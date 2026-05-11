"""Conditional flow matching for climate downscaling (32x32 -> 128x128).

Flow matching trains a velocity field v(x_t, t, c) where:
  x_t = (1-t) * x_0 + t * x_1    (linear interpolation)
  x_0 ~ N(0, I)                   (source noise)
  x_1 = HR target                  (data)
  c   = bicubic-upsampled LR       (condition)
  v*  = x_1 - x_0                  (target velocity)

Inference: Euler ODE from t=0 to t=1 with N steps.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset

# ── OT coupling ──────────────────────────────────────────────────────────────

@torch.no_grad()
def ot_coupling(x0, x1):
    """Minibatch optimal transport coupling.

    Given source x0 and target x1, find permutation of x0 that minimizes
    total transport cost (L2 distance). Cost matrix computed on GPU,
    only the small BxB matrix transferred to CPU for Hungarian algorithm.

    Args:
        x0: (B, ...) source samples (noise) on GPU
        x1: (B, ...) target samples (data) on GPU
    Returns:
        x0 reordered to match x1 via OT assignment (on same device)
    """
    B = x0.shape[0]
    # Flatten on GPU
    x0_flat = x0.reshape(B, -1)  # (B, D)
    x1_flat = x1.reshape(B, -1)  # (B, D)
    # Cost matrix on GPU: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    x0_sq = (x0_flat ** 2).sum(dim=1, keepdim=True)  # (B, 1)
    x1_sq = (x1_flat ** 2).sum(dim=1, keepdim=True)  # (B, 1)
    cost = x0_sq + x1_sq.T - 2 * x0_flat @ x1_flat.T  # (B, B) on GPU
    # Only transfer small BxB matrix to CPU for Hungarian
    cost_np = cost.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)
    # Reorder x0 so x0[i] is matched to x1[i]
    inv_perm = np.argsort(col_ind)
    return x0[inv_perm]


# ── Architecture ──────────────────────────────────────────────────────────────

def sinusoidal_embedding(t, dim):
    """Sinusoidal time embedding (like Transformer positional encoding)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    """Residual block with time conditioning via scale+shift (AdaGN style)."""

    def __init__(self, channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, channels * 2),
        )

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Time conditioning: scale + shift
        t_out = self.time_mlp(t_emb)[:, :, None, None]
        scale, shift = t_out.chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = F.silu(h)
        h = self.conv2(h)

        return x + h


class SelfAttention(nn.Module):
    """Multi-head self-attention for spatial features at coarse resolutions."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)  # each: (B, heads, head_dim, H*W)
        q = q.permute(0, 1, 3, 2)  # (B, heads, H*W, head_dim)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        h = F.scaled_dot_product_attention(q, k, v)  # (B, heads, H*W, head_dim)
        h = h.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj(h)


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {'decay': self.decay, 'shadow': dict(self.shadow)}

    def load_state_dict(self, state):
        self.decay = state['decay']
        self.shadow = state['shadow']


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, n_res=1):
        super().__init__()
        self.res_blocks = nn.ModuleList([ResBlock(in_ch, time_dim) for _ in range(n_res)])
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        h = x
        for res in self.res_blocks:
            h = res(h, t_emb)
        return self.down(h), h  # downsampled, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_dim, n_res=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.conv = nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1)
        self.res_blocks = nn.ModuleList([ResBlock(out_ch, time_dim) for _ in range(n_res)])

    def forward(self, x, skip, t_emb):
        h = self.up(x)
        h = torch.cat([h, skip], dim=1)
        h = self.conv(h)
        for res in self.res_blocks:
            h = res(h, t_emb)
        return h


class FlowUNet(nn.Module):
    """Small UNet for flow matching velocity prediction.

    Input: (B, 2, 128, 128) — channel 0 = x_t, channel 1 = LR condition
    Output: (B, 1, 128, 128) — predicted velocity
    """

    def __init__(self, channels=(32, 64, 128), time_dim=128, use_attention=False, n_res_blocks=1):
        super().__init__()
        self.time_dim = time_dim
        self.use_attention = use_attention
        c0, c1, c2 = channels

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Input: 2 channels (x_t + condition)
        self.input_conv = nn.Conv2d(2, c0, 3, padding=1)

        # Encoder: 128 -> 64 -> 32 -> 16
        self.down0 = DownBlock(c0, c1, time_dim, n_res=n_res_blocks)   # 128 -> 64
        self.down1 = DownBlock(c1, c2, time_dim, n_res=n_res_blocks)   # 64 -> 32
        self.down2 = DownBlock(c2, c2, time_dim, n_res=n_res_blocks)   # 32 -> 16

        # Middle
        self.mid = ResBlock(c2, time_dim)
        if use_attention:
            self.mid_attn = SelfAttention(c2)

        # Decoder: 16 -> 32 -> 64 -> 128
        self.up2 = UpBlock(c2, c2, c2, time_dim, n_res=n_res_blocks)   # 16 -> 32
        self.up1 = UpBlock(c2, c1, c1, time_dim, n_res=n_res_blocks)   # 32 -> 64
        self.up0 = UpBlock(c1, c0, c0, time_dim, n_res=n_res_blocks)   # 64 -> 128

        # Output
        self.out_norm = nn.GroupNorm(8, c0)
        self.out_conv = nn.Conv2d(c0, 1, 1)

    def forward(self, x_t, t, cond):
        """
        x_t: (B, 1, 128, 128) — noisy/interpolated sample
        t: (B,) — time in [0, 1]
        cond: (B, 1, 128, 128) — bicubic-upsampled LR condition
        """
        # Time embedding
        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        # Concatenate input and condition
        h = torch.cat([x_t, cond], dim=1)  # (B, 2, 128, 128)
        h = self.input_conv(h)

        # Encoder
        h, s0 = self.down0(h, t_emb)  # 64
        h, s1 = self.down1(h, t_emb)  # 32
        h, s2 = self.down2(h, t_emb)  # 16

        # Middle
        h = self.mid(h, t_emb)
        if self.use_attention:
            h = self.mid_attn(h)

        # Decoder
        h = self.up2(h, s2, t_emb)  # 32
        h = self.up1(h, s1, t_emb)  # 64
        h = self.up0(h, s0, t_emb)  # 128

        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        return self.out_conv(h)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_dir, batch_size, split='train'):
    """Load and normalize data matching baseline preprocessing."""
    data_dir = Path(data_dir)

    inp_train = torch.load(data_dir / 'train' / 'input_train.pt', weights_only=False)
    tgt_train = torch.load(data_dir / 'train' / 'target_train.pt', weights_only=False)

    # Normalization stats from training targets (match baseline)
    max_val = tgt_train[:, 0, 0, ...].max()
    min_val = tgt_train[:, 0, 0, ...].min()

    if split == 'test':
        inp = torch.load(data_dir / 'test' / 'input_test.pt', weights_only=False)
        tgt = torch.load(data_dir / 'test' / 'target_test.pt', weights_only=False)
    elif split == 'val':
        inp = torch.load(data_dir / 'val' / 'input_val.pt', weights_only=False)
        tgt = torch.load(data_dir / 'val' / 'target_val.pt', weights_only=False)
    else:
        inp, tgt = inp_train, tgt_train

    # Normalize to [0, 1]
    inp = (inp - min_val) / (max_val - min_val)
    tgt = (tgt - min_val) / (max_val - min_val)

    # Extract from (N, 1, 1, H, W) to (N, 1, H, W)
    inp = inp[:, 0, :, :, :]   # (N, 1, 32, 32)
    tgt = tgt[:, 0, :, :, :]   # (N, 1, 128, 128)

    dataset = TensorDataset(inp, tgt)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=(split == 'train'), drop_last=(split == 'train'),
                        num_workers=0, pin_memory=True)
    return loader, float(min_val), float(max_val)


def bicubic_upsample(lr, size=128):
    """Bicubic upsample LR to HR resolution for conditioning."""
    return F.interpolate(lr, size=(size, size), mode='bicubic', align_corners=False)


def bilinear_upsample(lr, size=128):
    """Bilinear upsample LR to HR resolution."""
    return F.interpolate(lr, size=(size, size), mode='bilinear', align_corners=False)


# ── Constraint layers ────────────────────────────────────────────────────────

def apply_constraint(hr, lr_32, constraint_type='none', eps=1e-6):
    """Apply conservation constraint to HR prediction (post-hoc).

    Enforces: AvgPool_4x4(output) = lr_32 exactly.

    Args:
        hr: (B, 1, 128, 128) HR prediction (normalized)
        lr_32: (B, 1, 32, 32) LR input (normalized)
        constraint_type: 'softmax' (exp + scale), 'mult' (clamp + scale), or 'none'
        eps: numerical stability
    Returns:
        (B, 1, 128, 128) constrained HR prediction
    """
    if constraint_type == 'none':
        return hr.clamp(0, 1)

    if constraint_type == 'addcl':
        # Additive correction: avgpool(hr) == lr_32
        pooled = F.avg_pool2d(hr, kernel_size=4)  # (B, 1, 32, 32)
        correction = lr_32 - pooled
        correction_up = correction.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)
        return hr + correction_up

    if constraint_type == 'softmax':
        y = torch.exp(hr)
    elif constraint_type == 'mult':
        y = hr.clamp(min=eps)
    else:
        raise ValueError(f"Unknown constraint: {constraint_type}")

    # Downsample to 32×32 via block average
    sum_y = F.avg_pool2d(y, kernel_size=4)  # (B, 1, 32, 32)

    # Ratio: lr / sum_y, then broadcast back to 128×128
    ratio = lr_32 / (sum_y + eps)
    ratio_up = ratio.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)

    return y * ratio_up


# ── Training ──────────────────────────────────────────────────────────────────

def apply_constraint_differentiable(hr, lr_32, eps=1e-6):
    """Differentiable mult constraint for use in training loss.

    Uses softplus instead of clamp for smooth gradients.
    """
    y = F.softplus(hr, beta=10)  # smooth positive approximation
    sum_y = F.avg_pool2d(y, kernel_size=4)  # (B, 1, 32, 32)
    ratio = lr_32 / (sum_y + eps)
    ratio_up = ratio.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)
    return y * ratio_up


def train(args):
    device = torch.device('cuda')
    constraint_aware = getattr(args, 'constraint_aware', False)
    constraint_weight = getattr(args, 'constraint_weight', 0.1)
    crps_loss_enabled = getattr(args, 'crps_loss', False)
    crps_weight = getattr(args, 'crps_weight', 0.1)
    lr_anchor = getattr(args, 'lr_anchor', False)
    noise_std = getattr(args, 'noise_std', 0.5)
    augment = getattr(args, 'augment', False)
    residual = getattr(args, 'residual', False)
    use_ot = getattr(args, 'ot', False)
    print(f"Training flow matching model on {device}")
    print(f"Args: epochs={args.epochs}, bs={args.batch_size}, lr={args.lr}, "
          f"channels={args.channels}, euler_steps={args.euler_steps}")
    if residual:
        print("Residual mode: learning on (HR - bilinear(LR)) space")
    if use_ot:
        print("OT-CFM: minibatch optimal transport coupling enabled")
    if lr_anchor:
        print(f"LR-anchor mode: noise_std={noise_std}")
    if augment:
        print("Data augmentation: random h/v flips enabled")
    if crps_loss_enabled:
        print(f"CRPS-aware training: weight={crps_weight}")
    elif constraint_aware:
        print(f"Constraint-aware training: weight={constraint_weight}")

    # Data
    train_loader, min_val, max_val = load_data(args.data_dir, args.batch_size, 'train')
    val_loader, _, _ = load_data(args.data_dir, args.batch_size, 'val')
    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")
    print(f"Normalization: min={min_val:.4f}, max={max_val:.4f}")

    # Model
    channels = [int(c) for c in args.channels.split(',')]
    use_attention = getattr(args, 'attention', False)
    n_res_blocks = getattr(args, 'n_res_blocks', 1)
    model = FlowUNet(channels=channels, use_attention=use_attention, n_res_blocks=n_res_blocks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,} (n_res_blocks={n_res_blocks})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')

    # EMA
    use_ema = getattr(args, 'ema', False)
    ema_decay = getattr(args, 'ema_decay', 0.999)
    ema = EMA(model, decay=ema_decay) if use_ema else None
    if use_ema:
        print(f"EMA enabled: decay={ema_decay}")

    # Training loop
    best_val_loss = float('inf')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0

    # Resume from checkpoint if available
    ckpt_path = save_dir / 'flow_checkpoint.pth'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        if ema is not None and 'ema' in ckpt:
            ema.load_state_dict(ckpt['ema'])
        print(f"Resumed from epoch {start_epoch}, best_val={best_val_loss:.6f}")

    # Save config early so eval works even if training is interrupted
    torch.save({'min_val': min_val, 'max_val': max_val,
                'channels': channels, 'euler_steps': args.euler_steps,
                'constraint_aware': constraint_aware,
                'crps_loss': crps_loss_enabled,
                'lr_anchor': lr_anchor, 'noise_std': noise_std,
                'use_attention': use_attention,
                'n_res_blocks': n_res_blocks,
                'residual': residual,
                'ot': use_ot},
               save_dir / 'flow_config.pth')

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        train_aux_loss = 0.0
        n_batches = 0

        for lr_input, hr_target in train_loader:
            lr_input = lr_input.to(device)      # (B, 1, 32, 32)
            hr_target = hr_target.to(device)     # (B, 1, 128, 128)

            # Data augmentation: random h/v flips (applied consistently to LR and HR)
            if augment:
                if torch.rand(1).item() > 0.5:
                    lr_input = torch.flip(lr_input, [-1])
                    hr_target = torch.flip(hr_target, [-1])
                if torch.rand(1).item() > 0.5:
                    lr_input = torch.flip(lr_input, [-2])
                    hr_target = torch.flip(hr_target, [-2])

            B = lr_input.shape[0]

            # Upsample LR for conditioning
            cond = bilinear_upsample(lr_input) if residual else bicubic_upsample(lr_input)

            # In residual mode, flow target is the residual (HR - bilinear(LR))
            flow_target = (hr_target - cond) if residual else hr_target

            # Sample source and time
            if lr_anchor:
                x0 = cond + noise_std * torch.randn_like(flow_target)
            else:
                x0 = torch.randn_like(flow_target)

            # OT coupling: reorder x0 to minimize transport cost to flow_target
            if use_ot:
                x0 = ot_coupling(x0, flow_target)

            t = torch.rand(B, device=device)

            # Interpolate: x_t = (1-t)*x0 + t*x1
            t_expand = t[:, None, None, None]
            x_t = (1 - t_expand) * x0 + t_expand * flow_target

            # Target velocity: v = x1 - x0
            v_target = flow_target - x0

            # Forward + loss with AMP
            with torch.amp.autocast('cuda'):
                v_pred = model(x_t, t, cond)
                loss = F.mse_loss(v_pred, v_target)

                # Auxiliary loss for t > 0.5
                aux_loss_val = 0.0
                if crps_loss_enabled:
                    mask = (t > 0.5).float()
                    if mask.sum() > 0:
                        remaining = (1.0 - t_expand)
                        # Prediction 1: from current forward pass
                        x_hat_1 = x_t + v_pred * remaining
                        x_hat_1c = apply_constraint_differentiable(x_hat_1, lr_input)

                        # Prediction 2: different noise realization
                        if lr_anchor:
                            x0_2 = cond + noise_std * torch.randn_like(hr_target)
                        else:
                            x0_2 = torch.randn_like(hr_target)
                        x_t_2 = (1 - t_expand) * x0_2 + t_expand * hr_target
                        v_pred_2 = model(x_t_2, t, cond)
                        x_hat_2 = x_t_2 + v_pred_2 * remaining
                        x_hat_2c = apply_constraint_differentiable(x_hat_2, lr_input)

                        # CRPS energy form: E|X-y| - 0.5*E|X-X'|
                        mae_term = 0.5 * (
                            torch.abs(x_hat_1c - hr_target).mean(dim=(1, 2, 3)) +
                            torch.abs(x_hat_2c - hr_target).mean(dim=(1, 2, 3))
                        )
                        spread_term = 0.5 * torch.abs(x_hat_1c - x_hat_2c).mean(dim=(1, 2, 3))
                        crps_per_sample = mae_term - spread_term
                        aux_loss = (crps_per_sample * mask).sum() / mask.sum()
                        loss = loss + crps_weight * aux_loss
                        aux_loss_val = aux_loss.item()
                elif constraint_aware:
                    mask = (t > 0.5).float()
                    if mask.sum() > 0:
                        # One-step prediction to t=1
                        remaining = (1.0 - t_expand)
                        x_hat = x_t + v_pred * remaining
                        # Apply differentiable constraint
                        x_constrained = apply_constraint_differentiable(x_hat, lr_input)
                        # Reconstruction loss (masked to t > 0.5 samples)
                        per_sample = ((x_constrained - hr_target) ** 2).mean(dim=(1, 2, 3))
                        aux_loss = (per_sample * mask).sum() / mask.sum()
                        loss = loss + constraint_weight * aux_loss
                        aux_loss_val = aux_loss.item()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model)

            train_loss += loss.item()
            train_aux_loss += aux_loss_val
            n_batches += 1

        scheduler.step()
        avg_train = train_loss / n_batches
        avg_aux = train_aux_loss / n_batches if (constraint_aware or crps_loss_enabled) else 0

        # Validation (use EMA weights if enabled)
        if ema is not None:
            ema.apply_shadow(model)
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for lr_input, hr_target in val_loader:
                lr_input = lr_input.to(device)
                hr_target = hr_target.to(device)
                B = lr_input.shape[0]
                cond = bilinear_upsample(lr_input) if residual else bicubic_upsample(lr_input)
                flow_target = (hr_target - cond) if residual else hr_target
                if lr_anchor:
                    x0 = cond + noise_std * torch.randn_like(flow_target)
                else:
                    x0 = torch.randn_like(flow_target)
                t = torch.rand(B, device=device)
                t_expand = t[:, None, None, None]
                x_t = (1 - t_expand) * x0 + t_expand * flow_target
                v_target = flow_target - x0
                with torch.amp.autocast('cuda'):
                    v_pred = model(x_t, t, cond)
                    val_loss += F.mse_loss(v_pred, v_target).item()
                n_val += 1
        avg_val = val_loss / n_val

        elapsed = time.time() - t_start
        if (epoch + 1) % 10 == 0 or epoch == 0:
            aux_str = f" aux={avg_aux:.6f}" if (constraint_aware or crps_loss_enabled) else ""
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"train={avg_train:.6f}{aux_str} val={avg_val:.6f} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e} | "
                  f"{elapsed/60:.1f}min")

        # Checkpoint best (EMA weights saved when EMA enabled)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_dir / 'flow_best.pth')

        # Restore raw weights after EMA validation
        if ema is not None:
            ema.restore(model)

        # Save resumable checkpoint every epoch
        ckpt_data = {
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_val_loss': best_val_loss,
        }
        if ema is not None:
            ckpt_data['ema'] = ema.state_dict()
        torch.save(ckpt_data, save_dir / 'flow_checkpoint.pth')

    # Save final model (EMA weights if enabled)
    if ema is not None:
        ema.apply_shadow(model)
    torch.save(model.state_dict(), save_dir / 'flow_last.pth')
    if ema is not None:
        ema.restore(model)
    total_min = (time.time() - t_start) / 60
    print(f"\nTraining complete in {total_min:.1f} min. Best val loss: {best_val_loss:.6f}")
    print(f"Saved to {save_dir}")

    # Save normalization stats
    torch.save({'min_val': min_val, 'max_val': max_val,
                'channels': channels, 'euler_steps': args.euler_steps,
                'constraint_aware': constraint_aware,
                'crps_loss': crps_loss_enabled,
                'lr_anchor': lr_anchor, 'noise_std': noise_std,
                'use_attention': use_attention,
                'n_res_blocks': n_res_blocks,
                'residual': residual,
                'ot': use_ot},
               save_dir / 'flow_config.pth')


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_ensemble(model, lr_input, n_members, n_steps, device,
                      constraint='none', solver='euler',
                      lr_anchor=False, noise_std=1.0,
                      residual=False):
    """Generate ensemble predictions via ODE integration.

    Args:
        model: trained FlowUNet
        lr_input: (B, 1, 32, 32) normalized LR
        n_members: number of ensemble members
        n_steps: integration steps
        device: torch device
        constraint: 'none', 'softmax', 'mult', or 'addcl'
        solver: 'euler' (1st order) or 'heun' (2nd order, 2x NFE)
        lr_anchor: if True, start from LR + noise instead of pure noise
        noise_std: noise scale for LR-anchor mode
        residual: if True, model predicts residual, add bilinear(LR) at the end

    Returns:
        (B, M, 1, 128, 128) ensemble predictions (normalized)
    """
    B = lr_input.shape[0]
    cond = bilinear_upsample(lr_input) if residual else bicubic_upsample(lr_input)
    dt = 1.0 / n_steps

    all_members = []
    for _m in range(n_members):
        if lr_anchor:
            x = cond + noise_std * torch.randn(B, 1, 128, 128, device=device)
        else:
            x = torch.randn(B, 1, 128, 128, device=device)

        for step in range(n_steps):
            t_cur = torch.full((B,), step * dt, device=device)
            v1 = model(x, t_cur, cond)

            if solver == 'heun':
                x_euler = x + v1 * dt
                t_next = torch.full((B,), (step + 1) * dt, device=device)
                v2 = model(x_euler, t_next, cond)
                x = x + 0.5 * dt * (v1 + v2)
            else:
                x = x + v1 * dt

        # In residual mode, add back the bilinear upsampled LR
        if residual:
            x = x + cond

        x = apply_constraint(x, lr_input, constraint)
        all_members.append(x)

    return torch.stack(all_members, dim=1)  # (B, M, 1, 128, 128)


def evaluate(args):
    """Generate ensemble predictions and evaluate CRPS."""
    device = torch.device('cuda')

    # Load config and model
    save_dir = Path(args.save_dir)
    config = torch.load(save_dir / 'flow_config.pth', weights_only=False)
    min_val = config['min_val']
    max_val = config['max_val']
    channels = config['channels']
    euler_steps = args.euler_steps or config['euler_steps']

    lr_anchor = config.get('lr_anchor', False)
    noise_std = config.get('noise_std', 1.0)
    residual = config.get('residual', False)
    if args.eval_noise_std is not None:
        print(f"Overriding noise_std: {noise_std} -> {args.eval_noise_std}")
        noise_std = args.eval_noise_std

    use_attention = config.get('use_attention', False)
    n_res_blocks = config.get('n_res_blocks', 1)
    model = FlowUNet(channels=channels, use_attention=use_attention, n_res_blocks=n_res_blocks).to(device)
    ckpt = 'flow_best.pth' if (save_dir / 'flow_best.pth').exists() else 'flow_last.pth'
    state_dict = torch.load(save_dir / ckpt, weights_only=False)
    # Remap old checkpoint keys: down0.res.* -> down0.res_blocks.0.*
    remapped = {}
    for k, v in state_dict.items():
        for prefix in ['down0', 'down1', 'down2', 'up0', 'up1', 'up2']:
            if k.startswith(f'{prefix}.res.'):
                k = k.replace(f'{prefix}.res.', f'{prefix}.res_blocks.0.', 1)
                break
        remapped[k] = v
    model.load_state_dict(remapped)
    model.eval()
    constraint = getattr(args, 'constraint', 'none')
    solver = getattr(args, 'solver', 'euler')
    print(f"Loaded {ckpt}, euler_steps={euler_steps}, n_members={args.n_members}, "
          f"constraint={constraint}, solver={solver}, lr_anchor={lr_anchor}, "
          f"noise_std={noise_std}, residual={residual}")

    # Load test data (normalized)
    test_loader, _, _ = load_data(args.data_dir, args.eval_batch_size, 'test')
    print(f"Test samples: {len(test_loader.dataset)}")

    # Generate ensemble predictions
    all_preds = []
    all_targets = []
    all_inputs = []
    t_start = time.time()

    for i, (lr_input, hr_target) in enumerate(test_loader):
        lr_input = lr_input.to(device)
        ensemble = generate_ensemble(model, lr_input, args.n_members, euler_steps,
                                     device, constraint=constraint, solver=solver,
                                     lr_anchor=lr_anchor, noise_std=noise_std,
                                     residual=residual)
        # Denormalize
        ensemble = ensemble * (max_val - min_val) + min_val
        hr_denorm = hr_target * (max_val - min_val) + min_val

        all_preds.append(ensemble.cpu())
        all_targets.append(hr_denorm)
        all_inputs.append(lr_input.cpu() * (max_val - min_val) + min_val)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  Batch {i+1}/{len(test_loader)} ({elapsed:.0f}s)")

    preds = torch.cat(all_preds, dim=0)       # (N, M, 1, 128, 128)
    targets = torch.cat(all_targets, dim=0)     # (N, 1, 128, 128)
    inputs = torch.cat(all_inputs, dim=0)       # (N, 1, 32, 32)

    gen_time = time.time() - t_start
    print(f"Generation done in {gen_time:.0f}s")

    # Compute metrics first (before save, which can fail on large files)
    compute_metrics(preds, targets, inputs)

    # Save predictions in baseline-compatible format: (N, M, 1, 1, H, W)
    try:
        preds_save = preds.unsqueeze(2)             # (N, M, 1, 1, 128, 128)
        pred_dir = Path(args.data_dir) / 'prediction'
        pred_dir.mkdir(exist_ok=True)
        pred_path = pred_dir / f'flow_{args.model_id}_test_ensemble.pt'
        torch.save(preds_save, pred_path)
        print(f"Saved predictions to {pred_path}")
    except Exception as e:
        print(f"Warning: Failed to save predictions: {e}")


def compute_metrics(preds, targets, inputs):
    """Compute CRPS and other metrics.

    preds: (N, M, 1, H, W)
    targets: (N, 1, H, W)
    inputs: (N, 1, 32, 32)
    """
    from skimage import transform as sktransform

    N, M = preds.shape[0], preds.shape[1]
    pred_np = preds[:, :, 0, :, :].numpy()    # (N, M, H, W)
    target_np = targets[:, 0, :, :].numpy()     # (N, H, W)
    mean_pred = preds.mean(dim=1)[:, 0, :, :].numpy()  # (N, H, W)
    input_np = inputs[:, 0, :, :].numpy()       # (N, 32, 32)

    # MSE, RMSE, MAE (of ensemble mean)
    mse = float(np.mean((mean_pred - target_np) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(mean_pred - target_np)))

    # CRPS (fast sorted method)
    crps_vals = []
    for i in range(N):
        fc = pred_np[i]             # (M, H, W)
        obs = target_np[i]          # (H, W)
        mae_term = np.mean(np.abs(fc - obs[None, ...]), axis=0)
        fc_sorted = np.sort(fc, axis=0)
        spread = np.zeros_like(obs)
        for j in range(M):
            w = (2.0 * (j + 1) - M - 1.0) / (M * M)
            spread += w * fc_sorted[j]
        crps_vals.append(float(np.mean(mae_term - spread)))
    crps = float(np.mean(crps_vals))

    # Ensemble spread
    spread_val = float(np.mean(np.std(pred_np, axis=1)))

    # Mass violation (ensemble mean vs LR input)
    mass_viol = 0.0
    for i in range(N):
        ds = sktransform.downscale_local_mean(mean_pred[i], (4, 4))
        mass_viol += float(np.mean(np.abs(ds - input_np[i])))
    mass_viol /= N

    print(f"\n{'='*40}")
    print(f"CRPS:            {crps:.6f}")
    print(f"MSE:             {mse:.6f}")
    print(f"RMSE:            {rmse:.6f}")
    print(f"MAE:             {mae:.6f}")
    print(f"Ensemble Spread: {spread_val:.6f}")
    print(f"Mass Violation:  {mass_viol:.6f}")
    print(f"Ensemble Size:   {M}")
    print(f"N Samples:       {N}")
    print(f"{'='*40}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval', 'both'], default='both')
    parser.add_argument('--data-dir', default='external/constrained-downscaling/data/era5_sr_data')
    parser.add_argument('--save-dir', default='models/flow')
    parser.add_argument('--model-id', default='flow_none')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--channels', default='32,64,128')
    parser.add_argument('--euler-steps', type=int, default=20)
    parser.add_argument('--n-members', type=int, default=10)
    parser.add_argument('--constraint', choices=['none', 'softmax', 'mult', 'addcl'],
                        default='none', help='Post-hoc conservation constraint')
    parser.add_argument('--solver', choices=['euler', 'heun'], default='euler',
                        help='ODE solver for inference')
    parser.add_argument('--constraint-aware', action='store_true',
                        help='Enable constraint-aware auxiliary loss during training')
    parser.add_argument('--constraint-weight', type=float, default=0.1,
                        help='Weight for constraint-aware auxiliary loss')
    parser.add_argument('--crps-loss', action='store_true',
                        help='Enable CRPS-aware auxiliary loss (replaces constraint-aware)')
    parser.add_argument('--crps-weight', type=float, default=0.1,
                        help='Weight for CRPS auxiliary loss')
    parser.add_argument('--lr-anchor', action='store_true',
                        help='Start ODE from LR + noise instead of pure noise')
    parser.add_argument('--noise-std', type=float, default=0.5,
                        help='Noise scale for LR-anchor mode')
    parser.add_argument('--eval-noise-std', type=float, default=None,
                        help='Override noise_std at eval time (default: use training value)')
    parser.add_argument('--attention', action='store_true',
                        help='Add self-attention at bottleneck (16x16)')
    parser.add_argument('--ema', action='store_true',
                        help='Enable EMA of model weights during training')
    parser.add_argument('--ema-decay', type=float, default=0.999,
                        help='EMA decay rate (default: 0.999)')
    parser.add_argument('--augment', action='store_true',
                        help='Enable random h/v flip augmentation during training')
    parser.add_argument('--residual', action='store_true',
                        help='Learn on residual (HR - bilinear(LR)) space')
    parser.add_argument('--ot', action='store_true',
                        help='Use minibatch OT coupling (OT-CFM)')
    parser.add_argument('--n-res-blocks', type=int, default=1,
                        help='Number of ResBlocks per level (default: 1, research2 used 2)')
    args = parser.parse_args()

    if args.mode in ('train', 'both'):
        train(args)
    if args.mode in ('eval', 'both'):
        evaluate(args)
