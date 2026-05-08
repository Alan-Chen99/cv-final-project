"""
Flow Matching v2: UNet + self-attention at bottleneck for 32x32 -> 128x128 downscaling.

Changes from v1:
  - Multi-head self-attention between mid_block1 and mid_block2 (at 16x16 resolution)
  - ~0.8M extra parameters (256 channels, 4 heads)

Usage:
  python scripts/flow_matching_v2.py --mode train --epochs 40 --batch_size 64
  python scripts/flow_matching_v2.py --mode eval --n_ensemble 10 --split test
"""

import argparse
import copy
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------- Modules ----------

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch * 2)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_mlp(F.silu(t_emb)).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention for 2D feature maps (residual connection)."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W)
        qkv = self.qkv(h).view(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(dim=1)  # each: (B, heads, head_dim, HW)
        # Transpose to (B, heads, HW, head_dim) for attention
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        # Use PyTorch's efficient attention
        out = F.scaled_dot_product_attention(q, k, v)  # (B, heads, HW, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H * W)  # (B, C, HW)
        out = self.proj(out).view(B, C, H, W)
        return x + out


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class AttentionUNet(nn.Module):
    """UNet with self-attention at bottleneck for velocity prediction.

    Input: interpolated state (1 ch) + LR condition (1 ch) = 2 ch
    Output: predicted velocity (1 ch)
    """

    def __init__(self, in_channels=2, out_channels=1, base_channels=64,
                 channel_mults=(1, 2, 4), time_emb_dim=256, dropout=0.1,
                 attn_heads=4):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        ch = base_channels
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.down_blocks.append(nn.ModuleList([
                ResBlock(ch, out_ch, time_emb_dim, dropout),
                ResBlock(out_ch, out_ch, time_emb_dim, dropout),
            ]))
            self.down_samples.append(Downsample(out_ch))
            ch = out_ch

        # Bottleneck with self-attention
        self.mid_block1 = ResBlock(ch, ch, time_emb_dim, dropout)
        self.mid_attn = SelfAttention(ch, num_heads=attn_heads)
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim, dropout)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.up_blocks.append(nn.ModuleList([
                ResBlock(ch + out_ch, out_ch, time_emb_dim, dropout),
                ResBlock(out_ch, out_ch, time_emb_dim, dropout),
            ]))
            self.up_samples.append(Upsample(ch))
            ch = out_ch

        self.final_norm = nn.GroupNorm(min(32, ch), ch)
        self.final_conv = nn.Conv2d(ch, out_channels, 1)

    def forward(self, x, t, condition):
        x = torch.cat([x, condition], dim=1)
        t_emb = self.time_mlp(t * 1000.0)
        h = self.init_conv(x)

        skips = []
        for blocks, downsample in zip(self.down_blocks, self.down_samples):
            for block in blocks:
                h = block(h, t_emb)
            skips.append(h)
            h = downsample(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for blocks, upsample in zip(self.up_blocks, self.up_samples):
            h = upsample(h)
            h = torch.cat([h, skips.pop()], dim=1)
            for block in blocks:
                h = block(h, t_emb)

        return self.final_conv(F.silu(self.final_norm(h)))


# ---------- EMA ----------

class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


# ---------- Timestep sampling ----------

def sample_timesteps_uniform(batch_size, device):
    return torch.rand(batch_size, device=device)


def sample_timesteps_logit_normal(batch_size, device, mean=0.0, std=1.0):
    """Logit-normal distribution (from SD3). Concentrates on intermediate t."""
    u = torch.randn(batch_size, device=device) * std + mean
    return torch.sigmoid(u)


# ---------- Data loading ----------

def load_tcw4_data(basedir, split='train'):
    """Load TCW4 data. Returns (lr_up, residual, hr, lr_orig)."""
    inp = torch.load(f'{basedir}/data/era5_sr_data/{split}/input_{split}.pt', weights_only=False)
    tgt = torch.load(f'{basedir}/data/era5_sr_data/{split}/target_{split}.pt', weights_only=False)

    lr = inp[:, 0, :, :, :]  # (N, 1, 32, 32)
    hr = tgt[:, 0, :, :, :]  # (N, 1, 128, 128)
    lr_up = F.interpolate(lr, size=(128, 128), mode='bilinear', align_corners=False)
    residual = hr - lr_up

    return lr_up, residual, hr, lr


# ---------- CRPS functions ----------

def crps_ensemble(observation, forecasts):
    """CRPS (paper-compatible version with known asymmetric weighting)."""
    fc = forecasts.copy()
    fc.sort(axis=0)
    obs = observation
    fc_below = fc < obs[None, ...]
    crps = np.zeros_like(obs)
    for i in range(fc.shape[0]):
        below = fc_below[i, ...]
        weight = ((i + 1) ** 2 - i ** 2) / fc.shape[-1] ** 2
        crps[below] += weight * (obs[below] - fc[i, ...][below])
    for i in range(fc.shape[0] - 1, -1, -1):
        above = ~fc_below[i, ...]
        k = fc.shape[0] - 1 - i
        weight = ((k + 1) ** 2 - k ** 2) / fc.shape[0] ** 2
        crps[above] += weight * (fc[i, ...][above] - obs[above])
    return np.mean(crps)


def crps_ensemble_correct(observation, forecasts):
    """Correct CRPS: E|X-y| - 0.5*E|X-X'|."""
    M = forecasts.shape[0]
    abs_diff = np.mean(np.abs(forecasts - observation[None, ...]), axis=0)
    fc_sorted = np.sort(forecasts, axis=0)
    spread = 0.0
    for i in range(M):
        for j in range(i + 1, M):
            spread += np.abs(fc_sorted[j] - fc_sorted[i])
    spread = spread * 2.0 / (M * (M - 1)) if M > 1 else 0.0
    crps = abs_diff - 0.5 * spread
    return np.mean(crps)


# ---------- Constraint layers ----------

def apply_addcl(pred_hr, lr_orig, upsampling_factor=4):
    """AddCL: additive correction so avgpool(pred_hr) == lr_orig."""
    pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
    pooled = pool(pred_hr)
    correction = lr_orig - pooled
    correction_hr = correction.repeat_interleave(upsampling_factor, dim=-2).repeat_interleave(upsampling_factor, dim=-1)
    return pred_hr + correction_hr


def apply_smcl(pred_hr, lr_orig, upsampling_factor=4):
    """SmCL (Softmax Constraint): exp + multiplicative renorm.
    Enforces non-negativity AND conservation: avgpool(out) == lr_orig, out >= 0."""
    pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
    y = torch.exp(pred_hr)
    pooled = pool(y)
    # Multiplicative correction: scale so that block means match lr_orig
    correction = lr_orig / (pooled + 1e-8)
    correction_hr = correction.repeat_interleave(upsampling_factor, dim=-2).repeat_interleave(upsampling_factor, dim=-1)
    return y * correction_hr


# ---------- ODE Sampling ----------

@torch.no_grad()
def euler_sample(model, condition, shape, steps=25):
    """Euler ODE integration from noise (t=0) to data (t=1)."""
    device = condition.device
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((shape[0],), i * dt, device=device)
        v = model(x, t, condition)
        x = x + v * dt
    return x


@torch.no_grad()
def midpoint_sample(model, condition, shape, steps=25):
    """Midpoint (2nd-order Runge-Kutta) ODE integration from noise (t=0) to data (t=1).
    Uses 2 function evaluations per step (2*steps NFE total)."""
    device = condition.device
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((shape[0],), i * dt, device=device)
        t_mid = torch.full((shape[0],), (i + 0.5) * dt, device=device)
        v1 = model(x, t, condition)
        x_mid = x + v1 * (0.5 * dt)
        v2 = model(x_mid, t_mid, condition)
        x = x + v2 * dt
    return x


# ---------- Training ----------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    basedir = args.basedir

    print("Loading data...")
    lr_up_train, res_train, _, _ = load_tcw4_data(basedir, 'train')
    lr_up_val, res_val, _, _ = load_tcw4_data(basedir, 'val')

    # Normalize residuals
    res_mean = res_train.mean()
    res_std = res_train.std()
    res_train_norm = (res_train - res_mean) / res_std
    res_val_norm = (res_val - res_mean) / res_std

    # Normalize LR condition
    lr_mean = lr_up_train.mean()
    lr_std = lr_up_train.std()
    lr_up_train_norm = (lr_up_train - lr_mean) / lr_std
    lr_up_val_norm = (lr_up_val - lr_mean) / lr_std

    # Save normalization stats
    stats = {'res_mean': res_mean.item(), 'res_std': res_std.item(),
             'lr_mean': lr_mean.item(), 'lr_std': lr_std.item()}
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(stats, os.path.join(args.save_dir, 'norm_stats.pt'))

    train_ds = TensorDataset(lr_up_train_norm, res_train_norm)
    val_ds = TensorDataset(lr_up_val_norm, res_val_norm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Model
    model = AttentionUNet(
        in_channels=2, out_channels=1,
        base_channels=args.base_channels,
        channel_mults=args.channel_mults_tuple,
        time_emb_dim=256,
        dropout=0.1,
        attn_heads=args.attn_heads,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"#params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # EMA
    use_ema = getattr(args, 'use_ema', False)
    ema_decay = getattr(args, 'ema_decay', 0.9999)
    ema = None

    # Timestep sampling
    t_sampling = getattr(args, 't_sampling', 'uniform')
    t_logit_mean = getattr(args, 't_logit_mean', 0.0)
    t_logit_std = getattr(args, 't_logit_std', 1.0)
    print(f"Timestep sampling: {t_sampling}" +
          (f" (mean={t_logit_mean}, std={t_logit_std})" if t_sampling == 'logit_normal' else ''))
    if use_ema:
        print(f"EMA enabled: decay={ema_decay}")

    best_val_loss = float('inf')
    start_epoch = 0

    # Resume from checkpoint
    ckpt_path = os.path.join(args.save_dir, 'best_flow.pt')
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['val_loss']
        if 'optimizer' in ckpt and not args.finetune_lr:
            optimizer.load_state_dict(ckpt['optimizer'])
        if args.finetune_lr:
            # Fresh optimizer + cosine schedule for fine-tuning
            ft_lr = args.finetune_lr
            remaining = args.epochs - start_epoch
            for pg in optimizer.param_groups:
                pg['lr'] = ft_lr
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining)
            print(f"Resumed from epoch {start_epoch}, FRESH schedule: "
                  f"lr={ft_lr}, T_max={remaining}, best val loss: {best_val_loss:.6f}")
        else:
            for _ in range(start_epoch):
                scheduler.step()
            print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.6f}")

    # Initialize EMA after potential resume
    if use_ema:
        ema = EMA(model, decay=ema_decay)
        if args.resume and os.path.exists(ckpt_path) and 'ema' in ckpt:
            ema.load_state_dict(ckpt['ema'])
            print("Loaded EMA state from checkpoint")

    def _sample_t(bs, dev):
        if t_sampling == 'logit_normal':
            return sample_timesteps_logit_normal(bs, dev, t_logit_mean, t_logit_std)
        return sample_timesteps_uniform(bs, dev)

    # AMP setup
    use_amp = getattr(args, 'amp', False)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if use_amp:
        print("AMP (mixed precision) enabled")

    # Data augmentation
    use_augment = getattr(args, 'augment', False)
    if use_augment:
        print("Data augmentation enabled (random h/v flip)")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        for lr_batch, res_batch in train_loader:
            lr_batch = lr_batch.to(device)
            res_batch = res_batch.to(device)

            # Data augmentation: random flips (same transform for both)
            if use_augment:
                if torch.rand(1).item() > 0.5:
                    lr_batch = torch.flip(lr_batch, [-1])
                    res_batch = torch.flip(res_batch, [-1])
                if torch.rand(1).item() > 0.5:
                    lr_batch = torch.flip(lr_batch, [-2])
                    res_batch = torch.flip(res_batch, [-2])

            bs = lr_batch.shape[0]
            t = _sample_t(bs, device)
            x_0 = torch.randn_like(res_batch)
            t_expand = t[:, None, None, None]
            x_t = (1 - t_expand) * x_0 + t_expand * res_batch
            target_v = res_batch - x_0

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                pred_v = model(x_t, t, lr_batch)
                loss = F.mse_loss(pred_v, target_v)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model)

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validation (always uniform t for comparability)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for lr_batch, res_batch in val_loader:
                lr_batch = lr_batch.to(device)
                res_batch = res_batch.to(device)
                bs = lr_batch.shape[0]
                t = torch.rand(bs, device=device)
                x_0 = torch.randn_like(res_batch)
                t_expand = t[:, None, None, None]
                x_t = (1 - t_expand) * x_0 + t_expand * res_batch
                target_v = res_batch - x_0
                with torch.amp.autocast('cuda', enabled=use_amp):
                    pred_v = model(x_t, t, lr_batch)
                    val_loss += F.mse_loss(pred_v, target_v).item()
        val_loss /= len(val_loader)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs}, Train: {train_loss:.6f}, Val: {val_loss:.6f}, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}, Time: {elapsed/60:.1f}min")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'args': vars(args),
            }
            if ema is not None:
                save_dict['ema'] = ema.state_dict()
            torch.save(save_dict, os.path.join(args.save_dir, 'best_flow.pt'))

    # Save final EMA separately for evaluation
    if ema is not None:
        torch.save({
            'model': ema.state_dict(),
            'epoch': args.epochs - 1,
            'val_loss': val_loss,
            'args': vars(args),
            'is_ema': True,
        }, os.path.join(args.save_dir, 'best_flow_ema.pt'))
        print(f"Saved EMA model to {args.save_dir}/best_flow_ema.pt")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Total time: {(time.time() - start_time)/60:.1f} min")


# ---------- Evaluation ----------

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    basedir = args.basedir

    stats = torch.load(os.path.join(args.save_dir, 'norm_stats.pt'), weights_only=False,
                        map_location=device)

    lr_up, residual, hr, lr_orig = load_tcw4_data(basedir, args.split)
    lr_up_norm = (lr_up - stats['lr_mean']) / stats['lr_std']

    ckpt = torch.load(os.path.join(args.save_dir, 'best_flow.pt'), weights_only=False,
                       map_location=device)
    saved_args = ckpt['args']
    saved_mults = saved_args.get('channel_mults_tuple', (1, 2, 4))
    if isinstance(saved_mults, str):
        saved_mults = tuple(int(x) for x in saved_mults.split(','))
    model = AttentionUNet(
        in_channels=2, out_channels=1,
        base_channels=saved_args.get('base_channels', 64),
        channel_mults=saved_mults,
        time_emb_dim=256,
        dropout=0.0,
        attn_heads=saved_args.get('attn_heads', 4),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    n_samples = min(lr_up.shape[0], args.max_samples) if args.max_samples else lr_up.shape[0]
    n_ensemble = args.n_ensemble
    batch_size = args.eval_batch_size
    ode_steps = args.ode_steps
    use_constraint = args.constraint
    sampler_name = getattr(args, 'sampler', 'euler')
    sampler = midpoint_sample if sampler_name == 'midpoint' else euler_sample

    print(f"Evaluating {n_samples} samples, {n_ensemble} ensemble, "
          f"{ode_steps} {sampler_name} steps, constraint={use_constraint}...")
    print(f"Model epoch: {ckpt['epoch']+1}, val_loss: {ckpt['val_loss']:.6f}")

    all_crps = []
    all_crps_std = []
    all_mae = []
    all_rmse = []
    all_mass_viol = []

    pool = torch.nn.AvgPool2d(kernel_size=4)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_lr = lr_up_norm[start_idx:end_idx].to(device)
        batch_hr = hr[start_idx:end_idx]
        batch_lr_up = lr_up[start_idx:end_idx]
        batch_lr_orig = lr_orig[start_idx:end_idx]
        bs = batch_lr.shape[0]

        ensemble_preds = []
        use_tta = getattr(args, 'tta', False)
        for e in range(n_ensemble):
            with torch.no_grad():
                # TTA: flip input horizontally for half the ensemble members
                do_flip = use_tta and (e % 2 == 1)
                cond = torch.flip(batch_lr, [-1]) if do_flip else batch_lr
                sampled_res_norm = sampler(
                    model, cond,
                    shape=(bs, 1, 128, 128),
                    steps=ode_steps,
                )
                if do_flip:
                    sampled_res_norm = torch.flip(sampled_res_norm, [-1])
                sampled_res = sampled_res_norm.cpu() * stats['res_std'] + stats['res_mean']
                pred_hr = batch_lr_up + sampled_res

                if use_constraint == 'addcl':
                    pred_hr = apply_addcl(pred_hr, batch_lr_orig)
                elif use_constraint == 'smcl':
                    pred_hr = apply_smcl(pred_hr, batch_lr_orig)

                ensemble_preds.append(pred_hr.numpy())

        ensemble_preds = np.stack(ensemble_preds, axis=1)

        for i in range(bs):
            gt = batch_hr[i, 0, ...].numpy()
            ens = ensemble_preds[i, :, 0, ...]
            ens_mean = ens.mean(axis=0)

            all_crps.append(crps_ensemble(gt, ens))
            all_crps_std.append(crps_ensemble_correct(gt, ens))
            all_mae.append(np.mean(np.abs(gt - ens_mean)))
            all_rmse.append(np.mean((gt - ens_mean) ** 2))

            pred_mean_t = torch.from_numpy(ens_mean).unsqueeze(0).unsqueeze(0)
            pooled = pool(pred_mean_t).squeeze()
            lr_i = batch_lr_orig[i, 0, ...]
            all_mass_viol.append(torch.mean(torch.abs(pooled - lr_i)).item())

        if (start_idx // batch_size) % 10 == 0:
            print(f"  Processed {end_idx}/{n_samples}...")

    crps = np.mean(all_crps)
    crps_std = np.mean(all_crps_std)
    mae = np.mean(all_mae)
    rmse = np.sqrt(np.mean(all_rmse))
    mass_viol = np.mean(all_mass_viol)

    print(f"\nResults ({args.split}, {n_ensemble} ens, {ode_steps} {sampler_name} steps, constraint={use_constraint}):")
    print(f"  CRPS (paper): {crps:.6f}")
    print(f"  CRPS (std):   {crps_std:.6f}")
    print(f"  MAE:          {mae:.6f}")
    print(f"  RMSE:         {rmse:.6f}")
    print(f"  Mass viol:    {mass_viol:.6f}")

    return crps, mae, rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--basedir", default="external/constrained-downscaling")
    parser.add_argument("--save_dir", default="models/flow_v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finetune_lr", type=float, default=None,
                        help="Fresh LR + cosine schedule for fine-tuning after resume")
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--channel_mults", type=str, default="1,2,4")
    parser.add_argument("--attn_heads", type=int, default=4)
    # EMA
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    # Timestep sampling
    parser.add_argument("--t_sampling", default="uniform",
                        choices=["uniform", "logit_normal"])
    parser.add_argument("--t_logit_mean", type=float, default=0.0)
    parser.add_argument("--t_logit_std", type=float, default=1.0)
    # AMP + augmentation
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--augment", action="store_true", help="Random h/v flip augmentation")
    # Eval args
    parser.add_argument("--n_ensemble", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--ode_steps", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--constraint", default="none", choices=["none", "addcl", "smcl"])
    parser.add_argument("--sampler", default="euler", choices=["euler", "midpoint"])
    parser.add_argument("--tta", action="store_true",
                        help="Test-time augmentation: flip half of ensemble members")
    args = parser.parse_args()

    args.channel_mults_tuple = tuple(int(x) for x in args.channel_mults.split(','))

    if args.mode == "train":
        train(args)
    else:
        evaluate(args)
