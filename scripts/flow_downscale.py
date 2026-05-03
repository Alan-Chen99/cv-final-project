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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


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


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.res = ResBlock(in_ch, time_dim)
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        h = self.res(x, t_emb)
        return self.down(h), h  # downsampled, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.conv = nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1)
        self.res = ResBlock(out_ch, time_dim)

    def forward(self, x, skip, t_emb):
        h = self.up(x)
        h = torch.cat([h, skip], dim=1)
        h = self.conv(h)
        return self.res(h, t_emb)


class FlowUNet(nn.Module):
    """Small UNet for flow matching velocity prediction.

    Input: (B, 2, 128, 128) — channel 0 = x_t, channel 1 = LR condition
    Output: (B, 1, 128, 128) — predicted velocity
    """

    def __init__(self, channels=(32, 64, 128), time_dim=128):
        super().__init__()
        self.time_dim = time_dim
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
        self.down0 = DownBlock(c0, c1, time_dim)   # 128 -> 64
        self.down1 = DownBlock(c1, c2, time_dim)   # 64 -> 32
        self.down2 = DownBlock(c2, c2, time_dim)   # 32 -> 16

        # Middle
        self.mid = ResBlock(c2, time_dim)

        # Decoder: 16 -> 32 -> 64 -> 128
        self.up2 = UpBlock(c2, c2, c2, time_dim)   # 16 -> 32
        self.up1 = UpBlock(c2, c1, c1, time_dim)   # 32 -> 64
        self.up0 = UpBlock(c1, c0, c0, time_dim)   # 64 -> 128

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
                        num_workers=4, pin_memory=True)
    return loader, float(min_val), float(max_val)


def bicubic_upsample(lr, size=128):
    """Bicubic upsample LR to HR resolution for conditioning."""
    return F.interpolate(lr, size=(size, size), mode='bicubic', align_corners=False)


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

def train(args):
    device = torch.device('cuda')
    print(f"Training flow matching model on {device}")
    print(f"Args: epochs={args.epochs}, bs={args.batch_size}, lr={args.lr}, "
          f"channels={args.channels}, euler_steps={args.euler_steps}")

    # Data
    train_loader, min_val, max_val = load_data(args.data_dir, args.batch_size, 'train')
    val_loader, _, _ = load_data(args.data_dir, args.batch_size, 'val')
    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")
    print(f"Normalization: min={min_val:.4f}, max={max_val:.4f}")

    # Model
    channels = [int(c) for c in args.channels.split(',')]
    model = FlowUNet(channels=channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')

    # Training loop
    best_val_loss = float('inf')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for lr_input, hr_target in train_loader:
            lr_input = lr_input.to(device)      # (B, 1, 32, 32)
            hr_target = hr_target.to(device)     # (B, 1, 128, 128)

            B = lr_input.shape[0]

            # Bicubic upsample LR for conditioning
            cond = bicubic_upsample(lr_input)    # (B, 1, 128, 128)

            # Sample noise and time
            x0 = torch.randn_like(hr_target)     # (B, 1, 128, 128)
            t = torch.rand(B, device=device)      # (B,) in [0, 1]

            # Interpolate: x_t = (1-t)*x0 + t*x1
            t_expand = t[:, None, None, None]
            x_t = (1 - t_expand) * x0 + t_expand * hr_target

            # Target velocity: v = x1 - x0
            v_target = hr_target - x0

            # Forward + loss with AMP
            with torch.amp.autocast('cuda'):
                v_pred = model(x_t, t, cond)
                loss = F.mse_loss(v_pred, v_target)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = train_loss / n_batches

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for lr_input, hr_target in val_loader:
                lr_input = lr_input.to(device)
                hr_target = hr_target.to(device)
                B = lr_input.shape[0]
                cond = bicubic_upsample(lr_input)
                x0 = torch.randn_like(hr_target)
                t = torch.rand(B, device=device)
                t_expand = t[:, None, None, None]
                x_t = (1 - t_expand) * x0 + t_expand * hr_target
                v_target = hr_target - x0
                with torch.amp.autocast('cuda'):
                    v_pred = model(x_t, t, cond)
                    val_loss += F.mse_loss(v_pred, v_target).item()
                n_val += 1
        avg_val = val_loss / n_val

        elapsed = time.time() - t_start
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"train={avg_train:.6f} val={avg_val:.6f} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e} | "
                  f"{elapsed/60:.1f}min")

        # Checkpoint best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_dir / 'flow_best.pth')

    torch.save(model.state_dict(), save_dir / 'flow_last.pth')
    total_min = (time.time() - t_start) / 60
    print(f"\nTraining complete in {total_min:.1f} min. Best val loss: {best_val_loss:.6f}")
    print(f"Saved to {save_dir}")

    # Save normalization stats
    torch.save({'min_val': min_val, 'max_val': max_val,
                'channels': channels, 'euler_steps': args.euler_steps},
               save_dir / 'flow_config.pth')


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_ensemble(model, lr_input, n_members, n_steps, device,
                      constraint='none'):
    """Generate ensemble predictions via Euler ODE integration.

    Args:
        model: trained FlowUNet
        lr_input: (B, 1, 32, 32) normalized LR
        n_members: number of ensemble members
        n_steps: Euler integration steps
        device: torch device
        constraint: 'none', 'softmax', or 'mult'

    Returns:
        (B, M, 1, 128, 128) ensemble predictions (normalized)
    """
    B = lr_input.shape[0]
    cond = bicubic_upsample(lr_input)  # (B, 1, 128, 128)
    dt = 1.0 / n_steps

    all_members = []
    for m in range(n_members):
        x = torch.randn(B, 1, 128, 128, device=device)  # Start from noise

        for step in range(n_steps):
            t = torch.full((B,), step * dt, device=device)
            v = model(x, t, cond)
            x = x + v * dt

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

    model = FlowUNet(channels=channels).to(device)
    ckpt = 'flow_best.pth' if (save_dir / 'flow_best.pth').exists() else 'flow_last.pth'
    model.load_state_dict(torch.load(save_dir / ckpt, weights_only=False))
    model.eval()
    constraint = getattr(args, 'constraint', 'none')
    print(f"Loaded {ckpt}, euler_steps={euler_steps}, n_members={args.n_members}, "
          f"constraint={constraint}")

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
                                     device, constraint=constraint)
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

    # Save predictions in baseline-compatible format: (N, M, 1, 1, H, W)
    preds_save = preds.unsqueeze(2)             # (N, M, 1, 1, 128, 128)
    targets_save = targets.unsqueeze(1)         # (N, 1, 1, 128, 128)
    inputs_save = inputs.unsqueeze(1)           # (N, 1, 1, 32, 32)

    pred_dir = Path(args.data_dir) / 'prediction'
    pred_dir.mkdir(exist_ok=True)
    pred_path = pred_dir / f'flow_{args.model_id}_test_ensemble.pt'
    torch.save(preds_save, pred_path)
    print(f"Saved predictions to {pred_path}")

    # Compute metrics
    compute_metrics(preds, targets, inputs)


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
    parser.add_argument('--constraint', choices=['none', 'softmax', 'mult'],
                        default='none', help='Post-hoc conservation constraint')
    args = parser.parse_args()

    if args.mode in ('train', 'both'):
        train(args)
    if args.mode in ('eval', 'both'):
        evaluate(args)
