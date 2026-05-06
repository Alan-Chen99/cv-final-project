"""CorrDiff-style residual flow matching: SwinIR deterministic mean + flow matching on residuals.

Two-stage approach:
  Stage 1 (frozen): Finetuned SwinIR predicts deterministic mean (MAE=0.250)
  Stage 2 (trained): FlowUNet learns residual distribution r = hr - swinir_pred
  Inference: sample_k = swinir_pred + flow_residual_k

This separates mean accuracy from stochastic diversity (CorrDiff principle).
The flow matching produces structured spatial diversity unlike multi-head approaches.

Usage:
    python src/exp-pretrained-sr/train_residual_flow.py --mode precompute  # step 1
    python src/exp-pretrained-sr/train_residual_flow.py --mode train       # step 2
    python src/exp-pretrained-sr/train_residual_flow.py --mode eval        # evaluate
    python src/exp-pretrained-sr/train_residual_flow.py --mode all         # all three
"""

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

POOL = Path("/home/chenxy/orcd/pool/datasets")
DATA_DIR = POOL / "era5_sr_data"
SAVE_DIR = POOL / "research5" / "models" / "residual_flow"


# ── FlowUNet (from flow_downscale.py, adapted) ──────────────────────────────


def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, channels * 2))

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        t_out = self.time_mlp(t_emb)[:, :, None, None]
        scale, shift = t_out.chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = F.silu(h)
        h = self.conv2(h)
        return x + h


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj(h)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.res = ResBlock(in_ch, time_dim)
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        h = self.res(x, t_emb)
        return self.down(h), h


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


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
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


class FlowUNet(nn.Module):
    """UNet for flow matching velocity prediction on residuals.

    Input: (B, 2, 128, 128) — channel 0 = r_t (interpolated residual), channel 1 = LR cond
    Output: (B, 1, 128, 128) — predicted velocity
    """

    def __init__(self, channels=(64, 128, 256), time_dim=128):
        super().__init__()
        self.time_dim = time_dim
        c0, c1, c2 = channels

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2), nn.SiLU(), nn.Linear(time_dim * 2, time_dim))

        self.input_conv = nn.Conv2d(2, c0, 3, padding=1)

        # Encoder: 128 -> 64 -> 32 -> 16
        self.down0 = DownBlock(c0, c1, time_dim)
        self.down1 = DownBlock(c1, c2, time_dim)
        self.down2 = DownBlock(c2, c2, time_dim)

        # Middle with attention at 16x16
        self.mid = ResBlock(c2, time_dim)
        self.mid_attn = SelfAttention(c2, num_heads=4)

        # Decoder
        self.up2 = UpBlock(c2, c2, c2, time_dim)
        self.up1 = UpBlock(c2, c1, c1, time_dim)
        self.up0 = UpBlock(c1, c0, c0, time_dim)

        self.out_norm = nn.GroupNorm(min(8, c0), c0)
        self.out_conv = nn.Conv2d(c0, 1, 1)

    def forward(self, x_t, t, cond):
        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        h = torch.cat([x_t, cond], dim=1)
        h = self.input_conv(h)

        h, s0 = self.down0(h, t_emb)
        h, s1 = self.down1(h, t_emb)
        h, s2 = self.down2(h, t_emb)

        h = self.mid(h, t_emb)
        h = self.mid_attn(h)

        h = self.up2(h, s2, t_emb)
        h = self.up1(h, s1, t_emb)
        h = self.up0(h, s0, t_emb)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


# ── Data ─────────────────────────────────────────────────────────────────────


def load_raw_data(split='train'):
    """Load ERA5 data. Returns (N, 1, H, W) tensors."""
    inputs = torch.load(DATA_DIR / split / f"input_{split}.pt", map_location="cpu", weights_only=True)
    targets = torch.load(DATA_DIR / split / f"target_{split}.pt", map_location="cpu", weights_only=True)
    lr = inputs[:, 0, :, :, :]  # (N, 1, 32, 32)
    hr = targets[:, 0, :, :, :]  # (N, 1, 128, 128)
    return lr, hr


def normalize(x, vmin, vmax):
    return (x - vmin) / (vmax - vmin + 1e-8)


def denormalize(x, vmin, vmax):
    return x * (vmax - vmin) + vmin


def bicubic_upsample(lr, size=128):
    return F.interpolate(lr, size=(size, size), mode='bicubic', align_corners=False)


# ── Precompute SwinIR predictions ────────────────────────────────────────────


def precompute_swinir_preds(args):
    """Run finetuned SwinIR on all splits and save predictions + residuals."""
    device = torch.device('cuda')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load finetuned SwinIR
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from finetune_swinir import load_swinir_1ch

    ft_ckpt = torch.load(
        POOL / "research5" / "models" / "swinir_ft" / "best_swinir.pt",
        map_location="cpu", weights_only=False)
    vmin = ft_ckpt['vmin']
    vmax = ft_ckpt['vmax']

    weights_path = POOL / "research5" / "pretrained_weights" / "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
    model = load_swinir_1ch(str(weights_path))
    model.load_state_dict(ft_ckpt['model'])
    model = model.to(device).eval()
    print(f"Loaded finetuned SwinIR, vmin={vmin:.4f}, vmax={vmax:.4f}")

    for split in ['train', 'val', 'test']:
        print(f"\nPrecomputing {split}...")
        lr, hr = load_raw_data(split)
        lr_n = normalize(lr, vmin, vmax)
        hr_n = normalize(hr, vmin, vmax)

        # Run SwinIR in batches
        preds = []
        bs = 64
        with torch.no_grad():
            for i in range(0, len(lr_n), bs):
                batch = lr_n[i:i+bs].to(device)
                pred = model(batch)
                preds.append(pred.cpu())
                if (i // bs + 1) % 20 == 0:
                    print(f"  {i+bs}/{len(lr_n)}")
        preds = torch.cat(preds, dim=0)  # (N, 1, 128, 128) normalized

        # Compute residuals in normalized space
        residuals = hr_n - preds  # (N, 1, 128, 128)

        # Stats
        rmean = residuals.mean().item()
        rstd = residuals.std().item()
        rmin = residuals.min().item()
        rmax = residuals.max().item()
        mae = residuals.abs().mean().item()
        print(f"  {split} residuals: mean={rmean:.6f}, std={rstd:.6f}, "
              f"range=[{rmin:.4f}, {rmax:.4f}], MAE={mae:.6f}")

        # Save
        torch.save(residuals, save_dir / f'residuals_{split}.pt')
        torch.save(preds, save_dir / f'swinir_preds_{split}.pt')

    # Save config
    config = {'vmin': float(vmin), 'vmax': float(vmax)}
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved precomputed data to {save_dir}")


# ── Training ─────────────────────────────────────────────────────────────────


def train(args):
    device = torch.device('cuda')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(save_dir / 'config.json') as f:
        config = json.load(f)
    vmin, vmax = config['vmin'], config['vmax']

    # Load precomputed residuals and LR data
    print("Loading precomputed data...")
    residuals_train = torch.load(save_dir / 'residuals_train.pt', weights_only=True)
    residuals_val = torch.load(save_dir / 'residuals_val.pt', weights_only=True)
    lr_train, _ = load_raw_data('train')
    lr_val, _ = load_raw_data('val')
    lr_train_n = normalize(lr_train, vmin, vmax)
    lr_val_n = normalize(lr_val, vmin, vmax)

    # Bicubic-upsampled LR as conditioning (precompute to save time)
    print("Precomputing bicubic upsampled conditions...")
    cond_train = bicubic_upsample(lr_train_n)  # (N, 1, 128, 128)
    cond_val = bicubic_upsample(lr_val_n)

    print(f"Train: {len(residuals_train)} samples, Val: {len(residuals_val)} samples")
    print(f"Residual stats: mean={residuals_train.mean():.6f}, std={residuals_train.std():.6f}")

    train_ds = TensorDataset(residuals_train, cond_train)
    val_ds = TensorDataset(residuals_val, cond_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Model
    channels = tuple(int(c) for c in args.channels.split(','))
    model = FlowUNet(channels=channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FlowUNet params: {n_params:,} (channels={channels})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Estimate epochs within wall time: ~2-3 min/epoch with precomputed data
    expected_epochs = min(args.epochs, int(args.wall_hours * 60 / 2.5) + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=expected_epochs)
    scaler = torch.amp.GradScaler('cuda')
    ema = EMA(model, decay=0.999)

    # Resume if checkpoint exists
    start_epoch = 0
    best_val_loss = float('inf')
    ckpt_path = save_dir / 'flow_checkpoint.pth'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        if 'ema' in ckpt:
            ema.load_state_dict(ckpt['ema'])
        print(f"Resumed from epoch {start_epoch}, best_val={best_val_loss:.6f}")

    # Save training config
    train_config = {
        'channels': list(channels), 'lr': args.lr, 'batch_size': args.batch_size,
        'euler_steps': args.euler_steps, 'n_params': n_params,
        'expected_epochs': expected_epochs, 'vmin': vmin, 'vmax': vmax,
    }
    with open(save_dir / 'train_config.json', 'w') as f:
        json.dump(train_config, f, indent=2)

    # Training loop
    train_losses = []
    val_losses = []
    start_time = time.time()
    wall_limit = args.wall_hours * 3600

    print(f"\nTraining for up to {args.epochs} epochs ({args.wall_hours}h wall limit)...")
    print(f"  channels={channels}, LR={args.lr}, BS={args.batch_size}, EMA=0.999")
    print(f"  Expected ~{expected_epochs} epochs, cosine T_max={expected_epochs}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for residual, cond in train_loader:
            residual = residual.to(device)  # (B, 1, 128, 128) target residual
            cond = cond.to(device)          # (B, 1, 128, 128) bicubic LR

            B = residual.shape[0]

            # Flow matching: interpolate between noise and target residual
            x0 = torch.randn_like(residual)
            t = torch.rand(B, device=device)
            t_expand = t[:, None, None, None]

            x_t = (1 - t_expand) * x0 + t_expand * residual
            v_target = residual - x0

            with torch.amp.autocast('cuda'):
                v_pred = model(x_t, t, cond)
                loss = F.mse_loss(v_pred, v_target)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = epoch_loss / n_batches
        train_losses.append(avg_train)

        # Validation with EMA weights
        ema.apply_shadow(model)
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for residual, cond in val_loader:
                residual = residual.to(device)
                cond = cond.to(device)
                B = residual.shape[0]
                x0 = torch.randn_like(residual)
                t = torch.rand(B, device=device)
                t_expand = t[:, None, None, None]
                x_t = (1 - t_expand) * x0 + t_expand * residual
                v_target = residual - x0
                with torch.amp.autocast('cuda'):
                    v_pred = model(x_t, t, cond)
                    val_loss += F.mse_loss(v_pred, v_target).item()
                n_val += 1
        avg_val = val_loss / n_val
        val_losses.append(avg_val)
        ema.restore(model)

        elapsed = time.time() - start_time
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"train={avg_train:.6f} val={avg_val:.6f} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed/60:.1f}min")

        # Save best (EMA weights)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            ema.apply_shadow(model)
            torch.save(model.state_dict(), save_dir / 'flow_best.pth')
            ema.restore(model)

        # Save checkpoint
        ckpt_data = {
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(), 'best_val_loss': best_val_loss,
            'ema': ema.state_dict(),
        }
        torch.save(ckpt_data, save_dir / 'flow_checkpoint.pth')

        # Wall time check
        if elapsed > wall_limit:
            print(f"Wall time limit reached ({elapsed/3600:.2f}h)")
            break

    # Save final EMA weights
    ema.apply_shadow(model)
    torch.save(model.state_dict(), save_dir / 'flow_last.pth')
    ema.restore(model)

    # Save losses
    torch.save({'train': train_losses, 'val': val_losses}, save_dir / 'losses.pt')

    total_min = (time.time() - start_time) / 60
    print(f"\nTraining complete in {total_min:.1f} min. Best val loss: {best_val_loss:.6f}")
    print(f"Trained {epoch+1 - start_epoch} epochs (from {start_epoch} to {epoch})")


# ── Inference ────────────────────────────────────────────────────────────────


@torch.no_grad()
def generate_residual_ensemble(model, cond, n_members, n_steps, device):
    """Generate residual samples via Euler ODE integration.

    Args:
        model: trained FlowUNet
        cond: (B, 1, 128, 128) bicubic-upsampled LR condition (normalized)
        n_members: number of ensemble members
        n_steps: Euler integration steps

    Returns:
        (B, M, 1, 128, 128) residual samples (normalized space)
    """
    B = cond.shape[0]
    dt = 1.0 / n_steps
    all_members = []

    for m in range(n_members):
        x = torch.randn(B, 1, 128, 128, device=device)
        for step in range(n_steps):
            t_cur = torch.full((B,), step * dt, device=device)
            v = model(x, t_cur, cond)
            x = x + v * dt
        all_members.append(x)

    return torch.stack(all_members, dim=1)


def evaluate(args):
    """Generate ensemble predictions and evaluate CRPS."""
    device = torch.device('cuda')
    save_dir = Path(args.save_dir)

    # Load configs
    with open(save_dir / 'config.json') as f:
        config = json.load(f)
    vmin, vmax = config['vmin'], config['vmax']

    with open(save_dir / 'train_config.json') as f:
        train_config = json.load(f)
    channels = tuple(train_config['channels'])
    euler_steps = args.euler_steps or train_config['euler_steps']

    # Load flow model
    model = FlowUNet(channels=channels).to(device)
    ckpt = 'flow_best.pth' if (save_dir / 'flow_best.pth').exists() else 'flow_last.pth'
    model.load_state_dict(torch.load(save_dir / ckpt, weights_only=False, map_location=device))
    model.eval()
    print(f"Loaded {ckpt}, channels={channels}, euler_steps={euler_steps}, "
          f"n_members={args.n_members}")

    # Load SwinIR predictions (precomputed)
    swinir_preds = torch.load(save_dir / 'swinir_preds_test.pt', weights_only=True)

    # Load test data
    lr_test, hr_test = load_raw_data('test')
    lr_test_n = normalize(lr_test, vmin, vmax)
    hr_test_n = normalize(hr_test, vmin, vmax)

    # Generate ensemble: swinir_pred + flow_residual_k
    all_preds = []
    all_targets = []
    all_inputs = []
    eval_bs = args.eval_batch_size
    t_start = time.time()

    for i in range(0, len(lr_test_n), eval_bs):
        lr_batch = lr_test_n[i:i+eval_bs].to(device)
        swinir_batch = swinir_preds[i:i+eval_bs].to(device)
        cond = bicubic_upsample(lr_batch)

        # Generate residuals from flow
        residuals = generate_residual_ensemble(
            model, cond, args.n_members, euler_steps, device)  # (B, M, 1, 128, 128)

        # Ensemble = swinir_pred + residual
        ensemble = swinir_batch.unsqueeze(1) + residuals  # (B, M, 1, 128, 128)

        # Denormalize
        ensemble_denorm = denormalize(ensemble, vmin, vmax)
        hr_denorm = denormalize(hr_test_n[i:i+eval_bs], vmin, vmax)
        lr_denorm = denormalize(lr_test_n[i:i+eval_bs], vmin, vmax)

        all_preds.append(ensemble_denorm.cpu())
        all_targets.append(hr_denorm)
        all_inputs.append(lr_denorm)

        if (i // eval_bs + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  Batch {i+eval_bs}/{len(lr_test_n)} ({elapsed:.0f}s)")

    preds = torch.cat(all_preds, dim=0)      # (N, M, 1, 128, 128)
    targets = torch.cat(all_targets, dim=0)    # (N, 1, 128, 128)
    inputs = torch.cat(all_inputs, dim=0)      # (N, 1, 32, 32)

    gen_time = time.time() - t_start
    print(f"Generation done in {gen_time:.0f}s")

    compute_metrics(preds, targets, inputs)

    # Also evaluate with AddCL constraint
    print("\n--- With AddCL constraint ---")
    compute_metrics_addcl(preds, targets, inputs)


def compute_metrics(preds, targets, inputs):
    """Compute CRPS and other metrics."""
    N, M = preds.shape[0], preds.shape[1]
    pred_np = preds[:, :, 0, :, :].numpy()
    target_np = targets[:, 0, :, :].numpy()
    mean_pred = preds.mean(dim=1)[:, 0, :, :].numpy()
    input_np = inputs[:, 0, :, :].numpy()

    mse = float(np.mean((mean_pred - target_np) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(mean_pred - target_np)))

    # CRPS (energy score, correct formula)
    crps_vals = []
    for i in range(N):
        fc = pred_np[i]
        obs = target_np[i]
        mae_term = np.mean(np.abs(fc - obs[None, ...]), axis=0)
        fc_sorted = np.sort(fc, axis=0)
        spread = np.zeros_like(obs)
        for j in range(M):
            w = (2.0 * (j + 1) - M - 1.0) / (M * M)
            spread += w * fc_sorted[j]
        crps_vals.append(float(np.mean(mae_term - spread)))
    crps = float(np.mean(crps_vals))

    spread_val = float(np.mean(np.std(pred_np, axis=1)))

    # Mass violation
    mass_viol = 0.0
    for i in range(N):
        ds = mean_pred[i].reshape(32, 4, 32, 4).mean(axis=(1, 3))
        mass_viol += float(np.mean(np.abs(ds - input_np[i])))
    mass_viol /= N

    print(f"\n{'='*40}")
    print(f"CRPS:            {crps:.6f}")
    print(f"RMSE:            {rmse:.6f}")
    print(f"MAE:             {mae:.6f}")
    print(f"Spread:          {spread_val:.6f}")
    print(f"Mass Violation:  {mass_viol:.6f}")
    print(f"Ensemble Size:   {M}")
    print(f"N Samples:       {N}")
    print(f"{'='*40}")


def compute_metrics_addcl(preds, targets, inputs):
    """Same metrics but with AddCL constraint applied to each member."""
    N, M = preds.shape[0], preds.shape[1]
    pred_np = preds[:, :, 0, :, :].numpy().copy()
    target_np = targets[:, 0, :, :].numpy()
    input_np = inputs[:, 0, :, :].numpy()

    # Apply AddCL to each ensemble member
    for m in range(M):
        for i in range(N):
            ds = pred_np[i, m].reshape(32, 4, 32, 4).mean(axis=(1, 3))
            correction = input_np[i] - ds
            correction_hr = np.repeat(np.repeat(correction, 4, axis=0), 4, axis=1)
            pred_np[i, m] += correction_hr

    mean_pred = pred_np.mean(axis=1)
    mse = float(np.mean((mean_pred - target_np) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(mean_pred - target_np)))

    crps_vals = []
    for i in range(N):
        fc = pred_np[i]
        obs = target_np[i]
        mae_term = np.mean(np.abs(fc - obs[None, ...]), axis=0)
        fc_sorted = np.sort(fc, axis=0)
        spread = np.zeros_like(obs)
        for j in range(M):
            w = (2.0 * (j + 1) - M - 1.0) / (M * M)
            spread += w * fc_sorted[j]
        crps_vals.append(float(np.mean(mae_term - spread)))
    crps = float(np.mean(crps_vals))

    spread_val = float(np.mean(np.std(pred_np, axis=1)))

    mass_viol = 0.0
    for i in range(N):
        ds = mean_pred[i].reshape(32, 4, 32, 4).mean(axis=(1, 3))
        mass_viol += float(np.mean(np.abs(ds - input_np[i])))
    mass_viol /= N

    print(f"CRPS (AddCL):    {crps:.6f}")
    print(f"RMSE (AddCL):    {rmse:.6f}")
    print(f"MAE (AddCL):     {mae:.6f}")
    print(f"Spread (AddCL):  {spread_val:.6f}")
    print(f"Mass Viol (AddCL): {mass_viol:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['precompute', 'train', 'eval', 'all'], default='all')
    parser.add_argument('--save-dir', default=str(SAVE_DIR))
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--channels', default='64,128,256')
    parser.add_argument('--euler-steps', type=int, default=20)
    parser.add_argument('--n-members', type=int, default=10)
    parser.add_argument('--wall-hours', type=float, default=2.0)
    args = parser.parse_args()

    if args.mode == 'all':
        precompute_swinir_preds(args)
        train(args)
        evaluate(args)
    elif args.mode == 'precompute':
        precompute_swinir_preds(args)
    elif args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        evaluate(args)
