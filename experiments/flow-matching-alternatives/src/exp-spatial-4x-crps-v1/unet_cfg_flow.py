"""
UNet + Classifier-Free Guidance (CFG) Flow Matching for 32x32 -> 128x128 downscaling.

Based on flow_matching_v2.py AttentionUNet with OT-CFM, adding:
  - Training: random condition dropout (replace LR with zeros) with probability cfg_prob
  - Inference: guided sampling v = v_uncond + s*(v_cond - v_uncond)
  - s=1.0 is standard conditional; s>1.0 amplifies conditioning effect
  - Correct Gneiting M^2 CRPS formula throughout

Usage:
  python unet_cfg_flow.py --mode train --epochs 80 --cfg_prob 0.1
  python unet_cfg_flow.py --mode eval --guidance_scale 1.0 --n_ensemble 10
  python unet_cfg_flow.py --mode eval_sweep  # sweep guidance scales
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------- Architecture (AttentionUNet from flow_matching_v2.py) ----------

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
        q, k, v = qkv.unbind(dim=1)
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H * W)
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

        self.mid_block1 = ResBlock(ch, ch, time_emb_dim, dropout)
        self.mid_attn = SelfAttention(ch, num_heads=attn_heads)
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim, dropout)

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


# ---------- Data loading ----------

def load_tcw4_data(basedir, split='train'):
    inp = torch.load(f'{basedir}/data/era5_sr_data/{split}/input_{split}.pt', weights_only=False)
    tgt = torch.load(f'{basedir}/data/era5_sr_data/{split}/target_{split}.pt', weights_only=False)
    lr = inp[:, 0, :, :, :]
    hr = tgt[:, 0, :, :, :]
    lr_up = F.interpolate(lr, size=(128, 128), mode='bilinear', align_corners=False)
    residual = hr - lr_up
    return lr_up, residual, hr, lr


# ---------- CRPS (Gneiting M^2 formula) ----------

def crps_gneiting(observation, forecasts):
    """Standard energy CRPS: E|X-y| - 0.5*E|X-X'| using Gneiting sorted formula.
    Uses M^2 denominator (includes self-pairs). This is the canonical definition."""
    M = forecasts.shape[0]
    mae_term = np.mean(np.abs(forecasts - observation[None, ...]), axis=0)
    fc_sorted = np.sort(forecasts, axis=0)
    spread = np.zeros_like(observation)
    for j in range(M):
        w = (2.0 * (j + 1) - M - 1.0) / (M * M)
        spread += w * fc_sorted[j]
    return np.mean(mae_term - spread)


# ---------- Constraint layers ----------

def apply_addcl(pred_hr, lr_orig, upsampling_factor=4):
    pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
    pooled = pool(pred_hr)
    correction = lr_orig - pooled
    correction_hr = correction.repeat_interleave(upsampling_factor, dim=-2).repeat_interleave(upsampling_factor, dim=-1)
    return pred_hr + correction_hr


def apply_smcl(pred_hr, lr_orig, upsampling_factor=4):
    """Softmax constraint layer (SmCL) from Harder et al. 2208.05424."""
    pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
    y = torch.exp(pred_hr)
    sum_y = pool(y)
    ratio = lr_orig / sum_y
    ratio_hr = ratio.repeat_interleave(upsampling_factor, dim=-2).repeat_interleave(upsampling_factor, dim=-1)
    return y * ratio_hr


# ---------- Spectral Loss ----------

def spectral_loss(pred, target):
    """L1 loss in Fourier domain — penalizes power spectrum mismatch.
    Operates on 2D spatial dims of (B,C,H,W) tensors."""
    fft_pred = torch.fft.rfft2(pred, norm='ortho')
    fft_target = torch.fft.rfft2(target, norm='ortho')
    return F.l1_loss(torch.abs(fft_pred), torch.abs(fft_target))


# ---------- Data Augmentation ----------

def random_flip_batch(lr_batch, res_batch):
    """Apply consistent random horizontal + vertical flips to LR and residual."""
    if torch.rand(1).item() > 0.5:
        lr_batch = torch.flip(lr_batch, [-1])
        res_batch = torch.flip(res_batch, [-1])
    if torch.rand(1).item() > 0.5:
        lr_batch = torch.flip(lr_batch, [-2])
        res_batch = torch.flip(res_batch, [-2])
    return lr_batch, res_batch


# ---------- ODE Sampling with CFG ----------

@torch.no_grad()
def euler_sample_cfg(model, condition, shape, steps=10, guidance_scale=1.0):
    """Euler ODE with classifier-free guidance.
    guidance_scale=1.0 is standard conditional; >1.0 amplifies conditioning."""
    device = condition.device
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    use_guidance = (guidance_scale != 1.0)
    uncond = torch.zeros_like(condition) if use_guidance else None

    for i in range(steps):
        t = torch.full((shape[0],), i * dt, device=device)
        v_cond = model(x, t, condition)
        if use_guidance:
            v_uncond = model(x, t, uncond)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            v = v_cond
        x = x + v * dt
    return x


@torch.no_grad()
def heun_sample_cfg(model, condition, shape, steps=10, guidance_scale=1.0):
    """Heun's method (2nd-order) ODE solver with optional CFG.
    Each step uses 2 function evaluations (NFE = 2*steps)."""
    device = condition.device
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    use_guidance = (guidance_scale != 1.0)
    uncond = torch.zeros_like(condition) if use_guidance else None

    def get_velocity(x_in, t_val):
        t_tensor = torch.full((shape[0],), t_val, device=device)
        v_cond = model(x_in, t_tensor, condition)
        if use_guidance:
            v_uncond = model(x_in, t_tensor, uncond)
            return v_uncond + guidance_scale * (v_cond - v_uncond)
        return v_cond

    for i in range(steps):
        t_i = i * dt
        v1 = get_velocity(x, t_i)
        x_euler = x + v1 * dt
        t_next = min((i + 1) * dt, 1.0)
        v2 = get_velocity(x_euler, t_next)
        x = x + 0.5 * (v1 + v2) * dt
    return x


# ---------- Training ----------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    basedir = args.basedir

    print("Loading data...")
    lr_up_train, res_train, _, _ = load_tcw4_data(basedir, 'train')
    lr_up_val, res_val, _, _ = load_tcw4_data(basedir, 'val')

    res_mean = res_train.mean()
    res_std = res_train.std()
    res_train_norm = (res_train - res_mean) / res_std
    res_val_norm = (res_val - res_mean) / res_std

    lr_mean = lr_up_train.mean()
    lr_std = lr_up_train.std()
    lr_up_train_norm = (lr_up_train - lr_mean) / lr_std
    lr_up_val_norm = (lr_up_val - lr_mean) / lr_std

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

    model = AttentionUNet(
        in_channels=2, out_channels=1,
        base_channels=args.base_channels,
        channel_mults=args.channel_mults_tuple,
        time_emb_dim=256,
        dropout=0.1,
        attn_heads=args.attn_heads,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"AttentionUNet: base_ch={args.base_channels}, mults={args.channel_mults_tuple}, "
          f"attn_heads={args.attn_heads}")
    print(f"#params: {n_params:,}")
    print(f"CFG prob: {args.cfg_prob}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    start_epoch = 0

    ckpt_path = os.path.join(args.save_dir, 'best_flow.pt')
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['val_loss']
        if 'optimizer' in ckpt and not args.finetune_lr:
            optimizer.load_state_dict(ckpt['optimizer'])
        if args.finetune_lr:
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

    start_time = time.time()

    use_spectral = getattr(args, 'spectral_weight', 0.0) > 0
    use_augment = getattr(args, 'augment', False)
    use_logit_normal = getattr(args, 't_schedule', 'uniform') == 'logit_normal'
    if use_spectral:
        print(f"Spectral loss weight: {args.spectral_weight}")
    if use_augment:
        print("Data augmentation: random H/V flips")
    if use_logit_normal:
        print(f"Time schedule: logit-normal (mean={args.logit_normal_mean}, std={args.logit_normal_std})")
    else:
        print("Time schedule: uniform")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        for lr_batch, res_batch in train_loader:
            lr_batch = lr_batch.to(device)
            res_batch = res_batch.to(device)
            bs = lr_batch.shape[0]

            # Data augmentation: random flips (applied before CFG dropout)
            if use_augment:
                lr_batch, res_batch = random_flip_batch(lr_batch, res_batch)

            # CFG: per-sample condition dropout
            if args.cfg_prob > 0:
                mask = (torch.rand(bs, 1, 1, 1, device=device) > args.cfg_prob).float()
                lr_batch_cond = lr_batch * mask
            else:
                lr_batch_cond = lr_batch

            # Time sampling: uniform or logit-normal
            if use_logit_normal:
                z = torch.randn(bs, device=device)
                t = torch.sigmoid(args.logit_normal_mean + args.logit_normal_std * z)
            else:
                t = torch.rand(bs, device=device)
            x_0 = torch.randn_like(res_batch)
            t_expand = t[:, None, None, None]
            x_t = (1 - t_expand) * x_0 + t_expand * res_batch
            target_v = res_batch - x_0

            pred_v = model(x_t, t, lr_batch_cond)
            loss = F.mse_loss(pred_v, target_v)

            # Spectral loss: penalize FFT mismatch on reconstructed x_1
            if use_spectral:
                x1_pred = x_t + (1 - t_expand) * pred_v
                x1_target = res_batch
                loss = loss + args.spectral_weight * spectral_loss(x1_pred, x1_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validation (always conditional, no dropout)
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
                pred_v = model(x_t, t, lr_batch)
                val_loss += F.mse_loss(pred_v, target_v).item()
        val_loss /= len(val_loader)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs}, Train: {train_loss:.6f}, Val: {val_loss:.6f}, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}, Time: {elapsed/60:.1f}min")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'args': vars(args),
            }, ckpt_path)

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
    guidance_scale = args.guidance_scale

    solver = getattr(args, 'solver', 'euler')
    sample_fn = heun_sample_cfg if solver == 'heun' else euler_sample_cfg

    print(f"Evaluating {n_samples} samples, {n_ensemble} ensemble, "
          f"{ode_steps} {solver} steps, constraint={use_constraint}, guidance={guidance_scale}")
    print(f"Model epoch: {ckpt['epoch']+1}, val_loss: {ckpt['val_loss']:.6f}")
    print(f"CFG training prob: {saved_args.get('cfg_prob', 0.0)}")

    all_crps = []
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
        for e in range(n_ensemble):
            with torch.no_grad():
                sampled_res_norm = sample_fn(
                    model, batch_lr,
                    shape=(bs, 1, 128, 128),
                    steps=ode_steps,
                    guidance_scale=guidance_scale,
                )
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

            all_crps.append(crps_gneiting(gt, ens))
            all_mae.append(np.mean(np.abs(gt - ens_mean)))
            all_rmse.append(np.mean((gt - ens_mean) ** 2))

            pred_mean_t = torch.from_numpy(ens_mean).unsqueeze(0).unsqueeze(0)
            pooled = pool(pred_mean_t).squeeze()
            lr_i = batch_lr_orig[i, 0, ...]
            all_mass_viol.append(torch.mean(torch.abs(pooled - lr_i)).item())

        if (start_idx // batch_size) % 10 == 0:
            print(f"  Processed {end_idx}/{n_samples}...")

    crps = np.mean(all_crps)
    mae = np.mean(all_mae)
    rmse = np.sqrt(np.mean(all_rmse))
    mass_viol = np.mean(all_mass_viol)

    # Compute ensemble spread for diagnostics
    # (mean pairwise distance, Gneiting formula)
    print(f"\nResults ({args.split}, {n_ensemble} ens, {ode_steps} steps, "
          f"constraint={use_constraint}, guidance={guidance_scale}):")
    print(f"  CRPS (Gneiting M^2): {crps:.6f}")
    print(f"  MAE:                 {mae:.6f}")
    print(f"  RMSE:                {rmse:.6f}")
    print(f"  Mass violation:      {mass_viol:.6f}")

    return crps, mae, rmse, mass_viol


def eval_sweep(args):
    """Sweep guidance scales and report results."""
    scales = [float(s) for s in args.sweep_scales.split(',')]
    print(f"Sweeping guidance scales: {scales}")
    results = []
    for s in scales:
        args.guidance_scale = s
        crps, mae, rmse, mass_viol = evaluate(args)
        results.append((s, crps, mae, rmse, mass_viol))
        print()

    print("\n" + "="*80)
    print(f"Guidance Scale Sweep Results ({args.split}, {args.n_ensemble} ens, "
          f"{args.ode_steps} steps, constraint={args.constraint}):")
    print(f"{'Scale':>8} {'CRPS':>10} {'MAE':>10} {'RMSE':>10} {'MassViol':>10}")
    print("-"*50)
    for s, crps, mae, rmse, mv in results:
        print(f"{s:>8.1f} {crps:>10.6f} {mae:>10.6f} {rmse:>10.6f} {mv:>10.6f}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "eval_sweep"], required=True)
    parser.add_argument("--basedir", default="external/constrained-downscaling")
    parser.add_argument("--save_dir", default="models/unet_cfg")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finetune_lr", type=float, default=None)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--channel_mults", type=str, default="1,2,4")
    parser.add_argument("--attn_heads", type=int, default=4)
    # CFG args
    parser.add_argument("--cfg_prob", type=float, default=0.1,
                        help="Probability of dropping condition during training (0=no CFG)")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="Guidance scale for sampling (1.0=standard conditional)")
    parser.add_argument("--sweep_scales", type=str, default="0.5,1.0,1.5,2.0,3.0",
                        help="Comma-separated guidance scales for eval_sweep mode")
    # Training recipe args
    parser.add_argument("--augment", action="store_true",
                        help="Enable random H/V flip augmentation")
    parser.add_argument("--spectral_weight", type=float, default=0.0,
                        help="Weight for spectral (FFT) loss (0=disabled)")
    parser.add_argument("--t_schedule", default="uniform", choices=["uniform", "logit_normal"],
                        help="Time sampling schedule (uniform or logit_normal)")
    parser.add_argument("--logit_normal_mean", type=float, default=0.0,
                        help="Mean of logit-normal distribution for t sampling")
    parser.add_argument("--logit_normal_std", type=float, default=1.0,
                        help="Std of logit-normal distribution for t sampling")
    # Eval args
    parser.add_argument("--n_ensemble", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--ode_steps", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--constraint", default="addcl", choices=["none", "addcl", "smcl"])
    parser.add_argument("--solver", default="euler", choices=["euler", "heun"],
                        help="ODE solver: euler (1st-order) or heun (2nd-order, 2x NFE)")
    args = parser.parse_args()

    args.channel_mults_tuple = tuple(int(x) for x in args.channel_mults.split(','))

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
    elif args.mode == "eval_sweep":
        eval_sweep(args)
