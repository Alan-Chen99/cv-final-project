"""
DiT Flow Matching: Diffusion Transformer + OT-CFM for 32x32 -> 128x128 downscaling.

Replaces the UNet backbone from flow_matching_v2.py with a Vision Transformer
using AdaLN-Zero conditioning (as in DiT paper). All data loading, CRPS,
constraint layers, sampling, and evaluation code is copied from flow_matching_v2.py.

Architecture:
  - Patchify: 128x128 with patch_size=8 -> 16x16 = 256 tokens
  - Input: 2 channels (noisy state + LR condition) -> patch embedding
  - Learnable 2D positional embedding for 16x16 grid
  - N transformer blocks with AdaLN-Zero time conditioning
  - Unpatchify: linear projection back to patch_size^2 * out_channels

Usage:
  python dit_flow.py --mode train --epochs 40 --batch_size 64
  python dit_flow.py --mode eval --n_ensemble 10 --split test
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


# ---------- DiT Modules ----------

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


class TimestepEmbedder(nn.Module):
    """Maps scalar timestep to embedding vector."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            SinusoidalPositionEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        return self.mlp(t * 1000.0)


class PatchEmbed(nn.Module):
    """Convert 2D image into patch tokens via convolution."""

    def __init__(self, patch_size, in_channels, hidden_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, hidden_dim, H/ps, W/ps) -> (B, N, hidden_dim)
        x = self.proj(x)  # (B, hidden_dim, grid_h, grid_w)
        x = x.flatten(2).transpose(1, 2)  # (B, N, hidden_dim)
        return x


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning.

    AdaLN-Zero: time embedding produces (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
    where gamma/beta modulate LayerNorm and alpha gates the residual addition.
    All alpha params initialized to zero so each block starts as identity.
    """

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )
        # AdaLN-Zero: 6 modulation params per block
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        # Initialize last linear to zero so blocks start as identity
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, t_emb):
        # t_emb: (B, hidden_dim) -> modulation params (B, 6*hidden_dim)
        mod = self.adaLN_modulation(t_emb)  # (B, 6*hidden_dim)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)
        # Each is (B, hidden_dim) -> unsqueeze for (B, 1, hidden_dim) broadcast

        # Attention branch with AdaLN
        h = self.norm1(x)
        h = h * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + alpha1.unsqueeze(1) * h

        # MLP branch with AdaLN
        h = self.norm2(x)
        h = h * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        h = self.mlp(h)
        x = x + alpha2.unsqueeze(1) * h

        return x


class FinalLayer(nn.Module):
    """Final AdaLN + linear projection to unpatchify."""

    def __init__(self, hidden_dim, patch_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        # Initialize to zero
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, t_emb):
        mod = self.adaLN_modulation(t_emb)
        gamma, beta = mod.chunk(2, dim=-1)
        h = self.norm(x)
        h = h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        h = self.linear(h)
        return h


class DiT(nn.Module):
    """Diffusion Transformer for velocity prediction.

    Input: noisy state (1 ch) + LR condition (1 ch) = 2 ch at 128x128
    Output: predicted velocity (1 ch) at 128x128
    """

    def __init__(self, img_size=128, patch_size=8, in_channels=2, out_channels=1,
                 hidden_dim=512, depth=12, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.grid_size = img_size // patch_size  # 16
        self.num_patches = self.grid_size ** 2   # 256

        # Patch embedding
        self.patch_embed = PatchEmbed(patch_size, in_channels, hidden_dim)

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Timestep embedder
        self.time_embed = TimestepEmbedder(hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Final layer
        self.final_layer = FinalLayer(hidden_dim, patch_size, out_channels)

    def unpatchify(self, x):
        """Convert patch tokens back to image.

        x: (B, N, patch_size^2 * out_channels) -> (B, out_channels, H, W)
        """
        ps = self.patch_size
        gs = self.grid_size
        c = self.out_channels
        x = x.reshape(-1, gs, gs, ps, ps, c)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, gs, ps, gs, ps)
        x = x.reshape(-1, c, gs * ps, gs * ps)
        return x

    def forward(self, x, t, condition):
        """
        x: (B, 1, 128, 128) - noisy state
        t: (B,) - timestep in [0, 1]
        condition: (B, 1, 128, 128) - LR condition
        Returns: (B, 1, 128, 128) - predicted velocity
        """
        # Concatenate noisy state and condition
        x = torch.cat([x, condition], dim=1)  # (B, 2, 128, 128)

        # Patchify + position embedding
        x = self.patch_embed(x)  # (B, 256, hidden_dim)
        x = x + self.pos_embed   # (B, 256, hidden_dim)

        # Timestep embedding
        t_emb = self.time_embed(t)  # (B, hidden_dim)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, t_emb)

        # Final layer -> unpatchify
        x = self.final_layer(x, t_emb)  # (B, 256, patch_size^2 * out_channels)
        x = self.unpatchify(x)           # (B, 1, 128, 128)

        return x


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
    """Standard energy CRPS: E|X-y| - 0.5*E|X-X'| (Gneiting sorted method, M^2 denom).
    This is the canonical energy score for a finite M-member ensemble."""
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
    model = DiT(
        img_size=128,
        patch_size=args.patch_size,
        in_channels=2,
        out_channels=1,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=0.1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DiT config: patch_size={args.patch_size}, hidden_dim={args.hidden_dim}, "
          f"depth={args.depth}, num_heads={args.num_heads}, mlp_ratio={args.mlp_ratio}")
    print(f"#params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        for lr_batch, res_batch in train_loader:
            lr_batch = lr_batch.to(device)
            res_batch = res_batch.to(device)

            bs = lr_batch.shape[0]
            t = torch.rand(bs, device=device)
            x_0 = torch.randn_like(res_batch)
            t_expand = t[:, None, None, None]
            x_t = (1 - t_expand) * x_0 + t_expand * res_batch
            target_v = res_batch - x_0

            pred_v = model(x_t, t, lr_batch)
            loss = F.mse_loss(pred_v, target_v)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validation
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
            }, os.path.join(args.save_dir, 'best_flow.pt'))

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
    model = DiT(
        img_size=128,
        patch_size=saved_args.get('patch_size', 8),
        in_channels=2,
        out_channels=1,
        hidden_dim=saved_args.get('hidden_dim', 512),
        depth=saved_args.get('depth', 12),
        num_heads=saved_args.get('num_heads', 8),
        mlp_ratio=saved_args.get('mlp_ratio', 4.0),
        dropout=0.0,
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
    print(f"DiT config: patch_size={saved_args.get('patch_size', 8)}, "
          f"hidden_dim={saved_args.get('hidden_dim', 512)}, "
          f"depth={saved_args.get('depth', 12)}, "
          f"num_heads={saved_args.get('num_heads', 8)}")

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
    parser.add_argument("--save_dir", default="models/dit_flow")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finetune_lr", type=float, default=None,
                        help="Fresh LR + cosine schedule for fine-tuning after resume")
    # DiT architecture args
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
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

    if args.mode == "train":
        train(args)
    else:
        evaluate(args)
