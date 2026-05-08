"""Diffusion Transformer (DiT) for flow matching on climate downscaling residuals.

All downscaling diffusion/flow papers use UNet backbones. This tests whether
transformer-based score networks (DiT) can match or exceed UNets for climate
downscaling — explicitly identified as under-explored in research direction #7.

Architecture:
  - Patches 128x128 input into 4x4 patches → 1024 tokens
  - N transformer blocks with adaLN-Zero (time conditioning)
  - Global self-attention captures long-range spatial correlations
  - Unpatchify → velocity field for flow matching

Training: OT-CFM on residuals (HR - bilinear(LR)), same as flow_matching_v2.py.

Usage:
    python experiments/pretrained-sr-downscaling/src/train_dit_flow.py --mode train --epochs 50 --batch_size 64
    python experiments/pretrained-sr-downscaling/src/train_dit_flow.py --mode eval --n_ensemble 10 --split test
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
from pathlib import Path

POOL = Path("/home/chenxy/orcd/pool/datasets")


# ---------- DiT Architecture ----------

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


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning (from DiT paper).
    Uses F.scaled_dot_product_attention for flash attention (memory efficient)."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # QKV projection (single linear for efficiency)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attn_proj = nn.Linear(dim, dim)
        self.attn_drop = dropout
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        # adaLN-Zero: produces (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        # Initialize adaLN output to zero (adaLN-Zero)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        # c: (B, dim) conditioning vector
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        # Self-attention branch with flash attention
        B, N, C = x.shape
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, heads, N, head_dim)
        h = F.scaled_dot_product_attention(q, k, v,
                                           dropout_p=self.attn_drop if self.training else 0.0)
        h = h.transpose(1, 2).reshape(B, N, C)
        h = self.attn_proj(h)
        x = x + gate_msa.unsqueeze(1) * h

        # MLP branch
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * h

        return x


class FlowDiT(nn.Module):
    """Diffusion/Flow Transformer for 128x128 velocity prediction.

    Replaces UNet with patch-based transformer + adaLN-Zero time conditioning.
    """

    def __init__(self, in_channels=2, out_channels=1, patch_size=8,
                 dim=384, depth=8, num_heads=6, mlp_ratio=4.0, dropout=0.1,
                 img_size=128):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.dim = dim
        self.img_size = img_size
        num_patches = (img_size // patch_size) ** 2  # 256 for 8x8 patches

        # Patch embedding via conv
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size,
                                     stride=patch_size)

        # Learnable position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])

        # Final layer: adaLN + linear to patch pixels
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True),
        )
        self.final_linear = nn.Linear(dim, patch_size * patch_size * out_channels)

        # Small conv refinement to smooth patch boundary artifacts
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, out_channels, 3, padding=1),
        )

        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # Zero-init final layer for stable training start
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)
        # Zero-init refinement so it starts as identity
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def unpatchify(self, x):
        """(B, num_patches, patch_size**2 * C) -> (B, C, H, W)"""
        p = self.patch_size
        c = self.out_channels
        h = w = self.img_size // p
        x = x.reshape(-1, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.reshape(-1, c, h * p, w * p)

    def forward(self, x_t, t, condition):
        """
        x_t: (B, 1, 128, 128) - noisy/interpolated state
        t: (B,) - time in [0, 1]
        condition: (B, 1, 128, 128) - LR upsampled
        Returns: (B, 1, 128, 128) - predicted velocity
        """
        # Concatenate input and condition
        x = torch.cat([x_t, condition], dim=1)  # (B, 2, 128, 128)

        # Patch embedding + position embedding
        x = self.patch_embed(x)  # (B, dim, 32, 32)
        x = x.flatten(2).transpose(1, 2)  # (B, 1024, dim)
        x = x + self.pos_embed

        # Time conditioning
        c = self.time_embed(t * 1000.0)  # (B, dim)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # Final layer
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.final_linear(x)  # (B, 1024, patch_size**2)

        x = self.unpatchify(x)
        return x + self.refine(x)  # residual refinement


# ---------- Data loading ----------

def load_tcw4_data(basedir, split='train'):
    """Load TCW4 data. Returns (lr_up, residual, hr, lr_orig)."""
    # Try pool path first, then legacy basedir path
    pool_path = POOL / "era5_sr_data"
    if (pool_path / split).exists():
        data_root = pool_path
    else:
        data_root = Path(basedir) / "data" / "era5_sr_data"
    inp = torch.load(f'{data_root}/{split}/input_{split}.pt', weights_only=False)
    tgt = torch.load(f'{data_root}/{split}/target_{split}.pt', weights_only=False)
    lr = inp[:, 0, :, :, :]  # (N, 1, 32, 32)
    hr = tgt[:, 0, :, :, :]  # (N, 1, 128, 128)
    lr_up = F.interpolate(lr, size=(128, 128), mode='bilinear', align_corners=False)
    residual = hr - lr_up
    return lr_up, residual, hr, lr


# ---------- CRPS ----------

def crps_ensemble_correct(observation, forecasts):
    """Correct energy CRPS: E|X-y| - 0.5*E|X-X'|."""
    M = forecasts.shape[0]
    abs_diff = np.mean(np.abs(forecasts - observation[None, ...]), axis=0)
    spread = 0.0
    if M > 1:
        for i in range(M):
            for j in range(i + 1, M):
                spread += np.mean(np.abs(forecasts[j] - forecasts[i]))
        spread = spread * 2.0 / (M * (M - 1))
    crps = np.mean(abs_diff) - 0.5 * spread
    return crps


# ---------- Constraint layers ----------

def apply_addcl(pred_hr, lr_orig, upsampling_factor=4):
    """AddCL: additive correction so avgpool(pred_hr) == lr_orig."""
    pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
    pooled = pool(pred_hr)
    correction = lr_orig - pooled
    correction_hr = correction.repeat_interleave(upsampling_factor, dim=-2) \
                              .repeat_interleave(upsampling_factor, dim=-1)
    return pred_hr + correction_hr


# ---------- ODE Sampling ----------

@torch.no_grad()
def euler_sample(model, condition, shape, steps=10):
    """Euler ODE from noise (t=0) to data (t=1)."""
    device = condition.device
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((shape[0],), i * dt, device=device)
        v = model(x, t, condition)
        x = x + v * dt
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
    os.makedirs(args.save_dir, exist_ok=True)
    stats = {'res_mean': res_mean.item(), 'res_std': res_std.item(),
             'lr_mean': lr_mean.item(), 'lr_std': lr_std.item()}
    torch.save(stats, os.path.join(args.save_dir, 'norm_stats.pt'))

    train_ds = TensorDataset(lr_up_train_norm, res_train_norm)
    val_ds = TensorDataset(lr_up_val_norm, res_val_norm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Model
    model = FlowDiT(
        in_channels=2, out_channels=1,
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: FlowDiT (patch={args.patch_size}, dim={args.dim}, depth={args.depth}, "
          f"heads={args.num_heads})")
    print(f"#params: {n_params:,} total, {n_train:,} trainable")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    start_time = time.time()
    wall_limit = args.wall_limit * 60  # minutes to seconds

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for lr_batch, res_batch in train_loader:
            lr_batch = lr_batch.to(device)
            res_batch = res_batch.to(device)
            bs = lr_batch.shape[0]

            # OT-CFM: x_t = (1-t)*noise + t*data, velocity = data - noise
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
                'epoch': epoch,
                'val_loss': val_loss,
                'args': vars(args),
            }, os.path.join(args.save_dir, 'best_flow.pt'))

        # Wall-clock limit
        if elapsed > wall_limit:
            print(f"Wall-clock limit ({args.wall_limit}min) reached at epoch {epoch+1}")
            break

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
    model = FlowDiT(
        in_channels=2, out_channels=1,
        patch_size=saved_args.get('patch_size', 8),
        dim=saved_args.get('dim', 384),
        depth=saved_args.get('depth', 8),
        num_heads=saved_args.get('num_heads', 6),
        mlp_ratio=saved_args.get('mlp_ratio', 4.0),
        dropout=0.0,
        img_size=128,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    n_samples = min(lr_up.shape[0], args.max_samples) if args.max_samples else lr_up.shape[0]
    n_ensemble = args.n_ensemble
    batch_size = args.eval_batch_size
    ode_steps = args.ode_steps
    use_constraint = args.constraint

    print(f"Evaluating {n_samples} samples, {n_ensemble} ensemble, "
          f"{ode_steps} Euler steps, constraint={use_constraint}...")
    print(f"Model: epoch {ckpt['epoch']+1}, val_loss: {ckpt['val_loss']:.6f}")
    print(f"DiT config: dim={saved_args.get('dim')}, depth={saved_args.get('depth')}, "
          f"heads={saved_args.get('num_heads')}")

    all_crps = []
    all_mae = []
    all_rmse = []
    all_mass_viol = []
    all_spread = []

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
                sampled_res_norm = euler_sample(
                    model, batch_lr,
                    shape=(bs, 1, 128, 128),
                    steps=ode_steps,
                )
                sampled_res = sampled_res_norm.cpu() * stats['res_std'] + stats['res_mean']
                pred_hr = batch_lr_up + sampled_res

                if use_constraint == 'addcl':
                    pred_hr = apply_addcl(pred_hr, batch_lr_orig)

                ensemble_preds.append(pred_hr.numpy())

        ensemble_preds = np.stack(ensemble_preds, axis=1)  # (bs, n_ens, 1, H, W)

        for i in range(bs):
            gt = batch_hr[i, 0, ...].numpy()
            ens = ensemble_preds[i, :, 0, ...]  # (n_ens, H, W)
            ens_mean = ens.mean(axis=0)

            all_crps.append(crps_ensemble_correct(gt, ens))
            all_mae.append(np.mean(np.abs(gt - ens_mean)))
            all_rmse.append(np.mean((gt - ens_mean) ** 2))

            # Spread: mean pairwise absolute difference
            M = ens.shape[0]
            sp = 0.0
            for m1 in range(M):
                for m2 in range(m1 + 1, M):
                    sp += np.mean(np.abs(ens[m1] - ens[m2]))
            sp = sp * 2.0 / (M * (M - 1)) if M > 1 else 0.0
            all_spread.append(sp)

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
    spread = np.mean(all_spread)

    print(f"\nResults ({args.split}, {n_ensemble} ens, {ode_steps} steps, "
          f"constraint={use_constraint}):")
    print(f"  CRPS (corrected): {crps:.6f}")
    print(f"  MAE:              {mae:.6f}")
    print(f"  RMSE:             {rmse:.6f}")
    print(f"  Spread:           {spread:.6f}")
    print(f"  Mass viol:        {mass_viol:.6f}")

    return crps, mae, rmse, spread, mass_viol


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--basedir", default="external/constrained-downscaling")
    parser.add_argument("--save_dir", default=str(POOL / "research5" / "models" / "dit_flow"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wall_limit", type=float, default=120.0,
                        help="Training wall-clock limit in minutes")
    # DiT architecture
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Eval
    parser.add_argument("--n_ensemble", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--ode_steps", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--constraint", default="none", choices=["none", "addcl"])
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        evaluate(args)
