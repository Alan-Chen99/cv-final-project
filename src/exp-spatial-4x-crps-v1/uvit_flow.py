"""
U-ViT Flow Matching: Transformer + Skip Connections for 32x32 -> 128x128 downscaling.

Based on "All are Worth Words: A ViT Backbone for Diffusion Models" (U-ViT, 2022).
Extends the DiT architecture with:
  - Long skip connections between corresponding encoder/decoder blocks
  - 3x3 conv refinement after unpatchify (critical for spatial quality)

Addresses the main weakness of pure DiT: lack of multi-scale information flow.

Usage:
  python src/exp-spatial-4x-crps-v1/uvit_flow.py --mode train --epochs 200 --batch_size 64
  python src/exp-spatial-4x-crps-v1/uvit_flow.py --mode eval --n_ensemble 10
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


# ---------- U-ViT Modules ----------

class PatchEmbed(nn.Module):
    """Convert 2D image to patch tokens."""
    def __init__(self, img_size=128, patch_size=8, in_channels=2, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class TimestepEmbedder(nn.Module):
    """Embed scalar timesteps into vector representations."""
    def __init__(self, hidden_size, frequency_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_dim = frequency_dim

    def forward(self, t):
        half = self.frequency_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.mlp(emb)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x


class FinalLayer(nn.Module):
    """Final layer: AdaLN + linear projection to patches."""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class UViT(nn.Module):
    """U-ViT: Transformer with long skip connections for flow matching.

    Key difference from DiT:
    - First depth//2 blocks are "encoder", remaining are "decoder"
    - Skip connections from encoder block i to decoder block (depth//2 - 1 - i)
    - 3x3 conv refinement after unpatchify for spatial coherence
    """
    def __init__(self, img_size=128, patch_size=8, in_channels=2, out_channels=1,
                 hidden_size=256, depth=12, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.num_patches = (img_size // patch_size) ** 2
        self.depth = depth
        assert depth % 2 == 0, "depth must be even for U-ViT skip connections"

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_size)

        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Time embedding
        self.time_embed = TimestepEmbedder(hidden_size)

        # Encoder blocks (first half)
        half = depth // 2
        self.encoder_blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(half)
        ])

        # Middle block
        self.mid_block = DiTBlock(hidden_size, num_heads, mlp_ratio)

        # Decoder blocks (second half) — receive skip connections
        self.decoder_blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(half)
        ])

        # Skip connection projections: concat(hidden, hidden) -> hidden
        self.skip_projs = nn.ModuleList([
            nn.Linear(hidden_size * 2, hidden_size) for _ in range(half)
        ])

        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)

        # Conv refinement after unpatchify (from U-ViT paper)
        self.conv_refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels * 4, out_channels, 3, padding=1),
        )

    def unpatchify(self, x):
        """Convert patch tokens back to image: (B, N, P*P*C) -> (B, C, H, W)."""
        p = self.patch_size
        c = self.out_channels
        h = w = self.img_size // p
        x = x.reshape(-1, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(-1, c, h * p, w * p)
        return x

    def forward(self, x, t, condition):
        """
        x: (B, 1, H, W) - noisy state
        t: (B,) - timestep
        condition: (B, 1, H, W) - LR condition
        """
        x_in = torch.cat([x, condition], dim=1)  # (B, 2, H, W)

        # Patch embed + positional embed
        x = self.patch_embed(x_in) + self.pos_embed  # (B, N, D)

        # Time conditioning
        c = self.time_embed(t * 1000.0)  # (B, D)

        # Encoder: save skip connections
        skips = []
        for block in self.encoder_blocks:
            x = block(x, c)
            skips.append(x)

        # Middle block
        x = self.mid_block(x, c)

        # Decoder: apply skip connections (reverse order)
        for i, block in enumerate(self.decoder_blocks):
            skip = skips.pop()  # last encoder output first
            x = torch.cat([x, skip], dim=-1)  # (B, N, 2*D)
            x = self.skip_projs[i](x)  # (B, N, D)
            x = block(x, c)

        # Final layer
        x = self.final_layer(x, c)  # (B, N, P*P*C)

        # Unpatchify + conv refinement
        x = self.unpatchify(x)  # (B, C, H, W)
        x = self.conv_refine(x)
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
    """CRPS (paper-compatible, BUGGY — uses fc.shape[-1]**2)."""
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
    model = UViT(
        img_size=128,
        patch_size=args.patch_size,
        in_channels=2,
        out_channels=1,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-ViT: patch_size={args.patch_size}, hidden={args.hidden_size}, "
          f"depth={args.depth}, heads={args.num_heads}")
    print(f"#params: {n_params:,}")
    print(f"#patches: {model.num_patches}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler('cuda')
    use_amp = device.type == 'cuda'

    best_val_loss = float('inf')
    start_epoch = 0

    # Resume from checkpoint
    ckpt_path = os.path.join(args.save_dir, 'best_flow.pt')
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['val_loss']
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
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

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast('cuda'):
                    pred_v = model(x_t, t, lr_batch)
                    loss = F.mse_loss(pred_v, target_v)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_v = model(x_t, t, lr_batch)
                loss = F.mse_loss(pred_v, target_v)
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
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        pred_v = model(x_t, t, lr_batch)
                        val_loss += F.mse_loss(pred_v, target_v).item()
                else:
                    pred_v = model(x_t, t, lr_batch)
                    val_loss += F.mse_loss(pred_v, target_v).item()
        val_loss /= len(val_loader)

        elapsed = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{args.epochs}, Train: {train_loss:.6f}, Val: {val_loss:.6f}, "
              f"LR: {current_lr:.6f}, Time: {elapsed/60:.1f}min")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'args': vars(args),
                'architecture': 'UViT',
            }, os.path.join(args.save_dir, 'best_flow.pt'))
            print(f"  -> New best! Saved checkpoint.")

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
    model = UViT(
        img_size=128,
        patch_size=saved_args.get('patch_size', 8),
        in_channels=2,
        out_channels=1,
        hidden_size=saved_args.get('hidden_size', 256),
        depth=saved_args.get('depth', 12),
        num_heads=saved_args.get('num_heads', 4),
        mlp_ratio=saved_args.get('mlp_ratio', 4.0),
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
    print(f"Model: U-ViT, patch_size={saved_args.get('patch_size', 8)}, "
          f"hidden={saved_args.get('hidden_size', 256)}, depth={saved_args.get('depth', 12)}")
    print(f"Model epoch: {ckpt['epoch']+1}, val_loss: {ckpt['val_loss']:.6f}")

    all_crps_paper = []
    all_crps_correct = []
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
                if device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        sampled_res_norm = euler_sample(
                            model, batch_lr,
                            shape=(bs, 1, 128, 128),
                            steps=ode_steps,
                        )
                else:
                    sampled_res_norm = euler_sample(
                        model, batch_lr,
                        shape=(bs, 1, 128, 128),
                        steps=ode_steps,
                    )
                sampled_res = sampled_res_norm.float().cpu() * stats['res_std'] + stats['res_mean']
                pred_hr = batch_lr_up + sampled_res

                if use_constraint == 'addcl':
                    pred_hr = apply_addcl(pred_hr, batch_lr_orig)

                ensemble_preds.append(pred_hr.numpy())

        ensemble_preds = np.stack(ensemble_preds, axis=1)

        for i in range(bs):
            gt = batch_hr[i, 0, ...].numpy()
            ens = ensemble_preds[i, :, 0, ...]
            ens_mean = ens.mean(axis=0)

            all_crps_paper.append(crps_ensemble(gt, ens))
            all_crps_correct.append(crps_ensemble_correct(gt, ens))
            all_mae.append(np.mean(np.abs(gt - ens_mean)))
            all_rmse.append(np.mean((gt - ens_mean) ** 2))

            pred_mean_t = torch.from_numpy(ens_mean).unsqueeze(0).unsqueeze(0)
            pooled = pool(pred_mean_t).squeeze()
            lr_i = batch_lr_orig[i, 0, ...]
            all_mass_viol.append(torch.mean(torch.abs(pooled - lr_i)).item())

        if (start_idx // batch_size) % 10 == 0:
            print(f"  Processed {end_idx}/{n_samples}...")

    crps_paper = np.mean(all_crps_paper)
    crps_correct = np.mean(all_crps_correct)
    mae = np.mean(all_mae)
    rmse = np.sqrt(np.mean(all_rmse))
    mass_viol = np.mean(all_mass_viol)

    print(f"\nResults ({args.split}, {n_ensemble} ens, {ode_steps} Euler steps, "
          f"constraint={use_constraint}):")
    print(f"  CRPS (paper/buggy): {crps_paper:.6f}")
    print(f"  CRPS (correct):     {crps_correct:.6f}")
    print(f"  MAE:                {mae:.6f}")
    print(f"  RMSE:               {rmse:.6f}")
    print(f"  Mass violation:     {mass_viol:.6f}")

    return crps_correct, mae, rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--basedir", default="external/constrained-downscaling")
    parser.add_argument("--save_dir", default="models/uvit_flow")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    # U-ViT architecture args
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    # Eval args
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
