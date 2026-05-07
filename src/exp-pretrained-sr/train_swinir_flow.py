"""SwinIR-Conditioned OT-CFM Flow Matching for 32x32 -> 128x128 downscaling.

Uses precomputed SwinIR predictions as additional conditioning channel
alongside LR (bilinear upsampled). Target is bilinear residual (HR - bilinear(LR)),
same as standard OT-CFM — avoids the source-target mismatch that failed in
residual flow (iteration 5).

Architecture: AttentionUNet with in_channels=3 [x_t, lr_up, swinir_pred].

Usage:
  python src/exp-pretrained-sr/train_swinir_flow.py --mode train --epochs 150
  python src/exp-pretrained-sr/train_swinir_flow.py --mode eval --n_ensemble 10
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
DATA_DIR = POOL / "era5_sr_data"
SWINIR_DIR = POOL / "research5" / "models" / "residual_flow"  # has swinir_preds_{split}.pt
SAVE_DIR = POOL / "research5" / "models" / "swinir_flow"


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
    """UNet with self-attention at bottleneck for velocity prediction.

    Input: interpolated state (1 ch) + conditioning channels
    Output: predicted velocity (1 ch)
    """

    def __init__(self, in_channels=3, out_channels=1, base_channels=64,
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
        """x: (B,1,H,W) noisy state, condition: (B,C,H,W) conditioning channels."""
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

def load_data(split='train'):
    """Load LR, HR, and precomputed SwinIR predictions."""
    inp = torch.load(DATA_DIR / split / f'input_{split}.pt', weights_only=False)
    tgt = torch.load(DATA_DIR / split / f'target_{split}.pt', weights_only=False)

    lr = inp[:, 0, :, :, :]  # (N, 1, 32, 32)
    hr = tgt[:, 0, :, :, :]  # (N, 1, 128, 128)
    lr_up = F.interpolate(lr, size=(128, 128), mode='bilinear', align_corners=False)
    residual = hr - lr_up

    # SwinIR predictions (precomputed in iteration 5)
    swinir_pred = torch.load(SWINIR_DIR / f'swinir_preds_{split}.pt', weights_only=False)

    return lr_up, residual, hr, lr, swinir_pred


# ---------- CRPS ----------

def crps_ensemble_correct(observation, forecasts):
    """Correct CRPS: E|X-y| - 0.5*E|X-X'|."""
    M = forecasts.shape[0]
    abs_diff = np.mean(np.abs(forecasts - observation[None, ...]), axis=0)
    spread = 0.0
    if M > 1:
        for i in range(M):
            for j in range(i + 1, M):
                spread += np.mean(np.abs(forecasts[j] - forecasts[i]))
        spread = spread * 2.0 / (M * (M - 1))
    crps = np.mean(abs_diff) - 0.5 * spread
    return crps, np.mean(abs_diff), spread


# ---------- Constraint layers ----------

def apply_addcl(pred_hr, lr_orig, upsampling_factor=4):
    pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
    pooled = pool(pred_hr)
    correction = lr_orig - pooled
    correction_hr = correction.repeat_interleave(upsampling_factor, dim=-2).repeat_interleave(upsampling_factor, dim=-1)
    return pred_hr + correction_hr


# ---------- ODE Sampling ----------

@torch.no_grad()
def euler_sample(model, condition, shape, steps=20):
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

    print("Loading data...")
    lr_up_train, res_train, _, _, swinir_train = load_data('train')
    lr_up_val, res_val, _, _, swinir_val = load_data('val')

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

    # Normalize SwinIR predictions
    sw_mean = swinir_train.mean()
    sw_std = swinir_train.std()
    swinir_train_norm = (swinir_train - sw_mean) / sw_std
    swinir_val_norm = (swinir_val - sw_mean) / sw_std

    # Save normalization stats
    stats = {
        'res_mean': res_mean.item(), 'res_std': res_std.item(),
        'lr_mean': lr_mean.item(), 'lr_std': lr_std.item(),
        'sw_mean': sw_mean.item(), 'sw_std': sw_std.item(),
    }
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(stats, os.path.join(args.save_dir, 'norm_stats.pt'))

    # Concatenate conditioning: [lr_up, swinir_pred] → 2 channels
    cond_train = torch.cat([lr_up_train_norm, swinir_train_norm], dim=1)  # (N, 2, 128, 128)
    cond_val = torch.cat([lr_up_val_norm, swinir_val_norm], dim=1)

    train_ds = TensorDataset(cond_train, res_train_norm)
    val_ds = TensorDataset(cond_val, res_val_norm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Model: in_channels=3 (x_t + 2 conditioning channels)
    model = AttentionUNet(
        in_channels=3, out_channels=1,
        base_channels=args.base_channels,
        channel_mults=args.channel_mults_tuple,
        time_emb_dim=256,
        dropout=0.1,
        attn_heads=args.attn_heads,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")
    print(f"Stats: res_mean={res_mean:.6f}, res_std={res_std:.6f}")
    print(f"       lr_mean={lr_mean:.6f}, lr_std={lr_std:.6f}")
    print(f"       sw_mean={sw_mean:.6f}, sw_std={sw_std:.6f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # EMA
    ema_model = None
    if args.ema_decay > 0:
        import copy
        ema_model = copy.deepcopy(model)
        ema_model.eval()

    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for cond_batch, res_batch in train_loader:
            cond_batch = cond_batch.to(device)
            res_batch = res_batch.to(device)

            bs = cond_batch.shape[0]
            t = torch.rand(bs, device=device)
            x_0 = torch.randn_like(res_batch)
            t_expand = t[:, None, None, None]
            x_t = (1 - t_expand) * x_0 + t_expand * res_batch
            target_v = res_batch - x_0

            pred_v = model(x_t, t, cond_batch)
            loss = F.mse_loss(pred_v, target_v)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # EMA update
            if ema_model is not None:
                with torch.no_grad():
                    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                        p_ema.data.mul_(args.ema_decay).add_(p.data, alpha=1 - args.ema_decay)

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validation
        eval_model = ema_model if ema_model is not None else model
        eval_model.eval()
        val_loss = 0
        with torch.no_grad():
            for cond_batch, res_batch in val_loader:
                cond_batch = cond_batch.to(device)
                res_batch = res_batch.to(device)
                bs = cond_batch.shape[0]
                t = torch.rand(bs, device=device)
                x_0 = torch.randn_like(res_batch)
                t_expand = t[:, None, None, None]
                x_t = (1 - t_expand) * x_0 + t_expand * res_batch
                target_v = res_batch - x_0
                pred_v = eval_model(x_t, t, cond_batch)
                val_loss += F.mse_loss(pred_v, target_v).item()
        val_loss /= len(val_loader)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs}, Train: {train_loss:.6f}, Val: {val_loss:.6f}, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}, Time: {elapsed/60:.1f}min")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dict = {
                'model': (ema_model if ema_model is not None else model).state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'args': vars(args),
            }
            torch.save(save_dict, os.path.join(args.save_dir, 'best_flow.pt'))

        # Time limit check
        if elapsed > args.time_limit * 60:
            print(f"Time limit ({args.time_limit}min) reached at epoch {epoch+1}")
            break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Total time: {(time.time() - start_time)/60:.1f} min")


# ---------- Evaluation ----------

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stats = torch.load(os.path.join(args.save_dir, 'norm_stats.pt'), weights_only=False,
                        map_location=device)

    lr_up, residual, hr, lr_orig, swinir_pred = load_data(args.split)

    # Normalize conditioning
    lr_up_norm = (lr_up - stats['lr_mean']) / stats['lr_std']
    swinir_norm = (swinir_pred - stats['sw_mean']) / stats['sw_std']

    ckpt = torch.load(os.path.join(args.save_dir, 'best_flow.pt'), weights_only=False,
                       map_location=device)
    saved_args = ckpt['args']
    saved_mults = saved_args.get('channel_mults_tuple', (1, 2, 4))
    if isinstance(saved_mults, str):
        saved_mults = tuple(int(x) for x in saved_mults.split(','))

    model = AttentionUNet(
        in_channels=3, out_channels=1,
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

    print(f"Evaluating {n_samples} samples, {n_ensemble} ensemble, "
          f"{ode_steps} euler steps, constraint={use_constraint}...")
    print(f"Model epoch: {ckpt['epoch']+1}, val_loss: {ckpt['val_loss']:.6f}")

    all_crps = []
    all_mae_list = []
    all_rmse = []
    all_mass_viol = []

    pool = torch.nn.AvgPool2d(kernel_size=4)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_lr_up_norm = lr_up_norm[start_idx:end_idx].to(device)
        batch_swinir_norm = swinir_norm[start_idx:end_idx].to(device)
        batch_hr = hr[start_idx:end_idx]
        batch_lr_up = lr_up[start_idx:end_idx]
        batch_lr_orig = lr_orig[start_idx:end_idx]
        bs = batch_lr_up_norm.shape[0]

        # Concatenate conditioning
        batch_cond = torch.cat([batch_lr_up_norm, batch_swinir_norm], dim=1)

        ensemble_preds = []
        for e in range(n_ensemble):
            with torch.no_grad():
                sampled_res_norm = euler_sample(
                    model, batch_cond,
                    shape=(bs, 1, 128, 128),
                    steps=ode_steps,
                )
                sampled_res = sampled_res_norm.cpu() * stats['res_std'] + stats['res_mean']
                pred_hr = batch_lr_up + sampled_res

                if use_constraint == 'addcl':
                    pred_hr = apply_addcl(pred_hr, batch_lr_orig)

                ensemble_preds.append(pred_hr.numpy())

        ensemble_preds = np.stack(ensemble_preds, axis=0)  # (K, B, 1, 128, 128)

        for i in range(bs):
            gt = batch_hr[i, 0, ...].numpy()
            ens = ensemble_preds[:, i, 0, ...]  # (K, 128, 128)
            ens_mean = ens.mean(axis=0)

            crps, mae_val, spread = crps_ensemble_correct(gt, ens)
            all_crps.append(crps)
            all_mae_list.append(mae_val)
            all_rmse.append(np.mean((gt - ens_mean) ** 2))

            pred_mean_t = torch.from_numpy(ens_mean).unsqueeze(0).unsqueeze(0)
            pooled = pool(pred_mean_t).squeeze()
            lr_i = batch_lr_orig[i, 0, ...]
            all_mass_viol.append(torch.mean(torch.abs(pooled - lr_i)).item())

        if (start_idx // batch_size) % 20 == 0:
            print(f"  Processed {end_idx}/{n_samples}...")

    crps = np.mean(all_crps)
    mae = np.mean(all_mae_list)
    rmse = np.sqrt(np.mean(all_rmse))
    mass_viol = np.mean(all_mass_viol)

    print(f"\nResults ({args.split}, {n_ensemble} ens, {ode_steps} steps, constraint={use_constraint}):")
    print(f"  CRPS:      {crps:.6f}")
    print(f"  MAE:       {mae:.6f}")
    print(f"  RMSE:      {rmse:.6f}")
    print(f"  Mass viol: {mass_viol:.6f}")

    return crps, mae, rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--save_dir", default=str(SAVE_DIR))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--channel_mults", type=str, default="1,2,4")
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--time_limit", type=float, default=120,
                        help="Wall-clock time limit in minutes")
    # Eval args
    parser.add_argument("--n_ensemble", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--ode_steps", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--constraint", default="none", choices=["none", "addcl"])
    args = parser.parse_args()

    args.channel_mults_tuple = tuple(int(x) for x in args.channel_mults.split(','))

    if args.mode == "train":
        train(args)
    else:
        evaluate(args)
