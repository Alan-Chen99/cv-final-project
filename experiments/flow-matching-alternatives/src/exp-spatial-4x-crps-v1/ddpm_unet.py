"""
DDPM (VP-SDE) score-based diffusion for 32x32 -> 128x128 downscaling.

Different generative framework from OT-CFM flow matching:
  - Forward: x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*eps
  - Loss: MSE(eps_pred, eps) — predict noise
  - Sampling: DDIM (deterministic) with configurable steps
  - Noise schedule: linear beta from beta_start to beta_end, T=1000

Same AttentionUNet architecture as unet_cfg_flow.py for fair comparison.

Usage:
  python ddpm_unet.py --mode train --epochs 60 --batch_size 64
  python ddpm_unet.py --mode eval --ddim_steps 20 --n_ensemble 10
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


# ---------- Architecture (same AttentionUNet as unet_cfg_flow.py) ----------

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
        t_emb = self.time_mlp(t)
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


# ---------- Noise schedule ----------

class DDPMSchedule:
    """Linear VP-SDE noise schedule."""

    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alpha_bar = alpha_bar.to(device)
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar).to(device)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar).to(device)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        return self

    def q_sample(self, x_0, t, noise=None):
        """Forward process: add noise at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self.sqrt_alpha_bar[t][:, None, None, None]
        sqrt_1mab = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return sqrt_ab * x_0 + sqrt_1mab * noise, noise


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
    """Standard energy CRPS: E|X-y| - 0.5*E|X-X'| using Gneiting sorted formula."""
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
    """Softmax constraint layer (SmCL) from Harder et al. 2208.05424.
    Ensures non-negativity and exact mass conservation."""
    pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
    y = torch.exp(pred_hr)
    sum_y = pool(y)
    ratio = lr_orig / sum_y
    ratio_hr = ratio.repeat_interleave(upsampling_factor, dim=-2).repeat_interleave(upsampling_factor, dim=-1)
    return y * ratio_hr


# ---------- DDIM Sampling ----------

@torch.no_grad()
def ddim_sample(model, condition, shape, schedule, steps=20, eta=0.0):
    """DDIM sampling. eta=0 is deterministic, eta=1 is DDPM."""
    device = condition.device
    T = schedule.T

    # Create DDIM timestep subsequence (evenly spaced)
    step_indices = torch.linspace(T - 1, 0, steps + 1, dtype=torch.long)
    timesteps = step_indices[:-1]  # [T-1, ..., 0] with `steps` entries
    timesteps_prev = step_indices[1:]  # shifted by 1

    x = torch.randn(shape, device=device)

    for i in range(steps):
        t_idx = timesteps[i].item()
        t_prev_idx = timesteps_prev[i].item()

        t_batch = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)

        # Predict noise
        t_continuous = t_batch.float() / T  # normalize to [0, 1] for the model
        eps_pred = model(x, t_continuous, condition)

        # DDIM update
        alpha_bar_t = schedule.alpha_bar[t_idx]
        alpha_bar_prev = schedule.alpha_bar[max(t_prev_idx, 0)] if t_prev_idx >= 0 else torch.tensor(1.0, device=device)

        # Predict x_0
        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

        # Direction pointing to x_t
        sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
        dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t ** 2) * eps_pred

        # DDIM step
        noise = torch.randn_like(x) if (eta > 0 and i < steps - 1) else torch.zeros_like(x)
        x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma_t * noise

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

    schedule = DDPMSchedule(T=args.T, beta_start=args.beta_start,
                            beta_end=args.beta_end, device=device)

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
    print(f"DDPM schedule: T={args.T}, beta=[{args.beta_start}, {args.beta_end}]")

    # EMA
    ema_model = None
    if args.ema_decay > 0:
        import copy
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
        print(f"EMA enabled with decay={args.ema_decay}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Warmup + cosine schedule
    warmup_steps = args.warmup_epochs * len(train_loader)
    total_steps = args.epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    start_epoch = 0

    ckpt_path = os.path.join(args.save_dir, 'best_ddpm.pt')
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['val_loss']
        if ema_model and 'ema_model' in ckpt:
            ema_model.load_state_dict(ckpt['ema_model'])
        print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.6f}")

    start_time = time.time()
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        for lr_batch, res_batch in train_loader:
            lr_batch = lr_batch.to(device)
            res_batch = res_batch.to(device)
            bs = lr_batch.shape[0]

            # Sample random timesteps
            t = torch.randint(0, args.T, (bs,), device=device)

            # Forward process: add noise
            noise = torch.randn_like(res_batch)
            x_t, _ = schedule.q_sample(res_batch, t, noise)

            # Model predicts noise, conditioned on t/T (normalized to [0,1])
            t_normalized = t.float() / args.T
            eps_pred = model(x_t, t_normalized, lr_batch)
            loss = F.mse_loss(eps_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # EMA update
            if ema_model is not None:
                with torch.no_grad():
                    for p_ema, p_model in zip(ema_model.parameters(), model.parameters()):
                        p_ema.data.mul_(args.ema_decay).add_(p_model.data, alpha=1 - args.ema_decay)

            train_loss += loss.item()
            global_step += 1

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for lr_batch, res_batch in val_loader:
                lr_batch = lr_batch.to(device)
                res_batch = res_batch.to(device)
                bs = lr_batch.shape[0]
                t = torch.randint(0, args.T, (bs,), device=device)
                noise = torch.randn_like(res_batch)
                x_t, _ = schedule.q_sample(res_batch, t, noise)
                t_normalized = t.float() / args.T
                eps_pred = model(x_t, t_normalized, lr_batch)
                val_loss += F.mse_loss(eps_pred, noise).item()
        val_loss /= len(val_loader)

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs}, Train: {train_loss:.6f}, Val: {val_loss:.6f}, "
              f"LR: {current_lr:.6f}, Time: {elapsed/60:.1f}min")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dict = {
                'model': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'args': vars(args),
            }
            if ema_model is not None:
                save_dict['ema_model'] = ema_model.state_dict()
            torch.save(save_dict, ckpt_path)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Total time: {(time.time() - start_time)/60:.1f} min")


# ---------- Evaluation ----------

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    basedir = args.basedir
    print(f"[eval] Device: {device}, basedir: {basedir}", flush=True)

    print("[eval] Loading norm stats...", flush=True)
    stats = torch.load(os.path.join(args.save_dir, 'norm_stats.pt'), weights_only=False,
                        map_location=device)

    print("[eval] Loading test data...", flush=True)
    lr_up, residual, hr, lr_orig = load_tcw4_data(basedir, args.split)
    print(f"[eval] Data loaded: {lr_up.shape[0]} samples", flush=True)
    lr_up_norm = (lr_up - stats['lr_mean']) / stats['lr_std']

    ckpt = torch.load(os.path.join(args.save_dir, 'best_ddpm.pt'), weights_only=False,
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

    # Use EMA weights if available
    if args.use_ema and 'ema_model' in ckpt:
        model.load_state_dict(ckpt['ema_model'])
        print("Using EMA model weights")
    else:
        model.load_state_dict(ckpt['model'])
        print("Using standard model weights")
    model.eval()

    T = saved_args.get('T', 1000)
    schedule = DDPMSchedule(T=T,
                            beta_start=saved_args.get('beta_start', 1e-4),
                            beta_end=saved_args.get('beta_end', 0.02),
                            device=device)

    n_samples = min(lr_up.shape[0], args.max_samples) if args.max_samples else lr_up.shape[0]
    n_ensemble = args.n_ensemble
    batch_size = args.eval_batch_size
    ddim_steps = args.ddim_steps

    print(f"Evaluating {n_samples} samples, {n_ensemble} ensemble, "
          f"{ddim_steps} DDIM steps, constraint={args.constraint}")
    print(f"Model epoch: {ckpt['epoch']+1}, val_loss: {ckpt['val_loss']:.6f}")
    print(f"DDPM T={T}")

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
                sampled_res_norm = ddim_sample(
                    model, batch_lr,
                    shape=(bs, 1, 128, 128),
                    schedule=schedule,
                    steps=ddim_steps,
                    eta=args.ddim_eta,
                )
                sampled_res = sampled_res_norm.cpu() * stats['res_std'] + stats['res_mean']
                pred_hr = batch_lr_up + sampled_res

                if args.constraint == 'addcl':
                    pred_hr = apply_addcl(pred_hr, batch_lr_orig)
                elif args.constraint == 'smcl':
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
            print(f"  Processed {end_idx}/{n_samples}...", flush=True)

    crps = np.mean(all_crps)
    mae = np.mean(all_mae)
    rmse = np.sqrt(np.mean(all_rmse))
    mass_viol = np.mean(all_mass_viol)

    ema_str = " (EMA)" if (args.use_ema and 'ema_model' in ckpt) else ""
    print(f"\nResults ({args.split}, {n_ensemble} ens, {ddim_steps} DDIM steps, "
          f"eta={args.ddim_eta}, constraint={args.constraint}){ema_str}:")
    print(f"  CRPS (Gneiting M^2): {crps:.6f}")
    print(f"  MAE:                 {mae:.6f}")
    print(f"  RMSE:                {rmse:.6f}")
    print(f"  Mass violation:      {mass_viol:.6f}")

    return crps, mae, rmse, mass_viol


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--basedir", default="external/constrained-downscaling")
    parser.add_argument("--save_dir", default="models/ddpm")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--channel_mults", type=str, default="1,2,4")
    parser.add_argument("--attn_heads", type=int, default=4)
    # DDPM schedule
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    # EMA
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    # Eval args
    parser.add_argument("--n_ensemble", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--ddim_eta", type=float, default=0.0,
                        help="DDIM eta: 0=deterministic, 1=DDPM stochastic")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--constraint", default="addcl", choices=["none", "addcl", "smcl"])
    parser.add_argument("--use_ema", action="store_true", help="Use EMA weights for eval")
    args = parser.parse_args()

    args.channel_mults_tuple = tuple(int(x) for x in args.channel_mults.split(','))

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
