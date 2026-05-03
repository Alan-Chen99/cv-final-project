"""
Simple conditional DDPM for 32×32 → 128×128 climate downscaling.
Predicts residual (HR - bilinear_upsample(LR)) conditioned on LR input.

Usage:
  python scripts/simple_diffusion.py --mode train --epochs 100 --batch_size 64
  python scripts/simple_diffusion.py --mode eval --n_ensemble 20 --split test
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


# ---------- Diffusion schedule ----------

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


class GaussianDiffusion:
    def __init__(self, timesteps=1000, schedule='cosine'):
        if schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        else:
            betas = cosine_beta_schedule(timesteps)

        self.timesteps = timesteps
        self.betas = betas
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)

    def _extract(self, a, t, shape):
        return a.to(t.device).gather(-1, t).reshape(-1, *([1] * (len(shape) - 1)))

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: add noise to x0."""
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    def predict_x0_from_noise(self, xt, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, xt.shape) * xt
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, xt.shape) * noise
        )

    def p_mean_variance(self, model, xt, t, condition):
        """Predict mean and variance for reverse step."""
        pred_noise = model(xt, t, condition)
        x0_pred = self.predict_x0_from_noise(xt, t, pred_noise)
        x0_pred = torch.clamp(x0_pred, -5, 5)  # clamp for stability

        mean = (
            self._extract(self.posterior_mean_coef1, t, xt.shape) * x0_pred
            + self._extract(self.posterior_mean_coef2, t, xt.shape) * xt
        )
        var = self._extract(self.posterior_variance, t, xt.shape)
        log_var = self._extract(self.posterior_log_variance, t, xt.shape)
        return mean, var, log_var

    @torch.no_grad()
    def p_sample(self, model, xt, t, condition):
        mean, _, log_var = self.p_mean_variance(model, xt, t, condition)
        noise = torch.randn_like(xt)
        nonzero_mask = (t != 0).float().reshape(-1, *([1] * (len(xt.shape) - 1)))
        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def sample(self, model, condition, shape, ddim_steps=None, eta=1.0):
        """Reverse sampling with optional DDIM acceleration."""
        device = condition.device
        if ddim_steps is not None and ddim_steps < self.timesteps:
            return self.ddim_sample(model, condition, shape, ddim_steps, eta)
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, condition)
        return x

    @torch.no_grad()
    def ddim_sample(self, model, condition, shape, steps=50, eta=1.0):
        """DDIM sampling with fewer steps. eta=0 deterministic, eta=1 stochastic."""
        device = condition.device
        # Subsequence of timesteps
        step_size = self.timesteps // steps
        timesteps = list(range(0, self.timesteps, step_size))[:steps]
        timesteps = list(reversed(timesteps))

        x = torch.randn(shape, device=device)
        for i, t_cur in enumerate(timesteps):
            t = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)
            pred_noise = model(x, t, condition)
            x0_pred = self.predict_x0_from_noise(x, t, pred_noise)
            x0_pred = torch.clamp(x0_pred, -5, 5)

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_bar_t = self.alphas_cumprod[t_cur]
                alpha_bar_next = self.alphas_cumprod[t_next]
                sigma = eta * torch.sqrt(
                    (1 - alpha_bar_next) / (1 - alpha_bar_t) *
                    (1 - alpha_bar_t / alpha_bar_next)
                )
                dir_xt = torch.sqrt(1 - alpha_bar_next - sigma ** 2) * pred_noise
                noise = torch.randn_like(x) if eta > 0 else 0
                x = torch.sqrt(alpha_bar_next) * x0_pred + dir_xt + sigma * noise
            else:
                x = x0_pred
        return x


# ---------- UNet architecture ----------

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
        # Time conditioning: scale + shift
        scale, shift = self.time_mlp(F.silu(t_emb)).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


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


class SimpleUNet(nn.Module):
    """Conditional UNet for noise prediction.

    Input: noisy residual (1 ch) concatenated with LR condition (1 ch) = 2 ch
    Output: predicted noise (1 ch)
    """

    def __init__(self, in_channels=2, out_channels=1, base_channels=64,
                 channel_mults=(1, 2, 4, 8), time_emb_dim=256, dropout=0.1):
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

        # Bottleneck
        self.mid_block1 = ResBlock(ch, ch, time_emb_dim, dropout)
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim, dropout)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.up_blocks.append(nn.ModuleList([
                ResBlock(ch + out_ch, out_ch, time_emb_dim, dropout),  # skip connection
                ResBlock(out_ch, out_ch, time_emb_dim, dropout),
            ]))
            self.up_samples.append(Upsample(ch))  # upsample input channels, not output
            ch = out_ch

        self.final_norm = nn.GroupNorm(min(32, ch), ch)
        self.final_conv = nn.Conv2d(ch, out_channels, 1)

    def forward(self, x, t, condition):
        """
        x: noisy residual [B, 1, 128, 128]
        t: timestep [B]
        condition: upsampled LR [B, 1, 128, 128]
        """
        # Concatenate along channel dim
        x = torch.cat([x, condition], dim=1)

        t_emb = self.time_mlp(t)
        h = self.init_conv(x)

        # Encoder with skip connections
        skips = []
        for blocks, downsample in zip(self.down_blocks, self.down_samples):
            for block in blocks:
                h = block(h, t_emb)
            skips.append(h)
            h = downsample(h)

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        # Decoder
        for blocks, upsample in zip(self.up_blocks, self.up_samples):
            h = upsample(h)
            h = torch.cat([h, skips.pop()], dim=1)
            for block in blocks:
                h = block(h, t_emb)

        return self.final_conv(F.silu(self.final_norm(h)))


# ---------- Data loading ----------

def load_tcw4_data(basedir, split='train', normalize=True):
    """Load TCW4 data and compute residuals."""
    inp = torch.load(f'{basedir}/data/era5_sr_data/{split}/input_{split}.pt', weights_only=False)
    tgt = torch.load(f'{basedir}/data/era5_sr_data/{split}/target_{split}.pt', weights_only=False)

    # Shapes: inp (N,1,1,32,32), tgt (N,1,1,128,128)
    lr = inp[:, 0, :, :, :]  # (N, 1, 32, 32)
    hr = tgt[:, 0, :, :, :]  # (N, 1, 128, 128)

    # Bilinear upsample LR to HR resolution
    lr_up = F.interpolate(lr, size=(128, 128), mode='bilinear', align_corners=False)

    # Residual = HR - LR_upsampled
    residual = hr - lr_up

    if normalize:
        # Compute normalization stats from training data
        return lr_up, residual, hr

    return lr_up, residual, hr


def crps_ensemble(observation, forecasts):
    """CRPS for ensemble forecasts."""
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


# ---------- Training ----------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    basedir = args.basedir

    print("Loading data...")
    lr_up_train, res_train, hr_train = load_tcw4_data(basedir, 'train')
    lr_up_val, res_val, hr_val = load_tcw4_data(basedir, 'val')

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
    model = SimpleUNet(
        in_channels=2, out_channels=1,
        base_channels=args.base_channels,
        channel_mults=args.channel_mults_tuple,
        time_emb_dim=256,
        dropout=0.1,
    ).to(device)
    print(f"#params: {sum(p.numel() for p in model.parameters()):,}")

    diffusion = GaussianDiffusion(timesteps=args.timesteps, schedule=args.schedule)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    start_epoch = 0

    # Resume from checkpoint
    ckpt_path = os.path.join(args.save_dir, 'best_diffusion.pt')
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['val_loss']
        # Advance scheduler to correct position
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

            # Random timestep
            t = torch.randint(0, args.timesteps, (lr_batch.shape[0],), device=device)
            noise = torch.randn_like(res_batch)
            noisy_res = diffusion.q_sample(res_batch, t, noise)

            # Predict noise
            pred_noise = model(noisy_res, t, lr_batch)
            loss = F.mse_loss(pred_noise, noise)

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
                t = torch.randint(0, args.timesteps, (lr_batch.shape[0],), device=device)
                noise = torch.randn_like(res_batch)
                noisy_res = diffusion.q_sample(res_batch, t, noise)
                pred_noise = model(noisy_res, t, lr_batch)
                val_loss += F.mse_loss(pred_noise, noise).item()
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
            }, os.path.join(args.save_dir, 'best_diffusion.pt'))

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Total time: {(time.time() - start_time)/60:.1f} min")


# ---------- Evaluation ----------

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    basedir = args.basedir

    # Load normalization stats
    stats = torch.load(os.path.join(args.save_dir, 'norm_stats.pt'), weights_only=False)

    # Load test data
    lr_up, residual, hr = load_tcw4_data(basedir, args.split)
    lr_up_norm = (lr_up - stats['lr_mean']) / stats['lr_std']

    # Load model
    ckpt = torch.load(os.path.join(args.save_dir, 'best_diffusion.pt'), weights_only=False)
    saved_mults = ckpt['args'].get('channel_mults_tuple', (1, 2, 4))
    if isinstance(saved_mults, str):
        saved_mults = tuple(int(x) for x in saved_mults.split(','))
    model = SimpleUNet(
        in_channels=2, out_channels=1,
        base_channels=ckpt['args'].get('base_channels', 64),
        channel_mults=saved_mults,
        time_emb_dim=256,
        dropout=0.0,  # no dropout at inference
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    diffusion = GaussianDiffusion(
        timesteps=ckpt['args'].get('timesteps', 1000),
        schedule=ckpt['args'].get('schedule', 'cosine'),
    )

    n_samples = min(lr_up.shape[0], args.max_samples) if args.max_samples else lr_up.shape[0]
    n_ensemble = args.n_ensemble
    batch_size = args.eval_batch_size

    # Move diffusion schedule to GPU for faster sampling
    diffusion.alphas_cumprod = diffusion.alphas_cumprod.to(device)
    diffusion.sqrt_alphas_cumprod = diffusion.sqrt_alphas_cumprod.to(device)
    diffusion.sqrt_one_minus_alphas_cumprod = diffusion.sqrt_one_minus_alphas_cumprod.to(device)
    diffusion.sqrt_recip_alphas_cumprod = diffusion.sqrt_recip_alphas_cumprod.to(device)
    diffusion.sqrt_recipm1_alphas_cumprod = diffusion.sqrt_recipm1_alphas_cumprod.to(device)
    diffusion.posterior_variance = diffusion.posterior_variance.to(device)
    diffusion.posterior_log_variance = diffusion.posterior_log_variance.to(device)
    diffusion.posterior_mean_coef1 = diffusion.posterior_mean_coef1.to(device)
    diffusion.posterior_mean_coef2 = diffusion.posterior_mean_coef2.to(device)

    print(f"Evaluating {n_samples} samples with {n_ensemble} ensemble members...")

    # Generate ensemble predictions
    all_crps = []
    all_mae = []
    all_rmse = []

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_lr = lr_up_norm[start_idx:end_idx].to(device)
        batch_hr = hr[start_idx:end_idx].numpy()
        batch_lr_up = lr_up[start_idx:end_idx].numpy()  # un-normalized for reconstruction
        bs = batch_lr.shape[0]

        ensemble_preds = []
        for e in range(n_ensemble):
            with torch.no_grad():
                # Sample from diffusion (DDIM for speed)
                sampled_res_norm = diffusion.sample(
                    model, batch_lr,
                    shape=(bs, 1, 128, 128),
                    ddim_steps=args.ddim_steps,
                    eta=args.ddim_eta,
                )
                # Un-normalize residual
                sampled_res = sampled_res_norm.cpu() * stats['res_std'] + stats['res_mean']
                # Reconstruct HR = LR_up + residual
                pred_hr = torch.from_numpy(batch_lr_up) + sampled_res
                ensemble_preds.append(pred_hr.numpy())

        ensemble_preds = np.stack(ensemble_preds, axis=1)  # (bs, n_ensemble, 1, 128, 128)

        for i in range(bs):
            gt = batch_hr[i, 0, ...]  # (128, 128)
            ens = ensemble_preds[i, :, 0, ...]  # (n_ensemble, 128, 128)
            ens_mean = ens.mean(axis=0)

            all_crps.append(crps_ensemble(gt, ens))
            all_mae.append(np.mean(np.abs(gt - ens_mean)))
            all_rmse.append(np.mean((gt - ens_mean) ** 2))

        if (start_idx // batch_size) % 10 == 0:
            print(f"  Processed {end_idx}/{n_samples}...")

    crps = np.mean(all_crps)
    mae = np.mean(all_mae)
    rmse = np.sqrt(np.mean(all_rmse))

    print(f"\nResults ({args.split}, {n_ensemble} ensemble members):")
    print(f"  CRPS: {crps:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")

    return crps, mae, rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--basedir", default="external/constrained-downscaling")
    parser.add_argument("--save_dir", default="models/diffusion_v1")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--schedule", default="cosine")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--channel_mults", type=str, default="1,2,4",
                        help="Channel multipliers (comma-separated)")
    # Eval args
    parser.add_argument("--n_ensemble", type=int, default=20)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM steps for eval (None=full DDPM)")
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="DDIM stochasticity (0=deterministic, 1=DDPM-like)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit test samples")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    args.channel_mults_tuple = tuple(int(x) for x in args.channel_mults.split(','))

    if args.mode == "train":
        train(args)
    else:
        evaluate(args)
