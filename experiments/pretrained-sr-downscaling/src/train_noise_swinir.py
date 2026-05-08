"""Noise-conditioned SwinIR for stochastic ensemble via CRPS loss.

Architecture:
  - Frozen SwinIR backbone → features (B, 180, 32, 32)
  - Noise injection: concat z~N(0,I) of shape (B, noise_dim, 32, 32)
  - Single reconstruction tail: Conv(180+noise_dim, 64) → PixelShuffle(4x) → Conv(64, 1)
  - Train with energy score (CRPS) loss: K noise draws per sample

Key difference from multi-head approach:
  - Multi-head: K fixed functions (discrete diversity, gradient conflict)
  - Noise-conditioned: continuous mapping from noise space (unbounded diversity, single head)
  - Can sample any K at test time without retraining

Usage:
    python experiments/pretrained-sr-downscaling/src/train_noise_swinir.py --noise_dim 16 --K 8 --epochs 100
    python experiments/pretrained-sr-downscaling/src/train_noise_swinir.py --mode eval --K 10
"""

import argparse
import copy
import json
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
SAVE_DIR = POOL / "research5" / "models" / "noise_swinir"


class NoiseCondSwinIR(nn.Module):
    """SwinIR with noise-conditioned reconstruction tail.

    The backbone is frozen and produces features (B, 180, 32, 32).
    A noise map z~N(0,I) is concatenated to features, and a single
    trainable tail reconstructs the HR output. Different noise → different output.
    """

    def __init__(self, backbone, noise_dim=16):
        super().__init__()
        self.noise_dim = noise_dim

        # Shared frozen backbone
        self.conv_first = backbone.conv_first
        self.patch_embed = backbone.patch_embed
        self.patch_unembed = backbone.patch_unembed
        self.pos_drop = backbone.pos_drop
        self.layers = backbone.layers
        self.norm = backbone.norm
        self.conv_after_body = backbone.conv_after_body
        self.register_buffer('mean', backbone.mean.clone())
        self.img_range = backbone.img_range

        # Forward features attributes
        self.num_layers = backbone.num_layers
        self.patches_resolution = backbone.patches_resolution
        self.num_features = backbone.num_features

        # Freeze all backbone parameters
        for param in self.conv_first.parameters():
            param.requires_grad = False
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        for param in self.pos_drop.parameters():
            param.requires_grad = False
        for param in self.layers.parameters():
            param.requires_grad = False
        for param in self.norm.parameters():
            param.requires_grad = False
        for param in self.conv_after_body.parameters():
            param.requires_grad = False

        # Noise projection: project noise channels to feature space
        # Use a small MLP-style conv to give noise channels meaningful structure
        self.noise_proj = nn.Sequential(
            nn.Conv2d(noise_dim, 64, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
        )

        # Single reconstruction tail: processes features + projected noise
        in_ch = 180 + 64  # backbone features + projected noise
        self.tail = nn.Sequential(
            # conv_before_upsample equivalent
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            # upsample: 2x PixelShuffle stages (32→64→128)
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),
            # conv_last: 64→1
            nn.Conv2d(64, 1, 3, padding=1),
        )

    def init_tail_from_finetuned(self, backbone):
        """Initialize tail conv layers from finetuned backbone weights."""
        # Map backbone tail layers to our tail sequential indices
        # tail[0] = conv_before_upsample[0] (Conv2d in_ch→64)
        # tail[2] = upsample[0] (Conv2d 64→256)
        # tail[4] = upsample[2] (Conv2d 64→256)
        # tail[6] = conv_last (Conv2d 64→1)
        src_layers = [
            ('conv_before_upsample.0', 0),
            ('upsample.0', 2),
            ('upsample.2', 4),
            ('conv_last', 6),
        ]

        for src_name, dst_idx in src_layers:
            src_mod = dict(backbone.named_modules())[src_name]
            dst_mod = self.tail[dst_idx]
            with torch.no_grad():
                if dst_idx == 0:
                    # First conv: input channels differ (180+64 vs 180)
                    # Copy backbone weights for first 180 channels, zero-init for noise channels
                    dst_mod.weight[:, :180, :, :].copy_(src_mod.weight)
                    dst_mod.weight[:, 180:, :, :].zero_()
                    if src_mod.bias is not None:
                        dst_mod.bias.copy_(src_mod.bias)
                else:
                    dst_mod.weight.copy_(src_mod.weight)
                    if src_mod.bias is not None and dst_mod.bias is not None:
                        dst_mod.bias.copy_(src_mod.bias)

    def forward_features(self, x):
        """Forward through Swin Transformer body."""
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward_backbone(self, x):
        """Get backbone features (frozen). x: (B, 1, 32, 32) normalized."""
        x = (x - self.mean) * self.img_range
        feat = self.conv_first(x)
        body_out = self.conv_after_body(self.forward_features(feat))
        shared = body_out + feat  # (B, 180, 32, 32)
        return shared

    def forward_single(self, shared_features, noise):
        """Generate one sample from features + noise.

        shared_features: (B, 180, 32, 32) — frozen backbone output
        noise: (B, noise_dim, 32, 32)

        Returns: (B, 1, 128, 128)
        """
        noise_feat = self.noise_proj(noise)  # (B, 64, 32, 32)
        combined = torch.cat([shared_features, noise_feat], dim=1)  # (B, 244, 32, 32)
        out = self.tail(combined)  # (B, 1, 128, 128)
        # Undo normalization
        return out / self.img_range + self.mean

    def forward(self, x, K=8):
        """Generate K ensemble members.

        x: (B, 1, 32, 32) normalized input
        K: number of noise samples

        Returns: (B, K, 1, 128, 128)
        """
        B = x.shape[0]
        device = x.device

        # Compute backbone features once (frozen)
        with torch.no_grad():
            shared = self.forward_backbone(x)  # (B, 180, 32, 32)

        # Generate K members with different noise
        members = []
        for k in range(K):
            z = torch.randn(B, self.noise_dim, 32, 32, device=device)
            out = self.forward_single(shared, z)
            members.append(out)

        return torch.stack(members, dim=1)  # (B, K, 1, 128, 128)


def energy_score_loss(ensemble, target):
    """Energy score loss (CRPS for multivariate).

    ensemble: (B, K, 1, H, W)
    target: (B, 1, H, W)

    Loss = (1/K) Σ_k |y_k - y| - (1/(2K²)) ΣΣ |y_k - y_k'|
    """
    B, K = ensemble.shape[:2]
    target_exp = target.unsqueeze(1)  # (B, 1, 1, H, W)
    term1 = torch.mean(torch.abs(ensemble - target_exp))

    ens_i = ensemble.unsqueeze(2)  # (B, K, 1, 1, H, W)
    ens_j = ensemble.unsqueeze(1)  # (B, 1, K, 1, H, W)
    term2 = torch.mean(torch.abs(ens_i - ens_j))

    loss = term1 - 0.5 * term2
    return loss, term1.item(), term2.item()


def load_data(split='train'):
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


def apply_addcl_batch(predictions, lr):
    """Vectorized AddCL. predictions: (N, 128, 128), lr: (N, 32, 32)."""
    pred_ds = predictions.reshape(-1, 32, 4, 32, 4).mean(axis=(2, 4))
    correction = lr - pred_ds
    correction_hr = np.repeat(np.repeat(correction, 4, axis=1), 4, axis=2)
    return predictions + correction_hr


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print("Loading data...")
    lr_train, hr_train = load_data('train')
    lr_val, hr_val = load_data('val')

    # Load normalization stats from finetuned checkpoint
    ft_ckpt = torch.load(
        POOL / "research5" / "models" / "swinir_ft" / "best_swinir.pt",
        map_location="cpu", weights_only=False
    )
    vmin = ft_ckpt['vmin']
    vmax = ft_ckpt['vmax']
    print(f"  Normalization range: [{vmin:.4f}, {vmax:.4f}]")

    # Normalize
    lr_train_n = normalize(lr_train, vmin, vmax)
    hr_train_n = normalize(hr_train, vmin, vmax)
    lr_val_n = normalize(lr_val, vmin, vmax)
    hr_val_n = normalize(hr_val, vmin, vmax)

    train_ds = TensorDataset(lr_train_n, hr_train_n)
    val_ds = TensorDataset(lr_val_n, hr_val_n)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Build model
    print("Loading SwinIR backbone...")
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from finetune_swinir import load_swinir_1ch

    weights_path = POOL / "research5" / "pretrained_weights" / "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
    backbone = load_swinir_1ch(str(weights_path))
    backbone.load_state_dict(ft_ckpt['model'])

    model = NoiseCondSwinIR(backbone, noise_dim=args.noise_dim)
    model.init_tail_from_finetuned(backbone)
    model = model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Noise dim: {args.noise_dim}")
    print(f"  Parameters: {n_total:,} total, {n_trainable:,} trainable")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    expected_epochs = min(args.epochs, int(args.wall_hours * 60 / 7) + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=expected_epochs)

    # Save dir
    save_dir = SAVE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config['vmin'] = vmin
    config['vmax'] = vmax
    config['n_total'] = n_total
    config['n_trainable'] = n_trainable
    config['expected_epochs'] = expected_epochs
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_terms = []
    start_time = time.time()
    wall_limit = args.wall_hours * 3600

    print(f"\nTraining for up to {args.epochs} epochs ({args.wall_hours}h wall limit)...")
    print(f"  noise_dim={args.noise_dim}, K={args.K} draws/sample, LR={args.lr}, BS={args.batch_size}")
    print(f"  Expected ~{expected_epochs} epochs, cosine T_max={expected_epochs}")

    for epoch in range(args.epochs):
        elapsed = time.time() - start_time
        if elapsed > wall_limit:
            print(f"\nWall time limit reached ({elapsed/3600:.2f}h). Stopping.")
            break

        model.train()
        epoch_loss = 0.0
        epoch_t1 = 0.0
        epoch_t2 = 0.0
        n_batches = 0

        for lr_batch, hr_batch in train_loader:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            ensemble = model(lr_batch, K=args.K)  # (B, K, 1, H, W)
            loss, t1, t2 = energy_score_loss(ensemble, hr_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0
            )
            optimizer.step()

            epoch_loss += loss.item()
            epoch_t1 += t1
            epoch_t2 += t2
            n_batches += 1

        scheduler.step()
        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)
        train_terms.append((epoch_t1 / n_batches, epoch_t2 / n_batches))

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for lr_batch, hr_batch in val_loader:
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)
                ensemble = model(lr_batch, K=args.K)
                loss, _, _ = energy_score_loss(ensemble, hr_batch)
                val_loss += loss.item()
                n_val += 1
        val_loss /= n_val
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_data = {
                'tail': model.tail.state_dict(),
                'noise_proj': model.noise_proj.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'optimizer': optimizer.state_dict(),
                'vmin': vmin,
                'vmax': vmax,
                'noise_dim': args.noise_dim,
            }
            torch.save(ckpt_data, save_dir / 'best_noise_swinir.pt')

        elapsed = time.time() - start_time
        lr_now = scheduler.get_last_lr()[0]
        t1_avg, t2_avg = train_terms[-1]
        print(f"  Ep {epoch+1:3d} | loss: {train_loss:.6f} (t1={t1_avg:.4f} t2={t2_avg:.4f}) | "
              f"val: {val_loss:.6f} | best: {best_val_loss:.6f} | lr: {lr_now:.2e} | {elapsed/60:.1f}min")

    # Save final
    final_data = {
        'tail': model.tail.state_dict(),
        'noise_proj': model.noise_proj.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
        'vmin': vmin,
        'vmax': vmax,
        'noise_dim': args.noise_dim,
    }
    torch.save(final_data, save_dir / 'final_noise_swinir.pt')
    torch.save({
        'train': train_losses, 'val': val_losses, 'terms': train_terms
    }, save_dir / 'losses.pt')

    elapsed = time.time() - start_time
    print(f"\nTraining complete. {elapsed/60:.1f} minutes, {epoch+1} epochs.")
    print(f"Best val energy score: {best_val_loss:.6f}")


def evaluate(args):
    """Evaluate noise-conditioned SwinIR on test set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = SAVE_DIR
    ckpt_name = f'{args.checkpoint}_noise_swinir.pt'
    ckpt = torch.load(save_dir / ckpt_name, map_location='cpu', weights_only=False)
    vmin = ckpt['vmin']
    vmax = ckpt['vmax']
    noise_dim = ckpt['noise_dim']
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.6f}, noise_dim={noise_dim}")

    # Build model
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from finetune_swinir import load_swinir_1ch

    weights_path = POOL / "research5" / "pretrained_weights" / "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
    ft_ckpt = torch.load(
        POOL / "research5" / "models" / "swinir_ft" / "best_swinir.pt",
        map_location='cpu', weights_only=False
    )
    backbone = load_swinir_1ch(str(weights_path))
    backbone.load_state_dict(ft_ckpt['model'])

    model = NoiseCondSwinIR(backbone, noise_dim=noise_dim)
    model.tail.load_state_dict(ckpt['tail'])
    model.noise_proj.load_state_dict(ckpt['noise_proj'])
    model = model.to(device).eval()

    K = args.K
    print(f"Evaluating with K={K} ensemble members on {args.split}")

    # Load test data
    lr, hr = load_data(args.split)
    N = lr.shape[0]
    if args.n_samples:
        N = min(N, args.n_samples)
        lr = lr[:N]
        hr = hr[:N]

    lr_n = normalize(lr, vmin, vmax)

    # Generate ensemble predictions
    all_ensemble = []
    batch_size = args.batch_size
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = lr_n[start:end].to(device)
            ens = model(batch, K=K)  # (B, K, 1, H, W)
            ens_phys = denormalize(ens, vmin, vmax)
            all_ensemble.append(ens_phys.cpu())
            if start % (batch_size * 10) == 0:
                print(f"  {end}/{N}")

    all_ensemble = torch.cat(all_ensemble, dim=0).numpy()
    all_ensemble = all_ensemble[:, :, 0, :, :]  # (N, K, 128, 128)

    hr_np = hr[:N, 0].numpy()
    lr_np = lr[:N, 0].numpy()

    # CRPS
    crps_values = []
    for i in range(N):
        samples = all_ensemble[i]  # (K, 128, 128)
        obs = hr_np[i]
        M = samples.shape[0]
        t1 = np.mean(np.abs(samples - obs[None, ...]))
        fc_sorted = np.sort(samples, axis=0)
        spread = np.zeros_like(obs)
        for j in range(M):
            w = (2.0 * (j + 1) - M - 1.0) / (M * M)
            spread += w * fc_sorted[j]
        crps_values.append(float(np.mean(t1 - spread)))

    crps = np.mean(crps_values)

    # Ensemble mean metrics
    ens_mean = all_ensemble.mean(axis=1)
    mae = np.mean(np.abs(hr_np - ens_mean))
    rmse = np.sqrt(np.mean((hr_np - ens_mean) ** 2))
    spread_val = np.mean(np.std(all_ensemble, axis=1))

    # Mass violation
    ens_mean_ds = ens_mean.reshape(N, 32, 4, 32, 4).mean(axis=(2, 4))
    mass_viol = np.mean(np.abs(ens_mean_ds - lr_np))

    # With AddCL on each member
    crps_addcl = []
    for i in range(N):
        members_c = apply_addcl_batch(all_ensemble[i], np.tile(lr_np[i:i+1], (K, 1, 1)))
        obs = hr_np[i]
        t1 = np.mean(np.abs(members_c - obs[None, ...]))
        fc_sorted = np.sort(members_c, axis=0)
        spread_c = np.zeros_like(obs)
        for j in range(K):
            w = (2.0 * (j + 1) - K - 1.0) / (K * K)
            spread_c += w * fc_sorted[j]
        crps_addcl.append(float(np.mean(t1 - spread_c)))
    crps_addcl = np.mean(crps_addcl)

    # AddCL ensemble mean
    ens_mean_c = apply_addcl_batch(ens_mean, lr_np)
    mae_c = np.mean(np.abs(hr_np - ens_mean_c))
    ens_mean_c_ds = ens_mean_c.reshape(N, 32, 4, 32, 4).mean(axis=(2, 4))
    mass_viol_c = np.mean(np.abs(ens_mean_c_ds - lr_np))

    print(f"\n{'='*60}")
    print(f"Results: Noise-Conditioned SwinIR ({args.split}, N={N}, K={K})")
    print(f"{'='*60}")
    print(f"  CRPS (energy):     {crps:.6f}")
    print(f"  MAE (ens. mean):   {mae:.6f}")
    print(f"  RMSE (ens. mean):  {rmse:.6f}")
    print(f"  Spread (std):      {spread_val:.6f}")
    print(f"  Mass violation:    {mass_viol:.6f}")
    print(f"\n  With AddCL:")
    print(f"  CRPS (energy):     {crps_addcl:.6f}")
    print(f"  MAE (ens. mean):   {mae_c:.6f}")
    print(f"  Mass violation:    {mass_viol_c:.6f}")
    print(f"\n  Reference:")
    print(f"  Multi-head K=8:    CRPS=0.183")
    print(f"  OT-CFM (research2) CRPS: 0.171")
    print(f"{'='*60}")

    # Save results
    results = {
        'CRPS': crps, 'CRPS_addcl': crps_addcl,
        'MAE': mae, 'MAE_addcl': mae_c,
        'RMSE': rmse, 'Spread': spread_val,
        'Mass_viol': mass_viol, 'Mass_viol_addcl': mass_viol_c,
        'K': K, 'N': N, 'noise_dim': noise_dim,
    }
    torch.save(results, save_dir / 'eval_results.pt')
    print(f"Saved results to {save_dir / 'eval_results.pt'}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", choices=["train", "eval"])
    parser.add_argument("--noise_dim", type=int, default=16)
    parser.add_argument("--K", type=int, default=8, help="Noise draws per sample (train) or ensemble size (eval)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--wall_hours", type=float, default=2.0)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
