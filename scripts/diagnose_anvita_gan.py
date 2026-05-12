"""Diagnose ERA5 GAN none discrepancy between ours and Anvita."""

import sys
sys.path.insert(0, "/workspace/external/constrained-downscaling")

import numpy as np
import torch
import models
from pathlib import Path

ANVITA = Path("/orcd/pool/007/chenxy/datasets/anvita")
POOL = Path("/home/chenxy/orcd/pool/datasets")

# Load ERA5 test data (shape: N, 1, 1, H, W)
lr = torch.load(str(POOL / "era5_sr_data/test/input_test.pt"), weights_only=False)[:10]
hr = torch.load(str(POOL / "era5_sr_data/test/target_test.pt"), weights_only=False)[:10]
tgt_train = torch.load(str(POOL / "era5_sr_data/train/target_train.pt"), weights_only=False)
min_val = float(tgt_train[:, 0, 0, ...].min())
max_val = float(tgt_train[:, 0, 0, ...].max())
val_range = max_val - min_val
print(f"ERA5 min={min_val:.6f} max={max_val:.6f} range={val_range:.6f}")
print(f"LR shape: {lr.shape}, HR shape: {hr.shape}")
print(f"LR value range: [{lr.min():.4f}, {lr.max():.4f}]")
print(f"HR value range: [{hr.min():.4f}, {hr.max():.4f}]")

# Fixed seed for comparable noise
torch.manual_seed(42)
z = torch.randn(10, 100, 1, 1, device="cuda")

for label, path in [
    ("ours", POOL / "organize2/models/harder/twc_gan_none.pth"),
    ("anvita", ANVITA / "twc_gan_noconstraints.pth"),
]:
    m = models.ResNet(
        number_channels=32, number_residual_blocks=4,
        upsampling_factor=4, noise=True, constraints="none", dim=1,
    )
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    m.load_state_dict(ckpt["state_dict"])
    m.to("cuda").eval()

    # Normalize LR (already (B,1,1,32,32) — no unsqueeze needed)
    lr_norm = (lr - min_val) / val_range
    lr_in = lr_norm.to("cuda")

    with torch.no_grad():
        out = m(lr_in, z.clone())

    # Raw model output (before denorm)
    raw = out.cpu()
    print(f"\n{label}:")
    print(f"  raw output range: [{raw.min():.6f}, {raw.max():.6f}]")
    print(f"  raw output mean: {raw.mean():.6f}, std: {raw.std():.6f}")

    # Denormalize
    pred = raw.squeeze(1) * val_range + min_val
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)

    truth = hr[:, 0, 0].numpy()
    p = pred[:, 0].numpy()
    mae = np.mean(np.abs(truth - p))
    print(f"  denorm pred range: [{p.min():.2f}, {p.max():.2f}]")
    print(f"  truth range: [{truth.min():.2f}, {truth.max():.2f}]")
    print(f"  pred mean={p.mean():.4f} std={p.std():.4f}")
    print(f"  truth mean={truth.mean():.4f} std={truth.std():.4f}")
    print(f"  MAE={mae:.4f}")

    del m
    torch.cuda.empty_cache()
