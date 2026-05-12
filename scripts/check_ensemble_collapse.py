"""Check ensemble diversity for ours vs Anvita GAN models."""

import sys
sys.path.insert(0, "/workspace/external/constrained-downscaling")

import numpy as np
import torch
import models
from pathlib import Path

ANVITA = Path("/orcd/pool/007/chenxy/datasets/anvita")
POOL = Path("/home/chenxy/orcd/pool/datasets")

CONFIGS = [
    # (label, path, dataset, upsampling_factor, constraints)
    ("ours ERA5 GAN none", POOL / "organize2/models/harder/twc_gan_none.pth", "era5", 4, "none"),
    ("anvita ERA5 GAN none", ANVITA / "twc_gan_noconstraints.pth", "era5", 4, "none"),
    ("ours ERA5 GAN SmCL", POOL / "organize2/models/harder/twc_gan_softmax.pth", "era5", 4, "softmax"),
    ("anvita ERA5 GAN SmCL", ANVITA / "twc_gan_softmax.pth", "era5", 4, "softmax"),
    ("ours NorESM GAN SmCL", POOL / "noresm-dataset/models/harder/twc_gan_softmax.pth", "noresm", 2, "softmax"),
    ("anvita NorESM GAN SmCL", ANVITA / "noresm_gan_softmax.pth", "noresm", 2, "softmax"),
    ("anvita NorESM GAN none", ANVITA / "noresm_gan_none.pth", "noresm", 2, "none"),
]

N_ENSEMBLE = 10
N_SAMPLES = 50

# Load data
print("Loading data...")
lr_era5 = torch.load(str(POOL / "era5_sr_data/test/input_test.pt"), weights_only=False)[:N_SAMPLES]
tgt_era5 = torch.load(str(POOL / "era5_sr_data/train/target_train.pt"), weights_only=False)
min_e, max_e = float(tgt_era5[:, 0, 0].min()), float(tgt_era5[:, 0, 0].max())
del tgt_era5

from downscaling.data import load_noresm_tas
_, _, _, lr_noresm = load_noresm_tas(str(POOL), "test")
lr_noresm = lr_noresm[:N_SAMPLES]
from downscaling.evaluation.harder import compute_minmax_stats
min_n, max_n = compute_minmax_stats(POOL, "noresm")

print(f"ERA5: {lr_era5.shape}, NorESM: {lr_noresm.shape}")

for label, path, dataset, uf, constraints in CONFIGS:
    if not path.exists():
        print(f"\n{label}: NOT FOUND")
        continue

    m = models.ResNet(
        number_channels=32, number_residual_blocks=4,
        upsampling_factor=uf, noise=True, constraints=constraints, dim=1,
    )
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    m.load_state_dict(ckpt["state_dict"])
    m.to("cuda").eval()

    if dataset == "era5":
        lr, min_val, max_val = lr_era5, min_e, max_e
    else:
        lr, min_val, max_val = lr_noresm, min_n, max_n

    val_range = max_val - min_val
    # NorESM lr is (N,1,H,W), ERA5 is (N,1,1,H,W)
    if lr.ndim == 4:
        lr_norm = ((lr - min_val) / val_range).unsqueeze(1).to("cuda")
    else:
        lr_norm = ((lr - min_val) / val_range).to("cuda")

    # Generate ensemble
    all_preds = []
    with torch.no_grad():
        for _ in range(N_ENSEMBLE):
            z = torch.randn(N_SAMPLES, 100, 1, 1, device="cuda")
            out = m(lr_norm, z)
            pred = out.cpu().squeeze() * val_range + min_val
            all_preds.append(pred.numpy())

    ens = np.stack(all_preds, axis=1)  # (N, M, H, W)

    # Ensemble statistics
    ens_std = ens.std(axis=1)  # (N, H, W) — per-pixel std across members
    ens_spread = ens_std.mean()  # mean spread
    ens_max_diff = (ens.max(axis=1) - ens.min(axis=1)).mean()  # mean max-min range

    # Pairwise correlation between members (sample a few pairs)
    corrs = []
    for i in range(min(5, N_SAMPLES)):
        m1 = ens[i, 0].flatten()
        m2 = ens[i, 1].flatten()
        corrs.append(np.corrcoef(m1, m2)[0, 1])
    mean_corr = np.mean(corrs)

    print(f"\n{label}:")
    print(f"  ensemble spread (mean pixel std): {ens_spread:.6f}")
    print(f"  ensemble max-min range (mean):    {ens_max_diff:.6f}")
    print(f"  member pairwise correlation:      {mean_corr:.6f}")
    print(f"  pred value range:                 [{ens.min():.2f}, {ens.max():.2f}]")

    collapsed = ens_spread < 0.01 * val_range
    print(f"  COLLAPSED: {'YES' if collapsed else 'NO'} (spread/range = {ens_spread/val_range:.6f})")

    del m
    torch.cuda.empty_cache()
