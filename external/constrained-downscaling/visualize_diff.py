"""Error visualization: prediction - target, plus per-sample mass violation."""
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform

def main():
    dataset = "era5_sr_data"
    model_id = sys.argv[1] if len(sys.argv) > 1 else "test_cnn_none"
    split = sys.argv[2] if len(sys.argv) > 2 else "val"
    n_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    upsample = 4

    lr = torch.load(f"./data/{dataset}/{split}/input_{split}.pt", weights_only=False)
    hr = torch.load(f"./data/{dataset}/{split}/target_{split}.pt", weights_only=False)
    pred = torch.load(f"./data/prediction/{dataset}_{model_id}_{split}.pt", weights_only=False)

    indices = np.linspace(0, lr.shape[0] - 1, n_samples, dtype=int)

    fig, axes = plt.subplots(n_samples, 2, figsize=(8, 3 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]

    for row, idx in enumerate(indices):
        lr_img = lr[idx].squeeze().numpy()
        hr_img = hr[idx].squeeze().numpy()
        pred_img = pred[idx].squeeze().numpy()

        diff = pred_img - hr_img
        abs_max = max(abs(diff.min()), abs(diff.max()))

        # Error map
        im = axes[row, 0].imshow(diff, vmin=-abs_max, vmax=abs_max, cmap="RdBu_r")
        rmse = np.sqrt(np.mean(diff**2))
        axes[row, 0].set_title(f"[{idx}] Error (RMSE={rmse:.2f})")
        axes[row, 0].axis("off")
        plt.colorbar(im, ax=axes[row, 0], fraction=0.046)

        # Mass violation map: downscale pred and compare to LR
        pred_ds = transform.downscale_local_mean(pred_img, (upsample, upsample))
        mass_diff = pred_ds - lr_img
        abs_max_m = max(abs(mass_diff.min()), abs(mass_diff.max()), 1e-6)

        im2 = axes[row, 1].imshow(mass_diff, vmin=-abs_max_m, vmax=abs_max_m, cmap="RdBu_r")
        mv = np.mean(np.abs(mass_diff))
        axes[row, 1].set_title(f"[{idx}] Mass viol (MAE={mv:.3f})")
        axes[row, 1].axis("off")
        plt.colorbar(im2, ax=axes[row, 1], fraction=0.046)

    fig.suptitle(f"{model_id} — error analysis ({split})", fontsize=14)
    fig.tight_layout()
    out = f"./data/{model_id}_errors.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.close()

if __name__ == "__main__":
    main()
