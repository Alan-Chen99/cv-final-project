"""Quick visual check of downscaling predictions vs ground truth."""
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

def main():
    dataset = "era5_sr_data"
    model_id = sys.argv[1] if len(sys.argv) > 1 else "test_cnn_none"
    split = sys.argv[2] if len(sys.argv) > 2 else "val"
    n_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 6

    lr = torch.load(f"./data/{dataset}/{split}/input_{split}.pt", weights_only=False)
    hr = torch.load(f"./data/{dataset}/{split}/target_{split}.pt", weights_only=False)
    pred = torch.load(f"./data/prediction/{dataset}_{model_id}_{split}.pt", weights_only=False)

    # Pick evenly spaced samples
    indices = np.linspace(0, lr.shape[0] - 1, n_samples, dtype=int)

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]

    vmin = hr.min().item()
    vmax = hr.max().item()

    for row, idx in enumerate(indices):
        lr_img = lr[idx].squeeze().numpy()
        hr_img = hr[idx].squeeze().numpy()
        pred_img = pred[idx].squeeze().numpy()

        axes[row, 0].imshow(lr_img, vmin=vmin, vmax=vmax, cmap="viridis")
        axes[row, 0].set_title(f"LR input [{idx}]" if row == 0 else f"[{idx}]")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(hr_img, vmin=vmin, vmax=vmax, cmap="viridis")
        axes[row, 1].set_title(f"HR target [{idx}]" if row == 0 else f"[{idx}]")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_img, vmin=vmin, vmax=vmax, cmap="viridis")
        axes[row, 2].set_title(f"Prediction [{idx}]" if row == 0 else f"[{idx}]")
        axes[row, 2].axis("off")

    fig.suptitle(f"{model_id} — {split} set", fontsize=14)
    fig.tight_layout()
    out = f"./data/{model_id}_visual.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.close()

if __name__ == "__main__":
    main()
