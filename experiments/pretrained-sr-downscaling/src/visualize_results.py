"""Generate visualizations for SwinIR finetuning experiment."""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

POOL = Path("/home/chenxy/orcd/pool/datasets")
DATA_DIR = POOL / "era5_sr_data"
SAVE_DIR = POOL / "research5" / "models" / "swinir_ft"
OUT_DIR = Path("/home/chenxy/repos/workspace/research5/experiments/pretrained-sr-downscaling/figures")


def plot_loss_curves():
    """Plot training and validation loss curves."""
    losses = torch.load(SAVE_DIR / "losses.pt", weights_only=False)
    train_losses = losses['train']
    val_losses = losses['val']

    # Convert to physical MAE (approximate)
    vmin, vmax = 0.0444, 130.8358
    scale = vmax - vmin
    train_mae = [l * scale for l in train_losses]
    val_mae = [l * scale for l in val_losses]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(train_losses) + 1)

    # Normalized loss
    ax1.plot(epochs, train_losses, 'b-', label='Train', linewidth=1.5)
    ax1.plot(epochs, val_losses, 'r-', label='Val', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('L1 Loss (normalized)')
    ax1.set_title('SwinIR Finetuning: Normalized L1 Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Physical MAE
    ax2.plot(epochs, train_mae, 'b-', label='Train', linewidth=1.5)
    ax2.plot(epochs, val_mae, 'r-', label='Val', linewidth=1.5)
    ax2.axhline(y=0.317, color='gray', linestyle='--', alpha=0.7, label='Zero-shot MAE')
    ax2.axhline(y=0.247, color='green', linestyle='--', alpha=0.7, label='OT-CFM MAE (research2)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE (physical units, kg/m²)')
    ax2.set_title('SwinIR Finetuning: Physical MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved loss curves to {OUT_DIR / 'loss_curves.png'}")


def plot_sample_predictions():
    """Plot sample LR, HR, and predictions."""
    from spandrel import ModelLoader
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from finetune_swinir import load_swinir_1ch, normalize, denormalize

    # Load model
    weights_path = POOL / "research5" / "pretrained_weights" / "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
    ckpt = torch.load(SAVE_DIR / "best_swinir.pt", map_location='cpu', weights_only=False)
    model = load_swinir_1ch(str(weights_path))
    model.load_state_dict(ckpt['model'])
    model.eval()
    vmin, vmax = ckpt['vmin'], ckpt['vmax']

    # Load test data
    inputs = torch.load(DATA_DIR / "test" / "input_test.pt", map_location="cpu", weights_only=True)
    targets = torch.load(DATA_DIR / "test" / "target_test.pt", map_location="cpu", weights_only=True)

    # Pick diverse samples
    indices = [0, 100, 500, 1000, 2000, 5000]

    fig, axes = plt.subplots(len(indices), 4, figsize=(16, 4 * len(indices)))

    for row, idx in enumerate(indices):
        lr = inputs[idx, 0, 0]  # (32, 32)
        hr = targets[idx, 0, 0]  # (128, 128)
        lr_up = F.interpolate(lr.unsqueeze(0).unsqueeze(0), size=(128, 128),
                               mode='bicubic', align_corners=False)[0, 0]

        # SwinIR prediction
        lr_n = normalize(inputs[idx:idx+1, 0], vmin, vmax)
        with torch.no_grad():
            pred_n = model(lr_n)
        pred = denormalize(pred_n, vmin, vmax)[0, 0]

        # Common colorbar range
        vmin_plot = min(hr.min().item(), pred.min().item(), lr_up.min().item())
        vmax_plot = max(hr.max().item(), pred.max().item(), lr_up.max().item())

        for col, (data, title) in enumerate([
            (lr.numpy(), f'LR (32x32)\nidx={idx}'),
            (lr_up.numpy(), 'Bicubic (128x128)'),
            (pred.numpy(), 'SwinIR-FT (128x128)'),
            (hr.numpy(), 'HR Ground Truth'),
        ]):
            ax = axes[row, col]
            if col == 0:
                im = ax.imshow(data, cmap='viridis', vmin=vmin_plot, vmax=vmax_plot)
            else:
                im = ax.imshow(data, cmap='viridis', vmin=vmin_plot, vmax=vmax_plot)
            ax.set_title(title, fontsize=9)
            ax.axis('off')

            if col == 3:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('SwinIR Finetuned: Sample Predictions (TCW 4x Downscaling)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'sample_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sample predictions to {OUT_DIR / 'sample_predictions.png'}")


def plot_error_maps():
    """Plot error maps comparing bicubic vs SwinIR."""
    from spandrel import ModelLoader
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from finetune_swinir import load_swinir_1ch, normalize, denormalize

    # Load model
    weights_path = POOL / "research5" / "pretrained_weights" / "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
    ckpt = torch.load(SAVE_DIR / "best_swinir.pt", map_location='cpu', weights_only=False)
    model = load_swinir_1ch(str(weights_path))
    model.load_state_dict(ckpt['model'])
    model.eval()
    vmin, vmax = ckpt['vmin'], ckpt['vmax']

    # Load test data
    inputs = torch.load(DATA_DIR / "test" / "input_test.pt", map_location="cpu", weights_only=True)
    targets = torch.load(DATA_DIR / "test" / "target_test.pt", map_location="cpu", weights_only=True)

    indices = [0, 100, 500, 1000]
    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4 * len(indices)))

    for row, idx in enumerate(indices):
        hr = targets[idx, 0, 0]  # (128, 128)
        lr_up = F.interpolate(inputs[idx:idx+1, 0, 0:1], size=(128, 128),
                               mode='bicubic', align_corners=False)[0, 0]

        lr_n = normalize(inputs[idx:idx+1, 0], vmin, vmax)
        with torch.no_grad():
            pred_n = model(lr_n)
        pred = denormalize(pred_n, vmin, vmax)[0, 0]

        err_bicubic = (hr - lr_up).abs().numpy()
        err_swinir = (hr - pred).abs().numpy()

        vmax_err = max(err_bicubic.max(), err_swinir.max())

        ax = axes[row, 0]
        ax.imshow(hr.numpy(), cmap='viridis')
        ax.set_title(f'HR (idx={idx})', fontsize=9)
        ax.axis('off')

        ax = axes[row, 1]
        im = ax.imshow(err_bicubic, cmap='hot', vmin=0, vmax=vmax_err)
        ax.set_title(f'|Bicubic - HR| (MAE={err_bicubic.mean():.3f})', fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[row, 2]
        im = ax.imshow(err_swinir, cmap='hot', vmin=0, vmax=vmax_err)
        ax.set_title(f'|SwinIR - HR| (MAE={err_swinir.mean():.3f})', fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Error Maps: Bicubic vs SwinIR Finetuned', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'error_maps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved error maps to {OUT_DIR / 'error_maps.png'}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_loss_curves()
    plot_sample_predictions()
    plot_error_maps()
    print("\nAll visualizations generated.")
