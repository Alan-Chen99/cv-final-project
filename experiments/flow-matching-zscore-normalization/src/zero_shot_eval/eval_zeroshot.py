"""
Zero-shot evaluation of pretrained image SR models on ERA5 TCW 4x downscaling.

Methods:
  - bicubic: torch bicubic interpolation
  - swinir: SwinIR classical SR x4 (pretrained on DF2K natural images)
  - swinir-finetune: SwinIR finetuned on ERA5 data

Data: 32x32 -> 128x128, single-channel TCW
CRPS: Energy CRPS = E|X-y| - 0.5*E|X-X'| (corrected formula)

For deterministic models, CRPS = MAE (no ensemble spread term).
Ensemble can be created via test-time augmentation (flips/rotations).

Usage:
  python src/zero_shot_eval/eval_zeroshot.py --method bicubic
  python src/zero_shot_eval/eval_zeroshot.py --method swinir --weights path/to/swinir.pth
  python src/zero_shot_eval/eval_zeroshot.py --method swinir --tta  # 8-fold TTA ensemble
  python src/zero_shot_eval/eval_zeroshot.py --method swinir-finetune --mode train --epochs 50
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Add the zero_shot_eval directory to path for SwinIR import
sys.path.insert(0, os.path.dirname(__file__))


# ---------- Data Loading ----------

def load_data(basedir, split):
    """Load ERA5 TCW data.

    Returns:
        lr: (N, 1, 32, 32) low-resolution input
        hr: (N, 1, 128, 128) high-resolution target
        lr_up: (N, 1, 128, 128) bicubic-upsampled LR
    """
    inp = torch.load(f'{basedir}/{split}/input_{split}.pt', weights_only=False)
    tgt = torch.load(f'{basedir}/{split}/target_{split}.pt', weights_only=False)
    lr = inp[:, 0, :, :, :]  # (N, 1, 32, 32)
    hr = tgt[:, 0, :, :, :]  # (N, 1, 128, 128)
    lr_up = F.interpolate(lr, size=(128, 128), mode='bicubic', align_corners=False)
    return lr, hr, lr_up


# ---------- CRPS ----------

def crps_ensemble_correct(observation, forecasts):
    """Correct energy CRPS: E|X-y| - 0.5*E|X-X'|.

    Args:
        observation: (H, W) ground truth
        forecasts: (M, H, W) ensemble members
    Returns:
        Scalar CRPS value
    """
    M = forecasts.shape[0]
    mae_term = np.mean(np.abs(forecasts - observation[None, ...]), axis=0)
    if M > 1:
        spread = np.zeros_like(observation)
        for i in range(M):
            for j in range(i + 1, M):
                spread += np.abs(forecasts[i] - forecasts[j])
        spread = spread * 2.0 / (M * (M - 1))
    else:
        spread = 0.0
    crps = mae_term - 0.5 * spread
    return float(np.mean(crps))


# ---------- Constraint Layers ----------

def apply_addcl(pred_hr, lr_orig, factor=4):
    """AddCL: additive correction so avgpool(pred_hr) == lr_orig."""
    pooled = F.avg_pool2d(pred_hr, factor)
    correction = lr_orig - pooled
    correction_hr = correction.repeat_interleave(factor, dim=-2).repeat_interleave(factor, dim=-1)
    return pred_hr + correction_hr


# ---------- TTA (Test-Time Augmentation) ----------

def tta_augment(x, idx):
    """Apply augmentation idx (0-7) to a batch."""
    if idx == 0: return x
    if idx == 1: return x.flip(-1)       # h-flip
    if idx == 2: return x.flip(-2)       # v-flip
    if idx == 3: return x.flip(-1, -2)   # h+v flip
    if idx == 4: return x.rot90(1, [-2, -1])  # 90 CW
    if idx == 5: return x.rot90(1, [-2, -1]).flip(-1)
    if idx == 6: return x.rot90(1, [-2, -1]).flip(-2)
    if idx == 7: return x.rot90(1, [-2, -1]).flip(-1, -2)
    return x

def tta_deaugment(x, idx):
    """Reverse augmentation idx."""
    if idx == 0: return x
    if idx == 1: return x.flip(-1)
    if idx == 2: return x.flip(-2)
    if idx == 3: return x.flip(-1, -2)
    if idx == 4: return x.rot90(-1, [-2, -1])
    if idx == 5: return x.flip(-1).rot90(-1, [-2, -1])
    if idx == 6: return x.flip(-2).rot90(-1, [-2, -1])
    if idx == 7: return x.flip(-1, -2).rot90(-1, [-2, -1])
    return x


# ---------- SwinIR Setup ----------

def load_swinir(weights_path, device='cuda'):
    """Load SwinIR classical SR x4 model."""
    from network_swinir import SwinIR
    model = SwinIR(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    )
    pretrained = torch.load(weights_path, map_location='cpu', weights_only=False)
    if 'params' in pretrained:
        pretrained = pretrained['params']
    elif 'params_ema' in pretrained:
        pretrained = pretrained['params_ema']
    model.load_state_dict(pretrained, strict=True)
    model = model.to(device).eval()
    return model


def adapt_input_for_swinir(lr, min_val, max_val):
    """Convert 1-channel climate data to 3-channel [0,1] for SwinIR.

    Args:
        lr: (N, 1, 32, 32) raw climate values
        min_val, max_val: normalization range from training data
    Returns:
        (N, 3, 32, 32) normalized to [0, 1]
    """
    normalized = (lr - min_val) / (max_val - min_val + 1e-8)
    normalized = normalized.clamp(0, 1)
    return normalized.expand(-1, 3, -1, -1)  # replicate to 3 channels


def adapt_output_from_swinir(out, min_val, max_val):
    """Convert 3-channel [0,1] SwinIR output back to 1-channel climate values.

    Args:
        out: (N, 3, 128, 128) SwinIR output in [0, 1]
    Returns:
        (N, 1, 128, 128) denormalized climate values
    """
    # Average across channels
    out_1ch = out.mean(dim=1, keepdim=True)
    out_1ch = out_1ch.clamp(0, 1)
    return out_1ch * (max_val - min_val) + min_val


# ---------- SwinIR with 1-channel adapter ----------

class SwinIR1Ch(nn.Module):
    """Wrapper: 1-ch input → SwinIR (3-ch) → 1-ch output, with learnable adapters."""
    def __init__(self, swinir_model, freeze_backbone=True):
        super().__init__()
        self.swinir = swinir_model
        self.input_adapter = nn.Conv2d(1, 3, 1, bias=True)
        self.output_adapter = nn.Conv2d(3, 1, 1, bias=True)
        # Initialize: replicate input, average output
        with torch.no_grad():
            self.input_adapter.weight.fill_(1.0 / 3.0)
            self.input_adapter.bias.zero_()
            self.output_adapter.weight.fill_(1.0 / 3.0)
            self.output_adapter.bias.zero_()
        if freeze_backbone:
            for p in self.swinir.parameters():
                p.requires_grad = False

    def forward(self, x):
        x3 = self.input_adapter(x)
        out3 = self.swinir(x3)
        return self.output_adapter(out3)

    def unfreeze_backbone(self):
        for p in self.swinir.parameters():
            p.requires_grad = True


# ---------- Evaluation ----------

@torch.no_grad()
def evaluate(predictions_list, hr_np, lr_np, method_name, max_samples=None):
    """Evaluate predictions.

    Args:
        predictions_list: list of (N, 1, 128, 128) numpy arrays (ensemble members)
        hr_np: (N, 1, 128, 128) ground truth numpy
        lr_np: (N, 1, 32, 32) LR input numpy (for mass violation)
        method_name: string for logging
        max_samples: limit evaluation to first N samples
    """
    M = len(predictions_list)
    N = hr_np.shape[0]
    if max_samples:
        N = min(N, max_samples)

    # Stack ensemble: (M, N, H, W)
    ensemble = np.stack([p[:N, 0] for p in predictions_list], axis=0)  # (M, N, H, W)
    targets = hr_np[:N, 0]  # (N, H, W)

    # CRPS
    crps_values = []
    for i in range(N):
        c = crps_ensemble_correct(targets[i], ensemble[:, i])
        crps_values.append(c)
    crps = float(np.mean(crps_values))

    # Deterministic metrics (ensemble mean)
    mean_pred = ensemble.mean(axis=0)  # (N, H, W)
    rmse = float(np.sqrt(np.mean((mean_pred - targets) ** 2)))
    mae = float(np.mean(np.abs(mean_pred - targets)))

    # Mass violation
    lr_orig = lr_np[:N, 0]  # (N, 32, 32)
    ds_pred = mean_pred.reshape(N, 32, 4, 32, 4).mean(axis=(2, 4))  # avg_pool 4x
    mass_viol = float(np.mean(np.abs(ds_pred - lr_orig)))

    # Ensemble spread
    spread = float(np.mean(np.std(ensemble, axis=0))) if M > 1 else 0.0

    print(f"\n{'='*60}")
    print(f"  {method_name} (M={M}, N={N})")
    print(f"{'='*60}")
    print(f"  CRPS (corrected): {crps:.6f}")
    print(f"  RMSE:             {rmse:.6f}")
    print(f"  MAE:              {mae:.6f}")
    print(f"  Mass violation:   {mass_viol:.6f}")
    print(f"  Ensemble spread:  {spread:.6f}")
    print(f"{'='*60}\n")

    return {
        'method': method_name,
        'crps': crps,
        'rmse': rmse,
        'mae': mae,
        'mass_viol': mass_viol,
        'spread': spread,
        'M': M,
        'N': N,
    }


# ---------- Methods ----------

@torch.no_grad()
def run_bicubic(lr, device='cpu'):
    """Bicubic interpolation baseline."""
    lr_up = F.interpolate(lr.to(device), size=(128, 128), mode='bicubic', align_corners=False)
    return lr_up.cpu().numpy()


@torch.no_grad()
def run_swinir_zeroshot(lr, model, min_val, max_val, device='cuda', batch_size=64, tta=False):
    """Run SwinIR zero-shot on climate data.

    Args:
        lr: (N, 1, 32, 32) raw climate values
        model: loaded SwinIR model
        min_val, max_val: normalization range
        tta: if True, return 8 TTA members; else return 1 prediction

    Returns:
        list of (N, 1, 128, 128) numpy arrays
    """
    N = lr.shape[0]
    augments = list(range(8)) if tta else [0]
    all_predictions = []

    for aug_idx in augments:
        predictions = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = lr[start:end].to(device)

            # Augment
            batch = tta_augment(batch, aug_idx)

            # Adapt for SwinIR
            batch_3ch = adapt_input_for_swinir(batch, min_val, max_val)

            # Pad to multiple of window_size (8)
            _, _, h, w = batch_3ch.shape
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            if pad_h > 0 or pad_w > 0:
                batch_3ch = F.pad(batch_3ch, (0, pad_w, 0, pad_h), mode='reflect')

            out = model(batch_3ch)

            # Remove padding in output space (4x scale)
            if pad_h > 0 or pad_w > 0:
                out = out[:, :, :h*4, :w*4]

            # Convert back to climate space
            out_1ch = adapt_output_from_swinir(out, min_val, max_val)

            # De-augment
            out_1ch = tta_deaugment(out_1ch, aug_idx)

            predictions.append(out_1ch.cpu())

        pred_full = torch.cat(predictions, dim=0).numpy()
        all_predictions.append(pred_full)

    return all_predictions


@torch.no_grad()
def run_swinir1ch(lr, model, min_val, max_val, device='cuda', batch_size=64, tta=False):
    """Run SwinIR1Ch (1-channel adapter) on climate data."""
    N = lr.shape[0]
    augments = list(range(8)) if tta else [0]
    all_predictions = []

    # Normalize to [0,1]
    lr_norm = (lr - min_val) / (max_val - min_val + 1e-8)
    lr_norm = lr_norm.clamp(0, 1)

    for aug_idx in augments:
        predictions = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = lr_norm[start:end].to(device)
            batch = tta_augment(batch, aug_idx)

            # Pad
            _, _, h, w = batch.shape
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            if pad_h > 0 or pad_w > 0:
                batch = F.pad(batch, (0, pad_w, 0, pad_h), mode='reflect')

            out = model(batch)

            if pad_h > 0 or pad_w > 0:
                out = out[:, :, :h*4, :w*4]

            out = tta_deaugment(out, aug_idx)

            # Denormalize
            out_denorm = out.cpu() * (max_val - min_val) + min_val
            predictions.append(out_denorm)

        pred_full = torch.cat(predictions, dim=0).numpy()
        all_predictions.append(pred_full)

    return all_predictions


# ---------- Training (finetune) ----------

def train_swinir_finetune(args):
    """Finetune SwinIR on ERA5 data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    basedir = args.data_dir

    print("Loading data...")
    lr_train, hr_train, lr_up_train = load_data(basedir, 'train')
    lr_val, hr_val, lr_up_val = load_data(basedir, 'val')

    # Normalization stats
    min_val = hr_train.min().item()
    max_val = hr_train.max().item()

    # Normalize to [0, 1]
    lr_train_n = (lr_train - min_val) / (max_val - min_val)
    hr_train_n = (hr_train - min_val) / (max_val - min_val)
    lr_val_n = (lr_val - min_val) / (max_val - min_val)
    hr_val_n = (hr_val - min_val) / (max_val - min_val)

    train_ds = TensorDataset(lr_train_n, hr_train_n)
    val_ds = TensorDataset(lr_val_n, hr_val_n)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Load pretrained SwinIR
    print("Loading SwinIR backbone...")
    swinir_backbone = load_swinir(args.weights, device='cpu')
    model = SwinIR1Ch(swinir_backbone, freeze_backbone=(not args.unfreeze_all))
    model = model.to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_total:,} total, {n_trainable:,} trainable")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss
    loss_fn = nn.L1Loss() if args.loss == 'l1' else nn.MSELoss()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'min_val': min_val, 'max_val': max_val}, os.path.join(save_dir, 'norm_stats.pt'))

    best_val_loss = float('inf')
    start_time = time.time()

    # Phase control for progressive unfreezing
    unfreeze_epoch = args.unfreeze_epoch if hasattr(args, 'unfreeze_epoch') else -1

    for epoch in range(args.epochs):
        # Progressive unfreezing
        if epoch == unfreeze_epoch and not args.unfreeze_all:
            print(f"Epoch {epoch}: Unfreezing backbone!")
            model.unfreeze_backbone()
            # Reset optimizer with lower LR for backbone
            optimizer = torch.optim.AdamW([
                {'params': model.input_adapter.parameters(), 'lr': args.lr},
                {'params': model.output_adapter.parameters(), 'lr': args.lr},
                {'params': model.swinir.parameters(), 'lr': args.lr * 0.1},
            ], weight_decay=1e-5)
            remaining = args.epochs - epoch
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)

        model.train()
        train_loss = 0
        for lr_batch, hr_batch in train_loader:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            pred = model(lr_batch)
            loss = loss_fn(pred, hr_batch)

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
            for lr_batch, hr_batch in val_loader:
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)
                pred = model(lr_batch)
                val_loss += loss_fn(pred, hr_batch).item()
        val_loss /= len(val_loader)

        elapsed = time.time() - start_time
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} | train: {train_loss:.6f} | val: {val_loss:.6f} | "
              f"lr: {lr_now:.2e} | time: {elapsed/60:.1f}m")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'min_val': min_val,
                'max_val': max_val,
            }, os.path.join(save_dir, 'best_swinir1ch.pt'))
            print(f"  -> Saved best model (val_loss={val_loss:.6f})")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} min. Best val loss: {best_val_loss:.6f}")


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='bicubic',
                        choices=['bicubic', 'swinir', 'swinir-finetune'])
    parser.add_argument('--mode', default='eval', choices=['eval', 'train', 'both'])
    parser.add_argument('--data-dir', default='/home/chenxy/orcd/pool/datasets/era5_sr_data')
    parser.add_argument('--weights', default='/home/chenxy/orcd/pool/datasets/research6/weights/swinir_classical_x4.pth')
    parser.add_argument('--save-dir', default='/home/chenxy/orcd/pool/datasets/research6/models/swinir_finetune')
    parser.add_argument('--split', default='test')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--tta', action='store_true', help='8-fold TTA ensemble')
    parser.add_argument('--constraint', default='none', choices=['none', 'addcl'])
    # Training args
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loss', default='l1', choices=['l1', 'mse'])
    parser.add_argument('--unfreeze-all', action='store_true')
    parser.add_argument('--unfreeze-epoch', type=int, default=10)
    args = parser.parse_args()

    if args.mode in ('train', 'both') and args.method == 'swinir-finetune':
        train_swinir_finetune(args)

    if args.mode in ('eval', 'both'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")

        # Load data
        lr, hr, lr_up = load_data(args.data_dir, args.split)
        N = lr.shape[0]
        if args.max_samples:
            N = min(N, args.max_samples)
            lr = lr[:N]
            hr = hr[:N]
            lr_up = lr_up[:N]

        print(f"Data: {N} samples, LR {lr.shape}, HR {hr.shape}")

        # Normalization stats from training set
        train_hr = torch.load(f'{args.data_dir}/train/target_train.pt', weights_only=False)
        min_val = train_hr[:, 0, :, :, :].min().item()
        max_val = train_hr[:, 0, :, :, :].max().item()
        del train_hr
        print(f"Normalization: min={min_val:.4f}, max={max_val:.4f}")

        results = []

        if args.method == 'bicubic':
            pred_bicubic = run_bicubic(lr, device)
            r = evaluate([pred_bicubic], hr.numpy(), lr.numpy(), 'Bicubic', args.max_samples)
            results.append(r)

            # Also compute with AddCL constraint
            lr_up_constrained = apply_addcl(
                torch.from_numpy(pred_bicubic), lr, factor=4
            ).numpy()
            r2 = evaluate([lr_up_constrained], hr.numpy(), lr.numpy(), 'Bicubic + AddCL', args.max_samples)
            results.append(r2)

        elif args.method == 'swinir':
            print("Loading SwinIR...")
            model = load_swinir(args.weights, device=device)

            print(f"Running SwinIR zero-shot (TTA={args.tta})...")
            preds = run_swinir_zeroshot(lr, model, min_val, max_val, device,
                                        batch_size=args.batch_size, tta=args.tta)

            name = f'SwinIR zero-shot {"TTA8" if args.tta else ""}'
            r = evaluate(preds, hr.numpy(), lr.numpy(), name, args.max_samples)
            results.append(r)

            if args.constraint == 'addcl':
                # Apply AddCL to each member
                preds_c = []
                for p in preds:
                    p_t = torch.from_numpy(p)
                    p_c = apply_addcl(p_t, lr, factor=4)
                    preds_c.append(p_c.numpy())
                r2 = evaluate(preds_c, hr.numpy(), lr.numpy(), f'{name} + AddCL', args.max_samples)
                results.append(r2)

        elif args.method == 'swinir-finetune':
            # Load finetuned model
            ckpt_path = os.path.join(args.save_dir, 'best_swinir1ch.pt')
            if not os.path.exists(ckpt_path):
                print(f"No finetuned model at {ckpt_path}. Run with --mode train first.")
                return

            print("Loading finetuned SwinIR...")
            swinir_backbone = load_swinir(args.weights, device='cpu')
            model = SwinIR1Ch(swinir_backbone, freeze_backbone=False)
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model'])
            model = model.to(device).eval()
            min_val = ckpt['min_val']
            max_val = ckpt['max_val']

            preds = run_swinir1ch(lr, model, min_val, max_val, device,
                                  batch_size=args.batch_size, tta=args.tta)

            name = f'SwinIR-finetune {"TTA8" if args.tta else ""}'
            r = evaluate(preds, hr.numpy(), lr.numpy(), name, args.max_samples)
            results.append(r)

            if args.constraint == 'addcl':
                preds_c = []
                for p in preds:
                    p_t = torch.from_numpy(p)
                    p_c = apply_addcl(p_t, lr, factor=4)
                    preds_c.append(p_c.numpy())
                r2 = evaluate(preds_c, hr.numpy(), lr.numpy(), f'{name} + AddCL', args.max_samples)
                results.append(r2)

        # Summary
        print("\n" + "="*80)
        print("  SUMMARY")
        print("="*80)
        print(f"{'Method':<35} {'CRPS':>10} {'RMSE':>10} {'MAE':>10} {'Mass Viol':>10}")
        print("-"*80)
        for r in results:
            print(f"{r['method']:<35} {r['crps']:>10.6f} {r['rmse']:>10.6f} "
                  f"{r['mae']:>10.6f} {r['mass_viol']:>10.6f}")
        print("="*80)


if __name__ == '__main__':
    main()
