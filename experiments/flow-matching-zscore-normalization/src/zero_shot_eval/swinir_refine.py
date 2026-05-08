"""
SwinIR + Refinement: Use frozen SwinIR as preprocessor, train a small refinement CNN.

Approach:
  1. SwinIR zero-shot produces rough 128x128 output
  2. Small CNN refines the output (learns residual corrections)
  3. AddCL constraint applied post-hoc

This is fast because:
  - SwinIR features can be precomputed (one-time cost)
  - Refinement CNN is tiny (< 1M params)
  - Only the residual from SwinIR output to ground truth needs to be learned

For CRPS, we add noise-based ensemble diversity at test time.

Usage:
  python src/zero_shot_eval/swinir_refine.py --mode precompute  # precompute SwinIR features
  python src/zero_shot_eval/swinir_refine.py --mode train       # train refinement CNN
  python src/zero_shot_eval/swinir_refine.py --mode eval        # evaluate
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))


# ---------- Data ----------

def load_data(basedir, split):
    """Load ERA5 TCW data."""
    inp = torch.load(f'{basedir}/{split}/input_{split}.pt', weights_only=False)
    tgt = torch.load(f'{basedir}/{split}/target_{split}.pt', weights_only=False)
    lr = inp[:, 0, :, :, :]  # (N, 1, 32, 32)
    hr = tgt[:, 0, :, :, :]  # (N, 1, 128, 128)
    return lr, hr


# ---------- CRPS ----------

def crps_ensemble_correct(observation, forecasts):
    """Correct energy CRPS."""
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


# ---------- Constraint ----------

def apply_addcl(pred_hr, lr_orig, factor=4):
    pooled = F.avg_pool2d(pred_hr, factor)
    correction = lr_orig - pooled
    correction_hr = correction.repeat_interleave(factor, dim=-2).repeat_interleave(factor, dim=-1)
    return pred_hr + correction_hr


# ---------- Refinement CNN ----------

class RefineCNN(nn.Module):
    """Small residual refinement network.

    Input: 2 channels (SwinIR output + bicubic LR) at 128x128
    Output: 1 channel residual at 128x128
    """
    def __init__(self, in_channels=2, hidden=64, n_blocks=8):
        super().__init__()
        self.head = nn.Conv2d(in_channels, hidden, 3, padding=1)

        blocks = []
        for _ in range(n_blocks):
            blocks.append(ResBlock(hidden))
        self.body = nn.Sequential(*blocks)

        self.tail = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )

    def forward(self, swinir_out, lr_bicubic):
        x = torch.cat([swinir_out, lr_bicubic], dim=1)
        h = self.head(x)
        h = self.body(h) + h  # global residual
        residual = self.tail(h)
        return swinir_out + residual  # refine SwinIR output


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return h + x


# ---------- Direct UNet Regression ----------

class SimpleUNetRegression(nn.Module):
    """Direct LR->HR regression with UNet architecture.

    Input: 1 channel LR at 32x32
    Output: 1 channel HR at 128x128
    """
    def __init__(self, base_ch=64, ch_mults=(1, 2, 4)):
        super().__init__()
        # Encoder at 32x32
        dims = [base_ch * m for m in ch_mults]

        self.init_conv = nn.Conv2d(1, dims[0], 3, padding=1)

        # Encoder: 32->16->8
        self.enc1 = self._make_block(dims[0], dims[0])
        self.down1 = nn.Conv2d(dims[0], dims[1], 4, 2, 1)
        self.enc2 = self._make_block(dims[1], dims[1])
        self.down2 = nn.Conv2d(dims[1], dims[2], 4, 2, 1)

        # Bottleneck at 8x8
        self.mid = self._make_block(dims[2], dims[2])

        # Decoder: 8->16->32->64->128
        self.up1 = nn.ConvTranspose2d(dims[2], dims[1], 4, 2, 1)
        self.dec1 = self._make_block(dims[1]*2, dims[1])
        self.up2 = nn.ConvTranspose2d(dims[1], dims[0], 4, 2, 1)
        self.dec2 = self._make_block(dims[0]*2, dims[0])

        # Upsample 32->128 (4x)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(dims[0], dims[0], 4, 2, 1),  # 32->64
            nn.SiLU(),
            nn.ConvTranspose2d(dims[0], base_ch//2, 4, 2, 1),  # 64->128
            nn.SiLU(),
        )

        self.out = nn.Conv2d(base_ch//2, 1, 3, padding=1)

    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.SiLU(),
        )

    def forward(self, x):
        h = self.init_conv(x)  # (B, 64, 32, 32)

        e1 = self.enc1(h)  # (B, 64, 32, 32)
        e2 = self.enc2(self.down1(e1))  # (B, 128, 16, 16)

        mid = self.mid(self.down2(e2))  # (B, 256, 8, 8)

        d1 = self.dec1(torch.cat([self.up1(mid), e2], dim=1))  # (B, 128, 16, 16)
        d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1))  # (B, 64, 32, 32)

        up = self.upsample(d2)  # (B, 32, 128, 128)
        return self.out(up)  # (B, 1, 128, 128)


# ---------- Precompute SwinIR Features ----------

@torch.no_grad()
def precompute_swinir(args):
    """Run SwinIR zero-shot on all splits and save outputs."""
    from eval_zeroshot import load_swinir, adapt_input_for_swinir, adapt_output_from_swinir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_swinir(args.weights, device=device)

    # Get normalization range from training data
    train_hr = torch.load(f'{args.data_dir}/train/target_train.pt', weights_only=False)
    min_val = train_hr[:, 0, :, :, :].min().item()
    max_val = train_hr[:, 0, :, :, :].max().item()
    del train_hr

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'min_val': min_val, 'max_val': max_val}, os.path.join(save_dir, 'norm_stats.pt'))

    for split in ['train', 'val', 'test']:
        print(f"Processing {split}...")
        lr, hr = load_data(args.data_dir, split)
        N = lr.shape[0]

        outputs = []
        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)
            batch = lr[start:end].to(device)

            # Adapt for SwinIR (1ch -> 3ch, normalize)
            batch_3ch = adapt_input_for_swinir(batch, min_val, max_val)
            # Pad to multiple of 8
            _, _, h, w = batch_3ch.shape
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            if pad_h > 0 or pad_w > 0:
                batch_3ch = F.pad(batch_3ch, (0, pad_w, 0, pad_h), mode='reflect')

            out = model(batch_3ch)
            if pad_h > 0 or pad_w > 0:
                out = out[:, :, :h*4, :w*4]

            # Convert back to 1ch physical values
            out_1ch = adapt_output_from_swinir(out, min_val, max_val)
            outputs.append(out_1ch.cpu())

            if (start // args.batch_size) % 50 == 0:
                print(f"  {start}/{N}")

        outputs = torch.cat(outputs, dim=0)
        torch.save(outputs, os.path.join(save_dir, f'swinir_output_{split}.pt'))
        print(f"  Saved: {outputs.shape}")


# ---------- Training ----------

def train_refine(args):
    """Train refinement CNN on precomputed SwinIR outputs."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading data...")
    lr_train, hr_train = load_data(args.data_dir, 'train')
    lr_val, hr_val = load_data(args.data_dir, 'val')

    # Normalization
    min_val = hr_train.min().item()
    max_val = hr_train.max().item()

    # Normalize to [0, 1]
    hr_train_n = (hr_train - min_val) / (max_val - min_val)
    hr_val_n = (hr_val - min_val) / (max_val - min_val)
    lr_train_n = (lr_train - min_val) / (max_val - min_val)
    lr_val_n = (lr_val - min_val) / (max_val - min_val)

    # Bicubic upsampled LR as input (or SwinIR precomputed)
    lr_up_train = F.interpolate(lr_train_n, size=(128, 128), mode='bicubic', align_corners=False)
    lr_up_val = F.interpolate(lr_val_n, size=(128, 128), mode='bicubic', align_corners=False)

    if args.model == 'unet':
        # Direct UNet regression: LR -> HR
        print("Training SimpleUNet regression...")
        model = SimpleUNetRegression(base_ch=args.base_ch).to(device)
        train_ds = TensorDataset(lr_train_n, hr_train_n, lr_train_n)
        val_ds = TensorDataset(lr_val_n, hr_val_n, lr_val_n)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.L1Loss()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'min_val': min_val, 'max_val': max_val}, os.path.join(save_dir, 'norm_stats.pt'))

    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            lr_batch = batch[0].to(device)
            hr_batch = batch[1].to(device)

            if args.model == 'unet':
                pred = model(lr_batch)
            else:
                swinir_batch = batch[2].to(device)
                lr_up_batch = F.interpolate(lr_batch, size=(128, 128), mode='bicubic', align_corners=False)
                pred = model(swinir_batch, lr_up_batch)

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
            for batch in val_loader:
                lr_batch = batch[0].to(device)
                hr_batch = batch[1].to(device)

                if args.model == 'unet':
                    pred = model(lr_batch)
                else:
                    swinir_batch = batch[2].to(device)
                    lr_up_batch = F.interpolate(lr_batch, size=(128, 128), mode='bicubic', align_corners=False)
                    pred = model(swinir_batch, lr_up_batch)

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
                'model_type': args.model,
                'base_ch': args.base_ch,
            }, os.path.join(save_dir, 'best_model.pt'))
            print(f"  -> Saved best (val={val_loss:.6f})")

    total_time = time.time() - start_time
    print(f"\nDone in {total_time/60:.1f} min. Best val: {best_val_loss:.6f}")


# ---------- Evaluation ----------

@torch.no_grad()
def eval_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    ckpt = torch.load(os.path.join(args.save_dir, 'best_model.pt'),
                      map_location='cpu', weights_only=False)
    min_val = ckpt['min_val']
    max_val = ckpt['max_val']

    # Build model
    model_type = ckpt.get('model_type', args.model)
    if model_type == 'unet':
        model = SimpleUNetRegression(base_ch=ckpt.get('base_ch', args.base_ch))
    model.load_state_dict(ckpt['model'])
    model = model.to(device).eval()

    # Load data
    lr, hr = load_data(args.data_dir, args.split)
    N = lr.shape[0]
    if args.max_samples:
        N = min(N, args.max_samples)
        lr = lr[:N]
        hr = hr[:N]

    lr_n = (lr - min_val) / (max_val - min_val)

    # TTA augmentations
    augments = list(range(8)) if args.tta else [0]
    all_preds = []

    for aug_idx in augments:
        preds = []
        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)
            batch = lr_n[start:end].to(device)

            # Augment
            if aug_idx > 0:
                if aug_idx & 1: batch = batch.flip(-1)
                if aug_idx & 2: batch = batch.flip(-2)
                if aug_idx & 4: batch = batch.rot90(1, [-2, -1])

            pred = model(batch)

            # De-augment
            if aug_idx > 0:
                if aug_idx & 4: pred = pred.rot90(-1, [-2, -1])
                if aug_idx & 2: pred = pred.flip(-2)
                if aug_idx & 1: pred = pred.flip(-1)

            # Denormalize
            pred_phys = pred.cpu() * (max_val - min_val) + min_val
            preds.append(pred_phys)

        pred_full = torch.cat(preds, dim=0).numpy()
        all_preds.append(pred_full)

    # Evaluate
    M = len(all_preds)
    ensemble = np.stack([p[:N, 0] for p in all_preds], axis=0)
    targets = hr[:N, 0].numpy()
    lr_np = lr[:N, 0].numpy()

    crps_values = []
    for i in range(N):
        c = crps_ensemble_correct(targets[i], ensemble[:, i])
        crps_values.append(c)
    crps = float(np.mean(crps_values))

    mean_pred = ensemble.mean(axis=0)
    rmse = float(np.sqrt(np.mean((mean_pred - targets) ** 2)))
    mae = float(np.mean(np.abs(mean_pred - targets)))

    ds_pred = mean_pred.reshape(N, 32, 4, 32, 4).mean(axis=(2, 4))
    mass_viol = float(np.mean(np.abs(ds_pred - lr_np)))
    spread = float(np.mean(np.std(ensemble, axis=0))) if M > 1 else 0.0

    name = f"{model_type} {'TTA8' if args.tta else ''}"
    print(f"\n{'='*60}")
    print(f"  {name} (M={M}, N={N})")
    print(f"{'='*60}")
    print(f"  CRPS:           {crps:.6f}")
    print(f"  RMSE:           {rmse:.6f}")
    print(f"  MAE:            {mae:.6f}")
    print(f"  Mass violation: {mass_viol:.6f}")
    print(f"  Spread:         {spread:.6f}")

    # With AddCL
    preds_c = []
    for p in all_preds:
        p_t = torch.from_numpy(p)
        p_c = apply_addcl(p_t, lr, factor=4)
        preds_c.append(p_c.numpy())

    ensemble_c = np.stack([p[:N, 0] for p in preds_c], axis=0)
    crps_c_values = []
    for i in range(N):
        c = crps_ensemble_correct(targets[i], ensemble_c[:, i])
        crps_c_values.append(c)
    crps_c = float(np.mean(crps_c_values))

    mean_pred_c = ensemble_c.mean(axis=0)
    rmse_c = float(np.sqrt(np.mean((mean_pred_c - targets) ** 2)))
    mae_c = float(np.mean(np.abs(mean_pred_c - targets)))
    ds_pred_c = mean_pred_c.reshape(N, 32, 4, 32, 4).mean(axis=(2, 4))
    mass_viol_c = float(np.mean(np.abs(ds_pred_c - lr_np)))
    spread_c = float(np.mean(np.std(ensemble_c, axis=0))) if M > 1 else 0.0

    print(f"\n  {name} + AddCL (M={M}, N={N})")
    print(f"  CRPS:           {crps_c:.6f}")
    print(f"  RMSE:           {rmse_c:.6f}")
    print(f"  MAE:            {mae_c:.6f}")
    print(f"  Mass violation: {mass_viol_c:.6f}")
    print(f"  Spread:         {spread_c:.6f}")
    print(f"{'='*60}\n")


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', choices=['precompute', 'train', 'eval', 'both'])
    parser.add_argument('--model', default='unet', choices=['unet', 'refine'])
    parser.add_argument('--data-dir', default='/home/chenxy/orcd/pool/datasets/era5_sr_data')
    parser.add_argument('--weights', default='/home/chenxy/orcd/pool/datasets/research6/weights/swinir_classical_x4.pth')
    parser.add_argument('--save-dir', default='/home/chenxy/orcd/pool/datasets/research6/models/unet_regression')
    parser.add_argument('--split', default='test')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--base-ch', type=int, default=64)
    parser.add_argument('--tta', action='store_true')
    args = parser.parse_args()

    if args.mode == 'precompute':
        precompute_swinir(args)
    elif args.mode in ('train', 'both'):
        train_refine(args)
        if args.mode == 'both':
            args.tta = True
            eval_model(args)
    elif args.mode == 'eval':
        eval_model(args)


if __name__ == '__main__':
    main()
