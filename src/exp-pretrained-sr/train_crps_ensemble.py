"""Multi-head SwinIR with direct CRPS energy score loss.

Architecture:
  - Frozen SwinIR backbone (through conv_after_body + skip connection)
  - K parallel output branches, each: Conv(180→64)+ReLU → PixelShuffle(4x) → Conv(64→1)
  - Each branch produces one ensemble member at 128x128
  - Initialized from finetuned SwinIR checkpoint (iteration 1)

Loss:
  Energy score (proper scoring rule):
    L = (1/K) Σ_k |y_k - y| - (1/(2K²)) ΣΣ_{k,k'} |y_k - y_k'|

This directly optimizes the CRPS metric used for evaluation.

Usage:
    python src/exp-pretrained-sr/train_crps_ensemble.py --K 8 --epochs 100 --lr 5e-4
    python src/exp-pretrained-sr/train_crps_ensemble.py --mode eval --checkpoint best
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
SAVE_DIR = POOL / "research5" / "models" / "crps_ensemble"


class MultiHeadSwinIR(nn.Module):
    """SwinIR backbone with K parallel output heads for ensemble prediction.

    When residual=True, keeps the finetuned deterministic tail (frozen) and each
    head predicts a residual correction. Output_k = det_mean + residual_k.
    This separates mean accuracy from stochastic diversity (CorrDiff principle).
    """

    def __init__(self, backbone, K=8, residual=False, unfreeze_layers=0):
        super().__init__()
        self.K = K
        self.residual = residual
        self.unfreeze_layers = unfreeze_layers

        # Shared backbone: conv_first through conv_after_body + skip
        self.conv_first = backbone.conv_first
        self.patch_embed = backbone.patch_embed
        self.patch_unembed = backbone.patch_unembed
        self.pos_drop = backbone.pos_drop
        self.layers = backbone.layers
        self.norm = backbone.norm
        self.conv_after_body = backbone.conv_after_body
        self.register_buffer('mean', backbone.mean.clone())
        self.img_range = backbone.img_range

        # Copy forward_features method's needed attributes
        self.num_layers = backbone.num_layers
        self.patches_resolution = backbone.patches_resolution
        self.num_features = backbone.num_features

        # Freeze all backbone parameters first
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

        # Unfreeze last N Swin Transformer layers + norm + conv_after_body
        if unfreeze_layers > 0:
            n_layers = len(self.layers)
            for i in range(n_layers - unfreeze_layers, n_layers):
                for param in self.layers[i].parameters():
                    param.requires_grad = True
            for param in self.norm.parameters():
                param.requires_grad = True
            for param in self.conv_after_body.parameters():
                param.requires_grad = True

        if residual:
            # Frozen deterministic tail from finetuned backbone
            self.det_conv_before_upsample = copy.deepcopy(backbone.conv_before_upsample)
            self.det_upsample = copy.deepcopy(backbone.upsample)
            self.det_conv_last = copy.deepcopy(backbone.conv_last)
            for mod in [self.det_conv_before_upsample, self.det_upsample, self.det_conv_last]:
                for param in mod.parameters():
                    param.requires_grad = False

        # K parallel output branches (trainable)
        # Each branch: conv_before_upsample + upsample + conv_last
        self.heads = nn.ModuleList()
        for k in range(K):
            head = nn.Sequential(
                # conv_before_upsample equivalent
                nn.Conv2d(180, 64, 3, padding=1),
                nn.LeakyReLU(0.01, inplace=True),
                # upsample: 2x PixelShuffle stages (32→64→128)
                nn.Conv2d(64, 256, 3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(64, 256, 3, padding=1),
                nn.PixelShuffle(2),
                # conv_last: 64→1
                nn.Conv2d(64, 1, 3, padding=1),
            )
            self.heads.append(head)

    def init_heads_from_finetuned(self, backbone):
        """Initialize each head from the finetuned backbone's tail weights + noise."""
        # Map backbone tail layers to our head sequential indices
        # head[0] = conv_before_upsample[0] (Conv2d 180→64)
        # head[2] = upsample[0] (Conv2d 64→256)
        # head[4] = upsample[2] (Conv2d 64→256)
        # head[6] = conv_last (Conv2d 64→1)
        src_layers = [
            ('conv_before_upsample.0', 0),
            ('upsample.0', 2),
            ('upsample.2', 4),
            ('conv_last', 6),
        ]

        for k, head in enumerate(self.heads):
            for src_name, dst_idx in src_layers:
                src_mod = dict(backbone.named_modules())[src_name]
                dst_mod = head[dst_idx]
                with torch.no_grad():
                    dst_mod.weight.copy_(src_mod.weight)
                    if src_mod.bias is not None and dst_mod.bias is not None:
                        dst_mod.bias.copy_(src_mod.bias)
                    # Add small perturbation for diversity (skip head 0 to keep one "mean" head)
                    if k > 0:
                        noise_scale = 0.02
                        w_std = dst_mod.weight.std() if dst_mod.weight.numel() > 1 else dst_mod.weight.abs().mean()
                        dst_mod.weight.add_(torch.randn_like(dst_mod.weight) * noise_scale * w_std)
                        if dst_mod.bias is not None:
                            b_std = dst_mod.bias.std() if dst_mod.bias.numel() > 1 else dst_mod.bias.abs().mean()
                            dst_mod.bias.add_(torch.randn_like(dst_mod.bias) * noise_scale * b_std)

    def init_heads_residual(self):
        """Initialize residual heads with small random weights for near-zero initial output.

        Used in residual mode: heads predict small corrections to the frozen det. mean.
        Small init → initial ensemble ≈ det_mean (all members identical) → CRPS starts at MAE.
        """
        for head in self.heads:
            for module in head.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.xavier_uniform_(module.weight, gain=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward_features(self, x):
        """Forward through Swin Transformer body (mirrors SwinIR's method)."""
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x):
        """Forward pass returning K ensemble members.

        Args:
            x: (B, 1, 32, 32) normalized input

        Returns:
            (B, K, 1, 128, 128) ensemble predictions
        """
        # Normalize by img_range and mean
        x = (x - self.mean) * self.img_range

        # Shared backbone
        feat = self.conv_first(x)
        body_out = self.conv_after_body(self.forward_features(feat))
        shared = body_out + feat  # (B, 180, 32, 32)

        if self.residual:
            # Deterministic mean from frozen tail (no grad needed)
            with torch.no_grad():
                det_out = self.det_conv_before_upsample(shared)
                det_out = self.det_upsample(det_out)
                det_mean = self.det_conv_last(det_out)  # (B, 1, 128, 128)
            # Each head predicts a residual; output = det_mean + residual
            outputs = []
            for head in self.heads:
                residual = head(shared)  # (B, 1, 128, 128)
                outputs.append(det_mean + residual)
        else:
            # Direct prediction from each head
            outputs = []
            for head in self.heads:
                out = head(shared)  # (B, 1, 128, 128)
                outputs.append(out)

        # Stack: (B, K, 1, 128, 128)
        ensemble = torch.stack(outputs, dim=1)

        # Undo normalization
        ensemble = ensemble / self.img_range + self.mean

        return ensemble


def energy_score_loss(ensemble, target):
    """Energy score loss (CRPS for multivariate).

    ensemble: (B, K, 1, H, W)
    target: (B, 1, H, W)

    Loss = (1/K) Σ_k |y_k - y| - (1/(2K²)) ΣΣ |y_k - y_k'|
    """
    B, K = ensemble.shape[:2]

    # Term 1: (1/K) Σ_k E|y_k - y|
    # ensemble: (B, K, 1, H, W), target: (B, 1, 1, H, W) for broadcasting
    target_exp = target.unsqueeze(1)  # (B, 1, 1, H, W)
    term1 = torch.mean(torch.abs(ensemble - target_exp))

    # Term 2: (1/(2K²)) ΣΣ E|y_k - y_k'|
    # Use broadcasting: (B, K, 1, 1, H, W) - (B, 1, K, 1, H, W)
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


def apply_addcl(predictions, lr):
    """Apply additive constraint: shift so block-means match LR.

    predictions: (N, 128, 128)
    lr: (N, 32, 32)
    """
    N = predictions.shape[0]
    result = predictions.copy()
    for i in range(N):
        pred_ds = predictions[i].reshape(32, 4, 32, 4).mean(axis=(1, 3))
        correction = lr[i] - pred_ds
        correction_hr = np.repeat(np.repeat(correction, 4, axis=0), 4, axis=1)
        result[i] = predictions[i] + correction_hr
    return result


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

    model = MultiHeadSwinIR(backbone, K=args.K, residual=args.residual,
                            unfreeze_layers=args.unfreeze_layers)
    if args.residual:
        model.init_heads_residual()
        print(f"  Mode: RESIDUAL — heads predict corrections to frozen det. mean")
    else:
        model.init_heads_from_finetuned(backbone)
        print(f"  Mode: DIRECT — heads predict full output")
    model = model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_backbone_trainable = sum(p.numel() for n, p in model.named_parameters()
                               if p.requires_grad and 'heads' not in n)
    print(f"  Parameters: {n_total:,} total, {n_trainable:,} trainable")
    if args.unfreeze_layers > 0:
        print(f"  Unfrozen backbone params: {n_backbone_trainable:,} (last {args.unfreeze_layers} layers + norm + conv_after_body)")

    # Optimizer with discriminative LR
    if args.unfreeze_layers > 0:
        backbone_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and 'heads' not in n]
        head_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and 'heads' in n]
        backbone_lr = args.lr * args.backbone_lr_mult
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params, 'lr': args.lr},
        ], weight_decay=args.weight_decay)
        print(f"  Discriminative LR: backbone={backbone_lr:.2e}, heads={args.lr:.2e}")
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=args.weight_decay
        )
    # Cosine schedule based on expected epochs within wall time
    # Estimate: ~6.5 min/epoch with BS=32, 40K samples → ~18 epochs in 2hr
    expected_epochs = min(args.epochs, int(args.wall_hours * 60 / 7) + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=expected_epochs)

    # Save dir — separate dir per experiment variant
    if args.unfreeze_layers > 0:
        save_dir = POOL / "research5" / "models" / f"crps_unfreeze{args.unfreeze_layers}"
    elif args.residual:
        save_dir = POOL / "research5" / "models" / "crps_residual"
    else:
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
    train_terms = []  # (term1, term2) per epoch
    start_time = time.time()
    wall_limit = args.wall_hours * 3600

    mode_str = "RESIDUAL" if args.residual else "DIRECT"
    if args.unfreeze_layers > 0:
        mode_str += f"+UNFREEZE{args.unfreeze_layers}"
    print(f"\nTraining for up to {args.epochs} epochs ({args.wall_hours}h wall limit)...")
    print(f"  Mode={mode_str}, K={args.K} heads, LR={args.lr}, BS={args.batch_size}")
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

            ensemble = model(lr_batch)  # (B, K, 1, H, W)
            loss, t1, t2 = energy_score_loss(ensemble, hr_batch)

            optimizer.zero_grad()
            loss.backward()
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
                ensemble = model(lr_batch)
                loss, _, _ = energy_score_loss(ensemble, hr_batch)
                val_loss += loss.item()
                n_val += 1
        val_loss /= n_val
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_data = {
                'heads': {f'head_{k}': head.state_dict() for k, head in enumerate(model.heads)},
                'epoch': epoch,
                'val_loss': val_loss,
                'optimizer': optimizer.state_dict(),
                'vmin': vmin,
                'vmax': vmax,
                'K': args.K,
                'residual': args.residual,
                'unfreeze_layers': args.unfreeze_layers,
            }
            # Save unfrozen backbone state if applicable
            if args.unfreeze_layers > 0:
                ckpt_data['backbone_state'] = {
                    n: p.data.clone() for n, p in model.named_parameters()
                    if p.requires_grad and 'heads' not in n
                }
            torch.save(ckpt_data, save_dir / 'best_ensemble.pt')

        elapsed = time.time() - start_time
        lr_now = scheduler.get_last_lr()[0]
        t1_avg, t2_avg = train_terms[-1]
        print(f"  Ep {epoch+1:3d} | loss: {train_loss:.6f} (t1={t1_avg:.4f} t2={t2_avg:.4f}) | "
              f"val: {val_loss:.6f} | best: {best_val_loss:.6f} | lr: {lr_now:.2e} | {elapsed/60:.1f}min")

    # Save final
    final_data = {
        'heads': {f'head_{k}': head.state_dict() for k, head in enumerate(model.heads)},
        'epoch': epoch,
        'val_loss': val_loss,
        'vmin': vmin,
        'vmax': vmax,
        'K': args.K,
        'residual': args.residual,
        'unfreeze_layers': args.unfreeze_layers,
    }
    if args.unfreeze_layers > 0:
        final_data['backbone_state'] = {
            n: p.data.clone() for n, p in model.named_parameters()
            if p.requires_grad and 'heads' not in n
        }
    torch.save(final_data, save_dir / 'final_ensemble.pt')
    torch.save({
        'train': train_losses, 'val': val_losses, 'terms': train_terms
    }, save_dir / 'losses.pt')

    elapsed = time.time() - start_time
    print(f"\nTraining complete. {elapsed/60:.1f} minutes, {epoch+1} epochs.")
    print(f"Best val energy score: {best_val_loss:.6f}")


def evaluate(args):
    """Evaluate ensemble CRPS on test set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    residual = getattr(args, 'residual', False)
    # Determine save dir based on flags or explicit --eval_dir
    if hasattr(args, 'eval_dir') and args.eval_dir:
        save_dir = Path(args.eval_dir)
    elif residual:
        save_dir = POOL / "research5" / "models" / "crps_residual"
    else:
        save_dir = SAVE_DIR
    ckpt = torch.load(save_dir / f'{args.checkpoint}_ensemble.pt', map_location='cpu', weights_only=False)
    vmin = ckpt['vmin']
    vmax = ckpt['vmax']
    K = ckpt['K']
    residual = ckpt.get('residual', False)
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.6f}, K={K}, residual={residual}")

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

    unfreeze_layers = ckpt.get('unfreeze_layers', 0)
    model = MultiHeadSwinIR(backbone, K=K, residual=residual, unfreeze_layers=unfreeze_layers)
    # Load trained heads
    for k, head in enumerate(model.heads):
        head.load_state_dict(ckpt['heads'][f'head_{k}'])
    # Load unfrozen backbone state if applicable
    if 'backbone_state' in ckpt:
        model_dict = dict(model.named_parameters())
        for name, tensor in ckpt['backbone_state'].items():
            if name in model_dict:
                model_dict[name].data.copy_(tensor)
        print(f"  Loaded unfrozen backbone state ({len(ckpt['backbone_state'])} params)")
    model = model.to(device).eval()

    # Load test data
    print(f"Loading {args.split} data...")
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
            ens = model(batch)  # (B, K, 1, H, W)
            ens_phys = denormalize(ens, vmin, vmax)
            all_ensemble.append(ens_phys.cpu())
            if start % (batch_size * 10) == 0:
                print(f"  {end}/{N}")

    # (N, K, 1, 128, 128)
    all_ensemble = torch.cat(all_ensemble, dim=0).numpy()
    all_ensemble = all_ensemble[:, :, 0, :, :]  # (N, K, 128, 128)

    hr_np = hr[:N, 0].numpy()  # (N, 128, 128)
    lr_np = lr[:N, 0].numpy()  # (N, 32, 32)

    # Compute CRPS per sample
    crps_values = []
    for i in range(N):
        samples = all_ensemble[i]  # (K, 128, 128)
        obs = hr_np[i]  # (128, 128)
        M = samples.shape[0]
        t1 = np.mean(np.abs(samples - obs[None, ...]))
        diff = np.abs(samples[:, None, ...] - samples[None, :, ...])
        t2 = np.mean(diff)
        crps = t1 - 0.5 * t2
        crps_values.append(crps)

    crps = np.mean(crps_values)

    # Ensemble mean metrics
    ens_mean = all_ensemble.mean(axis=1)  # (N, 128, 128)
    mae = np.mean(np.abs(hr_np - ens_mean))
    rmse = np.sqrt(np.mean((hr_np - ens_mean) ** 2))
    spread = np.mean(np.std(all_ensemble, axis=1))

    # Mass violation (ensemble mean)
    ens_mean_ds = ens_mean.reshape(N, 32, 4, 32, 4).mean(axis=(2, 4))
    mass_viol = np.mean(np.abs(ens_mean_ds - lr_np))

    # With AddCL on each member
    crps_addcl = []
    for i in range(N):
        members_c = apply_addcl_batch(all_ensemble[i], np.tile(lr_np[i:i+1], (K, 1, 1)))
        obs = hr_np[i]
        t1 = np.mean(np.abs(members_c - obs[None, ...]))
        diff = np.abs(members_c[:, None, ...] - members_c[None, :, ...])
        t2 = np.mean(diff)
        crps_addcl.append(t1 - 0.5 * t2)
    crps_addcl = np.mean(crps_addcl)

    # AddCL ensemble mean
    ens_mean_c = apply_addcl_batch(ens_mean, lr_np)
    mae_c = np.mean(np.abs(hr_np - ens_mean_c))
    ens_mean_c_ds = ens_mean_c.reshape(N, 32, 4, 32, 4).mean(axis=(2, 4))
    mass_viol_c = np.mean(np.abs(ens_mean_c_ds - lr_np))

    print(f"\n{'='*60}")
    print(f"Results: Multi-Head SwinIR CRPS Ensemble ({args.split}, N={N}, K={K})")
    print(f"{'='*60}")
    print(f"  CRPS (energy):     {crps:.6f}")
    print(f"  MAE (ens. mean):   {mae:.6f}")
    print(f"  RMSE (ens. mean):  {rmse:.6f}")
    print(f"  Spread (std):      {spread:.6f}")
    print(f"  Mass violation:    {mass_viol:.6f}")
    print(f"\n  With AddCL:")
    print(f"  CRPS (energy):     {crps_addcl:.6f}")
    print(f"  MAE (ens. mean):   {mae_c:.6f}")
    print(f"  Mass violation:    {mass_viol_c:.6f}")
    print(f"\n  Reference:")
    print(f"  OT-CFM (research2) CRPS: 0.171")
    print(f"  SwinIR-FT det. CRPS: 0.250")
    print(f"{'='*60}")

    # Save results
    results = {
        'CRPS': crps, 'CRPS_addcl': crps_addcl,
        'MAE': mae, 'MAE_addcl': mae_c,
        'RMSE': rmse, 'Spread': spread,
        'Mass_viol': mass_viol, 'Mass_viol_addcl': mass_viol_c,
        'K': K, 'N': N,
    }
    torch.save(results, save_dir / 'eval_results.pt')
    print(f"Saved results to {save_dir / 'eval_results.pt'}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", choices=["train", "eval"])
    parser.add_argument("--K", type=int, default=8, help="Number of ensemble heads")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--wall_hours", type=float, default=2.0)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--residual", action="store_true",
                        help="Residual mode: heads predict corrections to frozen det. mean")
    parser.add_argument("--unfreeze_layers", type=int, default=0,
                        help="Number of Swin Transformer layers to unfreeze (from the end)")
    parser.add_argument("--backbone_lr_mult", type=float, default=0.1,
                        help="Learning rate multiplier for unfrozen backbone layers")
    parser.add_argument("--eval_dir", type=str, default=None,
                        help="Explicit model directory for evaluation")
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
