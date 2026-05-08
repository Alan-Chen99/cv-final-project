"""Quick CPU evaluation of base64 model on small subset.
Same 100 samples as wide96 for fair comparison."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'exp-spatial-4x-crps-v1'))
from flow_matching_v2 import (
    AttentionUNet, load_tcw4_data,
    midpoint_sample, apply_addcl, crps_ensemble_correct,
)
import numpy as np
import torch

MODEL_DIR = "/home/chenxy/orcd/pool/datasets/research3/models/unet_uniform_amp"
BASEDIR = "external/constrained-downscaling"
MAX_SAMPLES = 100
N_ENSEMBLE = 10
STEPS = 5

device = torch.device('cpu')

stats = torch.load(os.path.join(MODEL_DIR, 'norm_stats.pt'), weights_only=False, map_location='cpu')
ckpt = torch.load(os.path.join(MODEL_DIR, 'best_flow.pt'), weights_only=False, map_location='cpu')
saved_args = ckpt['args']
saved_mults = saved_args.get('channel_mults_tuple', (1, 2, 4))
if isinstance(saved_mults, str):
    saved_mults = tuple(int(x) for x in saved_mults.split(','))

model = AttentionUNet(
    in_channels=2, out_channels=1,
    base_channels=saved_args.get('base_channels', 64),
    channel_mults=saved_mults,
    time_emb_dim=256, dropout=0.0,
    attn_heads=saved_args.get('attn_heads', 4),
).to(device)
model.load_state_dict(ckpt['model'])
model.eval()
print(f"Model: base_channels={saved_args.get('base_channels')}, epoch={ckpt['epoch']+1}, val_loss={ckpt['val_loss']:.6f}")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

lr_up, residual, hr, lr_orig = load_tcw4_data(BASEDIR, 'test')
lr_up_norm = (lr_up - stats['lr_mean']) / stats['lr_std']
n_samples = min(MAX_SAMPLES, lr_up.shape[0])
print(f"Evaluating {n_samples} samples, {N_ENSEMBLE} ensemble, midpoint {STEPS} steps")

pool = torch.nn.AvgPool2d(kernel_size=4)
all_crps, all_mae, all_rmse, all_mass = [], [], [], []
t0 = time.time()

for i in range(n_samples):
    batch_lr = lr_up_norm[i:i+1].to(device)
    ensemble_preds = []
    for e in range(N_ENSEMBLE):
        with torch.no_grad():
            sampled_res_norm = midpoint_sample(model, batch_lr, shape=(1, 1, 128, 128), steps=STEPS)
            sampled_res = sampled_res_norm.cpu() * stats['res_std'] + stats['res_mean']
            pred_hr = lr_up[i:i+1] + sampled_res
            pred_hr = apply_addcl(pred_hr, lr_orig[i:i+1])
            ensemble_preds.append(pred_hr.numpy())

    ensemble_preds = np.stack(ensemble_preds, axis=1)
    gt = hr[i, 0, ...].numpy()
    ens = ensemble_preds[0, :, 0, ...]
    ens_mean = ens.mean(axis=0)

    all_crps.append(crps_ensemble_correct(gt, ens))
    all_mae.append(np.mean(np.abs(gt - ens_mean)))
    all_rmse.append(np.mean((gt - ens_mean) ** 2))
    pred_mean_t = torch.from_numpy(ens_mean).unsqueeze(0).unsqueeze(0)
    pooled = pool(pred_mean_t).squeeze()
    all_mass.append(torch.mean(torch.abs(pooled - lr_orig[i, 0, ...])).item())

    if (i + 1) % 10 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (n_samples - i - 1)
        print(f"  [{i+1}/{n_samples}] CRPS={np.mean(all_crps):.6f} elapsed={elapsed:.0f}s ETA={eta:.0f}s")

elapsed = time.time() - t0
print(f"\n=== Results ({n_samples} samples, midpoint {STEPS}, AddCL) ===")
print(f"CRPS: {np.mean(all_crps):.6f}")
print(f"RMSE: {np.sqrt(np.mean(all_rmse)):.6f}")
print(f"MAE:  {np.mean(all_mae):.6f}")
print(f"Mass: {np.mean(all_mass):.6f}")
print(f"Time: {elapsed:.0f}s")
