"""Inspect Anvita vs ours checkpoint contents — state_dict keys, shapes, value stats."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "external" / "constrained-downscaling"))

ANVITA = Path("/orcd/pool/007/chenxy/datasets/anvita")
POOL = Path("/home/chenxy/orcd/pool/datasets")

PAIRS = [
    ("noresm CNN none", POOL / "noresm-dataset/models/harder/twc_cnn_none.pth", ANVITA / "noresm_cnn_none.pth"),
    ("noresm CNN smcl", POOL / "noresm-dataset/models/harder/twc_cnn_softmax.pth", ANVITA / "noresm_cnn_softmax.pth"),
    ("noresm GAN smcl", POOL / "noresm-dataset/models/harder/twc_gan_softmax.pth", ANVITA / "noresm_gan_softmax.pth"),
    ("era5 CNN none", POOL / "organize2/models/harder/twc_cnn_none.pth", ANVITA / "twc_cnn_noconstraints.pth"),
    ("era5 CNN smcl", POOL / "organize2/models/harder/twc_cnn_softmax.pth", ANVITA / "twc_cnn_softmax.pth"),
    ("era5 GAN none", POOL / "organize2/models/harder/twc_gan_none.pth", ANVITA / "twc_gan_noconstraints.pth"),
    ("era5 GAN smcl", POOL / "organize2/models/harder/twc_gan_softmax.pth", ANVITA / "twc_gan_softmax.pth"),
]


def inspect(label: str, path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    info = {"path": str(path), "top_keys": sorted(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt).__name__}
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        info["n_params"] = sum(p.numel() for p in sd.values())
        info["sd_keys"] = sorted(sd.keys())
        # Check a few weight stats
        for k in ["conv1.0.weight", "conv4.weight", "upsampling.0.weight"]:
            if k in sd:
                t = sd[k]
                info[f"{k}_shape"] = list(t.shape)
                info[f"{k}_mean"] = f"{t.float().mean():.6f}"
                info[f"{k}_std"] = f"{t.float().std():.6f}"
        # Extra metadata
        for mk in ["epoch", "best_loss", "optimizer", "training_time"]:
            if mk in ckpt:
                v = ckpt[mk]
                if mk == "optimizer":
                    info[mk] = "present"
                else:
                    info[mk] = v
    return info


for pair_label, ours_path, anvita_path in PAIRS:
    print(f"\n{'='*80}")
    print(f"  {pair_label}")
    print(f"{'='*80}")
    for source, path in [("OURS", ours_path), ("ANVITA", anvita_path)]:
        if not path.exists():
            print(f"  {source}: NOT FOUND ({path})")
            continue
        info = inspect(f"{source} {pair_label}", path)
        print(f"\n  {source}: {path.name} ({path.stat().st_size} bytes)")
        print(f"    top_keys: {info.get('top_keys')}")
        print(f"    n_params: {info.get('n_params')}")
        n_sd_keys = len(info.get("sd_keys", []))
        print(f"    state_dict keys ({n_sd_keys}): {info.get('sd_keys', [])[:5]}...")
        for k in ["conv1.0.weight", "conv4.weight", "upsampling.0.weight"]:
            if f"{k}_shape" in info:
                print(f"    {k}: shape={info[f'{k}_shape']} mean={info[f'{k}_mean']} std={info[f'{k}_std']}")
        for mk in ["epoch", "best_loss"]:
            if mk in info:
                print(f"    {mk}: {info[mk]}")

    # Key comparison
    if ours_path.exists() and anvita_path.exists():
        ours_info = inspect("ours", ours_path)
        anv_info = inspect("anvita", anvita_path)
        ours_keys = set(ours_info.get("sd_keys", []))
        anv_keys = set(anv_info.get("sd_keys", []))
        if ours_keys == anv_keys:
            print(f"\n    KEY MATCH: identical state_dict keys ({len(ours_keys)})")
        else:
            print(f"\n    KEY MISMATCH!")
            print(f"      only in ours: {ours_keys - anv_keys}")
            print(f"      only in anvita: {anv_keys - ours_keys}")
        print(f"    n_params: ours={ours_info.get('n_params')} anvita={anv_info.get('n_params')}")
