"""Evaluate all trained models and produce a results table.

Run after training completes:
    python scripts/eval_all.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_crps import evaluate_crps

DATA_DIR = Path("external/constrained-downscaling/data")
PRED_DIR = DATA_DIR / "prediction"


def evaluate_model(model_id: str, model_type: str, split: str = "test") -> dict:
    """Evaluate a single model."""
    target_path = DATA_DIR / "era5_sr_data" / split / f"target_{split}.pt"
    input_path = DATA_DIR / "era5_sr_data" / split / f"input_{split}.pt"

    if model_type == "gan":
        pred_path = PRED_DIR / f"era5_sr_data_{model_id}_{split}_ensemble.pt"
    else:
        pred_path = PRED_DIR / f"era5_sr_data_{model_id}_{split}.pt"

    if not pred_path.exists():
        print(f"  Predictions not found: {pred_path}")
        return {}

    results = evaluate_crps(str(pred_path), str(target_path), str(input_path))
    return results


def main():
    models = [
        ("gan_none", "gan", "GAN, no constraints"),
        ("gan_softmax", "gan", "GAN, SmCL"),
        ("cnn_none", "cnn", "CNN, no constraints"),
        ("cnn_softmax", "cnn", "CNN, SmCL"),
    ]

    all_results = {}
    print(f"{'Model':<25} {'CRPS':>10} {'RMSE':>10} {'MAE':>10} {'MassViol':>10} {'Spread':>10}")
    print("-" * 75)

    for model_id, model_type, desc in models:
        results = evaluate_model(model_id, model_type)
        if not results:
            print(f"{desc:<25} {'N/A':>10}")
            continue
        all_results[model_id] = results
        print(f"{desc:<25} {results['CRPS']:10.4f} {results['RMSE']:10.4f} {results['MAE']:10.4f} {results['Mass_Violation']:10.4f} {results['Ensemble_Spread']:10.4f}")

    # Save results as JSON
    out_path = Path("reports/baseline_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
