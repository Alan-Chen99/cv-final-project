"""Integration tests for visualization module.

Tests verify that plotting functions produce valid PNG files without errors.
Uses synthetic data — no GPU or model weights required.
"""

import json

import numpy as np
import pytest

from downscaling.visualization import (
    plot_constraint_effect,
    plot_ensemble_members,
    plot_mass_violation,
    plot_metrics_comparison,
    plot_sample_grid,
)


@pytest.fixture
def synthetic_data():
    """Create synthetic data matching expected shapes."""
    rng = np.random.default_rng(42)
    n_samples = 4
    n_ensemble = 5
    h, w = 128, 128

    lr_up = rng.uniform(0, 100, (n_samples, 1, h, w)).astype(np.float32)
    hr = lr_up + rng.normal(0, 5, (n_samples, 1, h, w)).astype(np.float32)
    ensemble_preds = np.stack(
        [hr + rng.normal(0, 2, (n_samples, 1, h, w)).astype(np.float32) for _ in range(n_ensemble)],
        axis=1,
    )
    return lr_up, hr, ensemble_preds


@pytest.fixture
def results_json(tmp_path):
    """Create a synthetic results JSON."""
    data = {
        "timestamp": "2026-01-01 00:00:00",
        "config": {"max_samples": 100, "n_ensemble": 10, "ode_steps": 10, "sampler": "midpoint"},
        "results": [
            {
                "method": "bilinear",
                "constraint": "none",
                "crps": 0.5,
                "crps_paper": 0.5,
                "mae": 0.5,
                "rmse": 0.9,
                "mass_violation": 0.3,
                "spread": 0.0,
                "n_samples": 100,
                "n_ensemble": 1,
            },
            {
                "method": "bilinear",
                "constraint": "addcl",
                "crps": 0.4,
                "crps_paper": 0.4,
                "mae": 0.4,
                "rmse": 0.8,
                "mass_violation": 0.0,
                "spread": 0.0,
                "n_samples": 100,
                "n_ensemble": 1,
            },
            {
                "method": "bicubic",
                "constraint": "none",
                "crps": 0.38,
                "crps_paper": 0.38,
                "mae": 0.38,
                "rmse": 0.77,
                "mass_violation": 0.15,
                "spread": 0.0,
                "n_samples": 100,
                "n_ensemble": 1,
            },
            {
                "method": "bicubic",
                "constraint": "addcl",
                "crps": 0.35,
                "crps_paper": 0.35,
                "mae": 0.35,
                "rmse": 0.73,
                "mass_violation": 0.0,
                "spread": 0.0,
                "n_samples": 100,
                "n_ensemble": 1,
            },
            {
                "method": "flow_model",
                "constraint": "none",
                "crps": 0.18,
                "crps_paper": 0.10,
                "mae": 0.26,
                "rmse": 0.47,
                "mass_violation": 0.005,
                "spread": 0.28,
                "n_samples": 100,
                "n_ensemble": 10,
            },
            {
                "method": "flow_model",
                "constraint": "addcl",
                "crps": 0.18,
                "crps_paper": 0.10,
                "mae": 0.26,
                "rmse": 0.47,
                "mass_violation": 0.0,
                "spread": 0.28,
                "n_samples": 100,
                "n_ensemble": 10,
            },
        ],
    }
    path = tmp_path / "results.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return path


class TestSampleGrid:
    def test_produces_valid_png(self, synthetic_data, tmp_path):
        lr_up, hr, ensemble_preds = synthetic_data
        out = tmp_path / "grid.png"
        plot_sample_grid(lr_up, hr, ensemble_preds, [0, 1, 2], str(out))
        assert out.exists()
        assert out.stat().st_size > 1000  # non-trivial PNG

    def test_single_sample(self, synthetic_data, tmp_path):
        lr_up, hr, ensemble_preds = synthetic_data
        out = tmp_path / "grid_single.png"
        plot_sample_grid(lr_up, hr, ensemble_preds, [0], str(out))
        assert out.exists()


class TestEnsembleMembers:
    def test_produces_valid_png(self, synthetic_data, tmp_path):
        lr_up, hr, ensemble_preds = synthetic_data
        out = tmp_path / "ensemble.png"
        plot_ensemble_members(lr_up, hr, ensemble_preds, 0, str(out), n_show=3)
        assert out.exists()
        assert out.stat().st_size > 1000


class TestMetricsComparison:
    def test_produces_valid_png(self, results_json, tmp_path):
        out = tmp_path / "metrics.png"
        plot_metrics_comparison(results_json, str(out), metric="crps")
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_mae_metric(self, results_json, tmp_path):
        out = tmp_path / "metrics_mae.png"
        plot_metrics_comparison(results_json, str(out), metric="mae")
        assert out.exists()


class TestConstraintEffect:
    def test_produces_valid_png(self, results_json, tmp_path):
        out = tmp_path / "constraint.png"
        plot_constraint_effect(results_json, str(out))
        assert out.exists()
        assert out.stat().st_size > 1000


class TestMassViolation:
    def test_produces_valid_png(self, results_json, tmp_path):
        out = tmp_path / "mass.png"
        plot_mass_violation(results_json, str(out))
        assert out.exists()
        assert out.stat().st_size > 1000
