"""Integration tests for spectral plotting functions."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from downscaling.plotting.spectral import (
    plot_extended_metrics_panel,
    plot_psd_comparison,
    plot_ralsd_comparison,
    plot_spectral_bias,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def freq():
    return np.linspace(0.01, 0.5, 26)


@pytest.fixture
def sample_results():
    """Minimal results dict with all 8 metrics."""
    return {
        "flow-wide96-amp (28M)": {
            "crps": 0.10,
            "mae": 0.08,
            "rmse": 0.12,
            "mass_violation": 1e-6,
            "ralsd": 2.5,
            "ssim": 0.95,
            "psnr": 35.0,
            "emd": 0.05,
        },
        "bicubic": {
            "crps": 0.39,
            "mae": 0.39,
            "rmse": 0.78,
            "mass_violation": 0.15,
            "ralsd": 8.1,
            "ssim": 0.72,
            "psnr": 25.0,
            "emd": 0.25,
        },
        "harder-cnn": {
            "crps": 0.25,
            "mae": 0.20,
            "rmse": 0.35,
            "mass_violation": 0.02,
            "ralsd": 5.3,
            "ssim": 0.85,
            "psnr": 30.0,
            "emd": 0.15,
        },
    }


class TestPSDComparison:
    def test_creates_figure(self, freq, rng):
        psd_truth = 1.0 / (freq**2 + 0.01)
        method_psds = {
            "flow-wide96-amp (28M)": psd_truth * 0.9,
            "bicubic": psd_truth * 0.5,
        }
        fig = plot_psd_comparison(freq, psd_truth, method_psds)
        assert fig is not None
        assert len(fig.axes) == 1
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_saves_to_file(self, freq, rng, tmp_path):
        psd_truth = 1.0 / (freq**2 + 0.01)
        method_psds = {"bicubic": psd_truth * 0.5}
        out = tmp_path / "psd.png"
        fig = plot_psd_comparison(freq, psd_truth, method_psds, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 1000
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestSpectralBias:
    def test_creates_figure(self, freq):
        biases = {
            "flow-wide96-amp (28M)": np.random.randn(26) * 2,
            "bicubic": np.random.randn(26) * 5,
        }
        fig = plot_spectral_bias(freq, biases)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_handles_nan(self, freq):
        bias = np.random.randn(26) * 2
        bias[0] = np.nan
        bias[25] = np.nan
        fig = plot_spectral_bias(freq, {"test": bias})
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestExtendedMetricsPanel:
    def test_creates_8_metric_panel(self, sample_results):
        fig = plot_extended_metrics_panel(sample_results)
        assert fig is not None
        # 3x3 grid, 8 visible + 1 hidden
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) == 8
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_saves_to_file(self, sample_results, tmp_path):
        out = tmp_path / "panel.png"
        fig = plot_extended_metrics_panel(sample_results, output_path=out)
        assert out.exists()
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestRALSDComparison:
    def test_creates_figure(self, sample_results):
        fig = plot_ralsd_comparison(sample_results)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_no_ralsd_data(self):
        results = {"method": {"crps": 0.1, "mae": 0.1, "rmse": 0.2, "mass_violation": 0.0}}
        fig = plot_ralsd_comparison(results)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_saves_to_file(self, sample_results, tmp_path):
        out = tmp_path / "ralsd.png"
        fig = plot_ralsd_comparison(sample_results, output_path=out)
        assert out.exists()
        import matplotlib.pyplot as plt

        plt.close(fig)
