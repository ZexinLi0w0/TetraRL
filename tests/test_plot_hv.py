"""Unit test for the HV convergence plot script."""

import tempfile
from pathlib import Path

from scripts.plot_hv_convergence import plot_hv_convergence


class TestPlotHVConvergence:

    def test_creates_png(self):
        progress = [
            {"frames": i * 1000, "hv": 50.0 + i * 20.0, "n_pareto": 3,
             "episode": i, "elapsed_s": i * 2.0, "epsilon": 1.0, "loss": 0.1}
            for i in range(1, 9)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "test_hv.png"
            result = plot_hv_convergence(progress, out)
            assert result.exists()
            assert result.stat().st_size > 0

    def test_no_reference_line(self):
        progress = [
            {"frames": i * 5000, "hv": 100.0 + i * 10.0, "n_pareto": 5,
             "episode": i * 2, "elapsed_s": i * 3.0, "epsilon": 0.5, "loss": 0.05}
            for i in range(1, 6)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "no_ref.png"
            result = plot_hv_convergence(progress, out, reference=None)
            assert result.exists()
            assert result.stat().st_size > 0

    def test_nested_output_dir(self):
        progress = [
            {"frames": 10000, "hv": 200.0, "n_pareto": 8,
             "episode": 50, "elapsed_s": 30.0, "epsilon": 0.1, "loss": 0.01}
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sub" / "dir" / "plot.png"
            result = plot_hv_convergence(progress, out)
            assert result.exists()
