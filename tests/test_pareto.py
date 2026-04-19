"""Tests for ``tetrarl.eval.pareto`` (Week 7 Task 4)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot import in tests
import matplotlib.pyplot as plt  # noqa: E402, F401
import numpy as np  # noqa: E402

from tetrarl.eval.pareto import (  # noqa: E402
    compute_hv,
    pareto_front,
    pareto_summary_table,
    plot_2d_projections,
)


def test_compute_hv_2d_unit_square() -> None:
    points = np.array([[1.0, 1.0]])
    hv = compute_hv(points, np.array([0.0, 0.0]))
    assert hv == 1.0


def test_compute_hv_2d_two_points() -> None:
    # Two non-dominated points -- inclusion-exclusion over the two
    # axis-aligned rectangles to (0, 0).
    points = np.array([[2.0, 1.0], [1.0, 2.0]])
    hv = compute_hv(points, np.array([0.0, 0.0]))
    assert hv == 3.0


def test_compute_hv_4d_known_box() -> None:
    points = np.array([[1.0, 1.0, 1.0, 1.0]])
    hv = compute_hv(points, np.array([0.0, 0.0, 0.0, 0.0]))
    assert hv == 1.0


def test_pareto_front_removes_dominated() -> None:
    # (1, 1) dominates (0.5, 0.5); (2, 0) and (0, 2) are non-dominated.
    points = np.array(
        [
            [1.0, 1.0],
            [0.5, 0.5],
            [2.0, 0.5],
            [0.5, 2.0],
        ]
    )
    front = pareto_front(points)
    front_sorted = front[np.lexsort(front.T)]
    expected = np.array([[0.5, 2.0], [1.0, 1.0], [2.0, 0.5]])
    expected_sorted = expected[np.lexsort(expected.T)]
    np.testing.assert_allclose(front_sorted, expected_sorted)


def test_pareto_front_empty() -> None:
    assert pareto_front(np.array([])).size == 0
    assert pareto_front([]).size == 0


def test_pareto_front_all_equal() -> None:
    points = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    front = pareto_front(points)
    # No point STRICTLY dominates another; all should be returned.
    assert len(front) == 3
    np.testing.assert_allclose(front, points)


def test_plot_2d_projections_writes_files(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    points = rng.normal(size=(20, 4))
    out = plot_2d_projections(
        points,
        tmp_path,
        ref_point=np.array([-10.0, -10.0, -10.0, -10.0]),
    )
    # 3 PNGs + 1 SVG -> 4 entries in the returned dict.
    assert len(out) == 4
    png_count = 0
    svg_count = 0
    for key, path in out.items():
        p = Path(path)
        assert p.exists(), f"expected {p} to exist"
        assert p.stat().st_size > 0, f"expected {p} to be non-empty"
        if p.suffix == ".png":
            png_count += 1
        elif p.suffix == ".svg":
            svg_count += 1
    assert png_count == 3
    assert svg_count == 1


def test_pareto_summary_table_format() -> None:
    points = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, 4.0, 3.0],
            [0.5, 0.5, 0.5, 0.5],
        ]
    )
    md = pareto_summary_table(
        points,
        np.array([-1.0, -1.0, -1.0, -1.0]),
    )
    assert "HV" in md
    assert "| Dimension |" in md
    # Header + separator + 4 dimension rows = 6 table lines
    table_rows = [ln for ln in md.splitlines() if ln.startswith("|")]
    # 1 header + 1 separator + 4 dim rows = 6
    assert len(table_rows) == 6


def test_pareto_summary_table_empty() -> None:
    md = pareto_summary_table(np.array([]), np.array([0.0, 0.0, 0.0, 0.0]))
    assert "no points" in md.lower()


def test_compute_hv_dominated_points_filtered() -> None:
    # (2, 2) dominates (1, 1); HV of {(2,2)} alone equals HV of the mix.
    pareto_only = np.array([[2.0, 2.0]])
    mixed = np.array([[2.0, 2.0], [1.0, 1.0], [0.5, 1.5]])
    ref = np.array([0.0, 0.0])
    hv_only = compute_hv(pareto_only, ref)
    hv_mixed = compute_hv(mixed, ref)
    # The (0.5, 1.5) point has y > 2 false but x < 2; in 2-D HV it's
    # also dominated by (2, 2) since 2 >= 0.5 and 2 >= 1.5. Both equal 4.
    assert hv_only == 4.0
    assert hv_mixed == 4.0
