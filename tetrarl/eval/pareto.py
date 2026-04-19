"""4-D Pareto-front analysis and visualization helpers (Week 7 Task 4).

Thin orchestration layer over ``tetrarl.eval.hypervolume``: provides

    * ``compute_hv``         -- HV indicator wrapper.
    * ``pareto_front``       -- non-dominated subset wrapper.
    * ``plot_2d_projections``-- 3 x 2-D projection scatter plots of a
                                4-D objective cloud, highlighting the
                                per-projection Pareto subset, with a
                                combined SVG figure for the paper.
    * ``pareto_summary_table``-- Markdown summary table for reports.

All functions accept lists or numpy arrays. Empty inputs short-circuit
to graceful zero / empty results so the caller never has to special-case
"no data yet" situations.

Plotting uses the headless ``Agg`` backend so the helpers are safe to
call from CI / cron jobs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tetrarl.eval.hypervolume import hypervolume, pareto_filter  # noqa: E402

_DEFAULT_DIM_LABELS: list[str] = ["Throughput", "Latency", "Energy", "Memory"]
_DEFAULT_PAIRS: list[tuple[int, int]] = [(0, 1), (0, 2), (0, 3)]


def _as_2d(points: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Coerce input to a ``(n, d)`` float64 numpy array.

    Empty inputs become a ``(0, 0)`` array; the caller is expected to
    handle the empty case explicitly (typically by returning early).
    """
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def compute_hv(
    points: np.ndarray | Sequence[Sequence[float]],
    ref_point: np.ndarray | Sequence[float],
) -> float:
    """Hypervolume indicator (maximization) of ``points`` vs ``ref_point``.

    Thin wrapper around :func:`tetrarl.eval.hypervolume.hypervolume`.
    The input is filtered to the Pareto subset first so callers can pass
    raw evaluation clouds without worrying about dominated points.

    Empty input returns 0.0.
    """
    arr = _as_2d(points)
    if arr.size == 0:
        return 0.0
    ref = np.asarray(ref_point, dtype=np.float64)
    front = pareto_filter(arr)
    return float(hypervolume(front, ref))


def pareto_front(
    points: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Return the non-dominated subset of ``points`` (maximization).

    Thin wrapper around :func:`tetrarl.eval.hypervolume.pareto_filter`.
    Empty input returns an empty array.
    """
    arr = _as_2d(points)
    if arr.size == 0:
        return arr
    return pareto_filter(arr)


def _resolve_pairs(
    pairs: Iterable[tuple[int, int]] | None,
    n_dims: int,
) -> list[tuple[int, int]]:
    if pairs is None:
        return [(i, j) for (i, j) in _DEFAULT_PAIRS if i < n_dims and j < n_dims]
    out: list[tuple[int, int]] = []
    for i, j in pairs:
        if i >= n_dims or j >= n_dims:
            raise ValueError(
                f"projection pair ({i}, {j}) out of range for n_dims={n_dims}"
            )
        out.append((int(i), int(j)))
    return out


def _resolve_labels(
    dim_labels: Sequence[str] | None,
    n_dims: int,
) -> list[str]:
    if dim_labels is None:
        if n_dims <= len(_DEFAULT_DIM_LABELS):
            return list(_DEFAULT_DIM_LABELS[:n_dims])
        return [f"obj{k}" for k in range(n_dims)]
    if len(dim_labels) < n_dims:
        raise ValueError(
            f"dim_labels has {len(dim_labels)} entries but points have "
            f"{n_dims} dimensions"
        )
    return list(dim_labels)


def _scatter_one(
    ax: plt.Axes,
    points: np.ndarray,
    pair: tuple[int, int],
    labels: list[str],
) -> None:
    i, j = pair
    sub = points[:, [i, j]]
    front_2d = pareto_filter(sub)
    ax.scatter(
        sub[:, 0],
        sub[:, 1],
        c="#bdbdbd",
        s=30,
        alpha=0.6,
        label="all points",
        zorder=1,
    )
    if len(front_2d) > 0:
        ax.scatter(
            front_2d[:, 0],
            front_2d[:, 1],
            c="#d62728",
            s=70,
            edgecolors="black",
            linewidth=0.6,
            label="Pareto subset",
            zorder=2,
        )
    ax.set_xlabel(labels[i])
    ax.set_ylabel(labels[j])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def plot_2d_projections(
    points: np.ndarray | Sequence[Sequence[float]],
    save_dir: str | Path,
    *,
    dim_labels: Sequence[str] | None = None,
    ref_point: np.ndarray | Sequence[float] | None = None,
    pairs: Iterable[tuple[int, int]] | None = None,
    dpi: int = 150,
) -> dict[str, str]:
    """Save 3 x 2-D scatter projections of a 4-D objective cloud.

    For each ``(i, j)`` pair (default ``[(0,1), (0,2), (0,3)]``):

      * scatter all points in light gray;
      * overlay the per-projection Pareto subset in red;
      * write ``projection_{labels[i]}_vs_{labels[j]}.png``.

    A combined ``pareto_combined.svg`` with all projections side-by-side
    is also saved for the paper figure.

    Title carries the *4-D* HV (computed via :func:`compute_hv`) when
    ``ref_point`` is supplied — this is the canonical aggregated metric
    even though each subplot only shows two of the four axes.

    Parameters
    ----------
    points
        ``(n, d)`` array (or list); ``d`` >= max index referenced by
        ``pairs`` (default needs ``d>=4``, but smaller ``d`` works if
        ``pairs`` is overridden).
    save_dir
        Directory to write artifacts to. Created if missing.
    dim_labels
        Per-dimension axis labels. Defaults to
        ``["Throughput","Latency","Energy","Memory"]``.
    ref_point
        Reference point for the 4-D HV displayed in titles. May be
        ``None``, in which case HV is omitted from titles.
    pairs
        Iterable of ``(i, j)`` index tuples. Defaults to
        ``[(0,1),(0,2),(0,3)]`` (Throughput vs each of L/E/M).
    dpi
        Raster DPI for PNG outputs.

    Returns
    -------
    dict[str, str]
        Mapping ``"projection_{x}_vs_{y}" -> absolute path``, plus the
        ``"combined"`` entry pointing at the SVG. For empty input the
        dict is empty and no files are written.
    """
    save_dir = Path(save_dir)
    arr = _as_2d(points)
    if arr.size == 0:
        return {}

    n_dims = arr.shape[1]
    labels = _resolve_labels(dim_labels, n_dims)
    pair_list = _resolve_pairs(pairs, n_dims)

    save_dir.mkdir(parents=True, exist_ok=True)

    hv_str: str | None = None
    if ref_point is not None:
        try:
            hv_val = compute_hv(arr, ref_point)
            hv_str = f"4D HV = {hv_val:.3f}"
        except Exception:
            hv_str = None

    out_paths: dict[str, str] = {}

    for i, j in pair_list:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        _scatter_one(ax, arr, (i, j), labels)
        title = f"{labels[i]} vs {labels[j]}"
        if hv_str is not None:
            title = f"{title}  ({hv_str})"
        ax.set_title(title)
        fname = f"projection_{labels[i]}_vs_{labels[j]}.png"
        out_path = save_dir / fname
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        key = f"projection_{labels[i]}_vs_{labels[j]}"
        out_paths[key] = str(out_path.resolve())

    fig, axes = plt.subplots(
        1, len(pair_list), figsize=(5 * len(pair_list), 4.5)
    )
    if len(pair_list) == 1:
        axes = [axes]
    for ax, (i, j) in zip(axes, pair_list):
        _scatter_one(ax, arr, (i, j), labels)
        ax.set_title(f"{labels[i]} vs {labels[j]}")
    if hv_str is not None:
        fig.suptitle(hv_str, fontsize=12)
    fig.tight_layout()
    combined_path = save_dir / "pareto_combined.svg"
    fig.savefig(combined_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    out_paths["combined"] = str(combined_path.resolve())

    return out_paths


def pareto_summary_table(
    points: np.ndarray | Sequence[Sequence[float]],
    ref_point: np.ndarray | Sequence[float],
    dim_labels: Sequence[str] | None = None,
) -> str:
    """Return a Markdown summary of a 4-D Pareto front.

    The table contains:

      * total number of points and number of Pareto points;
      * 4-D hypervolume (via :func:`compute_hv`);
      * per-dimension min/max/mean of the **Pareto** subset.

    Empty input returns the string ``"_no points_"`` (no crash).
    """
    arr = _as_2d(points)
    if arr.size == 0:
        return "_no points_"

    front = pareto_filter(arr)
    hv = compute_hv(arr, ref_point)
    labels = _resolve_labels(dim_labels, arr.shape[1])

    lines: list[str] = []
    lines.append("## Pareto summary")
    lines.append("")
    lines.append(f"- Total points: {len(arr)}")
    lines.append(f"- Pareto points: {len(front)}")
    lines.append(f"- 4-D HV: {hv:.4f}")
    lines.append("")
    lines.append("| Dimension | Min | Max | Mean |")
    lines.append("|---|---:|---:|---:|")
    if len(front) == 0:
        for lbl in labels:
            lines.append(f"| {lbl} | n/a | n/a | n/a |")
    else:
        mins = front.min(axis=0)
        maxs = front.max(axis=0)
        means = front.mean(axis=0)
        for k, lbl in enumerate(labels):
            lines.append(
                f"| {lbl} | {mins[k]:.4f} | {maxs[k]:.4f} | {means[k]:.4f} |"
            )

    return "\n".join(lines)
