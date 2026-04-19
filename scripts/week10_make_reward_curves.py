"""Week 10 Task 3 deliverable: reward-vs-{walltime, energy} curves.

Reads per-step JSONL files emitted by :class:`tetrarl.eval.runner.EvalRunner`
(records carry ``episode, step, reward, latency_ms, energy_j, memory_util,
omega``) for every entry in a sweep YAML manifest, smooths the per-step
reward with a rolling mean, and renders two figures:

  * ``reward_vs_walltime.{png,svg}``: smoothed reward vs. cumulative
    wall-clock seconds (``cumsum(latency_ms / 1000)``).
  * ``reward_vs_energy.{png,svg}``: smoothed reward vs. cumulative
    energy in joules (``cumsum(energy_j)``).

One curve per ``(agent, env, omega_idx)`` group; seeds are averaged on a
common x-grid (union of all seeds' x-points, then ``np.interp``) and the
figure renders a shaded ±1 std band around the mean. Curves are coloured
by agent so different ω at the same agent share a hue.

Per the §9.6 pitfall, this script intentionally does NOT emit a
``reward_vs_steps`` figure — wall-clock and energy axes are the fair
comparison for on-policy methods.

Manifest entries that point at a missing JSONL are warned to stderr and
skipped, never crashing the whole render. The filename for each entry is
taken from ``extra["jsonl_name"]`` if present, else fall back to
``<ablation>__<agent>__seed<seed>.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib

# Force a non-interactive backend; safe for headless CI / Orin / Nano runs.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow ``matplotlib.use``)
import numpy as np  # noqa: E402
import yaml  # noqa: E402

# Stable per-agent colour palette. Falls back to the Matplotlib tab cycle
# when an unknown agent is encountered so the figure still renders.
_AGENT_COLORS: dict[str, str] = {
    "preference_ppo": "#1f77b4",   # blue
    "envelope_morl": "#d62728",    # red
    "ppo_lagrangian": "#2ca02c",   # green
    "max_p": "#ff7f0e",            # orange
    "random": "#9467bd",           # purple
    "static_dvfs": "#8c564b",      # brown
    "round_robin": "#e377c2",      # pink
    "greedy": "#17becf",           # teal
}

_FALLBACK_CYCLE: tuple[str, ...] = (
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
)

# Linestyle per omega_idx so ω corners stay distinguishable within an agent.
_OMEGA_LINESTYLES: tuple[str, ...] = ("-", "--", ":", "-.", (0, (3, 1, 1, 1)))


def _color_for_agent(agent: str, idx: int) -> str:
    """Return the stable colour for ``agent`` or fall back to the cycle."""
    if agent in _AGENT_COLORS:
        return _AGENT_COLORS[agent]
    return _FALLBACK_CYCLE[idx % len(_FALLBACK_CYCLE)]


def _linestyle_for_omega(omega_idx: int):
    """Return a stable linestyle for ``omega_idx`` (modulo the cycle)."""
    return _OMEGA_LINESTYLES[omega_idx % len(_OMEGA_LINESTYLES)]


def _resolve_jsonl_name(entry: dict) -> str:
    """Return the JSONL filename for one manifest entry.

    Prefers ``extra["jsonl_name"]`` so callers can pin per-ω disambiguating
    suffixes (e.g. ``__o0.jsonl``); falls back to the runner's default
    ``<ablation>__<agent>__seed<seed>.jsonl`` shape.
    """
    extra = entry.get("extra") or {}
    name = extra.get("jsonl_name")
    if isinstance(name, str) and name:
        return name
    ablation = str(entry.get("ablation", "none"))
    agent = str(entry.get("agent_type", "agent"))
    seed = int(entry.get("seed", 0))
    return f"{ablation}__{agent}__seed{seed}.jsonl"


def _load_jsonl(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(reward, latency_ms, energy_j)`` 1-D arrays from a JSONL file.

    Lines that fail to parse, or that lack one of the three required keys,
    are silently skipped so a partially-truncated tail does not abort the
    whole render. The arrays returned share the same length.
    """
    rewards: list[float] = []
    latencies: list[float] = []
    energies: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            try:
                r = float(rec["reward"])
                lat = float(rec["latency_ms"])
                ej = float(rec["energy_j"])
            except (KeyError, TypeError, ValueError):
                continue
            rewards.append(r)
            latencies.append(lat)
            energies.append(ej)
    return (
        np.asarray(rewards, dtype=np.float64),
        np.asarray(latencies, dtype=np.float64),
        np.asarray(energies, dtype=np.float64),
    )


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Return a rolling mean of ``x`` with the given ``window`` (>=1).

    Uses an "expanding-then-rolling" form: the first ``window-1`` outputs
    are means of the prefix so the smoothed series stays the same length
    as the input, which matters for matching it against the cumulative
    x-axis arrays.
    """
    n = x.size
    if n == 0 or window <= 1:
        return x.astype(np.float64, copy=True)
    w = int(min(window, n))
    out = np.empty(n, dtype=np.float64)
    csum = np.cumsum(x, dtype=np.float64)
    for i in range(n):
        if i < w - 1:
            out[i] = csum[i] / float(i + 1)
        else:
            lo = i - w + 1
            prev = csum[lo - 1] if lo > 0 else 0.0
            out[i] = (csum[i] - prev) / float(w)
    return out


def _interp_to_grid(
    xs: list[np.ndarray],
    ys: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate per-seed (x, y) pairs onto a common grid (union of x).

    Returns ``(grid, mean, std)`` where ``mean`` and ``std`` are the
    per-grid-point statistics across seeds. NaNs (empty seeds) are
    filtered before stacking.
    """
    if not xs:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty
    # Union of all x-points, deduplicated and sorted; ``np.interp``
    # requires a strictly increasing xp on each seed, which is given
    # since both axes are cumulative sums of non-negative quantities.
    grid = np.unique(np.concatenate(xs))
    if grid.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty
    stack: list[np.ndarray] = []
    for x, y in zip(xs, ys):
        if x.size == 0 or y.size == 0:
            continue
        # Clip the grid to each seed's [x[0], x[-1]] support so we never
        # extrapolate; np.interp would otherwise hold the endpoint
        # constant, which would bias the mean toward whichever seed
        # ran longest.
        lo, hi = float(x[0]), float(x[-1])
        clipped_grid = np.clip(grid, lo, hi)
        interp = np.interp(clipped_grid, x, y)
        # Mask grid points that lie outside the seed's support so they
        # don't bias the mean/std. Use NaN + nanmean/nanstd downstream.
        mask = (grid < lo) | (grid > hi)
        interp = interp.astype(np.float64, copy=True)
        interp[mask] = np.nan
        stack.append(interp)
    if not stack:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty
    arr = np.vstack(stack)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(arr, axis=0)
        if arr.shape[0] >= 2:
            std = np.nanstd(arr, axis=0, ddof=1)
        else:
            std = np.zeros_like(mean)
    # Drop grid points where every seed was masked out.
    keep = ~np.isnan(mean)
    return grid[keep], mean[keep], std[keep]


def _group_key(entry: dict) -> tuple[str, str, int]:
    """Return ``(agent, env, omega_idx)`` for grouping curves across seeds."""
    agent = str(entry.get("agent_type", "agent"))
    env = str(entry.get("env_name", "env"))
    extra = entry.get("extra") or {}
    omega_idx = int(extra.get("omega_idx", 0))
    return agent, env, omega_idx


def _load_groups(
    matrix_yaml: Path,
    runs_dir: Path,
    *,
    smoothing_window: int,
) -> dict[tuple[str, str, int], dict[str, list[np.ndarray]]]:
    """Load every JSONL referenced by the manifest, grouped by curve key.

    Returns ``{(agent, env, omega_idx): {"walltime_x": [...],
    "energy_x": [...], "reward_y": [...]}}`` where each list element is
    a per-seed numpy array, ready to feed into :func:`_interp_to_grid`.
    """
    with matrix_yaml.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    configs = doc.get("configs") or []
    groups: dict[tuple[str, str, int], dict[str, list[np.ndarray]]] = {}
    for entry in configs:
        if not isinstance(entry, dict):
            continue
        fname = _resolve_jsonl_name(entry)
        path = runs_dir / fname
        if not path.exists():
            print(
                f"warning: missing JSONL for {entry.get('agent_type')}/"
                f"seed{entry.get('seed')}: {path}",
                file=sys.stderr,
            )
            continue
        rewards, latencies, energies = _load_jsonl(path)
        if rewards.size == 0:
            print(
                f"warning: no usable records in {path}",
                file=sys.stderr,
            )
            continue
        walltime = np.cumsum(latencies / 1000.0)
        energy_cum = np.cumsum(energies)
        smoothed = _rolling_mean(rewards, smoothing_window)
        key = _group_key(entry)
        bucket = groups.setdefault(
            key,
            {"walltime_x": [], "energy_x": [], "reward_y": []},
        )
        bucket["walltime_x"].append(walltime)
        bucket["energy_x"].append(energy_cum)
        bucket["reward_y"].append(smoothed)
    return groups


def _render_axis(
    out_path_png: Path,
    out_path_svg: Path,
    groups: dict[tuple[str, str, int], dict[str, list[np.ndarray]]],
    *,
    x_key: str,
    xlabel: str,
    title: str,
) -> None:
    """Render one axis (walltime or energy) as PNG + SVG.

    ``x_key`` is the bucket key for the cumulative x-array
    (``"walltime_x"`` or ``"energy_x"``). One curve per group, coloured
    by agent and styled by ω-index, with a shaded ±1 std band.
    """
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    # Sorted iteration so the legend is deterministic across runs.
    sorted_keys = sorted(groups.keys())
    # Stable agent index for fallback colouring.
    seen_agents: dict[str, int] = {}
    for key in sorted_keys:
        agent, env, omega_idx = key
        if agent not in seen_agents:
            seen_agents[agent] = len(seen_agents)
        bucket = groups[key]
        xs = bucket[x_key]
        ys = bucket["reward_y"]
        grid, mean, std = _interp_to_grid(xs, ys)
        if grid.size == 0:
            continue
        color = _color_for_agent(agent, seen_agents[agent])
        ls = _linestyle_for_omega(omega_idx)
        label = f"{agent} | {env} | ω{omega_idx}"
        ax.plot(
            grid,
            mean,
            label=label,
            color=color,
            linestyle=ls,
            linewidth=1.6,
        )
        # Shade ±1 std around the mean. Skip when std is identically zero
        # (single-seed case) to keep the band invisible rather than a
        # degenerate flat ribbon.
        if np.any(std > 0):
            ax.fill_between(
                grid,
                mean - std,
                mean + std,
                color=color,
                alpha=0.15,
                linewidth=0,
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Smoothed reward (rolling mean)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    if groups:
        ax.legend(loc="best", fontsize="x-small", ncol=1)
    fig.tight_layout()
    out_path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path_png, dpi=150)
    out_path_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path_svg)
    plt.close(fig)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-yaml",
        required=True,
        type=Path,
        help="Sweep YAML mapping (agent, env, omega, seed) to a JSONL filename.",
    )
    parser.add_argument(
        "--runs-dir",
        required=True,
        type=Path,
        help="Directory containing the per-config JSONL files.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory where reward_vs_{walltime,energy}.{png,svg} are written.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=50,
        help="Rolling-mean window applied to per-step reward (default: 50).",
    )
    args = parser.parse_args(argv)

    matrix_yaml = Path(args.matrix_yaml)
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)

    if not matrix_yaml.is_file():
        print(
            f"error: --matrix-yaml {str(matrix_yaml)!r} does not exist",
            file=sys.stderr,
        )
        return 1
    if not runs_dir.is_dir():
        print(
            f"error: --runs-dir {str(runs_dir)!r} does not exist or is not a directory",
            file=sys.stderr,
        )
        return 1

    smoothing_window = max(1, int(args.smoothing_window))
    groups = _load_groups(
        matrix_yaml,
        runs_dir,
        smoothing_window=smoothing_window,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _render_axis(
        out_dir / "reward_vs_walltime.png",
        out_dir / "reward_vs_walltime.svg",
        groups,
        x_key="walltime_x",
        xlabel="Cumulative wall-clock time (s)",
        title="Reward vs wall-clock (per-agent × env × ω; mean ± 1 std over seeds)",
    )
    _render_axis(
        out_dir / "reward_vs_energy.png",
        out_dir / "reward_vs_energy.svg",
        groups,
        x_key="energy_x",
        xlabel="Cumulative energy (J)",
        title="Reward vs energy (per-agent × env × ω; mean ± 1 std over seeds)",
    )

    print(
        f"wrote {out_dir / 'reward_vs_walltime.png'}, "
        f"{out_dir / 'reward_vs_walltime.svg'}, "
        f"{out_dir / 'reward_vs_energy.png'}, "
        f"{out_dir / 'reward_vs_energy.svg'} "
        f"({len(groups)} groups)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
