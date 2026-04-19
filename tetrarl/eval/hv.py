"""Per-run hypervolume from JSONL eval logs + Welch's t-test.

Each TetraRL eval run produces a JSONL file (one record per env step)
containing the per-step keys: episode, step, action, reward,
latency_ms, energy_j, memory_util, omega. This module:

1. Computes a single per-run hypervolume scalar by collapsing each
   episode to a 4-D point (mean_reward, -mean_latency_ms,
   -mean_memory_util, -mean_energy_j) so that all four objectives are
   "higher is better", then taking the dominated HV w.r.t. a fixed
   reference point.
2. Aggregates HV across (agent, env, seed) into a list of HVRecord.
3. Provides a Welch two-sample t-test helper that normalises NaN
   p-values (zero-variance case) to 1.0.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from tetrarl.eval.hypervolume import hypervolume, pareto_filter


@dataclass
class HVRecord:
    """One (agent, env, seed) triple paired with its scalar HV."""

    agent: str
    env: str
    seed: int
    hv: float


def compute_run_hv(jsonl_path: Path, ref_point: np.ndarray) -> float:
    """Compute the per-run hypervolume from a JSONL of per-step records.

    Records are grouped by episode; each episode contributes a single
    4-D point (mean_reward, -mean_latency_ms, -mean_memory_util,
    -mean_energy_j). All four dims are framed as "higher is better".
    The Pareto front of those points is taken and its dominated HV
    relative to ``ref_point`` is returned.

    Empty file or all-points-dominated -> 0.0.
    """
    path = Path(jsonl_path)
    if not path.exists():
        return 0.0

    per_episode: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            per_episode[int(rec["episode"])].append((
                float(rec["reward"]),
                float(rec["latency_ms"]),
                float(rec["memory_util"]),
                float(rec["energy_j"]),
            ))

    if not per_episode:
        return 0.0

    points: list[list[float]] = []
    for ep in sorted(per_episode.keys()):
        steps = np.asarray(per_episode[ep], dtype=np.float64)
        mean_reward = float(steps[:, 0].mean())
        mean_latency = float(steps[:, 1].mean())
        mean_memory = float(steps[:, 2].mean())
        mean_energy = float(steps[:, 3].mean())
        # Negate the "lower is better" dims so all four become maximised.
        points.append([mean_reward, -mean_latency, -mean_memory, -mean_energy])

    arr = np.asarray(points, dtype=np.float64)
    front = pareto_filter(arr)
    return float(hypervolume(front, np.asarray(ref_point, dtype=np.float64)))


def aggregate_hv_table(
    run_dir: Path,
    manifest: dict[str, tuple[str, str, int]],
    ref_point: np.ndarray,
) -> list[HVRecord]:
    """Compute HV for each manifest-listed JSONL under ``run_dir``.

    Files in ``run_dir`` not present in ``manifest`` are silently
    skipped. Returned list is sorted by (agent, env, seed) for stable
    downstream rendering.
    """
    run_dir = Path(run_dir)
    records: list[HVRecord] = []
    for filename, (agent, env, seed) in manifest.items():
        path = run_dir / filename
        if not path.exists():
            continue
        hv = compute_run_hv(path, ref_point=ref_point)
        records.append(HVRecord(agent=agent, env=env, seed=int(seed), hv=hv))
    records.sort(key=lambda r: (r.agent, r.env, r.seed))
    return records


def welch_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    """Welch's two-sided t-test p-value with zero-variance fallback.

    scipy returns NaN when both samples have zero variance; in that
    case we return 1.0 (no evidence of separability).
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    result = stats.ttest_ind(a_arr, b_arr, equal_var=False)
    p = float(result.pvalue)
    if not np.isfinite(p):
        return 1.0
    return p
