"""Week 10 Task 4: per-ω winner table for the sensitivity sweep.

Consumes a directory of ``none__<agent>__seedN__o<idx>.jsonl`` files,
parses ``(agent, omega_idx)`` from each filename, computes the per-run
hypervolume via :func:`tetrarl.eval.hv.compute_run_hv`, then for each
ω index reports the per-agent mean HV and the winning agent. A Welch
two-sample t-test (p<0.05) is used to ask: at how many ω is the
TetraRL Native arbiter (``preference_ppo``) significantly better than
each baseline?

Output is a single Markdown report with:
  * The per-ω HV table (rows = ω index + ω vector, columns = agent
    mean HV, winner column).
  * Win/loss summary for ``preference_ppo``.
  * Per-baseline count of significant wins (p<0.05).
  * A "value-prop characterization" paragraph honestly stating that
    TetraRL's selling points are runtime adaptability and constraint
    respect (override layer), not always-best aggregate HV.

The 9 ω indices known to this script are the 5 anchors from
:mod:`week10_make_matrix_yaml` (0..4) plus the 4 intermediates from
:mod:`week10_make_intermediate_omega_yaml` (5..8).
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tetrarl.eval.hv import compute_run_hv, welch_pvalue  # noqa: E402

# Canonical labels and vectors for the nine ω indices the W10 sweep
# spans (5 anchors + 4 intermediates). Kept in sync with
# scripts.week10_make_matrix_yaml.OMEGAS_4D and
# scripts.week10_make_intermediate_omega_yaml.INTERMEDIATE_OMEGAS_4D.
OMEGA_LABELS: dict[int, tuple[str, list[float]]] = {
    0: ("reward-only", [1.0, 0.0, 0.0, 0.0]),
    1: ("latency-only", [0.0, 1.0, 0.0, 0.0]),
    2: ("memory-only", [0.0, 0.0, 1.0, 0.0]),
    3: ("energy-only", [0.0, 0.0, 0.0, 1.0]),
    4: ("uniform", [0.25, 0.25, 0.25, 0.25]),
    5: ("reward-leaning", [0.4, 0.3, 0.2, 0.1]),
    6: ("latency-leaning", [0.1, 0.4, 0.3, 0.2]),
    7: ("mem+energy-leaning", [0.2, 0.2, 0.3, 0.3]),
    8: ("reward+latency-leaning", [0.3, 0.3, 0.2, 0.2]),
}

DEFAULT_TETRARL_AGENT = "preference_ppo"
DEFAULT_REF_POINT = (-0.1, -1.0, -0.15, -0.01)
DEFAULT_P_THRESHOLD = 0.05

# Filenames follow the runner contract:
#   <ablation>__<agent>__seed<N>__o<IDX>.jsonl
# We anchor on the trailing __o<IDX>.jsonl segment so agent names with
# underscores (e.g. ``preference_ppo``) parse correctly.
_FILENAME_RE = re.compile(
    r"^(?P<ablation>[^_]+)__(?P<agent>.+)__seed(?P<seed>\d+)__o(?P<omega_idx>\d+)\.jsonl$"
)


def _parse_filename(name: str) -> Optional[tuple[str, str, int, int]]:
    """Return ``(ablation, agent, seed, omega_idx)`` or ``None`` if no match."""
    m = _FILENAME_RE.match(name)
    if m is None:
        return None
    return (
        m.group("ablation"),
        m.group("agent"),
        int(m.group("seed")),
        int(m.group("omega_idx")),
    )


def _parse_ref_point(s: str) -> np.ndarray:
    """Parse a 4-comma-separated string into a (4,) float64 array."""
    parts = [tok.strip() for tok in s.split(",") if tok.strip()]
    if len(parts) != 4:
        raise ValueError(
            f"--ref-point expected 4 comma-separated floats, got {len(parts)}: {s!r}"
        )
    return np.asarray([float(p) for p in parts], dtype=np.float64)


def collect_per_omega_hvs(
    runs_dir: Path,
    ref_point: np.ndarray,
) -> dict[int, dict[str, list[float]]]:
    """Walk ``runs_dir`` and bucket per-run HV by (omega_idx, agent).

    Returns a nested dict ``{omega_idx: {agent: [hv, hv, ...]}}``.
    """
    per_omega: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(Path(runs_dir).glob("none__*.jsonl")):
        parsed = _parse_filename(path.name)
        if parsed is None:
            continue
        _ablation, agent, _seed, omega_idx = parsed
        hv = compute_run_hv(path, ref_point=ref_point)
        per_omega[omega_idx][agent].append(float(hv))
    # freeze defaultdicts to plain dicts for stable iteration
    return {k: dict(v) for k, v in per_omega.items()}


def _per_omega_winner(per_agent: dict[str, list[float]]) -> tuple[Optional[str], float]:
    """Return ``(winner_agent, winner_mean_hv)`` for one ω bucket.

    Empty input -> ``(None, 0.0)``.
    """
    best_agent: Optional[str] = None
    best_mean = -float("inf")
    for agent, vals in per_agent.items():
        if not vals:
            continue
        m = float(np.mean(vals))
        if m > best_mean:
            best_mean = m
            best_agent = agent
    if best_agent is None:
        return None, 0.0
    return best_agent, best_mean


def _significant_win_counts(
    per_omega: dict[int, dict[str, list[float]]],
    tetrarl_agent: str,
    p_threshold: float,
) -> dict[str, int]:
    """For each baseline, count ω at which TetraRL is significantly better.

    "Significantly better" = Welch p < ``p_threshold`` AND mean(TetraRL)
    > mean(baseline). Baselines that never appear opposite TetraRL get
    a count of 0.
    """
    baselines: set[str] = set()
    for per_agent in per_omega.values():
        baselines.update(per_agent.keys())
    baselines.discard(tetrarl_agent)

    counts = {b: 0 for b in baselines}
    for omega_idx, per_agent in per_omega.items():
        tetra = per_agent.get(tetrarl_agent, [])
        if not tetra:
            continue
        tetra_arr = np.asarray(tetra, dtype=np.float64)
        for baseline in baselines:
            base = per_agent.get(baseline, [])
            if not base:
                continue
            base_arr = np.asarray(base, dtype=np.float64)
            mean_t = float(np.mean(tetra_arr))
            mean_b = float(np.mean(base_arr))
            p = welch_pvalue(tetra_arr, base_arr)
            if p < p_threshold and mean_t > mean_b:
                counts[baseline] += 1
    return counts


def _ordered_omegas(per_omega: dict[int, dict[str, list[float]]]) -> list[int]:
    """Return ω indices in ascending integer order."""
    return sorted(per_omega.keys())


def _ordered_agents(
    per_omega: dict[int, dict[str, list[float]]],
    tetrarl_agent: str,
) -> list[str]:
    """TetraRL Native first, then remaining agents alphabetically."""
    seen: set[str] = set()
    for per_agent in per_omega.values():
        seen.update(per_agent.keys())
    others = sorted(a for a in seen if a != tetrarl_agent)
    if tetrarl_agent in seen:
        return [tetrarl_agent, *others]
    return others


def _format_omega_label(idx: int) -> str:
    label, vec = OMEGA_LABELS.get(idx, (f"unknown_{idx}", []))
    if not vec:
        return f"ω{idx} ({label})"
    vec_str = ", ".join(f"{v:.2f}" for v in vec)
    return f"ω{idx} {label} [{vec_str}]"


def render_markdown(
    per_omega: dict[int, dict[str, list[float]]],
    tetrarl_agent: str,
    ref_point: np.ndarray,
    p_threshold: float,
) -> str:
    """Render the per-ω winner table + summary as Markdown text."""
    omegas = _ordered_omegas(per_omega)
    agents = _ordered_agents(per_omega, tetrarl_agent)

    # Per-omega winner identification.
    winners: dict[int, Optional[str]] = {}
    for o in omegas:
        winner, _ = _per_omega_winner(per_omega[o])
        winners[o] = winner

    # Significant-win counts vs each baseline.
    sig_counts = _significant_win_counts(per_omega, tetrarl_agent, p_threshold)

    # TetraRL win/loss tally.
    tetra_wins = sum(1 for o in omegas if winners[o] == tetrarl_agent)
    tetra_losses = sum(
        1
        for o in omegas
        if winners[o] is not None and winners[o] != tetrarl_agent
    )

    lines: list[str] = []
    lines.append(f"# Per-ω winner table (TetraRL = `{tetrarl_agent}`)")
    lines.append("")
    lines.append(
        f"Ref point (reward, -latency, -memory, -energy) = "
        f"({ref_point[0]:.4g}, {ref_point[1]:.4g}, "
        f"{ref_point[2]:.4g}, {ref_point[3]:.4g})."
    )
    lines.append("")

    # Per-ω HV table.
    header = ["ω"] + agents + ["winner"]
    align = ["---"] + [":-:"] * len(agents) + [":-:"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(align) + " |")
    for o in omegas:
        row = [_format_omega_label(o)]
        winner = winners[o]
        for ag in agents:
            vals = per_omega[o].get(ag, [])
            if not vals:
                row.append("-")
                continue
            mean = float(np.mean(vals))
            # Use scientific notation so the small dominated-HV magnitudes
            # (~1e-5 with the W10 ref point) remain comparable across cells.
            cell = f"{mean:.3e}"
            if ag == winner:
                cell = f"**{cell}**"
            row.append(cell)
        winner_label = winner if winner is not None else "-"
        if winner == tetrarl_agent:
            winner_label = f"**{winner_label}**"
        row.append(winner_label)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Summary paragraph (HONEST: do not cherry-pick).
    lines.append("## Summary")
    lines.append("")
    n_omegas = len(omegas)
    if tetra_wins == 0:
        win_phrase = (
            f"TetraRL Native (`{tetrarl_agent}`) does NOT win on any of the "
            f"{n_omegas} ω vectors evaluated."
        )
    elif tetra_wins == n_omegas:
        win_phrase = (
            f"TetraRL Native (`{tetrarl_agent}`) wins on all {n_omegas} of the "
            f"ω vectors evaluated."
        )
    else:
        win_phrase = (
            f"TetraRL Native (`{tetrarl_agent}`) wins on {tetra_wins} of "
            f"{n_omegas} ω vectors and loses on {tetra_losses}."
        )
    lines.append(win_phrase)
    lines.append("")

    # Per-baseline significant-win count.
    if sig_counts:
        lines.append(
            f"Significant wins (Welch t-test p<{p_threshold}, mean(TetraRL) "
            f"> mean(baseline)) per baseline:"
        )
        lines.append("")
        lines.append("| baseline | ω at which TetraRL is significantly better |")
        lines.append("| --- | :-: |")
        for baseline in sorted(sig_counts.keys()):
            lines.append(f"| {baseline} | {sig_counts[baseline]} / {n_omegas} |")
        lines.append("")
        total_sig = sum(sig_counts.values())
        lines.append(
            f"Aggregate: TetraRL has **{total_sig}** statistically significant "
            f"wins across all (baseline x ω) cells "
            f"({len(sig_counts)} baselines x {n_omegas} ω = {len(sig_counts) * n_omegas} cells)."
        )
        lines.append("")

    # Value-proposition characterisation.
    lines.append("## Value-proposition characterisation")
    lines.append("")
    lines.append(
        "TetraRL's selling points are NOT \"highest aggregate HV under any "
        "fixed ω\". The empirical numbers above honestly reflect that point: "
        "on Pareto-front anchors and intermediate ω alike, several MORL "
        "baselines often match or beat `preference_ppo` on the dominated-HV "
        "scalar."
    )
    lines.append("")
    lines.append(
        "What TetraRL provides instead is (a) **adaptability when the user "
        "preference shifts at runtime** — the preference plane can be "
        "re-targeted between episodes without retraining the arbiter — and "
        "(b) **constraint respect via the override layer**, which fires on "
        "telemetry breaches and clamps the executed action to a safe fallback "
        "(see `lagrangian_violation_table.md` and the Week 10 "
        "override-telemetry tests). Per-ω HV is therefore a complement to, "
        "not a substitute for, the dynamic-preference and "
        "constraint-violation evidence."
    )
    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Week 10 per-ω winner Markdown report from the JSONL "
            "files in --runs-dir."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Directory containing none__<agent>__seedN__oI.jsonl files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination Markdown path for the per-ω winner report.",
    )
    parser.add_argument(
        "--ref-point",
        type=str,
        default=",".join(str(x) for x in DEFAULT_REF_POINT),
        help=(
            "Comma-separated 4 floats for the HV reference point in the order "
            "(reward, -latency, -memory, -energy). Default matches the "
            "rest of the W10 analysis: -0.1,-1.0,-0.15,-0.01."
        ),
    )
    parser.add_argument(
        "--tetrarl-agent",
        type=str,
        default=DEFAULT_TETRARL_AGENT,
        help="agent_type label treated as TetraRL Native (default: preference_ppo).",
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=DEFAULT_P_THRESHOLD,
        help="Welch t-test p-value cutoff for significance (default: 0.05).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    ref_point = _parse_ref_point(str(args.ref_point))
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        print(f"[per-omega] runs-dir does not exist: {runs_dir}", file=sys.stderr)
        return 1

    per_omega = collect_per_omega_hvs(runs_dir, ref_point=ref_point)
    if not per_omega:
        print(f"[per-omega] no JSONL files matched under {runs_dir}", file=sys.stderr)
        return 1

    md = render_markdown(
        per_omega,
        tetrarl_agent=str(args.tetrarl_agent),
        ref_point=ref_point,
        p_threshold=float(args.p_threshold),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"[per-omega] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
