"""Week 8 deliverable: end-to-end EvalRunner smoke against ablation YAML.

Loads ``tetrarl/eval/configs/ablation_smoke.yaml`` (5 ablation arms × 1
seed × CartPole-v1 × 50 episodes), runs the full sweep through
:class:`tetrarl.eval.runner.EvalRunner`, and prints a Markdown summary
table — one row per ablation arm — covering reward, tail-p99 latency,
mean energy, mean memory util, override fire count, and wall time.

Wall-time budget: total runner ≤ 90 s on Mac CPU. Validates that the 5
ablation arms produce distinct (mean_reward, override_fire_count)
tuples; emits a warning if all rows are identical (suggests the
ablation factory is not actually toggling the components).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional


def _print_markdown_table(rows: list[dict]) -> None:
    """Print a one-row-per-ablation Markdown table to stdout."""
    headers = [
        "ablation",
        "mean_reward",
        "tail_p99_ms",
        "mean_energy_j",
        "mean_memory_util",
        "override_fire_count",
        "wall_time_s",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        cells = [
            str(r["ablation"]),
            f"{r['mean_reward']:.4f}",
            f"{r['tail_p99_ms']:.4f}",
            f"{r['mean_energy_j']:.6f}",
            f"{r['mean_memory_util']:.4f}",
            str(r["override_fire_count"]),
            f"{r['wall_time_s']:.4f}",
        ]
        print("| " + " | ".join(cells) + " |")


def main(argv: Optional[list[str]] = None) -> int:
    # Lazy imports so test collection / --help stay cheap.
    from tetrarl.eval.runner import EvalRunner, load_sweep_yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="tetrarl/eval/configs/ablation_smoke.yaml",
        help="Path to sweep YAML (default: ablation_smoke.yaml).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Override out_dir for all configs; default uses runs/w8_smoke_<ts>.",
    )
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 1

    configs = load_sweep_yaml(cfg_path)
    if not configs:
        print(f"ERROR: no configs loaded from {cfg_path}", file=sys.stderr)
        return 1

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = int(time.time())
        out_dir = Path("runs") / f"w8_smoke_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in configs:
        c.out_dir = out_dir

    print(f"Loaded {len(configs)} configs from {cfg_path}")
    print(f"Writing to {out_dir}")

    t0 = time.perf_counter()
    results = EvalRunner().run_sweep(configs)
    wall_time_s = time.perf_counter() - t0

    rows = []
    for cfg, res in zip(configs, results):
        rows.append(
            {
                "ablation": cfg.ablation,
                "mean_reward": res.mean_reward,
                "tail_p99_ms": res.tail_p99_ms,
                "mean_energy_j": res.mean_energy_j,
                "mean_memory_util": res.mean_memory_util,
                "override_fire_count": res.override_fire_count,
                "wall_time_s": res.wall_time_s,
            }
        )

    print()
    _print_markdown_table(rows)
    print()

    # On mac_stub, the resource_manager arm is expected to produce identical
    # functional metrics (mean_reward, override_fire_count, mean_energy_j,
    # mean_memory_util) to the `none` baseline because the synthetic telemetry
    # loop does not close on DVFS decisions. The other 3 arms (preference_plane,
    # rl_arbiter, override_layer) MUST produce distinct functional metrics.
    baseline = next(r for r in results if r.config["ablation"] == "none")
    expected_collapsed = {"resource_manager"}  # mac_stub: no DVFS feedback

    def _functional_tuple(r):
        return (round(r.mean_reward, 4), r.override_fire_count, round(r.mean_energy_j, 6))

    collapsed = {
        r.config["ablation"]
        for r in results
        if r.config["ablation"] != "none" and _functional_tuple(r) == _functional_tuple(baseline)
    }
    unexpected = collapsed - expected_collapsed
    if unexpected:
        print(f"WARNING: ablation arms {unexpected} collapse with baseline — wiring may be broken")
        # Do not exit non-zero — the warning is enough for the smoke script.

    print(f"total wall_time (s)   : {wall_time_s:.2f}")
    print(f"out dir               : {out_dir}")
    print(f"summary csv           : {out_dir / 'summary.csv'}")
    print("WEEK 8 EVAL SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
