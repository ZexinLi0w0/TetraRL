#!/usr/bin/env python3
"""Profile DVFS transition latency for all (from_freq, to_freq) pairs.

Run on the target hardware (e.g., Orin AGX) as root or with appropriate
sysfs write permissions. Emits a CSV and a markdown summary table.

On Mac dev (no sysfs), falls back to stub mode and produces a synthetic
table so the pipeline can be exercised end-to-end.

Usage:
    python scripts/profile_orin_dvfs.py [--domain cpu|gpu|both] \\
        [--n-iters 5] [--out-dir docs/]
"""
from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

from tetrarl.sys.dvfs import DVFSController, TransitionLatency


def write_csv(rows: list[TransitionLatency], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "from_freq", "to_freq", "latency_ms"])
        for r in rows:
            w.writerow([r.domain, r.from_freq, r.to_freq, f"{r.latency_ms:.4f}"])


def write_markdown_table(rows: list[TransitionLatency], path: Path,
                         meta: dict) -> None:
    by_domain: dict[str, list[TransitionLatency]] = defaultdict(list)
    for r in rows:
        by_domain[r.domain].append(r)

    lines: list[str] = []
    lines.append("# Orin AGX DVFS Transition Latency")
    lines.append("")
    lines.append(f"- Host: `{meta['host']}`")
    lines.append(f"- Mode: `{meta['mode']}`")
    lines.append(f"- Iterations per pair: `{meta['n_iters']}`")
    lines.append(f"- Timestamp (UTC): `{meta['timestamp']}`")
    lines.append("")

    for domain, items in by_domain.items():
        freqs = sorted({r.from_freq for r in items} | {r.to_freq for r in items})
        latency_map: dict[tuple[int, int], float] = {
            (r.from_freq, r.to_freq): r.latency_ms for r in items
        }
        scale = "kHz" if domain == "cpu" else "Hz"

        lines.append(f"## {domain.upper()} ({scale}) — mean ms per transition")
        lines.append("")
        header = "| from \\\\ to |" + "|".join(str(f) for f in freqs) + "|"
        sep = "|---|" + "|".join("---" for _ in freqs) + "|"
        lines.append(header)
        lines.append(sep)
        for fa in freqs:
            cells = []
            for fb in freqs:
                if fa == fb:
                    cells.append("—")
                else:
                    v = latency_map.get((fa, fb))
                    cells.append(f"{v:.3f}" if v is not None else "n/a")
            lines.append(f"| {fa} |" + "|".join(cells) + "|")
        lines.append("")

        flat = [r.latency_ms for r in items]
        lines.append(f"- min: {min(flat):.3f} ms")
        lines.append(f"- max: {max(flat):.3f} ms")
        lines.append(f"- mean: {sum(flat)/len(flat):.3f} ms")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["cpu", "gpu", "both"], default="both")
    ap.add_argument("--n-iters", type=int, default=5)
    ap.add_argument("--out-dir", type=Path, default=Path("docs"))
    ap.add_argument("--stub", action="store_true",
                    help="Force stub mode (synthetic latencies)")
    args = ap.parse_args()

    ctrl = DVFSController(stub=True if args.stub else None)
    mode = "stub" if ctrl.stub else "real"
    print(f"[profile_orin_dvfs] mode={mode} domain={args.domain} "
          f"n_iters={args.n_iters}")

    domains = ["cpu", "gpu"] if args.domain == "both" else [args.domain]
    all_results: list[TransitionLatency] = []
    for d in domains:
        print(f"  profiling {d}...", flush=True)
        t0 = time.perf_counter()
        rows = ctrl.profile_transition_latency(domain=d, n_iters=args.n_iters)
        t1 = time.perf_counter()
        print(f"  {d}: {len(rows)} pairs in {t1 - t0:.2f}s")
        all_results.extend(rows)

    meta = {
        "host": platform.node() or "unknown",
        "mode": mode,
        "n_iters": args.n_iters,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    csv_path = args.out_dir / "orin_dvfs_latency_table.csv"
    md_path = args.out_dir / "orin_dvfs_latency_table.md"
    write_csv(all_results, csv_path)
    write_markdown_table(all_results, md_path, meta)
    print(f"  wrote {csv_path}")
    print(f"  wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
