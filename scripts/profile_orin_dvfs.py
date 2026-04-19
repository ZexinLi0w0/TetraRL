#!/usr/bin/env python3
"""Profile DVFS transition latency for all (from_freq, to_freq) pairs.

Run on the target hardware (e.g., Orin AGX, Jetson Nano) as root or with
appropriate sysfs write permissions. Emits a CSV and a markdown summary
table.

On Mac dev (no sysfs), falls back to stub mode and produces a synthetic
table so the pipeline can be exercised end-to-end.

Output naming
-------------
Output files are prefixed by the selected platform so multiple runs do
not overwrite each other:

    <out-dir>/<platform>_dvfs_latency_table.csv
    <out-dir>/<platform>_dvfs_latency_table.md

For example ``--platform nano`` writes ``nano_dvfs_latency_table.{csv,md}``
and ``--platform orin_agx`` (the default) writes
``orin_agx_dvfs_latency_table.{csv,md}``. There is no implicit Orin-only
copy: every platform produces its own cleanly-named pair.

Real-mode safety
----------------
By default real-mode sysfs writes are disabled. Pass ``--allow-real-dvfs``
to permit writes to ``scaling_setspeed`` / devfreq (requires root and
``governor=userspace``). Without it, ``--stub`` is forced even on Jetson
hardware so that exploratory profiling cannot accidentally repin the
governor.

Usage:
    python scripts/profile_orin_dvfs.py [--platform orin_agx|nano] \\
        [--domain cpu|gpu|both] [--n-iters 5] [--out-dir docs/] \\
        [--stub] [--allow-real-dvfs]
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
from tetrarl.sys.platforms import Platform


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
    lines.append(f"# {meta['platform_name']} DVFS Transition Latency")
    lines.append("")
    lines.append(f"- Platform: `{meta['platform_name']}`")
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
    ap.add_argument("--platform", choices=["orin_agx", "nano"], default="orin_agx",
                    help="Target Jetson platform; selects freq tables and sysfs paths.")
    ap.add_argument("--allow-real-dvfs", action="store_true",
                    help="Permit real-mode sysfs writes (needs root + governor=userspace).")
    ap.add_argument("--domain", choices=["cpu", "gpu", "both"], default="both")
    ap.add_argument("--n-iters", type=int, default=5)
    ap.add_argument("--out-dir", type=Path, default=Path("docs"))
    ap.add_argument("--stub", action="store_true",
                    help="Force stub mode (synthetic latencies)")
    args = ap.parse_args()

    plat = Platform(args.platform)
    ctrl = DVFSController(platform=plat, stub=True if args.stub else None)
    if not ctrl.stub and not args.allow_real_dvfs:
        # Auto-detected real-mode path; refuse to write sysfs without explicit
        # opt-in so exploratory runs cannot accidentally repin the governor.
        print("[profile_orin_dvfs] WARNING: real-mode sysfs detected but "
              "--allow-real-dvfs not set; forcing stub mode.")
        ctrl = DVFSController(platform=plat, stub=True)
    mode = "stub" if ctrl.stub else "real"
    print(f"[profile_orin_dvfs] platform={args.platform} mode={mode} "
          f"domain={args.domain} n_iters={args.n_iters}")

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
        "platform_name": ctrl.profile.name,
        "host": platform.node() or "unknown",
        "mode": mode,
        "n_iters": args.n_iters,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    csv_path = args.out_dir / f"{args.platform}_dvfs_latency_table.csv"
    md_path = args.out_dir / f"{args.platform}_dvfs_latency_table.md"
    write_csv(all_results, csv_path)
    write_markdown_table(all_results, md_path, meta)
    print(f"  wrote {csv_path}")
    print(f"  wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
