"""P15 matrix coverage validator.

Loads ``tetrarl/eval/configs/p15_drl_matrix.yaml`` and checks that every
expected ``summary.json`` is present under ``--runs-root`` and that its
status / required fields match the matrix's ``expected`` field.

Cell directory convention: ``<runs-root>/<env>__<hw>__<algo>__<wrapper>__seed<N>/summary.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


REQUIRED_COMPLETED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "cumulative_reward_curve": list,
    "framework_overhead_pct": (int, float),
    "mean_deadline_miss_rate": (int, float),
    "mean_p99_step_ms": (int, float),
    "peak_gpu_memory_mb": (int, float),
    "mean_energy_j": (int, float),
    "time_to_converge_steps": int,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P15 matrix coverage validator")
    p.add_argument("--matrix", help="path to p15_drl_matrix.yaml (full mode)")
    p.add_argument("--runs-root", help="directory containing per-cell run dirs (full mode)")
    p.add_argument(
        "--runs-dir",
        help="directory containing per-cell run dirs (lightweight mode; alias for --runs-root)",
    )
    p.add_argument(
        "--expected-cells",
        type=int,
        help=(
            "lightweight mode: total expected COMPLETED summary.json files under "
            "--runs-dir (SKIPPED files are checked but counted separately)"
        ),
    )
    p.add_argument("--strict", action="store_true", help="fail on any missing required field")
    args = p.parse_args()
    if args.runs_dir and not args.runs_root:
        args.runs_root = args.runs_dir
    if not args.matrix and not (args.runs_root and args.expected_cells is not None):
        p.error("must provide either --matrix and --runs-root, OR --runs-dir and --expected-cells")
    return args


def _cell_dir_name(run: dict[str, Any]) -> str:
    return f"{run['env']}__{run['hw']}__{run['algo']}__{run['wrapper']}__seed{int(run['seed'])}"


def _check_completed(summary: dict[str, Any], strict: bool) -> list[str]:
    errs: list[str] = []
    if summary.get("status") != "COMPLETED":
        errs.append(f"status={summary.get('status')!r} (want COMPLETED)")
    for field, want_type in REQUIRED_COMPLETED_FIELDS.items():
        if field not in summary:
            errs.append(f"missing field {field!r}")
            continue
        v = summary[field]
        if not isinstance(v, want_type):
            errs.append(f"field {field!r} has type {type(v).__name__}")
            continue
        # Non-trivial / range checks.
        if field == "cumulative_reward_curve":
            if not isinstance(v, list) or len(v) == 0:
                errs.append("cumulative_reward_curve is empty")
        elif field == "framework_overhead_pct":
            if float(v) < 0.0:
                errs.append("framework_overhead_pct < 0")
        elif field == "mean_deadline_miss_rate":
            f = float(v)
            if f < 0.0 or f > 1.0:
                errs.append("mean_deadline_miss_rate out of [0,1]")
        elif field == "mean_p99_step_ms":
            if float(v) <= 0.0:
                errs.append("mean_p99_step_ms <= 0")
        elif field == "peak_gpu_memory_mb":
            if float(v) < 0.0:
                errs.append("peak_gpu_memory_mb < 0")
        elif field == "mean_energy_j":
            if float(v) < 0.0:
                errs.append("mean_energy_j < 0")
        elif field == "time_to_converge_steps":
            if not isinstance(v, int):
                errs.append("time_to_converge_steps is not int")
    if strict:
        return errs
    # In non-strict mode we still report all errors — strict only widens what
    # counts as a hard failure (currently identical here).
    return errs


def _check_skipped(summary: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    if summary.get("status") not in {"SKIPPED", "DEFERRED"}:
        errs.append(f"status={summary.get('status')!r} (want SKIPPED or DEFERRED)")
    reason = summary.get("reason", "")
    if not isinstance(reason, str) or not reason.strip():
        errs.append("missing/empty 'reason'")
    return errs


def main() -> int:
    args = _parse_args()
    runs_root = Path(args.runs_root).expanduser().resolve()

    # Lightweight mode: discover summary.json files directly under runs_root.
    if args.matrix is None:
        return _main_lightweight(runs_root, int(args.expected_cells), args.strict)

    matrix_path = Path(args.matrix).expanduser().resolve()
    with matrix_path.open() as f:
        doc = yaml.safe_load(f)
    runs: list[dict[str, Any]] = doc.get("runs", [])
    n_total = len(runs)
    n_expected_completed = sum(1 for r in runs if r["expected"] == "COMPLETED")
    n_expected_skipped = sum(1 for r in runs if r["expected"] == "SKIPPED")

    n_present = 0
    n_missing = 0
    n_mismatched = 0
    failures: list[str] = []

    for run in runs:
        cell_name = _cell_dir_name(run)
        summary_path = runs_root / cell_name / "summary.json"
        if not summary_path.exists():
            n_missing += 1
            failures.append(f"MISSING {cell_name}: {summary_path} not found")
            continue
        n_present += 1
        try:
            with summary_path.open() as f:
                summary = json.load(f)
        except Exception as exc:
            n_mismatched += 1
            failures.append(f"MISMATCH {cell_name}: failed to parse JSON ({exc!r})")
            continue
        expected = run["expected"]
        if expected == "COMPLETED":
            errs = _check_completed(summary, args.strict)
        elif expected == "SKIPPED":
            errs = _check_skipped(summary)
        else:
            errs = [f"unknown expected={expected!r}"]
        if errs:
            n_mismatched += 1
            failures.append(f"MISMATCH {cell_name}: " + "; ".join(errs))

    print(f"[P15-VALIDATE] matrix={matrix_path}  runs_root={runs_root}")
    print(
        f"total expected: {n_total}  "
        f"({n_expected_completed} COMPLETED, {n_expected_skipped} SKIPPED)"
    )
    print(f"present: {n_present}  missing: {n_missing}  mismatched: {n_mismatched}")
    ok = n_missing == 0 and n_mismatched == 0
    if not ok:
        print("FAILED")
        for line in failures[:50]:
            print("  -", line)
        if len(failures) > 50:
            print(f"  ... and {len(failures) - 50} more")
        return 1
    print("ALL OK")
    return 0


def _main_lightweight(runs_root: Path, expected_completed: int, strict: bool) -> int:
    """Lightweight mode: count summary.json files under runs_root, classify by status."""
    if not runs_root.exists():
        print(f"[P15-VALIDATE] runs_root={runs_root}")
        print(f"FAILED: runs_root does not exist")
        return 1
    cells = sorted(
        d for d in runs_root.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )
    n_completed = 0
    n_skipped = 0
    n_other = 0
    failures: list[str] = []
    for cell in cells:
        sp = cell / "summary.json"
        if not sp.exists():
            failures.append(f"MISSING {cell.name}: summary.json not found")
            continue
        try:
            with sp.open() as f:
                summary = json.load(f)
        except Exception as exc:
            failures.append(f"MISMATCH {cell.name}: failed to parse JSON ({exc!r})")
            continue
        status = summary.get("status")
        if status == "COMPLETED":
            errs = _check_completed(summary, strict)
            if errs:
                failures.append(f"MISMATCH {cell.name}: " + "; ".join(errs))
            else:
                n_completed += 1
        elif status in {"SKIPPED", "DEFERRED"}:
            errs = _check_skipped(summary)
            if errs:
                failures.append(f"MISMATCH {cell.name}: " + "; ".join(errs))
            else:
                n_skipped += 1
        else:
            n_other += 1
            failures.append(f"MISMATCH {cell.name}: unknown status={status!r}")

    print(f"[P15-VALIDATE-LIGHT] runs_root={runs_root}")
    print(f"expected COMPLETED: {expected_completed}")
    print(
        f"found: COMPLETED={n_completed}  SKIPPED/DEFERRED={n_skipped}  other={n_other}  "
        f"total cells={len(cells)}"
    )
    ok = (n_completed == expected_completed) and not failures
    if not ok:
        print("FAILED")
        if n_completed != expected_completed:
            print(
                f"  - expected {expected_completed} COMPLETED cells, found {n_completed}"
            )
        for line in failures[:50]:
            print("  -", line)
        if len(failures) > 50:
            print(f"  ... and {len(failures) - 50} more")
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
