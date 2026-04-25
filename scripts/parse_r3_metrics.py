#!/usr/bin/env python3
"""Parse one R3 training run into ``r3_native_metrics.json``.

R3 (Li, Liu et al., RTSS 2023) is the deadline-aware on-device DRL training
framework that the P11 head-to-head experiment compares against. R3 does NOT
target a Pareto front -- its native metrics are:

  * ``framework_overhead_pct``    -- wall-clock cost of the R3 runtime
                                     coordinator (per-episode batch / replay
                                     adaptation) as a fraction of total
                                     training time.
  * ``mean_deadline_miss_rate``   -- fraction of episodes that ran past their
                                     allotted runtime deadline.

The aggregator (``scripts/p11_aggregate.py``) keeps R3 in the manifest with
empty HV columns and reads its native metric from this JSON.

Parsing strategy (in order, anything found wins):

  (a) The R3 stdout log file (``--log-file``). We look for explicit
      ``framework_overhead=X.XX%`` / ``deadline_miss_rate=Y.YY`` lines that
      a future R3 wrapper might emit, and for the ``RUNTIME COORDINATOR:
      ALPHA: ..., BETA: ...`` lines that the current R3 RuntimeCoordinator
      already prints (alpha = episode_runtime / deadline; alpha > 1 implies
      a deadline miss).
  (b) The tensorboard event files under ``--runs-dir``, in case a future R3
      patch logs scalars like ``r3/framework_overhead`` or
      ``r3/deadline_miss_rate``.
  (c) Estimate ``framework_overhead_pct`` from the per-episode timing data
      already in the log: episode lines look like::

          episode: N, frame: F, fps: X, episode_length: L, returns: R

      Total training wall-clock is provided via ``--wall-clock-s``. The
      coordinator overhead is the difference between (1) the wall-clock
      total and (2) the productive env-step time inferred from
      ``episode_length / fps`` summed over all episodes; expressed as a
      percentage of the wall-clock total.

If NEITHER metric can be extracted, the JSON is still written -- with
``framework_overhead_pct=null`` and ``mean_deadline_miss_rate=null`` and
``parse_status="failed"`` plus a ``parse_notes`` string explaining what was
looked for. The aggregator handles missing values gracefully.

Stdlib + numpy + (optional) tensorboard. Never crashes on missing files,
malformed log lines, or empty tensorboard dirs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Regex patterns used by the log-file parser
# -----------------------------------------------------------------------------

# Explicit metric lines a future R3 wrapper might emit.
_RE_OVERHEAD_PCT = re.compile(
    r"framework[_\s]overhead[\s=:]+(?P<v>[-+]?\d*\.?\d+)\s*%?",
    re.IGNORECASE,
)
_RE_MISS_RATE = re.compile(
    r"deadline[_\s]miss[_\s]rate[\s=:]+(?P<v>[-+]?\d*\.?\d+)\s*%?",
    re.IGNORECASE,
)

# RuntimeCoordinator already prints "RUNTIME COORDINATOR: ALPHA: x, BETA: y".
# alpha = episode_runtime / deadline, so alpha > 1 implies a deadline miss.
_RE_COORD_ALPHA = re.compile(
    r"RUNTIME\s+COORDINATOR\s*:\s*ALPHA\s*:\s*(?P<alpha>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"\s*,\s*BETA\s*:\s*(?P<beta>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)

# ALL's _log_training_episode prints:
#   "episode: N, frame: F, fps: X, episode_length: L, returns: R"
_RE_EPISODE = re.compile(
    r"episode:\s*(?P<ep>\d+)\s*,\s*"
    r"frame:\s*(?P<frame>\d+)\s*,\s*"
    r"fps:\s*(?P<fps>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
    r"episode_length:\s*(?P<len>\d+)\s*,\s*"
    r"returns:\s*(?P<ret>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

# Tensorboard scalar tags an R3 logger might emit. Order is preference.
_TB_OVERHEAD_TAGS: Tuple[str, ...] = (
    "r3/framework_overhead",
    "r3/framework_overhead_pct",
    "info/framework_overhead",
    "info/r3/framework_overhead",
)
_TB_MISS_RATE_TAGS: Tuple[str, ...] = (
    "r3/deadline_miss_rate",
    "r3/mean_deadline_miss_rate",
    "info/deadline_miss_rate",
    "info/r3/deadline_miss_rate",
)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", type=Path, required=True,
                   help="Tensorboard run dir (the --logdir passed to all-classic).")
    p.add_argument("--log-file", type=Path, required=True,
                   help="Captured stdout/stderr from the R3 training run.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSON path (e.g. r3_native_metrics.json).")
    p.add_argument("--env", default="CartPole-v1",
                   help="Env name to record in the JSON.")
    p.add_argument("--agent", default="dqn",
                   help="Agent name to record in the JSON.")
    p.add_argument("--frames", type=int, default=None,
                   help="Frame budget (recorded as step_count if --runs-dir "
                        "yields no better signal).")
    p.add_argument("--wall-clock-s", type=float, default=None,
                   help="Wall-clock seconds of the training run, measured by "
                        "the runner script. Used to estimate overhead from the "
                        "per-episode fps lines if no explicit overhead line is "
                        "logged.")
    return p.parse_args()


# -----------------------------------------------------------------------------
# (a) Log-file parser
# -----------------------------------------------------------------------------

def parse_log_file(
    log_path: Path,
) -> Tuple[Optional[float], Optional[float], List[Dict[str, float]],
          List[Dict[str, float]], int, str]:
    """Parse the R3 stdout log.

    Returns:
        explicit_overhead_pct, explicit_miss_rate,
        coord_records (list of {alpha, beta}),
        episode_records (list of {ep, frame, fps, len, ret}),
        step_count (max frame seen, 0 if none),
        notes (one-line summary of what was found / why something was skipped).
    """
    notes_parts: List[str] = []

    if not log_path.exists():
        notes_parts.append(f"log_file_missing={log_path}")
        return None, None, [], [], 0, "; ".join(notes_parts)

    try:
        text = log_path.read_text(errors="replace")
    except Exception as exc:
        notes_parts.append(f"log_read_error={type(exc).__name__}:{exc}")
        return None, None, [], [], 0, "; ".join(notes_parts)

    # Look for the *last* explicit overhead / miss-rate line in the log
    # (training might log a running estimate plus a final one).
    explicit_overhead_pct: Optional[float] = None
    explicit_miss_rate: Optional[float] = None

    for m in _RE_OVERHEAD_PCT.finditer(text):
        try:
            explicit_overhead_pct = float(m.group("v"))
        except ValueError:
            continue

    for m in _RE_MISS_RATE.finditer(text):
        try:
            v = float(m.group("v"))
        except ValueError:
            continue
        # If the line had a trailing % sign, normalize to a fraction.
        if m.group(0).rstrip().endswith("%") and v > 1.0:
            v = v / 100.0
        explicit_miss_rate = v

    coord_records: List[Dict[str, float]] = []
    for m in _RE_COORD_ALPHA.finditer(text):
        try:
            alpha = float(m.group("alpha"))
            beta = float(m.group("beta"))
        except ValueError:
            continue
        coord_records.append({"alpha": alpha, "beta": beta})

    episode_records: List[Dict[str, float]] = []
    max_frame = 0
    for m in _RE_EPISODE.finditer(text):
        try:
            rec = {
                "ep": int(m.group("ep")),
                "frame": int(m.group("frame")),
                "fps": float(m.group("fps")),
                "len": int(m.group("len")),
                "ret": float(m.group("ret")),
            }
        except ValueError:
            continue
        episode_records.append(rec)
        if rec["frame"] > max_frame:
            max_frame = rec["frame"]

    notes_parts.append(f"explicit_overhead={'yes' if explicit_overhead_pct is not None else 'no'}")
    notes_parts.append(f"explicit_miss_rate={'yes' if explicit_miss_rate is not None else 'no'}")
    notes_parts.append(f"coord_lines={len(coord_records)}")
    notes_parts.append(f"episode_lines={len(episode_records)}")

    return (
        explicit_overhead_pct,
        explicit_miss_rate,
        coord_records,
        episode_records,
        max_frame,
        "; ".join(notes_parts),
    )


# -----------------------------------------------------------------------------
# (b) Tensorboard parser (best-effort; tensorboard import is optional)
# -----------------------------------------------------------------------------

def parse_tensorboard(
    runs_dir: Path,
) -> Tuple[Optional[float], Optional[float], int, str]:
    """Look for R3 native-metric scalars in tensorboard event files.

    Returns:
        overhead_pct, miss_rate, step_count, notes.

    Best-effort: missing tensorboard module, missing/empty runs_dir, missing
    tags all yield (None, None, 0, "<reason>") rather than raising.
    """
    if not runs_dir.exists():
        return None, None, 0, f"runs_dir_missing={runs_dir}"

    event_files = sorted(runs_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        return None, None, 0, f"no_event_files_under={runs_dir}"

    try:
        from tensorboard.backend.event_processing.event_accumulator import (  # type: ignore
            EventAccumulator,
        )
    except Exception as exc:
        return (
            None,
            None,
            0,
            f"tensorboard_import_failed={type(exc).__name__}:{exc}",
        )

    overhead_pct: Optional[float] = None
    miss_rate: Optional[float] = None
    step_count = 0
    seen_tags: List[str] = []
    failed_files: List[str] = []

    for ef in event_files:
        try:
            acc = EventAccumulator(str(ef))
            acc.Reload()
        except Exception as exc:
            failed_files.append(f"{ef.name}:{type(exc).__name__}")
            continue

        scalar_tags = list(acc.Tags().get("scalars", []))
        seen_tags.extend(scalar_tags)

        # Track max step across any scalar to estimate frame budget.
        for tag in scalar_tags:
            try:
                events = acc.Scalars(tag)
            except Exception:
                continue
            if events:
                last_step = int(events[-1].step)
                if last_step > step_count:
                    step_count = last_step

        for tag in _TB_OVERHEAD_TAGS:
            if tag in scalar_tags and overhead_pct is None:
                try:
                    events = acc.Scalars(tag)
                    if events:
                        overhead_pct = float(events[-1].value)
                except Exception:
                    pass

        for tag in _TB_MISS_RATE_TAGS:
            if tag in scalar_tags and miss_rate is None:
                try:
                    events = acc.Scalars(tag)
                    if events:
                        miss_rate = float(events[-1].value)
                except Exception:
                    pass

    note_parts = [
        f"event_files={len(event_files)}",
        f"unique_tags={len(set(seen_tags))}",
        f"overhead_tag={'hit' if overhead_pct is not None else 'miss'}",
        f"miss_tag={'hit' if miss_rate is not None else 'miss'}",
    ]
    if failed_files:
        note_parts.append(f"failed_files={','.join(failed_files[:3])}")
    return overhead_pct, miss_rate, step_count, "; ".join(note_parts)


# -----------------------------------------------------------------------------
# (c) Estimators from log-only data
# -----------------------------------------------------------------------------

def estimate_overhead_pct(
    episode_records: List[Dict[str, float]],
    wall_clock_s: Optional[float],
) -> Tuple[Optional[float], str]:
    """Estimate framework_overhead_pct from per-episode fps + wall-clock.

    Productive env-step time per episode = episode_length / fps. Sum over
    episodes -> productive_s. Overhead = wall_clock_s - productive_s.

    Returns (overhead_pct or None, notes).
    """
    if wall_clock_s is None or wall_clock_s <= 0:
        return None, "wall_clock_unavailable"
    if not episode_records:
        return None, "no_episode_records_for_estimation"

    productive_s = 0.0
    n_used = 0
    for r in episode_records:
        fps = r.get("fps", 0.0)
        ln = r.get("len", 0)
        if fps and fps > 0 and ln > 0:
            productive_s += ln / fps
            n_used += 1

    if n_used == 0:
        return None, "no_valid_fps_records"

    overhead_s = max(wall_clock_s - productive_s, 0.0)
    overhead_pct = 100.0 * overhead_s / wall_clock_s
    return (
        float(overhead_pct),
        f"estimated_from_log; episodes_used={n_used}; "
        f"productive_s={productive_s:.2f}; wall_s={wall_clock_s:.2f}",
    )


def estimate_miss_rate(
    coord_records: List[Dict[str, float]],
) -> Tuple[Optional[float], str]:
    """Estimate mean_deadline_miss_rate from RuntimeCoordinator alpha lines.

    alpha = episode_runtime / deadline -> alpha > 1 implies a miss.
    Returns (miss_rate or None, notes).
    """
    if not coord_records:
        return None, "no_coord_records"
    alphas = [r.get("alpha", 0.0) for r in coord_records]
    n = len(alphas)
    n_miss = sum(1 for a in alphas if a > 1.0)
    miss_rate = float(n_miss) / float(n)
    return (
        miss_rate,
        f"estimated_from_coord_alpha; n_episodes={n}; n_miss={n_miss}",
    )


# -----------------------------------------------------------------------------
# Main glue
# -----------------------------------------------------------------------------

def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> int:
    args = parse_args()

    notes: List[str] = []

    # (a) Log file
    (
        log_overhead,
        log_miss,
        coord_records,
        episode_records,
        log_max_frame,
        log_notes,
    ) = parse_log_file(args.log_file)
    notes.append(f"log[{log_notes}]")

    # (b) Tensorboard
    tb_overhead, tb_miss, tb_step_count, tb_notes = parse_tensorboard(args.runs_dir)
    notes.append(f"tb[{tb_notes}]")

    # Resolve metrics with the preference order: log-explicit > tb > estimate.
    overhead_pct: Optional[float] = None
    miss_rate: Optional[float] = None
    overhead_source = "none"
    miss_source = "none"

    if log_overhead is not None:
        overhead_pct = log_overhead
        overhead_source = "log_explicit"
    elif tb_overhead is not None:
        overhead_pct = tb_overhead
        overhead_source = "tensorboard"
    else:
        est, est_notes = estimate_overhead_pct(episode_records, args.wall_clock_s)
        notes.append(f"overhead_est[{est_notes}]")
        if est is not None:
            overhead_pct = est
            overhead_source = "estimated"

    if log_miss is not None:
        miss_rate = log_miss
        miss_source = "log_explicit"
    elif tb_miss is not None:
        miss_rate = tb_miss
        miss_source = "tensorboard"
    else:
        est, est_notes = estimate_miss_rate(coord_records)
        notes.append(f"miss_est[{est_notes}]")
        if est is not None:
            miss_rate = est
            miss_source = "estimated_from_coord"

    # parse_status: ok / partial / failed
    if overhead_pct is not None and miss_rate is not None:
        parse_status = "ok"
    elif overhead_pct is not None or miss_rate is not None:
        parse_status = "partial"
    else:
        parse_status = "failed"

    # step_count: prefer args.frames if provided, else log_max_frame, else tb.
    if args.frames is not None:
        step_count = int(args.frames)
    elif log_max_frame > 0:
        step_count = int(log_max_frame)
    elif tb_step_count > 0:
        step_count = int(tb_step_count)
    else:
        step_count = 0

    payload: Dict[str, object] = {
        "framework_overhead_pct": overhead_pct,
        "mean_deadline_miss_rate": miss_rate,
        "env": str(args.env),
        "agent": str(args.agent),
        "step_count": int(step_count),
        "wall_clock_s": (
            float(args.wall_clock_s) if args.wall_clock_s is not None else None
        ),
        "parse_status": parse_status,
        "parse_notes": (
            f"overhead_source={overhead_source}; miss_source={miss_source}; "
            + "; ".join(notes)
        ),
        "log_file": str(args.log_file),
        "tensorboard_runs_dir": str(args.runs_dir),
    }

    try:
        write_json(args.out, payload)
    except Exception as exc:
        # If we can't even write the JSON, fall back to stderr.
        print(f"[parse_r3_metrics] FATAL: failed to write {args.out}: {exc}",
              file=sys.stderr)
        return 2

    print(f"[parse_r3_metrics] wrote {args.out} (parse_status={parse_status}, "
          f"overhead_pct={overhead_pct}, miss_rate={miss_rate})")
    # Always exit 0 -- a "failed" parse_status is a data-shape problem the
    # aggregator handles, not a script crash.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
