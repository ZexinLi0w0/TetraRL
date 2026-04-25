"""P15 Atari memory headroom probe — Orin Nano (8GB shared CPU+GPU).

For each requested replay capacity:

1. Read MemAvailable BEFORE.
2. Allocate ``(capacity, frame_stack, frame_h, frame_w)`` uint8 obs + next_obs.
3. Allocate Nature-DQN small CNN, move to ``--device``.
4. Run a single forward pass with batch_size=32.
5. Read MemAvailable AFTER.
6. Conservative ``fits_in_8gb`` if MemAvailable_after > 200 MB.
7. Free buffers + CNN + ``torch.cuda.empty_cache()``.

Writes a markdown table report to ``--out-md``. Falls back to
``psutil.virtual_memory().available`` when ``/proc/meminfo`` is absent (e.g.
when smoke-running on macOS) and prefixes the report with a warning.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Atari replay-buffer memory headroom probe")
    p.add_argument("--capacities", default="1000,5000,10000,50000")
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--frame-h", type=int, default=84)
    p.add_argument("--frame-w", type=int, default=84)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--out-md", required=True)
    return p.parse_args()


def _read_meminfo_kb() -> Optional[int]:
    """Return MemAvailable in kB from /proc/meminfo, or None if absent."""
    path = "/proc/meminfo"
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    return int(parts[1])  # value in kB
    except Exception:
        return None
    return None


def _mem_available_mb() -> tuple[float, bool]:
    """Return (MemAvailable in MB, used_proc_meminfo flag)."""
    kb = _read_meminfo_kb()
    if kb is not None:
        return float(kb) / 1024.0, True
    try:
        import psutil

        return float(psutil.virtual_memory().available) / (1024.0 * 1024.0), False
    except Exception:
        return 0.0, False


class _NatureDQN(nn.Module):
    def __init__(self, frame_stack: int, n_actions: int = 4) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(frame_stack, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
        )
        # Compute flatten dim for 84x84 -> ((84-8)//4 + 1)=20 -> ((20-4)//2+1)=9 -> ((9-3)//1+1)=7 -> 64*7*7=3136
        self.flatten_dim = 64 * 7 * 7
        self.fc = nn.Sequential(nn.Linear(self.flatten_dim, 512), nn.ReLU(), nn.Linear(512, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


def _probe_capacity(
    capacity: int, frame_stack: int, frame_h: int, frame_w: int, device: str
) -> dict:
    # Lazy-allocation gotcha: np.zeros uses calloc on Linux, which maps every
    # page of the array to a single shared zero page until each page is
    # actually written. Touching one byte only commits one 4 KB page, so the
    # buffer's nominal RAM never appears in MemAvailable. We must .fill()
    # both buffers below to force every page resident before measuring.
    bytes_per_buffer = capacity * frame_stack * frame_h * frame_w
    replay_alloc_mb_estimate = (2.0 * bytes_per_buffer) / (1024.0 * 1024.0)

    mem_before_mb, used_proc = _mem_available_mb()

    obs = np.zeros((capacity, frame_stack, frame_h, frame_w), dtype=np.uint8)
    next_obs = np.zeros((capacity, frame_stack, frame_h, frame_w), dtype=np.uint8)
    # Force every page resident — np.zeros uses calloc on Linux which maps all
    # pages to a single shared zero page until written. Without .fill() the
    # replay buffer never actually consumes its nominal RAM and MemAvailable
    # barely moves, masking the real headroom on the Nano.
    obs.fill(1)
    next_obs.fill(1)

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    cnn = _NatureDQN(frame_stack=frame_stack, n_actions=4).to(dev)

    # Forward pass: batch_size=32 random frames as float32 in [0,1].
    batch = torch.from_numpy(obs[:32].astype(np.float32) / 255.0).to(dev)
    with torch.no_grad():
        _ = cnn(batch)
    if dev.type == "cuda":
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    import time
    time.sleep(0.1)
    mem_after_mb, _ = _mem_available_mb()
    delta_mb = mem_before_mb - mem_after_mb
    fits_in_8gb = mem_after_mb > 200.0

    # Free.
    del obs, next_obs, cnn, batch
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return {
        "capacity": int(capacity),
        "replay_alloc_mb_estimate": float(replay_alloc_mb_estimate),
        "mem_avail_before_mb": float(mem_before_mb),
        "mem_avail_after_mb": float(mem_after_mb),
        "delta_mb": float(delta_mb),
        "fits_in_8gb": bool(fits_in_8gb),
        "used_proc_meminfo": bool(used_proc),
    }


def main() -> int:
    args = _parse_args()
    out_md = Path(args.out_md).expanduser().resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)

    try:
        capacities = [int(x.strip()) for x in args.capacities.split(",") if x.strip()]
    except Exception as exc:
        print(f"failed to parse --capacities: {exc!r}", file=sys.stderr)
        return 1
    if not capacities:
        print("no capacities specified", file=sys.stderr)
        return 1

    rows: list[dict] = []
    any_proc = False
    any_fallback = False
    for cap in capacities:
        try:
            row = _probe_capacity(cap, args.frame_stack, args.frame_h, args.frame_w, args.device)
        except MemoryError as exc:
            row = {
                "capacity": int(cap),
                "replay_alloc_mb_estimate": float(
                    2.0 * cap * args.frame_stack * args.frame_h * args.frame_w / (1024.0 * 1024.0)
                ),
                "mem_avail_before_mb": 0.0,
                "mem_avail_after_mb": 0.0,
                "delta_mb": 0.0,
                "fits_in_8gb": False,
                "used_proc_meminfo": _read_meminfo_kb() is not None,
                "error": f"MemoryError: {exc}",
            }
        rows.append(row)
        if row.get("used_proc_meminfo"):
            any_proc = True
        else:
            any_fallback = True

    # Recommendation: largest capacity with ≥1 GB headroom (after CNN forward).
    GB = 1024.0
    fits_with_1gb = [r for r in rows if r["mem_avail_after_mb"] >= GB]
    if fits_with_1gb:
        recommended = max(r["capacity"] for r in fits_with_1gb)
        rec_line = (
            f"Recommendation: pick replay_capacity={recommended} "
            f"(largest capacity with >=1 GB headroom remaining)."
        )
    else:
        rec_line = (
            "Recommendation: NONE of the probed capacities leave >=1 GB headroom; "
            "consider smaller capacities or uint8 frame compression."
        )

    lines: list[str] = []
    lines.append("# P15 Atari Replay-Buffer Memory Headroom Probe\n")
    if any_fallback and not any_proc:
        lines.append(
            "> WARNING: /proc/meminfo not present — fell back to "
            "`psutil.virtual_memory().available` for ALL rows. Numbers are "
            "from a non-Linux host (likely macOS) and do not reflect Orin Nano headroom.\n"
        )
    elif any_fallback and any_proc:
        lines.append(
            "> WARNING: /proc/meminfo missing on some rows — psutil fallback used where flagged.\n"
        )
    lines.append(
        f"Config: frame_stack={args.frame_stack}, frame_h={args.frame_h}, "
        f"frame_w={args.frame_w}, device={args.device}\n"
    )
    lines.append(
        "| capacity | replay_alloc_mb_estimate | mem_avail_before_mb | mem_avail_after_mb | delta_mb | fits_in_8gb |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        lines.append(
            f"| {r['capacity']} | {r['replay_alloc_mb_estimate']:.1f} | "
            f"{r['mem_avail_before_mb']:.1f} | {r['mem_avail_after_mb']:.1f} | "
            f"{r['delta_mb']:.1f} | {r['fits_in_8gb']} |"
        )
    lines.append("")
    lines.append(rec_line)
    lines.append("")

    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
