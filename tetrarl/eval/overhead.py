"""Per-component overhead profiler (Week 8 Task 1).

Instruments the TetraRL pipeline so the paper can report wall-clock
distribution + python memory delta for each of the 7 sub-components
(Table 5 candidate). Six of them live inside ``framework.step()``;
the seventh, ``replay_buffer_add``, is timed by the training script.

Design choices:
  * ``time.perf_counter_ns()`` for monotonic wall clock (highest
    resolution; not affected by NTP/system-time updates).
  * ``tracemalloc`` for python-side memory deltas. Started lazily on
    the first ``ComponentTimer.__enter__`` when ``track_memory=True``,
    then kept running for the profiler's lifetime so per-sample
    deltas are comparable.
  * RSS deltas via ``psutil`` are an *additional* signal; they capture
    OS-level allocations (e.g. CUDA arenas, malloc growth) that
    tracemalloc would miss. Reported as ``rss_mb`` (max delta).
  * Public method ``_record_sample`` lets tests inject deterministic
    distributions without monkeypatching ``time.perf_counter_ns``.

The profiler is intentionally a passive aggregator; the framework
decides which components to wrap. This keeps the profiler reusable
for the training script's ``replay_buffer_add`` row.
"""
from __future__ import annotations

import csv
import tracemalloc
from pathlib import Path
from time import perf_counter_ns
from typing import Optional

import numpy as np
import psutil


class ComponentTimer:
    """Context manager that records elapsed ns + memory delta for a component."""

    __slots__ = ("_profiler", "_name", "_t0", "_mem0", "_rss0")

    def __init__(self, profiler: "OverheadProfiler", name: str):
        self._profiler = profiler
        self._name = name
        self._t0 = 0
        self._mem0 = 0
        self._rss0 = 0

    def __enter__(self) -> "ComponentTimer":
        prof = self._profiler
        if prof._track_memory:
            # Lazy-start tracemalloc so importing the module is cheap.
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            self._mem0 = tracemalloc.get_traced_memory()[0]
            self._rss0 = prof._proc.memory_info().rss
        self._t0 = perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed_ns = perf_counter_ns() - self._t0
        prof = self._profiler
        mem_delta = 0
        rss_delta = 0
        if prof._track_memory:
            mem_now = tracemalloc.get_traced_memory()[0]
            mem_delta = mem_now - self._mem0
            rss_delta = prof._proc.memory_info().rss - self._rss0
        prof._record_sample(self._name, elapsed_ns=elapsed_ns,
                            mem_delta_bytes=mem_delta, rss_delta_bytes=rss_delta)


class OverheadProfiler:
    """Aggregates per-component timing + memory samples across framework steps."""

    def __init__(self, track_memory: bool = True):
        self._track_memory = bool(track_memory)
        self._samples_ns: dict[str, list[int]] = {}
        self._samples_mem: dict[str, list[int]] = {}
        self._samples_rss: dict[str, list[int]] = {}
        self._rows: list[dict] = []
        self._step_idx: int = 0
        # Avoid importing psutil cost on hot path when track_memory=False.
        self._proc: Optional[psutil.Process] = (
            psutil.Process() if self._track_memory else None
        )

    # -- ctx mgr factory ----------------------------------------------------

    def time(self, name: str) -> ComponentTimer:
        return ComponentTimer(self, name)

    # -- step bookkeeping ---------------------------------------------------

    def step_marker(self) -> None:
        self._step_idx += 1

    # -- raw sample access --------------------------------------------------

    def samples_ns(self, name: str) -> list[int]:
        return list(self._samples_ns.get(name, []))

    def rows(self) -> list[dict]:
        # Return a shallow copy so callers can't mutate internal state.
        return [dict(r) for r in self._rows]

    # -- summary stats ------------------------------------------------------

    def summarize(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name, ns_list in self._samples_ns.items():
            if not ns_list:
                continue
            ns_arr = np.asarray(ns_list, dtype=np.int64)
            ms_arr = ns_arr.astype(np.float64) / 1_000_000.0
            mem_list = self._samples_mem.get(name, [])
            rss_list = self._samples_rss.get(name, [])
            if self._track_memory and mem_list:
                mem_mb = float(max(mem_list)) / (1024.0 * 1024.0)
            else:
                mem_mb = 0.0
            if self._track_memory and rss_list:
                rss_mb = float(max(rss_list)) / (1024.0 * 1024.0)
            else:
                rss_mb = 0.0
            out[name] = {
                "mean_ms": float(ms_arr.mean()),
                "p50_ms": float(np.percentile(ms_arr, 50)),
                "p99_ms": float(np.percentile(ms_arr, 99)),
                "mem_mb": mem_mb,
                "rss_mb": rss_mb,
                "n_samples": int(ns_arr.size),
            }
        return out

    # -- export -------------------------------------------------------------

    def to_csv(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["step", "component", "elapsed_ns",
                      "mem_delta_bytes", "rss_delta_bytes"]
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in self._rows:
                w.writerow({k: row.get(k, 0) for k in fieldnames})

    def to_markdown(self) -> str:
        summary = self.summarize()
        header = ("| component | mean_ms | p50_ms | p99_ms | mem_mb |"
                  " rss_mb | n_samples |")
        sep = "|---|---|---|---|---|---|---|"
        lines = [header, sep]
        if not summary:
            return "\n".join(lines) + "\n"
        for name in sorted(summary.keys()):
            s = summary[name]
            lines.append(
                f"| {name} | {s['mean_ms']:.4f} | {s['p50_ms']:.4f} |"
                f" {s['p99_ms']:.4f} | {s['mem_mb']:.4f} |"
                f" {s['rss_mb']:.4f} | {int(s['n_samples'])} |"
            )
        return "\n".join(lines) + "\n"

    # -- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        self._samples_ns.clear()
        self._samples_mem.clear()
        self._samples_rss.clear()
        self._rows.clear()
        self._step_idx = 0

    # -- internal hook for tests + ComponentTimer ---------------------------

    def _record_sample(
        self,
        name: str,
        elapsed_ns: int,
        mem_delta_bytes: int,
        rss_delta_bytes: int = 0,
    ) -> None:
        self._samples_ns.setdefault(name, []).append(int(elapsed_ns))
        self._samples_mem.setdefault(name, []).append(int(mem_delta_bytes))
        self._samples_rss.setdefault(name, []).append(int(rss_delta_bytes))
        self._rows.append({
            "step": self._step_idx,
            "component": name,
            "elapsed_ns": int(elapsed_ns),
            "mem_delta_bytes": int(mem_delta_bytes),
            "rss_delta_bytes": int(rss_delta_bytes),
        })
