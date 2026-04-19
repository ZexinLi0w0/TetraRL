"""ConcurrentDecisionLoop — DVFO-style thinking-while-moving for TetraRL.

DVFO (Zhang TMC 2023) overlaps the resource manager's DVFS decision with
the RL arbiter's forward pass: while the env executes the previous action
and the actor computes its next forward, a background thread computes
the upcoming DVFS frequency target. The decision applied at step ``t``
is the one computed during step ``t-1``; on step 0, no decision is yet
available, so ``apply_latest()`` falls back (or returns ``None``) without
crashing. This masks the DVFS decision overhead.

Threading model:
- A single daemon worker thread drains a single-slot ``queue.Queue``.
- ``submit`` is non-blocking: if the slot is occupied it drops the
  pending telemetry first (freshness > completeness — we want the most
  recent signal, never a backlog).
- A ``threading.Lock`` serialises ``submit`` calls AND the last-result
  swap so concurrent producers cannot race.
- ``decide_dvfs`` exceptions inside the worker are swallowed so a bad
  telemetry sample cannot deadlock the main loop. ``shutdown`` joins
  the thread and is idempotent; ``submit`` after ``shutdown`` is a
  safe no-op.
"""
from __future__ import annotations

import queue
import threading
from typing import Any, Optional

from tetrarl.morl.native.override import HardwareTelemetry


class ConcurrentDecisionLoop:
    """Background DVFS-decision loop for the TetraRL framework."""

    def __init__(
        self,
        resource_manager: Any,
        dvfs_controller: Any,
        n_levels: int,
        fallback_idx: int = -1,
    ):
        self._resource_manager = resource_manager
        self._dvfs_controller = dvfs_controller
        self._n_levels = int(n_levels)
        self._fallback_idx = int(fallback_idx)

        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._last_result: Optional[int] = None
        self._shutdown_done = False

        self._thread = threading.Thread(
            target=self._worker_loop,
            name="tetrarl-concurrent-dvfs",
            daemon=True,
        )
        self._thread.start()

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                telemetry = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if telemetry is None:
                break  # shutdown sentinel
            try:
                idx = self._resource_manager.decide_dvfs(
                    telemetry, self._n_levels
                )
            except Exception:
                # Background failures must not deadlock the main loop.
                continue
            with self._lock:
                self._last_result = int(idx)

    def submit(self, telemetry: HardwareTelemetry) -> None:
        if self._stop.is_set() or self._shutdown_done:
            return
        with self._lock:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(telemetry)
            except queue.Full:
                pass

    def latest(self) -> Optional[int]:
        with self._lock:
            return self._last_result

    def apply_latest(self) -> Optional[int]:
        with self._lock:
            idx = self._last_result
        if idx is None:
            if self._fallback_idx >= 0:
                self._dvfs_controller.set_freq(gpu_idx=self._fallback_idx)
                return self._fallback_idx
            return None
        self._dvfs_controller.set_freq(gpu_idx=idx)
        return idx

    def shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self._stop.set()
        # Wake the worker if it's blocked on get().
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        self._thread.join(timeout=2.0)
