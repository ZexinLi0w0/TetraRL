"""Tests for tetrarl/sys/concurrent.py — DVFO thinking-while-moving loop.

The ConcurrentDecisionLoop overlaps the resource manager's DVFS decision
computation with the RL arbiter's forward pass. Track A (arbiter.act) is
foreground and gates the env step; Track B (decide_dvfs) runs in a
background thread and is permitted to lag by 1 step. The decision applied
at step ``t`` is the one computed during step ``t-1``; on step 0, no
decision is yet available and ``apply_latest`` returns ``None``.

Per spec:
- single-slot queue (drop oldest under fast submit — freshness > completeness)
- thread-safe ``submit`` (Lock)
- idempotent ``shutdown`` that joins the worker
- background exceptions never deadlock the main loop
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import pytest

from tetrarl.morl.native.override import HardwareTelemetry
from tetrarl.sys.concurrent import ConcurrentDecisionLoop


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class _StubResourceManager:
    """Returns a deterministic dvfs idx; counts decide_dvfs calls."""

    def __init__(self, idx: int = 3, sleep_s: float = 0.0):
        self.idx = int(idx)
        self.sleep_s = float(sleep_s)
        self.calls: list[HardwareTelemetry] = []
        self._lock = threading.Lock()

    def decide_dvfs(self, telemetry: HardwareTelemetry, n_levels: int) -> int:
        if self.sleep_s > 0:
            time.sleep(self.sleep_s)
        with self._lock:
            self.calls.append(telemetry)
        return self.idx


class _RaisingResourceManager:
    def decide_dvfs(self, telemetry: HardwareTelemetry, n_levels: int) -> int:
        raise RuntimeError("synthetic ResourceManager failure")


class _StubDVFSController:
    """Records every set_freq call and returns a fake current state."""

    def __init__(self, n_levels: int = 14):
        self._n_levels = int(n_levels)
        self.set_freq_calls: list[Optional[int]] = []
        self._lock = threading.Lock()

    def available_frequencies(self) -> dict:
        return {"cpu": list(range(self._n_levels)), "gpu": list(range(self._n_levels))}

    def set_freq(self, cpu_idx: Optional[int] = None, gpu_idx: Optional[int] = None):
        with self._lock:
            self.set_freq_calls.append(gpu_idx)
        return None


def _hw(latency: float = 1.0, energy: float = 100.0, mem: float = 0.1) -> HardwareTelemetry:
    return HardwareTelemetry(
        latency_ema_ms=latency, energy_remaining_j=energy, memory_util=mem
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_submit_then_latest_returns_decision():
    """Happy path: submit telemetry, wait for worker, latest() yields the idx."""
    rm = _StubResourceManager(idx=5)
    dvfs = _StubDVFSController()
    loop = ConcurrentDecisionLoop(
        resource_manager=rm, dvfs_controller=dvfs, n_levels=14, fallback_idx=0
    )
    try:
        loop.submit(_hw())
        deadline = time.time() + 1.0
        while loop.latest() is None and time.time() < deadline:
            time.sleep(0.005)
        assert loop.latest() == 5
    finally:
        loop.shutdown()


def test_first_call_to_latest_returns_none_before_any_submit():
    rm = _StubResourceManager(idx=5)
    dvfs = _StubDVFSController()
    loop = ConcurrentDecisionLoop(
        resource_manager=rm, dvfs_controller=dvfs, n_levels=14, fallback_idx=-1
    )
    try:
        assert loop.latest() is None
        # apply_latest must also be safe before any submit — returns None.
        assert loop.apply_latest() is None
        # No DVFS write should have happened.
        assert dvfs.set_freq_calls == []
    finally:
        loop.shutdown()


def test_apply_latest_invokes_dvfs_set_freq_with_gpu_idx():
    rm = _StubResourceManager(idx=7)
    dvfs = _StubDVFSController()
    loop = ConcurrentDecisionLoop(
        resource_manager=rm, dvfs_controller=dvfs, n_levels=14, fallback_idx=0
    )
    try:
        loop.submit(_hw())
        deadline = time.time() + 1.0
        while loop.latest() is None and time.time() < deadline:
            time.sleep(0.005)
        applied = loop.apply_latest()
        assert applied == 7
        assert dvfs.set_freq_calls == [7]
    finally:
        loop.shutdown()


def test_shutdown_is_idempotent_and_joins_thread():
    rm = _StubResourceManager(idx=2)
    dvfs = _StubDVFSController()
    loop = ConcurrentDecisionLoop(
        resource_manager=rm, dvfs_controller=dvfs, n_levels=14, fallback_idx=0
    )
    loop.shutdown()
    loop.shutdown()  # second call must not raise / deadlock
    # Submitting after shutdown should be a no-op or safe (no exceptions).
    loop.submit(_hw())  # safe to ignore


def test_oldest_decision_dropped_under_fast_submit():
    """Single-slot queue: 100 fast submits never grow the queue beyond 1."""
    # Slow worker so submits queue up.
    rm = _StubResourceManager(idx=1, sleep_s=0.005)
    dvfs = _StubDVFSController()
    loop = ConcurrentDecisionLoop(
        resource_manager=rm, dvfs_controller=dvfs, n_levels=14, fallback_idx=0
    )
    try:
        for _ in range(100):
            loop.submit(_hw())
            # The internal queue must never exceed maxsize=1.
            assert loop._queue.qsize() <= 1
        # Drain.
        deadline = time.time() + 2.0
        while loop._queue.qsize() > 0 and time.time() < deadline:
            time.sleep(0.005)
        # Worker should have processed FAR fewer than 100 (most dropped).
        assert len(rm.calls) < 100
    finally:
        loop.shutdown()


def test_thread_safety_under_concurrent_submit():
    """4 threads pounding submit() for 0.5s — no exceptions, no deadlocks."""
    rm = _StubResourceManager(idx=4, sleep_s=0.001)
    dvfs = _StubDVFSController()
    loop = ConcurrentDecisionLoop(
        resource_manager=rm, dvfs_controller=dvfs, n_levels=14, fallback_idx=0
    )
    errors: list[BaseException] = []
    stop = threading.Event()

    def worker():
        try:
            while not stop.is_set():
                loop.submit(_hw())
        except BaseException as e:  # pragma: no cover - test instrumentation
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.5)
    stop.set()
    for t in threads:
        t.join(timeout=2.0)
        assert not t.is_alive(), "worker thread did not exit cleanly"
    loop.shutdown()
    assert errors == [], f"worker exceptions: {errors!r}"


def test_background_exception_does_not_deadlock_main_loop():
    """If decide_dvfs raises, the worker keeps running and shutdown still joins."""
    rm = _RaisingResourceManager()
    dvfs = _StubDVFSController()
    loop = ConcurrentDecisionLoop(
        resource_manager=rm, dvfs_controller=dvfs, n_levels=14, fallback_idx=-1
    )
    try:
        loop.submit(_hw())
        # Give the worker a chance to consume + raise.
        time.sleep(0.1)
        # latest() should return None (no successful decision was produced).
        assert loop.latest() is None
        # Submitting again must not raise from the main thread.
        loop.submit(_hw())
        time.sleep(0.05)
    finally:
        loop.shutdown()  # must join cleanly even after exceptions


def test_apply_latest_returns_fallback_when_no_decision_and_fallback_set():
    """When fallback_idx >= 0 and no decision yet, apply_latest applies fallback."""
    rm = _StubResourceManager(idx=5)
    dvfs = _StubDVFSController(n_levels=14)
    loop = ConcurrentDecisionLoop(
        resource_manager=rm, dvfs_controller=dvfs, n_levels=14, fallback_idx=0
    )
    try:
        # No submit yet — apply_latest should fall back to fallback_idx.
        applied = loop.apply_latest()
        assert applied == 0
        assert dvfs.set_freq_calls == [0]
    finally:
        loop.shutdown()


def test_latest_reflects_freshest_decision_after_settling():
    """After fast submits + settling, latest() reflects the most recent computation."""
    rm = _StubResourceManager(idx=9, sleep_s=0.002)
    dvfs = _StubDVFSController()
    loop = ConcurrentDecisionLoop(
        resource_manager=rm, dvfs_controller=dvfs, n_levels=14, fallback_idx=0
    )
    try:
        for _ in range(20):
            loop.submit(_hw())
        deadline = time.time() + 2.0
        while loop.latest() != 9 and time.time() < deadline:
            time.sleep(0.005)
        assert loop.latest() == 9
    finally:
        loop.shutdown()
