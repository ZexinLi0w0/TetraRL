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
from typing import Optional

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


# ---------------------------------------------------------------------------
# Framework integration tests (require new framework arg `concurrent_decision`)
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

from tetrarl.core.framework import (  # noqa: E402
    ResourceManager,
    StaticPreferencePlane,
    TetraRLFramework,
)
from tetrarl.morl.native.override import (  # noqa: E402
    OverrideLayer,
    OverrideThresholds,
)


class _ConstTelemetry:
    """latest() always returns the same HardwareTelemetry."""

    def __init__(self, hw: HardwareTelemetry):
        self._hw = hw

    def latest(self) -> HardwareTelemetry:
        return self._hw


class _FixedArbiter:
    """Returns a fixed action; records calls."""

    def __init__(self, action: int = 1):
        self.action = int(action)
        self.calls: list = []

    def act(self, state, omega):
        self.calls.append((np.asarray(state).copy(), np.asarray(omega).copy()))
        return self.action


def _telem_passthrough(hw: HardwareTelemetry) -> HardwareTelemetry:
    return hw


def _make_sequential_framework(arbiter, telemetry, dvfs):
    pref = StaticPreferencePlane(np.array([0.5, 0.5], dtype=np.float32))
    rm = ResourceManager()
    override = OverrideLayer(
        OverrideThresholds(max_latency_ms=10000.0, min_energy_j=0.5,
                           max_memory_util=0.95),
        fallback_action=0,
    )
    return TetraRLFramework(
        preference_plane=pref,
        rl_arbiter=arbiter,
        resource_manager=rm,
        override_layer=override,
        telemetry_source=telemetry,
        telemetry_adapter=_telem_passthrough,
        dvfs_controller=dvfs,
    )


def _make_concurrent_framework(arbiter, telemetry, dvfs, loop):
    pref = StaticPreferencePlane(np.array([0.5, 0.5], dtype=np.float32))
    rm = ResourceManager()
    override = OverrideLayer(
        OverrideThresholds(max_latency_ms=10000.0, min_energy_j=0.5,
                           max_memory_util=0.95),
        fallback_action=0,
    )
    return TetraRLFramework(
        preference_plane=pref,
        rl_arbiter=arbiter,
        resource_manager=rm,
        override_layer=override,
        telemetry_source=telemetry,
        telemetry_adapter=_telem_passthrough,
        dvfs_controller=dvfs,
        concurrent_decision=loop,
    )


def test_framework_records_concurrent_dvfs_used_flag_false_when_unset():
    """Backward compat: without the concurrent loop, the flag is False."""
    hw = _hw(latency=1.0, energy=100.0, mem=0.1)
    seq_fw = _make_sequential_framework(
        _FixedArbiter(action=1), _ConstTelemetry(hw), _StubDVFSController()
    )
    record = seq_fw.step(np.zeros(4, dtype=np.float32))
    assert record["concurrent_dvfs_used"] is False


def test_framework_first_step_with_concurrent_loop_does_not_crash():
    """At step 0, no decision exists yet; framework returns dvfs_idx=None safely."""
    hw = _hw(latency=1.0, energy=100.0, mem=0.1)
    rm = ResourceManager()
    dvfs = _StubDVFSController()
    loop = ConcurrentDecisionLoop(
        resource_manager=rm, dvfs_controller=dvfs, n_levels=14, fallback_idx=-1
    )
    try:
        fw = _make_concurrent_framework(
            _FixedArbiter(action=1), _ConstTelemetry(hw), dvfs, loop
        )
        record = fw.step(np.zeros(4, dtype=np.float32))
        assert record["action"] == 1
        assert record["concurrent_dvfs_used"] is True
        # No decision yet: dvfs_idx is None (no fallback configured).
        assert record["dvfs_idx"] is None
    finally:
        loop.shutdown()


def test_framework_concurrent_action_stream_matches_sequential_under_constant_telemetry():
    """With constant telemetry, the action stream must match step-for-step.

    The DVFS index is allowed to lag by 1 step on the concurrent path: at
    step 0 the concurrent record has dvfs_idx=None (no fallback set), and
    from step 1 onward it must match the sequential decision (which is
    constant when telemetry is constant).
    """
    hw = _hw(latency=1.0, energy=100.0, mem=0.1)

    seq_dvfs = _StubDVFSController()
    seq_fw = _make_sequential_framework(
        _FixedArbiter(action=1), _ConstTelemetry(hw), seq_dvfs
    )
    seq_records = [seq_fw.step(np.zeros(4, dtype=np.float32)) for _ in range(5)]

    conc_dvfs = _StubDVFSController()
    rm = ResourceManager()
    loop = ConcurrentDecisionLoop(
        resource_manager=rm, dvfs_controller=conc_dvfs, n_levels=14, fallback_idx=-1
    )
    try:
        conc_fw = _make_concurrent_framework(
            _FixedArbiter(action=1), _ConstTelemetry(hw), conc_dvfs, loop
        )
        conc_records: list = []
        for _ in range(5):
            conc_records.append(conc_fw.step(np.zeros(4, dtype=np.float32)))
            time.sleep(0.08)  # let the background worker drain the latest submit
    finally:
        loop.shutdown()

    # Action stream identical (deterministic arbiter + constant telemetry).
    for s, c in zip(seq_records, conc_records):
        assert s["action"] == c["action"], "action stream diverged"
        assert s["override_fired"] == c["override_fired"]

    # DVFS lag-by-one: concurrent step 0 is None, step >= 1 matches sequential.
    assert conc_records[0]["dvfs_idx"] is None
    seq_dvfs_idx = seq_records[0]["dvfs_idx"]  # constant under constant telemetry
    assert all(rec["dvfs_idx"] == seq_dvfs_idx for rec in seq_records)
    assert all(rec["dvfs_idx"] == seq_dvfs_idx for rec in conc_records[1:])

    # New flag set correctly on each path.
    assert all(r["concurrent_dvfs_used"] is False for r in seq_records)
    assert all(r["concurrent_dvfs_used"] is True for r in conc_records)


def test_framework_concurrent_per_step_time_not_egregiously_higher():
    """Concurrent per-step time must not be wildly higher than sequential.

    On Mac with stub DVFS, the decide_dvfs body is essentially free, so
    threading overhead can dominate. We allow up to 5x slack in this unit
    test (the smoke script enforces the tighter +10% bound over a longer
    run with real env dynamics).
    """
    hw = _hw(latency=1.0, energy=100.0, mem=0.1)
    n_steps = 200

    seq_dvfs = _StubDVFSController()
    seq_fw = _make_sequential_framework(
        _FixedArbiter(action=1), _ConstTelemetry(hw), seq_dvfs
    )
    t0 = time.perf_counter()
    for _ in range(n_steps):
        seq_fw.step(np.zeros(4, dtype=np.float32))
    seq_total = time.perf_counter() - t0
    seq_mean = seq_total / n_steps

    conc_dvfs = _StubDVFSController()
    rm = ResourceManager()
    loop = ConcurrentDecisionLoop(
        resource_manager=rm, dvfs_controller=conc_dvfs, n_levels=14, fallback_idx=0
    )
    try:
        conc_fw = _make_concurrent_framework(
            _FixedArbiter(action=1), _ConstTelemetry(hw), conc_dvfs, loop
        )
        t0 = time.perf_counter()
        for _ in range(n_steps):
            conc_fw.step(np.zeros(4, dtype=np.float32))
        conc_total = time.perf_counter() - t0
        conc_mean = conc_total / n_steps
    finally:
        loop.shutdown()

    # Sanity: both well under 50 ms/step on any modern laptop.
    assert seq_mean < 0.05, f"sequential too slow: {seq_mean*1000:.3f} ms/step"
    assert conc_mean < 0.05, f"concurrent too slow: {conc_mean*1000:.3f} ms/step"
    # Allow concurrent up to 5x sequential mean (timing is noisy on Mac).
    assert conc_mean <= max(seq_mean * 5.0, 0.001), (
        f"concurrent regression: seq={seq_mean*1000:.3f} ms, "
        f"conc={conc_mean*1000:.3f} ms"
    )
