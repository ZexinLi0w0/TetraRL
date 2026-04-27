"""P15 Phase 6 — verify MaxA forces batch=16 and MaxP forces batch=128.

These wrappers used to be DVFS-only, which made them indistinguishable in
matrix data (the bug Timmy caught 2026-04-26 22:25 PDT). The contract:

- MaxA.wrap(algo) sets algo.batch_size = 16 (and mini_batch_size if present).
- MaxP.wrap(algo) sets algo.batch_size = 128 (and mini_batch_size if present).
- step_hook echoes the same batch_size in WrapperKnobs.
- get_metrics()["batch_size"] is the wrapper's pinned batch.
"""
from __future__ import annotations

import pytest

from tetrarl.morl.system_wrappers import MaxAWrapper, MaxPWrapper


class _DummyOffPolicyAlgo:
    paradigm = "off_policy"

    def __init__(self, batch_size: int = 64) -> None:
        self.batch_size = int(batch_size)


class _DummyOnPolicyAlgo:
    paradigm = "on_policy"

    def __init__(self, mini_batch_size: int = 32) -> None:
        # Mirror algos.py: batch_size mirrors mini_batch_size on _OnPolicyBase.
        self.mini_batch_size = int(mini_batch_size)
        self.batch_size = int(mini_batch_size)


# --- MaxA forces batch=16 ---------------------------------------------------


def test_maxa_default_batch_is_16() -> None:
    w = MaxAWrapper()
    assert w.batch_size == 16


def test_maxa_wrap_sets_off_policy_batch_to_16() -> None:
    algo = _DummyOffPolicyAlgo(batch_size=64)
    MaxAWrapper().wrap(algo)
    assert algo.batch_size == 16


def test_maxa_wrap_sets_on_policy_mini_batch_to_16() -> None:
    algo = _DummyOnPolicyAlgo(mini_batch_size=32)
    MaxAWrapper().wrap(algo)
    assert algo.batch_size == 16
    assert algo.mini_batch_size == 16


def test_maxa_step_hook_emits_batch_16() -> None:
    w = MaxAWrapper()
    knobs = w.step_hook(0, {"last_step_ms": 0.0, "last_step_energy_j": 0.0, "memory_util": 0.0})
    assert knobs.batch_size == 16
    assert knobs.dvfs_idx == 11  # default max_dvfs_idx


def test_maxa_get_metrics_exposes_batch_size_16() -> None:
    w = MaxAWrapper()
    w.step_hook(0, {"last_step_ms": 0.0, "last_step_energy_j": 0.0, "memory_util": 0.0})
    m = w.get_metrics()
    assert m["batch_size"] == 16


def test_maxa_custom_batch_override_is_honored() -> None:
    w = MaxAWrapper(batch_size=32)
    algo = _DummyOffPolicyAlgo(batch_size=64)
    w.wrap(algo)
    assert algo.batch_size == 32
    assert w.step_hook(0, {"last_step_ms": 0.0, "last_step_energy_j": 0.0, "memory_util": 0.0}).batch_size == 32


# --- MaxP forces batch=128 --------------------------------------------------


def test_maxp_default_batch_is_128() -> None:
    w = MaxPWrapper()
    assert w.batch_size == 128


def test_maxp_wrap_sets_off_policy_batch_to_128() -> None:
    algo = _DummyOffPolicyAlgo(batch_size=64)
    MaxPWrapper().wrap(algo)
    assert algo.batch_size == 128


def test_maxp_wrap_sets_on_policy_mini_batch_to_128() -> None:
    algo = _DummyOnPolicyAlgo(mini_batch_size=32)
    MaxPWrapper().wrap(algo)
    assert algo.batch_size == 128
    assert algo.mini_batch_size == 128


def test_maxp_step_hook_emits_batch_128_at_dvfs_0() -> None:
    w = MaxPWrapper()
    knobs = w.step_hook(0, {"last_step_ms": 0.0, "last_step_energy_j": 0.0, "memory_util": 0.0})
    assert knobs.batch_size == 128
    assert knobs.dvfs_idx == 0


def test_maxp_get_metrics_exposes_batch_size_128() -> None:
    w = MaxPWrapper()
    w.step_hook(0, {"last_step_ms": 0.0, "last_step_energy_j": 0.0, "memory_util": 0.0})
    m = w.get_metrics()
    assert m["batch_size"] == 128


def test_maxp_custom_batch_override_is_honored() -> None:
    w = MaxPWrapper(batch_size=64)
    algo = _DummyOffPolicyAlgo(batch_size=32)
    w.wrap(algo)
    assert algo.batch_size == 64
    assert w.step_hook(0, {"last_step_ms": 0.0, "last_step_energy_j": 0.0, "memory_util": 0.0}).batch_size == 64


# --- MaxA vs MaxP differ ----------------------------------------------------


def test_maxa_and_maxp_emit_different_batches() -> None:
    """Regression for Issue 4: identical metrics when neither set batch_size."""
    a = MaxAWrapper()
    p = MaxPWrapper()
    state = {"last_step_ms": 0.0, "last_step_energy_j": 0.0, "memory_util": 0.0}
    assert a.step_hook(0, state).batch_size != p.step_hook(0, state).batch_size
