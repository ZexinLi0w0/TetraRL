"""Tests for the P15 SystemWrapper ABC + 5 concrete wrappers + factory.

Covers:
- ``WRAPPER_REGISTRY`` exposes exactly the 5 expected keys.
- ``make_wrapper`` returns the correct concrete class for each key and raises
  on unknown names.
- ``is_compatible`` truth table across (wrapper, real algo class) pairs.
- ``step_hook`` returns a :class:`WrapperKnobs` for every wrapper.
- MaxA / MaxP pin DVFS to the documented indices.
- R3 + DuoJoule shrink batch under sustained pressure.
- TetraRL fires the override layer when memory utilisation exceeds threshold.
- ``get_metrics`` always exposes ``wrapper`` + ``n_steps``.
"""
from __future__ import annotations

from typing import Any

import pytest

from tetrarl.morl.algos import (
    A2CAlgo,
    C51Algo,
    DDQNAlgo,
    DQNAlgo,
    PPOAlgo,
)
from tetrarl.morl.system_wrapper import WrapperKnobs
from tetrarl.morl.system_wrappers import (
    WRAPPER_REGISTRY,
    DuoJouleWrapper,
    MaxAWrapper,
    MaxPWrapper,
    R3Wrapper,
    TetraRLWrapper,
    make_wrapper,
)


class _FakeAlgo:
    """Cheap stand-in for an off-policy algo with a mutable batch_size."""

    paradigm = "off_policy"

    def __init__(self, batch_size: int = 128, replay_capacity: int = 10_000) -> None:
        self.batch_size = batch_size
        self.replay_capacity = replay_capacity


def _zero_state() -> dict[str, Any]:
    return {"last_step_ms": 0.0, "last_step_energy_j": 0.0, "memory_util": 0.0}


# --- registry + factory ----------------------------------------------------


def test_wrapper_registry_keys_exact() -> None:
    assert set(WRAPPER_REGISTRY.keys()) == {"maxa", "maxp", "r3", "duojoule", "tetrarl"}


@pytest.mark.parametrize(
    "name,cls",
    [
        ("maxa", MaxAWrapper),
        ("maxp", MaxPWrapper),
        ("r3", R3Wrapper),
        ("duojoule", DuoJouleWrapper),
        ("tetrarl", TetraRLWrapper),
    ],
)
def test_make_wrapper_returns_correct_class(name: str, cls: type) -> None:
    w = make_wrapper(name)
    assert isinstance(w, cls)


def test_make_wrapper_raises_on_unknown() -> None:
    with pytest.raises(ValueError):
        make_wrapper("nonsense")


# --- is_compatible truth table --------------------------------------------


@pytest.mark.parametrize(
    "wrapper_name,algo_class,expected_compat",
    [
        # R3 / DuoJoule: off-policy only.
        ("r3", DQNAlgo, True),
        ("r3", DDQNAlgo, True),
        ("r3", C51Algo, True),
        ("r3", A2CAlgo, False),
        ("r3", PPOAlgo, False),
        ("duojoule", DQNAlgo, True),
        ("duojoule", DDQNAlgo, True),
        ("duojoule", C51Algo, True),
        ("duojoule", A2CAlgo, False),
        ("duojoule", PPOAlgo, False),
        # MaxA / MaxP / TetraRL: compatible with everything.
        ("maxa", DQNAlgo, True),
        ("maxa", DDQNAlgo, True),
        ("maxa", C51Algo, True),
        ("maxa", A2CAlgo, True),
        ("maxa", PPOAlgo, True),
        ("maxp", DQNAlgo, True),
        ("maxp", DDQNAlgo, True),
        ("maxp", C51Algo, True),
        ("maxp", A2CAlgo, True),
        ("maxp", PPOAlgo, True),
        ("tetrarl", DQNAlgo, True),
        ("tetrarl", DDQNAlgo, True),
        ("tetrarl", C51Algo, True),
        ("tetrarl", A2CAlgo, True),
        ("tetrarl", PPOAlgo, True),
    ],
)
def test_is_compatible_truth_table(
    wrapper_name: str, algo_class: type, expected_compat: bool
) -> None:
    w = make_wrapper(wrapper_name)
    assert w.is_compatible(algo_class) is expected_compat


# --- step_hook returns WrapperKnobs ---------------------------------------


@pytest.mark.parametrize(
    "wrapper_name", ["maxa", "maxp", "r3", "duojoule", "tetrarl"]
)
def test_step_hook_returns_wrapper_knobs(wrapper_name: str) -> None:
    w = make_wrapper(wrapper_name)
    # R3 / DuoJoule require wrap() before step_hook to seed _current_batch.
    if wrapper_name in {"r3", "duojoule"}:
        w.wrap(_FakeAlgo(batch_size=64))
    knobs = w.step_hook(0, _zero_state())
    assert isinstance(knobs, WrapperKnobs)


# --- MaxA / MaxP DVFS pinning ---------------------------------------------


def test_maxa_pins_dvfs_to_max_index() -> None:
    w = MaxAWrapper()  # default max_dvfs_idx=11
    knobs = w.step_hook(0, _zero_state())
    assert knobs.dvfs_idx == w.max_dvfs_idx
    assert knobs.dvfs_idx == 11


def test_maxp_pins_dvfs_to_zero() -> None:
    w = MaxPWrapper()
    knobs = w.step_hook(0, _zero_state())
    assert knobs.dvfs_idx == 0


# --- R3: high latency shrinks batch ---------------------------------------


def test_r3_shrinks_batch_on_sustained_deadline_miss() -> None:
    w = R3Wrapper(deadline_ms=10.0, min_batch=16, max_batch=256)
    fake = _FakeAlgo(batch_size=128)
    w.wrap(fake)
    final_knobs: WrapperKnobs | None = None
    for step_idx in range(30):
        final_knobs = w.step_hook(
            step_idx,
            {
                "last_step_ms": 100.0,  # massive deadline miss
                "last_step_energy_j": 0.0,
                "memory_util": 0.0,
            },
        )
    assert final_knobs is not None
    # At least one halving must have occurred (128 -> 64 or smaller).
    assert final_knobs.batch_size is not None
    assert final_knobs.batch_size <= 64


# --- DuoJoule: high energy shrinks batch ----------------------------------


def test_duojoule_shrinks_batch_on_sustained_high_energy() -> None:
    w = DuoJouleWrapper(energy_target_j=0.05, min_batch=16, max_batch=256)
    fake = _FakeAlgo(batch_size=128)
    w.wrap(fake)
    final_knobs: WrapperKnobs | None = None
    for step_idx in range(30):
        final_knobs = w.step_hook(
            step_idx,
            {
                "last_step_ms": 0.0,
                "last_step_energy_j": 5.0,  # >> 0.05 target
                "memory_util": 0.0,
            },
        )
    assert final_knobs is not None
    assert final_knobs.batch_size is not None
    assert final_knobs.batch_size <= 64


# --- TetraRL override layer ------------------------------------------------


def test_tetrarl_override_fires_on_high_memory() -> None:
    w = TetraRLWrapper(memory_threshold=0.85)
    # No wrap() is required for the override path; algo_class is used only by
    # the off-policy probe inside step_hook for replay_capacity. The default
    # branch (no algo wrapped) still walks the override path correctly.
    knobs = w.step_hook(0, {"last_step_ms": 0.0, "last_step_energy_j": 0.0, "memory_util": 0.99})
    assert knobs.action_override is not None
    metrics = w.get_metrics()
    assert metrics["override_fire_count"] >= 1


# --- get_metrics surface --------------------------------------------------


@pytest.mark.parametrize(
    "wrapper_name", ["maxa", "maxp", "r3", "duojoule", "tetrarl"]
)
def test_get_metrics_includes_wrapper_and_n_steps(wrapper_name: str) -> None:
    w = make_wrapper(wrapper_name)
    if wrapper_name in {"r3", "duojoule"}:
        w.wrap(_FakeAlgo(batch_size=64))
    # Drive at least one step so n_steps reflects a real call.
    w.step_hook(0, _zero_state())
    m = w.get_metrics()
    assert isinstance(m, dict)
    assert m.get("wrapper") == wrapper_name
    assert "n_steps" in m
