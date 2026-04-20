"""Tests for tetrarl.runtime.preference_elicitation."""

from __future__ import annotations

import math

import pytest

from tetrarl.runtime.preference_elicitation import (
    OBJECTIVE_NAMES,
    PROFILES,
    from_ordinal,
    from_profile,
    list_profiles,
    simplex_normalize,
)

# -----------------------------------------------------------------------------
# simplex_normalize
# -----------------------------------------------------------------------------


def test_simplex_normalize_already_simplex():
    # Note: even when the input is already on the simplex, normalisation
    # divides by the (computed) sum, which can introduce ULP-level float
    # noise depending on platform/glibc. Use approx() rather than ==.
    omega = simplex_normalize([0.4, 0.3, 0.2, 0.1])
    assert omega == pytest.approx((0.4, 0.3, 0.2, 0.1))


def test_simplex_normalize_unnormalised():
    omega = simplex_normalize([4.0, 3.0, 2.0, 1.0])
    assert omega == pytest.approx((0.4, 0.3, 0.2, 0.1))
    assert math.isclose(sum(omega), 1.0)


def test_simplex_normalize_all_zero_falls_back_uniform():
    omega = simplex_normalize([0, 0, 0, 0])
    assert omega == (0.25, 0.25, 0.25, 0.25)


def test_simplex_normalize_rejects_wrong_length():
    with pytest.raises(ValueError, match="exactly 4"):
        simplex_normalize([1, 2, 3])
    with pytest.raises(ValueError, match="exactly 4"):
        simplex_normalize([1, 2, 3, 4, 5])


def test_simplex_normalize_rejects_negative():
    with pytest.raises(ValueError, match="negative"):
        simplex_normalize([0.5, 0.5, -0.1, 0.1])


def test_simplex_normalize_rejects_non_numeric():
    with pytest.raises(TypeError, match="not numeric"):
        simplex_normalize(["a", 0.1, 0.1, 0.1])  # type: ignore[list-item]


def test_simplex_normalize_rejects_nan_inf():
    with pytest.raises(ValueError, match="finite"):
        simplex_normalize([float("nan"), 0.1, 0.1, 0.1])
    with pytest.raises(ValueError, match="finite"):
        simplex_normalize([float("inf"), 0.1, 0.1, 0.1])


# -----------------------------------------------------------------------------
# Profiles (Layer 2)
# -----------------------------------------------------------------------------


def test_list_profiles_includes_canonical_set():
    profiles = list_profiles()
    for name in (
        "balanced",
        "performance",
        "low-latency",
        "low-memory",
        "battery-saver",
        "idle",
    ):
        assert name in profiles


def test_every_profile_is_on_simplex():
    for name, omega in PROFILES.items():
        assert len(omega) == 4, f"{name} wrong length"
        assert math.isclose(sum(omega), 1.0, abs_tol=1e-9), (
            f"{name} not on simplex"
        )
        assert all(x >= 0 for x in omega), f"{name} has negative"


def test_from_profile_known():
    assert from_profile("battery-saver") == (0.2, 0.2, 0.3, 0.3)
    # Case + whitespace tolerant.
    assert from_profile("  Performance  ") == (0.5, 0.3, 0.1, 0.1)


def test_from_profile_unknown_raises():
    with pytest.raises(KeyError, match="unknown profile"):
        from_profile("turbo-mode")


def test_from_profile_rejects_non_string():
    with pytest.raises(TypeError):
        from_profile(123)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Ordinal parsing (Layer 1)
# -----------------------------------------------------------------------------


def test_from_ordinal_strict_canonical_order():
    omega = from_ordinal("R > T > M > E")
    # Borda: R=0.4, T=0.3, M=0.2, E=0.1
    assert omega == pytest.approx((0.4, 0.3, 0.2, 0.1))


def test_from_ordinal_tie_at_top():
    # R and T tied at rank-1 -> share (0.4 + 0.3) / 2 = 0.35 each.
    # E rank-3 = 0.2, M rank-4 = 0.1.
    omega = from_ordinal("R >= T > E > M")
    assert omega == pytest.approx((0.35, 0.35, 0.1, 0.2))


def test_from_ordinal_full_aliases():
    omega = from_ordinal("Reward > Latency > Energy >= Memory")
    # Reward 0.4, Latency 0.3, Energy & Memory tied for rank-3 group share
    # (0.2 + 0.1)/2 = 0.15 each.
    assert omega == pytest.approx((0.4, 0.3, 0.15, 0.15))


def test_from_ordinal_preserves_simplex():
    for spec in (
        "R > T > M > E",
        "R >= T >= M >= E",
        "R > T >= M > E",
        "T > R > E > M",
    ):
        omega = from_ordinal(spec)
        assert math.isclose(sum(omega), 1.0, abs_tol=1e-9), spec


def test_from_ordinal_all_tied_is_uniform():
    omega = from_ordinal("R >= T >= M >= E")
    assert omega == pytest.approx((0.25, 0.25, 0.25, 0.25))


def test_from_ordinal_temperature_smooths_toward_uniform():
    sharp = from_ordinal("R > T > M > E", temperature=0.3)
    smooth = from_ordinal("R > T > M > E", temperature=10.0)
    # Sharp: top rank should be much larger than bottom.
    assert sharp[0] > 0.5
    # Smooth: all four close to 0.25.
    assert max(smooth) - min(smooth) < 0.1
    # Ordering must still respect ranks.
    assert sharp[0] > sharp[1] > sharp[2] > sharp[3]
    assert smooth[0] > smooth[1] > smooth[2] > smooth[3]


def test_from_ordinal_missing_objective_raises():
    with pytest.raises(ValueError, match="missing"):
        from_ordinal("R > T > M")  # E missing


def test_from_ordinal_duplicate_objective_raises():
    with pytest.raises(ValueError, match="duplicated"):
        from_ordinal("R > T > M > R")  # R repeated, E missing


def test_from_ordinal_rejects_lt_operator():
    with pytest.raises(ValueError, match="'<'"):
        from_ordinal("R < T < M < E")


def test_from_ordinal_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        from_ordinal("   ")


def test_from_ordinal_rejects_unknown_token():
    with pytest.raises(ValueError, match="unknown objective token"):
        from_ordinal("R > T > Cost > E")


def test_from_ordinal_rejects_stray_gt():
    with pytest.raises(ValueError, match="empty rank"):
        from_ordinal("R > > T > M > E")


def test_from_ordinal_rejects_invalid_temperature():
    with pytest.raises(ValueError, match="temperature"):
        from_ordinal("R > T > M > E", temperature=0)
    with pytest.raises(ValueError, match="temperature"):
        from_ordinal("R > T > M > E", temperature=-1)


def test_from_ordinal_rejects_non_string():
    with pytest.raises(TypeError):
        from_ordinal(["R", "T", "M", "E"])  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Objective name canonical order
# -----------------------------------------------------------------------------


def test_objective_names_canonical_order():
    # Order must be Reward, real-Time, Memory, Energy to match runner / yaml.
    assert OBJECTIVE_NAMES == ("R", "T", "M", "E")
