"""Tests for the per-omega flag added to the FFmpeg co-runner harness.

The W9 spec Task C requires the harness to accept an ``--omega`` keyword
(``energy_corner``, ``memory_corner`` or ``center``) so the per-omega CDF
figure (``scripts/week9_make_expanded_cdf.py``) can be populated from a
single script invocation per omega.

The 2-D omega vectors are fixed by the design doc Section 6.1:
* ``energy_corner`` = ``[1.0, 0.0]``
* ``memory_corner`` = ``[0.0, 1.0]``
* ``center``        = ``[0.5, 0.5]``
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.week6_e2e_smoke import make_framework
from scripts.week7_ffmpeg_corunner import OMEGA_PRESETS, parse_omega_name


def test_parse_omega_name_energy_corner_returns_unit_first_axis():
    omega = parse_omega_name("energy_corner")
    np.testing.assert_allclose(omega, np.array([1.0, 0.0], dtype=np.float32))


def test_parse_omega_name_memory_corner_returns_unit_second_axis():
    omega = parse_omega_name("memory_corner")
    np.testing.assert_allclose(omega, np.array([0.0, 1.0], dtype=np.float32))


def test_parse_omega_name_center_returns_uniform():
    omega = parse_omega_name("center")
    np.testing.assert_allclose(omega, np.array([0.5, 0.5], dtype=np.float32))


def test_parse_omega_name_unknown_raises():
    with pytest.raises(ValueError):
        parse_omega_name("not_a_real_omega")


def test_omega_presets_keys_are_the_three_names():
    assert sorted(OMEGA_PRESETS.keys()) == ["center", "energy_corner", "memory_corner"]


def test_make_framework_accepts_omega_kwarg_default_unchanged():
    """Default behaviour (omega=None) preserves the W6 [0.5, 0.5] preference."""
    fw, _, _ = make_framework(n_actions=2, seed=0)
    np.testing.assert_allclose(
        fw.preference_plane.get(),
        np.array([0.5, 0.5], dtype=np.float32),
    )


def test_make_framework_propagates_omega_to_preference_plane():
    """Passing ``omega=`` plumbs into ``StaticPreferencePlane``."""
    omega = np.array([1.0, 0.0], dtype=np.float32)
    fw, _, _ = make_framework(n_actions=2, seed=0, omega=omega)
    np.testing.assert_allclose(fw.preference_plane.get(), omega)
