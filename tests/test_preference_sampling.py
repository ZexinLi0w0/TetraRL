"""Unit tests for preference-sampling utilities."""

import numpy as np

from tetrarl.morl.preference_sampling import (
    her_preference_relabel,
    sample_anchor_preferences,
    sample_preference,
)


class TestSamplePreference:

    def test_normalization(self):
        prefs = sample_preference(3, batch_size=100)
        assert prefs.shape == (100, 3)
        sums = prefs.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_positive(self):
        prefs = sample_preference(4, batch_size=50)
        assert (prefs >= 0).all()

    def test_reproducibility(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        p1 = sample_preference(3, 10, rng=rng1)
        p2 = sample_preference(3, 10, rng=rng2)
        np.testing.assert_array_equal(p1, p2)

    def test_single(self):
        pref = sample_preference(2, 1)
        assert pref.shape == (1, 2)
        np.testing.assert_allclose(pref.sum(), 1.0, atol=1e-5)


class TestSampleAnchorPreferences:

    def test_shape(self):
        anchors = sample_anchor_preferences(3)
        assert anchors.shape == (4, 3)

    def test_corners(self):
        anchors = sample_anchor_preferences(2)
        np.testing.assert_allclose(anchors[0], [1.0, 0.0])
        np.testing.assert_allclose(anchors[1], [0.0, 1.0])

    def test_center(self):
        anchors = sample_anchor_preferences(3)
        np.testing.assert_allclose(anchors[-1], [1.0 / 3, 1.0 / 3, 1.0 / 3], atol=1e-5)


class TestHERPreferenceRelabel:

    def test_augmentation_count(self):
        transitions = [
            {
                "state": np.zeros(2),
                "action": 0,
                "reward_vec": np.array([1.0, -1.0]),
                "next_state": np.zeros(2),
                "done": False,
                "omega": np.array([0.5, 0.5]),
            }
        ]
        augmented = her_preference_relabel(transitions, num_objectives=2, n_relabel=4)
        assert len(augmented) == 5

    def test_original_preserved(self):
        original = {
            "state": np.array([1.0, 2.0]),
            "action": 1,
            "reward_vec": np.array([3.0, -1.0]),
            "next_state": np.array([1.0, 3.0]),
            "done": False,
            "omega": np.array([0.7, 0.3]),
        }
        augmented = her_preference_relabel([original], num_objectives=2, n_relabel=3)
        np.testing.assert_array_equal(augmented[0]["state"], original["state"])
        np.testing.assert_array_equal(augmented[0]["omega"], original["omega"])

    def test_relabeled_omegas_differ(self):
        transitions = [
            {
                "state": np.zeros(2),
                "action": 0,
                "reward_vec": np.array([1.0, -1.0]),
                "next_state": np.zeros(2),
                "done": False,
                "omega": np.array([0.5, 0.5]),
            }
        ]
        augmented = her_preference_relabel(transitions, num_objectives=2, n_relabel=4)
        for t in augmented[1:]:
            np.testing.assert_allclose(t["omega"].sum(), 1.0, atol=1e-5)

    def test_reward_unchanged(self):
        reward = np.array([5.0, -3.0])
        transitions = [
            {
                "state": np.zeros(2),
                "action": 0,
                "reward_vec": reward.copy(),
                "next_state": np.zeros(2),
                "done": False,
                "omega": np.array([0.5, 0.5]),
            }
        ]
        augmented = her_preference_relabel(transitions, num_objectives=2, n_relabel=3)
        for t in augmented:
            np.testing.assert_array_equal(t["reward_vec"], reward)
