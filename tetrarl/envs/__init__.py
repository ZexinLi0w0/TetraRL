"""Gymnasium environment wrappers for multi-objective reward augmentation."""

from tetrarl.envs.dst import DeepSeaTreasure  # noqa: F401
from tetrarl.envs.mo_mountaincar import MOMountainCarContinuous  # noqa: F401

__all__ = ["DeepSeaTreasure", "MOMountainCarContinuous"]
