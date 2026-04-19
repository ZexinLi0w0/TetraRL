"""Side-effect: register TetraRL's custom Gymnasium envs on import.

Imported by :mod:`tetrarl.envs.__init__` so the unified eval runner can
resolve ``env_name="dag_scheduler_mo-v0"`` via ``gym.make`` without
having to construct the env class directly. Registration is guarded
against re-registration so re-importing :mod:`tetrarl.envs` (e.g. by
multiple test modules) does not raise.
"""
from __future__ import annotations

from gymnasium.envs.registration import register, registry

_DAG_ENV_ID = "dag_scheduler_mo-v0"

if _DAG_ENV_ID not in registry:
    register(
        id=_DAG_ENV_ID,
        entry_point="tetrarl.envs.dag_scheduler:DAGSchedulerEnv",
        kwargs={"n_tasks": 6, "density": 0.3, "reward_dim": 4},
    )
