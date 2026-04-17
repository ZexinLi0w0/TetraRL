"""Algorithm implementations for TetraRL.

Each agent module exposes a stable class-level interface (act/update/save/load)
so that the runtime coordinator can dispatch across paradigms uniformly.
PD-MORL (pd_morl.py) is fully implemented; remaining agents are stubs whose
implementation is scheduled per docs/action-plan-weekly.md.
"""

# Stubs (full implementation pending; see docs/action-plan-weekly.md).
from tetrarl.morl.agents.a2c import A2CAgent  # noqa: F401
from tetrarl.morl.agents.a3c import A3CAgent  # noqa: F401
from tetrarl.morl.agents.c51 import C51Agent  # noqa: F401
from tetrarl.morl.agents.ddqn import DDQNAgent  # noqa: F401
from tetrarl.morl.agents.dqn import DQNAgent  # noqa: F401
from tetrarl.morl.agents.mo_sac_her import MOSACHERAgent  # noqa: F401
from tetrarl.morl.agents.pd_morl import PDMORLAgent  # noqa: F401
from tetrarl.morl.agents.ppo import PPOAgent  # noqa: F401
from tetrarl.morl.agents.sac import SACAgent  # noqa: F401

__all__ = [
    "PDMORLAgent",
    "DQNAgent",
    "DDQNAgent",
    "C51Agent",
    "SACAgent",
    "MOSACHERAgent",
    "A2CAgent",
    "A3CAgent",
    "PPOAgent",
]
