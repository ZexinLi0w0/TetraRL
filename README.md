# TetraRL: A Multi-Objective Runtime System for On-Device Reinforcement Learning Training


[![CI](https://github.com/ZexinLi0w0/TetraRL/actions/workflows/ci.yml/badge.svg)](https://github.com/ZexinLi0w0/TetraRL/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-Week%200%20%E2%80%94%20skeleton-lightgrey.svg)](docs/action-plan-weekly.md)

**Built around the R⁴ principle: Real-time × Reward × RAM × Recharge**

<!-- Badges -->
![CI](https://github.com/ZexinLi0w0/TetraRL/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![arXiv](https://img.shields.io/badge/arXiv-TBD-b31b1b.svg)

---

## Overview

TetraRL is a multi-objective runtime system that co-optimizes four dimensions —
**Timing**, **Algorithm performance**, **Memory footprint**, and **Energy
consumption** — during on-device deep reinforcement learning (DRL) training.
Unlike prior single-objective approaches that treat deadline satisfaction or
reward maximization in isolation, TetraRL formulates the problem as a
preference-conditioned multi-objective RL (MORL) task and navigates the
resulting 4-D Pareto front at runtime.  The system extends the on-device DRL
training infrastructure established by R³ (RTSS 2023) and the energy-aware
adaptation mechanisms of DuoJoule (RTSS 2024), unifying them under a single
four-component framework.  TetraRL targets NVIDIA Jetson embedded platforms
(AGX Orin, Xavier NX, Orin Nano) and is under submission to IEEE Transactions
on Computers.

## The R⁴ Principle

The name *TetraRL* derives from the four resource axes — the **R⁴ principle** —
that the system jointly optimizes:

| R | Axis | Metric |
|---|------|--------|
| **Real-time** | Timing | Per-step and per-episode latency; deadline miss rate; tail-latency CDF |
| **Reward** | Algorithm performance | Episodic return; convergence speed; Hypervolume indicator |
| **RAM** | Memory | Peak unified-memory utilization; replay-buffer footprint |
| **Recharge** | Energy | Per-step energy (Joules); cumulative energy over a training run |

Classical on-device DRL systems optimize one or two of these dimensions,
accepting degradation in the others.  R³ demonstrated that deadline-driven
batch-size and replay-buffer co-adaptation can maintain real-time guarantees
without catastrophic reward loss, but it treats energy and memory as secondary
concerns.  TetraRL closes this gap by embedding all four axes into a single
reward vector and employing a preference-conditioned MORL agent that can
smoothly trade off among them at runtime — for example, shifting toward
energy-conservative operation when the battery level drops, or toward
memory-conservative operation on platforms with limited unified memory.

## Relationship to Prior Work

TetraRL is a natural extension of the research line established by the
following prior works:

- **R³** (Li et al., RTSS 2023) provides the on-device DRL training
  infrastructure, including the deadline-driven feedback loop, replay-buffer
  memory management, and the runtime coordinator architecture.  TetraRL
  preserves and extends these mechanisms within its Resource Manager component.
  Repository: <https://github.com/ZexinLi0w0/R3>

- **DuoJoule** (Shirvani et al., RTSS 2024) introduces accurate energy
  profiling and energy-aware runtime adaptation for on-device DRL.  TetraRL
  integrates DuoJoule's energy measurement methodology into the Recharge axis
  of the R⁴ reward vector.

- **PD-MORL** (Basaklar et al., ICLR 2023) contributes the cosine-similarity
  envelope operator and the preference-conditioned multi-objective Q-learning
  framework.  TetraRL adopts this operator as the core of its RL Arbiter and
  extends it to continuous-action domains via MO-SAC-HER.

- **Lagrangian-Empirical** (Spoor et al., 2025) provides the PI-controller
  dual-variable update rule used in TetraRL's constraint satisfaction layer,
  along with empirical evidence that Lagrangian methods alone violate
  constraints 20–30% of the time — motivating the Hardware-Emergency Override
  Layer.

## System Architecture

TetraRL employs a **four-component framework** that coordinates algorithm-level
and system-level decisions within a unified control loop:

### 1. Preference Plane

The Preference Plane accepts a 4-D preference vector ω ∈ Δ³ (the probability
simplex over {Timing, Reward, Memory, Energy}) and maintains a structured
preference-sampling strategy.  It supports runtime preference switching —
enabling dynamic trade-off adjustment in response to changing operational
conditions (e.g., low battery, high co-runner interference) — and employs an
RBF interpolator to generalize across unseen preference vectors.

### 2. Resource Manager

The Resource Manager controls the system-level knobs that affect resource
consumption: DVFS frequency selection (CPU/GPU/EMC), replay-buffer soft
truncation, batch-size scaling, and on-policy rollout length.  It inherits the
co-adaptation strategy from R³ (batch size × replay buffer) and extends it with
DVFS actuation and energy budgeting.  A unified knob taxonomy maps both
off-policy (replay buffer size) and on-policy (n\_steps, mini-batch size) knobs
to a common `M_store` primitive.

### 3. RL Arbiter

The RL Arbiter is the preference-conditioned MORL agent (MO-SAC-HER for
continuous actions, MO-DQN-HER for discrete actions) that receives a
system-augmented state vector `S_sys = [s_t, L_prev, E_rem, M_util, ω]` and
outputs actions that jointly optimize the 4-D reward vector.  The
cosine-similarity envelope operator from PD-MORL guides the critic toward
Pareto-optimal Q-vectors, while HER-style preference replay enables sample
reuse across preference conditions.

### 4. Hardware-Emergency Override Layer

The Override Layer acts as a hard safety net beneath the learned policy.  When
any resource dimension exceeds a critical threshold (e.g., memory utilization
\> 95%, energy budget exhausted, deadline miss cascade), the Override Layer
bypasses the policy output and forces a conservative fallback action — inspired
by the CPO recovery step (Achiam et al., 2017).  This layer ensures that
Lagrangian constraint violations, which occur in 20–30% of episodes even under
well-tuned dual variables, do not translate to catastrophic system failures.

## Supported RL Algorithms

A core design goal of TetraRL is to validate the generality of 4-D
co-optimization across fundamentally different RL paradigms.  Prior systems in
this research line — R³ (Li et al., RTSS 2023) and DuoJoule (Shirvani et al.,
RTSS 2024) — exclusively support off-policy, replay-based algorithms (DQN,
DDQN, C51) whose replay buffers serve as the primary runtime knob.  TetraRL
extends algorithm coverage to on-policy, non-replay methods (A2C, A3C, PPO),
where rollout buffers, update epochs, and worker counts replace the replay
buffer as the dominant resource knobs.  Demonstrating that the same 4-D
Pareto-tracking runtime operates correctly under both paradigms is essential
evidence that the R⁴ framework is algorithm-agnostic rather than an artifact of
off-policy buffer manipulation.

| Algorithm | Paradigm | Action Space | Origin / Inheritance | TetraRL Module | R⁴ Knobs Available |
|-----------|----------|--------------|----------------------|----------------|--------------------|
| DQN | Off-policy / replay-based | Discrete | Inherited from R³ / DuoJoule baseline | `tetrarl/morl/agents/dqn.py` | `batch_size`, `replay_buffer_size`, DVFS, `mixed_precision` |
| DDQN | Off-policy / replay-based | Discrete | Inherited from R³ / DuoJoule baseline | `tetrarl/morl/agents/ddqn.py` | `batch_size`, `replay_buffer_size`, DVFS, `mixed_precision` |
| C51 | Off-policy / replay-based (distributional) | Discrete | Inherited from R³ / DuoJoule baseline | `tetrarl/morl/agents/c51.py` | `batch_size`, `replay_buffer_size`, DVFS, `mixed_precision` |
| SAC | Off-policy / replay-based (actor-critic) | Continuous | New for TetraRL (continuous control) | `tetrarl/morl/agents/sac.py` | `batch_size`, `replay_buffer_size`, DVFS, `mixed_precision` |
| A2C | On-policy / non-replay | Both | New for TetraRL (non-replay extension §9) | `tetrarl/morl/agents/a2c.py` | `n_steps`, `n_envs`, `n_epochs`, `mini_batch_size`, DVFS, `mixed_precision` |
| A3C | On-policy / non-replay (async) | Both | New for TetraRL (non-replay extension §9) | `tetrarl/morl/agents/a3c.py` | `n_steps`, `n_envs`, `n_workers`, DVFS, `mixed_precision` |
| PPO | On-policy / non-replay | Both | New for TetraRL (non-replay extension §9) | `tetrarl/morl/agents/ppo.py` | `n_steps`, `n_envs`, `n_epochs`, `mini_batch_size`, `KL_coef`, DVFS, `mixed_precision`, `gradient_checkpointing` |

**Unified Resource Primitives.**  Despite the apparent diversity of knobs
across algorithm families, all seven algorithms map to four resource primitives
at the system level: *C_sample* (environment interaction cost), *C_update*
(gradient computation cost), *M_store* (buffer or rollout memory occupancy),
and *τ* (communication and synchronization overhead).  Tuning
`replay_buffer_size` in an off-policy agent and tuning `n_steps` in an
on-policy agent are both mapped to capacity limits on *M_store*; adjusting DVFS
frequencies imposes a hard power cap on *C_update*.  This unified abstraction
enables the Resource Manager to operate identically regardless of the
underlying algorithm, which is the central systems contribution of TetraRL.

**Implementation roadmap** (per [`docs/action-plan-weekly.md`](docs/action-plan-weekly.md)):

- **Week 1**: Repo skeleton + Hypervolume eval module + GUDA-MountainCar smoke env (✅ Done)
- **Week 2**: **C-MORL (ICLR 2025) Integration** as the foundational building block — reproduce on `Building-3d` benchmark, validate two-stage Pareto discovery (Init + CPO/IPO Constraint stages)
- **Week 3**: 4-D objective vector `[performance, -energy, -memory, -latency]` + `mo-gymnasium` compatible DAG scheduling environment, mapping latency deadlines and memory budgets directly to C-MORL's CPO/IPO constraint thresholds
- **Week 4**: **Action Masking** (mask DVFS actions guaranteed to miss deadlines) + **GNN Feature Extractor** (topology-aware generalization across DAG sets) + Hardware-emergency override layer
- **Weeks 5–8**: Orin AGX systems integration — tegrastats sensor daemon, DVFS controller, four-component framework (Preference Plane / Resource Manager / RL Arbiter / Override), pre-allocated soft-truncation replay buffer, full closed-loop validation with FFmpeg co-runner interference
- **Weeks 9–12**: Multi-platform porting (Xavier NX + Jetson Nano), full evaluation matrix, ablation studies, paper draft for IEEE Transactions on Computers (TC) submission

## Repository Layout

```
TetraRL/
├── README.md
├── LICENSE
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml
├── pyproject.toml
├── setup.py
├── tetrarl/
│   ├── __init__.py
│   ├── core/                          # State/action/agent abstractions
│   │   └── __init__.py
│   ├── envs/                          # Gym/Gymnasium environment wrappers
│   │   └── __init__.py
│   ├── morl/                          # MORL algorithms (PD-MORL primary)
│   │   ├── __init__.py
│   │   ├── pd_morl.py                 # PD-MORL MO-DQN-HER agent
│   │   └── operators.py               # Cosine-similarity envelope operator
│   ├── sys/                           # System layer (DVFS, tegrastats, override)
│   │   ├── __init__.py
│   │   ├── tegra_daemon.py            # Async tegrastats sensor daemon (100Hz/10Hz, EMA filter)
│   │   ├── dvfs.py                    # DVFS frequency controller (Orin AGX, stub mode for dev)
│   │   └── override_layer.py          # Hardware-emergency override
│   ├── eval/                          # Evaluation utilities (HV, Pareto, CDF)
│   │   ├── __init__.py
│   │   ├── hypervolume.py             # Hypervolume indicator computation
│   │   └── tail_latency.py            # Tail-latency CDF analysis
│   └── utils/                         # Logging, config, registry
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── test_operators.py              # Envelope operator unit tests
│   └── test_preference_sampling.py    # Preference sampling tests
├── reproduce/
│   ├── README.md
│   ├── run_smoke.sh
│   ├── run_short.sh
│   └── run_full_paper.sh
├── docs/
│   ├── architecture.md
│   └── action-plan-weekly.md
└── scripts/
    └── README.md
```

## Installation

### On NVIDIA Jetson (JetPack 6.2)

JetPack 6.2 ships with Python 3.10.  Create a virtual environment under the
shared experiment directory:

```bash
python3.10 -m venv /experiment/zexin/venvs/tetrarl
source /experiment/zexin/venvs/tetrarl/bin/activate

# Install PyTorch (>=2.8) from the Jetson AI Lab wheel index (JP6 / CUDA 12.6)
pip install --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
    "torch>=2.8" torchvision torchaudio

# Install TetraRL in editable mode
pip install -e .
```

### On Desktop / CI (x86)

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

Three reproduction tiers are provided, following the same convention as R³:

```bash
# Tier 1 — Smoke test (5-10 min): CartPole DST sanity check
bash reproduce/run_smoke.sh

# Tier 2 — Short run (30-60 min): Pong + R⁴ tracking
bash reproduce/run_short.sh

# Tier 3 — Full paper reproduction (days): all paper-figure experiments
bash reproduce/run_full_paper.sh --list   # catalog available runs first
bash reproduce/run_full_paper.sh --only atari-pong-mosac
```

## Reproducing the IEEE TC Paper

The full 12-week project timeline, including per-week deliverables, validation
criteria, and risk mitigations, is documented in
[`docs/action-plan-weekly.md`](docs/action-plan-weekly.md).

Detailed reproduction instructions for each paper figure and table are provided
in [`reproduce/README.md`](reproduce/README.md).

## Citation

```bibtex
@article{tetrarl2026,
  title     = {TetraRL: A Multi-Objective Runtime System for On-Device
               Reinforcement Learning Training},
  author    = {[Authors TBD]},
  journal   = {IEEE Transactions on Computers (under review)},
  year      = {2026}
}

@inproceedings{r32023,
  title     = {R$^3$: On-Device Real-Time Deep Reinforcement Learning
               for Autonomous Robotics},
  author    = {Li, Zexin and Samanta, Aritra and Li, Yufei
               and Soltoggio, Andrea and Kim, Hyoseung and Liu, Cong},
  booktitle = {2023 IEEE Real-Time Systems Symposium (RTSS)},
  pages     = {131--144},
  year      = {2023},
  organization = {IEEE}
}

@inproceedings{duojoule2024,
  title     = {DuoJoule: Accurate On-Device Deep Reinforcement Learning
               for Energy and Timeliness},
  author    = {Shirvani, Soheil and Samanta, Aritra
               and Li, Zexin and Liu, Cong},
  booktitle = {2024 IEEE Real-Time Systems Symposium (RTSS)},
  year      = {2024},
  organization = {IEEE}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE)
file for details.

## Acknowledgments

<!-- Acknowledgments will be added upon paper acceptance. -->
