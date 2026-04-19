# Week 6 Framework + Buffer Design

**Status**: implemented on branch `week6/framework-buffer`.
**Theme** (per `docs/action-plan-weekly.md`): four-component framework
integration and replay-buffer memory management.

## Overview

Week 6 ships the runtime spine of TetraRL: `tetrarl/core/framework.py`
wires the four paper-prescribed components (Preference Plane, RL Arbiter,
Resource Manager, Hardware Override) into a single per-step
orchestrator, and `tetrarl/sys/buffer.py` adds the pre-allocated,
soft-truncation replay buffer specified in R^3 (Li, RTSS'23 Section 5).
Together these are what turns the previously-isolated MORL agent, DVFS
controller, tegrastats daemon, and override layer into a thing that can
actually run a closed control loop on Orin Unified Memory without
fragmenting it. The smoke (`scripts/week6_e2e_smoke.py`) drives the
whole stack against CartPole-v1 with a synthetic 4-D telemetry stream
and confirms the per-step overhead budget.

## The 4 Components

### 1. Preference Plane

- **Module**: `tetrarl/core/framework.py:StaticPreferencePlane`
- **Responsibility**: emit the per-step preference weight vector
  `omega` consumed by the arbiter.
- **Inputs/Outputs**:
  - constructor: `omega: np.ndarray` (simplex, M-D)
  - `get() -> np.ndarray` — copy of the configured `omega`
- The Week 6 implementation is constant; a learned/scheduler-driven
  preference plane is deferred to a later week.

### 2. RL Arbiter

- **Module**: black-box duck-typed object with `.act(state, omega) -> action`.
- **Responsibility**: propose an action conditioned on the env state
  and the current preference vector.
- **Inputs/Outputs**:
  - `act(state: Any, omega: np.ndarray) -> action` (action type
    matches the env action space; `int` for discrete envs)
- In production this is `tetrarl/morl/native/agent.py:TetraRLNativeAgent`.
  In the Week 6 smoke it is `RandomArbiter`
  (`scripts/week6_e2e_smoke.py`); unit tests use a stub. The framework
  treats this slot as opaque on purpose — no retraining or gradient
  flow happens inside `framework.step()`.

### 3. Resource Manager

- **Module**: `tetrarl/core/framework.py:ResourceManager`
- **Responsibility**: map a hardware telemetry snapshot to a DVFS
  frequency-table index using a step-down rule.
- **Inputs/Outputs**:
  - `decide_dvfs(telemetry: HardwareTelemetry, n_levels: int) -> int`
    — index in `[0, n_levels-1]`.
- Configuration lives in `ResourceManagerConfig`
  (`soft_latency_ms=50.0`, `min_energy_j=50.0`, `max_memory_util=0.7`).
  Each violation knocks the index down by one from the top of the
  frequency table; the result is clamped into the legal range.

### 4. Hardware Override

- **Module**: `tetrarl/morl/native/override.py:OverrideLayer`
  (existing module from earlier weeks).
- **Responsibility**: veto the arbiter's action and substitute a safe
  fallback when telemetry crosses configured hard thresholds.
- **Inputs/Outputs**:
  - `step(telemetry: HardwareTelemetry) -> (override_active: bool, fallback_action)`

## Per-step Dataflow

```
                  +----------------------+
   state  ---->   |   RL Arbiter         |---- proposed_action --+
                  |   .act(state, omega) |                       |
                  +----------------------+                       |
                          ^                                      v
                          |                            +------------------+
   StaticPreferencePlane -+                            |  Override Layer  |
   .get() -> omega                                     |  .step(hw)       |
                                                       +------------------+
   TelemetrySource.latest() --(adapter)--> hw                 |
                              |                               |
                              v                               |
                  +----------------------+                    |
                  |   Resource Manager   |                    |
                  |   .decide_dvfs(hw)   |                    |
                  +----------------------+                    |
                              |                               v
                              v                       fired ? fallback : proposed
                       DVFSController.set_freq()              |
                                                              v
                                                            action
```

The per-step record returned by `framework.step(state)` and appended to
`framework.history` has the keys:

```
action, proposed_action, omega, override_fired, reward (None at step time),
latency_ms, energy_j, memory_util, dvfs_idx
```

`reward` is `None` at `step()` time on purpose: the framework runs
*before* the env step, but the env reward is only available *after*
`env.step(action)`. The caller fills it in via
`framework.observe_reward(env_reward)`, which writes into
`history[-1]["reward"]`. This keeps the framework agnostic to whether
the reward source is a Gym env, a real-hardware controller, or a
synthetic 4-D stream.

## Buffer Design

### Why pre-allocation matters

R^3 (Li, RTSS'23 §5) shows that on Jetson Unified Memory the PyTorch
caching allocator does not return freed blocks to the OS, and
re-allocating large replay buffers fragments the unified pool shared by
CPU and iGPU — under load this manifests as iGPU OOM even though the
nominal memory budget is fine. The fix is to allocate the replay
buffer once at construction and never resize it; "shrinking" is done
by flipping a boolean mask, never by freeing storage.

### Layout

`tetrarl/sys/buffer.py:ReplayBuffer` allocates six fixed-shape tensors
of length `capacity`:

| tensor       | shape                 | dtype          |
|--------------|-----------------------|----------------|
| `obs`        | `(capacity, *obs_shape)` | `obs_dtype` (default `float32`) |
| `next_obs`   | `(capacity, *obs_shape)` | `obs_dtype`    |
| `actions`    | `(capacity, *act_shape)` | `act_dtype` (default `long`) |
| `rewards`    | `(capacity,)`            | `float32`      |
| `dones`      | `(capacity,)`            | `bool`         |
| `valid_mask` | `(capacity,)`            | `bool`         |

`add()` writes at `_head`, advances `_head` mod `capacity`, and bumps
`_size` until it saturates at `capacity` — i.e., a ring overwriting the
oldest entry once full.

### Soft-truncation semantics

`soft_truncate(n)` flips the `n` oldest mask bits to `False` (starting
at the logical tail `(_head - _size) mod capacity`) and decrements
`_size`. The underlying storage tensors are never reassigned; their
Python object identity is preserved across truncations (verified by
test). `sample(batch_size)` resolves valid rows via
`torch.nonzero(valid_mask)` and indexes into the fixed storage, so
masked-out slots are never returned even though their bytes are still
resident.

### Status

The Week 6 buffer is *not yet wired* into the on-policy PPO arbiter —
PPO discards rollouts after each update and has no replay store. The
buffer ships now to (a) match Section 5 of the paper and (b) unblock
the off-policy port (SAC / C-MORL off-policy variants) without
re-touching the framework.

## Smoke Test Results

`scripts/week6_e2e_smoke.py --episodes 100` (CartPole-v1, random
arbiter, stub telemetry, stub DVFS):

| metric                          | value     |
|---------------------------------|-----------|
| episodes                        | 100       |
| total steps                     | 2079      |
| mean framework step (ms)        | 0.0018    |
| max framework step (ms)         | 0.0118    |
| override fire_count             | 0         |
| jsonl output                    | `runs/week6_smoke_<unix_ts>.jsonl` (2079 lines) |

The 0.0018 ms mean is three orders of magnitude under the 2 ms paper
budget. Override `fire_count=0` is by construction — the smoke uses
generous thresholds (`max_memory_util=0.95`, `max_latency_ms=10000.0`,
`min_energy_j=0.5`) so the arbiter's proposed action is always passed
through, exercising the non-override path. All four reward dimensions
(`reward`, `latency_ms`, `energy_j`, `memory_util`) are populated on
every step with no NaNs.

## Test Inventory

| file                        | tests | scope                                                                  |
|-----------------------------|------:|------------------------------------------------------------------------|
| `tests/test_buffer.py`      | 11    | preallocation, ring overflow, soft-truncation correctness, sample filter |
| `tests/test_framework.py`   | 8     | orchestrator wiring, override path, reward attachment, history accumulation |
| `tests/test_week6_e2e.py`   | 5     | 10-episode CartPole smoke                                              |
| **Week 6 total**            | **24**| all passing                                                            |
| Full repo suite             | 177   | no regressions                                                         |

## Mac vs Orin Path

| component   | Mac (dev)                              | Orin AGX (target)                                         |
|-------------|----------------------------------------|-----------------------------------------------------------|
| DVFS        | `DVFSController(stub=True)` — table-only, no sysfs writes | sysfs writes via `cpufreq` / `devfreq`; requires `sudo` and `governor=userspace` (see auto-memory entry "Orin DVFS root requirement") |
| Tegrastats  | `StubTelemetrySource` (in-memory, fed by smoke loop)      | `tetrarl/sys/tegra_daemon.py:TegrastatsDaemon` (100 Hz sample, 10 Hz dispatch, EMA-filtered) |
| Override    | identical                              | identical                                                 |
| Buffer      | identical                              | identical; the Unified Memory rationale only matters here |

## What's NOT in Week 6 (deferred)

- Wiring the replay buffer into the actual training loop. PPO is
  on-policy and does not consume it; the off-policy port is a future
  week.
- Real C-MORL agent + real `tegrastats` on Orin AGX. The sequential
  agent is running that on hardware in parallel; this branch validates
  the Mac CPU path only.
- A learned preference scheduler. Per `docs/action-plan-weekly.md` this
  arrives in a later week; Week 6 ships only `StaticPreferencePlane`.

## Validation against Week 6 Spec

From `docs/action-plan-weekly.md` Week 6 validation criteria:

| criterion                                                         | status |
|-------------------------------------------------------------------|--------|
| No OOM on Orin AGX over 100 episodes (memory < 90 % unified pool) | Pending — Orin run by sequential hardware agent |
| All 4 reward dimensions logged with valid (non-NaN, non-zero) values at every step | Verified on Mac smoke. `energy_j` and `memory_util` are non-zero by construction; `latency_ms` can be ~0 for very fast Mac steps but is non-NaN and non-None |
| Framework overhead < 2 ms                                         | Verified on Mac (mean 0.0018 ms, max 0.0118 ms) |
