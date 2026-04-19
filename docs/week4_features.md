# Week 4 Features: Masking, GNN, and Hardware Override

## Goal

Week 4 adds three optional, default-OFF extension points to the TetraRL
native preference-conditioned PPO agent: per-step discrete-action masking
(for DVFS-style deadline constraints), a pure-PyTorch GCN feature
extractor (for graph-structured observations), and a post-policy
hardware override layer (for runtime safety). All three plug into
`TetraRLNativeAgent` via boolean flags so the existing baseline
(`tests/test_native_dst.py`) continues to pass without changes.

## Module 1: Action Masking

The masking subsystem lives in `tetrarl/morl/native/masking.py` and
exposes a small strategy-pattern interface:

- `ActionMask` (ABC) — `compute(state, act_dim) -> bool ndarray` of
  shape `(act_dim,)`. `as_tensor(...)` returns the same mask as a
  `torch.bool` tensor on the requested device.
- `NoOpMask` — identity mask; safe default for envs without deadlines.
- `DeadlineMask` — DVFS-aware heuristic. Actions are ordered slowest to
  fastest by `freq_scale`; predicted execution time is
  `latency_ms / freq_scale[a]`. Anything predicted to miss
  `deadline_ms` is masked. The latency EMA is updated online via
  `update_latency(observed_ms)`.

Internally the rollout calls `mask.as_tensor(obs, act_dim, device)`
once per step and `apply_logit_mask` rewrites disallowed action logits
with `-1e9` (not `-inf`, to keep the softmax finite).

```python
from tetrarl.morl.native.agent import TetraRLNativeAgent
from tetrarl.morl.native.masking import DeadlineMask

mask = DeadlineMask(freq_scale=[0.5, 1.0, 1.5], deadline_ms=2.0)
agent = TetraRLNativeAgent(env_name="dst", use_masking=True, action_mask=mask)
agent.train()
```

`DeadlineMask` guarantees the mask is never empty: if every action
would miss the deadline, the fastest action (argmax of `freq_scale`)
stays allowed. The policy therefore never sees a zero-probability
action set.

## Module 2: GNN Feature Extractor

`tetrarl/morl/native/gnn_extractor.py` provides a pure-PyTorch GCN
based on the symmetric normalization of Kipf & Welling (2017):

```
H' = sigma(D^-1/2 (A + I) D^-1/2 H W)
```

It is a two-layer GCN followed by per-graph mean / sum / max pooling.
Edge index uses the standard `(2, E)` COO convention; batched graphs
are encoded by a `(N,)` `batch` LongTensor (PyG-compatible) so we can
swap in `torch_geometric` later without touching call sites. There is
no `torch_geometric` dependency today.

`PreferenceNetwork` accepts a `gnn_extractor` kwarg and sizes its
heads to `gnn_out_dim + pref_dim`. **However**, integration with
`TetraRLNativeAgent` is wiring-only as of Week 4: no current env (DST,
mo-gymnasium) produces graph observations. The rollout still concatenates
flat `obs` with `omega`, which would shape-mismatch against the GNN-sized
heads on the first forward pass.

To make this fail loudly rather than mysteriously, `train_preference_ppo`
raises `NotImplementedError` as soon as a non-`None` `gnn_extractor` is
passed. End-to-end use of the GCN therefore requires a graph-aware env,
which is Week 5+ scope. The extractor itself is fully unit-tested as a
standalone module (`tests/test_gnn_extractor.py`).

## Module 3: Hardware Override Layer

`tetrarl/morl/native/override.py` provides a runtime-safety layer
decoupled from the policy gradient.

- `HardwareTelemetry` — snapshot dataclass with optional
  `latency_ema_ms`, `energy_remaining_j`, `memory_util` fields.
- `OverrideThresholds` — corresponding optional limits. A field that
  is `None` is never checked.
- `OverrideLayer.step(telemetry) -> (override_active, fallback)` —
  evaluates thresholds; on violation, fires immediately, arms a
  `cooldown_steps` counter, and returns `(True, fallback_action)`. On
  subsequent calls, while the cooldown counter is non-zero, it keeps
  firing and decrementing even if telemetry recovers — this is the
  hysteresis that prevents oscillation around the threshold edge.
  Otherwise returns `(False, None)`.

The override is **post-policy**: in `train_preference_ppo`, the policy
samples an action and stores its own logprob/value/reward as usual.
The executor then asks `OverrideLayer.step(telemetry_fn())` whether to
swap in `fallback_action` before calling `env.step(...)`. The training
buffers contain the policy's proposed action and the resulting reward.
The gradient is therefore identical to the no-override case.
`results["override_fire_count"]` exposes the audit counter.

## Backward Compatibility

All three flags (`use_masking`, `use_override`, `use_gnn`) default to
`False`. With defaults the `TetraRLNativeAgent` constructor and
`train()` call are byte-identical to Week 3. The pre-existing baseline
test `tests/test_native_dst.py` runs unchanged.

## Test Coverage

| File                              | Tests |
| --------------------------------- | ----- |
| `tests/test_masking.py`           | 13    |
| `tests/test_gnn_extractor.py`     | 10    |
| `tests/test_override.py`          | 11    |
| `tests/test_week4_integration.py` | 4     |
| `tests/test_native_dst.py`        | 7     |
| **Total**                         | **45** |

## Roadmap (Week 5)

- A graph-aware env (DAG task model with node features + edges) so the
  `gnn_extractor` integration path can be exercised end-to-end and the
  `NotImplementedError` guard can be removed.
- A real DVFS-aware `DeadlineMask` plug-in tied to the embedded target's
  available frequency steps, with online calibration of `freq_scale`
  from measured per-step latency.
- A telemetry adapter for Orin AGX that reads `sysfs` / `tegrastats`
  and produces `HardwareTelemetry` snapshots, so `OverrideLayer` can be
  evaluated against real thermal and energy traces.
