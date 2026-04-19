# DAG Scheduler Env: GCN Integration Design

## Goal

This env exists specifically to validate the GCN feature extractor that
landed in PR #11 (`tetrarl/morl/native/gnn_extractor.py`). At Week 4
the GCN was wired into `PreferenceNetwork` and unit-tested as a
standalone module, but it had no integration test against a real
graph-structured multi-objective env -- the agent constructor accepted
`use_gnn=True` and the network would build, but no rollout had ever
actually pushed `(node_features, edge_index, batch)` tensors through a
full PPO loop and back into a deterministic eval. `DAGSchedulerEnv`
closes that gap with a minimal, deterministic, Gymnasium-compatible env
that exercises the entire graph code path end-to-end.

## Why a synthetic env, not a real DAG

Real DAG scheduling simulators (AppleSched, GraphRL benchmarks, Pegasus
workflow traces) carry heavyweight dependencies, opaque state, and
non-trivial action semantics that would dominate any debugging signal
coming out of the GCN integration. We want a minimal env that exercises
the GNN code path with deterministic, seed-reproducible graphs so any
training instability is unambiguously the policy or the extractor, not
the simulator.

## Env Spec

- **Observation**: `gym.spaces.Dict({...})` with
  - `node_features`: `Box(N, 4)` float32 -- per-node features
    `(compute_cost, memory_cost, deadline_window, done_flag)`.
  - `edge_index`: `Box(2, E_max)` int64, padded with `-1` so the shape
    is fixed across episodes (`E_max = N * (N-1) / 2`).
  - `num_edges`: `Box` scalar int64 -- the actual number of valid edges
    (the rest of `edge_index` is padding).
  - `valid_mask`: `Box(N,)` int8 -- 1 for ready (not-done and all
    predecessors done), 0 otherwise.
- **Action**: `Discrete(N)` -- pick which task to schedule next.
- **Reward**: `[throughput, -energy, -peak_memory]`, 3 objectives, all
  maximization-aligned. Per-step semantics:
  - `throughput`: `+1` on a successful schedule, `0` on a no-op
    (invalid action).
  - `energy`: `-compute_cost[a]` on success, `0` on no-op.
  - `peak_memory`: `-(current_cumulative_memory - prior_peak)` when the
    current cumulative live memory crosses a new high-water mark, else
    `0`.
- **Termination**: all `N` tasks done (terminated). **Truncation**:
  `4*N` steps elapsed.

## DAG Generator

`generate_random_dag(n_tasks, density, rng)` iterates ordered pairs
`(u, v)` with `u < v` and samples each edge with probability
`density`. Acyclicity is guaranteed structurally by the topological
numbering -- no separate DAG-ification pass is needed. Determinism
comes from the caller passing `np.random.default_rng(seed)`. Per-task
costs `(compute, memory, deadline)` are independent uniforms drawn from
the same generator.

## Integration with `PreferenceNetwork`

Stage 2 added a `graph_obs={node_features, edge_index, batch}` plus
`omega` keyword path to `PreferenceNetwork.get_value`,
`get_action_and_value`, and `get_deterministic_action`. The graph path
runs the GCN extractor over `(node_features, edge_index, batch)` and
concatenates the resulting graph-level embedding with `omega`, replacing
the flat `obs_aug = concat(obs, omega)` concat used by the legacy MLP
path.

In `train_preference_ppo`:

- **Per-step rollout**: a single graph (`batch=None`) flows through the
  network, producing one logit / value / action per timestep.
- **Per-minibatch PPO update**: `B` rollout-step graphs are merged into
  one big disjoint graph by stacking node features (`(B*N, F)`),
  concatenating edge tensors with offsets `+ k * n_nodes`, and emitting
  a `batch` LongTensor of length `B*N` whose entries mark the source
  graph index. Pooling in the GCN reduces this back to `(B, gnn_out)`
  before concatenation with the per-step `omega`.

Backward compat is preserved: flat envs (DST, `mo-mountaincar`) still
take the old `obs_aug` path and never construct a `graph_obs` dict.
The selector is `is_graph = isinstance(env.observation_space,
gym.spaces.Dict)` and is sanity-checked against the presence of
`gnn_extractor` (mismatch raises immediately).

## Action Masking

`DAGReadyMask` is the env-specific `ActionMask` implementation. It
reads `state["valid_mask"]` directly from the dict observation and
returns a `(N,)` bool ndarray. From there the existing
`apply_logit_mask` machinery in `PreferenceNetwork` rewrites masked
logits to `-1e9`, so no change to the `ActionMask` ABC was needed --
`compute(state, act_dim)` already accepts an arbitrary state object.

## Sample HV results

```
N tasks         : 8
density         : 0.3
total timesteps : 30000
final HV        : 2435.188
|PF|            : 9
runtime         : 7.23 s
GNN params      : 1216
```

HV history (selected rows from `hv_history`):

| global_step | HV       | \|PF\| |
| ----------: | -------: | -----: |
|       2,560 | 1230.212 |      9 |
|       5,120 | 2425.467 |      4 |
|       7,680 | 2433.990 |      3 |
|      12,800 | 1224.535 |      9 |
|      20,480 | 1228.529 |      9 |
|      25,600 | 2435.188 |      9 |
|      29,952 | 1223.208 |      8 |

The 30k-step run is intentionally short -- enough to confirm the GCN
code path actually moves the policy off the all-zeros baseline (HV>0)
and finds non-degenerate trade-offs (|PF|>1), but not enough to
converge. HV oscillates between ~1.2k and ~2.4k as the policy bounces
between high-throughput-narrow-spread and lower-throughput-wider-spread
regimes; the agent retains the best-HV checkpoint.

## Limitations

- Synthetic costs (uniform sampling) -- not derived from real workload
  traces.
- Single resource per node (no heterogeneous CPU/GPU dispatch).
- Memory penalty is cumulative-completed, not lifetime-aware -- once a
  task is "done" its memory still counts toward the running peak,
  which is a stand-in for a proper allocate/free model.

## Roadmap

- Real DAG benchmark (e.g., parallel scientific workflows from Pegasus
  traces) so the env reflects realistic edge density and skewed cost
  distributions.
- Heterogeneous resource model (per-task `(compute, memory)` per
  device class).
- Deadline-aware scheduling -- consume `deadline_window` as a hard
  constraint with a tardiness penalty objective.
