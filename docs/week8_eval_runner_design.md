# Week 8 — Unified Evaluation Runner Design

## 1. Motivation

The TetraRL paper requires two evaluation tables grounded in uniform
methodology: Table 4 (ablation study, ≥4 rows × 4 metrics) and Table 5
(per-component overhead). Both tables compare runs of the same
framework with one component swapped for a null variant, then diff
metrics across rows. If each row is produced by a hand-rolled script,
the diffs are apples-to-oranges: the runs use different RNG seeds,
different telemetry shapes, different aggregation windows, and write
incompatible artefact formats. Reviewers cannot reproduce the table.

The Week 8 deliverable, `tetrarl/eval/runner.py`, replaces the
hand-rolled scripts with a single `EvalRunner` driven by a YAML sweep.
Every run uses the same `TetraRLFramework` wiring path, the same Mac
stub or Jetson telemetry adapter, and the same RNG seeding policy. The
runner emits a uniform per-step JSONL plus a per-sweep `summary.csv`
that downstream analysis (the `scripts/week7_make_cdf.py`-style
aggregators) can ingest as a single artefact.

The runner explicitly stays platform-agnostic: the same code drives
Mac smoke runs and Orin/Nano physical runs. The platform field
controls only the telemetry / DVFS factory, never the loop structure.

## 2. EvalConfig schema

Each row of a sweep is one `EvalConfig`:

| Field         | Type        | Notes                                                |
| ------------- | ----------- | ---------------------------------------------------- |
| `env_name`    | `str`       | Gymnasium env id, e.g. `CartPole-v1`.                |
| `agent_type`  | `str`       | `random`, `fixed`, `preference_ppo`.                 |
| `ablation`    | `str`       | `none`, `preference_plane`, `resource_manager`, `rl_arbiter`, `override_layer`. |
| `platform`    | `str`       | `mac_stub`, `orin_agx`, `nano`.                      |
| `n_episodes`  | `int`       | Number of Gym episodes per run.                      |
| `seed`        | `int`       | Both `np.random` and `random` are seeded.            |
| `out_dir`     | `Path`      | JSONL + CSV output directory.                        |
| `extra`       | `dict`      | Free-form, reserved for env-specific knobs.          |

Example YAML config block:

```yaml
configs:
  - env_name: CartPole-v1
    agent_type: random
    ablation: none
    platform: mac_stub
    n_episodes: 50
    seed: 0
    out_dir: runs/w8_ablation_smoke
```

`EvalConfig.from_dict`/`to_dict` and `from_yaml`/`to_yaml` provide a
round-trippable representation; `load_sweep_yaml(path)` returns a
`list[EvalConfig]`.

## 3. Ablation factory wrappers

The framework exposes four pluggable slots; each ablation arm
substitutes one slot with a documented null variant:

```
TetraRLFramework
├── preference_plane  ──► StaticPreferencePlane (real)
│                         _NullPreferencePlane (uniform omega when ablation=preference_plane)
├── resource_manager  ──► ResourceManager (real)
│                         _NullResourceManager (always max idx when ablation=resource_manager)
├── rl_arbiter        ──► _PreferencePPOArbiter | _RandomArbiter | _FixedActionArbiter
│                         (forced _RandomArbiter when ablation=rl_arbiter)
└── override_layer    ──► OverrideLayer (real)
                          _NullOverrideLayer (never fires when ablation=override_layer)
```

The factory functions `_make_preference_plane`, `_make_resource_manager`,
`_make_rl_arbiter`, and `_make_override_layer` each take the
`ablation` string (and `agent_type` for the arbiter) and return either
the real implementation or its `_Null...` counterpart. Because all
four factories share the same string key space, exactly one of the
four slots is replaced per run, which is what the table-4 ablation
analysis needs.

> Default `omega` for the real `StaticPreferencePlane` is `[0.7, 0.3]` (intentionally non-uniform) so omega-aware arbiters such as `_PreferencePPOArbiter` produce different action distributions when the preference plane is ablated to its uniform `_NullPreferencePlane` variant.

## 4. How to add a new ablation arm

1. Extend the `AblationArm` literal (or its equivalent string set) to
   include the new arm name, e.g. `"reward_shaper"`.
2. Implement a `_NullX` class that satisfies the same protocol as the
   real component (same method names, same return types). Document
   what "null" means for that component.
3. Extend the matching `_make_X(ablation)` factory: add a guard
   `if ablation == "<new_arm>": return _NullX(...)` before the
   real-component fallthrough.
4. Add a unit test in `tests/test_eval_runner.py` modelled on
   `test_ablation_preference_plane_uses_null_variant`: build a config
   with the new arm, call `EvalRunner()._build_framework(cfg)`, and
   assert both the swapped slot's class name and the null behaviour
   (e.g. uniform omega, zero fire count, max DVFS index, etc.).
5. Add the new arm to `tetrarl/eval/configs/ablation_smoke.yaml` (and
   the `ablation_full.yaml` template) so the smoke catches regressions.

## 5. Mac vs Orin/Nano differences

Only the `platform` field changes the construction of the telemetry
source and DVFS controller. `_make_telemetry(platform)` dispatches:

- `mac_stub` -> `_MacStubTelemetry` + `_telemetry_to_hw` adapter, no
  DVFS controller.
- `orin_agx` / `nano` -> `TegrastatsDaemon`-backed source + a real
  `DVFSController` constructed by the platform-specific scripts (which
  require `sudo` and `governor=userspace`; see the user memory note
  on Orin DVFS root requirements).

The eval loop, JSONL schema, summary CSV, and seeding policy in
`runner.py` are identical across platforms. Physical scripts simply
import the runner and pass an `EvalConfig` whose `platform` field is
`orin_agx` or `nano`.

> **mac_stub limitations.** The synthetic 4-D telemetry on `mac_stub` is open-loop: DVFS decisions issued by the `ResourceManager` are not fed back into the synthetic latency/energy stream, so the `resource_manager` ablation arm produces identical functional metrics (mean_reward, override_fire_count, mean_energy_j, mean_memory_util) to the `none` baseline. On Orin (`platform=orin_agx`) the DVFS decision physically scales GPU/CPU frequency, which moves measured energy and latency, so all 5 arms produce distinct results. Smoke runs on Mac validate component WIRING (verified by `_build_framework` class identity in tests 5–9); functional separation of all 5 arms is validated on Orin in the W8 ablation-orin spec.

## 6. Out of scope for this PR

- Actual ablation experiments on Orin (handled by the W8
  ablation-orin spec).
- Overhead measurements on Nano (handled by the W8 overhead-nano
  spec).
- Hypervolume (HV) computation for multi-objective envs. `RunResult.hv`
  is `None` for CartPole (single-objective); a multi-objective
  computation will land alongside the DST/multi-objective sweeps.
- Trained Preference-PPO arbiter. `_PreferencePPOArbiter` is a
  Mac-side stub with a seeded RNG over the discrete action space; the
  W8 ablation-orin agent will swap in the trained checkpoint without
  changing the runner API.

## 7. CLI usage

```bash
# Run a sweep YAML directly via the runner module
python -m tetrarl.eval.runner --config tetrarl/eval/configs/ablation_smoke.yaml

# Override the output directory for all configs
python -m tetrarl.eval.runner --config tetrarl/eval/configs/ablation_smoke.yaml --out-dir runs/my_eval

# Mac smoke wrapper that prints a Markdown summary table
python scripts/week8_eval_smoke.py
```

## 8. Output schema

### Per-step JSONL (`<ablation>__<agent_type>__seed<seed>.jsonl`)

One record per env step. Keys:

- `episode` — episode index within the run.
- `step` — step index within the episode.
- `action` — discrete action taken after override gating.
- `reward` — env reward at this step.
- `latency_ms` — framework-step + env-step wall time, in ms.
- `energy_j` — synthetic per-step energy delta (Mac stub) or measured
  delta (Jetson).
- `memory_util` — fraction in [0, 1].
- `omega` — list-form preference vector (currently 2-D).
- `override_fired` — bool, true when the override layer triggered.
- `dvfs_idx` — DVFS index chosen by the resource manager (None on
  Mac stub).

### Per-sweep summary CSV (`summary.csv` in the sweep `out_dir`)

One row per `EvalConfig`. Columns:

- `env_name`
- `agent_type`
- `ablation`
- `platform`
- `seed`
- `n_episodes`
- `n_steps`
- `mean_reward`
- `override_fire_count`
- `tail_p99_ms`
- `mean_energy_j`
- `mean_memory_util`
- `wall_time_s`
