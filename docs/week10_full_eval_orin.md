# Week 10 — Full Eval Matrix on Orin AGX

## 1. Methodology
- Hardware: Orin AGX (port 8002), 12-core ARM, governor set to `userspace` for the duration of the runs, then reset to `schedutil`.
- Working dir on Orin: `/experiment/zexin/TetraRL/`. Mutex held at `/experiment/zexin/TetraRL/.orin_busy_w10.lock` for the entire run, released at end (Task 7).
- Env: `dag_scheduler_mo-v0` (DAG scheduler, 4-D vector reward; scalarised through `MOAggregateWrapper` using `omega @ reward_vec`).
  - Note: PyBullet `HalfCheetahBulletEnv-v0` was the second env in the spec but is NOT in the runtime; matrix dropped to DAG-only fallback per spec line 54 ("If PyBullet env not yet integrated... fall back to DAG-scheduler-MO only").
- Agents (8): `preference_ppo` (TetraRL Native, treated as PD-MORL surrogate), `envelope_morl`, `ppo_lagrangian`, `focops`, `duojoule`, `max_action` (MAX-A), `max_performance` (MAX-P), `pcn`. Each baseline is a behavioral surrogate (deterministic / omega-conditioned categorical) intended to capture the *shape* of each method's policy on a discrete-action env, not a full re-implementation.
- 5 ω vectors covering Pareto-front corners + uniform centroid:
  - `[1,0,0,0]` reward-only
  - `[0,1,0,0]` latency-only
  - `[0,0,1,0]` memory-only
  - `[0,0,0,1]` energy-only
  - `[0.25,0.25,0.25,0.25]` uniform
- 3 seeds (0, 1, 2). 50 episodes per (agent, env, ω, seed).
- Total: 8 × 1 × 5 × 3 = **120 runs**.
- Telemetry source: `_MacStubTelemetry` (the runner's stub). The real `TegrastatsDaemon` is not wired through `EvalRunner` (lives in `tetrarl/sys/tegra_daemon.py` and is exercised only by platform-specific scripts under `scripts/`). The runner emits a RuntimeWarning when `platform="orin_jetson"` to make this explicit. **Wall-clock latency in the JSONLs is real (measured around the env+framework step on Orin); energy and memory are stub-synthesized.** This affects the violation table interpretation (see Limitations).

## 2. Pareto-front matrix
- Matrix YAML: `tetrarl/eval/configs/w10_full_matrix_orin.yaml` (auto-generated via `scripts/week10_make_matrix_yaml.py`).
- Per-run JSONLs: `runs/w10_orin_full/none__<agent>__seed<N>__o<I>.jsonl`.
- Long-form summary: `runs/w10_orin_full/summary.csv`.
- Wall-clock: 18.6 s for the full 120-run sweep.

## 3. Hypervolume comparison vs 7 MORL baselines (Task 2)
- Reference point used (4-D, after the "all higher = better" sign-flip on latency / memory / energy): `(-0.1, -1.0, -0.15, -0.01)`.
  - Observed per-step ranges across the matrix were reward∈[0,1], latency∈[0.04,0.6] ms, memory_util∈[0.10,0.13], energy∈[0.001,0.006] J/step. The ref point sits just below the worst-observed value on each axis so HV is informative on every dimension.
- Implementation: `tetrarl/eval/hv.py` (`compute_run_hv`, `welch_pvalue`).
- HV chart: `runs/w10_orin_full/hv_comparison.png` + `.svg`. Comparison Markdown: `runs/w10_orin_full/hv_comparison.md`. Long-form CSV: `runs/w10_orin_full/hv_comparison.csv`.

Verbatim copy of the Markdown table (n=15 per agent = 5 ω × 3 seeds):

| Method | mean HV | std | n | p-value vs preference_ppo |
| --- | ---: | ---: | ---: | ---: |
| preference_ppo | 0.000022 | 0.000013 | 15 | - |
| duojoule | 0.000019 | 0.000007 | 15 | 0.4695 |
| envelope_morl | 0.000027 | 0.000010 | 15 | 0.2847 |
| focops | 0.000016 | 0.000014 | 15 | 0.2662 |
| max_action | 0.000054 | 0.000094 | 15 | 0.2061 |
| max_performance | 0.000054 | 0.000094 | 15 | 0.2061 |
| pcn | 0.000053 | 0.000074 | 15 | 0.1336 |
| ppo_lagrangian | 0.000019 | 0.000019 | 15 | 0.5573 |

Discussion:
- No baseline reaches Welch p<0.05 against TetraRL Native at n=15. The closest gap is PCN (p=0.13).
- The MAX-A / MAX-P / PCN surrogates score numerically higher in mean HV, with very wide std (std/mean ≈ 1.7), reflecting these surrogates committing to a single high-reward arm and getting either lucky or unlucky on a per-omega basis.
- The principled MORL baselines (duojoule / envelope_morl / focops / ppo_lagrangian) cluster within ±25% of TetraRL Native's mean.
- Caveat: these are **behavioral surrogates** of the baselines built to fit the W10 evaluation timeline, not the canonical training implementations of each method. The HV ranking should be read as an order-of-magnitude sanity check, not a publication-grade comparison. Real implementations of FOCOPS / Envelope MORL etc. would likely separate further from the surrogates after several thousand training steps.

## 4. Reward-vs-walltime + reward-vs-energy curves (Task 3)
- `runs/w10_orin_full/reward_vs_walltime.png` / `.svg`
- `runs/w10_orin_full/reward_vs_energy.png` / `.svg`
- 40 (agent × ω) groups (8 agents × 5 ω) plotted; each line is the mean across 3 seeds.
- Per the §9.6 pitfall, both x-axes are real-time / energy axes (no `Reward vs Steps`).

## 5. Dynamic preference switching demo (Task 4)
- Script: `scripts/week10_dynamic_pref_switch.py` (uses local `MutableStaticPreferencePlane`).
- Setup: 100 episodes, switch at episode 50.
  - pre-switch ω: `[0.5, 0.5, 0.0, 0.0]` (reward + latency)
  - post-switch ω: `[0.0, 0.0, 0.2, 0.8]` ("low-battery": memory + energy heavy)
- Outputs:
  - `runs/w10_orin_full/dynamic_pref_switch.png` / `.svg` (time-series)
  - `runs/w10_orin_full/dynamic_pref_switch.csv` (per-episode)
  - `runs/w10_orin_full/dynamic_pref_switch_summary.md` (acceptance summary)
- Result:
  - pre-switch mean reward (last 10 eps before switch): **9.300**
  - post-switch mean reward (eps 60..69, after a 10-episode adjustment window): **9.100**
  - reward-collapse criterion (post < 0.5 * pre = 4.65): **PASSED** (9.10 ≥ 4.65, no collapse)
  - Smooth transition < 10 episodes ✅

## 6. Lagrangian constraint-violation table (Task 5)
- Script: `scripts/week10_lagrangian_violation_table.py`. Sweeps:
  - `tetrarl/eval/configs/w10_violation_orin_off.yaml` (15 runs, ablation=`override_layer` → override_off)
  - `tetrarl/eval/configs/w10_violation_orin_on.yaml` (15 runs, ablation=`none` → override_on)
  - Combined for analysis: `tetrarl/eval/configs/w10_violation_orin_combined.yaml`.
- Per-step violation thresholds: latency > 0.2 ms OR memory_util > 0.115 OR cumsum(energy_j) > 5.0 J.
- Outputs:
  - `runs/w10_orin_full/lagrangian_violation_table.md`
  - `runs/w10_orin_full/lagrangian_violation_table.csv`

| Variant                              | Violation Rate | Std    | N Runs |
| ------------------------------------ | -------------- | ------ | ------ |
| Lagrangian only (no override)        |          0.471 |  0.084 |     15 |
| Lagrangian + override                |          0.471 |  0.084 |     15 |

Spec expectation was "Lagrangian alone violates ~25%, override + Lagrangian < 5%". The empirical result is identical (47.1%) for both variants. **This is a known artefact of the synthetic stub telemetry** — `_MacStubTelemetry` produces an action-independent telemetry stream (`memory_util = 0.1 + 0.001 * episode_step`, energy synthesised, latency reflects only env+framework wall-clock), so the override layer has no signal to respond to. `override_fire_count == 0` for every preference_ppo run in `summary.csv` confirms the override never triggered. Closing this gap requires wiring the real `TegrastatsDaemon` through `EvalRunner` (currently only the platform-specific scripts under `scripts/run_orin_*.py` build it directly). Tracked under Limitations.

## 7. Reference list of generated artefacts
- `runs/w10_orin_full/summary.csv` — long-form per-run summary.
- `runs/w10_orin_full/none__<agent>__seed<N>__o<I>.jsonl` — 120 raw per-step JSONL traces.
- `runs/w10_orin_full/hv_comparison.{png,svg,md,csv}` — Task 2 artefacts.
- `runs/w10_orin_full/reward_vs_walltime.{png,svg}` — Task 3a.
- `runs/w10_orin_full/reward_vs_energy.{png,svg}` — Task 3b.
- `runs/w10_orin_full/dynamic_pref_switch.{png,svg,csv}` + `dynamic_pref_switch_summary.md` — Task 4.
- `runs/w10_orin_full/lagrangian_violation_table.{md,csv}` — Task 5.
- Matrix YAMLs: `tetrarl/eval/configs/w10_full_matrix_orin.yaml`, `w10_violation_orin_{off,on,combined}.yaml`.

## 8. Known limitations
- **PyBullet env not in this PR.** `HalfCheetahBulletEnv-v0` is missing from the runtime. Matrix is DAG-only (45→120 swap because we ran 8 agents instead of the spec-default 3). Tracked separately.
- **Nano data deferred.** `nano2` is unreachable (per spec line 17). Nano-side work will land in a separate `w10-nano-eval` PR when nano2 recovers.
- **Stub telemetry.** `EvalRunner` uses `_MacStubTelemetry` even on Orin. Wall-clock latency is real; energy and memory are synthesized and action-independent. Concretely affects:
  - the `-mean_energy` / `-mean_memory` axes of the HV computation (degenerate axes near constant)
  - the override-on/off violation table (override never fires under stub telemetry; empirical override_off ≡ override_on)
  Wiring the real `TegrastatsDaemon` through the runner is the unblocker here.
- **Behavioral baselines, not full reimplementations.** Each baseline is a deterministic / omega-conditioned categorical surrogate that captures the *shape* of the method on a discrete-action env. Use the HV ranking as an order-of-magnitude sanity check, not a publication-grade comparison.
