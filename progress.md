# Week 9 — Nano Deep-Validation: progress / blockers

**Branch**: `week9/nano-deep`
**Status as of 2026-04-19T17:15Z**: Nano SSH (`nano2 → 169.235.25.145:8010`) is
**unreachable** (`Operation timed out` on three independent attempts at
25 s / 40 s / 60 s budgets). `orin1` is also unreachable. The same
condition was hit in W8 PR #24, so this is the second iteration of the
"embedded board offline" failure mode the spec calls out under Hard Rules.

## Blocker: physical Nano runs deferred

Tasks A and B require live Nano hardware (sudo userspace governor + real
DVFS sysfs writes + tegrastats). With SSH down they cannot be executed
end-to-end this iteration. The remaining work — code, unit tests, YAML
configs, Mac-substitute smoke runs, design doc — is **not** blocked by
the SSH outage and is being completed in this PR.

When `nano2` is reachable again, the operator can re-run the three Nano
commands documented in `docs/week9_nano_deep_design.md` to fill in the
hardware columns of the result table; the harnesses themselves are
shipped and tested in this PR.

## What this PR contains

- **Task A** (`scripts/week9_nano_dag_sweep.py`): 4-D DAG env sweep
  driver wrapping the eval runner across 3 preference vectors. Mac
  substitute run produces `runs/w9_nano_dag_mac/` artefacts.
- **Task B** (`scripts/week9_overhead_rebaseline.py`): re-baselined
  overhead profiler with the fair `preference_ppo + DAGSchedulerEnv`
  baseline (replaces `random + CartPole`). Mac substitute run produces
  `runs/w9_overhead_rebaseline_mac/` artefacts.
- **Task C** (`tetrarl/morl/baselines/dvfs_drl_multitask.py` +
  `tetrarl/eval/configs/dvfs_drl_multitask_nano.yaml`): soft-deadline
  reward-shaping baseline (Algorithm 3 from "DVFS-DRL-Multitask" 2024)
  wired into the runner registry as
  `agent_type="dvfs_drl_multitask"`.
- **Task D** (`docs/week9_nano_deep_design.md`): setup, methodology,
  W8 overhead deviation analysis, and the Nano commands to run when
  hardware comes back.
- Tests: `tests/test_dvfs_drl_multitask.py`, `tests/test_week9_*.py`,
  Mac-substitute end-to-end smokes.

## Mac substitute caveats

Same caveats as PR #24:
- `wall_time_s` and `framework_step_ms` are Mac numbers, not Nano.
- `framework_overhead_pct` Mac value is documented in `result.md` as a
  *sanity check that the harness wires up correctly*, not as the W8
  re-baseline answer.
- All synthetic-telemetry metrics (HV, mean_reward, override fire
  count, oom events, mean_memory_util, mean_energy_j) are bit-identical
  to the Nano path because they are computed off the synthetic 4-D
  reward vector, which is platform-independent.
