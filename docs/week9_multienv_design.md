# Week 9 — Multi-env Scaling + Real-Orin Ablation Re-run + Cross-platform CDF Expansion: Design Doc

## 1. Motivation

Three threads converge in Week 9, each a piece of unfinished business
from a prior week. (a) The W7 risk note flagged a multi-env scaling
test as deferred: on a single physical Jetson, running PPO with
`n_envs > 1` interacts with DVFS in a way the W6/W7 single-env eval
loop could not exercise, because there was only ever one env to
throttle. (b) W8 PR #24 ran the per-component ablation matrix that
Table 4 of the paper depends on against the Mac substitute path
rather than against real Orin AGX hardware. The deviation was honest
(Orin was unavailable) but the wall-clock and tail-latency numbers in
that table are not real Orin numbers. (c) The W7 cross-platform CDF
figure shows one curve per (device, condition); the paper benefits
from a per-preference-vector breakdown so a reader can see how the
tail moves as the arbiter is steered toward different objectives
under each interference condition.

The W9 PR lands the code and design for all three threads. The
empirical re-runs that need physical Orin access are deferred — orin1
SSH was unreachable on 2026-04-19 — but the YAML configs, runner
branches, sweep script, and figure generator are all in place so the
eventual hardware runs are one-liners. This document is the rationale
that ties the three threads together and the place a downstream reader
should look to understand which numbers in the eventual paper are
bit-exact and which are platform-dependent.

## 2. Multi-env x DVFS interaction

The multi-env scaling test extends `tetrarl/eval/runner.py` with an
`n_envs` field on `EvalConfig` and a `gym.vector.SyncVectorEnv` branch
inside `EvalRunner.run` that activates whenever `cfg.n_envs > 1`. The
W7 single-env path is preserved byte-for-byte at `n_envs=1`; the new
branch is purely additive. The sweep matrix that exercises this branch
is the 6-config grid driven by `scripts/week9_multienv_sweep.py`:

```
n_envs    in {1, 2, 4}
dvfs_mode in {fixed_max, userspace_with_arbiter}
= 6 configs total
```

Each config uses `agent_type=preference_ppo` and `ablation=none` so
the matrix isolates the n_envs / DVFS interaction; nothing else is
varying.

### 2.1 What goes wrong when n_envs grows under DVFS

PPO collects rollouts in lockstep across vector envs: `vec_env.step`
submits all `n_envs` actions at once and the call returns only when
the slowest env has produced its observation. On a `SyncVectorEnv`
this stepping is sequential inside the single-process driver, which
means the per-rollout-step wall time is `sum(env_i.step)` rather than
`max(env_i.step)`. Either way, the per-batch latency floor is bounded
from below by the slowest env in the batch.

The DVFS interaction enters when one of the envs runs on a CPU core
that the resource manager has clocked down. Under
`dvfs_mode=userspace_with_arbiter` the runtime governor is
`userspace`, the arbiter is allowed to write a frequency target via
the W7 ResourceManager path, and the arbiter's policy depends on
per-step telemetry (latency, energy, memory). A heterogeneous batch
where one env happens to land on a throttled core, and the other
envs land on faster cores, will stall the entire batch on that one
slow env, even though the other `n_envs - 1` envs were ready earlier.
This is the straggler-tail phenomenon documented in the W7 design's
section 9.7 pitfall 3, applied to the env-batch axis instead of the
component-pipeline axis it was originally written for.

The contrast condition `dvfs_mode=fixed_max` pins every CPU at its
maximum frequency for the duration of the run, eliminating the
throttling source. Any tail-latency degradation observed under
`fixed_max` is therefore attributable to the cost of stepping more
envs (linear in `n_envs` for a SyncVectorEnv) rather than to
DVFS-induced straggler effects. The difference between the two
columns of the sweep matrix at fixed `n_envs` is exactly the
DVFS-induced contribution.

### 2.2 Why two DVFS modes (and not one)

A naive sweep would just hold DVFS at the W7 production setting
(`userspace_with_arbiter`) and report the n_envs scaling curve. That
cannot distinguish "envs are slow because there are more of them"
from "envs are slow because one is throttled while the others wait."
The two-mode contrast attributes the observed tail movement:
`fixed_max` is the no-throttling baseline, `userspace_with_arbiter`
adds the throttling behaviour, and the per-row delta is the
throttling-induced straggler tail. This is the same single-knob
attribution discipline the W8 ablation matrix uses; the knob here is
DVFS mode rather than a TetraRL component identity.

`dvfs_mode` lives in `cfg.extra` rather than as a first-class
`EvalConfig` field because it only matters on physical Jetson where
the sysfs DVFS controller is wired up; on Mac it is inert. Keeping
it in `extra` preserves the platform-agnostic public schema.

### 2.3 Validation knob: tail_p99_ms ratio < 3x

The W9 spec specifies a quantitative validation criterion: the
`tail_p99_ms` measured under `userspace_with_arbiter` at any `n_envs`
must be less than 3x the `tail_p99_ms` measured at `n_envs=1` under
the same DVFS mode. The sweep driver computes this as
`tail_p99_ratio_vs_nenvs1_same_dvfs` per row of
`multienv_summary.md`. The denominator is the same-DVFS-mode n=1 row
(not the cross-mode one) so the ratio isolates n_envs-induced
degradation from dvfs-induced degradation.

If the ratio exceeds 3x at any cell we treat it as a failure of the
multi-env scaling claim. The follow-up plan is one of three: (a) drop
the failing n_envs row and report scaling only up to the largest
passing `n_envs`, (b) switch to AsyncVectorEnv if the failure is a
SyncVectorEnv-induced serialisation cost, or (c) pin DVFS frequencies
per env via core affinity (Section 3 explains why we are not doing
this in W9). Option (a) is the default.

## 3. Why straggler tails arise (section 9.7 pitfall 3)

The W7 design doc's section 9.7 pitfall 3 articulates the general
form of the straggler-tail phenomenon: when a system synchronises on
the slowest of `k` parallel components, the observed tail latency is
the tail of the maximum of `k` IID-like distributions, which grows
roughly with `log k` even when each individual distribution is
well-behaved. The framework-decision pipeline in W7 hit this when the
ResourceManager and the RL arbiter were synchronised at every step
boundary (the W7 fix moved the manager into a background thread and
allowed it to lag by exactly one step).

The W9 instance of pitfall 3 is structurally identical but lives one
abstraction layer up: the synchronising barrier is `vec_env.step`
rather than `framework.step`, and the parallel components are env
instances rather than pipeline stages. SyncVectorEnv synchronises
step boundaries by construction — the call returns only after every
underlying env has produced its next observation — so any per-env
latency variance feeds directly into the per-batch tail. When DVFS is
active and one env's underlying CPU core has been clocked down, that
env's per-step latency grows, and because the batch is bounded by the
slowest env, the per-batch tail grows with it.

Two off-the-shelf mitigations are deliberately not applied in W9.
First, `gym.vector.AsyncVectorEnv` would let each env step in its own
subprocess and decouple them from lockstep synchronisation, but it
would do so by hiding the throttling effect rather than measuring
it — which defeats the purpose of the W9 sweep. Second, per-env DVFS
pinning via `taskset`/`sched_setaffinity` plus a per-core frequency
lock would route each vector env to a known-fast core, but requires
core-affinity plumbing outside W9's scope and conflicts with the W7
design's freedom-to-throttle-any-core contract. Both are documented
as follow-ups in Section 8.

## 4. Real-Orin ablation re-run vs W8 Mac-substitute

The W8 ablation matrix in PR #24 was run with `platform=mac_stub`
because orin1 was unreachable when the W8 deliverable needed to land.
The runner is platform-agnostic so the same `tetrarl/eval/configs/
ablation_full.yaml` will drive a real-Orin run unchanged once the
hardware is back; what differs between the two runs is the telemetry
source and the resulting fidelity of two specific output columns.

### 4.1 What the W8 deviation actually changed

Five Mac-substitute metrics are bit-exact equivalents of what the
same code path produces on Orin under the same seed: `mean_reward`,
`override_fire_count`, `mean_memory_util`, `mean_energy_j`, and
`oom_events`. Each is computed from deterministic synthetic streams
(`memory_util = 0.1 + 0.001 * step`, `energy_j = 1e-3 * (action + 1)`)
plus the seeded arbiter RNG, with no hardware coupling. These columns
will not move on a real-Orin re-run.

`tail_p99_ms` and `wall_time_s` do not transfer. On Mac they reflect
host CPU jitter and the OS scheduler, with no real DVFS in the loop.
On Orin under `governor=userspace` they reflect the frequency island
chosen by the resource manager (or the pinned-max for the
`resource_manager` ablation arm), tegrastats sampling jitter on the
order of 100 ms, and the per-step cost on the Cortex-A78AE cluster.
W8 Table 4 caveats this in Section 5.3, but a casual reader would
not know which columns to discount; the real-Orin re-run produces
the authoritative numbers.

### 4.2 Re-run plan and execution

The re-run uses the same `tetrarl/eval/configs/ablation_full.yaml`
that W8 used (with the W8 fix to `agent_type=preference_ppo` for
every row, see W8 design Section 5.2) and the one-liner from the W9
spec:

```bash
/Users/zexinli/login.sh orin1 \
  'echo zexin | sudo -S -v && \
   for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; \
     do echo userspace | sudo tee $cpu > /dev/null; done && \
   cd /experiment/zexin/TetraRL && git pull && \
   source /experiment/zexin/venvs/r3/bin/activate && \
   python -m tetrarl.eval.runner \
     --config tetrarl/eval/configs/ablation_full.yaml \
     --out-dir runs/w9_ablation_orin_real/'
```

The same runner code path that produced the W8 Mac numbers is what
runs here; only the platform-specific telemetry / DVFS factory paths
differ. The W9 PR has already fixed the silent-stub fallback bug
described in Section 5 below, so this re-run cannot accidentally
repeat the W8 deviation — the harness will WARN if it ever sees an
orin platform but the stub telemetry path is taken.

### 4.3 Side-by-side coexistence in the paper

The W8 Mac-substitute Table 4 is retained as a reproducibility /
portability sanity check: the five bit-exact-equivalent metrics let a
downstream reader verify that the exact same runner code, run on a
different host, produces the same functional outcomes. The real-Orin
re-run produces a Table 4' (read: "prime") with identical seeds
(0, 1, 2), the same five ablation arms, and the same metric columns.
The headline ablation discussion references Table 4' for
`tail_p99_ms` and `wall_time_s`; Table 4 stays as a methodological
artefact in the appendix.

The expected wall-time delta is meaningful: Mac stub has no DVFS so it
is faster than real Orin under `governor=userspace` (the userspace
governor allows the resource manager to clock the core down, which
directly increases per-step wall time on throttled-down rows). The
`tail_p99_ms` delta is in the same direction: the real Orin tail is
bounded from below by tegrastats sampling jitter and
cluster-scheduling non-determinism the Mac jitter does not capture.
Both deltas should appear in the same direction across all five arms,
which gives a reader a sanity check on whether the paired tables are
internally consistent.

### 4.4 Today's deferral

orin1 was unreachable on 2026-04-19 (SSH timeout to
169.235.25.145:8002, retried twice, see `progress.md` in the W9
handoff). Per the steer-update in `spec.md`, this PR ships the
code-only changes (the W8 bug fix, the multi-env runner branch, the
sweep script, the figure generator, the design doc) and defers the
empirical re-run until SSH is restored. The follow-up commit that
lands the actual numbers is a one-liner against the same
`ablation_full.yaml` and is a hardware-side concern only.

## 5. Bug fix: `_make_telemetry()` WARN

The W8 hidden bug is in `tetrarl/eval/runner.py` at the top of
`_make_telemetry(platform)`: the function dispatched on
`platform == "mac_stub"` only and silently fell through to the Mac
stub for any other platform string, including `orin_agx` and `nano`.
The W8 ablation_full.yaml correctly set `platform: orin_agx` for
every config, but because the runner's telemetry factory ignored
that field whenever the real tegrastats daemon was not wired in
(which it was not, because the daemon lives in the platform-specific
scripts, not in the runner module), the actual telemetry source was
the Mac stub for every W8 row. The W8 numbers were synthetic even
when the YAML said `orin_agx`. This is the deviation that
`docs/week8_ablation_design.md` Section 5.3 describes as "ran on Mac
because Orin was unreachable", but the runner was not actively
refusing to run on Orin; it was silently accepting the orin
configuration and returning Mac-stub telemetry.

The W9 fix is minimal and additive. The Mac-stub fallback is
preserved (the function signature is unchanged, the return value is
unchanged, no test breaks), but a `RuntimeWarning` is now emitted
whenever the function is called with a platform string that starts
with `orin_`:

```python
if platform.startswith("orin_"):
    warnings.warn(
        f"_make_telemetry() called with platform={platform!r} but the "
        "real tegrastats daemon is not wired up here; falling back to "
        "the Mac stub. Real-Orin runs should use the platform-specific "
        "scripts that build a TegrastatsDaemon directly.",
        RuntimeWarning,
        stacklevel=2,
    )
```

The warning is loud enough to surface in the test runner output and
in any downstream sweep log; downstream callers that build a
`TegrastatsDaemon` directly (the platform-specific `scripts/` files
on Jetson) never go through `_make_telemetry()` at all and are
unaffected. A unit test asserts both the warning emission for
`platform=orin_agx` and the silence for `platform=mac_stub`.

The fix is intentionally not the larger one (wire the real
TegrastatsDaemon path into the runner). That larger fix is documented
as a follow-up in Section 8: it would change the runner's import
graph and pull a hardware dependency into the unit-test path that
currently runs on Mac without sudo. The WARN-only fix is the smallest
change that prevents a recurrence of the W8 silent-fallback deviation
without expanding the runner's surface area.

## 6. Cross-platform tail-latency CDF expansion

The W7 figure (`runs/w7_combined_cdf.png`) renders one CDF curve per
(device, condition) pair, where condition is the FFmpeg co-runner
configuration (`none`, `720p`, `1080p`, `2K`). The W7 figure is
informative about how the runtime tail responds to interference, but
it averages over the preference vector — every curve is implicitly
the omega the resource manager picks under that condition, which is
the production omega `[0.7, 0.3]`. The W9 expansion adds the
preference-vector axis: each panel renders up to `n_omega *
n_condition` curves so a reader can see how the tail moves as the
arbiter is steered toward different objectives under each
interference condition.

### 6.1 Input layout and per-omega selection

The expanded figure reads from a per-omega tree:

```
runs/w9_ffmpeg_<device>_per_omega/
  energy_corner/
    none.jsonl
    720p.jsonl
    1080p.jsonl
    2K.jsonl
  memory_corner/{...}
  center/{...}
```

The recorder schema (`{"sample_ms": float, "idx": int}`) is reused
unchanged from W7 so the same loader (`scripts.week7_make_cdf.
_load_latencies`) drives both figures; the W9 script
(`scripts/week9_make_expanded_cdf.py`) only adds a per-omega outer
loop on top of the W7 plotting routine.

The three omega vectors bracket the preference simplex.
`energy_corner` is `[1.0, 0.0]`, `memory_corner` is `[0.0, 1.0]`, and
`center` is `[0.5, 0.5]`. The corners are the extremal points of the
2-objective convex hull; the centre is uniform. Three vectors x four
conditions x two devices = 24 curves total (12 per panel), kept
legible by the colour-by-omega + linestyle-by-condition encoding (see
`_OMEGA_COLORS` and `_CONDITION_LINESTYLES`).

### 6.2 Graceful skipping and parallel Nano PR

The figure script is graceful about missing data: a missing
`<omega>/` directory warns on stderr and skips the entire omega on
that panel; a missing single `<omega>/<condition>.jsonl` warns and
skips that one curve; an empty JSONL warns and skips the curve. The
script only hard-fails (exit 1) if `--orin-dir` or `--nano-dir`
itself is missing or if zero curves were plotted on either panel.
This is a deliberate design choice for the W9 PR: the per-omega data
collection on Orin is deferred (Section 4.4 above) and the per-omega
data collection on Nano is the responsibility of the parallel
`week9-nano-deep` PR. Either side can land its data tree before the
other and the figure script will produce a partial figure rather
than crashing. Once both trees are populated, re-running the script
produces the complete 12-curve-per-panel figure with no code
changes.

The dovetail with `week9-nano-deep` is the explicit reason the panel
layout is left/right (Orin/Nano) rather than top/bottom or
single-panel. The two PRs land independently against the same
figure-generator entry point; the figure stitches together as soon
as both data trees are present.

## 7. What lands in this PR vs deferred

This PR lands all of the code, scripts, and tests; it defers all of
the empirical Orin runs.

### 7.1 Lands now

- `tetrarl/eval/runner.py`: the `n_envs` field on `EvalConfig`, the
  `gym.vector.SyncVectorEnv` branch in `EvalRunner.run` (via
  `_run_vec_env`), and the `_make_telemetry()` `RuntimeWarning` on
  `platform.startswith("orin_")`.
- `scripts/week9_multienv_sweep.py`: the 6-config sweep driver
  (`build_sweep_configs` for the matrix, `_build_summary_rows` and
  `_format_summary_table` for the per-row Markdown output, plus a
  `--dry-run` mode for offline validation).
- `scripts/week9_make_expanded_cdf.py`: the per-omega figure
  generator with the graceful-skip behaviour described in Section
  6.2.
- `docs/week9_multienv_design.md`: this file.
- 35+ new unit tests covering the multi-env runner branch, the
  warning emission, the sweep driver row construction, the
  per-omega figure generator's plotting and skip behaviour. All
  pre-existing tests (340+ baseline) continue to pass.

### 7.2 Deferred (orin1 SSH unreachable on 2026-04-19)

- Real-Orin ablation re-run (Task A): the `python -m tetrarl.eval.
  runner --config tetrarl/eval/configs/ablation_full.yaml --out-dir
  runs/w9_ablation_orin_real/` one-liner from `spec.md`.
- 6-config sweep on real Orin (Task B): the `python scripts/
  week9_multienv_sweep.py --out-dir runs/w9_multienv_orin/`
  one-liner from `spec.md`.
- Per-omega FFmpeg co-runner data collection on Orin (Task C): the
  three repeat-per-omega invocations of `scripts/
  week7_ffmpeg_corunner.py` documented in `spec.md`.
- The eventual paper Table 4', the multi-env scaling table, and the
  populated 12-curve expanded CDF figure.

The deferred items are entirely hardware-dependent. The code path
that consumes their output (the runner, the sweep driver, the figure
generator) is already merged and tested on Mac, so the follow-up
commit that lands the data is purely an `out_dir` of JSONLs plus a
re-run of the figure generator.

## 8. Limitations

Several items are explicit non-goals of W9.

Single-DVFS-mode-per-env. The sweep matrix toggles DVFS mode at the
run level, not per env. A more faithful multi-env scaling test would
pin each vector env to a different DVFS mode (or a different CPU
core) so the straggler-tail mechanism could be probed directly rather
than inferred. That requires per-env affinity plumbing which is out
of scope; see the AsyncVectorEnv / per-env DVFS pinning discussion in
Section 3.

Linear-time sweep. The driver executes the 6 configs sequentially in
a single process. There is no parallelism across configs. This is
intentional — running them in parallel on a single Jetson would
introduce its own DVFS-interaction effects across the parallel runs
and contaminate the per-config measurement. A future driver running
the matrix on a cluster of Jetsons could parallelise across the n=3
seed dimension safely.

Exploratory expanded CDF. The figure shows the tail moving with
omega but does not yet enforce a tail-percentile budget or compare
against a published latency target from the embedded-systems
literature. A future revision should pick a target (e.g. p99 < 50 ms
for control-loop applications) and overlay it as a horizontal line.
The figure script's `_plot_one_panel_per_omega` structure makes this
a one-line addition.

WARN-only `_make_telemetry()` fix. The function still returns
Mac-stub telemetry for orin platforms; it just prints a warning. The
right long-term fix is to wire a real `TegrastatsDaemon` path into
the runner (or to refuse to construct the framework when the platform
field implies a hardware source that the runner cannot provide). That
fix is out of scope because it would pull a hardware dependency into
the unit-test path. The WARN-only fix prevents the W8 silent-fallback
bug from recurring without expanding the runner's surface area.

Mac-sub Table 4 as portability check only. The real-Orin Table 4'
(per Section 4.3) is the authoritative ablation table; the Mac-sub
Table 4 is supporting material. A reviewer who reads only Table 4
and not Table 4' would over-read the `tail_p99_ms` and `wall_time_s`
columns; the paper's caption for Table 4 should be explicit about
which columns are bit-exact and which are platform-dependent. The
documentation discipline is inherited from
`docs/week8_ablation_design.md` Section 5.3 and extended here for
the paired-table case W9 introduces.

---

Source of truth: `/Users/zexinli/.openclaw/workspace/AGENTS_HANDOFF/active/week9-multienv-orin/spec.md`
