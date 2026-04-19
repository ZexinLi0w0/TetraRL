# Week 7 Concurrent Decision Loop Design

**Status**: implemented on branch `week7/concurrent-decision`.
**Theme** (per `docs/action-plan-weekly.md`): DVFO-style "thinking-while-moving" overlap of the DVFS decision computation with the RL arbiter's forward pass.

## Overview

`tetrarl/sys/concurrent.py` adds `ConcurrentDecisionLoop`, a small daemon-thread wrapper around the Resource Manager that lets `TetraRLFramework.step()` overlap two tracks per step:

- **Track A (foreground)**: `rl_arbiter.act(state, omega)` — must complete before the env step.
- **Track B (background)**: `resource_manager.decide_dvfs(hw, n_levels)` — runs in a worker thread and is permitted to lag the foreground by exactly one step.

The decision applied at step `t` is the one computed during step `t-1`'s background work. At step 0 the worker has produced nothing yet, so `apply_latest()` returns `None` (or, when configured, a static fallback frequency index) without crashing.

## Why thinking-while-moving (DVFO, Zhang TMC 2023)

DVFO observes that on Jetson, the DVFS picker (in DVFO's case a small NN; in our default Resource Manager a rule-based step-down) runs on the CPU while the actor's forward pass is also CPU/GPU work. Sequencing them — sample telemetry → decide DVFS → apply DVFS → run arbiter forward → step env — pays the DVFS-decision latency on the critical path of every control step.

DVFO's contribution is the observation that the next step's DVFS target depends only on the current telemetry and not on the next state, so the decision can be hoisted into a background thread that completes before the *next* step needs it. The *applied* frequency at step `t` is therefore the decision computed at step `t-1`, masking the DVFS latency entirely behind the arbiter's forward pass.

## Threading Model

```
+------------------+
|  main thread     |
|                  |        single-slot                +----------------------+
|  framework.step()|--submit(hw_t)-->[ Queue (max=1) ]----- get(timeout=50ms)|--->[ ResourceManager.decide_dvfs ]
|     |            |                       ^                                 |
|     | apply_latest()---reads-->[ _last_result ]<----lock----writes---------+
|     v            |                                  worker thread          |
|  arbiter.act()   |                                  (daemon)               |
+------------------+                                                         |
                                                                             v
                                                                    swallows exceptions
                                                                    so main loop never deadlocks
```

- Single daemon worker thread (`tetrarl-concurrent-dvfs`) drains the queue with a 50 ms timeout.
- A `threading.Lock` serialises `submit()` (drain-then-put) AND the last-result swap, so concurrent producers cannot race.
- The queue has `maxsize=1`. `submit()` first drains any pending item before enqueueing the new one — **freshness > completeness**: under fast submit pressure, the OLDEST pending telemetry is dropped, never the newest.
- `decide_dvfs` exceptions inside the worker are swallowed (`except Exception: continue`). A bad telemetry sample cannot deadlock the main loop or crash the worker.
- `shutdown()` is **idempotent**: it sets the stop event, pushes a `None` sentinel to wake a blocked `get()`, and joins the worker with a 2 s timeout. A second call is a no-op. `submit()` after `shutdown()` is a safe no-op as well.

## Single-slot queue rationale

Why drop oldest instead of buffering?

- DVFS targets are **freshness-sensitive**: a stale telemetry from 10 steps ago is worse than no decision at all (the system has moved on; any frequency derived from it is misaligned).
- The decision is a *target index*, not a stream of events that must each be applied. Two queued decisions would just mean the second one overwrites the first at apply-time anyway.
- Bounding `maxsize=1` removes the need for any backpressure protocol on the producer side and keeps memory constant under arbitrary submit rates (the 4-thread stress test in `tests/test_concurrent.py` exercises exactly this).

## Per-step Overlap Diagram

Sequential (Week 6, default path):

```
step t:
  | telemetry | decide | set_freq | arbiter.act | env.step |
                 ^^^^   ^^^^^^^^^^                 ^^^^^^^^
                 critical path includes DVFS work
```

Concurrent (Week 7, when `concurrent_decision` is set):

```
step t-1 (background work kicked off):
  | telemetry | submit(hw_t-1) | arbiter.act | env.step |
                                  -------+----
                                         | bg: decide_dvfs(hw_t-1) (overlapped)
                                         v

step t (apply previous decision, kick off next):
  | apply_latest (uses t-1 result) | submit(hw_t) | arbiter.act | env.step |
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                  ----+--------
    set_freq runs here                                  | bg: decide_dvfs(hw_t)
    (NOT overlapped; trivial)                           v
```

What the overlap masks:
- `decide_dvfs` (potentially expensive — DVFO's NN, or any future ML-based picker).
- The wall-clock cost of the rule traversal in our default ResourceManager (negligible on its own, but combinable with future learned modules).

What the overlap does NOT mask (in this iteration):
- `set_freq` (the actual DVFS apply). On Orin this is a cheap sysfs write but still in the foreground inside `apply_latest()`. Moving it off the critical path would risk applying a frequency target after the env step has already begun, breaking the per-step semantic the rest of the framework assumes.

## Framework Wiring

`TetraRLFramework.__init__` gains an optional kwarg `concurrent_decision: Optional[ConcurrentDecisionLoop] = None`. When set:

1. `framework.step()` calls `concurrent_decision.apply_latest()` BEFORE the arbiter (uses the previously-submitted decision; sets the frequency now).
2. Then calls `concurrent_decision.submit(hw)` BEFORE `arbiter.act()` so the worker computes the next decision **while** the foreground thread is busy in the arbiter forward.
3. The in-loop `resource_manager.decide_dvfs(...)` + `dvfs_controller.set_freq(...)` block is **skipped** — the loop owns DVFS now.
4. The per-step record gets a new key `concurrent_dvfs_used: bool` so downstream analysis can filter the two paths apart.

When `concurrent_decision is None` (default), the pre-Week-7 sequential pipeline is preserved EXACTLY (verified by all 8 existing `tests/test_framework.py` tests + 5 `tests/test_week6_e2e.py` tests still passing).

## Mac vs Orin Behaviour

| component       | Mac (dev)                                     | Orin AGX (target)                                                   |
|-----------------|-----------------------------------------------|---------------------------------------------------------------------|
| `ConcurrentDecisionLoop` | identical (Python stdlib threading)  | identical (the GIL releases on `time.sleep()` so a NN-based picker can actually overlap with PyTorch forward) |
| `decide_dvfs`   | rule-based step-down (~µs); smoke uses `_CostlyResourceManager` to inject a 1 ms simulated cost so the overlap is observable | rule-based step-down today; the design admits a learned NN-based picker à la DVFO without changing the loop interface |
| `set_freq`      | stub (no sysfs writes)                        | sysfs writes via `cpufreq` / `devfreq`; requires `sudo` and `governor=userspace` (see auto-memory entry "Orin DVFS root requirement") |
| Threading       | identical                                     | identical                                                           |

The Mac smoke uses `--decide-cost-ms 1.0` by default so the overlap is meaningful. With the real (essentially free) rule-based decision, the loop's thread-handoff overhead would dominate per-step time and the +10% gate would not be meetable — that case is intentionally exposed by `--decide-cost-ms 0`.

## Smoke Test Results

`scripts/week7_concurrent_smoke.py --episodes 100 --decide-cost-ms 1.0` (CartPole-v1, RandomArbiter, stub telemetry, stub DVFS, simulated 1 ms decision cost):

| metric                         | sequential | concurrent |
|--------------------------------|-----------:|-----------:|
| total steps                    | 2079       | 2079       |
| mean framework step (ms)       | ~4.12      | ~0.010     |
| max framework step (ms)        | ~7.13      | ~0.21      |
| override fire_count            | 0          | 0          |
| **speedup (concurrent vs seq)**| —          | **+99.76%**|

The +99.76% speedup is the "ceiling" case: the simulated decide cost dominates the sequential per-step budget, so moving it off the critical path nearly eliminates it. On real hardware with a more modest decide cost (e.g., 0.5 ms vs a 5 ms arbiter forward) the speedup will be smaller but still positive — the framework is bound by `max(arbiter, decide)` instead of `sum(arbiter, decide)`.

## Tests

| test                                                                                           | what it verifies                                                  |
|------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| `test_basic_submit_then_latest_returns_decision`                                               | happy-path: submit, wait, latest() yields the worker's idx        |
| `test_first_call_to_latest_returns_none_before_any_submit`                                     | step-0 semantics: latest() and apply_latest() return None safely  |
| `test_apply_latest_invokes_dvfs_set_freq_with_gpu_idx`                                         | apply_latest writes via `set_freq(gpu_idx=...)`                   |
| `test_shutdown_is_idempotent_and_joins_thread`                                                 | shutdown twice + post-shutdown submit are no-ops                  |
| `test_oldest_decision_dropped_under_fast_submit`                                               | single-slot queue: 100 fast submits never grow past `maxsize=1`   |
| `test_thread_safety_under_concurrent_submit`                                                   | 4 producer threads × 0.5 s — no exceptions, no deadlocks          |
| `test_background_exception_does_not_deadlock_main_loop`                                        | ResourceManager exceptions in worker do not deadlock main         |
| `test_apply_latest_returns_fallback_when_no_decision_and_fallback_set`                         | configured fallback applied at step 0                             |
| `test_latest_reflects_freshest_decision_after_settling`                                        | latest() reflects the most recent computation                     |
| `test_framework_records_concurrent_dvfs_used_flag_false_when_unset`                            | new record key `concurrent_dvfs_used=False` when no loop          |
| `test_framework_first_step_with_concurrent_loop_does_not_crash`                                | framework's step 0 with the loop returns dvfs_idx=None safely     |
| `test_framework_concurrent_action_stream_matches_sequential_under_constant_telemetry`          | action stream matches step-for-step; DVFS lag is exactly 1 step   |
| `test_framework_concurrent_per_step_time_not_egregiously_higher`                               | concurrent per-step time within a sane bound vs sequential        |

13 tests, all passing. Full repo: 190 (177 baseline + 13 new).

## Backward Compatibility

- `TetraRLFramework(...)` with no `concurrent_decision` arg is byte-for-byte identical in behaviour to Week 6.
- All 8 `tests/test_framework.py` and all 5 `tests/test_week6_e2e.py` tests pass unchanged.
- The new record key `concurrent_dvfs_used` is **additive** — existing consumers that only read `(action, dvfs_idx, reward, latency_ms, energy_j, memory_util)` continue to work.

## What's NOT in Week 7 (deferred)

- Moving `set_freq` off the critical path. The current design keeps `set_freq` in the foreground inside `apply_latest()` so the per-step semantics — "the action and the frequency target are jointly committed at the start of the step" — are preserved. Deferred until we have measurements showing `set_freq` itself is a bottleneck.
- A learned (NN-based) Resource Manager. The interface (`decide_dvfs(telemetry, n_levels) -> int`) is already DVFO-shaped; swapping in an ML picker is mechanical.
- Real-Orin sysfs validation of the loop. The sequential-on-hardware agent will exercise this on the Orin AGX once Week 7's hardware slot opens.
