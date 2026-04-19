# Week 7: PPO-Lagrangian for TetraRL — design notes

## What this delivers

A constrained-RL training driver (`tetrarl/morl/native/lagrangian.py`) that
shapes the per-step reward by Lagrangian multipliers updated with the
Spoor (2025) PI dual rule. Mirrors the structure of
`tetrarl/morl/native/preference_ppo.py` but drops preference conditioning
(PPO-Lagrangian doesn't condition on `omega`).

Files:

- `tetrarl/morl/native/lagrangian.py` — `LagrangianConfig`, `LagrangianDual`,
  `shaped_reward`, `train_ppo_lagrangian`.
- `scripts/week7_ppo_lagrangian_train.py` — CLI driver with synthetic
  telemetry and matplotlib convergence plots.
- `tests/test_lagrangian.py` — dual-update unit tests + CartPole
  smoke + override wire-up test.

## Spoor 2025 PI dual update

For each constraint `i`:

```
error_i        = max(0, measured_i - target_i)             # one-sided over-shoot
integral_i    += error_i        # clamped to +/- integral_max (anti-windup I)
delta_i        = K_p * error_i + K_i * integral_i
lambda_i       = clip(lambda_i + delta_i,                  # anti-windup II
                     lambda_min, lambda_max)
```

This is the classical Lagrangian dual ascent with a PI controller in
place of the pure I term. The proportional term gives fast response to
sudden violations; the integral term ensures the multiplier converges to
the value that drives the constraint to its target asymptotically.

Per Spoor 2025, the dual variable is updated at a SLOWER cadence than
the policy: in our implementation the dual update fires once per rollout
(end of each PPO iteration), using the rollout-mean violation as the
error signal. This matches the intuition that PPO is already doing many
gradient steps per env-step batch and the dual should react to the
average behaviour of the rollout, not to per-step noise.

## Anti-windup design

We apply BOTH defences against integral windup:

1. **Integral clamp** (`integral_max`): bounds the I-state itself so a
   long stretch of large violations cannot create an arbitrarily large
   integral that would later prevent lambda from coming back down once
   the system recovers.
2. **Lambda clamp** (`lambda_max`): bounds the multiplier directly. Even
   if the integral and proportional terms produce a huge delta, the
   final multiplier is clipped, preventing the shaped reward from being
   dominated by a single constraint penalty term.

Belt-and-suspenders: integral clamp protects controller MEMORY, lambda
clamp protects current ACTION. Either alone is insufficient; together
they keep the closed-loop stable across the variety of environments
we'll target on the Orin/Nano.

## One-sided multipliers (lambda >= 0)

`lambda_min` defaults to `0.0`. We enforce this because a NEGATIVE
multiplier would *reward* violation rather than penalize it, which is
nonsensical for safety constraints. The Spoor rule naturally drives
lambda back toward zero when the system is well below target (negative
violation), but the clamp guarantees we never cross.

## Override-layer integration

The override layer is **decoupled from the policy gradient** — the same
pattern as `train_preference_ppo` in `preference_ppo.py`:

* Policy proposes an action `a_pi` and we record `(a_pi, logprob, value)`
  in the rollout buffer.
* If `with_override` is True and telemetry triggers the override, we
  swap in `fallback_action` for `env.step(...)`.
* The reward we observe IS the reward from running `fallback_action`,
  but PPO trains its policy on the actor's `a_pi` choice.

This keeps learning unbiased by the safety mechanism while letting us
audit `override_fire_count` to understand how often the safety net was
needed.

## Knob mapper

The `knob_mapper` argument is one of `{n_steps, n_epochs,
mini_batch_size}` and is currently RECORDED in the result dict but does
not yet adjust the corresponding PPO knob. The closed-loop coupling
between lambda values and PPO knobs is deferred to a later week's
deliverable; this wrapper establishes the contract so that downstream
code can read `result["knob_mapper"]` and decide which dimension to
modulate.

The eventual design intent (out of scope here): when `lambda_T` (latency
multiplier) grows above a threshold, the knob mapper would shorten
`num_steps` so the policy reacts to safety pressure faster, even at the
cost of a bit more variance in the value-function update.

## Logging

Per-step JSONL records contain:

```
{
  step, iteration,
  latency_ms, energy_j, memory_util,
  reward_raw, reward_shaped,
  lambda_T, lambda_E, lambda_M,
  override_fired,
  violation_T, violation_E, violation_M
}
```

Buffered per-iteration and flushed at the end of each rollout to keep
disk I/O off the critical path. Numpy scalars / arrays are coerced to
plain Python types at append time so JSON serialization can never fail
on a stray `np.float32`.

## TODO

- [ ] Convergence plot from a real Orin run (currently filled by the
      Mac smoke test).
- [ ] Per-constraint violation table populated from
      `mean_violations` once we have meaningful real telemetry.
- [ ] Closed-loop coupling between `lambda_T` and `num_steps` (deferred
      from this week).
