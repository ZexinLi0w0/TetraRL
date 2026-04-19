# Week 8 — Per-component Ablation Study: Design + Methodology

## 1. Motivation

TetraRL is presented in the paper as a four-component framework: a
preference plane that exposes a user-facing scalarization vector, a
resource manager that converts hardware telemetry into DVFS decisions,
an RL arbiter that proposes actions conditioned on that preference
vector, and an override layer that vetoes the arbiter when telemetry
indicates an unsafe operating point. Each of those components is named
in the paper's contribution paragraph and each is justified individually
by prior work — the C-MORL preference-plane line of [Yang 2019] and
[Xu 2020], the on-device DVFS work of [Zhang 2021] and [Liu 2019], the
preference-conditioned policy work of [Yang 2019] and [Basaklar 2023],
and the runtime safety / Lagrangian work of [Achiam 2017] and
[Stooke 2020]. A reviewer who reads that contribution paragraph will
demand evidence that each of the four components actually does
something — that the system is not three components plus one
load-bearing decoration. The conventional answer in the systems-ML
literature is a per-component ablation table: substitute each component
in turn for a documented "null" variant, hold every other knob fixed,
and report whether the headline metrics move.

This document is the design rationale for the Table 4 candidate that
satisfies that demand. It explains what the five ablation arms are,
what the substitute for each one does, what metric movement we expect
to see (and what we expect not to see, which is just as important for
attribution), and how the resulting numbers should be interpreted under
the statistical methodology we adopted. It also documents the
deviations from the original W8 plan — most importantly that this
attempt was forced onto the Mac substitute path because the Orin lab
bastion was unreachable when the runs needed to land — so a downstream
reader can tell which numbers are bit-exact equivalents of the Orin
result and which need to be re-run on the physical platform.

The five arms are `none` (the unmodified framework, used as the
reference), `preference_plane`, `resource_manager`, `rl_arbiter`, and
`override_layer`. The substitute for each non-`none` arm is a
purpose-built null class that lives next to the real component in
`tetrarl/eval/runner.py`; the wiring path that selects the null variant
is the single `ablation` string field of `EvalConfig`. That
single-field design is deliberate: exactly one slot is replaced per
run, so any movement in the headline metrics can be attributed to the
swapped component without disentangling interaction effects between
multiple simultaneous swaps. If a future paper revision wants joint
ablations (e.g. preference_plane plus rl_arbiter both nulled in the
same run) it would extend the field to a list, but the analysis
methodology in Section 4 below is written for the single-component
case that the current sweep produces.

## 2. The 5 ablation arms

### 2.1 `none` (reference arm)

The `none` arm runs the unmodified `TetraRLFramework` with the real
`StaticPreferencePlane` (returning `[0.7, 0.3]`), the real
`ResourceManager`, a `_PreferencePPOArbiter` driven by the
omega-conditioned stochastic policy described in
`tetrarl/eval/runner.py`, and the real `OverrideLayer` thresholded at
`max_latency_ms=2.0`, `min_energy_j=0.5`, and `max_memory_util=0.13`.
This is the row every other arm is compared against. The expected
metric profile under CartPole-v1 with the synthetic memory ramp
`memory_util = 0.1 + 0.001 * step` is a non-trivial `mean_reward`
(CartPole rewards saturate at 200 per episode under the v1 cap, so a
moderately good policy lands somewhere in the tens), an
`override_fire_count` that increases as the per-episode step count
grows past 31 (the threshold-crossing step under the synthetic ramp),
and zero `oom_events` because the synthetic ramp never reaches 1.0
within a single CartPole episode. This row is also the one whose
`tail_p99_ms` and `wall_time_s` we treat as nominal; on the Mac
substitute these timing numbers reflect host CPU jitter and not the
Orin frequency island, so they are not bit-exact equivalents to the
physical run.

### 2.2 `preference_plane` arm

The substitute is `_NullPreferencePlane`, which returns the uniform
omega `[0.5, 0.5]` regardless of step. This contrasts with the real
`StaticPreferencePlane`, which returns the non-uniform `[0.7, 0.3]`
that biases the omega-conditioned arbiter toward action 0 with
probability 0.7. Because `_PreferencePPOArbiter.act` computes
`P(a=0) = omega[0] / sum(omega)` and then samples, the action
distribution under the null variant collapses from the 70/30 bias of
the real plane to a 50/50 split. On CartPole-v1, where action 0 and
action 1 push the cart in opposite directions, this shift in marginal
action probability shifts the trajectory the cart takes through state
space and therefore shifts the per-episode return.

The expected metric movement is a change in `mean_reward` relative to
the `none` arm — the direction depends on whether the cart's initial
balancing problem is better served by a 70/30 left-bias policy or a
uniform policy, but a movement of some magnitude is the testable
prediction. The metric we expect to NOT move under this arm is
`override_fire_count`. The override layer reads `memory_util` and
`latency_ms` directly from telemetry, with no dependence on omega
whatsoever, so the rate at which the override fires under the
synthetic memory ramp should be statistically indistinguishable from
the `none` arm. A row that moved on `mean_reward` but did not move on
`override_fire_count` is the cleanest possible attribution we can
produce on this substitute env: the preference plane affects the
arbiter's choice and nothing else.

### 2.3 `resource_manager` arm

The substitute is `_NullResourceManager`, whose `decide_dvfs` returns
the highest available DVFS index regardless of telemetry — the
hardware-equivalent of pinning the device at maximum frequency for the
entire run. The real `ResourceManager` instead adapts the index
downward when telemetry indicates the device should throttle, so on a
real platform this arm should move energy upward and latency downward.

The honest answer for the Mac substitute path used in this
measurement is that this arm has no observable effect on any of the
four headline metrics. The runner's `_build_framework` constructs the
framework with `dvfs_controller=None` (see `tetrarl/eval/runner.py`,
the `TetraRLFramework(...)` call passes `dvfs_controller=None`
explicitly). The framework still calls `resource_manager.decide_dvfs`
but the returned index is never written to a real DVFS sysfs node
because there is no controller to write through. Furthermore the
synthetic latency / energy / memory streams emitted by the eval loop
are open-loop with respect to the DVFS index: they are computed as
`memory_util = 0.1 + 0.001 * step` and `energy_j = 1e-3 * (action +
1)` per step, neither of which reads the DVFS choice back. The result
is that on `platform=mac_stub` the `resource_manager` arm produces
identical functional metrics to the `none` arm. This is documented
in `docs/week8_eval_runner_design.md` (Section 5, "mac_stub
limitations") and is called out again in Section 5 below as a known
deviation that needs the Jetson DVFS path to validate.

The expected metric movement on the Orin path (deferred until SSH is
restored) is upward `mean_energy_j` and downward `tail_p99_ms`,
because pinning the highest DVFS index disables the throttle-down
behaviour the real manager uses to trade latency for energy. The
metric we expect to NOT move on the Orin path is `mean_reward`, since
DVFS choices on CartPole do not affect the policy's view of the env
state — they only affect the wall-clock cost of stepping it.

### 2.4 `rl_arbiter` arm

The substitute is `_RandomArbiter`, which samples uniformly over the
discrete action space, ignoring both the state and the omega vector.
This is the strongest of the four ablations in the sense that it
removes the most behaviourally significant component on a control task
like CartPole. A uniform-random policy on CartPole-v1 reliably falls
over within ten or twenty steps because the cart's pendulum is
unstable and uncorrelated action sequences cannot stabilise it; the
trained or omega-biased arbiter, by contrast, produces a coherent
balancing policy that survives substantially longer.

The expected metric movement is a large downward shift in
`mean_reward` relative to the `none` arm. We anticipate this is the
arm most likely to clear the `p<0.001` (`***`) bar in the Welch test
described in Section 4, even at n=3, because the effect size is large
relative to the seed-to-seed variance of either the random policy or
the omega-biased one. The metrics we expect to NOT move significantly
are `mean_energy_j` and `mean_memory_util` under the synthetic stub,
because both are computed from the action and the step counter
without any policy-dependence beyond which discrete action was
sampled — and a uniform sampler and a 70/30-biased sampler produce
very similar marginal distributions over many steps. A movement in
`override_fire_count` is plausible but second-order: shorter episodes
under the random policy mean fewer steps past the memory-threshold
crossover at step 31, so the per-run absolute count of override fires
should drop, but the per-step rate should be unchanged.

### 2.5 `override_layer` arm

The substitute is `_NullOverrideLayer`, which returns
`(False, None)` from every `step` call and keeps `fire_count` at zero
for the entire run. The real `OverrideLayer` constructed by
`_make_override_layer` triggers when telemetry crosses any of three
thresholds: `max_latency_ms=2.0`, `min_energy_j=0.5`, or
`max_memory_util=0.13`. Under the synthetic memory ramp `0.1 + 0.001
* step`, the memory threshold is crossed at step 31 and remains
crossed for the rest of the episode, so the real override fires
every step from 31 onward (modulo the latency threshold being
crossed earlier on a particularly slow step). The latency threshold
is set tight enough that the real override may fire on the first few
steps of a run while the host scheduler warms up, but the dominant
firing source on the synthetic stub is the memory ramp.

The expected metric movement is precisely `override_fire_count → 0`
by construction (the null layer never fires) and a possible upward
shift in `mean_memory_util` and `oom_events`, the latter only if a
particular episode runs long enough for the `0.1 + 0.001 * step`
ramp to reach 1.0. CartPole episodes capped at 200 steps see the
ramp reach `0.1 + 0.001 * 199 = 0.299` at most, so within the
substitute env `oom_events` will remain at 0 for both the `none` and
the `override_layer` arms — this is a known limitation of the
synthetic ramp's slope and is documented as a follow-up in
Section 5.

The metric we expect to NOT move significantly under this arm is
`mean_reward`. The override layer in the runner is wired to swap a
`fallback_action=0` for the arbiter's chosen action when it fires, so
removing the override returns control to the arbiter for those
late-episode steps. On CartPole-v1 the difference between "always
push left for the last hundred steps" and "follow the arbiter for the
last hundred steps" is small relative to the episode return variance,
so we predict `mean_reward` will not clear the `p<0.05` bar in the
Welch test even though the override-firing behaviour clearly differs.
This is the row a reviewer should look at to confirm the override
layer is doing what we say it does — fire-count and memory-pressure
movement, NOT reward movement — rather than as evidence that the
override layer changes the policy.

## 3. Why each component is expected to matter (with paper citations)

The contribution paragraph of the paper roots each TetraRL component in
a strand of prior work, and the per-component ablation is the
mechanism by which we test whether our integration of those strands
preserves the property each strand was meant to provide. This section
maps each arm to the prior-work anchor that motivates it and explains
what theoretical claim the ablation is testing.

### 3.1 Preference plane

The preference plane in TetraRL exposes a user-facing scalarization
vector `omega` that the arbiter conditions on, in the spirit of the
preference-set learning line that begins with [Yang 2019]'s
generalized-MORL formulation and is extended by [Xu 2020]'s
prediction-guided MORL toward sample-efficient Pareto-front
discovery. The theoretical claim is that the policy
`pi(a | s, omega)` should produce visibly different action
distributions when `omega` changes, because if it does not, the user
preference is decorative and the multi-objective framing collapses to
single-objective. The `preference_plane` ablation tests precisely
this: by clamping omega to uniform we remove the only mechanism by
which user preference can influence action selection, so any
preserved or improved metric under the null variant is evidence that
the omega-conditioning is not actually doing work. This is the
standard ablation form for any preference-conditioned policy and
matches the protocol used by [Basaklar 2023] in their PD-MORL
evaluation table. The expected outcome on CartPole is a shift in
`mean_reward` in either direction relative to `none`, because the
70/30 omega bias and the uniform 50/50 baseline correspond to
different stationary distributions over the cart's left-right
choices; the size of the shift depends on which choice happens to
better stabilise the pendulum from the v1 reset state.

### 3.2 Resource manager

The resource manager converts hardware telemetry (memory, energy,
latency) into a DVFS index, which a real platform writes to a sysfs
node to scale CPU/GPU frequency. The justifying line of work is the
energy-aware DNN scheduling literature, in particular [Zhang 2021] on
NeuOS-style co-runner scheduling and [Liu 2019] on real-time DVFS for
embedded ML inference. The theoretical claim is that adapting DVFS in
response to the workload's instantaneous resource pressure produces
better energy-latency trade-offs than running the device pinned at
its maximum frequency. The `resource_manager` ablation tests this by
substituting the always-max-index policy and measuring whether
`mean_energy_j` rises and `tail_p99_ms` falls — the predicted
direction if the real manager was successfully shaping that trade-off.
As Section 2.3 documents, the Mac substitute path cannot test this
prediction because no DVFS controller is wired in and the synthetic
telemetry is open-loop with respect to the DVFS choice; the test
exists only on the Orin path that this PR could not exercise.

### 3.3 RL arbiter

The arbiter is the locus of the preference-conditioned policy. The
prior work it is rooted in is the same line as the preference plane
([Yang 2019], [Basaklar 2023]) but the ablation is testing a
different claim: that the trained policy is meaningfully better than
a uniform-random baseline on the headline reward metric. This is the
weakest possible bar — it's the difference between "we trained
something" and "we trained nothing" — but it is also the most likely
to clear the statistical significance threshold at small n. If the
`rl_arbiter` arm does not show a `***` significance against `none`,
that is a strong signal that either the substitute env is too easy
(the random baseline is competitive) or the trained policy did not
actually fit (a Week 9 issue). On CartPole-v1, where uniform-random
typically achieves episode lengths of 10-20 against the v1 cap of
200, the expected effect size is large.

### 3.4 Override layer

The override layer corresponds to the constrained-RL safety-shielding
line of work, most directly [Achiam 2017]'s constrained policy
optimization and [Stooke 2020]'s responsive-safety layer designs. The
theoretical claim is that a runtime safety shield can absorb
constraint violations that the policy gradient does not know about,
without biasing the policy gradient itself (the override is recorded
but the actor still learns on its own action choices, the same
decoupling pattern documented in `docs/week7_ppo_lagrangian_design.md`
under "Override-layer integration"). The `override_layer` ablation
tests that the shield is actually catching real violations rather than
firing on noise: under the null variant, `override_fire_count` drops to
zero by construction and any consequent rise in `mean_memory_util` or
`oom_events` is evidence that the shield was absorbing real pressure.
The cleanest reviewer-facing reading of this row is the one in
Section 6 below: look at fire-count and memory-pressure delta, not at
reward delta.

## 4. Statistical methodology

Each of the five arms is run for `n=3` independent seeds (0, 1, 2),
producing three independent realisations of `mean_reward`,
`override_fire_count`, `tail_p99_ms`, and `mean_energy_j` per arm.
The four non-`none` arms are then compared against the `none` arm
pairwise on each metric using Welch's two-sample t-test, which
extends the standard two-sample t-test to the case of unequal
variances. The three core decisions in this design — Welch over
Student, two-sided over one-sided, and t-test over rank-based
alternatives — are explained below.

### 4.1 Why Welch (and not Student's t)

Student's two-sample t-test assumes the two samples are drawn from
populations with equal variance. That assumption is unsafe at n=3,
because it cannot be tested with any power: an F-test for variance
equality at n1=n2=3 has wide enough confidence intervals to be
uninformative. Welch's t-test relaxes the assumption — it estimates
each sample's variance separately and uses the Welch–Satterthwaite
approximation for the degrees of freedom — at the cost of slightly
lower statistical power when the variances actually are equal. For
this study the variance-inequality risk is concrete: the
`rl_arbiter` arm's per-seed reward variance is plausibly very
different from the `none` arm's per-seed reward variance (the
random-arbiter trajectories are short and noisy, the
preference-arbiter trajectories may be long and stable or short and
collapsed depending on the seed), and the `override_layer` arm's
per-seed `override_fire_count` variance is mechanically zero (it's
always zero) which is the maximally pathological input for an
equal-variance test. Welch is the right default in this regime.

### 4.2 Two-sided alternative, p<0.05 / p<0.01 / p<0.001

The alternative hypothesis is two-sided: we are interested in whether
the metric moves in either direction relative to the `none` arm, not
only in whether it moves in a predicted direction. This matters for
two of the arms in particular. The `preference_plane` arm could move
`mean_reward` either up or down (Section 2.2), and reporting only a
one-sided p-value would either miss a "moves the wrong direction"
signal or require us to commit to the direction in advance, neither
of which is desirable for an ablation study whose purpose is to
establish that the component DOES SOMETHING. The `override_layer`
arm could similarly move `mean_memory_util` upward (the predicted
direction) or downward (if the override fallback action was
unexpectedly worsening memory pressure, an outcome we would want to
flag and not silently miss). Significance markers follow the
conventional three-tier scheme: `*` for `p<0.05`, `**` for
`p<0.01`, `***` for `p<0.001`, and `ns` for not-significant, all
applied to the two-sided p-value.

### 4.3 The n=3 caveat

Three seeds is the bare minimum at which the t-test is even
defined — at n=2 there is no within-group variance estimate, and at
n=1 the test is undefined entirely. At n=3 the test is defined but
its statistical power is low: the effect size has to be large
relative to the seed-to-seed variance for the p-value to clear even
the `*` threshold. A 1.5x reward reduction with seed-to-seed
standard deviation comparable to the mean will typically NOT clear
`p<0.05` at n=3, even though a reasonable reader would call that a
real effect. The honest interpretation of an `ns` row at n=3 is
therefore "we lack the statistical power to distinguish this arm
from the baseline" rather than "this arm has the same effect as the
baseline" — a distinction Section 6 reiterates for reviewer-facing
clarity. The follow-up plan is to re-run with 10+ seeds once the
Orin path is unblocked, at which point the per-arm power increases
substantially and effects of the size we expect from the preference
plane (small but real) become detectable.

### 4.4 Why not Mann-Whitney U or another rank-based test

Mann-Whitney U is a defensible alternative to Welch's t-test when the
underlying distributions are non-normal and when the sample size is
large enough to make the rank statistic informative. At n=3 versus
n=3, the U-statistic takes one of a small finite set of values
(specifically, U can range from 0 to 9, and the smallest two-sided
p-value attainable in the discrete null distribution is 0.10 — i.e.
no n=3 vs n=3 outcome can reach p<0.05 under Mann-Whitney). This
makes the rank-based alternative useless for the present sample size.
A bootstrap resampling test would have similar issues — the
bootstrap of three values cannot generate a sampling distribution
informative enough to support a p<0.05 conclusion. Welch's t-test,
even with its parametric normality assumption, is the only standard
option that is well-defined and informative at n=3.

### 4.5 Why not paired t-test

The paired t-test would be appropriate if each seed of the `none` arm
were matched to a corresponding seed of the ablated arm in some way
that controlled for seed-specific noise — for instance, if both arms
shared the same env reset sequence and only the substituted component
changed. In the runner's current design, every aspect of randomness
that depends on the seed (env reset, framework RNG, arbiter RNG) is
seeded from the same `cfg.seed`, so seeds 0/1/2 of `none` and seeds
0/1/2 of `rl_arbiter` do share the env reset sequence. A paired test
would therefore not be unreasonable. We chose Welch (unpaired) for
two reasons. First, the substituted components themselves consume
random bytes in different patterns — the `_PreferencePPOArbiter`
calls `rng.random()` once per step, the `_RandomArbiter` calls
`rng.randint(...)` once per step, and they consume from different
RNG instances — so even with the same seed the post-substitution RNG
state is not directly comparable across arms. The "matched seed"
intuition that motivates a paired test is therefore weaker than it
looks. Second, at n=3 the gain in power from a paired test is small
relative to the loss in degrees of freedom (paired t at n=3 has 2 df
versus Welch's typical 3-4 df), and the unpaired test is the more
defensible default for an audience unfamiliar with the runner's
seeding internals. We document the alternative here so a future
revision with more seeds can switch.

## 5. Substitute env and known deviations

This section documents three deviations from the original W8 ablation
spec, each of which a downstream reader needs to know about before
citing the resulting numbers.

### 5.1 CartPole-v1 as a sanity-only substitute

The original spec called for a 4-D MORL env so the four entries of the
preference plane could each have a meaningful objective channel.
Two candidates were considered: pybullet HalfCheetah, which is the
canonical multi-objective continuous-control benchmark in the MORL
literature, and the in-house DAG env that ships with TetraRL. Neither
was usable for this run. Pybullet HalfCheetah is not available in this
venv (a Week 7 finding documented in the corresponding progress
file — the wheel build fails on the dev machine), and the in-house
DAG env emits 3-D rewards at present, not the 4-D vector the
preference plane is sized for. CartPole-v1, a single-objective
discrete-action env, was the available fallback.

This means CartPole is a sanity proxy, NOT the headline result. The
paper Table 4 caveats this exactly that way: the row labels still
include the five arms, and the qualitative patterns we expect to see
(arbiter ablation devastates reward, override ablation drops fire
count to zero, preference ablation moves reward, resource ablation is
inert on the substitute path) are all observable on CartPole, but the
quantitative effect sizes do not transfer to the 4-D MORL setting the
paper headlines. The 4-D ablation on a properly-shaped DAG env or on
a freshly-built MORL benchmark is deferred as a follow-up; the
hypothesis-testing methodology in Section 4 is env-agnostic and will
carry over without modification.

### 5.2 The `ablation_full.yaml` `agent_type` fix

The `tetrarl/eval/configs/ablation_full.yaml` template originally
shipped with `agent_type: random` for every config. With
`agent_type: random`, the `_make_rl_arbiter` factory returns
`_RandomArbiter` regardless of the `ablation` field — the random
arbiter is what the `rl_arbiter` ablation arm substitutes IN, so a
random base policy makes three of the four non-`none` arms no-ops:
the `preference_plane` arm has nothing to omega-condition because the
random arbiter ignores omega, the `rl_arbiter` arm substitutes
random-for-random and produces an identical arbiter, and the
`override_layer` arm fires the override against an already-bad
arbiter so the reward delta is dominated by random-arbiter noise
rather than by the override's behaviour. The fix in this PR is to
update `ablation_full.yaml` to `agent_type: preference_ppo` for every
row, matching the `ablation_smoke.yaml` template, so the `none` arm
runs the omega-conditioned arbiter and the four non-`none` arms each
have a meaningful contrast to draw against. The yaml-level diff is
small; the semantic change is that the matrix is now an actual
ablation matrix rather than a no-op grid.

### 5.3 Mac substitute path vs. Orin physical path

This attempt ran on Mac because the Orin lab bastion was unreachable
(SSH timeout to both `nano2` and `orin1`) at the time the W8 ablation
deliverable needed to land. The runner is platform-agnostic (Section
5 of `docs/week8_eval_runner_design.md`), so the same code drives
both paths, but the metrics produced on Mac fall into two classes
with very different fidelity to the eventual Orin numbers.

The first class is the bit-exact-equivalent metrics. The synthetic
telemetry stream the runner uses is deterministic given the seed, the
`memory_util = 0.1 + 0.001 * step` ramp does not depend on hardware
behaviour, and the synthetic `energy_j = 1e-3 * (action + 1)` per-step
energy is computed from the action choice alone. As a result,
`mean_reward`, `override_fire_count`, `mean_memory_util`,
`mean_energy_j`, and `oom_events` are all bit-exact equivalent to
what the same code path would produce on Orin under the same seed.
These five metrics can be cited from the Mac run with confidence.

The second class is the platform-dependent metrics. `tail_p99_ms`
and `wall_time_s` reflect host-side timing (Mac CPU jitter, OS
scheduling, no real DVFS) and are NOT bit-exact equivalents to Orin.
On Orin these numbers would reflect the frequency island chosen by
the resource manager (or pinned-max for the `resource_manager`
ablation arm), the tegrastats sampling jitter, and the actual cost
of the framework step on the Cortex-A78AE cluster. The
`tail_p99_ms` column should be re-run on Orin once SSH is restored;
the re-run command and `out_dir` convention are documented in
`progress.md` for this PR.

## 6. Results interpretation guide

For a reviewer who is reading Table 4 cold, the following three
patterns are the ones to look for, in this order.

First, a row with `***` in the `p (reward)` column is strong evidence
that ablating the named component hurts (or, more rarely, helps)
reward. The `rl_arbiter` arm is the row most likely to show this
pattern, because the effect size of replacing a coherent policy with
uniform-random sampling on CartPole is large relative to the
seed-to-seed reward variance. A `***` here is the cleanest possible
evidence that the named component is doing real work; the absence of
`***` here would be a serious problem and would motivate either a
larger seed count or a harder substitute env.

Second, a row marked `ns` is NOT evidence that the component does
nothing. There are two distinct ways to land on `ns` at n=3, and only
one of them is the conclusion a casual reader will jump to. The
benign reading is "the substitute is actually similar to the real
component on this metric / on this env" — for instance, the
`resource_manager` arm on the Mac substitute path is mechanically
equivalent to `none` because no DVFS is wired (Section 2.3, Section
5.3), so an `ns` here just tells the reader that this row is not
informative on this platform. The other reading is "the test lacks
statistical power at n=3 to detect the effect size that is actually
present" — the `preference_plane` arm is the most likely candidate
for this reading, because the predicted effect size on `mean_reward`
(a shift driven by changing the marginal action probability from
0.7 to 0.5) is moderate and may not clear the `*` bar with three
seeds. Either reading is consistent with `ns`; the design of the
follow-up (more seeds, real DVFS, harder env) is what disambiguates
between them.

Third, the `override_layer` arm is the row where the most informative
movement is in `override_fire_count` (which by construction drops to
0 under the null variant) and in `mean_memory_util` and `oom_events`
(which can rise once the safety shield is removed). It is NOT the row
where to look for reward movement, because the override layer's
behaviour on CartPole is to swap a single fallback action in for the
arbiter's choice on a small fraction of late-episode steps, which
does not move episode return enough to clear the t-test's bar at
n=3. A reviewer who is looking only at the reward column for this
arm is looking at the wrong column; the design intent is that
override-layer evidence comes from the safety-relevant metrics, not
the reward metric.

A fourth, more subtle pattern is worth flagging for completeness.
Across all four ablation arms, if `oom_events` is zero for every row
(as it is expected to be on CartPole-v1, since the synthetic memory
ramp tops out at 0.299 within the v1 step cap), the `oom_events`
column carries no information. The paper Table 4 should either
suppress the column for the CartPole result or include it with an
explicit note that `0` is the same in every row by construction; a
silent inclusion would invite the reader to over-read the absence of
movement.
