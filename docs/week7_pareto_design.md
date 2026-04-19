# Week 7 Task 4 -- 4-D Pareto front visualization

## Goals

* Provide a small, well-tested helper module
  (`tetrarl/eval/pareto.py`) that wraps the existing HV / Pareto-filter
  utilities under spec-friendly names and adds visualization +
  Markdown-summary helpers.
* Provide a CLI driver (`scripts/week7_pareto_sweep.py`) that trains
  the preference-conditioned PPO at the four 4-D corners of the
  preference simplex plus one uniform interior point, aggregates the
  five resulting 4-D objective returns, and saves Pareto-front
  visualizations + a Markdown summary.

## Why 3 x 2-D projections (not parallel coordinates)?

A 4-D Pareto front cannot be plotted directly. Two standard options:

1. **Parallel-coordinate plots** -- one polyline per Pareto point
   crossing four parallel axes. Information-dense but visually noisy
   for paper figures: lines cross, ordering of axes biases perception,
   and the eye cannot easily judge dominance relationships.
2. **2-D scatter projections** -- one scatter per pair of objectives.
   Projections lose joint information (a point dominated in 2-D may be
   non-dominated in 4-D and vice versa) but each panel reads exactly
   like a familiar bi-objective Pareto plot, which is essential for a
   paper-figure narrative.

We adopt option (2). The 4-D HV indicator is shown in each subplot's
title to keep the canonical aggregated metric in scope, while the
in-panel red points are the **per-projection** Pareto subset (the
non-dominated set within those two axes only). This deliberate
asymmetry -- aggregate metric in title, projection-local front in
markers -- prevents the misleading suggestion that grey points are
dominated in 4-D.

The default projections fix Throughput on the X axis and pair it with
each of (Latency, Energy, Memory) on Y. This matches the T-A, E-A, M-A
phrasing from the action plan; the `pairs` kwarg lets callers pick
other combinations.

A combined SVG (`pareto_combined.svg`) places the three panels
side-by-side for inclusion as a single figure.

## Pareto-filter semantics

We assume **maximization on all objectives** throughout the codebase
(`tetrarl/eval/hypervolume.py::pareto_filter`). Concretely a point
`p` is *strictly dominated* by `q` iff
`np.all(q >= p) and np.any(q > p)`.

For objectives that are naturally minimized (latency, energy, memory),
the convention is to negate them at env / agent boundaries so the agent
sees rewards. This keeps a single sign convention in the eval code.
The default reference point in the CLI is therefore
`(0, -100, -100, -100)`: the throughput axis can be 0 or larger; the
three negated penalty axes have a finite lower bound large enough to
be dominated by any reasonable agent.

## HV indicator definition + reference point

The dominated hypervolume of a Pareto front `P` w.r.t. a reference
point `r` is the Lebesgue measure of

    {y in R^d  :  y <= p for some p in P  and  y >= r}

The reference point must be dominated by every Pareto point for the
measure to make sense; in `compute_hv` points that fail this check are
silently filtered before the measure is computed.

We use the existing recursive inclusion-exclusion implementation in
`hypervolume._hv_nd` for d >= 3. For our typical 5-point sweep the
runtime is negligible.

## Mapping to the action-plan four objectives

The preference vector layout in the rest of TetraRL is

    omega = [throughput, accuracy, energy_efficiency, memory_efficiency]

with all four phrased as "more is better". The default
`dim_labels = ["Throughput", "Latency", "Energy", "Memory"]` in
`plot_2d_projections` re-maps the latter three to the user-facing
"penalty" names so the paper figure reads naturally; callers that want
the agent-internal labels can pass `dim_labels` explicitly.

## DAG-scheduler-MO objective dimensions (current state)

The DAG scheduler env at `tetrarl/envs/dag_scheduler.py` currently
returns a 3-vector reward,
`[throughput, -energy, -peak_memory_delta]` (see the docstring on
`DAGSchedulerEnv` and the `reward_dim = 3` field). This is **not yet
4-D**: there is no separate accuracy / latency channel.

For Task 4's visualization we therefore document the following
deviation:

* The CLI driver attempts to train at the four 4-D corners. When
  `--obj-num=4` is requested with the 3-D DAG env, the inner dot
  product `omega . r_vec` raises a shape error (caught by the script's
  try/except).
* On any such failure the driver falls back to a **synthetic** 4-D
  point cloud (5 points sampled from `np.random.default_rng(seed).normal`)
  so the visualization pipeline still produces a non-empty Pareto plot
  and `summary.md` for development.
* The synthetic-mode fallback is loudly marked in `summary.md` with
  "synthetic dev points" and a `synthetic: true` flag in
  `result.json`. Once a future week extends the DAG env to a 4-D
  reward (e.g. by adding a deadline-miss accuracy term), the fallback
  branch is bypassed automatically and the same artifacts will reflect
  real training outcomes.

The HalfCheetah path (`--env halfcheetah_mo`) requires MuJoCo, which
is not installed in the Mac venv. The driver attempts
`mo-halfcheetah-v4` first and falls back to
`mo-mountaincarcontinuous-v0` if MuJoCo is missing. The 2-D MountainCar
fallback also will not match `--obj-num=4`, in which case the
synthetic fallback kicks in for the same reason.

## Files

* `tetrarl/eval/pareto.py` -- library (compute_hv, pareto_front,
  plot_2d_projections, pareto_summary_table).
* `scripts/week7_pareto_sweep.py` -- CLI driver.
* `tests/test_pareto.py` -- 10 unit tests including a tmp-dir test
  that verifies the PNG + SVG artifacts are written to disk.
* `docs/week7_pareto_design.md` -- this file.
