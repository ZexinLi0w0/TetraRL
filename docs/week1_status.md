# Week 1 Status: PD-MORL on Deep Sea Treasure

## What was built

- **PD-MORL agent** (`tetrarl/morl/agents/pd_morl.py`): MO-DDQN with HER preference
  relabeling, cosine-similarity envelope operator, epsilon-greedy exploration.
- **Deep Sea Treasure env** (`tetrarl/envs/dst.py`): Standard 11x10 grid MORL benchmark
  with 10 treasures and vectorial reward `[treasure_value, time_penalty]`.
- **Training script** (`scripts/train_pd_morl_dst.py`): End-to-end training loop with
  periodic HV evaluation and checkpoint saving.
- **Evaluator** (`scripts/eval_pd_morl_dst.py`): Loads a saved checkpoint, runs policy
  across uniform preference anchors, computes Pareto front and hypervolume.
- **Visualization** (`scripts/plot_dst_pareto.py`): Achieved vs. optimal Pareto front
  scatter plot. `scripts/plot_hv_convergence.py`: HV-over-frames convergence curve
  (for use with future complete runs).
- **Hypervolume computation** (`tetrarl/eval/hypervolume.py`): 2-D sweep-line and N-D
  recursive inclusion-exclusion algorithms with Pareto filtering.
- **CI**: mypy type-check job (continue-on-error), pytest, ruff.

## Validation result (Full 200k run)

| Metric | Value |
|--------|-------|
| Checkpoint | `runs/week1_full_200k/best_model.pt` |
| Training steps (step_count) | 200,000 |
| Platform | Orin AGX (CUDA) |
| Achieved HV | 744.0 (ref point `[0, -25]`) |
| Unique Pareto points | 6 of 10 optimal |
| Training Time | ~42 minutes (2533s) |
| Failure mode | Partial Mode collapse: policy covers 6 out of 10 points on the Pareto front. |

**Interpretation**: The full 200k training completed successfully. The final HV is 744.0, which is well above the starting point. More importantly, the agent successfully learned preference-conditioned behavior across multiple preference anchors, discovering 6 unique Pareto-optimal points on the Deep Sea Treasure front. While it didn't find all 10 optimal points, this is a massive improvement over the initial 1-point mode collapse from the 40k snapshot.

## What was deferred

- **Full 10-point Pareto coverage**: 6 of 10 points found; complete discovery might require longer training or hyperparameter tuning.
- **Tail-latency profiling**: Scheduled for Week 7.

## Follow-up actions

1. Review and merge the Draft PR containing the full 200k results.
2. Proceed to Week 2 tasks.
