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

## Validation result (snapshot evaluation)

| Metric | Value |
|--------|-------|
| Checkpoint | `runs/week1_orin_validation/best_model.pt` |
| Training steps (step_count) | 39,988 |
| Platform | Orin AGX (CUDA) |
| Achieved HV | 744.0 (ref point `[0, -25]`) |
| Unique Pareto points | 1 of 10 optimal |
| Achieved point | `(124, -19)` — deepest treasure only |
| Failure mode | Mode collapse: treasure found only for omega heavily favoring obj-1 |

**Interpretation**: The HV of 744 is mathematically valid but degenerate — it comes
from a single point `(124, -19)` which yields `124 * 6 = 744` area above the
reference. The agent has not learned preference-conditioned behavior: for 9 of 11
preference anchors, the policy times out at `(0, -200)`. This is expected at ~40k
optimization steps (training was killed at ~35 min into a 200k-frame run due to VPN
interruption).

## What was deferred

- **Full 200k-frame convergence curve**: The training run was interrupted before
  `progress.json` was written. A follow-up run with periodic JSON checkpointing
  (every 5k frames) will produce the time-series HV convergence plot.
- **Multi-point Pareto coverage**: The agent needs the full training curriculum
  (epsilon decay through 50k steps + continued HER relabeling) to learn diverse
  preference-conditioned policies across all 10 DST treasures.
- **Tail-latency profiling**: Scheduled for Week 7.

## Follow-up actions

1. Re-run complete 200k training on Orin with `--eval-freq 5000` to generate
   `progress.json` (estimated ~2 hours wall time).
2. Generate HV convergence figure using `scripts/plot_hv_convergence.py`.
3. Validate that final HV approaches the 229+ target (known optimal HV with
   ref `[0, -25]` is ~1155 for full 10-point DST front).
