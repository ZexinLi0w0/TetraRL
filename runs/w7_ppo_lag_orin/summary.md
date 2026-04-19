# PPO-Lagrangian run summary

- env: `CartPole-v1` (requested: `CartPole-v1`)
- fell_back_to_pendulum: `False`
- total_steps: requested=50000, completed=49920
- knob_mapper: `n_steps` (closed-loop coupling deferred)
- with_override: `True`, fires=0
- seed: 0

## Final lambdas
| constraint | target | final lambda | mean violation |
|---|---|---|---|
| latency_ms | 30.000 | 0.0000 | 0.0000 |
| energy_j   | 5.000 | 100.0000 | 5.0000 |
| memory_util| 0.850 | 0.0000 | 0.0000 |

Per-step JSONL: `training_log.jsonl`. Plots: `lambdas_convergence.png`, `violation_rate.png` (if matplotlib available).