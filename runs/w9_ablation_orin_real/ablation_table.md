### Table 4. Ablation study (CartPole-v1, n=3 seeds, 200 episodes; reward saturates at 1.0/step on CartPole — see n_steps for the action signal)

| Ablation | n_steps (μ ± σ) | Override fires (μ ± σ) | Tail p99 ms (μ ± σ) | Mean energy J (μ ± σ) | Mean Reward (μ ± σ) | p (n_steps) | p (override) | p (tail p99) | p (reward) | sig |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 3122.667 ± 38.083 | 51.000 ± 8.718 | 0.065 ± 0.002 | 0.001 ± 0.000 | 1.000 ± 0.000 | — | — | — | — |  |
| preference_plane | 4354.667 ± 213.149 | 292.333 ± 56.501 | 0.069 ± 0.001 | 0.001 ± 0.000 | 1.000 ± 0.000 | 0.008 | 0.016 | 0.041 | — | ** |
| resource_manager | 3122.667 ± 38.083 | 51.000 ± 8.718 | 0.064 ± 0.001 | 0.001 ± 0.000 | 1.000 ± 0.000 | 1.000 | 1.000 | 0.735 | — | ns |
| rl_arbiter | 4264.667 ± 119.525 | 292.667 ± 40.377 | 0.044 ± 0.000 | 0.001 ± 0.000 | 1.000 ± 0.000 | 0.002 | 0.007 | 0.003 | — | ** |
| override_layer | 3135.667 ± 16.258 | 0.000 ± 0.000 | 0.060 ± 0.000 | 0.001 ± 0.000 | 1.000 ± 0.000 | 0.628 | 0.010 | 0.050 | — | ** |
