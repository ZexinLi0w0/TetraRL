# Multi-tenant Nano-GRPO comparison

- n_seeds: 3
- n_steps: 200
- model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- host: ubuntu

| Metric | with-critic (PPO) | without-critic (GRPO) | Δ (without − with) | Δ % |
|---|---|---|---|---|
| GRPO step latency (mean ms) | 1.18e+03 ± 13.8 (n=3) | 1.15e+03 ± 9.14 (n=3) | -22.7 | -1.93 |
| GRPO step latency (p99 ms) | 1.42e+03 ± 19.5 (n=3) | 1.39e+03 ± 26.6 (n=3) | -21.1 | -1.49 |
| GRPO energy/step (J) | 22.8 ± 0.138 (n=3) | 22.4 ± 0.0605 (n=3) | -0.394 | -1.73 |
| GRPO energy total (J) | 4.55e+03 ± 27.6 (n=3) | 4.47e+03 ± 12.1 (n=3) | -78.8 | -1.73 |
| GRPO mem peak (MB) | 2.05e+04 ± 326 (n=3) | 2.03e+04 ± 204 (n=3) | -171 | -0.836 |
| GRPO torch peak alloc (MB) | 1.05e+04 ± 0 (n=3) | 1.05e+04 ± 0 (n=3) | -11.1 | -0.105 |
| GRPO GPU util mean (%) | 71.9 ± 0.629 (n=3) | 72.1 ± 0.309 (n=3) | 0.165 | +0.23 |
| Override fire count | 1 ± 0 (n=3) | 1 ± 0 (n=3) | 0 | +0 |
| Override fire rate (/step) | 0.005 ± 0 (n=3) | 0.005 ± 0 (n=3) | 0 | +0 |
| LLM inference latency mean (ms) | 645 ± 14.4 (n=3) | 638 ± 6.79 (n=3) | -7.2 | -1.12 |
| LLM inference latency p99 (ms) | 739 ± 11.8 (n=3) | 732 ± 6.93 (n=3) | -7.6 | -1.03 |
| Perception fps actual | 29.9 ± 0.000397 (n=3) | 29.9 ± 0.000383 (n=3) | 0.000659 | +0.0022 |
| Perception latency mean (ms) | 7.32 ± 0.0411 (n=3) | 7.29 ± 0.0152 (n=3) | -0.0283 | -0.387 |

Note: Δ%>0 means without-critic is HIGHER than with-critic; for latency/energy/memory, NEGATIVE Δ% favors GRPO (without-critic).
