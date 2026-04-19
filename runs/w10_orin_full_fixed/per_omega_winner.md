# Per-ω winner table (TetraRL = `preference_ppo`)

Ref point (reward, -latency, -memory, -energy) = (-0.1, -1, -0.15, -0.01).

| ω | preference_ppo | duojoule | envelope_morl | focops | max_action | max_performance | pcn | pd_morl | ppo_lagrangian | winner |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| ω0 reward-only [1.00, 0.00, 0.00, 0.00] | 1.485e-05 | 1.090e-06 | 1.397e-05 | 4.566e-06 | 1.097e-06 | **2.080e-04** | 9.701e-05 | - | 6.940e-06 | max_performance |
| ω1 latency-only [0.00, 1.00, 0.00, 0.00] | 4.602e-07 | 4.682e-07 | **4.270e-06** | 2.767e-06 | 4.705e-07 | 2.411e-06 | 2.652e-06 | - | 2.774e-06 | envelope_morl |
| ω2 memory-only [0.00, 0.00, 1.00, 0.00] | 4.437e-07 | 4.515e-07 | **2.538e-06** | 2.402e-06 | 4.539e-07 | 2.180e-06 | 2.501e-06 | - | 2.407e-06 | envelope_morl |
| ω3 energy-only [0.00, 0.00, 0.00, 1.00] | 5.643e-07 | **1.030e-05** | 3.166e-06 | 7.641e-06 | 5.771e-07 | 7.210e-06 | 6.278e-06 | - | 8.810e-06 | duojoule |
| ω4 uniform [0.25, 0.25, 0.25, 0.25] | 4.554e-06 | 5.749e-07 | 4.647e-06 | 4.160e-06 | 5.779e-07 | 5.531e-06 | **5.869e-06** | - | 5.020e-06 | pcn |
| ω5 reward-leaning [0.40, 0.30, 0.20, 0.10] | 7.468e-06 | - | - | - | - | **1.478e-05** | 1.224e-05 | 1.477e-05 | - | max_performance |
| ω6 latency-leaning [0.10, 0.40, 0.30, 0.20] | 2.965e-06 | - | - | - | - | 4.406e-06 | **4.633e-06** | 4.407e-06 | - | pcn |
| ω7 mem+energy-leaning [0.20, 0.20, 0.30, 0.30] | 4.914e-06 | - | - | - | - | 5.919e-06 | 5.783e-06 | **5.922e-06** | - | pd_morl |
| ω8 reward+latency-leaning [0.30, 0.30, 0.20, 0.20] | 6.095e-06 | - | - | - | - | 6.592e-06 | **6.939e-06** | 6.589e-06 | - | pcn |

## Summary

TetraRL Native (`preference_ppo`) does NOT win on any of the 9 ω vectors evaluated.

Significant wins (Welch t-test p<0.05, mean(TetraRL) > mean(baseline)) per baseline:

| baseline | ω at which TetraRL is significantly better |
| --- | :-: |
| duojoule | 2 / 9 |
| envelope_morl | 0 / 9 |
| focops | 1 / 9 |
| max_action | 2 / 9 |
| max_performance | 0 / 9 |
| pcn | 0 / 9 |
| pd_morl | 0 / 9 |
| ppo_lagrangian | 1 / 9 |

Aggregate: TetraRL has **6** statistically significant wins across all (baseline x ω) cells (8 baselines x 9 ω = 72 cells).

## Value-proposition characterisation

TetraRL's selling points are NOT "highest aggregate HV under any fixed ω". The empirical numbers above honestly reflect that point: on Pareto-front anchors and intermediate ω alike, several MORL baselines often match or beat `preference_ppo` on the dominated-HV scalar.

What TetraRL provides instead is (a) **adaptability when the user preference shifts at runtime** — the preference plane can be re-targeted between episodes without retraining the arbiter — and (b) **constraint respect via the override layer**, which fires on telemetry breaches and clamps the executed action to a safe fallback (see `lagrangian_violation_table.md` and the Week 10 override-telemetry tests). Per-ω HV is therefore a complement to, not a substitute for, the dynamic-preference and constraint-violation evidence.
