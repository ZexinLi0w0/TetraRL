# Week 8 — Per-component Overhead (Table 5 candidate)

- device: `Linux ubuntu 5.10.120-tegra aarch64`
- platform: `nano`
- n_steps: 5000
- mean_bare_step_ms: 0.0306
- mean_framework_step_ms: 0.1009
- framework_overhead_pct: 229.7952
- total_components_profiled: 8

| component | mean_ms | p50_ms | p99_ms | mem_mb | rss_mb | n_samples |
|---|---|---|---|---|---|---|
| dvfs_controller_set | 0.0316 | 0.0064 | 0.0204 | 0.0000 | 0.0000 | 5000 |
| lag_feature_extract | 0.0495 | 0.0476 | 0.0694 | 0.0000 | 0.0000 | 5000 |
| override_layer_step | 0.0027 | 0.0026 | 0.0032 | 0.0000 | 0.0000 | 5000 |
| preference_plane_get | 0.0031 | 0.0030 | 0.0036 | 0.0000 | 0.0000 | 5000 |
| replay_buffer_add | 0.1816 | 0.1729 | 0.3022 | 0.0000 | 0.0000 | 5000 |
| resource_manager_decide | 0.0041 | 0.0040 | 0.0050 | 0.0000 | 0.0000 | 5000 |
| rl_arbiter_act | 0.0057 | 0.0055 | 0.0075 | 0.0000 | 0.0000 | 5000 |
| tegra_daemon_sample | 0.0114 | 0.0110 | 0.0216 | 0.0000 | 0.0000 | 5000 |
