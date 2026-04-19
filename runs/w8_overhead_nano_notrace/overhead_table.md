# Week 8 — Per-component Overhead (Table 5 candidate)

- device: `Linux ubuntu 5.10.120-tegra aarch64`
- platform: `nano`
- n_steps: 5000
- mean_bare_step_ms: 0.0302
- mean_framework_step_ms: 0.0761
- framework_overhead_pct: 152.0740
- total_components_profiled: 8

| component | mean_ms | p50_ms | p99_ms | mem_mb | rss_mb | n_samples |
|---|---|---|---|---|---|---|
| dvfs_controller_set | 0.0076 | 0.0059 | 0.0200 | 0.0000 | 0.0000 | 5000 |
| lag_feature_extract | 0.0474 | 0.0461 | 0.0631 | 0.0000 | 0.0000 | 5000 |
| override_layer_step | 0.0028 | 0.0028 | 0.0036 | 0.0000 | 0.0000 | 5000 |
| preference_plane_get | 0.0028 | 0.0028 | 0.0036 | 0.0000 | 0.0000 | 5000 |
| replay_buffer_add | 0.1823 | 0.1767 | 0.2950 | 0.0000 | 0.0000 | 5000 |
| resource_manager_decide | 0.0040 | 0.0039 | 0.0048 | 0.0000 | 0.0000 | 5000 |
| rl_arbiter_act | 0.0056 | 0.0055 | 0.0069 | 0.0000 | 0.0000 | 5000 |
| tegra_daemon_sample | 0.0112 | 0.0111 | 0.0130 | 0.0000 | 0.0000 | 5000 |
