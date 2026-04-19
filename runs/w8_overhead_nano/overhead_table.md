# Week 8 — Per-component Overhead (Table 5 candidate)

- device: `Linux ubuntu 5.10.120-tegra aarch64`
- platform: `nano`
- n_steps: 5000
- mean_bare_step_ms: 0.0301
- mean_framework_step_ms: 1.4898
- framework_overhead_pct: 4841.4924
- total_components_profiled: 8

| component | mean_ms | p50_ms | p99_ms | mem_mb | rss_mb | n_samples |
|---|---|---|---|---|---|---|
| dvfs_controller_set | 0.0215 | 0.0212 | 0.0240 | 0.0083 | 0.2578 | 5000 |
| lag_feature_extract | 0.0882 | 0.0866 | 0.1111 | 0.0087 | 0.2578 | 5000 |
| override_layer_step | 0.0074 | 0.0073 | 0.0084 | 0.0017 | 0.2578 | 5000 |
| preference_plane_get | 0.0085 | 0.0084 | 0.0095 | 0.0108 | 0.6562 | 5000 |
| replay_buffer_add | 0.2369 | 0.2294 | 0.4764 | 0.0083 | 1.4336 | 5000 |
| resource_manager_decide | 0.0118 | 0.0116 | 0.0130 | 0.0006 | 0.0000 | 5000 |
| rl_arbiter_act | 0.0103 | 0.0102 | 0.0118 | 0.0004 | 0.0000 | 5000 |
| tegra_daemon_sample | 0.0268 | 0.0265 | 0.0312 | 0.0080 | 0.2578 | 5000 |
