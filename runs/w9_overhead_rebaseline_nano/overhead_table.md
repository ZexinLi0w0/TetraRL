# Week 9 — Re-baselined Per-component Overhead

- device: `Linux ubuntu 5.10.120-tegra aarch64`
- platform: `orin_nano`
- agent: `preference_ppo`
- env: `dag_scheduler_mo`
- n_steps: 5000
- mean_bare_step_ms: 0.0660
- mean_framework_step_ms: 3.0513
- framework_overhead_pct: 4520.6664
- acceptance_threshold_pct: 30.0

| component | mean_ms | p50_ms | p99_ms | mem_mb | rss_mb | n_samples |
|---|---|---|---|---|---|---|
| dvfs_controller_set | 1.0490 | 1.0173 | 1.3510 | 0.0084 | 4.0000 | 5000 |
| override_layer_step | 0.0102 | 0.0100 | 0.0116 | 0.0011 | 0.0000 | 5000 |
| preference_plane_get | 0.0086 | 0.0085 | 0.0097 | 0.0029 | 0.0000 | 5000 |
| resource_manager_decide | 0.0125 | 0.0124 | 0.0139 | 0.0005 | 0.2578 | 5000 |
| rl_arbiter_act | 0.0550 | 0.0546 | 0.0675 | 0.0007 | 0.0000 | 5000 |
| tegra_daemon_sample | 0.0297 | 0.0293 | 0.0347 | 0.0026 | 0.0000 | 5000 |
