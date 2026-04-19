# Week 7 Pareto sweep summary

- env: `dag_scheduler_mo`
- steps-per-omega: 10000
- n_omegas: 5
- obj_num: 4
- ref_point: [0.0, -100.0, -100.0, -100.0]

**NOTE: synthetic dev points used.** Reason: omega 0 ([1.0, 0.0, 0.0, 0.0]): ValueError: shapes (4,) and (3,) not aligned: 4 (dim 0) != 3 (dim 0)

## Pareto summary

- Total points: 5
- Pareto points: 3
- 4-D HV: 126500.8269

| Dimension | Min | Max | Mean |
|---|---:|---:|---:|
| Throughput | -0.5443 | 0.1257 | -0.3181 |
| Latency | -0.3163 | 0.3616 | -0.0289 |
| Energy | 0.4116 | 1.3040 | 0.7854 |
| Memory | 0.1049 | 1.0425 | 0.6982 |

## Artifacts

- points.csv: `/experiment/zexin/TetraRL/runs/w7_pareto_orin/points.csv`
- projection_Throughput_vs_Latency: `/experiment/zexin/TetraRL/runs/w7_pareto_orin/projection_Throughput_vs_Latency.png`
- projection_Throughput_vs_Energy: `/experiment/zexin/TetraRL/runs/w7_pareto_orin/projection_Throughput_vs_Energy.png`
- projection_Throughput_vs_Memory: `/experiment/zexin/TetraRL/runs/w7_pareto_orin/projection_Throughput_vs_Memory.png`
- combined: `/experiment/zexin/TetraRL/runs/w7_pareto_orin/pareto_combined.svg`
