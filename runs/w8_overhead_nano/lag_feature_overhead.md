# Week 8 — LAG Feature Extract Overhead

- device: `Linux ubuntu 5.10.120-tegra aarch64`
- platform: `nano`
- n_steps: 5000
- W8 criterion: `lag_feature_extract.p99_ms < 0.5`
- result: **PASS**

| component | mean_ms | p50_ms | p99_ms | mem_mb | rss_mb | n_samples |
|---|---|---|---|---|---|---|
| lag_feature_extract | 0.0882 | 0.0866 | 0.1111 | 0.0087 | 0.2578 | 5000 |
