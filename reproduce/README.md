# Reproduction Scripts

This directory contains scripts for reproducing the experiments in the
TetraRL IEEE TC paper.

## Tiers

| Script | Duration | Purpose |
|--------|----------|---------|
| `run_smoke.sh` | 5–10 min | Sanity check: CartPole DST with MO-DQN-HER |
| `run_short.sh` | 30–60 min | Short validation: Pong + R⁴ metric tracking |
| `run_full_paper.sh` | Days | Full paper reproduction: all figures and tables |

## Output Convention

All logs are written to:
```
/experiment/zexin/TetraRL/reproduce/runs/{smoke,short,full}_<UTC-timestamp>/
```

## Usage

```bash
bash reproduce/run_smoke.sh
bash reproduce/run_short.sh
bash reproduce/run_full_paper.sh --list
bash reproduce/run_full_paper.sh --only <experiment-name>
```

<!-- Detailed per-experiment instructions will be added as experiments are implemented. -->
