#!/usr/bin/env bash
# Operator runbook for W10 Nano evaluation matrix (60 cells).
#
# Prereqs (on local Mac):
#   * nano2 SSH reachable: /Users/zexinli/login.sh nano2 'echo OK'
#   * Local worktree: /Users/zexinli/Downloads/TetraRL-w10-nano on branch
#     week10/nano-eval, with this commit (matrix YAML + helper scripts) merged.
#
# This script does NOT acquire the .nano_busy_w10nano.lock for you. Acquire it
# manually first (see Step 1) and release it on exit (Step 7).
set -euo pipefail

LOCK=/experiment/zexin/.nano_busy_w10nano.lock
NANO_REPO=/experiment/zexin/TetraRL
NANO_VENV=/experiment/zexin/venvs/tetrarl-nano
LOCAL_REPO=/Users/zexinli/Downloads/TetraRL-w10-nano
LOCAL_RUNS=$LOCAL_REPO/runs/w10_nano_matrix
MATRIX_YAML=tetrarl/eval/configs/w10_nano_matrix.yaml

echo "[w10-nano] Step 1/7: acquire mutex"
/Users/zexinli/login.sh nano2 "test -e $LOCK && cat $LOCK || echo no-lock"
/Users/zexinli/login.sh nano2 "echo p2-w10-nano-\$(date -u +%Y%m%dT%H%M%SZ) > $LOCK"

echo "[w10-nano] Step 2/7: pre-flight + repo sync + userspace governor"
for i in 1 2 3; do /Users/zexinli/login.sh nano2 'echo OK && date'; sleep 5; done
/Users/zexinli/login.sh nano2 "cd $NANO_REPO && git fetch origin && git checkout main && git pull --ff-only && git log --oneline -3"
/Users/zexinli/login.sh nano2 'echo zexin | sudo -S -v && for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo userspace | sudo tee $cpu > /dev/null; done'

echo "[w10-nano] Step 3/7: run 60-cell matrix on nano2 (~30-60 min)"
/Users/zexinli/login.sh nano2 "cd $NANO_REPO && source $NANO_VENV/bin/activate && python -m tetrarl.eval.runner --config $MATRIX_YAML --out-dir runs/w10_nano_matrix/"

echo "[w10-nano] Step 4/7: copy results back to local"
mkdir -p "$LOCAL_RUNS"
scp -P 8010 -r zexin@169.235.25.145:$NANO_REPO/runs/w10_nano_matrix/* "$LOCAL_RUNS/"

echo "[w10-nano] Step 5/7: HV summary + plots (local)"
cd "$LOCAL_REPO"
python scripts/week10_nano_hv_and_plots.py \
  --matrix-yaml "$MATRIX_YAML" \
  --runs-dir "$LOCAL_RUNS" \
  --out-dir "$LOCAL_RUNS"

echo "[w10-nano] Step 6/7: reset governor to schedutil"
/Users/zexinli/login.sh nano2 'echo zexin | sudo -S -v && for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo schedutil | sudo tee $cpu > /dev/null; done'

echo "[w10-nano] Step 7/7: release mutex"
/Users/zexinli/login.sh nano2 "rm -f $LOCK"

echo "[w10-nano] DONE. Inspect $LOCAL_RUNS/hv_summary.csv + hv_bar.png + pareto_scatter.png"
