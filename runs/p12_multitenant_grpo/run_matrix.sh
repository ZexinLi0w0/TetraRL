#!/bin/bash
set -e
cd /experiment/zexin/TetraRL
source /experiment/zexin/venvs/r3/bin/activate
export HF_HOME=/experiment/zexin/.cache/huggingface
export TRANSFORMERS_VERBOSITY=error

LOGDIR=/experiment/zexin/TetraRL/runs/p12_multitenant_grpo
mkdir -p "$LOGDIR"
echo "[$(date +%T)] start matrix (pid=$$)" > "$LOGDIR/matrix.log"

for seed in 0 1 2; do
  for cond in with_critic without_critic; do
    out="$LOGDIR/${cond}_seed${seed}"
    mkdir -p "$out"
    echo "[$(date +%T)] start  $cond seed=$seed -> $out" >> "$LOGDIR/matrix.log"
    set +e
    python scripts/p12_multitenant_nano_grpo_orin.py \
      --mode grpo --condition "$cond" --seed "$seed" --n-steps 200 \
      --out-dir "$out" --warmup-s 12 > "$out/run.log" 2>&1
    rc=$?
    set -e
    echo "[$(date +%T)] finish $cond seed=$seed rc=$rc" >> "$LOGDIR/matrix.log"
    if [ "$rc" -ne 0 ]; then
      echo "[$(date +%T)] WARN: $cond seed=$seed exited rc=$rc, continuing" >> "$LOGDIR/matrix.log"
    fi
  done
done
echo "[$(date +%T)] matrix complete" >> "$LOGDIR/matrix.log"
touch "$LOGDIR/.done"
