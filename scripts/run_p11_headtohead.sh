#!/usr/bin/env bash
# =============================================================================
# P11 head-to-head sweep launcher (server1 / 8x RTX A4500)
# =============================================================================
#
# Launches the full P11 head-to-head experiment:
#   4 baselines x 3 seeds = 12 jobs (default)
#
#   tetrarl_pref_ppo : scripts/train_pref_ppo_dst.py
#   pd_morl          : scripts/train_pd_morl_dst.py
#   duojoule         : scripts/train_duojoule_dst.py
#   r3               : scripts/run_p11_r3_native.sh   (native dispatcher)
#
# Each job runs in its own tmux window inside the session "p11-headtohead".
# GPUs are pinned round-robin (i % 8), so with 12 jobs four GPUs host two
# jobs each (~1 GB resident per job, fits comfortably in 20 GB).
#
# Usage
# -----
#   bash scripts/run_p11_headtohead.sh                           # full sweep (12 jobs)
#   bash scripts/run_p11_headtohead.sh --smoke                   # 4 jobs, 1k frames, seed 0
#   bash scripts/run_p11_headtohead.sh --dry-run                 # print plan, no launch
#   bash scripts/run_p11_headtohead.sh --smoke --dry-run         # plan smoke run
#   FRAMES=200000 SEEDS="0 1 2" \
#     BASELINES="tetrarl_pref_ppo pd_morl duojoule r3" \
#     bash scripts/run_p11_headtohead.sh                         # explicit overrides
#
# Expected runtime
# ----------------
#   Full (200k frames, 12 jobs in parallel on 8 GPUs)  : ~5-6 hours wall clock
#   Smoke (1k frames, 4 jobs)                          : ~5 minutes
#
# Monitor
# -------
#   tmux attach -t p11-headtohead          # interactive attach (Ctrl-b d to detach)
#   tmux list-windows -t p11-headtohead    # list all 12 job windows
#   tail -f runs/p11_dst_headtohead/<baseline>_seed<N>/train.log
#   ls runs/p11_dst_headtohead/*/.done runs/p11_dst_headtohead/*/.failed 2>/dev/null | wc -l
#       # -> reaches 12 when the sweep is finished (any mix of .done/.failed)
#
# Exit policy
# -----------
#   We use `set -uo pipefail` (NOT -e). Individual job failures must surface as
#   .failed sentinels in their per-job logdir, NOT abort the launcher mid-flight.
# =============================================================================

set -uo pipefail

# -----------------------------------------------------------------------------
# Defaults (overridable via env)
# -----------------------------------------------------------------------------
FRAMES="${FRAMES:-200000}"
SEEDS_DEFAULT="0 1 2"
BASELINES_DEFAULT="tetrarl_pref_ppo pd_morl duojoule r3"
SEEDS="${SEEDS:-$SEEDS_DEFAULT}"
BASELINES="${BASELINES:-$BASELINES_DEFAULT}"

REPO_DIR="/data/zli/TetraRL-headtohead/TetraRL"
CONDA_ACTIVATE='source ~/anaconda3/etc/profile.d/conda.sh && conda activate tetrarl-baselines'
OUT_ROOT="runs/p11_dst_headtohead"
SESSION="p11-headtohead"
NUM_GPUS=8

# -----------------------------------------------------------------------------
# Flag parsing
# -----------------------------------------------------------------------------
SMOKE=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --smoke)   SMOKE=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,40p' "$0"
      exit 0
      ;;
    *)
      echo "[p11-headtohead] ERROR: unknown flag '$arg' (valid: --smoke, --dry-run, --help)" >&2
      exit 2
      ;;
  esac
done

if [[ "$SMOKE" -eq 1 ]]; then
  FRAMES=1000
  SEEDS="0"
  echo "[p11-headtohead] SMOKE mode: FRAMES=$FRAMES SEEDS='$SEEDS' (one seed per baseline)"
fi

# -----------------------------------------------------------------------------
# Pre-flight checks (skipped in dry-run mode so the plan can be inspected anywhere)
# -----------------------------------------------------------------------------
if [[ "$DRY_RUN" -eq 0 ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "[p11-headtohead] ERROR: tmux not found on PATH." >&2
    echo "                Install with: sudo apt install tmux" >&2
    exit 3
  fi

  # Activate env so the CUDA probe runs in the right interpreter.
  # shellcheck disable=SC1090
  source ~/anaconda3/etc/profile.d/conda.sh
  conda activate tetrarl-baselines || {
    echo "[p11-headtohead] ERROR: failed to activate conda env tetrarl-baselines" >&2
    exit 4
  }

  if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
    echo "[p11-headtohead] ERROR: CUDA not available in tetrarl-baselines env." >&2
    echo "                Verify nvidia-smi works and torch.cuda.is_available() returns True." >&2
    exit 5
  fi

  if [[ ! -d "$REPO_DIR" ]]; then
    echo "[p11-headtohead] ERROR: repo dir not found: $REPO_DIR" >&2
    exit 6
  fi
  cd "$REPO_DIR"

  mkdir -p "$OUT_ROOT"
  LAUNCH_LOG="$OUT_ROOT/launch.log"
  : > "$LAUNCH_LOG"
  echo "[p11-headtohead] $(date -u +%Y-%m-%dT%H:%M:%SZ) launching sweep" | tee -a "$LAUNCH_LOG"
  echo "[p11-headtohead] FRAMES=$FRAMES SEEDS='$SEEDS' BASELINES='$BASELINES'" | tee -a "$LAUNCH_LOG"
fi

# -----------------------------------------------------------------------------
# Build the (baseline, seed, gpu, logdir, command) plan
# -----------------------------------------------------------------------------
build_train_cmd() {
  # Args: $1=baseline $2=seed $3=logdir
  local baseline="$1" seed="$2" logdir="$3"
  case "$baseline" in
    tetrarl_pref_ppo)
      echo "python scripts/train_pref_ppo_dst.py --frames $FRAMES --seed $seed --device cuda --logdir $logdir"
      ;;
    pd_morl)
      echo "python scripts/train_pd_morl_dst.py --frames $FRAMES --seed $seed --device cuda --logdir $logdir"
      ;;
    duojoule)
      echo "python scripts/train_duojoule_dst.py --frames $FRAMES --seed $seed --device cuda --logdir $logdir"
      ;;
    r3)
      # r3 is a native bash dispatcher; SEED+OUT_DIR are honored via env.
      echo "SEED=$seed OUT_DIR=$logdir bash scripts/run_p11_r3_native.sh"
      ;;
    *)
      echo "__INVALID__"
      ;;
  esac
}

declare -a JOB_BASELINES JOB_SEEDS JOB_GPUS JOB_LOGDIRS JOB_CMDS
i=0
for baseline in $BASELINES; do
  for seed in $SEEDS; do
    gpu=$(( i % NUM_GPUS ))
    logdir="$OUT_ROOT/${baseline}_seed${seed}"
    cmd="$(build_train_cmd "$baseline" "$seed" "$logdir")"
    if [[ "$cmd" == "__INVALID__" ]]; then
      echo "[p11-headtohead] ERROR: unknown baseline '$baseline' (valid: tetrarl_pref_ppo pd_morl duojoule r3)" >&2
      exit 7
    fi
    JOB_BASELINES+=("$baseline")
    JOB_SEEDS+=("$seed")
    JOB_GPUS+=("$gpu")
    JOB_LOGDIRS+=("$logdir")
    JOB_CMDS+=("$cmd")
    i=$(( i + 1 ))
  done
done
NUM_JOBS=${#JOB_BASELINES[@]}

# -----------------------------------------------------------------------------
# Print plan (always)
# -----------------------------------------------------------------------------
echo
echo "[p11-headtohead] Plan: $NUM_JOBS jobs (mode: $([[ $SMOKE -eq 1 ]] && echo SMOKE || echo FULL)$([[ $DRY_RUN -eq 1 ]] && echo ', DRY-RUN'))"
printf '%-3s  %-20s  %-5s  %-4s  %-50s  %s\n' "#" "BASELINE" "SEED" "GPU" "LOGDIR" "COMMAND"
printf '%-3s  %-20s  %-5s  %-4s  %-50s  %s\n' "---" "--------------------" "-----" "----" "--------------------------------------------------" "-------"
for ((j=0; j<NUM_JOBS; j++)); do
  printf '%-3d  %-20s  %-5s  %-4d  %-50s  CUDA_VISIBLE_DEVICES=%d %s\n' \
    "$j" "${JOB_BASELINES[$j]}" "${JOB_SEEDS[$j]}" "${JOB_GPUS[$j]}" \
    "${JOB_LOGDIRS[$j]}" "${JOB_GPUS[$j]}" "${JOB_CMDS[$j]}"
done
echo

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[p11-headtohead] DRY-RUN: no tmux session created, no jobs launched."
  exit 0
fi

# -----------------------------------------------------------------------------
# Launch tmux session + one window per job
# -----------------------------------------------------------------------------
echo "[p11-headtohead] Launching tmux session '$SESSION' with $NUM_JOBS windows..." | tee -a "$LAUNCH_LOG"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[p11-headtohead] ERROR: tmux session '$SESSION' already exists." >&2
  echo "                Kill it first: tmux kill-session -t $SESSION" >&2
  exit 8
fi

for ((j=0; j<NUM_JOBS; j++)); do
  baseline="${JOB_BASELINES[$j]}"
  seed="${JOB_SEEDS[$j]}"
  gpu="${JOB_GPUS[$j]}"
  logdir="${JOB_LOGDIRS[$j]}"
  cmd="${JOB_CMDS[$j]}"
  win_name="${baseline}_seed${seed}"

  mkdir -p "$logdir"

  # The window command: cd into repo, activate env, run train with GPU pin and tee,
  # then write .done or .failed sentinel based on the train pipeline's exit status.
  # We wrap with `bash -lc` so the conda activate call (which uses 'source') works.
  win_cmd="cd $REPO_DIR && $CONDA_ACTIVATE && \
CUDA_VISIBLE_DEVICES=$gpu $cmd 2>&1 | tee $logdir/train.log; \
ec=\${PIPESTATUS[0]}; \
if [[ \$ec -eq 0 ]]; then touch $logdir/.done; else echo \"exit=\$ec\" > $logdir/.failed; fi; \
echo \"[p11-headtohead] $win_name finished with exit=\$ec\"; \
exec bash"

  if [[ "$j" -eq 0 ]]; then
    tmux new-session -d -s "$SESSION" -n "$win_name" "bash -lc \"$win_cmd\""
  else
    tmux new-window -t "$SESSION" -n "$win_name" "bash -lc \"$win_cmd\""
  fi

  echo "[p11-headtohead]   launched job $j: $win_name on GPU $gpu -> $logdir" | tee -a "$LAUNCH_LOG"
done

# -----------------------------------------------------------------------------
# Final status banner
# -----------------------------------------------------------------------------
cat <<EOF | tee -a "$LAUNCH_LOG"

[p11-headtohead] tmux session: $SESSION ($NUM_JOBS windows)

[p11-headtohead] Attach:
  tmux attach -t $SESSION
  # (Ctrl-b d to detach; Ctrl-b n / Ctrl-b p to cycle windows)

[p11-headtohead] Monitor sentinels:
  ls $OUT_ROOT/*/.done $OUT_ROOT/*/.failed 2>/dev/null | wc -l
  # should reach $NUM_JOBS when the sweep is finished

[p11-headtohead] Per-job logs:
  ls $OUT_ROOT/*/train.log
EOF
