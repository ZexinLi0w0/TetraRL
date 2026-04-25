#!/usr/bin/env bash
# =============================================================================
# scripts/run_p11_r3_native.sh
#
# P11 head-to-head — R3 native dispatcher.
#
# R3 (Li, Liu et al., RTSS 2023) is a deadline-aware on-device DRL training
# framework. It does NOT target a Pareto front — it dynamically adapts batch
# size and replay-buffer size to honour a per-episode timing deadline.
#
# Per the spec hard rule ("If R3 targets a different metric, report on its
# native metric -- don't force HV onto it"), this script:
#
#   1. Trains R3 (DQN preset, CartPole-v1 -- the R3-unit-tested combination)
#      for 50k frames per seed via the autonomous-learning-library `all-classic`
#      console script that R3 ships.
#   2. Captures stdout (episode lines, runtime-coordinator alpha/beta, fps)
#      and the tensorboard event files written by ALL's ExperimentLogger.
#   3. Invokes scripts/parse_r3_metrics.py to emit r3_native_metrics.json
#      with `framework_overhead_pct` and `mean_deadline_miss_rate`.
#
# Usage (server1, inside the tetrarl-baselines conda env):
#
#   SEED=0 OUT_DIR=runs/p11_dst_headtohead/r3_seed0 \
#       bash scripts/run_p11_r3_native.sh
#
#   # or via positional args:
#   bash scripts/run_p11_r3_native.sh 0 runs/p11_dst_headtohead/r3_seed0
#
# This script is invoked directly by scripts/run_p11_headtohead.sh as:
#
#   SEED=$seed OUT_DIR=$logdir bash scripts/run_p11_r3_native.sh
#
# Output layout (relative to OUT_DIR):
#
#   $OUT_DIR/
#       runs/                       <- ALL tensorboard subdir (one event file)
#       r3_train.log                <- captured stdout/stderr from training
#       r3_native_metrics.json      <- aggregator-readable metrics
#       .done | .failed             <- sentinel for the orchestrator
#
# Exit policy: -uo pipefail (NOT -e). Failures surface as a non-zero exit and
# a `.failed` sentinel; we never abort silently mid-pipeline.
# =============================================================================

set -uo pipefail

# -----------------------------------------------------------------------------
# Parse args / env
# -----------------------------------------------------------------------------
SEED="${1:-${SEED:-0}}"
OUT_DIR="${2:-${OUT_DIR:-runs/p11_dst_headtohead/r3_seed${SEED}}}"

# Knobs (override via env)
FRAMES="${FRAMES:-50000}"
ENV_NAME="${ENV:-CartPole-v1}"
AGENT_NAME="${AGENT:-dqn}"
DEVICE="${DEVICE:-cuda}"

# Where R3 lives on server1. Allow override for portability / testing.
R3_REPO="${R3_REPO:-/data/zli/TetraRL-headtohead/R3}"
R3_ALL="${R3_ALL:-${R3_REPO}/autonomous-learning-library}"

# -----------------------------------------------------------------------------
# Locate this script + the parser script (sibling) regardless of cwd
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARSER="${SCRIPT_DIR}/parse_r3_metrics.py"

# Make OUT_DIR absolute so subprocess cwd changes don't break it.
mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"
RUNS_DIR="${OUT_DIR}/runs"
LOG_FILE="${OUT_DIR}/r3_train.log"
METRICS_JSON="${OUT_DIR}/r3_native_metrics.json"
DONE_SENTINEL="${OUT_DIR}/.done"
FAIL_SENTINEL="${OUT_DIR}/.failed"

mkdir -p "${RUNS_DIR}"
rm -f "${DONE_SENTINEL}" "${FAIL_SENTINEL}"

# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------
log() {
  echo "[r3-native] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"
}

fail() {
  local code="$1"; shift
  log "FAIL ($code): $*"
  echo "$*" > "${FAIL_SENTINEL}"
  exit "$code"
}

log "seed=${SEED} env=${ENV_NAME} agent=${AGENT_NAME} device=${DEVICE} frames=${FRAMES}"
log "out_dir=${OUT_DIR}"
log "r3_all=${R3_ALL}"
log "log_file=${LOG_FILE}"

# -----------------------------------------------------------------------------
# Pre-flight: R3 source tree present?
# -----------------------------------------------------------------------------
if [[ ! -d "${R3_ALL}" ]]; then
  fail 10 "R3 autonomous-learning-library not found at ${R3_ALL} (set R3_ALL=...)"
fi

# -----------------------------------------------------------------------------
# Pre-flight: ensure the `all` package is importable. We share the
# tetrarl-baselines conda env (the orchestrator already activated it), but R3
# ships its own modernized fork of autonomous-learning-library. If it's not
# already installed we pip-install it editable.
# -----------------------------------------------------------------------------
log "checking that the R3 'all' package is importable..."
if ! python -c "import all; import all.r3" >/dev/null 2>&1; then
  log "'all' or 'all.r3' not importable -- installing R3's fork editable"
  if ! pip install -e "${R3_ALL}" 2>&1 | tee -a "${LOG_FILE}"; then
    fail 11 "pip install -e ${R3_ALL} failed -- see ${LOG_FILE}"
  fi
  if ! python -c "import all; import all.r3" >/dev/null 2>&1; then
    fail 12 "R3 'all' package still not importable after pip install -e -- see ${LOG_FILE}"
  fi
fi

# -----------------------------------------------------------------------------
# Pre-flight: 'all-classic' console script on PATH?
# -----------------------------------------------------------------------------
if ! command -v all-classic >/dev/null 2>&1; then
  log "WARN: 'all-classic' not on PATH; falling back to 'python -m all.scripts.train_classic'"
  ALL_CLASSIC=("python" "-m" "all.scripts.train_classic")
else
  ALL_CLASSIC=("all-classic")
fi

# -----------------------------------------------------------------------------
# Seed: ALL's CLI does not expose a --seed flag, so we set it via env vars
# that PyTorch / NumPy / Python pick up. The R3 fork has the same CLI shape.
# -----------------------------------------------------------------------------
export PYTHONHASHSEED="${SEED}"
# Many ALL preset implementations honour these conventions.
export TORCH_SEED="${SEED}"
export NUMPY_SEED="${SEED}"
export RANDOM_SEED="${SEED}"

# -----------------------------------------------------------------------------
# Run training. We tee stdout/stderr to the log so the parser can read both
# the episode lines ("episode: N, frame: F, fps: X, episode_length: L,
# returns: R") and the runtime-coordinator alpha/beta lines.
# -----------------------------------------------------------------------------
log "launching training..."
WALL_START="$(date +%s)"

(
  cd "${R3_ALL}" || exit 99
  "${ALL_CLASSIC[@]}" "${ENV_NAME}" "${AGENT_NAME}" \
      --device "${DEVICE}" \
      --frames "${FRAMES}" \
      --logdir "${RUNS_DIR}"
) 2>&1 | tee -a "${LOG_FILE}"
TRAIN_RC="${PIPESTATUS[0]}"
WALL_END="$(date +%s)"
WALL_S="$(( WALL_END - WALL_START ))"

if [[ "${TRAIN_RC}" -ne 0 ]]; then
  fail 20 "training exited with code ${TRAIN_RC} after ${WALL_S}s -- see ${LOG_FILE}"
fi
log "training finished OK (wall=${WALL_S}s)"

# -----------------------------------------------------------------------------
# Parse the run into r3_native_metrics.json
# -----------------------------------------------------------------------------
if [[ ! -f "${PARSER}" ]]; then
  fail 30 "parser script not found at ${PARSER}"
fi

log "parsing metrics -> ${METRICS_JSON}"
if ! python "${PARSER}" \
      --runs-dir "${RUNS_DIR}" \
      --log-file "${LOG_FILE}" \
      --out "${METRICS_JSON}" \
      --env "${ENV_NAME}" \
      --agent "${AGENT_NAME}" \
      --frames "${FRAMES}" \
      --wall-clock-s "${WALL_S}" 2>&1 | tee -a "${LOG_FILE}"; then
  fail 40 "parser exited non-zero -- see ${LOG_FILE}"
fi

if [[ ! -f "${METRICS_JSON}" ]]; then
  fail 41 "parser claimed success but ${METRICS_JSON} missing"
fi

touch "${DONE_SENTINEL}"
log "DONE -- metrics at ${METRICS_JSON}"
exit 0
