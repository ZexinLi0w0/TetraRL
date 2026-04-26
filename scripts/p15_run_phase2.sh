#!/usr/bin/env bash
# P15 Phase 2 — CartPole × Orin Nano, 75 cells (63 COMPLETED + 12 SKIPPED) × 50k frames.
# Runs N-way parallel with xargs; resumable (skips cells whose summary.json already exists).
#
# Differences vs Phase 1 (Orin AGX):
#   - PARALLEL defaults to 2 (Orin Nano has 6 cores + 8 GB unified RAM, vs AGX 12 cores + 32 GB).
#   - OUT_ROOT points to runs/p15_phase2_orin_nano_cartpole.
#   - PLATFORM tag is orin_nano (used in cell dir name and summary.json hw field).

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/experiment/zexin/TetraRL}"
VENV_ROOT="${VENV_ROOT:-/experiment/zexin/venvs/r3}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/runs/p15_phase2_orin_nano_cartpole}"
PARALLEL="${PARALLEL:-2}"
FRAMES="${FRAMES:-50000}"
PLATFORM="${PLATFORM:-orin_nano}"
SEEDS=(0 1 2)
ALGOS=(dqn ddqn c51 a2c ppo)
WRAPPERS=(maxa maxp r3 duojoule tetrarl)
LOG_DIR="${OUT_ROOT}/_logs"

is_compatible() {
  local algo="$1"
  local wrapper="$2"
  if [[ "${wrapper}" == "r3" || "${wrapper}" == "duojoule" ]]; then
    if [[ "${algo}" == "a2c" || "${algo}" == "ppo" ]]; then
      return 1
    fi
  fi
  return 0
}

build_jobs() {
  for algo in "${ALGOS[@]}"; do
    for wrapper in "${WRAPPERS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        local cell="cartpole__${PLATFORM}__${algo}__${wrapper}__seed${seed}"
        local out_dir="${OUT_ROOT}/${cell}"
        local sm="${out_dir}/summary.json"
        if [[ -f "${sm}" ]]; then
          # Already done (resumable).
          continue
        fi
        echo "${algo}|${wrapper}|${seed}|${out_dir}|${cell}"
      done
    done
  done
}

run_one() {
  IFS='|' read -r algo wrapper seed out_dir cell <<<"$1"
  mkdir -p "${out_dir}"
  local log="${LOG_DIR}/${cell}.log"
  echo "[$(date +%H:%M:%S)] START ${cell}" >> "${LOG_DIR}/_progress.log"
  "${VENV_PY}" "${REPO_ROOT}/scripts/p15_unified_runner.py" \
    --algo "${algo}" \
    --wrapper "${wrapper}" \
    --env cartpole \
    --platform "${PLATFORM}" \
    --seed "${seed}" \
    --frames "${FRAMES}" \
    --out-dir "${out_dir}" \
    > "${log}" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] END   ${cell} rc=${rc}" >> "${LOG_DIR}/_progress.log"
  return ${rc}
}

# Guard: only execute the launch body when this script is run directly,
# not when it is sourced (so build_jobs / run_one can be tested in isolation).
# shellcheck disable=SC2128
if [[ "${BASH_SOURCE}" != "$0" ]]; then
  return 0 2>/dev/null || true
fi

# --- launch body --------------------------------------------------------------

VENV_PY="${VENV_ROOT}/bin/python"
if [[ ! -x "${VENV_PY}" ]]; then
  echo "ERROR: venv python not found at ${VENV_PY}" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

cd "${REPO_ROOT}"

export -f run_one
export REPO_ROOT VENV_PY LOG_DIR FRAMES PLATFORM

n_jobs=$(build_jobs | wc -l | tr -d ' ')
echo "P15 Phase 2: ${n_jobs} pending cells, parallel=${PARALLEL}, frames=${FRAMES}, platform=${PLATFORM}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "VENV_PY=${VENV_PY}"
echo "OUT_ROOT=${OUT_ROOT}"

if [[ "${n_jobs}" == "0" ]]; then
  echo "Nothing to do."
  exit 0
fi

t_start=$(date +%s)
build_jobs | xargs -I {} -n 1 -P "${PARALLEL}" bash -c 'run_one "$@"' _ {}
xargs_rc=$?
t_end=$(date +%s)
echo "Elapsed: $((t_end - t_start))s, xargs rc=${xargs_rc}"

n_done=$(find "${OUT_ROOT}" -mindepth 2 -name summary.json | wc -l | tr -d ' ')
echo "Cells with summary.json under OUT_ROOT: ${n_done}"
exit ${xargs_rc}
