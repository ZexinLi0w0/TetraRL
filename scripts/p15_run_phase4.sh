#!/usr/bin/env bash
# P15 Phase 4 — Atari Breakout × Orin Nano, 25 cells (21 active + 4 SKIPPED) × 1 seed × 50k frames.
# Runs serial (PARALLEL=1) — Atari is memory-heavy on the 8 GB Nano per Phase 0 probe.
# Resumable (skips cells whose summary.json already exists).
#
# Differences vs Phase 2 (CartPole × Orin Nano):
#   - PARALLEL=1 (vs 2): Atari replay buffer ~2.7 GB per worker leaves no room for a second.
#   - SEEDS=(0): single-seed reduced budget per Phase 0 8 GB memory probe.
#   - FRAMES=50000: same frame budget as Phase 2 (matches Phase 0 §1 budget recommendation).
#   - OUT_ROOT=runs/p15_phase4_orin_nano_atari.
#   - VENV_ROOT=/experiment/zexin/venvs/tetrarl-nano (NOT r3 — that's the orin1 venv).
#   - Cell name prefix = atari__ (instead of cartpole__) per Phase 4 naming convention.
#   - --env breakout (the runner's actual valid choice; "atari" is the domain tag).

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/experiment/zexin/TetraRL}"
VENV_ROOT="${VENV_ROOT:-/experiment/zexin/venvs/tetrarl-nano}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/runs/p15_phase4_orin_nano_atari}"
PARALLEL="${PARALLEL:-1}"
FRAMES="${FRAMES:-50000}"
PLATFORM="${PLATFORM:-orin_nano}"
SEEDS=(0)
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
        local cell="atari__${PLATFORM}__${algo}__${wrapper}__seed${seed}"
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
    --env breakout \
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
echo "P15 Phase 4: ${n_jobs} pending cells, parallel=${PARALLEL}, frames=${FRAMES}, platform=${PLATFORM}"
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
