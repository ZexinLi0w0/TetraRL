#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="/experiment/zexin/TetraRL/reproduce/runs/full_${TIMESTAMP}"

usage() {
    echo "Usage: $0 [--list] [--dry-run] [--only <experiment>]"
    echo ""
    echo "Options:"
    echo "  --list       List all available experiments"
    echo "  --dry-run    Show what would run without executing"
    echo "  --only NAME  Run only the named experiment"
    exit 0
}

EXPERIMENTS=(
    "dst-cartpole-modqn"
    "mujoco-walker2d-mosac"
    "mujoco-halfcheetah-mosac"
    "atari-pong-mosac"
    "donkeycar-mosac-orin"
    "donkeycar-mosac-xavier"
    "classic-cartpole-nano"
    "ablation-4component"
    "ffmpeg-corunner"
    "preference-switching"
)

if [[ "${1:-}" == "--list" ]]; then
    echo "Available experiments:"
    for exp in "${EXPERIMENTS[@]}"; do
        echo "  - ${exp}"
    done
    exit 0
fi

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "Dry run — would execute:"
    for exp in "${EXPERIMENTS[@]}"; do
        echo "  ${exp}"
    done
    exit 0
fi

ONLY=""
if [[ "${1:-}" == "--only" ]]; then
    ONLY="${2:-}"
    if [[ -z "$ONLY" ]]; then
        echo "Error: --only requires an experiment name"
        exit 1
    fi
fi

echo "=== TetraRL Full Paper Reproduction ==="
echo "Output directory: ${OUT_DIR}"
mkdir -p "${OUT_DIR}"

for exp in "${EXPERIMENTS[@]}"; do
    if [[ -n "$ONLY" && "$exp" != "$ONLY" ]]; then
        continue
    fi
    echo "[RUN] ${exp}..."
    echo "TODO: Implement ${exp} (see docs/action-plan-weekly.md)" | tee "${OUT_DIR}/${exp}.log"
done

echo "=== Full paper reproduction complete ==="
echo "Logs: ${OUT_DIR}"
