#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${PROJECT_ROOT}/reproduce/runs/smoke_${TIMESTAMP}"

echo "=== TetraRL Smoke Test ==="
echo "Output directory: ${OUT_DIR}"
mkdir -p "${OUT_DIR}"

# Step 1: Run unit tests
echo "[1/3] Running unit tests..."
cd "${PROJECT_ROOT}"
pytest tests/ -v --tb=short 2>&1 | tee "${OUT_DIR}/pytest.log"

# Step 2: Verify imports
echo "[2/3] Verifying imports..."
python -c "
import tetrarl
from tetrarl.morl.agents import pd_morl
from tetrarl.morl import operators, preference_sampling
from tetrarl.envs.dst import DeepSeaTreasure
from tetrarl.eval import hypervolume, tail_latency
from tetrarl.sys import tegrastats_daemon, dvfs_controller, override_layer
print('All imports OK')
" 2>&1 | tee "${OUT_DIR}/imports.log"

# Step 3: PD-MORL DST short training (1000 frames)
echo "[3/3] PD-MORL DST sanity check (1000 frames)..."
python scripts/train_pd_morl_dst.py \
    --frames 1000 \
    --seed 0 \
    --logdir "${OUT_DIR}/pd_morl_dst" \
    2>&1 | tee "${OUT_DIR}/train.log"

echo "=== Smoke test complete ==="
echo "Logs: ${OUT_DIR}"
