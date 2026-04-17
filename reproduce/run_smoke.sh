#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="/experiment/zexin/TetraRL/reproduce/runs/smoke_${TIMESTAMP}"

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
from tetrarl.morl import pd_morl, operators
from tetrarl.sys import tegrastats_daemon, dvfs_controller, override_layer
from tetrarl.eval import hypervolume, tail_latency
print('All imports OK')
" 2>&1 | tee "${OUT_DIR}/imports.log"

# Step 3: CartPole DST sanity (placeholder)
echo "[3/3] CartPole DST sanity check..."
echo "TODO: Run MO-DQN-HER on CartPole DST for 20k steps (Week 1)" | tee "${OUT_DIR}/cartpole.log"

echo "=== Smoke test complete ==="
echo "Logs: ${OUT_DIR}"
