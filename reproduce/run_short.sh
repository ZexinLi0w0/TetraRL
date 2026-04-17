#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="/experiment/zexin/TetraRL/reproduce/runs/short_${TIMESTAMP}"

echo "=== TetraRL Short Validation ==="
echo "Output directory: ${OUT_DIR}"
mkdir -p "${OUT_DIR}"

# Step 1: Pong + R⁴ tracking (placeholder)
echo "[1/1] Pong + R⁴ metric tracking..."
echo "TODO: Run MO-SAC-HER on Pong for 250k frames with R⁴ logging (Week 2)" | tee "${OUT_DIR}/pong_r4.log"

echo "=== Short validation complete ==="
echo "Logs: ${OUT_DIR}"
