#!/usr/bin/env bash
# End-to-end LIT data preparation.
# Run from the lit/ directory: bash scripts/prepare_data.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=== LIT Data Pipeline ==="
echo "Project: $PROJECT_DIR"

# 1. Install dependencies
echo "[setup] Installing requirements..."
pip install -r requirements.txt --quiet

# 2. Run the pipeline
echo "[pipeline] Starting..."
python -m src.data.pipeline --config configs/data_config.yaml

echo "[done] Data preparation complete."
