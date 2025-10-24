#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIGS=(
  "${ROOT_DIR}/config/adam_config.yaml"
  "${ROOT_DIR}/config/sgd_config.yaml"
)

echo ">>> Starting batch training runs"
for cfg in "${CONFIGS[@]}"; do
  echo "---- Running with config: ${cfg}"
  "${PYTHON_BIN}" "${ROOT_DIR}/train.py" --cfg "${cfg}" "$@"
done

echo ">>> All runs completed"
