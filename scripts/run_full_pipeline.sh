#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
GPUS="${GPUS:-0,1,2,3}"
cd "$ROOT"

exec "$PY" -u run_v6_full_pipeline.py \
  --gpus ${GPUS//,/ } \
  --python "$PY" \
  "$@"
