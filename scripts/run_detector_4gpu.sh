#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
cd "$ROOT"
mkdir -p logs_v6_detector

run_one() {
  CUDA_VISIBLE_DEVICES="$1" "$PY" -u train_eval_csda_v6.py \
    --category "$2" --evaluate > "logs_v6_detector/$2.log" 2>&1
}

(run_one 0 breakfast_box; run_one 0 splicing_connectors) & p0=$!
run_one 1 juice_bottle & p1=$!
run_one 2 pushpins & p2=$!
run_one 3 screw_bag & p3=$!
status=0
for pid in "$p0" "$p1" "$p2" "$p3"; do wait "$pid" || status=1; done
exit "$status"
