#!/usr/bin/env bash
# End-to-end Track 2 run on the MI300X droplet.
#
# Assumes:
#   - setup_rocm.sh has already been executed (venv + torch-rocm + deps present)
#   - CCMT dataset lives at $DATA_ROOT

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/ccmt/CCMT Dataset-Augmented}"
RUN_NAME="${RUN_NAME:-dinov2_l_$(date +%Y%m%d_%H%M)}"
OUTPUT="runs/${RUN_NAME}"

# shellcheck disable=SC1091
source venv/bin/activate

echo "[run] building splits"
python prepare_data.py --data-root "$DATA_ROOT" --out splits.json

echo "[run] throughput benchmark (quick)"
python benchmark.py --config config.yaml --batch-sizes 32,64,128 --steps 10 \
    --output "${OUTPUT%.}_benchmark.json" || true

echo "[run] training -> $OUTPUT"
python train.py --config config.yaml --splits splits.json --output "$OUTPUT"

echo "[run] evaluation"
python eval.py --checkpoint "$OUTPUT/best.pt" --splits splits.json --tta 10 \
    --output "$OUTPUT"

echo "[run] done. Results:"
ls -la "$OUTPUT"
