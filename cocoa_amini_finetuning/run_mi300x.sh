#!/usr/bin/env bash
# End-to-end Amini cocoa contamination 3-class run on the MI300X droplet.
#
# Assumes:
#   - setup_rocm.sh has already been executed (venv + ROCm torch + deps)
#   - kaggle.json is in ~/.kaggle/  (or KAGGLE_USERNAME + KAGGLE_KEY are exported)
#
# Pipeline:
#   1) stage   — pull the Amini Kaggle dataset (~1-2 min)
#   2) prepare — filter boxes + image-level split → splits.json (no pre-cropping;
#                random pad happens at train time)
#   3) train   — fine-tune the chosen backbone (DINOv2-L @ 224 by default)
#   4) eval    — TTA × 10 on the held-out test split
#
# Switch backbones with CONFIG=config_eva02.yaml bash run_mi300x.sh

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/data/amini}"
CONFIG="${CONFIG:-config_dinov2.yaml}"
RUN_NAME="${RUN_NAME:-amini_$(basename "${CONFIG%.yaml}" | sed 's/config_//')_$(date +%Y%m%d_%H%M)}"
OUTPUT="runs/${RUN_NAME}"

# shellcheck disable=SC1091
source "${VENV_DIR:-$HOME/venv}/bin/activate"

# ---------------------------------------------------------------
# 1) Stage Kaggle dataset (idempotent — skips download if zip present)
# ---------------------------------------------------------------
echo "[run] staging amini dataset to $DATA_ROOT"
DATA_ROOT="$DATA_ROOT" bash stage_amini_dataset.sh

CSV_PATH=$(find "$DATA_ROOT" -maxdepth 4 -type f -iname "Train.csv" | head -1)
if [ -z "$CSV_PATH" ]; then
  echo "[run] ERROR: Train.csv not found after staging. Looked under $DATA_ROOT"
  exit 1
fi
DATASET_ROOT=$(dirname "$CSV_PATH")

# ---------------------------------------------------------------
# 2) Build splits  (deterministic; no pre-cropping)
# ---------------------------------------------------------------
echo "[run] filtering boxes and building splits.json"
python prepare_amini_data.py \
    --dataset-root "$DATASET_ROOT" \
    --out          splits.json

# ---------------------------------------------------------------
# 3) Train
# ---------------------------------------------------------------
echo "[run] training -> $OUTPUT  (config=$CONFIG)"
python train.py --config "$CONFIG" --splits splits.json --output "$OUTPUT"

# ---------------------------------------------------------------
# 4) Evaluate with TTA
# ---------------------------------------------------------------
echo "[run] evaluation"
python eval.py --checkpoint "$OUTPUT/best.pt" --splits splits.json --tta 10 \
    --output "$OUTPUT"

echo "[run] done. Results:"
ls -la "$OUTPUT"
