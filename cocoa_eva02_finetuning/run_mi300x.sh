#!/usr/bin/env bash
# End-to-end cocoa EVA-02-L run on the MI300X droplet.
#
# Assumes:
#   - setup_rocm.sh has already been executed (venv + ROCm torch + deps present)
#   - kaggle.json is in ~/.kaggle/  (or KAGGLE_USERNAME + KAGGLE_KEY are exported)
#
# Pipeline:
#   1) stage   — pull both Kaggle YOLO datasets  (~5 min, network-bound)
#   2) prepare — crop YOLO boxes + image-level split → splits.json
#   3) train   — EVA-02-L fine-tune (~?? hours on MI300X — bench first run)
#   4) eval    — TTA × 10 evaluation on the held-out test split

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/data/cocoa}"
RUN_NAME="${RUN_NAME:-eva02_l_$(date +%Y%m%d_%H%M)}"
OUTPUT="runs/${RUN_NAME}"

# shellcheck disable=SC1091
source "${VENV_DIR:-$HOME/venv}/bin/activate"

# ---------------------------------------------------------------
# 1) Stage Kaggle datasets (idempotent — skips download if zips present)
# ---------------------------------------------------------------
echo "[run] staging cocoa datasets to $DATA_ROOT"
DATA_ROOT="$DATA_ROOT" bash stage_cocoa_dataset.sh

# Locate the canonical sub-paths that stage_cocoa_dataset.sh extracted.
LATAM_ROOT=$(find "${DATA_ROOT}/latam" -maxdepth 4 -type d -iname "Enfermedades Cacao" | head -1)
PERU_IMAGES=$(find "${DATA_ROOT}/peru"  -maxdepth 5 -type d -iname "images"            | head -1)
PERU_LABELS=$(find "${DATA_ROOT}/peru"  -maxdepth 5 -type d -iname "labels"            | head -1)

if [ -z "$LATAM_ROOT" ] || [ -z "$PERU_IMAGES" ] || [ -z "$PERU_LABELS" ]; then
  echo "[run] ERROR: could not locate dataset roots after staging."
  echo "  LATAM_ROOT = $LATAM_ROOT"
  echo "  PERU_IMAGES= $PERU_IMAGES"
  echo "  PERU_LABELS= $PERU_LABELS"
  exit 1
fi

# ---------------------------------------------------------------
# 2) Crop + split  (deterministic; rebuilds crops dir on every run)
# ---------------------------------------------------------------
echo "[run] cropping YOLO boxes and building splits.json"
python prepare_cocoa_data.py \
    --latam-root  "$LATAM_ROOT" \
    --peru-images "$PERU_IMAGES" \
    --peru-labels "$PERU_LABELS" \
    --crop-dir    "${DATA_ROOT}/crops" \
    --out         splits.json

# ---------------------------------------------------------------
# 3) Train
# ---------------------------------------------------------------
echo "[run] training -> $OUTPUT"
python train.py --config config.yaml --splits splits.json --output "$OUTPUT"

# ---------------------------------------------------------------
# 4) Evaluate with TTA
# ---------------------------------------------------------------
echo "[run] evaluation"
python eval.py --checkpoint "$OUTPUT/best.pt" --splits splits.json --tta 10 \
    --output "$OUTPUT"

echo "[run] done. Results:"
ls -la "$OUTPUT"
