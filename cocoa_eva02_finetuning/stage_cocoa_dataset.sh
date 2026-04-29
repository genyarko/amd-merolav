#!/usr/bin/env bash
# Download the two YOLO cocoa-disease datasets from Kaggle using the direct
# API endpoint (bypasses the kaggle CLI's metadata call which 403s on some
# datasets).
#
# Datasets:
#   LatAm — serranosebas/enfermedades-cacao-yolov4
#           Folder layout: Enfermedades Cacao/{Fitoftora,Monilia,Sana}/{*.jpg,*.txt}
#   Peru  — bryandarquea/cocoa-diseases
#           Folder layout: cocoa_diseases/images/*.jpg + cocoa_diseases/labels/*.txt
#
# Credentials: reads ~/.kaggle/kaggle.json OR $KAGGLE_USERNAME + $KAGGLE_KEY.
# Click "Download" once on each dataset page in the browser to accept terms.
#
# Usage:
#   DATA_ROOT=/workspace/data/cocoa bash stage_cocoa_dataset.sh

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/data/cocoa}"
LATAM_DATASET="${LATAM_DATASET:-serranosebas/enfermedades-cacao-yolov4}"
PERU_DATASET="${PERU_DATASET:-bryandarquea/cocoa-diseases}"

# ---------------------------------------------------------------
# 1) Load credentials from kaggle.json if env vars aren't already set
# ---------------------------------------------------------------
if [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; then
  if [ -f "$HOME/.kaggle/kaggle.json" ]; then
    KAGGLE_USERNAME=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
    KAGGLE_KEY=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['key'])")
  else
    echo "[stage] ERROR: set KAGGLE_USERNAME + KAGGLE_KEY env vars, OR create ~/.kaggle/kaggle.json"
    exit 1
  fi
fi
echo "[stage] authenticated as: $KAGGLE_USERNAME"

command -v unzip >/dev/null || { echo "[stage] need 'unzip' — apt install -y unzip"; exit 1; }
command -v curl  >/dev/null || { echo "[stage] need 'curl' — apt install -y curl";  exit 1; }

# ---------------------------------------------------------------
# 2) Helper: download + unzip one dataset into a sub-folder
# ---------------------------------------------------------------
fetch_dataset() {
  local slug="$1"
  local subdir="$2"
  local target="${DATA_ROOT}/${subdir}"
  local zip_path="${target}.zip"

  mkdir -p "$target"

  if [ -f "$zip_path" ]; then
    echo "[stage] zip already present at $zip_path — skipping download"
  else
    echo "[stage] downloading $slug -> $zip_path"
    curl -L -f -C - --retry 5 \
      -u "${KAGGLE_USERNAME}:${KAGGLE_KEY}" \
      -o "$zip_path" \
      "https://www.kaggle.com/api/v1/datasets/download/${slug}"
  fi

  ls -lh "$zip_path"

  echo "[stage] extracting $zip_path -> $target"
  unzip -q -n "$zip_path" -d "$target"
}

# ---------------------------------------------------------------
# 3) Pull both datasets
# ---------------------------------------------------------------
mkdir -p "$DATA_ROOT"

fetch_dataset "$LATAM_DATASET" "latam"
fetch_dataset "$PERU_DATASET"  "peru"

# ---------------------------------------------------------------
# 4) Locate dataset roots and report counts
# ---------------------------------------------------------------
LATAM_ROOT=$(find "${DATA_ROOT}/latam" -maxdepth 4 -type d -iname "Enfermedades Cacao" | head -1)
PERU_IMAGES=$(find "${DATA_ROOT}/peru"  -maxdepth 5 -type d -iname "images"            | head -1)
PERU_LABELS=$(find "${DATA_ROOT}/peru"  -maxdepth 5 -type d -iname "labels"            | head -1)

if [ -z "$LATAM_ROOT" ]; then
  echo "[stage] ERROR: 'Enfermedades Cacao' folder not found under ${DATA_ROOT}/latam"
  find "${DATA_ROOT}/latam" -maxdepth 3 -type d | head -20
  exit 1
fi
if [ -z "$PERU_IMAGES" ] || [ -z "$PERU_LABELS" ]; then
  echo "[stage] ERROR: peru images/labels folders not found under ${DATA_ROOT}/peru"
  find "${DATA_ROOT}/peru" -maxdepth 4 -type d | head -20
  exit 1
fi

LATAM_IMG=$(find "$LATAM_ROOT" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
PERU_IMG=$(find  "$PERU_IMAGES" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)

echo
echo "[stage] LatAm root : $LATAM_ROOT  ($LATAM_IMG images)"
echo "[stage] Peru imgs  : $PERU_IMAGES ($PERU_IMG images)"
echo "[stage] Peru lbls  : $PERU_LABELS"
echo
echo "[stage] Done. Next:"
echo "  python prepare_cocoa_data.py \\"
echo "    --latam-root \"$LATAM_ROOT\" \\"
echo "    --peru-images \"$PERU_IMAGES\" \\"
echo "    --peru-labels \"$PERU_LABELS\" \\"
echo "    --crop-dir \"${DATA_ROOT}/crops\" \\"
echo "    --out splits.json"
