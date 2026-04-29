#!/usr/bin/env bash
# Download the Amini cocoa contamination dataset from Kaggle using the direct
# API endpoint (bypasses the kaggle CLI's metadata call which 403s on some
# datasets).
#
# Dataset:
#   ohagwucollinspatrick/amini-cocoa-contamination-dataset
#   Provides Train.csv with bounding boxes
#   (columns: Image_ID, ImagePath, class, xmin, ymin, xmax, ymax) and the
#   referenced source images.
#
# Credentials: reads ~/.kaggle/kaggle.json OR $KAGGLE_USERNAME + $KAGGLE_KEY.
# Click "Download" once on the dataset page in the browser to accept terms.
#
# Usage:
#   DATA_ROOT=/workspace/data/amini bash stage_amini_dataset.sh

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/data/amini}"
AMINI_DATASET="${AMINI_DATASET:-ohagwucollinspatrick/amini-cocoa-contamination-dataset}"

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
# 2) Download + unzip
# ---------------------------------------------------------------
mkdir -p "$DATA_ROOT"
ZIP_PATH="${DATA_ROOT}.zip"

if [ -f "$ZIP_PATH" ]; then
  echo "[stage] zip already present at $ZIP_PATH — skipping download"
else
  echo "[stage] downloading $AMINI_DATASET -> $ZIP_PATH"
  curl -L -f -C - --retry 5 \
    -u "${KAGGLE_USERNAME}:${KAGGLE_KEY}" \
    -o "$ZIP_PATH" \
    "https://www.kaggle.com/api/v1/datasets/download/${AMINI_DATASET}"
fi

ls -lh "$ZIP_PATH"

echo "[stage] extracting $ZIP_PATH -> $DATA_ROOT"
unzip -q -n "$ZIP_PATH" -d "$DATA_ROOT"

# ---------------------------------------------------------------
# 3) Locate Train.csv (zip layout sometimes has a top-level dir)
# ---------------------------------------------------------------
CSV_PATH=$(find "$DATA_ROOT" -maxdepth 4 -type f -iname "Train.csv" | head -1)

if [ -z "$CSV_PATH" ]; then
  echo "[stage] ERROR: Train.csv not found under $DATA_ROOT"
  find "$DATA_ROOT" -maxdepth 3 -type d | head -20
  exit 1
fi

DATASET_ROOT=$(dirname "$CSV_PATH")
IMG_COUNT=$(find "$DATASET_ROOT" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)

echo
echo "[stage] dataset root : $DATASET_ROOT"
echo "[stage] Train.csv    : $CSV_PATH"
echo "[stage] images       : $IMG_COUNT"
echo
echo "[stage] Done. Next:"
echo "  python prepare_amini_data.py --dataset-root \"$DATASET_ROOT\" --out splits.json"
