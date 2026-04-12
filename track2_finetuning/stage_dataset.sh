#!/usr/bin/env bash
# Download the CCMT crop-disease dataset from Kaggle using the direct API endpoint
# (bypasses the kaggle CLI's metadata call which some datasets 403 on).
#
# Credentials: reads ~/.kaggle/kaggle.json OR $KAGGLE_USERNAME + $KAGGLE_KEY.
# Make sure you've clicked "Download" once on the dataset page in the browser
# (that's what accepts the dataset's terms):
#   https://www.kaggle.com/datasets/merolavtechnology/dataset-for-crop-pest-and-disease-detection
#
# Usage:
#   DATA_ROOT=/workspace/data/ccmt bash stage_dataset.sh

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/data/ccmt}"
KAGGLE_DATASET="${KAGGLE_DATASET:-merolavtechnology/dataset-for-crop-pest-and-disease-detection}"

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
# 2) Download
# ---------------------------------------------------------------
mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

ZIP_PATH="${DATA_ROOT}/ccmt.zip"
if [ -f "$ZIP_PATH" ]; then
  echo "[stage] zip already present at $ZIP_PATH — skipping download"
else
  echo "[stage] downloading $KAGGLE_DATASET"
  # -L follow redirects; -f fail on HTTP error; -C - resume partial download;
  # --retry to survive brief hiccups.
  curl -L -f -C - --retry 5 \
    -u "${KAGGLE_USERNAME}:${KAGGLE_KEY}" \
    -o "$ZIP_PATH" \
    "https://www.kaggle.com/api/v1/datasets/download/${KAGGLE_DATASET}"
fi

ls -lh "$ZIP_PATH"

# ---------------------------------------------------------------
# 3) Unzip
# ---------------------------------------------------------------
echo "[stage] extracting (this can take a few minutes)"
unzip -q -n "$ZIP_PATH" -d "$DATA_ROOT"

# ---------------------------------------------------------------
# 4) Locate CCMT Dataset-Augmented
# ---------------------------------------------------------------
AUG_DIR=$(find "$DATA_ROOT" -maxdepth 5 -type d -name "CCMT Dataset-Augmented" | head -1)
if [ -z "$AUG_DIR" ]; then
  echo "[stage] ERROR: 'CCMT Dataset-Augmented' folder not found after extract."
  echo "        Layout under $DATA_ROOT:"
  find "$DATA_ROOT" -maxdepth 4 -type d | head -40
  exit 1
fi

IMG_COUNT=$(find "$AUG_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
echo "[stage] augmented dataset: $AUG_DIR"
echo "[stage] image count:       $IMG_COUNT  (expect ~105k)"
echo
echo "[stage] Done. Next:"
echo "  python prepare_data.py --data-root \"$AUG_DIR\" --out splits.json"
