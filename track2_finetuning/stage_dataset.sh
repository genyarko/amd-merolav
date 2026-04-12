#!/usr/bin/env bash
# Download the CCMT crop-disease dataset from Kaggle on the MI300X droplet.
#
# Requires a Kaggle API token at ~/.kaggle/kaggle.json (chmod 600).
# Get one at: https://www.kaggle.com/settings → "Create New Token".
#
# One-liner to set it up if you have the token content:
#   mkdir -p ~/.kaggle && nano ~/.kaggle/kaggle.json   # paste, save
#   chmod 600 ~/.kaggle/kaggle.json
#
# Usage:
#   DATA_ROOT=/workspace/data/ccmt bash stage_dataset.sh
#
# Override the dataset slug if you want a different copy:
#   KAGGLE_DATASET=merolavtechnology/dataset-for-crop-pest-and-disease-detection

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/data/ccmt}"
KAGGLE_DATASET="${KAGGLE_DATASET:-merolavtechnology/dataset-for-crop-pest-and-disease-detection}"

# ---------------------------------------------------------------
# 1) Credential check
# ---------------------------------------------------------------
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "[stage] ERROR: ~/.kaggle/kaggle.json not found."
  echo "        1) Go to https://www.kaggle.com/settings"
  echo "        2) Click 'Create New Token' — downloads kaggle.json"
  echo "        3) On the droplet:"
  echo "             mkdir -p ~/.kaggle"
  echo "             nano ~/.kaggle/kaggle.json    # paste JSON, save"
  echo "             chmod 600 ~/.kaggle/kaggle.json"
  exit 1
fi
chmod 600 "$HOME/.kaggle/kaggle.json"

# ---------------------------------------------------------------
# 2) Install kaggle CLI (inside container / venv)
# ---------------------------------------------------------------
if ! command -v kaggle >/dev/null 2>&1; then
  echo "[stage] installing kaggle CLI"
  pip install --quiet kaggle || pip install --quiet --break-system-packages kaggle
fi

# ---------------------------------------------------------------
# 3) Download + unzip
# ---------------------------------------------------------------
mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

echo "[stage] kaggle datasets download -d $KAGGLE_DATASET"
kaggle datasets download -d "$KAGGLE_DATASET" --unzip

echo "[stage] layout:"
find "$DATA_ROOT" -maxdepth 3 -type d | head -40

# ---------------------------------------------------------------
# 4) Locate CCMT Dataset-Augmented
# ---------------------------------------------------------------
AUG_DIR=$(find "$DATA_ROOT" -maxdepth 5 -type d -name "CCMT Dataset-Augmented" | head -1)
if [ -z "$AUG_DIR" ]; then
  echo "[stage] ERROR: 'CCMT Dataset-Augmented' folder not found after extract."
  echo "        Inspect $DATA_ROOT manually."
  exit 1
fi

IMG_COUNT=$(find "$AUG_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
echo "[stage] augmented dataset: $AUG_DIR"
echo "[stage] image count:       $IMG_COUNT  (expect ~105k)"
echo
echo "[stage] Done. Next:"
echo "  python prepare_data.py --data-root \"$AUG_DIR\" --out splits.json"
