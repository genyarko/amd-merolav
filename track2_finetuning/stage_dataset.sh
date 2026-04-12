#!/usr/bin/env bash
# Download and extract the Mendeley CCMT dataset on the MI300X droplet.
#
# The Mendeley presigned URL expires in ~5 minutes, so pass it fresh each time:
#
#   DATA_URL='https://prod-dcd-datasets-cache-zipfiles.s3...' \
#   DATA_ROOT=/data/ccmt \
#   bash stage_dataset.sh
#
# Get a fresh URL from: https://data.mendeley.com/datasets/bwh3zbpkpv/1
# (right-click the download button → copy link).
#
# After extraction, $DATA_ROOT/CCMT Dataset-Augmented/ should contain
# Cashew/ Cassava/ Maize/ Tomato/.

set -euo pipefail

: "${DATA_URL:?Set DATA_URL to the Mendeley presigned S3 URL (fresh, 5min expiry)}"
DATA_ROOT="${DATA_ROOT:-/data/ccmt}"

echo "[stage] target: $DATA_ROOT"
mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

ZIP_PATH="${DATA_ROOT}/ccmt.zip"

if [ -f "$ZIP_PATH" ]; then
  echo "[stage] zip already present at $ZIP_PATH — skipping download"
else
  echo "[stage] downloading zip (~several GB, this takes a few minutes)"
  # -L follow redirects, -f fail on HTTP errors, -C - resume if partial
  curl -L -f -C - -o "$ZIP_PATH" "$DATA_URL"
fi

echo "[stage] size on disk:"
ls -lh "$ZIP_PATH"

echo "[stage] extracting"
# -n = never overwrite, so re-runs are cheap
unzip -q -n "$ZIP_PATH" -d "$DATA_ROOT"

echo "[stage] inspecting layout"
find "$DATA_ROOT" -maxdepth 3 -type d | head -40

# Sanity check: locate "CCMT Dataset-Augmented"
AUG_DIR=$(find "$DATA_ROOT" -maxdepth 4 -type d -name "CCMT Dataset-Augmented" | head -1)
if [ -z "$AUG_DIR" ]; then
  echo "[stage] ERROR: could not find 'CCMT Dataset-Augmented' after extract"
  echo "        Inspect $DATA_ROOT manually and set --data-root accordingly."
  exit 1
fi

echo "[stage] augmented dataset: $AUG_DIR"
echo "[stage] class folders:"
ls "$AUG_DIR" || true

IMG_COUNT=$(find "$AUG_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
echo "[stage] total images under augmented dir: $IMG_COUNT"

echo
echo "[stage] Done. Use this path with prepare_data.py:"
echo "  python prepare_data.py --data-root \"$AUG_DIR\" --out splits.json"
