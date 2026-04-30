#!/usr/bin/env bash
# Publish a trained Amini Cocoa Contamination checkpoint + metrics to Hugging Face Hub.
#
# Reads:
#   HF_TOKEN  (required) — an HF access token with write scope
#   HF_REPO   (optional) — target repo, defaults to $(huggingface-cli whoami)/amini-cocoa-dinov2-l-mi300x
#
# Usage:
#   bash publish.sh runs/amini_dinov2_20260430_1200
#
# Requires: huggingface_hub (installed by setup_rocm.sh).

set -euo pipefail

RUN_DIR="${1:-}"
if [ -z "$RUN_DIR" ]; then
  # auto-pick the latest run
  RUN_DIR=$(ls -1dt runs/*/ 2>/dev/null | head -1 || true)
  if [ -z "$RUN_DIR" ]; then
    echo "Usage: bash publish.sh <run_dir>    (no runs/ found to auto-pick)"
    exit 1
  fi
  RUN_DIR="${RUN_DIR%/}"
  echo "[publish] auto-picked latest run: $RUN_DIR"
fi

[ -d "$RUN_DIR" ] || { echo "[publish] not a directory: $RUN_DIR"; exit 1; }
[ -f "$RUN_DIR/best.pt" ] || { echo "[publish] missing $RUN_DIR/best.pt"; exit 1; }

: "${HF_TOKEN:?HF_TOKEN is not set. Run: export HF_TOKEN='hf_...'}"

# Log in to HF using the token (no interactive prompt)
python - <<'PY'
import os
from huggingface_hub import HfApi, login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
me = HfApi().whoami()["name"]
print(f"[publish] authenticated as: {me}")
PY

HF_USER=$(python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])")

# Identify backbone from config
BACKBONE_NAME=$(python -c "import yaml; print(yaml.safe_load(open('$RUN_DIR/config.yaml'))['model']['name'])" 2>/dev/null || echo "dinov2-l")
SHORT_NAME=$(echo "$BACKBONE_NAME" | grep -q "eva02" && echo "eva02-l" || echo "dinov2-l")

HF_REPO="${HF_REPO:-${HF_USER}/amini-cocoa-${SHORT_NAME}-mi300x}"
echo "[publish] target repo: $HF_REPO"

# --------------------------------------------------------------
# Build a clean release folder
# --------------------------------------------------------------
REL_DIR="${RUN_DIR}/_release"
rm -rf "$REL_DIR"
mkdir -p "$REL_DIR"

cp "$RUN_DIR/best.pt"                   "$REL_DIR/best.pt"
cp "$RUN_DIR/config.yaml"               "$REL_DIR/config.yaml"           2>/dev/null || true
cp "$RUN_DIR/metrics.json"              "$REL_DIR/metrics.json"          2>/dev/null || true
cp "$RUN_DIR/classification_report.txt" "$REL_DIR/classification_report.txt" 2>/dev/null || true
cp "$RUN_DIR/confusion_matrix.csv"      "$REL_DIR/confusion_matrix.csv"  2>/dev/null || true

# --------------------------------------------------------------
# Generate a model card (README.md)
# --------------------------------------------------------------
python - "$REL_DIR" "$HF_REPO" "$BACKBONE_NAME" <<'PY'
import json, sys, yaml
from pathlib import Path

rel_dir = Path(sys.argv[1])
repo = sys.argv[2]
backbone = sys.argv[3]

def maybe_read_json(p):
    p = rel_dir / p
    return json.loads(p.read_text()) if p.exists() else {}

metrics = maybe_read_json("metrics.json")
config = {}
if (rel_dir / "config.yaml").exists():
    config = yaml.safe_load((rel_dir / "config.yaml").read_text())

std = metrics.get("standard", {})
tta = metrics.get("tta", {})

base_model = "facebook/dinov2-large"
if "eva02" in backbone:
    base_model = "YTViT/eva-02-large-patch14-448" # approximate tag

card = f"""---
license: apache-2.0
library_name: timm
tags:
  - image-classification
  - plant-disease
  - cocoa
  - rocm
  - mi300x
  - amd
base_model: {base_model}
datasets:
  - ohagwucollinspatrick/amini-cocoa-contamination-dataset
---

# {backbone} — Amini Cocoa Contamination (MI300X fine-tune)

Fine-tuned **{backbone}** on the **Amini cocoa contamination** dataset
(3 classes: anthracnose, cssvd, healthy).

Trained on a single **AMD Instinct MI300X** using PyTorch + ROCm, as part of the
AMD hackathon. This model uses on-the-fly bbox cropping with randomized context-padding
to improve robustness against detector imprecision.

## Results

| Metric              | This model ({short_name if 'short_name' in locals() else 'ViT-L'} / MI300X) |
|---------------------|-------------------------------:|
| Test accuracy (TTA) | {tta.get('acc', 0):.4f} |
| Macro F1 (TTA)      | {tta.get('macro_f1', 0):.4f} |
| Standard acc        | {std.get('acc', 0):.4f} |

TTA rounds: {tta.get('rounds', 10)}.

## Training

- **Backbone:** {backbone}
- **Precision:** bf16 (native MI300X)
- **Optimizer:** AdamW, cosine schedule
- **Augmentation:** RandAugment + Mixup/CutMix + Random context-pad [0.0, 0.15]

See `config.yaml` for the full hyperparameter set.

## Usage

```python
import timm, torch

model = timm.create_model(
    "{backbone}",
    pretrained=False,
    num_classes=3,
    img_size={config.get('model', {}).get('img_size', 224)},
)
ckpt = torch.load("best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

## Artifacts

- `best.pt` — model weights + training config
- `config.yaml` — hyperparameters used for this run
- `classification_report.txt` — per-class precision / recall / F1
- `confusion_matrix.csv` — 3x3 confusion matrix
- `metrics.json` — standard + TTA scores

## Source

Training code: <https://github.com/genyarko/amd-merolav/tree/main/cocoa_amini_finetuning>
"""
(rel_dir / "README.md").write_text(card)
print(f"[publish] wrote model card -> {rel_dir/'README.md'}")
PY

# --------------------------------------------------------------
# Create repo (if missing) and upload
# --------------------------------------------------------------
python - "$REL_DIR" "$HF_REPO" <<'PY'
import os, sys
from huggingface_hub import HfApi, create_repo

rel_dir = sys.argv[1]
repo = sys.argv[2]

api = HfApi()
try:
    create_repo(repo, token=os.environ["HF_TOKEN"], repo_type="model", exist_ok=True)
    print(f"[publish] repo ready: {repo}")
except Exception as e:
    print(f"[publish] create_repo warning: {e}")

print(f"[publish] uploading {rel_dir} -> {repo}")
api.upload_folder(
    folder_path=rel_dir,
    repo_id=repo,
    repo_type="model",
    token=os.environ["HF_TOKEN"],
    commit_message="Upload Amini cocoa contamination checkpoint (MI300X)",
)
print(f"[publish] done -> https://huggingface.co/{repo}")
PY
