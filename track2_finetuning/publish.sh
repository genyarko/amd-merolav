#!/usr/bin/env bash
# Publish a trained DINOv2-L checkpoint + metrics to Hugging Face Hub.
#
# Reads:
#   HF_TOKEN  (required) — an HF access token with write scope
#   HF_REPO   (optional) — target repo, defaults to $(huggingface-cli whoami)/dinov2-l-ccmt-mi300x
#
# Usage:
#   bash publish.sh runs/dinov2_l_20260412_0830
#   HF_REPO=myuser/my-model bash publish.sh runs/dinov2_l_20260412_0830
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
HF_REPO="${HF_REPO:-${HF_USER}/dinov2-l-ccmt-mi300x}"
echo "[publish] target repo: $HF_REPO"

# --------------------------------------------------------------
# Build a clean release folder so we don't upload TensorBoard
# event files, optimizer state, etc.
# --------------------------------------------------------------
REL_DIR="${RUN_DIR}/_release"
rm -rf "$REL_DIR"
mkdir -p "$REL_DIR"

cp "$RUN_DIR/best.pt"                   "$REL_DIR/best.pt"
cp "$RUN_DIR/config.yaml"               "$REL_DIR/config.yaml"           2>/dev/null || true
cp "$RUN_DIR/final_metrics.json"        "$REL_DIR/final_metrics.json"    2>/dev/null || true
cp "$RUN_DIR/metrics.json"              "$REL_DIR/metrics.json"          2>/dev/null || true
cp "$RUN_DIR/classification_report.txt" "$REL_DIR/classification_report.txt" 2>/dev/null || true
cp "$RUN_DIR/confusion_matrix.csv"      "$REL_DIR/confusion_matrix.csv"  2>/dev/null || true

# --------------------------------------------------------------
# Generate a model card (README.md)
# --------------------------------------------------------------
python - "$REL_DIR" "$HF_REPO" <<'PY'
import json, sys
from pathlib import Path

rel_dir = Path(sys.argv[1])
repo = sys.argv[2]

def maybe_read_json(p):
    p = rel_dir / p
    return json.loads(p.read_text()) if p.exists() else {}

metrics = maybe_read_json("metrics.json")       # eval.py output (has tta)
final   = maybe_read_json("final_metrics.json") # train.py output (phase1/phase2)

std = metrics.get("standard", {})
tta = metrics.get("tta", {})
baseline_acc = metrics.get("baseline_p100_effnetb0_acc", 0.9316)
baseline_f1  = metrics.get("baseline_p100_effnetb0_macro_f1", 0.9348)

card = f"""---
license: apache-2.0
library_name: timm
tags:
  - image-classification
  - plant-disease
  - dinov2
  - rocm
  - mi300x
  - amd
base_model: facebook/dinov2-large
datasets:
  - mendeley/crop-pest-and-disease-detection
---

# DINOv2-Large — CCMT Crop & Disease (MI300X fine-tune)

Fine-tuned **DINOv2-Large** (304M params) on the **CCMT crop-pest-and-disease** dataset
(22 classes across cashew, cassava, maize, tomato).

Trained on a single **AMD Instinct MI300X** using PyTorch + ROCm, as a submission to the
lablab.ai AMD hackathon **Track 2 — Fine-Tuning on AMD GPUs**.

## Results

| Metric              | This model (DINOv2-L / MI300X) | Baseline (EfficientNetB0 / P100) |
|---------------------|-------------------------------:|---------------------------------:|
| Test accuracy       | {tta.get('acc', 0):.4f} (TTA) | {baseline_acc:.4f} (TTA) |
| Macro F1            | {tta.get('macro_f1', 0):.4f}  | {baseline_f1:.4f} |
| Standard acc (no TTA) | {std.get('acc', 0):.4f}     | — |

TTA rounds: {tta.get('rounds', 10)}.

## Training

- **Backbone:** DINOv2-L ViT-L/14 (self-supervised, LVD-142M pretrain)
- **Precision:** bf16 (native MI300X)
- **Schedule:** 2-phase — linear probe → full fine-tune with layer-wise LR decay
- **Optimizer:** AdamW, cosine schedule, grad-clip 1.0
- **Augmentation:** RandAugment + Mixup/CutMix + RandomErasing

See `config.yaml` for the full hyperparameter set.

## Usage

```python
import timm, torch

model = timm.create_model(
    "vit_large_patch14_dinov2.lvd142m",
    pretrained=False,
    num_classes=22,
    img_size=224,
)
ckpt = torch.load("best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

Class index map is embedded inside the checkpoint under `cfg`; see the training repo
for `splits.json` which defines the `class_to_idx` mapping.

## Artifacts

- `best.pt` — model weights + training config
- `config.yaml` — hyperparameters used for this run
- `classification_report.txt` — per-class precision / recall / F1
- `confusion_matrix.csv` — 22×22 confusion matrix
- `metrics.json` — standard + TTA scores

## Source

Training code: <https://github.com/genyarko/amd-merolav/tree/main/track2_finetuning>
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
    commit_message="Upload trained checkpoint + metrics (MI300X)",
)
print(f"[publish] done -> https://huggingface.co/{repo}")
PY
