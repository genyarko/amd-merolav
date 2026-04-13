#!/usr/bin/env bash
# Publish trained Llama 3.2 Vision LoRA adapters + metrics to Hugging Face Hub.
#
# Reads:
#   HF_TOKEN  (required) — an HF access token with write scope
#   HF_REPO   (optional) — target repo, defaults to <user>/llama32-vision-ccmt-mi300x
#
# Usage:
#   bash vision/publish_vlm.sh runs/llama_vision_v1
#   HF_REPO=myuser/my-model bash vision/publish_vlm.sh runs/llama_vision_v1
#
# Requires: huggingface_hub

set -euo pipefail

RUN_DIR="${1:-}"
if [ -z "$RUN_DIR" ]; then
  RUN_DIR=$(ls -1dt runs/*/ 2>/dev/null | head -1 || true)
  if [ -z "$RUN_DIR" ]; then
    echo "Usage: bash vision/publish_vlm.sh <run_dir>"
    exit 1
  fi
  RUN_DIR="${RUN_DIR%/}"
  echo "[publish] auto-picked latest run: $RUN_DIR"
fi

[ -d "$RUN_DIR" ] || { echo "[publish] not a directory: $RUN_DIR"; exit 1; }
[ -d "$RUN_DIR/best_adapter" ] || { echo "[publish] missing $RUN_DIR/best_adapter"; exit 1; }

: "${HF_TOKEN:?HF_TOKEN is not set. Run: export HF_TOKEN='hf_...'}"

# Authenticate
python3 - <<'PY'
import os
from huggingface_hub import HfApi, login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
me = HfApi().whoami()["name"]
print(f"[publish] authenticated as: {me}")
PY

HF_USER=$(python3 -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])")
HF_REPO="${HF_REPO:-${HF_USER}/llama32-vision-ccmt-mi300x}"
echo "[publish] target repo: $HF_REPO"

# --------------------------------------------------------------
# Build release folder — best adapter + optional epoch3 + metrics
# --------------------------------------------------------------
REL_DIR="${RUN_DIR}/_release"
rm -rf "$REL_DIR"
mkdir -p "$REL_DIR/best_adapter"

cp -r "$RUN_DIR/best_adapter/"* "$REL_DIR/best_adapter/"
cp "$RUN_DIR/config.yaml"       "$REL_DIR/config.yaml"  2>/dev/null || true
cp "$RUN_DIR/metrics.json"      "$REL_DIR/metrics.json"  2>/dev/null || true

if [ -d "$RUN_DIR/epoch3_adapter" ]; then
  mkdir -p "$REL_DIR/epoch3_adapter"
  cp -r "$RUN_DIR/epoch3_adapter/"* "$REL_DIR/epoch3_adapter/"
  echo "[publish] including epoch3_adapter"
fi

# --------------------------------------------------------------
# Generate model card
# --------------------------------------------------------------
python3 - "$REL_DIR" "$HF_REPO" "$RUN_DIR" <<'PYCARD'
import json, sys, yaml
from pathlib import Path

rel_dir = Path(sys.argv[1])
repo = sys.argv[2]
run_dir = Path(sys.argv[3])

# Load training metrics
metrics = {}
metrics_path = run_dir / "metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())

config = {}
config_path = run_dir / "config.yaml"
if config_path.exists():
    config = yaml.safe_load(config_path.read_text())

best_val = metrics.get("best_val_loss", "N/A")
history = metrics.get("history", [])

epoch_table = ""
for h in history:
    epoch_table += (
        f"| {h['epoch']} | {h['train_loss']:.4f} | {h['val_loss']:.4f} "
        f"| {h.get('examples_per_sec', 'N/A')} | {h.get('elapsed_sec', 'N/A')}s |\n"
    )

has_epoch3 = (run_dir / "epoch3_adapter").exists()
epoch3_note = ""
if has_epoch3:
    epoch3_note = (
        "\n### Epoch 3 adapter\n\n"
        "An `epoch3_adapter/` checkpoint is included for A/B comparison. "
        "Epoch 3 had val_loss=0.0151 vs epoch 5's 0.0146 — the difference is marginal "
        "and epoch 3 may generalize equally well in practice.\n"
    )

lora_cfg = config.get("lora", {})
train_cfg = config.get("train", {})

card = f"""---
license: apache-2.0
library_name: peft
tags:
  - llama-3.2-vision
  - plant-disease
  - lora
  - rocm
  - mi300x
  - amd
  - vlm
  - image-text-to-text
base_model: meta-llama/Llama-3.2-11B-Vision-Instruct
---

# Llama 3.2 Vision 11B LoRA — Plant Disease Diagnosis (MI300X fine-tune)

Fine-tuned **Llama 3.2 Vision 11B** with **LoRA** on a plant disease QA dataset
(cashew, cassava, maize, tomato — 22 disease classes) for visual diagnosis and
treatment recommendations.

Trained on a single **AMD Instinct MI300X** using PyTorch + ROCm, as a submission to the
lablab.ai AMD hackathon **Track 3 — Building AI-Powered Applications on AMD GPUs**.

## Results

| Epoch | Train Loss | Val Loss | Throughput | Wall Time |
|------:|-----------:|---------:|:----------:|:---------:|
{epoch_table}
**Best val_loss: {best_val}**
{epoch3_note}
## Training Details

- **Base model:** meta-llama/Llama-3.2-11B-Vision-Instruct (11B params, ~4B vision + ~7B language)
- **Method:** LoRA (rank={lora_cfg.get('rank', 16)}, alpha={lora_cfg.get('alpha', 32)}, dropout={lora_cfg.get('dropout', 0.05)})
- **Target modules:** {', '.join(lora_cfg.get('target_modules', ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))}
- **Precision:** bf16 (native MI300X)
- **Epochs:** {train_cfg.get('epochs', 5)}
- **Effective batch size:** {train_cfg.get('batch_size', 4) * train_cfg.get('grad_accum_steps', 4)}
- **Learning rate:** {train_cfg.get('lr', 2e-5)} with cosine decay + {train_cfg.get('warmup_ratio', 0.1)} warmup
- **Optimizer:** AdamW (weight_decay={train_cfg.get('weight_decay', 0.01)})
- **Max sequence length:** {train_cfg.get('max_length', 2048)}
- **Hardware:** 1x AMD Instinct MI300X (192 GB HBM3)

## Usage

```python
from peft import PeftModel
from transformers import AutoProcessor, MllamaForConditionalGeneration

base = MllamaForConditionalGeneration.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "best_adapter")
processor = AutoProcessor.from_pretrained("best_adapter")
```

## Artifacts

- `best_adapter/` — LoRA weights from the best validation epoch
- `epoch3_adapter/` — LoRA weights from epoch 3 (for A/B comparison)
- `config.yaml` — training hyperparameters
- `metrics.json` — per-epoch training history

See `config.yaml` for the full hyperparameter set.

## Source

Training code: <https://github.com/genyarko/amd-merolav/tree/main/vision>
"""
(rel_dir / "README.md").write_text(card)
print(f"[publish] wrote model card -> {{rel_dir / 'README.md'}}")
PYCARD

# --------------------------------------------------------------
# Create repo and upload
# --------------------------------------------------------------
python3 - "$REL_DIR" "$HF_REPO" <<'PY'
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
    commit_message="Upload Llama 3.2 Vision LoRA adapter — plant disease QA (MI300X)",
)
print(f"[publish] done -> https://huggingface.co/{repo}")
PY
