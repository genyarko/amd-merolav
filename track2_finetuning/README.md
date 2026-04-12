# Track 2 — DINOv2-Large Fine-Tuning on MI300X

**Hackathon track:** Fine-Tuning on AMD GPUs (lablab.ai AMD hackathon)
**Goal:** Fine-tune DINOv2-Large (304M params) on the CCMT crop-disease dataset using
PyTorch + ROCm on a single MI300X, beating the Kaggle P100 EfficientNetB0 baseline.

## Baseline (to beat)

| Model | Hardware | Test Acc (TTA) | Macro F1 | Classes |
|---|---|---|---|---|
| EfficientNetB0 | Kaggle P100 | 93.16% | 0.9348 | 22 |

## Target

| Model | Hardware | Target Test Acc | Target Macro F1 |
|---|---|---|---|
| DINOv2-Large | MI300X (1× 192GB) | 96%+ | 0.95+ |

## Pipeline layout

```
track2_finetuning/
├── README.md            # this file
├── config.yaml          # hyperparameters
├── requirements.txt     # pip deps (PyTorch-ROCm, timm, etc.)
├── setup_rocm.sh        # one-shot environment setup on MI300X
├── prepare_data.py      # build same grouped split as Kaggle notebook (seed=123)
├── train.py             # DINOv2-L fine-tuning (linear probe → full FT)
├── eval.py              # standard + TTA evaluation, per-class F1 report
├── benchmark.py         # throughput / VRAM profiling vs P100 baseline
└── run_mi300x.sh        # orchestrate a full training run on the droplet
```

## Quickstart (on the MI300X droplet)

```bash
# 1. SSH into the droplet and clone this repo
ssh root@129.212.184.180
git clone https://github.com/genyarko/amd-merolav && cd amd-merolav/track2_finetuning

# 2. Install ROCm PyTorch + deps (uses official rocm/pytorch container)
bash setup_rocm.sh

# 3. Stage the CCMT dataset (expects /data/ccmt/CCMT Dataset-Augmented/)
python prepare_data.py --data-root /data/ccmt --out splits.json

# 4. Train
python train.py --config config.yaml --splits splits.json --output runs/dinov2_l_v1

# 5. Evaluate (writes metrics.json, classification_report.txt, confusion_matrix.csv)
python eval.py --checkpoint runs/dinov2_l_v1/best.pt --splits splits.json --tta 10

# 6. Publish to Hugging Face Hub
#    - Create a write-scope token at https://huggingface.co/settings/tokens
#    - publish.sh bundles best.pt + config.yaml + metrics into _release/ and
#      generates a model card with the P100 baseline comparison
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
bash publish.sh runs/dinov2_l_v1
# → https://huggingface.co/<your-user>/dinov2-l-ccmt-mi300x
# Override the target repo with HF_REPO=myuser/custom-name bash publish.sh ...
```

## Design notes

- **Split parity:** `prepare_data.py` reproduces the exact grouped split from the
  Kaggle notebook (seed=123, 80/10/10, augmentation-aware `group_id` extraction)
  so accuracy numbers are directly comparable.
- **Two-phase training:** Linear probe (3 epochs) → full fine-tune with
  discriminative LRs (12 epochs). Backbone LR 5e-5, head LR 1e-3.
- **Augmentation:** Light RandAugment + Mixup(0.2) + RandomErasing — DINOv2
  is pretrained heavily so aggressive aug hurts.
- **Loss:** Label-smoothed cross-entropy. Class weights are computed but default
  to uniform (DINOv2 is robust to mild imbalance).
- **MI300X leverage:** batch 128 @ 224² fits in <30GB, so we can push to bs=256
  or move to 448² resolution for the final run.
