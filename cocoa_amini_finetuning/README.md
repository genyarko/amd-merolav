# Amini cocoa contamination 3-class — DINOv2-L / EVA-02-L fine-tune (ROCm / MI300X)

3-class crop-based classifier on the Amini cocoa contamination Kaggle dataset.
Beats the EfficientNetB3 Kaggle baseline (~92.13% test acc) by trading the
TF/Keras stack for a stronger ViT backbone on a 192GB MI300X.

Two backbones are included so we can compare:

| | Track 2 (ref) | This — DINOv2 | This — EVA-02 |
|---|---|---|---|
| Backbone | `vit_large_patch14_dinov2.lvd142m` | same | `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` |
| Image size | 224 | 224 | 448 |
| Num classes | 22 (PlantVillage) | 3 (amini) | 3 (amini) |
| Batch size | 256 | 256 | 64 |
| Dataset | ImageFolder | YOLO bbox crop | YOLO bbox crop |

## What's different from the Kaggle notebooks

The amini notebooks (`amini-cocoa-contamination-dataset-based-1.ipynb` etc.)
pre-crop each bbox to disk with a fixed 5% context pad and then train on
those static crops.

This pipeline keeps the bbox metadata in `splits.json` (no pre-cropping) and
crops on-the-fly inside the `Dataset.__getitem__`. The context-pad is sampled
uniformly per sample from `[pad_min, pad_max]` (default `[0.0, 0.15]` for
training, fixed `0.05` for eval). That makes the classifier robust to bbox
imprecision at deploy time, which the notebooks couldn't easily simulate.

## Dataset

| Dataset slug | License | Classes |
|---|---|---|
| `ohagwucollinspatrick/amini-cocoa-contamination-dataset` | click "Download" once | anthracnose, cssvd, healthy |

Provides `Train.csv` (columns `Image_ID, ImagePath, class, xmin, ymin, xmax, ymax`)
and the source images. Per the v2 notebook output: **9,792 boxes / 5,529
unique images / 3 classes**. After filtering boxes with area<1000 px² or
side<20 px, ~9,700 valid samples remain (`anthracnose 2,263 / cssvd 3,229 /
healthy 4,208`). The class skew is mild (1.85:1) — `WeightedRandomSampler`
in `train.py` handles it.

## Bootstrap (fresh MI300X droplet)

```bash
# 1. Kaggle credentials
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USER","key":"YOUR_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 2. Clone repo
git clone https://github.com/genyarko/amd-merolav.git
cd amd-merolav/cocoa_amini_finetuning

pip3 install --break-system-packages \
    --index-url https://download.pytorch.org/whl/rocm6.2 \
    torch torchvision

    apt install python3.12-venv
    apt install -y unzip

# 3. Environment (ROCm-aware venv)
bash setup_rocm.sh
source ~/venv/bin/activate

# 4. End-to-end run (DINOv2-L by default)
bash run_mi300x.sh

# 4b. Or run EVA-02-L instead
CONFIG=config_eva02.yaml bash run_mi300x.sh
```

`run_mi300x.sh` is the orchestrator. Outputs land in `runs/amini_<backbone>_<timestamp>/`:
- `best.pt`               — checkpoint with the highest val macro-F1
- `metrics.json`          — full training history + best
- `classification_report.txt`, `confusion_matrix.csv` — produced by `eval.py`
- `config.yaml`           — frozen copy of the config used for the run

## Step-by-step (debugging)

```bash
# Pull the Kaggle dataset
DATA_ROOT=/workspace/data/amini bash stage_amini_dataset.sh

# Build splits.json (no crops written to disk)
python prepare_amini_data.py \
    --dataset-root /workspace/data/amini \
    --out splits.json

# Train DINOv2-L
python train.py --config config_dinov2.yaml --splits splits.json --output runs/dinov2_l_v1

# Or EVA-02-L
python train.py --config config_eva02.yaml --splits splits.json --output runs/eva02_l_v1

# Evaluate with TTA × 10
python eval.py --checkpoint runs/dinov2_l_v1/best.pt --splits splits.json --tta 10
```

## Memory + throughput notes

- DINOv2-L @ 224 with `batch_size=256` matches Track 2's settings — ~30-50 GB VRAM, no checkpointing needed.
- EVA-02-L @ 448 with `batch_size=64` + `grad_checkpointing=true` fits comfortably. If first-epoch headroom is generous, raise to 96/128.
- `compile: false` for the first stable run; toggle on once accuracy is in range — torch.compile + bf16 typically gives 1.3–1.6× speedup on MI300X.
- `bf16` autocast is the safe default. Don't switch to fp16 — EVA-02 has had issues with fp16 underflow in the past.

## Target

EfficientNetB3 (Kaggle baseline): **92.13% test acc / ~0.91 macro F1**.

DINOv2-L on Track 2 leaves hit 97.06% / 0.9713 macro-F1. On a cleaner 3-class
problem with on-the-fly random-pad augmentation, this pipeline should land
in the **94–97% test acc** range.

The binding constraint is `cssvd ↔ healthy` confusion — the v2 notebook
specifically called this out. Pay attention to the per-class F1 in
`classification_report.txt`.
