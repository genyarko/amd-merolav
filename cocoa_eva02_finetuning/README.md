# Cocoa Pod 5-Class — EVA-02-Large fine-tune (ROCm / MI300X)

EVA-02-L fine-tune on the merged LatAm + Peru cocoa-pod YOLO datasets.
Mirrors the Track 2 (`track2_finetuning/`) pipeline, swapped for:
- **Backbone**: `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` (was DINOv2-L @ 224)
- **Resolution**: 448 (was 224)
- **Classes**: 5 (carmenta / healthy / moniliasis / phytophthora / witches_broom)
- **Data prep**: YOLO box crops from two source datasets (was ImageFolder)

## Datasets

Two Kaggle datasets are merged into one 5-class label space:

| Dataset slug | License terms acceptance | Classes |
|---|---|---|
| `serranosebas/enfermedades-cacao-yolov4` | click "Download" once on the page | Fitoftora → phytophthora, Monilia → moniliasis, Sana → healthy |
| `bryandarquea/cocoa-diseases` | click "Download" once on the page | healthy, carmenta, witches_broom, moniliasis, phytophthora |

Expected merged crop count: ~4,870 boxes from ~1,800 source images.

## Bootstrap (fresh MI300X droplet)

```bash
# 1. Kaggle credentials
mkdir -p ~/.kaggle && \
  echo '{"username":"YOUR_USER","key":"YOUR_KEY"}' > ~/.kaggle/kaggle.json && \
  chmod 600 ~/.kaggle/kaggle.json

# 2. Clone repo
git clone https://github.com/genyarko/amd-merolav.git
cd amd-merolav/cocoa_eva02_finetuning

# 3. Environment (ROCm-aware venv)
bash setup_rocm.sh
source ~/venv/bin/activate

# 4. End-to-end run (stage → prepare → train → eval)
bash run_mi300x.sh
```

`run_mi300x.sh` is the orchestrator. Outputs land in `runs/eva02_l_<timestamp>/`:
- `best.pt`               — checkpoint with the highest val macro-F1
- `metrics.json`          — full training history + best
- `classification_report.txt`, `confusion_matrix.csv` — produced by `eval.py`
- `config.yaml`           — frozen copy of the config used for the run

## Step-by-step (debugging)

```bash
# Pull both Kaggle datasets
DATA_ROOT=/workspace/data/cocoa bash stage_cocoa_dataset.sh

# Crop YOLO boxes and build splits.json
python prepare_cocoa_data.py \
    --latam-root  "/workspace/data/cocoa/latam/Enfermedades Cacao" \
    --peru-images "/workspace/data/cocoa/peru/cocoa_diseases/images" \
    --peru-labels "/workspace/data/cocoa/peru/cocoa_diseases/labels" \
    --crop-dir    /workspace/data/cocoa/crops \
    --out         splits.json

# Train EVA-02-L
python train.py --config config.yaml --splits splits.json --output runs/eva02_l_v1

# Evaluate with TTA × 10
python eval.py --checkpoint runs/eva02_l_v1/best.pt --splits splits.json --tta 10
```

## Memory + throughput notes

- 448×448 inputs use 4× the activation memory of Track 2's 224×224 setup.
- `config.yaml` starts at `batch_size: 64` with `grad_checkpointing: true` to fit on a single MI300X. If the first epoch's VRAM headroom shows comfortably under 192 GB, raise to 96 or 128.
- `compile: false` for the first stable run. Toggle on once accuracy is in range — torch.compile + bf16 typically gives 1.3–1.6× speedup on MI300X but adds startup overhead.
- `bf16` autocast is the safe default (MI300X has native bf16). Don't switch to fp16 — EVA-02 has had issues with fp16 underflow in the past.

## Class imbalance

Training set distribution (post image-level split, pre-sampler):
- healthy        ≈ 2,750
- phytophthora   ≈ 460
- moniliasis     ≈ 420
- carmenta       ≈ 200
- witches_broom  ≈ 80

`train.py` uses `WeightedRandomSampler` (1/count weighting) so each batch sees roughly even class representation. Best metric is **val macro-F1** (not accuracy), which is what the rare classes will move.

## Differences from `track2_finetuning/`

| | Track 2 (DINOv2 leaves) | This (EVA-02 cocoa) |
|---|---|---|
| Backbone | `vit_large_patch14_dinov2.lvd142m` | `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` |
| Image size | 224 | 448 |
| Num classes | 22 | 5 |
| Batch size | 256 | 64 |
| Data prep | ImageFolder + grouped split | YOLO crop + image-level split |
| Stage source | one Kaggle dataset | two Kaggle datasets, merged |
| `prepare_*.py` | `prepare_data.py` | `prepare_cocoa_data.py` |
| `stage_*.sh` | `stage_dataset.sh` | `stage_cocoa_dataset.sh` |
| `vertical_flip` | false (leaves) | true (pods) |

## Target

DINOv2-L on Track 2 hit 97.06% / 0.9713 macro-F1. EVA-02-L at 448 on cleaner cocoa data should land in a similar neighborhood — **target 96–98% test accuracy, macro F1 ≥ 0.95**. Per-class F1 on `witches_broom` (n≈80) will be the binding constraint.
