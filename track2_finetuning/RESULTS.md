# Track 2 — Results Summary

**Shipped:** 2026-04-12
**Hackathon:** lablab.ai AMD Hackathon — Track 2 (Fine-Tuning on AMD GPUs)
**Published model:** https://huggingface.co/iamcode6/dinov2-l-ccmt-mi300x

## Headline numbers

| Metric            | This run (DINOv2-L / MI300X) | Baseline (EfficientNetB0 / Kaggle P100) | Delta     |
|-------------------|-----------------------------:|---------------------------------------:|----------:|
| Test accuracy     | **0.9706** (TTA x10)         | 0.9316                                  | **+3.90pp** |
| Test macro F1     | **0.9713**                   | 0.9348                                  | **+3.65pp** |
| Standard acc      | 0.9705 (no TTA)              | —                                       | —         |

TTA improvement on DINOv2 was only +0.02pp — the self-supervised LVD-142M pretrain already bakes in augmentation invariance, so TTA has nothing left to add.

## Training run

| Setting        | Value                                  |
|----------------|----------------------------------------|
| Model          | `vit_large_patch14_dinov2.lvd142m` (304M params, ViT-L/14) |
| Hardware       | Single AMD Instinct MI300X (192 GB HBM3) |
| Precision      | bf16 (native on MI300X, no GradScaler) |
| Batch size     | 256                                     |
| Learning rate  | 1e-4 unified (AdamW, cosine + 10% warmup) |
| Epochs         | 15 (no early stop — plateau around epoch 13) |
| Augmentation   | RandAugment (m=9, n=2) + Mixup(α=0.1) + CutMix(α=0.5) + RandomErasing (p=0.25) |
| Weight decay   | 0.05                                    |
| Grad clip      | 1.0 (via `clip_grad_norm_`)             |
| Duration       | ~51 min end-to-end                      |
| Throughput     | 414-426 img/s real, 499 img/s synthetic |
| Peak VRAM      | ~40 GB / 192 GB (20%)                   |

## Per-epoch trajectory

```
epoch  1 | acc=0.8586 f1=0.8633
epoch  2 | acc=0.8508 f1=0.8532   (slight dip — mixup/weighted-sampler variance)
epoch  3 | acc=0.8769 f1=0.8809
epoch  4 | acc=0.8950 f1=0.8968
epoch  5 | acc=0.9106 f1=0.9113
epoch  6 | acc=0.9207 f1=0.9240
epoch  7 | acc=0.9289 f1=0.9326
epoch  8 | acc=0.9438 f1=0.9453   ← crossed P100 baseline
epoch  9 | acc=0.9498 f1=0.9492
epoch 10 | acc=0.9553 f1=0.9563
epoch 11 | acc=0.9629 f1=0.9618
epoch 12 | acc=0.9656 f1=0.9654
epoch 13 | acc=0.9677 f1=0.9670
epoch 14 | acc=0.9680 f1=0.9670
epoch 15 | acc=0.9686 f1=0.9682   ← best val
```

## Dataset

- **Source:** Mendeley CCMT Dataset-Augmented (via Kaggle: `merolavtechnology/dataset-for-crop-pest-and-disease-detection`)
- **Classes:** 22 across 4 crops (cashew, cassava, maize, tomato)
- **Total images:** 105,252 valid
- **Split:** 84,201 train / 10,525 val / 10,526 test (grouped by source image, seed=123, 80/10/10)
- **Parity:** split reproduces the Kaggle P100 notebook exactly for apples-to-apples comparison

## Benchmark (synthetic batches, bs=256, bf16)

| Batch size | Train img/s | Train VRAM | Eval img/s | Eval VRAM |
|-----------:|------------:|-----------:|-----------:|----------:|
| 128        | 485.3       | 37.5 GB    | 1808.5     | 68.0 GB   |
| 256        | 499.2       | 70.5 GB    | 1849.5     | 133.4 GB  |
| 512        | 511.9       | 136.5 GB   | OOM        | —         |
| 1024       | OOM         | —          | OOM        | —         |

MI300X is compute-bound on ViT-L at 224², not memory-bound. Saturation hits at bs=256 — bs=512 buys only +2.5% throughput for 2× VRAM.

## Non-obvious lessons

1. **DINOv2 + TTA is a waste** — +0.02pp lift. Drop TTA for this family.
2. **Docker `/dev/shm` is 64 MB** by default — kills DataLoader workers with "Bus error" at `num_workers=8, pin_memory=True, bs=256`. Cap workers at 4 or use `--shm-size=8g`.
3. **timm Mixup requires even batch sizes** — always `drop_last=True`.
4. **MI300X is compute-bound, not memory-bound, on ViT-L.** Different tuning mindset than A100.
5. **Hardcoded overrides silently drift from config.** `train.py` used mixup 0.1/0.5 while config said 0.2/1.0; the shipped run is the 0.1/0.5 variant. Keep this in mind for reproducibility.
6. **Long runs need `nohup ... > log 2>&1 &` + `python -u`.** Lost the first attempt to an SSH disconnect with no log.

## Artifacts on HF Hub

Everything in `https://huggingface.co/iamcode6/dinov2-l-ccmt-mi300x`:
- `best.pt` — weights + embedded cfg
- `config.yaml` — exact hyperparameters
- `metrics.json` — standard + TTA numbers
- `classification_report.txt` — per-class precision / recall / F1 (all 22 classes)
- `confusion_matrix.csv` — 22×22
- `train.log` — full per-epoch output from the training run
- `benchmark_results.json` — throughput sweep

## What's next

- **Build-in-Public:** post the headline numbers to X and LinkedIn, tag `@lablab` and `@AIatAMD`
- **Destroy droplet** — all artifacts are persisted off-box, setup scripts reproduce the environment
- **Track 3:** multimodal plant assistant using this checkpoint as the vision backbone (DINOv2-L features + VLM for treatment advice)
