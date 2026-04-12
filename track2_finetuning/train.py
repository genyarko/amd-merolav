"""DINOv2-Large fine-tuning on CCMT crop-disease — PyTorch + ROCm.

Two-phase training:
  Phase 1 — Linear probe (backbone frozen, head only).
  Phase 2 — Full fine-tune with discriminative layer-wise LR decay.

Run:
    python train.py --config config.yaml --splits splits.json --output runs/dinov2_l_v1
"""
from __future__ import annotations

import argparse, json, math, random, shutil, time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import f1_score
from timm.data import Mixup, create_transform
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# ------------------ Utils ------------------ #
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def optimize_model(model, cfg):
    model = model.to(memory_format=torch.channels_last)

    if cfg["train"].get("compile", True):
        model = torch.compile(model)

    if cfg["train"].get("grad_checkpointing", False):
        if hasattr(model, "set_grad_checkpointing"):
            model.set_grad_checkpointing(True)

    return model


# ------------------ Dataset ------------------ #
class CCMTDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = Image.open(r["path"]).convert("RGB")
        return self.transform(img), r["label_idx"]


def build_transforms(cfg, is_train):
    size = cfg["model"]["img_size"]

    if is_train:
        return create_transform(
            input_size=size,
            is_training=True,
            auto_augment=f"rand-m{cfg['augment']['rand_augment_m']}-n{cfg['augment']['rand_augment_n']}",
            re_prob=cfg["augment"]["random_erasing_p"],
            re_mode="pixel",
            hflip=0.5 if cfg["augment"].get("horizontal_flip", True) else 0.0,
            vflip=0.5 if cfg["augment"].get("vertical_flip", False) else 0.0,
            interpolation="bicubic",
        )

    return create_transform(input_size=size, is_training=False, crop_pct=0.95)


# ------------------ Model ------------------ #
def build_model(cfg):
    return timm.create_model(
        cfg["model"]["name"],
        pretrained=True,
        num_classes=cfg["model"]["num_classes"],
        img_size=cfg["model"]["img_size"],
    )


# ------------------ Scheduler ------------------ #
def cosine_warmup(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


# ------------------ Eval ------------------ #
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            preds.append(logits.argmax(1).cpu())
            targets.append(y.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    acc = (preds == targets).float().mean().item()
    f1 = f1_score(targets, preds, average="macro")

    return acc, f1


# ------------------ Training ------------------ #
@dataclass
class Best:
    f1: float = 0.0
    patience: int = 0


def _unwrap_state_dict(sd):
    """Strip torch.compile's ``_orig_mod.`` key prefix for clean checkpoints."""
    return {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}


def train(cfg, splits, out_dir: Path):
    device = torch.device("cuda")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- datasets ---- #
    train_ds = CCMTDataset(splits["train"], build_transforms(cfg, True))
    val_ds = CCMTDataset(splits["val"], build_transforms(cfg, False))

    # ---- class imbalance handling ---- #
    labels = [r["label_idx"] for r in splits["train"]]
    class_counts = np.bincount(labels)
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum()

    sample_weights = [weights[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # ---- loaders ---- #
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        sampler=sampler,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,   # timm Mixup requires even batch sizes; drop the trailing partial
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    # ---- model ---- #
    model = build_model(cfg).to(device)
    model = optimize_model(model, cfg)

    # ---- optimizer ---- #
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    total_steps = cfg["train"]["epochs"] * len(train_loader)
    warmup_steps = int(0.1 * total_steps)

    # ------------------ REDUCED MIXUP ------------------ #
    mixup = Mixup(
        mixup_alpha=0.1,      # ↓ reduced
        cutmix_alpha=0.5,     # ↓ reduced
        prob=0.5,
        num_classes=cfg["model"]["num_classes"],
    )

    best = Best()
    history = []

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        t0 = time.time()

        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)

            # Apply reduced mixup
            x, y_mix = mixup(x, y)

            # LR schedule (correct, non-compounding)
            step_id = epoch * len(train_loader) + step
            lr_scale = cosine_warmup(step_id, total_steps, warmup_steps)

            for g in optimizer.param_groups:
                g["lr"] = cfg["train"]["lr"] * lr_scale

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = -(y_mix * F.log_softmax(logits, dim=-1)).sum(dim=1).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_clip = cfg["train"].get("grad_clip", 0.0)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        acc, f1 = evaluate(model, val_loader, device)
        speed = len(train_ds) / (time.time() - t0)

        print(f"epoch {epoch+1} | acc={acc:.4f} f1={f1:.4f} {speed:.0f} img/s")
        history.append({"epoch": epoch + 1, "val_acc": acc, "val_macro_f1": f1,
                        "images_per_sec": round(speed, 1)})

        # ---- early stopping on F1 ---- #
        if f1 > best.f1:
            best.f1 = f1
            best.patience = 0
            torch.save(
                {"state_dict": _unwrap_state_dict(model.state_dict()),
                 "cfg": cfg,
                 "epoch": epoch + 1,
                 "val_acc": acc,
                 "val_macro_f1": f1},
                out_dir / "best.pt",
            )
        else:
            best.patience += 1

        (out_dir / "metrics.json").write_text(json.dumps(
            {"history": history, "best": {"epoch_f1": best.f1}}, indent=2))

        if best.patience >= 5:
            print("Early stopping triggered")
            break

    print(f"[train] best val_macro_f1={best.f1:.4f}  artifacts in {out_dir}")


# ------------------ Main ------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("runs/dinov2_l_v1"),
                    help="run directory — best.pt, metrics.json, config.yaml land here")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    splits = json.loads(args.splits.read_text())

    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, args.output / "config.yaml")

    set_seed(cfg["seed"])
    train(cfg, splits, args.output)


if __name__ == "__main__":
    main()