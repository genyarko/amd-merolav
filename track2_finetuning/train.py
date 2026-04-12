"""DINOv2-Large fine-tuning on CCMT crop-disease — PyTorch + ROCm.

Two-phase training:
  Phase 1 — Linear probe (backbone frozen, head only).
  Phase 2 — Full fine-tune with discriminative layer-wise LR decay.

Run:
    python train.py --config config.yaml --splits splits.json --output runs/dinov2_l_v1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import f1_score
from timm.data import Mixup, create_transform
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# --------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------- #
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
            auto_augment=f"rand-m{cfg['augment']['rand_augment_m']}"
                         f"-n{cfg['augment']['rand_augment_n']}",
            re_prob=cfg["augment"]["random_erasing_p"],
            re_mode="pixel",
            hflip=0.5 if cfg["augment"]["horizontal_flip"] else 0.0,
            vflip=0.5 if cfg["augment"]["vertical_flip"] else 0.0,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            interpolation="bicubic",
        )
    return create_transform(
        input_size=size, is_training=False,
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
        interpolation="bicubic", crop_pct=0.95,
    )


# --------------------------------------------------------------------- #
# Model + parameter groups
# --------------------------------------------------------------------- #
def build_model(cfg):
    model = timm.create_model(
        cfg["model"]["name"],
        pretrained=True,
        num_classes=cfg["model"]["num_classes"],
        img_size=cfg["model"]["img_size"],
        drop_path_rate=cfg["model"]["drop_path_rate"],
    )
    return model


def param_groups_layer_decay(model, head_lr, backbone_lr, weight_decay, layer_decay):
    """Build AdamW parameter groups with ViT layer-wise LR decay.
    Later layers (closer to head) get larger LR. Head gets `head_lr`.
    """
    blocks = getattr(model, "blocks", None)
    n_layers = len(blocks) if blocks is not None else 0
    # layer_id 0 = patch_embed + cls + pos_embed; 1..n_layers = block i-1; n_layers+1 = head
    num_layers = n_layers + 1

    def layer_id_for(name):
        if name.startswith(("cls_token", "pos_embed", "patch_embed", "mask_token",
                           "register_tokens")):
            return 0
        if name.startswith("blocks."):
            return int(name.split(".")[1]) + 1
        return num_layers   # norm + head

    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lid = layer_id_for(name)
        is_head = lid == num_layers
        scale = layer_decay ** (num_layers - lid)
        lr = head_lr if is_head else backbone_lr * scale
        # no weight decay on biases / norm / tokens
        decay = 0.0 if (p.ndim == 1 or name.endswith(".bias")
                        or "token" in name or "pos_embed" in name) else weight_decay
        key = f"layer{lid}_{'wd' if decay > 0 else 'nowd'}"
        groups.setdefault(key, {"params": [], "lr": lr, "weight_decay": decay})
        groups[key]["params"].append(p)
    return list(groups.values())


# --------------------------------------------------------------------- #
# Scheduler (cosine with warmup)
# --------------------------------------------------------------------- #
def cosine_warmup(step, total_steps, warmup_steps, base_scale=1.0, min_scale=0.0):
    if step < warmup_steps:
        return base_scale * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_scale + (base_scale - min_scale) * 0.5 * (1 + math.cos(math.pi * progress))


# --------------------------------------------------------------------- #
# Train / eval loops
# --------------------------------------------------------------------- #
@dataclass
class PhaseResult:
    best_val_acc: float
    best_val_f1: float
    best_epoch: int


def evaluate(model, loader, device, amp_dtype):
    model.eval()
    all_pred, all_true = [], []
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            all_pred.append(logits.argmax(dim=1).cpu())
            all_true.append(y.cpu())
    pred = torch.cat(all_pred).numpy()
    true = torch.cat(all_true).numpy()
    acc = (pred == true).mean()
    f1 = f1_score(true, pred, average="macro")
    return {"loss": total_loss / n, "acc": float(acc), "macro_f1": float(f1)}


def run_phase(phase_cfg, cfg, model, train_loader, val_loader, device, writer,
              ckpt_dir, phase_name, global_step_offset=0):
    num_classes = cfg["model"]["num_classes"]
    amp_dtype = torch.bfloat16 if cfg["train"]["amp_dtype"] == "bfloat16" else torch.float16

    if phase_cfg["freeze_backbone"]:
        for n, p in model.named_parameters():
            p.requires_grad = not (n.startswith("blocks.") or
                                   n.startswith(("cls_token", "pos_embed", "patch_embed",
                                                "norm.", "mask_token", "register_tokens")))
        groups = [{
            "params": [p for p in model.parameters() if p.requires_grad],
            "lr": phase_cfg["lr_head"], "weight_decay": cfg["train"]["weight_decay"],
        }]
    else:
        for p in model.parameters():
            p.requires_grad = True
        groups = param_groups_layer_decay(
            model,
            head_lr=phase_cfg["lr_head"],
            backbone_lr=phase_cfg["lr_backbone"],
            weight_decay=cfg["train"]["weight_decay"],
            layer_decay=phase_cfg.get("layer_decay", 1.0),
        )

    optimizer = torch.optim.AdamW(groups, betas=tuple(cfg["train"]["betas"]))
    scaler = None  # bf16 doesn't need GradScaler; fp16 would

    mixup = None
    if cfg["train"]["mixup_alpha"] > 0 or cfg["train"]["cutmix_alpha"] > 0:
        mixup = Mixup(
            mixup_alpha=cfg["train"]["mixup_alpha"],
            cutmix_alpha=cfg["train"]["cutmix_alpha"],
            prob=cfg["train"]["mixup_prob"],
            label_smoothing=cfg["train"]["label_smoothing"],
            num_classes=num_classes,
        )
    # if no mixup, plain label-smoothed CE still applies
    base_loss = nn.CrossEntropyLoss(label_smoothing=cfg["train"]["label_smoothing"])

    epochs = phase_cfg["epochs"]
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = phase_cfg["warmup_epochs"] * steps_per_epoch
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    best = PhaseResult(0.0, 0.0, -1)

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running_loss, seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f"[{phase_name}] epoch {epoch+1}/{epochs}")
        for step, (x, y) in enumerate(pbar):
            global_step = global_step_offset + epoch * steps_per_epoch + step
            scale = cosine_warmup(global_step - global_step_offset,
                                  total_steps, warmup_steps)
            for g, base in zip(optimizer.param_groups, base_lrs):
                g["lr"] = base * scale

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if mixup is not None:
                x, y_soft = mixup(x, y)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(x)
                if mixup is not None:
                    loss = -(y_soft * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                else:
                    loss = base_loss(logits, y)

            loss.backward()
            if cfg["train"]["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg["train"]["grad_clip"])
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            seen += x.size(0)

            if global_step % cfg["log"]["every_n_steps"] == 0:
                writer.add_scalar(f"{phase_name}/train_loss", loss.item(), global_step)
                writer.add_scalar(f"{phase_name}/lr_head",
                                  optimizer.param_groups[-1]["lr"], global_step)

            pbar.set_postfix(loss=f"{running_loss/seen:.4f}",
                             lr=f"{optimizer.param_groups[-1]['lr']:.2e}")

        epoch_time = time.time() - t0
        samples_per_sec = seen / max(epoch_time, 1e-6)
        metrics = evaluate(model, val_loader, device, amp_dtype)
        writer.add_scalar(f"{phase_name}/val_loss", metrics["loss"], epoch)
        writer.add_scalar(f"{phase_name}/val_acc", metrics["acc"], epoch)
        writer.add_scalar(f"{phase_name}/val_macro_f1", metrics["macro_f1"], epoch)
        writer.add_scalar(f"{phase_name}/samples_per_sec", samples_per_sec, epoch)

        print(f"[{phase_name}] epoch {epoch+1}: train_loss={running_loss/seen:.4f}  "
              f"val_acc={metrics['acc']:.4f}  val_f1={metrics['macro_f1']:.4f}  "
              f"{samples_per_sec:.0f} img/s")

        metric_key = cfg["log"]["save_best_metric"]
        cur = metrics["macro_f1"] if metric_key == "val_macro_f1" else metrics["acc"]
        best_cur = best.best_val_f1 if metric_key == "val_macro_f1" else best.best_val_acc
        if cur > best_cur:
            best = PhaseResult(metrics["acc"], metrics["macro_f1"], epoch)
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "phase": phase_name,
                "metrics": metrics,
                "cfg": cfg,
            }, ckpt_dir / "best.pt")
            print(f"  ↳ new best ({metric_key}={cur:.4f}) — saved best.pt")

    return best, global_step_offset + epochs * steps_per_epoch


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    splits = json.loads(args.splits.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.yaml").write_text(yaml.safe_dump(cfg))

    set_seed(cfg["seed"])
    if not torch.cuda.is_available():
        raise SystemExit("CUDA (ROCm) device not available — check setup_rocm.sh output.")
    device = torch.device("cuda:0")
    print(f"[train] device: {torch.cuda.get_device_name(0)}")
    print(f"[train] ROCm/HIP: {getattr(torch.version, 'hip', None)}")

    # Data
    train_ds = CCMTDataset(splits["train"], build_transforms(cfg, is_train=True))
    val_ds = CCMTDataset(splits["val"],   build_transforms(cfg, is_train=False))
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"], pin_memory=cfg["data"]["pin_memory"],
        drop_last=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["eval"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=cfg["data"]["pin_memory"],
        persistent_workers=True,
    )

    # Model
    model = build_model(cfg).to(device)
    if cfg["train"].get("compile", False):
        model = torch.compile(model)
    if args.resume:
        sd = torch.load(args.resume, map_location=device)
        model.load_state_dict(sd["state_dict"])
        print(f"[train] resumed from {args.resume}")

    print(f"[train] params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    writer = SummaryWriter(log_dir=str(args.output / "tb"))

    # Phase 1 — linear probe
    p1, step_off = run_phase(
        cfg["train"]["phase1"], cfg, model, train_loader, val_loader, device,
        writer, args.output, phase_name="phase1_linear_probe", global_step_offset=0,
    )
    print(f"[train] phase1 best acc={p1.best_val_acc:.4f}  f1={p1.best_val_f1:.4f}")

    # Phase 2 — full fine-tune
    p2, _ = run_phase(
        cfg["train"]["phase2"], cfg, model, train_loader, val_loader, device,
        writer, args.output, phase_name="phase2_full_ft", global_step_offset=step_off,
    )
    print(f"[train] phase2 best acc={p2.best_val_acc:.4f}  f1={p2.best_val_f1:.4f}")

    (args.output / "final_metrics.json").write_text(json.dumps({
        "phase1": {"acc": p1.best_val_acc, "macro_f1": p1.best_val_f1},
        "phase2": {"acc": p2.best_val_acc, "macro_f1": p2.best_val_f1},
    }, indent=2))
    writer.close()


if __name__ == "__main__":
    main()
