"""Evaluate a trained EVA-02-L cocoa checkpoint with standard + TTA scoring.

Produces:
  - classification_report.txt (per-class precision/recall/F1)
  - confusion_matrix.csv
  - metrics.json (standard + TTA accuracy, macro F1)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from timm.data import create_transform
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class CocoaDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = Image.open(r["path"]).convert("RGB")
        return self.transform(img), r["label_idx"]


def clean_tf(cfg):
    return create_transform(
        input_size=cfg["model"]["img_size"], is_training=False,
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
        interpolation="bicubic", crop_pct=0.95,
    )


def tta_tf(cfg):
    # Mild aug to keep TTA cheap + aligned with training (no strong rand-aug).
    return create_transform(
        input_size=cfg["model"]["img_size"], is_training=True,
        auto_augment=None, re_prob=0.0,
        hflip=0.5,
        vflip=0.5 if cfg["augment"].get("vertical_flip", False) else 0.0,
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
        interpolation="bicubic", scale=(0.9, 1.0), ratio=(0.95, 1.05),
    )


@torch.no_grad()
def predict(model, loader, device, amp_dtype):
    model.eval()
    probs_chunks, ys = [], []
    for x, y in tqdm(loader, desc="predict", leave=False):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(x)
        probs_chunks.append(F.softmax(logits, dim=-1).float().cpu())
        ys.append(y)
    return torch.cat(probs_chunks).numpy(), torch.cat(ys).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--tta", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Tolerate both formats: wrapped {"state_dict": ..., "cfg": ...} or a raw state_dict.
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        ckpt_cfg = ckpt.get("cfg")
    else:
        state_dict = ckpt
        ckpt_cfg = None

    if args.config:
        cfg = yaml.safe_load(args.config.read_text())
    elif ckpt_cfg is not None:
        cfg = ckpt_cfg
    else:
        raise SystemExit("checkpoint has no embedded cfg — pass --config explicitly")

    # Strip torch.compile's ``_orig_mod.`` prefix if present.
    state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

    splits = json.loads(args.splits.read_text())
    idx_to_class = {v: k for k, v in splits["class_to_idx"].items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    out_dir = args.output or args.checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")
    amp_dtype = torch.bfloat16 if cfg["train"]["amp_dtype"] == "bfloat16" else torch.float16

    model = timm.create_model(
        cfg["model"]["name"], pretrained=False,
        num_classes=cfg["model"]["num_classes"],
        img_size=cfg["model"]["img_size"],
    )
    model.load_state_dict(state_dict)
    model.to(device)

    # Standard
    ds_std = CocoaDataset(splits["test"], clean_tf(cfg))
    loader_std = DataLoader(ds_std, batch_size=args.batch_size, shuffle=False,
                             num_workers=cfg["data"]["num_workers"], pin_memory=True)
    probs_std, y_true = predict(model, loader_std, device, amp_dtype)
    pred_std = probs_std.argmax(1)
    acc_std = (pred_std == y_true).mean()
    f1_std = f1_score(y_true, pred_std, average="macro")
    print(f"[eval] standard: acc={acc_std:.4f}  macro_f1={f1_std:.4f}")

    # TTA
    probs_tta = probs_std.copy()
    for r in range(args.tta - 1):
        random.seed(cfg["seed"] + r + 1)
        torch.manual_seed(cfg["seed"] + r + 1)
        ds = CocoaDataset(splits["test"], tta_tf(cfg))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=cfg["data"]["num_workers"], pin_memory=True)
        probs_r, _ = predict(model, loader, device, amp_dtype)
        probs_tta += probs_r
        print(f"[eval] tta pass {r+2}/{args.tta} done")
    probs_tta /= args.tta
    pred_tta = probs_tta.argmax(1)
    acc_tta = (pred_tta == y_true).mean()
    f1_tta = f1_score(y_true, pred_tta, average="macro")

    print(f"[eval] tta (x{args.tta}): acc={acc_tta:.4f}  macro_f1={f1_tta:.4f}")
    print(f"[eval] tta improvement: +{(acc_tta - acc_std) * 100:.2f}pp")

    report = classification_report(y_true, pred_tta, target_names=class_names, digits=4)
    (out_dir / "classification_report.txt").write_text(report)
    cm = confusion_matrix(y_true, pred_tta)
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    (out_dir / "metrics.json").write_text(json.dumps({
        "standard": {"acc": float(acc_std), "macro_f1": float(f1_std)},
        "tta": {"acc": float(acc_tta), "macro_f1": float(f1_tta), "rounds": args.tta},
        "improvement_pp": float((acc_tta - acc_std) * 100),
    }, indent=2))

    print(f"[eval] wrote classification_report.txt, confusion_matrix.csv, metrics.json "
          f"to {out_dir}")


if __name__ == "__main__":
    main()
