"""Build train/val/test splits for the merged LatAm + Peru cocoa-pod YOLO datasets.

Mirrors the Kaggle notebook (`5-cocoa-classes-pod-training`) crop pipeline:
  1. Read each YOLO label file (.txt).
  2. Convert normalized boxes to pixel coords (with optional context padding).
  3. Drop tiny boxes (< MIN_BOX_AREA / MIN_BOX_SIDE).
  4. Save crops as JPEGs grouped by unified class.
  5. Image-level stratified split (no leakage: crops from the same source image
     never appear in different splits).

Unified 5-class label space:
    carmenta, healthy, moniliasis, phytophthora, witches_broom

Output: splits.json with the SAME schema train.py / eval.py expect:
    {
      "seed": 123,
      "class_to_idx": {"carmenta": 0, ...},
      "num_classes": 5,
      "train": [{"path": "...", "label": "...", "label_idx": 0}, ...],
      "val":   [...],
      "test":  [...]
    }
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Crop hygiene — matches the notebook's pod-training settings.
MIN_BOX_AREA = 400
MIN_BOX_SIDE = 15
CONTEXT_PAD_FRAC = 0.05

# LatAm class IDs (from classes.txt): 0=Fitoftora, 1=Monilia, 2=Sana
LATAM_CLASS_MAP = {
    "0": "phytophthora",
    "1": "moniliasis",
    "2": "healthy",
    "Fitoftora": "phytophthora",
    "Monilia": "moniliasis",
    "Sana": "healthy",
}

# Peru class IDs (from notes.json): 0=Healthy 1=Carmenta 2=Witches 3=Moniliasis 4=Phytophthora
PERU_CLASS_MAP = {
    "0": "healthy",
    "1": "carmenta",
    "2": "witches_broom",
    "3": "moniliasis",
    "4": "phytophthora",
}

UNIFIED_CLASSES = [
    "carmenta",
    "healthy",
    "moniliasis",
    "phytophthora",
    "witches_broom",
]


# ============================================================
# YOLO crop helpers
# ============================================================
def parse_yolo_line(line: str):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    class_id = parts[0]
    try:
        cx, cy, bw, bh = (float(parts[1]), float(parts[2]),
                          float(parts[3]), float(parts[4]))
    except ValueError:
        return None
    return class_id, cx, cy, bw, bh


def yolo_to_pixel(cx, cy, bw, bh, img_w, img_h, pad_frac=CONTEXT_PAD_FRAC):
    x1 = (cx - bw / 2) * img_w
    y1 = (cy - bh / 2) * img_h
    x2 = (cx + bw / 2) * img_w
    y2 = (cy + bh / 2) * img_h
    box_w, box_h = x2 - x1, y2 - y1
    pad_x, pad_y = box_w * pad_frac, box_h * pad_frac
    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(img_w, int(x2 + pad_x))
    y2 = min(img_h, int(y2 + pad_y))
    return x1, y1, x2, y2


def crop_yolo_boxes(image_path: Path, label_path: Path, class_map: dict,
                    crop_dir: Path, source_name: str):
    """Return list of {path, label, source_image} dicts for valid boxes."""
    results = []
    if not label_path.exists():
        return results
    try:
        img = Image.open(image_path).convert("RGB")
        img_w, img_h = img.size
    except (UnidentifiedImageError, OSError, ValueError):
        return results

    lines = label_path.read_text().strip().split("\n")

    for i, line in enumerate(lines):
        parsed = parse_yolo_line(line)
        if parsed is None:
            continue
        class_id, cx, cy, bw, bh = parsed

        label = class_map.get(class_id)
        if label is None:
            continue

        x1, y1, x2, y2 = yolo_to_pixel(cx, cy, bw, bh, img_w, img_h)
        box_w, box_h = x2 - x1, y2 - y1
        if box_w < MIN_BOX_SIDE or box_h < MIN_BOX_SIDE:
            continue
        if box_w * box_h < MIN_BOX_AREA:
            continue

        crop = img.crop((x1, y1, x2, y2))
        label_dir = crop_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        crop_path = label_dir / f"{source_name}_{image_path.stem}_box{i}.jpg"
        crop.save(crop_path, quality=95)

        results.append({
            "path": str(crop_path),
            "label": label,
            "source_image": f"{source_name}_{image_path.name}",
        })

    return results


def load_latam(root: Path, crop_dir: Path) -> pd.DataFrame:
    """LatAm layout: <root>/<class_dir>/{*.jpg,*.txt}"""
    print(f"[latam] loading from {root}")
    rows, skipped = [], 0

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            txt_path = img_path.with_suffix(".txt")
            cropped = crop_yolo_boxes(img_path, txt_path, LATAM_CLASS_MAP,
                                       crop_dir, "latam")
            if cropped:
                rows.extend(cropped)
            else:
                skipped += 1

    df = pd.DataFrame(rows)
    print(f"[latam] {len(df):,} boxes  |  skipped {skipped:,} images")
    if len(df):
        print(df["label"].value_counts().sort_index().to_string(header=False))
    return df


def load_peru(images_dir: Path, labels_dir: Path,
              crop_dir: Path) -> pd.DataFrame:
    """Peru layout: flat images/ + flat labels/ folders."""
    print(f"[peru] loading from {images_dir}")
    rows, skipped = [], 0

    for txt_path in sorted(labels_dir.glob("*.txt")):
        if txt_path.name == "classes.txt":
            continue

        img_path = None
        for ext in IMAGE_EXTS:
            candidate = images_dir / f"{txt_path.stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            skipped += 1
            continue

        cropped = crop_yolo_boxes(img_path, txt_path, PERU_CLASS_MAP,
                                   crop_dir, "peru")
        if cropped:
            rows.extend(cropped)
        else:
            skipped += 1

    df = pd.DataFrame(rows)
    print(f"[peru] {len(df):,} boxes  |  skipped {skipped:,} images")
    if len(df):
        print(df["label"].value_counts().sort_index().to_string(header=False))
    return df


# ============================================================
# Image-level split
# ============================================================
def image_level_split(df: pd.DataFrame, train_size: float, val_size: float,
                      test_size: float, seed: int):
    """Split at the source-image level so crops from one photo don't leak."""
    assert abs(train_size + val_size + test_size - 1.0) < 1e-8

    # Each source image gets one majority label so we can stratify.
    image_labels = (
        df.groupby("source_image")["label"]
          .agg(lambda x: x.value_counts().index[0])
          .reset_index()
          .rename(columns={"label": "majority_label"})
    )

    train_imgs, temp_imgs = train_test_split(
        image_labels, train_size=train_size, random_state=seed,
        shuffle=True, stratify=image_labels["majority_label"],
    )
    rel_val = val_size / (val_size + test_size)
    val_imgs, test_imgs = train_test_split(
        temp_imgs, train_size=rel_val, random_state=seed,
        shuffle=True, stratify=temp_imgs["majority_label"],
    )

    train_df = df[df["source_image"].isin(train_imgs["source_image"])].reset_index(drop=True)
    val_df   = df[df["source_image"].isin(val_imgs["source_image"])].reset_index(drop=True)
    test_df  = df[df["source_image"].isin(test_imgs["source_image"])].reset_index(drop=True)

    # Leak guards
    assert not (set(train_df["source_image"]) & set(val_df["source_image"]))
    assert not (set(train_df["source_image"]) & set(test_df["source_image"]))
    assert not (set(val_df["source_image"]) & set(test_df["source_image"]))

    return train_df, val_df, test_df


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latam-root", required=True, type=Path,
                    help="Path to 'Enfermedades Cacao' folder (LatAm dataset)")
    ap.add_argument("--peru-images", required=True, type=Path,
                    help="Path to Peru images/ folder")
    ap.add_argument("--peru-labels", required=True, type=Path,
                    help="Path to Peru labels/ folder")
    ap.add_argument("--crop-dir", required=True, type=Path,
                    help="Where to write per-class JPEG crops")
    ap.add_argument("--out", type=Path, default=Path("splits.json"))
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--train-size", type=float, default=0.8)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--test-size", type=float, default=0.1)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    for p in (args.latam_root, args.peru_images, args.peru_labels):
        if not p.exists():
            raise SystemExit(f"path does not exist: {p}")

    # Fresh crop dir on every run (cheap; deterministic outputs)
    if args.crop_dir.exists():
        shutil.rmtree(args.crop_dir)
    args.crop_dir.mkdir(parents=True, exist_ok=True)

    latam_df = load_latam(args.latam_root, args.crop_dir)
    peru_df  = load_peru(args.peru_images, args.peru_labels, args.crop_dir)

    df = pd.concat([latam_df, peru_df], ignore_index=True)
    print(f"\n[merged] {len(df):,} crops from {df['source_image'].nunique():,} source images")
    print(df["label"].value_counts().sort_index().to_string(header=False))

    for cls in UNIFIED_CLASSES:
        if cls not in df["label"].values:
            print(f"[warn] class '{cls}' has zero samples!")

    train_df, val_df, test_df = image_level_split(
        df, args.train_size, args.val_size, args.test_size, args.seed,
    )

    print(f"\n[split] train {len(train_df):,}  val {len(val_df):,}  test {len(test_df):,}")
    print("[split] train class counts:")
    print(train_df["label"].value_counts().sort_index().to_string(header=False))

    class_to_idx = {c: i for i, c in enumerate(UNIFIED_CLASSES)}

    def to_records(d: pd.DataFrame):
        return [
            {"path": r.path, "label": r.label, "label_idx": class_to_idx[r.label]}
            for r in d.itertuples()
        ]

    payload = {
        "seed": args.seed,
        "class_to_idx": class_to_idx,
        "num_classes": len(UNIFIED_CLASSES),
        "train": to_records(train_df),
        "val":   to_records(val_df),
        "test":  to_records(test_df),
    }

    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\n[prepare] wrote {args.out}  (crops at {args.crop_dir})")


if __name__ == "__main__":
    main()
