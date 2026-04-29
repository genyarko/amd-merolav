"""Build train/val/test splits for the Amini cocoa contamination dataset.

Mirrors the Kaggle `amini-cocoa-contamination-dataset-based-1` notebook crop
hygiene, but stores BOX METADATA (source image + pixel coords) in splits.json
instead of pre-cropping. The on-the-fly crop in train.py randomizes the
context-pad in [pad_min, pad_max], which the notebooks couldn't do because
they wrote a single fixed-pad crop to disk.

Pipeline:
  1. Read Train.csv (columns: Image_ID, ImagePath, class, xmin, ymin, xmax, ymax).
  2. Drop rows whose bbox is too small (MIN_BOX_AREA / MIN_BOX_SIDE) or whose
     source image fails to open.
  3. Image-level stratified 80/10/10 split (no leakage — boxes from one source
     image never appear in different splits).
  4. Write splits.json with the schema train.py / eval.py expect.

Output schema:
    {
      "seed": 123,
      "class_to_idx": {"anthracnose": 0, "cssvd": 1, "healthy": 2},
      "num_classes": 3,
      "train": [{"image": "/abs/path.jpg",
                 "xmin": 10, "ymin": 20, "xmax": 200, "ymax": 300,
                 "label": "cssvd", "label_idx": 1,
                 "source_image": "abc.jpg"}, ...],
      "val":   [...],
      "test":  [...]
    }
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

# Crop hygiene — matches the v2 notebook (`amini-cocoa-contamination-dataset-based-1`).
MIN_BOX_AREA = 1000
MIN_BOX_SIDE = 20

CLASSES = ["anthracnose", "cssvd", "healthy"]


def normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def filter_boxes(df: pd.DataFrame, dataset_root: Path) -> pd.DataFrame:
    """Drop rows that fail the box-size or image-open check.

    Records the absolute path on disk so train.py doesn't need to know about
    the staging layout.
    """
    rows = []
    skipped_size = 0
    skipped_open = 0

    # itertuples mangles Python keywords like ``class`` — access via the
    # already-renamed ``class_name`` column.
    for r in df.itertuples():
        img_path = (dataset_root / r.ImagePath).resolve()

        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except (UnidentifiedImageError, OSError, ValueError, FileNotFoundError):
            skipped_open += 1
            continue

        x1 = max(0, int(r.xmin))
        y1 = max(0, int(r.ymin))
        x2 = min(w, int(r.xmax))
        y2 = min(h, int(r.ymax))

        bw, bh = x2 - x1, y2 - y1
        if bw < MIN_BOX_SIDE or bh < MIN_BOX_SIDE or bw * bh < MIN_BOX_AREA:
            skipped_size += 1
            continue

        rows.append({
            "image": str(img_path),
            "xmin": x1,
            "ymin": y1,
            "xmax": x2,
            "ymax": y2,
            "label": r.class_name,
            "source_image": r.Image_ID,
        })

    print(f"[filter] kept {len(rows):,}  skipped_size={skipped_size:,}  "
          f"skipped_open={skipped_open:,}")
    return pd.DataFrame(rows)


def image_level_split(df: pd.DataFrame, train_size: float, val_size: float,
                      test_size: float, seed: int):
    """Split at the source-image level so crops from one photo don't leak."""
    assert abs(train_size + val_size + test_size - 1.0) < 1e-8

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

    assert not (set(train_df["source_image"]) & set(val_df["source_image"]))
    assert not (set(train_df["source_image"]) & set(test_df["source_image"]))
    assert not (set(val_df["source_image"]) & set(test_df["source_image"]))

    return train_df, val_df, test_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, type=Path,
                    help="Path to the unpacked Kaggle dataset (contains Train.csv)")
    ap.add_argument("--csv", default="Train.csv",
                    help="CSV filename inside --dataset-root")
    ap.add_argument("--out", type=Path, default=Path("splits.json"))
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--train-size", type=float, default=0.8)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--test-size", type=float, default=0.1)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not args.dataset_root.exists():
        raise SystemExit(f"dataset root not found: {args.dataset_root}")

    csv_path = args.dataset_root / args.csv
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    raw_df = pd.read_csv(csv_path)
    # Rename ``class`` (Python keyword — itertuples mangles it) to ``class_name``.
    raw_df = raw_df.rename(columns={"class": "class_name"})
    raw_df["class_name"] = raw_df["class_name"].map(normalize_label)

    print(f"[load] {csv_path.name}: {len(raw_df):,} rows  "
          f"unique_images={raw_df['Image_ID'].nunique():,}  "
          f"classes={sorted(raw_df['class_name'].unique())}")

    unknown = set(raw_df["class_name"]) - set(CLASSES)
    if unknown:
        raise SystemExit(f"unknown classes in CSV: {unknown}  expected={CLASSES}")

    df = filter_boxes(raw_df, args.dataset_root)
    print(f"[filter] class counts:")
    print(df["label"].value_counts().sort_index().to_string(header=False))

    train_df, val_df, test_df = image_level_split(
        df, args.train_size, args.val_size, args.test_size, args.seed,
    )
    print(f"\n[split] train {len(train_df):,}  val {len(val_df):,}  test {len(test_df):,}")
    print("[split] train class counts:")
    print(train_df["label"].value_counts().sort_index().to_string(header=False))

    class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    def to_records(d: pd.DataFrame):
        return [
            {
                "image": r.image,
                "xmin": int(r.xmin),
                "ymin": int(r.ymin),
                "xmax": int(r.xmax),
                "ymax": int(r.ymax),
                "label": r.label,
                "label_idx": class_to_idx[r.label],
                "source_image": r.source_image,
            }
            for r in d.itertuples()
        ]

    payload = {
        "seed": args.seed,
        "class_to_idx": class_to_idx,
        "num_classes": len(CLASSES),
        "train": to_records(train_df),
        "val":   to_records(val_df),
        "test":  to_records(test_df),
    }

    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\n[prepare] wrote {args.out}  ({len(df):,} boxes total)")


if __name__ == "__main__":
    main()
