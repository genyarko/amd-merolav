"""Build the train/val/test split for the CCMT crop-disease dataset.

This mirrors the Kaggle notebook's `grouped_split` logic (seed=123, 80/10/10)
so accuracy numbers on the MI300X are directly comparable to the P100 baseline.

Output: splits.json with schema
    {
      "class_to_idx": {"cashew_anthracnose": 0, ...},
      "train": [{"path": "...", "label": "cashew_anthracnose", "label_idx": 0}, ...],
      "val":   [...],
      "test":  [...]
    }
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import GroupShuffleSplit, train_test_split

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Augmentation-marker suffixes the Kaggle notebook uses to recover the
# "original" image identity so augmented copies stay in the same split.
AUG_SUFFIX_PATTERNS = [
    r"(_aug\d*)$", r"(_copy\d*)$", r"(_flip(ped)?)$", r"(_[hv]flip)$",
    r"(_rot(at(e|ion))?[_-]?\d*)$", r"(_zoom\d*)$", r"(_shear\d*)$",
    r"(_shift\d*)$", r"(_bright(ness)?\d*)$", r"(_contrast\d*)$",
    r"(_blur\d*)$", r"(_noise\d*)$", r"(_crop\d*)$", r"(_enhanced?\d*)$",
    r"(_mirror(ed)?)$", r"(_distort(ed)?\d*)$", r"(_warp(ed)?\d*)$",
    r"(_jitter\d*)$", r"(_transform(ed)?\d*)$",
    r"[\s_]\(\d+\)$", r"_\d{1,4}$",
]


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("___", "_").replace("__", "_")
    s = s.replace(" ", "_").replace("-", "_")
    return re.sub(r"_+", "_", s).strip("_")


def normalize_condition(s: str) -> str:
    s = normalize_text(s)
    s = re.sub(r"\d+$", "", s)
    return re.sub(r"_+", "_", s).strip("_")


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.convert("RGB")
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def extract_group_id(path: Path) -> str:
    """Strip augmentation suffixes from the filename stem so augmented copies
    of the same source image share a group id."""
    base = path.stem.lower()
    changed = True
    while changed:
        changed = False
        for pat in AUG_SUFFIX_PATTERNS:
            new = re.sub(pat, "", base)
            if new != base:
                base = new
                changed = True
    parent = "/".join(p.lower() for p in path.parts[-4:-1])
    return f"{parent}/{base}"


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for fp in root.rglob("*"):
        if not (fp.is_file() and fp.suffix.lower() in IMAGE_EXTS):
            continue
        if len(fp.parts) < 4:
            continue
        condition = fp.parent.name
        split = fp.parent.parent.name
        crop = fp.parent.parent.parent.name
        if split not in {"train_set", "test_set"}:
            continue
        if not is_valid_image(fp):
            continue
        label = f"{normalize_text(crop)}_{normalize_condition(condition)}"
        rows.append({
            "path": str(fp),
            "label": label,
            "group_id": extract_group_id(fp),
        })
    df = pd.DataFrame(rows).drop_duplicates(subset=["path"]).reset_index(drop=True)
    print(f"[prepare] {len(df):,} valid images  |  {df['label'].nunique()} classes")
    return df


def filter_small_classes(df: pd.DataFrame, min_count: int = 2) -> pd.DataFrame:
    counts = df["label"].value_counts()
    keep = counts[counts >= min_count].index
    dropped = set(counts.index) - set(keep)
    if dropped:
        print(f"[prepare] dropping {len(dropped)} classes with <{min_count} images")
    return df[df["label"].isin(keep)].reset_index(drop=True)


def grouped_split(df: pd.DataFrame, seed: int, train_size: float, val_size: float,
                  test_size: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by group_id so augmented copies of one source image don't leak
    across splits. Falls back to stratified if groups would break class coverage.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6
    group_labels = df.groupby("group_id")["label"].agg(lambda x: x.iloc[0])
    mixed_groups = df.groupby("group_id")["label"].nunique()

    if (mixed_groups > 1).any() or (group_labels.value_counts() < 2).any():
        print("[prepare] group integrity broken; using stratified split")
        return stratified_split(df, seed, train_size, val_size, test_size)

    group_df = group_labels.reset_index().rename(columns={"label": "label"})
    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idx, temp_idx = next(gss1.split(group_df, y=group_df["label"],
                                           groups=group_df["group_id"]))
    train_groups = set(group_df.iloc[train_idx]["group_id"])
    temp = group_df[~group_df["group_id"].isin(train_groups)].reset_index(drop=True)

    rel_val = val_size / (val_size + test_size)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=rel_val, random_state=seed)
    v_idx, t_idx = next(gss2.split(temp, y=temp["label"], groups=temp["group_id"]))
    val_groups = set(temp.iloc[v_idx]["group_id"])
    test_groups = set(temp.iloc[t_idx]["group_id"])

    train = df[df["group_id"].isin(train_groups)].reset_index(drop=True)
    val = df[df["group_id"].isin(val_groups)].reset_index(drop=True)
    test = df[df["group_id"].isin(test_groups)].reset_index(drop=True)

    all_labels = set(df["label"])
    if all_labels != (set(train["label"]) & set(val["label"]) & set(test["label"])):
        print("[prepare] grouped split dropped classes; falling back to stratified")
        return stratified_split(df, seed, train_size, val_size, test_size)

    return train, val, test


def stratified_split(df, seed, train_size, val_size, test_size):
    train, temp = train_test_split(df, train_size=train_size, random_state=seed,
                                   stratify=df["label"])
    rel_val = val_size / (val_size + test_size)
    val, test = train_test_split(temp, train_size=rel_val, random_state=seed,
                                 stratify=temp["label"])
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path,
                    help="Path to 'CCMT Dataset-Augmented' folder")
    ap.add_argument("--out", type=Path, default=Path("splits.json"))
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--train-size", type=float, default=0.8)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--test-size", type=float, default=0.1)
    ap.add_argument("--min-class-count", type=int, default=2)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not args.data_root.exists():
        raise SystemExit(f"data-root does not exist: {args.data_root}")

    df = collect(args.data_root)
    df = filter_small_classes(df, args.min_class_count)

    train, val, test = grouped_split(df, args.seed, args.train_size,
                                      args.val_size, args.test_size)

    classes = sorted(df["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    def to_records(d):
        return [
            {"path": r.path, "label": r.label, "label_idx": class_to_idx[r.label]}
            for r in d.itertuples()
        ]

    payload = {
        "seed": args.seed,
        "class_to_idx": class_to_idx,
        "num_classes": len(classes),
        "train": to_records(train),
        "val": to_records(val),
        "test": to_records(test),
    }

    args.out.write_text(json.dumps(payload, indent=2))
    print(f"[prepare] train {len(train):,}  val {len(val):,}  test {len(test):,}")
    print(f"[prepare] wrote {args.out}")


if __name__ == "__main__":
    main()
