# ==========================================
# Continuation notebook (improved: Mixup + TTA)
# - Loads emergency_checkpoint.keras
# - Resumes fine-tuning with Mixup augmentation
# - Computes class weights dynamically from the training split
# - Uses cleaner evaluation generators
# - Uses grouped splitting heuristics to reduce augmentation leakage
# ==========================================

import math
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import random
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore")

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# -------------------------
# Config
# -------------------------
SEED = 123
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
RESUME_EPOCHS = 50
RESUME_LR = 1e-5
MIXUP_ALPHA = 0.2
TTA_ROUNDS = 10
LABEL_SMOOTHING = 0.1
MIN_CLASS_COUNT = 2
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1
USE_GROUPED_SPLIT = True

# UPDATE THIS to wherever you uploaded the checkpoint as a dataset
CHECKPOINT_PATH = (
    "/kaggle/input/datasets/merolavtechnology/"
    "mendeley-checkpoint-saved/emergency_checkpoint.keras"
)

AUGMENTED_DATASET = (
    "/kaggle/input/datasets/merolavtechnology/"
    "dataset-for-crop-pest-and-disease-detection/"
    "Dataset for Crop Pest and Disease Detection/"
    "CCMT Dataset-Augmented"
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# -------------------------
# Reproducibility
# -------------------------
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass


# -------------------------
# Helpers
# -------------------------
def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("___", "_")
    text = text.replace("__", "_")
    text = text.replace(" ", "_")
    text = text.replace("-", "_")
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def normalize_condition(condition: str) -> str:
    condition = normalize_text(condition)
    condition = re.sub(r"\d+$", "", condition)
    condition = re.sub(r"_+", "_", condition).strip("_")
    return condition


def is_valid_image(path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.convert("RGB")
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def extract_group_id(path_str: str) -> str:
    """
    Heuristic group id so likely augmented variants stay in the same split.
    """
    path = Path(path_str)
    stem = path.stem.lower()

    patterns = [
        r"(_aug\d+)$",
        r"(_copy\d+)$",
        r"(_flip)$",
        r"(_flipped)$",
        r"(_hflip)$",
        r"(_vflip)$",
        r"(_rot\d+)$",
        r"(_rotate[_-]?\d+)$",
        r"(_rotation[_-]?\d+)$",
        r"(_zoom\d*)$",
        r"(_shear\d*)$",
        r"(_shift\d*)$",
        r"(_bright\d*)$",
        r"(_brightness\d*)$",
        r"(_contrast\d*)$",
        r"(_blur\d*)$",
        r"(_noise\d*)$",
        r"(_crop\d*)$",
        r"(_enhanced\d*)$",
    ]

    base = stem
    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            new_base = re.sub(pattern, "", base)
            if new_base != base:
                base = new_base
                changed = True

    parent_parts = path.parts[-4:-1] if len(path.parts) >= 4 else path.parts[:-1]
    parent_context = "/".join([p.lower() for p in parent_parts])
    return f"{parent_context}/{base}"


def collect_augmented_images(root_dir: str) -> pd.DataFrame:
    root = Path(root_dir)
    if not root.exists():
        raise ValueError(f"Path not found: {root_dir}")

    rows = []
    bad_files = []

    for fp in root.rglob("*"):
        if not (fp.is_file() and fp.suffix.lower() in IMAGE_EXTS):
            continue

        if len(fp.parts) < 4:
            continue

        condition_folder = fp.parent.name
        split_folder = fp.parent.parent.name
        crop_folder = fp.parent.parent.parent.name

        if split_folder not in {"train_set", "test_set"}:
            continue

        crop = normalize_text(crop_folder)
        condition = normalize_condition(condition_folder)
        label = f"{crop}_{condition}"

        if is_valid_image(fp):
            rows.append(
                (
                    str(fp),
                    label,
                    crop,
                    condition,
                    split_folder,
                    extract_group_id(str(fp)),
                )
            )
        else:
            bad_files.append(str(fp))

    df = pd.DataFrame(
        rows,
        columns=[
            "filepaths",
            "labels",
            "crop",
            "condition",
            "split_source",
            "group_id",
        ],
    )
    df = df.drop_duplicates(subset=["filepaths"]).reset_index(drop=True)

    print(f"[INFO] Collected {len(df):,} valid images from augmented dataset")
    print(f"[INFO] Found {df['labels'].nunique()} classes")
    print(f"[INFO] Found {df['group_id'].nunique():,} estimated source groups")
    print(f"[INFO] Skipped {len(bad_files):,} invalid/corrupt files")

    return df


def filter_small_classes(df: pd.DataFrame, min_count: int = 2) -> pd.DataFrame:
    counts = df["labels"].value_counts()
    keep = counts[counts >= min_count].index
    removed = counts[counts < min_count]

    if len(removed) > 0:
        print(
            f"[INFO] Dropping {len(removed)} classes with fewer than {min_count} images"
        )

    return df[df["labels"].isin(keep)].reset_index(drop=True)


def stratified_split(
    df: pd.DataFrame,
    train_size: float = TRAIN_SIZE,
    val_size: float = VAL_SIZE,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
):
    assert abs(train_size + val_size + test_size - 1.0) < 1e-8

    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=seed,
        shuffle=True,
        stratify=df["labels"],
    )

    relative_val = val_size / (val_size + test_size)

    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val,
        random_state=seed,
        shuffle=True,
        stratify=temp_df["labels"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def grouped_split(
    df: pd.DataFrame,
    train_size: float = TRAIN_SIZE,
    val_size: float = VAL_SIZE,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
):
    """
    Group-aware split to reduce leakage from augmented variants.
    Falls back to normal stratified split if grouping is not feasible.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-8

    group_df = (
        df.groupby("group_id")
        .agg(
            labels=("labels", lambda x: x.iloc[0]),
            n_images=("filepaths", "count"),
        )
        .reset_index()
    )

    mixed_label_groups = df.groupby("group_id")["labels"].nunique()
    mixed_label_groups = mixed_label_groups[mixed_label_groups > 1]

    if len(mixed_label_groups) > 0:
        print(
            f"[WARN] {len(mixed_label_groups)} groups contain multiple labels. "
            "Falling back to standard stratified split."
        )
        return stratified_split(df, train_size, val_size, test_size, seed)

    label_counts = group_df["labels"].value_counts()
    if (label_counts < 2).any():
        print(
            "[WARN] Some labels have fewer than 2 groups. "
            "Falling back to standard stratified split."
        )
        return stratified_split(df, train_size, val_size, test_size, seed)

    try:
        gss1 = GroupShuffleSplit(
            n_splits=1,
            train_size=train_size,
            random_state=seed,
        )
        train_idx, temp_idx = next(
            gss1.split(group_df, y=group_df["labels"], groups=group_df["group_id"])
        )
        train_groups = set(group_df.iloc[train_idx]["group_id"])
        temp_groups = set(group_df.iloc[temp_idx]["group_id"])

        temp_group_df = group_df[group_df["group_id"].isin(temp_groups)].reset_index(
            drop=True
        )
        relative_val = val_size / (val_size + test_size)

        gss2 = GroupShuffleSplit(
            n_splits=1,
            train_size=relative_val,
            random_state=seed,
        )
        val_idx, test_idx = next(
            gss2.split(
                temp_group_df,
                y=temp_group_df["labels"],
                groups=temp_group_df["group_id"],
            )
        )
        val_groups = set(temp_group_df.iloc[val_idx]["group_id"])
        test_groups = set(temp_group_df.iloc[test_idx]["group_id"])

        train_df = df[df["group_id"].isin(train_groups)].reset_index(drop=True)
        val_df = df[df["group_id"].isin(val_groups)].reset_index(drop=True)
        test_df = df[df["group_id"].isin(test_groups)].reset_index(drop=True)

        all_labels = set(df["labels"].unique())
        split_labels = (
            set(train_df["labels"].unique())
            & set(val_df["labels"].unique())
            & set(test_df["labels"].unique())
        )

        if all_labels != split_labels:
            missing = sorted(all_labels - split_labels)
            print(
                "[WARN] Grouped split caused some classes to be absent from at least "
                f"one split: {missing}. Falling back to standard stratified split."
            )
            return stratified_split(df, train_size, val_size, test_size, seed)

        return train_df, val_df, test_df

    except Exception as e:
        print(f"[WARN] Grouped split failed ({e}). Falling back to standard stratified split.")
        return stratified_split(df, train_size, val_size, test_size, seed)


def plot_history(history):
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.title("Loss (Resumed Training)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.title("Accuracy (Resumed Training)")
    plt.legend()

    plt.tight_layout()
    plt.show()


def compute_class_weights_from_df(train_df: pd.DataFrame, class_indices: dict) -> dict:
    """
    Balanced class weights:
        weight_c = N / (K * n_c)
    """
    counts = train_df["labels"].value_counts().to_dict()
    total = len(train_df)
    num_classes = len(class_indices)

    class_weight = {}
    for label, idx in class_indices.items():
        count = counts.get(label, 0)
        if count > 0:
            class_weight[idx] = total / (num_classes * count)

    return class_weight


def build_generator(
    df: pd.DataFrame,
    datagen: ImageDataGenerator,
    img_size,
    batch_size,
    shuffle,
    seed=None,
):
    return datagen.flow_from_dataframe(
        df,
        x_col="filepaths",
        y_col="labels",
        target_size=img_size,
        class_mode="categorical",
        color_mode="rgb",
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        validate_filenames=True,
    )


# -------------------------
# Mixup generator with baked-in class weights
# -------------------------
def mixup_generator(generator, class_weight_dict, alpha=MIXUP_ALPHA):
    """
    Mixes examples within a single batch using a random permutation.
    Class weights are baked into sample_weights since Keras does not
    support class_weight with Python generators.
    """
    while True:
        x, y = next(generator)
        batch_len = len(x)

        if batch_len == 0:
            continue

        indices = np.random.permutation(batch_len)
        x2 = x[indices]
        y2 = y[indices]

        lam = np.random.beta(alpha, alpha, size=(batch_len, 1, 1, 1)).astype(np.float32)
        lam_y = lam.reshape(batch_len, 1)

        x_mixed = lam * x + (1.0 - lam) * x2
        y_mixed = lam_y * y + (1.0 - lam_y) * y2

        w1 = np.array(
            [class_weight_dict.get(int(np.argmax(row)), 1.0) for row in y],
            dtype=np.float32,
        )
        w2 = np.array(
            [class_weight_dict.get(int(np.argmax(row)), 1.0) for row in y2],
            dtype=np.float32,
        )
        sample_weights = lam_y.reshape(-1).astype(np.float32) * w1 + (
            1.0 - lam_y.reshape(-1).astype(np.float32)
        ) * w2

        yield x_mixed, y_mixed, sample_weights


# -------------------------
# Test-Time Augmentation
# -------------------------
def predict_with_tta(
    model,
    df: pd.DataFrame,
    img_size,
    num_rounds=TTA_ROUNDS,
    batch_size=BATCH_SIZE,
):
    """
    Run multiple passes and average predictions.
    Round 1 is clean; remaining rounds use mild augmentations.
    """
    print(f"\n[TTA] Running {num_rounds} passes...")

    clean_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )
    clean_gen = build_generator(
        df=df,
        datagen=clean_datagen,
        img_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )

    all_preds = model.predict(clean_gen, verbose=0)
    print(f"  Pass 1/{num_rounds} (clean) done")

    if num_rounds > 1:
        tta_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=8,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.08,
            horizontal_flip=True,
            fill_mode="nearest",
        )

        for r in range(1, num_rounds):
            aug_gen = build_generator(
                df=df,
                datagen=tta_datagen,
                img_size=img_size,
                batch_size=batch_size,
                shuffle=False,
                seed=SEED + r,
            )
            preds = model.predict(aug_gen, verbose=0)
            all_preds += preds
            print(f"  Pass {r + 1}/{num_rounds} done")

    avg_preds = all_preds / num_rounds
    return avg_preds, clean_gen.classes


def summarize_predictions(y_true, pred_probs, target_names, title="Report"):
    y_pred = np.argmax(pred_probs, axis=1)
    acc = np.mean(y_pred == y_true)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"\n{title}")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "y_pred": y_pred,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


# -------------------------
# Load dataset
# -------------------------
full_df = collect_augmented_images(AUGMENTED_DATASET)
full_df = filter_small_classes(full_df, min_count=MIN_CLASS_COUNT)

print(f"\n[INFO] Total images: {len(full_df):,} | Classes: {full_df['labels'].nunique()}")

# -------------------------
# Split
# -------------------------
if USE_GROUPED_SPLIT:
    print("\n[INFO] Using grouped split to reduce augmentation leakage")
    train_df, val_df, test_df = grouped_split(
        full_df,
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        seed=SEED,
    )
else:
    train_df, val_df, test_df = stratified_split(
        full_df,
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        seed=SEED,
    )

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# -------------------------
# Generators
# -------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=12,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.10,
    shear_range=0.08,
    horizontal_flip=True,
    fill_mode="nearest",
)

clean_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

train_gen = build_generator(
    df=train_df,
    datagen=train_datagen,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

train_eval_gen = build_generator(
    df=train_df,
    datagen=clean_datagen,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

val_gen = build_generator(
    df=val_df,
    datagen=clean_datagen,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_gen = build_generator(
    df=test_df,
    datagen=clean_datagen,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# -------------------------
# Class weights
# -------------------------
class_weight = compute_class_weights_from_df(train_df, train_gen.class_indices)

print("\n[INFO] Dynamic class weights:")
for class_name, class_idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1]):
    print(f"  {class_name:<30} -> {class_weight.get(class_idx, 1.0):.6f}")

# -------------------------
# Load checkpoint and resume with Mixup
# -------------------------
print(f"\n[INFO] Loading checkpoint: {CHECKPOINT_PATH}")
model = tf.keras.models.load_model(CHECKPOINT_PATH)
model.summary()

# Verify checkpoint accuracy
print("\n[INFO] Verifying checkpoint accuracy...")
val_check = model.evaluate(val_gen, verbose=1)
print(f"Checkpoint Val Loss: {val_check[0]:.4f} | Val Acc: {val_check[1]:.4f}")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1,
        min_lr=1e-7,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "/kaggle/working/best_model.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
]

model.compile(
    optimizer=Adam(learning_rate=RESUME_LR),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=["accuracy"],
)

mixup_gen = mixup_generator(train_gen, class_weight, alpha=MIXUP_ALPHA)
steps_per_epoch = math.ceil(len(train_df) / BATCH_SIZE)

print(
    f"\n[INFO] Resuming with Mixup (alpha={MIXUP_ALPHA}) "
    f"for {RESUME_EPOCHS} epochs at LR={RESUME_LR}"
)
history_resumed = model.fit(
    mixup_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=RESUME_EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1,
)

# -------------------------
# Plot resumed training
# -------------------------
plot_history(history_resumed)

# -------------------------
# Evaluation: Standard (no TTA)
# -------------------------
train_score = model.evaluate(train_eval_gen, verbose=1)
val_score = model.evaluate(val_gen, verbose=1)
test_score = model.evaluate(test_gen, verbose=1)

print("\n--- Standard Evaluation ---")
print(f"Train Loss: {train_score[0]:.4f} | Train Acc: {train_score[1]:.4f}")
print(f"Val Loss:   {val_score[0]:.4f} | Val Acc:   {val_score[1]:.4f}")
print(f"Test Loss:  {test_score[0]:.4f} | Test Acc:  {test_score[1]:.4f}")

idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

pred_probs_std = model.predict(test_gen, verbose=1)
y_pred_std = np.argmax(pred_probs_std, axis=1)
y_true = test_gen.classes

print("\nClassification Report (Standard)")
print(classification_report(y_true, y_pred_std, target_names=target_names, digits=4))

# -------------------------
# Evaluation: With TTA
# -------------------------
# First check if TTA helps on validation
pred_probs_val_tta, y_true_val_tta = predict_with_tta(
    model,
    val_df,
    IMG_SIZE,
    num_rounds=TTA_ROUNDS,
    batch_size=BATCH_SIZE,
)
val_tta_metrics = summarize_predictions(
    y_true_val_tta,
    pred_probs_val_tta,
    target_names,
    title=f"Classification Report (Val, TTA x{TTA_ROUNDS})",
)

pred_probs_tta, y_true_tta = predict_with_tta(
    model,
    test_df,
    IMG_SIZE,
    num_rounds=TTA_ROUNDS,
    batch_size=BATCH_SIZE,
)
y_pred_tta = np.argmax(pred_probs_tta, axis=1)

tta_acc = np.mean(y_pred_tta == y_true_tta)
print(f"\n--- TTA Evaluation ({TTA_ROUNDS} rounds) ---")
print(f"Test Acc (TTA): {tta_acc:.4f}")
print(f"Improvement:    {(tta_acc - test_score[1]) * 100:+.2f}%")

print("\nClassification Report (TTA)")
print(classification_report(y_true_tta, y_pred_tta, target_names=target_names, digits=4))

cm = confusion_matrix(y_true_tta, y_pred_tta)
print("Confusion Matrix shape:", cm.shape)

# -------------------------
# Save final model
# -------------------------
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
num_classes = len(class_names)

subject = "crop_pest_disease_b0_22class_mixup_tta"
best_val_acc = max(history_resumed.history.get("val_accuracy", [val_score[1]]))

model_path = f"/kaggle/working/efficientnetb0_{subject}_bestval_{best_val_acc * 100:.2f}.keras"
weights_path = f"/kaggle/working/efficientnetb0_{subject}.weights.h5"
labels_path = f"/kaggle/working/{subject}_labels.txt"
class_csv_path = f"/kaggle/working/{subject}_class_dict.csv"
metadata_path = f"/kaggle/working/{subject}_metadata.json"

model.save(model_path)
model.save_weights(weights_path)

with open(labels_path, "w") as f:
    for name in class_names:
        f.write(name + "\n")

class_df = pd.DataFrame(
    {
        "class_index": list(range(len(class_names))),
        "class": class_names,
        "height": [IMG_SIZE[0]] * len(class_names),
        "width": [IMG_SIZE[1]] * len(class_names),
    }
)
class_df.to_csv(class_csv_path, index=False)

metadata = {
    "seed": SEED,
    "img_size": list(IMG_SIZE),
    "batch_size": BATCH_SIZE,
    "resume_epochs": RESUME_EPOCHS,
    "resume_lr": RESUME_LR,
    "mixup_alpha": MIXUP_ALPHA,
    "tta_rounds": TTA_ROUNDS,
    "label_smoothing": LABEL_SMOOTHING,
    "use_grouped_split": USE_GROUPED_SPLIT,
    "num_classes": num_classes,
    "class_names": class_names,
    "best_val_accuracy": float(best_val_acc),
    "standard_test_accuracy": float(test_score[1]),
    "tta_test_accuracy": float(tta_acc),
}
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved:")
print("Model ->", model_path)
print("Weights ->", weights_path)
print("Labels ->", labels_path)
print("Classes ->", class_csv_path)
print("Metadata ->", metadata_path)

# -------------------------
# TFLite conversion
# -------------------------
def representative_dataset_gen():
    num_calibration_batches = min(100, math.ceil(len(train_df) / BATCH_SIZE))
    rep_gen = build_generator(
        df=train_df,
        datagen=clean_datagen,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    for i, (batch_images, _) in enumerate(rep_gen):
        if i >= num_calibration_batches:
            break
        for img in batch_images:
            yield [np.expand_dims(img, axis=0).astype(np.float32)]


artifact_tag = f"{subject}_bestval_{best_val_acc * 100:.2f}"

# Float32
tflite_f32_path = f"/kaggle/working/efficientnetb0_{artifact_tag}_float32.tflite"
converter_f32 = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_f32 = converter_f32.convert()
with open(tflite_f32_path, "wb") as f:
    f.write(tflite_f32)
print(f"\n[TFLite] float32  -> {tflite_f32_path}  ({len(tflite_f32) / 1e6:.1f} MB)")

# Float16
tflite_f16_path = f"/kaggle/working/efficientnetb0_{artifact_tag}_float16.tflite"
converter_f16 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_f16.optimizations = [tf.lite.Optimize.DEFAULT]
converter_f16.target_spec.supported_types = [tf.float16]
tflite_f16 = converter_f16.convert()
with open(tflite_f16_path, "wb") as f:
    f.write(tflite_f16)
print(f"[TFLite] float16  -> {tflite_f16_path}  ({len(tflite_f16) / 1e6:.1f} MB)")

# Dynamic range
tflite_dyn_path = f"/kaggle/working/efficientnetb0_{artifact_tag}_dynamic.tflite"
converter_dyn = tf.lite.TFLiteConverter.from_keras_model(model)
converter_dyn.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_dyn = converter_dyn.convert()
with open(tflite_dyn_path, "wb") as f:
    f.write(tflite_dyn)
print(f"[TFLite] dynamic  -> {tflite_dyn_path}  ({len(tflite_dyn) / 1e6:.1f} MB)")

# Int8
tflite_int8_path = f"/kaggle/working/efficientnetb0_{artifact_tag}_int8.tflite"
converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset_gen
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.uint8
converter_int8.inference_output_type = tf.uint8
tflite_int8 = converter_int8.convert()
with open(tflite_int8_path, "wb") as f:
    f.write(tflite_int8)
print(f"[TFLite] int8     -> {tflite_int8_path}  ({len(tflite_int8) / 1e6:.1f} MB)")

# Labels JSON
labels_json_path = f"/kaggle/working/{subject}_labels.json"
labels_meta = {
    "model": "EfficientNetB0",
    "input_size": list(IMG_SIZE),
    "num_classes": num_classes,
    "class_names": class_names,
    "class_index_map": {str(i): name for i, name in enumerate(class_names)},
}
with open(labels_json_path, "w") as f:
    json.dump(labels_meta, f, indent=2)
print(f"[TFLite] labels   -> {labels_json_path}")

# Verify float16
print("\n[TFLite] Verifying float16 model...")
interpreter = tf.lite.Interpreter(model_path=tflite_f16_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

sample_batch, _ = next(iter(test_gen))
sample_img = np.expand_dims(sample_batch[0], axis=0).astype(np.float32)

interpreter.set_tensor(input_details[0]["index"], sample_img)
interpreter.invoke()
tflite_pred = interpreter.get_tensor(output_details[0]["index"])
tflite_class = int(np.argmax(tflite_pred, axis=1)[0])

keras_pred = model.predict(sample_img, verbose=0)
keras_class = int(np.argmax(keras_pred, axis=1)[0])

print(f"  Keras:  {class_names[keras_class]} ({keras_pred[0][keras_class]:.4f})")
print(f"  TFLite: {class_names[tflite_class]} ({tflite_pred[0][tflite_class]:.4f})")
print(f"  Match: {keras_class == tflite_class}")

print("\nDone! All files saved to /kaggle/working/")

# Also save the current state (may be slightly ahead of best_model)
model.save("/kaggle/working/latest_model.keras")

# Download links
from IPython.display import FileLink, display
display(FileLink("best_model.keras"))
display(FileLink("latest_model.keras"))

import json
import numpy as np
import tensorflow as tf
from IPython.display import FileLink, display

# Save current model
model.save("/kaggle/working/latest_model.keras")

# Class names from the generator
idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

# Float16 (best accuracy-size tradeoff for mobile)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
with open("/kaggle/working/model_float16.tflite", "wb") as f:
    f.write(tflite_model)
print(f"Float16: {len(tflite_model) / 1e6:.1f} MB")

# Int8 (smallest, fastest on mobile NPU)
def representative_dataset_gen():
    for i, (batch, _) in enumerate(train_gen):
        if i >= 100:
            break
        for img in batch:
            yield [np.expand_dims(img, axis=0).astype(np.float32)]

converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset_gen
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.uint8
converter_int8.inference_output_type = tf.uint8
tflite_int8 = converter_int8.convert()
with open("/kaggle/working/model_int8.tflite", "wb") as f:
    f.write(tflite_int8)
print(f"Int8:    {len(tflite_int8) / 1e6:.1f} MB")

# Labels JSON
labels_meta = {
    "model": "EfficientNetB0",
    "input_size": [224, 224],
    "num_classes": len(class_names),
    "class_names": class_names,
    "class_index_map": {str(i): name for i, name in enumerate(class_names)},
}
with open("/kaggle/working/labels.json", "w") as f:
    json.dump(labels_meta, f, indent=2)

# Download links
print("\nDownload:")
display(FileLink("latest_model.keras"))
display(FileLink("best_model.keras"))
display(FileLink("model_float16.tflite"))
display(FileLink("labels.json"))