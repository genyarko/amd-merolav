"""Generate vision-language QA pairs for Llama 3.2 Vision fine-tuning.

Reads the CCMT splits.json (from Track 2) and treatment_knowledge.json,
then generates diverse conversational QA pairs for each image.

Output: JSONL files (train.jsonl, val.jsonl) where each line is a
conversation with an image path + multi-turn QA.

Usage:
    python create_qa_dataset.py \
        --splits /data/ccmt/splits.json \
        --knowledge vision/data/treatment_knowledge.json \
        --image-root /data/ccmt \
        --output vision/data/qa_dataset \
        --samples-per-class 200
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# --------------- Question templates --------------- #
# Each template is (question, answer_builder) where answer_builder is a
# function name that constructs the answer from the knowledge entry.

DIAGNOSIS_QUESTIONS = [
    "What disease does this plant have?",
    "Can you identify what's wrong with this plant?",
    "What is affecting this plant's leaves?",
    "Diagnose this plant disease from the photo.",
    "What condition is this plant suffering from?",
    "Look at this leaf — what disease is it?",
    "I found this in my field. What's the problem?",
    "Is this plant diseased? If so, what is it?",
]

HEALTHY_DIAGNOSIS_QUESTIONS = [
    "Is this plant healthy?",
    "Does this plant have any disease?",
    "What's the condition of this plant?",
    "Can you check if this plant is healthy?",
    "Is there anything wrong with this leaf?",
    "Diagnose this plant from the photo.",
]

TREATMENT_QUESTIONS = [
    "How should I treat this?",
    "What treatment do you recommend?",
    "How do I manage this disease?",
    "What should I spray on this plant?",
    "How can I save this crop?",
    "What's the best way to control this?",
]

PREVENTION_QUESTIONS = [
    "How can I prevent this in the future?",
    "What should I do to avoid this disease next season?",
    "How do I keep my crops safe from this?",
    "What preventive measures work against this?",
]

SEVERITY_QUESTIONS = [
    "How bad is this infection?",
    "What's the severity level?",
    "Is this a serious case?",
    "How advanced is this disease?",
    "Should I be worried about this?",
]

CROP_ID_QUESTIONS = [
    "What crop is this?",
    "Can you identify this plant?",
    "What kind of plant am I looking at?",
]


# --------------- Answer builders --------------- #

def build_diagnosis_answer(k: dict, severity: str) -> str:
    """Build a natural diagnosis answer from the knowledge entry."""
    crop = k["crop"]
    disease = k["disease"]
    pathogen = k["pathogen"]
    symptoms = k["symptoms"]

    ans = f"This {crop.lower()} leaf shows signs of **{disease}**"
    if pathogen:
        ans += f", caused by *{pathogen}*"
    ans += f". {symptoms}"

    sev_desc = k["severity_cues"].get(severity, "")
    if sev_desc:
        ans += f" Based on what I can see, this appears to be a {severity} case: {sev_desc.lower()}"

    return ans


def build_healthy_answer(k: dict) -> str:
    crop = k["crop"]
    return (
        f"This {crop.lower()} plant appears **healthy**. {k['symptoms']} "
        f"Keep monitoring regularly and continue your current care routine."
    )


def build_treatment_answer(k: dict) -> str:
    disease = k["disease"]
    treatment = k["treatment"]
    return f"For {disease.lower()} in {k['crop'].lower()}: {treatment}"


def build_prevention_answer(k: dict) -> str:
    return f"To prevent {k['disease'].lower()}: {k['prevention']}"


def build_severity_answer(k: dict, severity: str) -> str:
    sev_desc = k["severity_cues"].get(severity, "")
    if severity == "mild":
        concern = "This looks like an early-stage infection. With prompt treatment, the plant should recover well."
    elif severity == "moderate":
        concern = "This is a moderate infection that needs attention soon. Treatment can still save the crop, but act quickly."
    else:
        concern = "This is a severe case. Immediate action is needed to prevent further spread and limit crop loss."
    return f"{concern} {sev_desc}"


def build_crop_id_answer(k: dict) -> str:
    return f"This is a **{k['crop'].lower()}** plant ({k['crop']})."


# --------------- Conversation generators --------------- #

def make_diagnosis_conversation(k: dict, image_path: str, is_healthy: bool) -> dict:
    """Single-turn: diagnosis only."""
    severity = random.choice(["mild", "moderate", "severe"])

    if is_healthy:
        q = random.choice(HEALTHY_DIAGNOSIS_QUESTIONS)
        a = build_healthy_answer(k)
    else:
        q = random.choice(DIAGNOSIS_QUESTIONS)
        a = build_diagnosis_answer(k, severity)

    return {
        "image": image_path,
        "conversations": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ],
    }


def make_diagnosis_treatment_conversation(k: dict, image_path: str) -> dict:
    """Two-turn: diagnosis → treatment."""
    severity = random.choice(["mild", "moderate", "severe"])
    q1 = random.choice(DIAGNOSIS_QUESTIONS)
    a1 = build_diagnosis_answer(k, severity)
    q2 = random.choice(TREATMENT_QUESTIONS)
    a2 = build_treatment_answer(k)

    return {
        "image": image_path,
        "conversations": [
            {"role": "user", "content": q1},
            {"role": "assistant", "content": a1},
            {"role": "user", "content": q2},
            {"role": "assistant", "content": a2},
        ],
    }


def make_full_conversation(k: dict, image_path: str) -> dict:
    """Three-turn: diagnosis → treatment → prevention."""
    severity = random.choice(["mild", "moderate", "severe"])
    q1 = random.choice(DIAGNOSIS_QUESTIONS)
    a1 = build_diagnosis_answer(k, severity)
    q2 = random.choice(TREATMENT_QUESTIONS)
    a2 = build_treatment_answer(k)
    q3 = random.choice(PREVENTION_QUESTIONS)
    a3 = build_prevention_answer(k)

    return {
        "image": image_path,
        "conversations": [
            {"role": "user", "content": q1},
            {"role": "assistant", "content": a1},
            {"role": "user", "content": q2},
            {"role": "assistant", "content": a2},
            {"role": "user", "content": q3},
            {"role": "assistant", "content": a3},
        ],
    }


def make_severity_conversation(k: dict, image_path: str) -> dict:
    """Two-turn: severity assessment → treatment."""
    severity = random.choice(["mild", "moderate", "severe"])
    q1 = random.choice(SEVERITY_QUESTIONS)
    a1 = build_severity_answer(k, severity)
    q2 = random.choice(TREATMENT_QUESTIONS)
    a2 = build_treatment_answer(k)

    return {
        "image": image_path,
        "conversations": [
            {"role": "user", "content": q1},
            {"role": "assistant", "content": a1},
            {"role": "user", "content": q2},
            {"role": "assistant", "content": a2},
        ],
    }


def make_crop_id_conversation(k: dict, image_path: str, is_healthy: bool) -> dict:
    """Two-turn: crop ID → health check."""
    q1 = random.choice(CROP_ID_QUESTIONS)
    a1 = build_crop_id_answer(k)

    if is_healthy:
        q2 = random.choice(HEALTHY_DIAGNOSIS_QUESTIONS)
        a2 = build_healthy_answer(k)
    else:
        q2 = random.choice(DIAGNOSIS_QUESTIONS)
        severity = random.choice(["mild", "moderate", "severe"])
        a2 = build_diagnosis_answer(k, severity)

    return {
        "image": image_path,
        "conversations": [
            {"role": "user", "content": q1},
            {"role": "assistant", "content": a1},
            {"role": "user", "content": q2},
            {"role": "assistant", "content": a2},
        ],
    }


# --------------- Main dataset builder --------------- #

DISEASED_GENERATORS = [
    make_diagnosis_conversation,
    make_diagnosis_treatment_conversation,
    make_full_conversation,
    make_severity_conversation,
]

HEALTHY_GENERATORS = [
    lambda k, p: make_diagnosis_conversation(k, p, is_healthy=True),
    lambda k, p: make_crop_id_conversation(k, p, is_healthy=True),
]


def generate_examples(
    records: list[dict],
    knowledge: dict,
    samples_per_class: int,
    seed: int,
) -> list[dict]:
    """Generate QA pairs from image records + treatment knowledge."""
    rng = random.Random(seed)

    # Group records by label
    by_class: dict[str, list[dict]] = {}
    for r in records:
        by_class.setdefault(r["label"], []).append(r)

    examples = []
    missing_classes = []

    for label, class_records in sorted(by_class.items()):
        if label not in knowledge:
            missing_classes.append(label)
            continue

        k = knowledge[label]
        is_healthy = k["disease"] == "Healthy"

        # Sample images (with replacement if needed)
        sampled = rng.choices(class_records, k=samples_per_class)

        for rec in sampled:
            if is_healthy:
                gen = rng.choice(HEALTHY_GENERATORS)
                ex = gen(k, rec["path"])
            else:
                gen = rng.choice(DISEASED_GENERATORS)
                if gen == make_diagnosis_conversation:
                    ex = gen(k, rec["path"], is_healthy=False)
                else:
                    ex = gen(k, rec["path"])
            ex["label"] = label
            examples.append(ex)

    if missing_classes:
        print(f"[WARNING] No knowledge entry for: {missing_classes}")

    rng.shuffle(examples)
    return examples


def main():
    ap = argparse.ArgumentParser(description="Generate VLM QA dataset from CCMT splits")
    ap.add_argument("--splits", required=True, type=Path,
                    help="Path to splits.json from Track 2")
    ap.add_argument("--knowledge", type=Path,
                    default=Path(__file__).parent / "treatment_knowledge.json",
                    help="Path to treatment_knowledge.json")
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).parent / "qa_dataset",
                    help="Output directory for JSONL files")
    ap.add_argument("--samples-per-class", type=int, default=200,
                    help="Number of QA examples to generate per class")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load inputs
    splits = json.loads(args.splits.read_text())
    knowledge = json.loads(args.knowledge.read_text())

    print(f"[create_qa] Loaded {len(splits['class_to_idx'])} classes from splits.json")
    print(f"[create_qa] Loaded {len(knowledge)} disease entries from knowledge")
    print(f"[create_qa] Generating {args.samples_per_class} examples per class")

    # Verify class coverage
    split_classes = set(splits["class_to_idx"].keys())
    knowledge_classes = set(knowledge.keys())
    missing = split_classes - knowledge_classes
    if missing:
        print(f"[WARNING] Classes in splits but not in knowledge: {missing}")
    extra = knowledge_classes - split_classes
    if extra:
        print(f"[INFO] Extra knowledge entries (not in splits): {extra}")

    # Generate for train and val splits
    args.output.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val"]:
        records = splits[split_name]
        examples = generate_examples(
            records, knowledge, args.samples_per_class, args.seed
        )
        out_path = args.output / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"[create_qa] {split_name}: {len(examples)} examples → {out_path}")

    # Summary stats
    print("\n[create_qa] Done! Dataset summary:")
    print(f"  Classes covered: {len(knowledge_classes & split_classes)}/{len(split_classes)}")
    print(f"  Samples per class: {args.samples_per_class}")
    print(f"  Total examples: ~{len(split_classes) * args.samples_per_class * 2} (train + val)")


if __name__ == "__main__":
    main()
