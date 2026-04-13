"""Fine-tune Llama 3.2 Vision 11B on the plant-disease QA dataset with LoRA.

Uses HuggingFace transformers + peft on MI300X (ROCm, bf16).

Run:
    python vision/finetune_vlm.py \
        --config vision/config.yaml \
        --train-data vision/data/qa_dataset/train.jsonl \
        --val-data vision/data/qa_dataset/val.jsonl \
        --output runs/llama_vision_v1
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------- Dataset --------------- #

class VLMDataset(Dataset):
    """Loads JSONL QA examples and formats them for Llama 3.2 Vision."""

    def __init__(self, jsonl_path: Path, processor, max_length: int = 2048):
        self.examples = []
        with open(jsonl_path) as f:
            for line in f:
                self.examples.append(json.loads(line))
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        image = Image.open(ex["image"]).convert("RGB")
        convos = ex["conversations"]

        # Build the chat messages with the image in the first user turn
        messages = []
        for i, turn in enumerate(convos):
            if turn["role"] == "user" and i == 0:
                # First user message includes the image
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": turn["content"]},
                    ],
                })
            else:
                messages.append({
                    "role": turn["role"],
                    "content": [{"type": "text", "text": turn["content"]}],
                })

        # Use the processor's chat template to format
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Process image + text together
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        # Squeeze batch dim (processor returns [1, ...])
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0)

        # Labels = input_ids, but mask user/system tokens with -100
        # We only train on assistant responses
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # Mask everything before and including each assistant header,
        # keep only the assistant's actual response tokens
        labels = self._mask_non_assistant_tokens(input_ids, labels)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }

        # Include aspect_ratio_ids and cross_attention_mask if present
        if "aspect_ratio_ids" in inputs:
            result["aspect_ratio_ids"] = inputs["aspect_ratio_ids"].squeeze(0)
        if "aspect_ratio_mask" in inputs:
            result["aspect_ratio_mask"] = inputs["aspect_ratio_mask"].squeeze(0)
        if "cross_attention_mask" in inputs:
            result["cross_attention_mask"] = inputs["cross_attention_mask"].squeeze(0)

        return result

    def _mask_non_assistant_tokens(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Mask all tokens that are not part of assistant responses.

        Uses token ID matching instead of decode→re-encode, which avoids
        off-by-one errors from tokenizer round-trip inconsistencies.
        """
        tokenizer = self.processor.tokenizer

        # Get the token IDs for the boundary markers
        # Llama 3.2 uses <|start_header_id|>assistant<|end_header_id|> as markers
        header_start_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
        header_end_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        assistant_token_ids = tokenizer.encode("assistant", add_special_tokens=False)

        # Start with everything masked
        labels[:] = -100

        ids = input_ids.tolist()
        n = len(ids)
        i = 0

        while i < n:
            # Look for <|start_header_id|> assistant <|end_header_id|>
            if ids[i] == header_start_id:
                # Check if this is an assistant header
                ast_len = len(assistant_token_ids)
                ast_start = i + 1
                ast_end = ast_start + ast_len
                if (ast_end < n
                        and ids[ast_start:ast_end] == assistant_token_ids
                        and ids[ast_end] == header_end_id):
                    # Found assistant header — skip past it (and the newline after)
                    resp_start = ast_end + 1
                    # Skip newline token if present (token ID 271 = "\n" in Llama)
                    if resp_start < n and tokenizer.decode([ids[resp_start]]).strip() == "":
                        resp_start += 1

                    # Find the end: next <|eot_id|> or end of sequence
                    resp_end = resp_start
                    while resp_end < n and ids[resp_end] != eot_id:
                        resp_end += 1

                    # Unmask the assistant response tokens + the eot token
                    eot_end = min(resp_end + 1, n)
                    labels[resp_start:eot_end] = input_ids[resp_start:eot_end]

                    i = eot_end
                    continue
            i += 1

        return labels


def collate_fn(batch: list[dict]) -> dict:
    """Stack batch items. All tensors are pre-padded to max_length by the
    processor, so shapes match and torch.stack works directly."""
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        tensors = [item[k] for item in batch]
        try:
            collated[k] = torch.stack(tensors)
        except RuntimeError:
            # Fallback: if shapes don't match (shouldn't happen with
            # padding="max_length"), pad to the largest in the batch
            max_len = max(t.shape[-1] for t in tensors)
            padded = []
            for t in tensors:
                if t.shape[-1] < max_len:
                    pad_size = max_len - t.shape[-1]
                    t = torch.nn.functional.pad(t, (0, pad_size), value=0)
                padded.append(t)
            collated[k] = torch.stack(padded)
    return collated


# --------------- Model --------------- #

def build_model_and_processor(cfg: dict):
    """Load Llama 3.2 Vision with LoRA adapters."""
    from transformers import AutoProcessor, MllamaForConditionalGeneration
    from peft import LoraConfig, get_peft_model, TaskType

    model_name = cfg["model"]["name"]
    print(f"[model] Loading {model_name}...")

    # Llama 3.2 Vision is a gated model — needs HF token
    # Set via: huggingface-cli login, or HF_TOKEN env var
    import os
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("[model] Using HF_TOKEN from environment")

    processor = AutoProcessor.from_pretrained(model_name, token=hf_token)

    # Ensure pad token is set (must differ from eos so the model learns to emit eos)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = "<|finetune_right_pad_id|>"
    # Left-pad so the last token is always the response end, not a pad
    processor.tokenizer.padding_side = "right"

    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )

    # Freeze the vision encoder — only train the language model via LoRA
    # MllamaForConditionalGeneration wraps MllamaModel as self.model,
    # which has .vision_model and .language_model
    for param in model.model.vision_model.parameters():
        param.requires_grad = False

    # Apply LoRA
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, processor


# --------------- Scheduler --------------- #

def cosine_warmup(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


# --------------- Eval --------------- #

@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Return average loss on the validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch)
        # Count non-masked label tokens for proper averaging
        n_tokens = (batch["labels"] != -100).sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens

    model.train()
    return total_loss / max(1, total_tokens)


# --------------- Training --------------- #

def train(cfg: dict, train_path: Path, val_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    model, processor = build_model_and_processor(cfg)

    # With device_map="auto", the model places tensors on the right device.
    # We need to know which device to move batch tensors to.
    device = next(model.parameters()).device

    # Datasets
    print("[data] Loading training data...")
    train_ds = VLMDataset(train_path, processor, max_length=cfg["train"]["max_length"])
    print(f"[data] Train: {len(train_ds)} examples")

    val_ds = VLMDataset(val_path, processor, max_length=cfg["train"]["max_length"])
    print(f"[data] Val: {len(val_ds)} examples")

    # num_workers=0 because the dataset holds a reference to the processor
    # (tokenizer + image processor), which can fail to pickle across workers.
    # The bottleneck is GPU forward/backward, not data loading.
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        betas=tuple(cfg["train"]["betas"]),
    )

    total_steps = cfg["train"]["epochs"] * len(train_loader)
    warmup_steps = int(cfg["train"]["warmup_ratio"] * total_steps)
    grad_accum = cfg["train"]["grad_accum_steps"]

    print(f"[train] {cfg['train']['epochs']} epochs, {len(train_loader)} steps/epoch, "
          f"{total_steps} total steps, {warmup_steps} warmup")
    print(f"[train] Effective batch size: {cfg['train']['batch_size'] * grad_accum}")

    best_val_loss = float("inf")
    history = []

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        epoch_tokens = 0

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # LR schedule
            global_step = epoch * len(train_loader) + step
            lr_scale = cosine_warmup(global_step, total_steps, warmup_steps)
            for g in optimizer.param_groups:
                g["lr"] = cfg["train"]["lr"] * lr_scale

            # Forward
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum

            loss.backward()

            # Track loss
            n_tokens = (batch["labels"] != -100).sum().item()
            epoch_loss += outputs.loss.item() * n_tokens
            epoch_tokens += n_tokens

            # Gradient accumulation step
            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                grad_clip = cfg["train"].get("grad_clip", 0.0)
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Logging
            if (step + 1) % cfg["log"]["every_n_steps"] == 0:
                avg = epoch_loss / max(1, epoch_tokens)
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  step {step+1}/{len(train_loader)} | "
                      f"loss={avg:.4f} lr={lr_now:.2e}")

        # Epoch summary
        train_loss = epoch_loss / max(1, epoch_tokens)
        elapsed = time.time() - t0
        examples_per_sec = len(train_ds) / elapsed

        # Validation
        val_loss = evaluate(model, val_loader, device)

        print(f"epoch {epoch+1} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"{examples_per_sec:.0f} ex/s {elapsed:.0f}s")

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "examples_per_sec": round(examples_per_sec, 1),
            "elapsed_sec": round(elapsed, 1),
        })

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  -> New best val_loss={val_loss:.4f}, saving adapter...")
            model.save_pretrained(out_dir / "best_adapter")
            processor.save_pretrained(out_dir / "best_adapter")

        # Save metrics after every epoch
        (out_dir / "metrics.json").write_text(json.dumps({
            "history": history,
            "best_val_loss": round(best_val_loss, 4),
            "config": cfg,
        }, indent=2))

    # Save final adapter too
    model.save_pretrained(out_dir / "final_adapter")
    processor.save_pretrained(out_dir / "final_adapter")

    print(f"\n[train] Done! best val_loss={best_val_loss:.4f}")
    print(f"[train] Adapters saved to {out_dir}/best_adapter and {out_dir}/final_adapter")


# --------------- Main --------------- #

def main():
    ap = argparse.ArgumentParser(description="Fine-tune Llama 3.2 Vision on plant disease QA")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--train-data", type=Path, required=True,
                    help="Path to train.jsonl from create_qa_dataset.py")
    ap.add_argument("--val-data", type=Path, required=True,
                    help="Path to val.jsonl from create_qa_dataset.py")
    ap.add_argument("--output", type=Path, default=Path("runs/llama_vision_v1"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    set_seed(cfg.get("seed", 42))

    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, args.output / "config.yaml")

    train(cfg, args.train_data, args.val_data, args.output)


if __name__ == "__main__":
    main()
