"""Plant Disease Assistant — Hybrid Gradio App.

Uses DINOv2-L (Track 2, 97% acc) for classification, then either:
  - The fine-tuned Llama 3.2 Vision adapter for conversational responses, OR
  - Template-based responses from treatment_knowledge.json as fallback.

Run:
    python3 vision/app.py \
        --dinov2-checkpoint runs/dinov2_l_v1/best.pt \
        --splits track2_finetuning/splits.json \
        --knowledge vision/data/treatment_knowledge.json \
        [--vlm-adapter runs/llama_vision_v1/best_adapter]  # optional
        [--port 7860]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gradio as gr
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from timm.data import create_transform


# --------------- DINOv2 Classifier --------------- #

class PlantClassifier:
    """DINOv2-L classifier from Track 2."""

    def __init__(self, checkpoint_path: Path, splits_path: Path, device: str = "cuda"):
        self.device = torch.device(device)

        # Load splits for class names
        splits = json.loads(splits_path.read_text())
        self.idx_to_class = {v: k for k, v in splits["class_to_idx"].items()}
        self.class_names = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]
        self.num_classes = len(self.class_names)

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            cfg = ckpt.get("cfg", {})
        else:
            state_dict = ckpt
            cfg = {}

        # Strip torch.compile prefix
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

        # Build model
        model_name = cfg.get("model", {}).get("name", "vit_large_patch14_dinov2.lvd142m")
        img_size = cfg.get("model", {}).get("img_size", 224)

        self.model = timm.create_model(
            model_name, pretrained=False,
            num_classes=self.num_classes, img_size=img_size,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        # Transform
        self.transform = create_transform(
            input_size=img_size, is_training=False,
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
            interpolation="bicubic", crop_pct=0.95,
        )

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 3) -> list[dict]:
        """Return top-k predictions with class name, confidence, and index."""
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = self.model(x)
        probs = F.softmax(logits, dim=-1).squeeze(0).float().cpu().numpy()

        top_indices = np.argsort(probs)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "class": self.class_names[idx],
                "confidence": float(probs[idx]),
                "index": int(idx),
            })
        return results


# --------------- Knowledge Base Responder --------------- #

class KnowledgeResponder:
    """Template-based responses from treatment_knowledge.json."""

    def __init__(self, knowledge_path: Path):
        self.knowledge = json.loads(knowledge_path.read_text())

    def format_label(self, label: str) -> str:
        """Convert 'cashew_anthracnose' → 'Cashew Anthracnose'."""
        return label.replace("_", " ").title()

    def respond(self, predictions: list[dict]) -> str:
        """Generate a diagnosis + treatment response from predictions."""
        top = predictions[0]
        label = top["class"]
        confidence = top["confidence"]

        if label not in self.knowledge:
            return (
                f"**Prediction:** {self.format_label(label)} "
                f"(confidence: {confidence:.1%})\n\n"
                "No detailed information available for this condition."
            )

        k = self.knowledge[label]
        is_healthy = k["disease"] == "Healthy"

        # Build response
        lines = []

        # Header with confidence
        if is_healthy:
            lines.append(f"## {k['crop']} — Healthy")
            lines.append(f"**Confidence:** {confidence:.1%}\n")
            lines.append(f"{k['symptoms']}")
            lines.append("\nKeep monitoring regularly and continue your current care routine.")
        else:
            lines.append(f"## {k['crop']} — {k['disease']}")
            lines.append(f"**Confidence:** {confidence:.1%}\n")

            if k.get("pathogen"):
                lines.append(f"**Pathogen:** *{k['pathogen']}*\n")

            lines.append(f"### Symptoms")
            lines.append(f"{k['symptoms']}\n")

            # Severity guide
            lines.append("### Severity Guide")
            for level, desc in k["severity_cues"].items():
                lines.append(f"- **{level.title()}:** {desc}")
            lines.append("")

            lines.append(f"### Treatment")
            lines.append(f"{k['treatment']}\n")

            lines.append(f"### Prevention")
            lines.append(f"{k['prevention']}")

        # Alternative predictions
        if len(predictions) > 1:
            lines.append("\n---\n### Other Possibilities")
            for p in predictions[1:]:
                if p["confidence"] > 0.05:
                    lines.append(
                        f"- {self.format_label(p['class'])} ({p['confidence']:.1%})"
                    )

        return "\n".join(lines)


# --------------- VLM Responder (optional) --------------- #

class VLMResponder:
    """Fine-tuned Llama 3.2 Vision for conversational responses."""

    def __init__(self, adapter_path: Path, device: str = "cuda"):
        from transformers import AutoProcessor, MllamaForConditionalGeneration
        from peft import PeftModel

        base_model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

        self.processor = AutoProcessor.from_pretrained(adapter_path)
        base_model = MllamaForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def respond(self, image: Image.Image, question: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            images=image, text=text,
            return_tensors="pt",
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )

        # Decode only the generated tokens (skip the input)
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(generated, skip_special_tokens=True)


# --------------- Gradio App --------------- #

def build_app(
    classifier: PlantClassifier,
    knowledge: KnowledgeResponder,
    vlm: VLMResponder | None = None,
) -> gr.Blocks:

    def diagnose(image: Image.Image, use_vlm: bool, question: str):
        if image is None:
            return "Please upload an image.", ""

        image = image.convert("RGB")

        # Step 1: DINOv2 classification
        predictions = classifier.predict(image, top_k=3)
        top = predictions[0]

        classification_text = (
            f"**DINOv2-L Classification (97% accuracy)**\n\n"
            f"| Rank | Disease | Confidence |\n"
            f"|------|---------|------------|\n"
        )
        for i, p in enumerate(predictions):
            marker = " ←" if i == 0 else ""
            classification_text += (
                f"| {i+1} | {knowledge.format_label(p['class'])} | "
                f"{p['confidence']:.1%}{marker} |\n"
            )

        # Step 2: Generate response
        if use_vlm and vlm is not None:
            q = question if question.strip() else "What disease does this plant have and how should I treat it?"
            response = vlm.respond(image, q)
        else:
            response = knowledge.respond(predictions)

        return classification_text, response

    # Example images (if any exist)
    with gr.Blocks(
        title="Plant Disease Assistant",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# Plant Disease Assistant\n"
            "Upload a photo of a plant leaf to get an instant diagnosis, "
            "severity assessment, and treatment recommendations.\n\n"
            "*Powered by DINOv2-L fine-tuned on AMD Instinct MI300X (ROCm) — "
            "Track 2 & 3 of the lablab.ai AMD Hackathon*"
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload a plant leaf photo")

                with gr.Accordion("Advanced Options", open=False):
                    use_vlm_toggle = gr.Checkbox(
                        label="Use AI conversational mode (Llama 3.2 Vision)",
                        value=vlm is not None,
                        interactive=vlm is not None,
                    )
                    question_input = gr.Textbox(
                        label="Custom question (optional)",
                        placeholder="e.g., How should I treat this? Is this severe?",
                        visible=vlm is not None,
                    )

                diagnose_btn = gr.Button("Diagnose", variant="primary", size="lg")

            with gr.Column(scale=2):
                classification_output = gr.Markdown(label="Classification")
                response_output = gr.Markdown(label="Diagnosis & Treatment")

        diagnose_btn.click(
            fn=diagnose,
            inputs=[image_input, use_vlm_toggle, question_input],
            outputs=[classification_output, response_output],
        )

        # Also trigger on image upload
        image_input.change(
            fn=diagnose,
            inputs=[image_input, use_vlm_toggle, question_input],
            outputs=[classification_output, response_output],
        )

        gr.Markdown(
            "---\n"
            "**Models:**\n"
            "- Classification: DINOv2-Large (304M params) — 97.06% accuracy, 0.9713 macro F1\n"
            "- Conversational: Llama 3.2 Vision 11B + LoRA adapter (optional)\n\n"
            "**Hardware:** AMD Instinct MI300X (192 GB HBM3) via DigitalOcean\n\n"
            "**Dataset:** CCMT Crop Pest and Disease Detection (22 classes, 4 crops)\n\n"
            "*Built for the lablab.ai AMD Hackathon — Tracks 2 & 3*"
        )

    return app


def main():
    ap = argparse.ArgumentParser(description="Plant Disease Assistant — Hybrid Gradio App")
    ap.add_argument("--dinov2-checkpoint", type=Path, required=True,
                    help="Path to DINOv2-L best.pt from Track 2")
    ap.add_argument("--splits", type=Path, required=True,
                    help="Path to splits.json")
    ap.add_argument("--knowledge", type=Path,
                    default=Path(__file__).parent / "data" / "treatment_knowledge.json")
    ap.add_argument("--vlm-adapter", type=Path, default=None,
                    help="Path to fine-tuned Llama Vision adapter (optional)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true",
                    help="Create a public Gradio share link")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print("[app] Loading DINOv2-L classifier...")
    classifier = PlantClassifier(args.dinov2_checkpoint, args.splits, args.device)
    print(f"[app] Loaded {classifier.num_classes} classes")

    knowledge = KnowledgeResponder(args.knowledge)

    vlm = None
    if args.vlm_adapter and args.vlm_adapter.exists():
        print("[app] Loading Llama 3.2 Vision adapter...")
        vlm = VLMResponder(args.vlm_adapter, args.device)
        print("[app] VLM loaded")
    else:
        print("[app] No VLM adapter — using template-based responses")

    app = build_app(classifier, knowledge, vlm)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
