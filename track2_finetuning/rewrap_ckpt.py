"""Convert a raw state_dict checkpoint to the wrapped {state_dict, cfg} format.

Use this for a best.pt produced by an older train.py run that saved the raw
state_dict directly (and possibly with torch.compile's ``_orig_mod.`` prefix).

    python rewrap_ckpt.py --checkpoint best.pt --config config.yaml \
        --output runs/dinov2_l_v1/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    cfg = yaml.safe_load(args.config.read_text())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": sd, "cfg": cfg}, args.output)
    print(f"[rewrap] {args.checkpoint} -> {args.output}  ({len(sd)} tensors)")


if __name__ == "__main__":
    main()
