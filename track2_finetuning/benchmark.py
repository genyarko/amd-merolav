"""Throughput and VRAM profiling for DINOv2-L on the MI300X.

Compares MI300X numbers against the Kaggle P100 EfficientNetB0 baseline.
Does NOT need real data — uses synthetic batches so you can run it before
staging the CCMT dataset.

Usage:
    python benchmark.py --config config.yaml --batch-sizes 32,64,128,256,512
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import timm
import torch
import yaml

# Recorded from the Kaggle evaluation notebook (P100 / 224×224 / EfficientNetB0).
#   2632 steps × 32 bs / 238s = 353 images/sec
BASELINE_P100 = {
    "gpu": "Tesla P100-PCIE-16GB",
    "model": "efficientnet_b0",
    "img_size": 224,
    "batch_size": 32,
    "images_per_sec_train": None,     # not captured in notebook (training used T4/P100)
    "images_per_sec_eval": 353.0,     # 2632 * 32 / 238s  from the eval notebook
    "notes": "P100, eval pass only; training step time not cleanly logged",
}


def run_bench(model_name, img_size, batch_size, steps, device, amp_dtype,
              train=True, channels_last=False, compile_model=False):
    model = timm.create_model(model_name, pretrained=False, num_classes=22,
                              img_size=img_size).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    if compile_model:
        model = torch.compile(model)
    model.train(train)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    if channels_last:
        x = x.to(memory_format=torch.channels_last)
    y = torch.randint(0, 22, (batch_size,), device=device)

    # Warmup
    for _ in range(3):
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            out = model(x)
            loss = torch.nn.functional.cross_entropy(out, y)
        if train:
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            out = model(x)
            loss = torch.nn.functional.cross_entropy(out, y)
        if train:
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    imgs_per_sec = (steps * batch_size) / elapsed

    del model, optim, x, y, out, loss
    torch.cuda.empty_cache()
    return {
        "model": model_name,
        "img_size": img_size,
        "batch_size": batch_size,
        "mode": "train" if train else "eval",
        "steps": steps,
        "elapsed_s": round(elapsed, 3),
        "images_per_sec": round(imgs_per_sec, 1),
        "peak_vram_gb": round(peak_mem_gb, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--batch-sizes", default="32,64,128,256",
                    help="comma-separated list")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--output", type=Path, default=Path("benchmark_results.json"))
    ap.add_argument("--mode", choices=["train", "eval", "both"], default="both")
    ap.add_argument("--channels-last", action="store_true",
                    help="use NHWC memory format — usually faster for vision on ROCm")
    ap.add_argument("--compile", action="store_true",
                    help="wrap model in torch.compile")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if not torch.cuda.is_available():
        raise SystemExit("CUDA (ROCm) not available")

    device = torch.device("cuda:0")
    amp_dtype = torch.bfloat16 if cfg["train"]["amp_dtype"] == "bfloat16" else torch.float16

    print(f"[bench] device      : {torch.cuda.get_device_name(0)}")
    print(f"[bench] ROCm/HIP    : {getattr(torch.version, 'hip', None)}")
    print(f"[bench] total VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"[bench] model       : {cfg['model']['name']}")
    print(f"[bench] img_size    : {cfg['model']['img_size']}")
    print(f"[bench] amp_dtype   : {amp_dtype}")

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    modes = ["train", "eval"] if args.mode == "both" else [args.mode]

    results = []
    for bs in batch_sizes:
        for m in modes:
            try:
                r = run_bench(cfg["model"]["name"], cfg["model"]["img_size"],
                              bs, args.steps, device, amp_dtype, train=(m == "train"),
                              channels_last=args.channels_last, compile_model=args.compile)
                print(f"  bs={bs:>4d}  {m:<5s}  "
                      f"{r['images_per_sec']:>7.1f} img/s   "
                      f"peak VRAM {r['peak_vram_gb']:>5.1f} GB")
                results.append(r)
            except torch.cuda.OutOfMemoryError:
                print(f"  bs={bs:>4d}  {m:<5s}  OOM")
                results.append({"batch_size": bs, "mode": m, "error": "OOM"})
                torch.cuda.empty_cache()

    payload = {
        "device": torch.cuda.get_device_name(0),
        "rocm_hip": getattr(torch.version, "hip", None),
        "torch": torch.__version__,
        "model": cfg["model"]["name"],
        "img_size": cfg["model"]["img_size"],
        "amp_dtype": str(amp_dtype),
        "results": results,
        "baseline_p100_effnetb0": BASELINE_P100,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"[bench] wrote {args.output}")


if __name__ == "__main__":
    main()
