"""Upload auxiliary training artifacts (train.log, benchmark_results.json) to
an existing Hugging Face Hub repo. Use this after ``publish.sh`` if you want
the raw training log and benchmark numbers archived alongside the model.

Usage:
    # HF_REPO defaults to <hf_user>/dinov2-l-ccmt-mi300x
    HF_TOKEN=hf_xxx python upload_extras.py
    HF_TOKEN=hf_xxx HF_REPO=myuser/my-model python upload_extras.py runs/dinov2_l_v1
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

RUN_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/dinov2_l_v1")

EXTRAS = [
    (RUN_DIR / "train.log", "train.log"),
    (Path("benchmark_results.json"), "benchmark_results.json"),
]

token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN is not set — export it first")

api = HfApi()
repo = os.environ.get("HF_REPO") or f"{api.whoami()['name']}/dinov2-l-ccmt-mi300x"
print(f"[upload] target repo: {repo}")

for local, remote in EXTRAS:
    if not local.exists():
        print(f"[upload] skip {local} (not found)")
        continue
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=remote,
        repo_id=repo,
        token=token,
        commit_message=f"Add {remote}",
    )
    print(f"[upload] {local} -> {repo}/{remote}")

print("[upload] done")
