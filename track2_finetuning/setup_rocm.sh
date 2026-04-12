#!/usr/bin/env bash
# Track 2 — environment setup on the DigitalOcean "PyTorch 2.6.0 + ROCm 7.0" image.
#
# The droplet image ships with PyTorch + torchvision already built for ROCm,
# so this script ONLY installs the extra deps (timm, sklearn, tqdm, ...).
# DO NOT pip-install torch here — it would clobber the ROCm build with a CUDA
# wheel and silently disable the GPU.

set -euo pipefail

# ---------------------------------------------------------------
# 1) System sanity
# ---------------------------------------------------------------
echo "[setup] rocminfo check"
command -v rocminfo >/dev/null || { echo "rocminfo missing — is ROCm installed?"; exit 1; }
rocminfo | grep -E "Name:\s+gfx" | head -1 || true

# ---------------------------------------------------------------
# 2) Confirm preinstalled PyTorch has ROCm
# ---------------------------------------------------------------
python3 - <<'PY'
import sys, torch
hip = getattr(torch.version, "hip", None)
if not hip:
    print(f"[setup] ERROR: torch {torch.__version__} has no HIP/ROCm build.")
    print("        This script expects the 'PyTorch + ROCm' droplet image.")
    sys.exit(1)
print(f"[setup] torch {torch.__version__}  (ROCm/HIP {hip})")
PY

# ---------------------------------------------------------------
# 3) Create a venv that inherits the system's ROCm torch, then install extra deps
#
#    Ubuntu 24 (PEP 668) blocks `pip install` into the system Python. We also
#    don't want to shadow the image's ROCm-built torch, so we use
#    --system-site-packages: the venv sees system torch but gives us a writable
#    location for timm, sklearn, huggingface_hub, etc.
# ---------------------------------------------------------------
VENV_DIR="${VENV_DIR:-$HOME/venv}"
if [ ! -d "$VENV_DIR" ]; then
  echo "[setup] creating venv at $VENV_DIR (--system-site-packages)"
  python3 -m venv --system-site-packages "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install huggingface_hub

# ---------------------------------------------------------------
# 4) GPU visibility sanity
# ---------------------------------------------------------------
python - <<'PY'
import torch
print(f"cuda avail   : {torch.cuda.is_available()}")
print(f"device count : {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}  vram={p.total_memory / 1e9:.1f} GB")
PY

echo
echo "[setup] Done."
echo "[setup] Activate for future sessions with:  source $VENV_DIR/bin/activate"
echo "Next: stage the dataset (see stage_dataset.sh), then run run_mi300x.sh."
