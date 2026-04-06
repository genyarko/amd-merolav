"""PyTorch-specific CUDA to ROCm mappings and warnings.

Key insight: PyTorch on ROCm still uses 'cuda' as the device string.
torch.device("cuda") is CORRECT on ROCm — the HIP backend is transparent.
"""

from __future__ import annotations

# Things that work identically on ROCm (no change needed, but worth noting)
TORCH_PASSTHROUGH: dict[str, str] = {
    "torch.cuda.is_available()": "Works on ROCm — returns True if HIP device present",
    'torch.device("cuda")': "ROCm uses 'cuda' device string — no change needed",
    ".cuda()": "No change — works on ROCm via HIP backend",
    "torch.cuda.device_count()": "Works on ROCm",
    "torch.cuda.current_device()": "Works on ROCm",
    "torch.cuda.set_device()": "Works on ROCm",
    "torch.cuda.synchronize()": "Works on ROCm",
    "torch.cuda.memory_allocated()": "Works on ROCm",
    "torch.cuda.max_memory_allocated()": "Works on ROCm",
    "torch.cuda.empty_cache()": "Works on ROCm",
    "torch.cuda.Event": "Works on ROCm",
    "torch.cuda.Stream": "Works on ROCm",
}

# Things that need actual changes or awareness
TORCH_WARNINGS: dict[str, str] = {
    "torch.backends.cudnn": (
        "Replace with torch.backends.miopen on ROCm. "
        "MIOpen is the AMD equivalent of cuDNN."
    ),
    "torch.backends.cudnn.benchmark": (
        "Remove or replace. MIOpen auto-tunes by default. "
        "Set MIOPEN_FIND_MODE=3 for exhaustive tuning."
    ),
    "torch.backends.cudnn.deterministic": (
        "Use torch.backends.miopen.deterministic on ROCm."
    ),
    "torch.backends.cudnn.enabled": (
        "Use torch.backends.miopen.enabled on ROCm."
    ),
    "torch.cuda.amp": (
        "AMP works on ROCm. For PyTorch >= 2.0, prefer "
        "torch.amp.autocast('cuda') over torch.cuda.amp.autocast."
    ),
    "torch.cuda.nccl": (
        "Replace with RCCL. RCCL is API-compatible with NCCL. "
        "Set env NCCL_SOCKET_IFNAME for multi-GPU."
    ),
    "torch.cuda.nvtx": (
        "ROCm uses rocTX/roctx for profiling markers. "
        "Use rocm_smi or rocprof instead of nvtx."
    ),
}

# Import-level replacements
IMPORT_MAP: dict[str, tuple[str, str]] = {
    "import pycuda": ("import hip  # HIP Python bindings", "pycuda has no direct ROCm port; use hip-python"),
    "from pycuda": ("from hip", "Use hip-python package"),
    "import cupy": ("import cupy", "CuPy supports ROCm — install cupy-rocm-5-0 or build from source"),
    "import triton": ("import triton", "Triton supports ROCm backend — install triton-rocm"),
    "from apex": ("from apex", "apex supports ROCm — rebuild from source with ROCm"),
    "import tensorrt": ("import migraphx", "TensorRT → MIGraphX on AMD"),
    "from tensorrt": ("from migraphx", "TensorRT → MIGraphX on AMD"),
}

# torch.backends replacements (code-level)
BACKENDS_REPLACE: dict[str, str] = {
    "torch.backends.cudnn.benchmark = True": (
        "# MIOpen auto-tunes by default on ROCm\n"
        "# Set MIOPEN_FIND_MODE=3 env var for exhaustive tuning"
    ),
    "torch.backends.cudnn.benchmark = False": (
        "# MIOpen tuning disabled — set MIOPEN_FIND_MODE=0 if needed"
    ),
    "torch.backends.cudnn.deterministic = True": (
        "torch.backends.miopen.deterministic = True"
    ),
    "torch.backends.cudnn.deterministic = False": (
        "torch.backends.miopen.deterministic = False"
    ),
    "torch.backends.cudnn.enabled = True": (
        "torch.backends.miopen.enabled = True"
    ),
    "torch.backends.cudnn.enabled = False": (
        "torch.backends.miopen.enabled = False"
    ),
    "torch.backends.cudnn.allow_tf32 = True": (
        "# ROCm: MI300X supports BF16 natively — TF32 not applicable\n"
        "# torch.backends.miopen.allow_tf32 is not supported; use BF16 instead"
    ),
    "torch.backends.cudnn.allow_tf32 = False": (
        "# ROCm: TF32 not applicable on AMD GPUs"
    ),
    "torch.backends.cudnn.flash_sdp_enabled = True": (
        "# ROCm: Flash attention via composable_kernel is enabled by default\n"
        "# Use torch.backends.cuda.enable_flash_sdp(True) — works on ROCm"
    ),
    "torch.backends.cudnn.flash_sdp_enabled = False": (
        "# ROCm: torch.backends.cuda.enable_flash_sdp(False)"
    ),
}
