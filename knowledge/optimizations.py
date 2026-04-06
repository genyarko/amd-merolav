"""AMD-specific optimization suggestions triggered by code patterns."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OptimizationRule:
    trigger: str  # pattern to match in code (import, function call, etc.)
    category: str
    suggestion: str
    url: str = ""
    priority: int = 1  # 1=high, 2=medium, 3=low


OPTIMIZATION_RULES: list[OptimizationRule] = [
    # Hugging Face / Transformers
    OptimizationRule(
        trigger="transformers",
        category="Framework",
        suggestion=(
            "Consider using Hugging Face Optimum-AMD for ROCm-optimized "
            "transformer inference. Provides optimized ONNX Runtime and "
            "MIGraphX backends.\n"
            "  Install: pip install optimum[amd]"
        ),
        url="https://github.com/huggingface/optimum-amd",
        priority=1,
    ),

    # Convolution tuning
    OptimizationRule(
        trigger="torch.nn.Conv",
        category="Performance",
        suggestion=(
            "MIOpen auto-tunes convolution algorithms on first run. "
            "For best performance:\n"
            "  export MIOPEN_FIND_MODE=3  (exhaustive search)\n"
            "  export MIOPEN_USER_DB_PATH=~/.config/miopen  (cache tuning results)"
        ),
        priority=1,
    ),

    # torch.compile
    OptimizationRule(
        trigger="torch.compile",
        category="Compilation",
        suggestion=(
            "torch.compile works on ROCm with the Triton backend. "
            "Ensure triton-rocm is installed. For MI300X, the Triton "
            "backend can provide significant speedups."
        ),
        priority=1,
    ),

    # Flash Attention
    OptimizationRule(
        trigger="flash_attn",
        category="Attention",
        suggestion=(
            "Flash Attention 2 is supported on ROCm / MI300X. "
            "Use the composable_kernel backend for best performance. "
            "Install: pip install flash-attn --no-build-isolation"
        ),
        priority=1,
    ),

    # Multi-GPU / DataParallel
    OptimizationRule(
        trigger="DataParallel",
        category="Multi-GPU",
        suggestion=(
            "Prefer DistributedDataParallel (DDP) with RCCL backend "
            "over DataParallel for multi-GPU training on ROCm.\n"
            "  torch.distributed.init_process_group(backend='nccl')  # RCCL is NCCL-compatible"
        ),
        priority=1,
    ),

    # Mixed precision
    OptimizationRule(
        trigger="torch.cuda.amp",
        category="Mixed Precision",
        suggestion=(
            "AMP works on ROCm. For PyTorch >= 2.0, prefer the new API:\n"
            "  with torch.amp.autocast('cuda', dtype=torch.float16):\n"
            "MI300X also supports BF16 natively for better training stability."
        ),
        priority=2,
    ),

    # DeepSpeed
    OptimizationRule(
        trigger="deepspeed",
        category="Framework",
        suggestion=(
            "DeepSpeed has ROCm support. Use the ROCm-specific installation:\n"
            "  DS_BUILD_OPS=1 pip install deepspeed\n"
            "Ensure ROCm toolkit is in PATH."
        ),
        url="https://github.com/microsoft/DeepSpeed",
        priority=2,
    ),

    # FSDP
    OptimizationRule(
        trigger="FullyShardedDataParallel",
        category="Multi-GPU",
        suggestion=(
            "FSDP works on ROCm with RCCL backend. "
            "For MI300X, consider using FSDP with BF16 mixed precision "
            "for optimal memory efficiency."
        ),
        priority=2,
    ),

    # vLLM
    OptimizationRule(
        trigger="vllm",
        category="Inference",
        suggestion=(
            "vLLM has native ROCm support for MI300X. "
            "Use the ROCm Docker image or install from source:\n"
            "  pip install vllm  # with ROCm toolkit in PATH"
        ),
        url="https://docs.vllm.ai/en/latest/getting_started/amd-installation.html",
        priority=2,
    ),

    # bitsandbytes quantization
    OptimizationRule(
        trigger="bitsandbytes",
        category="Quantization",
        suggestion=(
            "bitsandbytes has ROCm support (bitsandbytes-rocm). "
            "Install: pip install bitsandbytes --prefer-binary\n"
            "For MI300X, 4-bit quantization with NF4 is supported."
        ),
        priority=2,
    ),

    # Environment variable
    OptimizationRule(
        trigger="CUDA_VISIBLE_DEVICES",
        category="Environment",
        suggestion=(
            "Replace CUDA_VISIBLE_DEVICES with:\n"
            "  HIP_VISIBLE_DEVICES (preferred on ROCm)\n"
            "  ROCR_VISIBLE_DEVICES (alternative)\n"
            "Both work; HIP_VISIBLE_DEVICES is most compatible."
        ),
        priority=1,
    ),

    # Memory management
    OptimizationRule(
        trigger="torch.cuda.memory",
        category="Memory",
        suggestion=(
            "ROCm memory management tips for MI300X (192GB HBM3):\n"
            "  - Use PYTORCH_HIP_ALLOC_CONF for memory allocator tuning\n"
            "  - torch.cuda.empty_cache() works on ROCm\n"
            "  - Monitor with rocm-smi instead of nvidia-smi"
        ),
        priority=3,
    ),

    # Profiling
    OptimizationRule(
        trigger="torch.profiler",
        category="Profiling",
        suggestion=(
            "PyTorch profiler works on ROCm. Additionally use:\n"
            "  - rocprof for kernel-level profiling\n"
            "  - rocm-smi for GPU monitoring\n"
            "  - omniperf for detailed performance analysis"
        ),
        priority=3,
    ),
]


def find_matching_optimizations(code: str) -> list[OptimizationRule]:
    """Find all optimization suggestions that match patterns in the code."""
    matches = []
    for rule in OPTIMIZATION_RULES:
        if rule.trigger in code:
            matches.append(rule)
    # Sort by priority (1=high first)
    matches.sort(key=lambda r: r.priority)
    return matches
