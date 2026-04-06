"""CUDA ecosystem library to ROCm equivalent mappings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LibraryMapping:
    cuda_name: str
    rocm_name: str | None
    description: str
    install_note: str = ""
    confidence: float = 1.0


LIBRARY_EQUIVALENTS: list[LibraryMapping] = [
    LibraryMapping(
        cuda_name="cuDNN",
        rocm_name="MIOpen",
        description="Deep learning primitives (conv, RNN, normalization, pooling)",
        install_note="pip install miopen or included in ROCm toolkit",
    ),
    LibraryMapping(
        cuda_name="cuBLAS",
        rocm_name="rocBLAS",
        description="BLAS (matrix multiply, GEMM) operations",
        install_note="Included in ROCm toolkit",
    ),
    LibraryMapping(
        cuda_name="cuSPARSE",
        rocm_name="rocSPARSE",
        description="Sparse matrix operations",
        install_note="Included in ROCm toolkit",
    ),
    LibraryMapping(
        cuda_name="cuFFT",
        rocm_name="rocFFT",
        description="Fast Fourier Transform",
        install_note="Included in ROCm toolkit",
    ),
    LibraryMapping(
        cuda_name="cuRAND",
        rocm_name="rocRAND",
        description="Random number generation",
        install_note="Included in ROCm toolkit",
    ),
    LibraryMapping(
        cuda_name="NCCL",
        rocm_name="RCCL",
        description="Multi-GPU collective communications",
        install_note="API-compatible with NCCL; included in ROCm toolkit",
    ),
    LibraryMapping(
        cuda_name="TensorRT",
        rocm_name="MIGraphX",
        description="Inference optimization and graph compilation engine",
        install_note="pip install migraphx or build from source",
    ),
    LibraryMapping(
        cuda_name="Thrust",
        rocm_name="rocThrust",
        description="Parallel algorithms library (sort, scan, reduce)",
        install_note="Included in ROCm toolkit",
    ),
    LibraryMapping(
        cuda_name="CUB",
        rocm_name="hipCUB",
        description="Block/warp/device cooperative primitives",
        install_note="Included in ROCm toolkit",
    ),
    LibraryMapping(
        cuda_name="cuSOLVER",
        rocm_name="rocSOLVER",
        description="Dense linear algebra solvers (LU, QR, SVD)",
        install_note="Included in ROCm toolkit",
    ),
    LibraryMapping(
        cuda_name="CUTLASS",
        rocm_name="composable_kernel",
        description="GEMM kernel templates and composable primitives",
        install_note="https://github.com/ROCm/composable_kernel",
        confidence=0.7,
    ),
    LibraryMapping(
        cuda_name="cuDLA",
        rocm_name=None,
        description="Deep Learning Accelerator (NVIDIA-specific hardware)",
        install_note="No AMD equivalent — DLA is NVIDIA Jetson-specific",
        confidence=0.0,
    ),
    LibraryMapping(
        cuda_name="NVRTC",
        rocm_name="hipRTC",
        description="Runtime compilation of GPU kernels",
        install_note="Included in ROCm toolkit",
    ),
    LibraryMapping(
        cuda_name="nvJPEG",
        rocm_name="rocJPEG",
        description="GPU-accelerated JPEG decoding",
        install_note="Available in ROCm 6.x+",
        confidence=0.8,
    ),
]


def get_library_map() -> dict[str, LibraryMapping]:
    """Return library mappings keyed by CUDA library name."""
    return {lib.cuda_name: lib for lib in LIBRARY_EQUIVALENTS}


def get_library_by_import(import_name: str) -> LibraryMapping | None:
    """Try to match an import string to a known library mapping."""
    import_lower = import_name.lower()
    for lib in LIBRARY_EQUIVALENTS:
        if lib.cuda_name.lower() in import_lower:
            return lib
    return None
