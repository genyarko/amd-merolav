"""CUDA to HIP/ROCm API mappings.

Sources:
- https://rocm.docs.amd.com/projects/HIP/en/latest/reference/api_syntax.html
- https://rocm.docs.amd.com/projects/HIPIFY/en/latest/
"""

from __future__ import annotations

# Format: cuda_symbol -> (hip_symbol, notes, confidence)
# confidence: 1.0 = safe direct replacement, <1.0 = needs review

RUNTIME_API_MAP: dict[str, tuple[str, str, float]] = {
    # Memory Management
    "cudaMalloc": ("hipMalloc", "", 1.0),
    "cudaFree": ("hipFree", "", 1.0),
    "cudaMemcpy": ("hipMemcpy", "", 1.0),
    "cudaMemcpyAsync": ("hipMemcpyAsync", "", 1.0),
    "cudaMemset": ("hipMemset", "", 1.0),
    "cudaMemsetAsync": ("hipMemsetAsync", "", 1.0),
    "cudaMallocManaged": ("hipMallocManaged", "", 1.0),
    "cudaMallocHost": ("hipHostMalloc", "Different name pattern", 1.0),
    "cudaFreeHost": ("hipHostFree", "Different name pattern", 1.0),
    "cudaHostAlloc": ("hipHostMalloc", "", 1.0),
    "cudaHostRegister": ("hipHostRegister", "", 1.0),
    "cudaHostUnregister": ("hipHostUnregister", "", 1.0),
    "cudaMallocPitch": ("hipMallocPitch", "", 1.0),
    "cudaMalloc3D": ("hipMalloc3D", "", 1.0),
    "cudaMemcpy2D": ("hipMemcpy2D", "", 1.0),
    "cudaMemcpy3D": ("hipMemcpy3D", "", 1.0),
    "cudaMemGetInfo": ("hipMemGetInfo", "", 1.0),
    "cudaMemcpyToSymbol": ("hipMemcpyToSymbol", "", 1.0),
    "cudaMemcpyFromSymbol": ("hipMemcpyFromSymbol", "", 1.0),

    # Device Management
    "cudaGetDeviceCount": ("hipGetDeviceCount", "", 1.0),
    "cudaSetDevice": ("hipSetDevice", "", 1.0),
    "cudaGetDevice": ("hipGetDevice", "", 1.0),
    "cudaGetDeviceProperties": ("hipGetDeviceProperties", "", 1.0),
    "cudaDeviceReset": ("hipDeviceReset", "", 1.0),
    "cudaDeviceSynchronize": ("hipDeviceSynchronize", "", 1.0),
    "cudaDeviceGetAttribute": ("hipDeviceGetAttribute", "", 1.0),
    "cudaChooseDevice": ("hipChooseDevice", "", 1.0),
    "cudaDeviceCanAccessPeer": ("hipDeviceCanAccessPeer", "", 1.0),
    "cudaDeviceEnablePeerAccess": ("hipDeviceEnablePeerAccess", "", 1.0),
    "cudaDeviceDisablePeerAccess": ("hipDeviceDisablePeerAccess", "", 1.0),
    "cudaSetDeviceFlags": ("hipSetDeviceFlags", "", 1.0),

    # Stream Management
    "cudaStreamCreate": ("hipStreamCreate", "", 1.0),
    "cudaStreamCreateWithFlags": ("hipStreamCreateWithFlags", "", 1.0),
    "cudaStreamCreateWithPriority": ("hipStreamCreateWithPriority", "", 1.0),
    "cudaStreamDestroy": ("hipStreamDestroy", "", 1.0),
    "cudaStreamSynchronize": ("hipStreamSynchronize", "", 1.0),
    "cudaStreamWaitEvent": ("hipStreamWaitEvent", "", 1.0),
    "cudaStreamQuery": ("hipStreamQuery", "", 1.0),
    "cudaStreamAddCallback": ("hipStreamAddCallback", "", 1.0),

    # Event Management
    "cudaEventCreate": ("hipEventCreate", "", 1.0),
    "cudaEventCreateWithFlags": ("hipEventCreateWithFlags", "", 1.0),
    "cudaEventRecord": ("hipEventRecord", "", 1.0),
    "cudaEventSynchronize": ("hipEventSynchronize", "", 1.0),
    "cudaEventElapsedTime": ("hipEventElapsedTime", "", 1.0),
    "cudaEventDestroy": ("hipEventDestroy", "", 1.0),
    "cudaEventQuery": ("hipEventQuery", "", 1.0),

    # Error Handling
    "cudaGetLastError": ("hipGetLastError", "", 1.0),
    "cudaPeekAtLastError": ("hipPeekAtLastError", "", 1.0),
    "cudaGetErrorString": ("hipGetErrorString", "", 1.0),
    "cudaGetErrorName": ("hipGetErrorName", "", 1.0),

    # Texture / Surface (partial support)
    "cudaCreateTextureObject": ("hipCreateTextureObject", "Check ROCm version for support", 0.8),
    "cudaDestroyTextureObject": ("hipDestroyTextureObject", "", 0.8),
    "cudaCreateSurfaceObject": ("hipCreateSurfaceObject", "Limited support", 0.6),

    # Occupancy
    "cudaOccupancyMaxActiveBlocksPerMultiprocessor": (
        "hipOccupancyMaxActiveBlocksPerMultiprocessor", "", 1.0
    ),
    "cudaOccupancyMaxPotentialBlockSize": (
        "hipOccupancyMaxPotentialBlockSize", "", 1.0
    ),

    # Unified Addressing
    "cudaPointerGetAttributes": ("hipPointerGetAttributes", "", 1.0),
}

DRIVER_API_MAP: dict[str, tuple[str, str, float]] = {
    "cuInit": ("hipInit", "", 1.0),
    "cuDeviceGet": ("hipDeviceGet", "", 1.0),
    "cuDeviceGetCount": ("hipGetDeviceCount", "", 1.0),
    "cuDeviceGetName": ("hipDeviceGetName", "", 1.0),
    "cuDeviceTotalMem": ("hipDeviceTotalMem", "", 1.0),
    "cuCtxCreate": ("hipCtxCreate", "", 1.0),
    "cuCtxDestroy": ("hipCtxDestroy", "", 1.0),
    "cuCtxSetCurrent": ("hipCtxSetCurrent", "", 1.0),
    "cuCtxGetCurrent": ("hipCtxGetCurrent", "", 1.0),
    "cuCtxSynchronize": ("hipCtxSynchronize", "", 1.0),
    "cuModuleLoad": ("hipModuleLoad", "", 1.0),
    "cuModuleUnload": ("hipModuleUnload", "", 1.0),
    "cuModuleGetFunction": ("hipModuleGetFunction", "", 1.0),
    "cuLaunchKernel": ("hipModuleLaunchKernel", "", 1.0),
    "cuMemAlloc": ("hipMalloc", "", 1.0),
    "cuMemFree": ("hipFree", "", 1.0),
    "cuMemcpyHtoD": ("hipMemcpyHtoD", "", 1.0),
    "cuMemcpyDtoH": ("hipMemcpyDtoH", "", 1.0),
}

ERROR_CODE_MAP: dict[str, str] = {
    "cudaSuccess": "hipSuccess",
    "cudaErrorMemoryAllocation": "hipErrorOutOfMemory",
    "cudaErrorInvalidValue": "hipErrorInvalidValue",
    "cudaErrorInvalidDevice": "hipErrorInvalidDevice",
    "cudaErrorNotReady": "hipErrorNotReady",
    "cudaErrorInvalidDevicePointer": "hipErrorInvalidDevicePointer",
    "cudaErrorLaunchFailure": "hipErrorLaunchFailure",
    "cudaErrorLaunchTimeout": "hipErrorLaunchTimeOut",
    "cudaErrorNoDevice": "hipErrorNoDevice",
    "cudaErrorPeerAccessAlreadyEnabled": "hipErrorPeerAccessAlreadyEnabled",
    "cudaErrorPeerAccessNotEnabled": "hipErrorPeerAccessNotEnabled",
    "cudaErrorInvalidResourceHandle": "hipErrorInvalidResourceHandle",
}

DEFINE_MAP: dict[str, str] = {
    "cudaMemcpyHostToDevice": "hipMemcpyHostToDevice",
    "cudaMemcpyDeviceToHost": "hipMemcpyDeviceToHost",
    "cudaMemcpyDeviceToDevice": "hipMemcpyDeviceToDevice",
    "cudaMemcpyHostToHost": "hipMemcpyHostToHost",
    "cudaMemcpyDefault": "hipMemcpyDefault",
    "cudaStreamDefault": "hipStreamDefault",
    "cudaStreamNonBlocking": "hipStreamNonBlocking",
    "cudaEventDefault": "hipEventDefault",
    "cudaEventBlockingSync": "hipEventBlockingSync",
    "cudaEventDisableTiming": "hipEventDisableTiming",
    "cudaHostAllocDefault": "hipHostMallocDefault",
    "cudaHostAllocPortable": "hipHostMallocPortable",
    "cudaHostAllocMapped": "hipHostMallocMapped",
    "cudaHostAllocWriteCombined": "hipHostMallocWriteCombined",
}

ENV_VAR_MAP: dict[str, str] = {
    "CUDA_VISIBLE_DEVICES": "HIP_VISIBLE_DEVICES",
    "CUDA_LAUNCH_BLOCKING": "HIP_LAUNCH_BLOCKING",
    "CUDA_DEVICE_ORDER": "HIP_DEVICE_ORDER",
}

HEADER_MAP: dict[str, str] = {
    "cuda_runtime.h": "hip/hip_runtime.h",
    "cuda_runtime_api.h": "hip/hip_runtime_api.h",
    "cuda.h": "hip/hip_runtime.h",
    "cuda_fp16.h": "hip/hip_fp16.h",
    "cuda_profiler_api.h": "hip/hip_runtime.h",
    "cublas_v2.h": "rocblas/rocblas.h",
    "cufft.h": "rocfft/rocfft.h",
    "curand.h": "rocrand/rocrand.h",
    "cusparse.h": "rocsparse/rocsparse.h",
    "cudnn.h": "miopen/miopen.h",
}


def get_all_mappings() -> dict[str, tuple[str, str, float]]:
    """Return all CUDA->HIP mappings combined."""
    all_maps: dict[str, tuple[str, str, float]] = {}
    all_maps.update(RUNTIME_API_MAP)
    all_maps.update(DRIVER_API_MAP)
    for k, v in ERROR_CODE_MAP.items():
        all_maps[k] = (v, "", 1.0)
    for k, v in DEFINE_MAP.items():
        all_maps[k] = (v, "", 1.0)
    return all_maps
