"""CUDA C/C++ to HIP/ROCm mappings — types, qualifiers, libraries, intrinsics.

Sources:
- https://rocm.docs.amd.com/projects/HIPIFY/en/latest/
- https://rocm.docs.amd.com/projects/hipCUB/en/latest/
- https://rocm.docs.amd.com/projects/rocThrust/en/latest/
- https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CUDA C qualifiers, types, and kernel launch syntax
# These are found in .cu/.cuh files and inline CUDA C strings in Python.
# ---------------------------------------------------------------------------

CUDA_C_QUALIFIERS_MAP: dict[str, tuple[str, str, float]] = {
    # Qualifiers — identical in HIP
    "__global__": ("__global__", "Same in HIP", 1.0),
    "__device__": ("__device__", "Same in HIP", 1.0),
    "__host__": ("__host__", "Same in HIP", 1.0),
    "__shared__": ("__shared__", "Same in HIP", 1.0),
    "__constant__": ("__constant__", "Same in HIP", 1.0),
    "__managed__": ("__managed__", "Same in HIP; requires managed memory support", 0.9),
    "__restrict__": ("__restrict__", "Same in HIP", 1.0),
    "__forceinline__": ("__forceinline__", "Same in HIP", 1.0),
    "__launch_bounds__": ("__launch_bounds__", "Same in HIP", 1.0),
}

CUDA_C_TYPES_MAP: dict[str, tuple[str, str, float]] = {
    # Types — most have direct HIP equivalents
    "cudaStream_t": ("hipStream_t", "", 1.0),
    "cudaEvent_t": ("hipEvent_t", "", 1.0),
    "cudaError_t": ("hipError_t", "", 1.0),
    "cudaDeviceProp": ("hipDeviceProp_t", "Note: _t suffix in HIP", 1.0),
    "cudaMemcpyKind": ("hipMemcpyKind", "", 1.0),
    "cudaFuncAttributes": ("hipFuncAttributes", "", 1.0),
    "cudaPointerAttributes": ("hipPointerAttribute_t", "", 1.0),
    "cudaChannelFormatDesc": ("hipChannelFormatDesc", "", 1.0),
    "cudaTextureObject_t": ("hipTextureObject_t", "", 0.8),
    "cudaSurfaceObject_t": ("hipSurfaceObject_t", "Limited support", 0.6),
    "cudaArray_t": ("hipArray_t", "", 0.9),
    "cudaPitchedPtr": ("hipPitchedPtr", "", 1.0),
    "cudaExtent": ("hipExtent", "", 1.0),
    "cudaPos": ("hipPos", "", 1.0),
    "cudaGraphicsResource_t": ("hipGraphicsResource_t", "Limited interop support", 0.5),
    # dim3 is the same in HIP
    "dim3": ("dim3", "Same in HIP", 1.0),
    # CUDA graph types
    "cudaGraph_t": ("hipGraph_t", "", 1.0),
    "cudaGraphExec_t": ("hipGraphExec_t", "", 1.0),
    "cudaGraphNode_t": ("hipGraphNode_t", "", 1.0),
}

CUDA_C_LAUNCH_MAP: dict[str, tuple[str, str, float]] = {
    # Kernel launch syntax — the triple-chevron syntax is NOT valid in HIP C++.
    # HIP uses hipLaunchKernelGGL() macro instead.
    "<<<...>>>": (
        "hipLaunchKernelGGL(kernel, gridDim, blockDim, sharedMem, stream, ...)",
        "HIP does not support <<<>>> syntax; use hipLaunchKernelGGL() macro. "
        "Format: hipLaunchKernelGGL(kernel, grid, block, sharedMem, stream, args...)",
        0.5,
    ),
}


# ---------------------------------------------------------------------------
# Thrust → rocThrust
# rocThrust is API-compatible — same headers, same namespace. The migration
# is mostly about build-system changes (linking rocThrust instead of Thrust).
# ---------------------------------------------------------------------------

THRUST_API_MAP: dict[str, tuple[str, str, float]] = {
    # Headers (used in CUDA C strings / pycuda SourceModule / inline code)
    "<thrust/device_vector.h>": ("<thrust/device_vector.h>", "rocThrust is API-compatible; same header", 1.0),
    "<thrust/host_vector.h>": ("<thrust/host_vector.h>", "rocThrust is API-compatible; same header", 1.0),
    "<thrust/sort.h>": ("<thrust/sort.h>", "rocThrust is API-compatible", 1.0),
    "<thrust/reduce.h>": ("<thrust/reduce.h>", "rocThrust is API-compatible", 1.0),
    "<thrust/transform.h>": ("<thrust/transform.h>", "rocThrust is API-compatible", 1.0),
    "<thrust/scan.h>": ("<thrust/scan.h>", "rocThrust is API-compatible", 1.0),
    "<thrust/copy.h>": ("<thrust/copy.h>", "rocThrust is API-compatible", 1.0),
    "<thrust/fill.h>": ("<thrust/fill.h>", "rocThrust is API-compatible", 1.0),
    "<thrust/functional.h>": ("<thrust/functional.h>", "rocThrust is API-compatible", 1.0),
    "<thrust/execution_policy.h>": ("<thrust/execution_policy.h>", "rocThrust is API-compatible", 1.0),
    # Namespace — thrust:: works as-is with rocThrust
    "thrust::device_vector": ("thrust::device_vector", "Works with rocThrust (API-compatible)", 1.0),
    "thrust::host_vector": ("thrust::host_vector", "Works with rocThrust (API-compatible)", 1.0),
    "thrust::sort": ("thrust::sort", "Works with rocThrust", 1.0),
    "thrust::stable_sort": ("thrust::stable_sort", "Works with rocThrust", 1.0),
    "thrust::reduce": ("thrust::reduce", "Works with rocThrust", 1.0),
    "thrust::transform": ("thrust::transform", "Works with rocThrust", 1.0),
    "thrust::inclusive_scan": ("thrust::inclusive_scan", "Works with rocThrust", 1.0),
    "thrust::exclusive_scan": ("thrust::exclusive_scan", "Works with rocThrust", 1.0),
    "thrust::copy": ("thrust::copy", "Works with rocThrust", 1.0),
    "thrust::fill": ("thrust::fill", "Works with rocThrust", 1.0),
    "thrust::count": ("thrust::count", "Works with rocThrust", 1.0),
    "thrust::find": ("thrust::find", "Works with rocThrust", 1.0),
    "thrust::for_each": ("thrust::for_each", "Works with rocThrust", 1.0),
    "thrust::min_element": ("thrust::min_element", "Works with rocThrust", 1.0),
    "thrust::max_element": ("thrust::max_element", "Works with rocThrust", 1.0),
}

# ---------------------------------------------------------------------------
# CUB → hipCUB
# hipCUB wraps CUB with a HIP backend. Namespace changes from cub:: to
# hipcub::, and headers change from <cub/...> to <hipcub/...>.
# ---------------------------------------------------------------------------

CUB_API_MAP: dict[str, tuple[str, str, float]] = {
    # Headers
    "<cub/cub.cuh>": ("<hipcub/hipcub.hpp>", "CUB → hipCUB header", 1.0),
    "<cub/device/device_reduce.cuh>": ("<hipcub/device/device_reduce.hpp>", "", 1.0),
    "<cub/device/device_scan.cuh>": ("<hipcub/device/device_scan.hpp>", "", 1.0),
    "<cub/device/device_select.cuh>": ("<hipcub/device/device_select.hpp>", "", 1.0),
    "<cub/device/device_sort.cuh>": ("<hipcub/device/device_radix_sort.hpp>", "", 1.0),
    "<cub/device/device_histogram.cuh>": ("<hipcub/device/device_histogram.hpp>", "", 1.0),
    "<cub/device/device_run_length_encode.cuh>": ("<hipcub/device/device_run_length_encode.hpp>", "", 1.0),
    "<cub/block/block_reduce.cuh>": ("<hipcub/block/block_reduce.hpp>", "", 1.0),
    "<cub/block/block_scan.cuh>": ("<hipcub/block/block_scan.hpp>", "", 1.0),
    "<cub/block/block_load.cuh>": ("<hipcub/block/block_load.hpp>", "", 1.0),
    "<cub/block/block_store.cuh>": ("<hipcub/block/block_store.hpp>", "", 1.0),
    "<cub/warp/warp_reduce.cuh>": ("<hipcub/warp/warp_reduce.hpp>", "", 1.0),
    "<cub/warp/warp_scan.cuh>": ("<hipcub/warp/warp_scan.hpp>", "", 1.0),
    # Namespace replacements
    "cub::DeviceReduce": ("hipcub::DeviceReduce", "", 1.0),
    "cub::DeviceScan": ("hipcub::DeviceScan", "", 1.0),
    "cub::DeviceSelect": ("hipcub::DeviceSelect", "", 1.0),
    "cub::DeviceRadixSort": ("hipcub::DeviceRadixSort", "", 1.0),
    "cub::DeviceHistogram": ("hipcub::DeviceHistogram", "", 1.0),
    "cub::DeviceRunLengthEncode": ("hipcub::DeviceRunLengthEncode", "", 1.0),
    "cub::DeviceSegmentedReduce": ("hipcub::DeviceSegmentedReduce", "", 1.0),
    "cub::BlockReduce": ("hipcub::BlockReduce", "", 1.0),
    "cub::BlockScan": ("hipcub::BlockScan", "", 1.0),
    "cub::BlockLoad": ("hipcub::BlockLoad", "", 1.0),
    "cub::BlockStore": ("hipcub::BlockStore", "", 1.0),
    "cub::WarpReduce": ("hipcub::WarpReduce", "", 1.0),
    "cub::WarpScan": ("hipcub::WarpScan", "", 1.0),
}

# ---------------------------------------------------------------------------
# CUDA Graphs → HIP Graphs
# HIP Graphs mirror the CUDA Graphs API. Most functions are a direct
# cuda→hip rename.  Some newer CUDA 12 features may lag on ROCm.
# ---------------------------------------------------------------------------

GRAPH_API_MAP: dict[str, tuple[str, str, float]] = {
    # Graph lifecycle
    "cudaGraphCreate": ("hipGraphCreate", "", 1.0),
    "cudaGraphDestroy": ("hipGraphDestroy", "", 1.0),
    "cudaGraphInstantiate": ("hipGraphInstantiate", "", 1.0),
    "cudaGraphInstantiateWithFlags": ("hipGraphInstantiateWithFlags", "ROCm 5.5+", 0.9),
    "cudaGraphLaunch": ("hipGraphLaunch", "", 1.0),
    "cudaGraphExecDestroy": ("hipGraphExecDestroy", "", 1.0),
    "cudaGraphExecUpdate": ("hipGraphExecUpdate", "ROCm 5.5+", 0.9),
    # Stream capture
    "cudaStreamBeginCapture": ("hipStreamBeginCapture", "", 1.0),
    "cudaStreamEndCapture": ("hipStreamEndCapture", "", 1.0),
    "cudaStreamIsCapturing": ("hipStreamIsCapturing", "", 1.0),
    # Graph nodes
    "cudaGraphAddKernelNode": ("hipGraphAddKernelNode", "", 1.0),
    "cudaGraphAddMemcpyNode": ("hipGraphAddMemcpyNode", "", 1.0),
    "cudaGraphAddMemsetNode": ("hipGraphAddMemsetNode", "", 1.0),
    "cudaGraphAddHostNode": ("hipGraphAddHostNode", "", 1.0),
    "cudaGraphAddChildGraphNode": ("hipGraphAddChildGraphNode", "", 1.0),
    "cudaGraphAddEmptyNode": ("hipGraphAddEmptyNode", "", 1.0),
    "cudaGraphAddDependencies": ("hipGraphAddDependencies", "", 1.0),
    "cudaGraphAddEventRecordNode": ("hipGraphAddEventRecordNode", "ROCm 5.5+", 0.9),
    "cudaGraphAddEventWaitNode": ("hipGraphAddEventWaitNode", "ROCm 5.5+", 0.9),
    # Graph query
    "cudaGraphGetNodes": ("hipGraphGetNodes", "", 1.0),
    "cudaGraphGetEdges": ("hipGraphGetEdges", "", 1.0),
    "cudaGraphNodeGetType": ("hipGraphNodeGetType", "", 1.0),
    "cudaGraphGetRootNodes": ("hipGraphGetRootNodes", "", 1.0),
    # Memory allocation in graphs (CUDA 11.4+, partial HIP support)
    "cudaGraphAddMemAllocNode": ("hipGraphAddMemAllocNode", "ROCm 6.0+; check support", 0.7),
    "cudaGraphAddMemFreeNode": ("hipGraphAddMemFreeNode", "ROCm 6.0+; check support", 0.7),
}

# ---------------------------------------------------------------------------
# Cooperative Groups & Warp Intrinsics
#
# CRITICAL: AMD wavefront size is 64 on MI200/MI300 (vs. NVIDIA warp size 32).
# Code that hardcodes warp_size=32 WILL produce wrong results on AMD.
# ---------------------------------------------------------------------------

WARP_INTRINSICS_MAP: dict[str, tuple[str, str, float]] = {
    # Warp shuffle — HIP drops the _sync suffix (no explicit mask needed)
    "__shfl_sync": ("__shfl", "HIP: no sync suffix; mask parameter ignored", 0.9),
    "__shfl_up_sync": ("__shfl_up", "HIP: no sync suffix", 0.9),
    "__shfl_down_sync": ("__shfl_down", "HIP: no sync suffix", 0.9),
    "__shfl_xor_sync": ("__shfl_xor", "HIP: no sync suffix", 0.9),
    # Ballot / vote
    "__ballot_sync": ("__ballot", "HIP: no sync suffix; returns 64-bit on AMD (wavefront=64)", 0.8),
    "__all_sync": ("__all", "HIP: no sync suffix", 0.9),
    "__any_sync": ("__any", "HIP: no sync suffix", 0.9),
    # Warp-level primitives
    "__activemask": ("__ballot(1)", "HIP equivalent; 64-bit on AMD", 0.8),
    "__syncwarp": ("__syncthreads", "HIP: use __syncthreads (no sub-warp sync)", 0.7),
    # Lane / warp size
    "warpSize": ("warpSize", "CRITICAL: 64 on AMD MI200/MI300 (vs 32 on NVIDIA). Check hardcoded warp_size=32!", 0.6),
}

COOPERATIVE_GROUPS_MAP: dict[str, tuple[str, str, float]] = {
    # Headers
    "<cooperative_groups.h>": ("<hip/hip_cooperative_groups.h>", "", 1.0),
    "<cooperative_groups/reduce.h>": ("<hip/hip_cooperative_groups.h>", "Combined header on HIP", 0.9),
    # Namespace
    "cooperative_groups::thread_block": ("cooperative_groups::thread_block", "Works on HIP", 1.0),
    "cooperative_groups::thread_block_tile": ("cooperative_groups::thread_block_tile", "Works on HIP; tile size must be ≤ wavefront size (64)", 0.9),
    "cooperative_groups::this_thread_block": ("cooperative_groups::this_thread_block", "Works on HIP", 1.0),
    "cooperative_groups::this_grid": ("cooperative_groups::this_grid", "Works on HIP", 1.0),
    "cooperative_groups::tiled_partition": ("cooperative_groups::tiled_partition", "Works on HIP; check tile size vs wavefront", 0.9),
    "cooperative_groups::coalesced_threads": ("cooperative_groups::coalesced_threads", "Works on HIP", 0.9),
}


# ---------------------------------------------------------------------------
# Aggregated accessor
# ---------------------------------------------------------------------------

def get_all_cuda_c_mappings() -> dict[str, tuple[str, str, float]]:
    """Return all CUDA C/C++ mappings combined (qualifiers, types, libraries, intrinsics)."""
    all_maps: dict[str, tuple[str, str, float]] = {}
    all_maps.update(CUDA_C_QUALIFIERS_MAP)
    all_maps.update(CUDA_C_TYPES_MAP)
    all_maps.update(CUDA_C_LAUNCH_MAP)
    all_maps.update(THRUST_API_MAP)
    all_maps.update(CUB_API_MAP)
    all_maps.update(GRAPH_API_MAP)
    all_maps.update(WARP_INTRINSICS_MAP)
    all_maps.update(COOPERATIVE_GROUPS_MAP)
    return all_maps
