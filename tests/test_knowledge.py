"""Tests for the CUDA→ROCm knowledge base completeness and correctness."""

import pytest

from knowledge.cuda_rocm_map import (
    DEFINE_MAP,
    DRIVER_API_MAP,
    ENV_VAR_MAP,
    ERROR_CODE_MAP,
    HEADER_MAP,
    RUNTIME_API_MAP,
    get_all_mappings,
)
from knowledge.library_map import (
    LIBRARY_EQUIVALENTS,
    get_library_by_import,
    get_library_map,
)
from knowledge.optimizations import OPTIMIZATION_RULES, find_matching_optimizations
from knowledge.torch_cuda_map import (
    BACKENDS_REPLACE,
    IMPORT_MAP,
    TORCH_PASSTHROUGH,
    TORCH_WARNINGS,
)


# --- cuda_rocm_map.py ---


class TestCudaRocmMap:
    def test_runtime_api_has_entries(self):
        assert len(RUNTIME_API_MAP) >= 50, "Expected at least 50 runtime API mappings"

    def test_driver_api_has_entries(self):
        assert len(DRIVER_API_MAP) >= 10, "Expected at least 10 driver API mappings"

    def test_all_runtime_entries_have_hip_prefix(self):
        for cuda, (hip, _notes, _conf) in RUNTIME_API_MAP.items():
            assert hip.startswith("hip"), f"{cuda} maps to {hip} which lacks 'hip' prefix"

    def test_all_driver_entries_have_hip_prefix(self):
        for cuda, (hip, _notes, _conf) in DRIVER_API_MAP.items():
            assert hip.startswith("hip"), f"{cuda} maps to {hip} which lacks 'hip' prefix"

    def test_confidence_range(self):
        for cuda, (_hip, _notes, conf) in RUNTIME_API_MAP.items():
            assert 0.0 <= conf <= 1.0, f"{cuda} has invalid confidence {conf}"

    def test_error_codes_map_correctly(self):
        assert ERROR_CODE_MAP["cudaSuccess"] == "hipSuccess"
        assert "hipError" in ERROR_CODE_MAP["cudaErrorMemoryAllocation"] or \
               "hipError" in ERROR_CODE_MAP.get("cudaErrorMemoryAllocation", "")

    def test_define_map_memcpy_directions(self):
        assert DEFINE_MAP["cudaMemcpyHostToDevice"] == "hipMemcpyHostToDevice"
        assert DEFINE_MAP["cudaMemcpyDeviceToHost"] == "hipMemcpyDeviceToHost"
        assert DEFINE_MAP["cudaMemcpyDeviceToDevice"] == "hipMemcpyDeviceToDevice"

    def test_env_var_map(self):
        assert ENV_VAR_MAP["CUDA_VISIBLE_DEVICES"] == "HIP_VISIBLE_DEVICES"
        assert "CUDA_LAUNCH_BLOCKING" in ENV_VAR_MAP

    def test_header_map_has_common_headers(self):
        assert "cuda_runtime.h" in HEADER_MAP
        assert "cudnn.h" in HEADER_MAP
        assert HEADER_MAP["cudnn.h"] == "miopen/miopen.h"

    def test_get_all_mappings_combines_maps(self):
        all_maps = get_all_mappings()
        # Should contain entries from runtime, driver, error, and define maps
        assert "cudaMalloc" in all_maps
        assert "cuInit" in all_maps
        assert "cudaSuccess" in all_maps
        assert "cudaMemcpyHostToDevice" in all_maps

    def test_no_duplicate_keys_across_maps(self):
        runtime_keys = set(RUNTIME_API_MAP.keys())
        driver_keys = set(DRIVER_API_MAP.keys())
        overlap = runtime_keys & driver_keys
        assert len(overlap) == 0, f"Duplicate keys across runtime/driver maps: {overlap}"

    def test_essential_apis_present(self):
        essential = [
            "cudaMalloc", "cudaFree", "cudaMemcpy",
            "cudaDeviceSynchronize", "cudaSetDevice",
            "cudaStreamCreate", "cudaEventCreate",
            "cudaGetDeviceCount", "cudaGetLastError",
        ]
        for api in essential:
            assert api in RUNTIME_API_MAP, f"Essential API {api} missing from runtime map"


# --- torch_cuda_map.py ---


class TestTorchCudaMap:
    def test_passthrough_has_common_ops(self):
        assert "torch.cuda.is_available()" in TORCH_PASSTHROUGH
        assert ".cuda()" in TORCH_PASSTHROUGH

    def test_warnings_cover_cudnn(self):
        assert "torch.backends.cudnn" in TORCH_WARNINGS
        assert "torch.backends.cudnn.benchmark" in TORCH_WARNINGS

    def test_backends_replace_has_benchmark(self):
        assert "torch.backends.cudnn.benchmark = True" in BACKENDS_REPLACE
        assert "torch.backends.cudnn.deterministic = True" in BACKENDS_REPLACE

    def test_backends_replace_targets_miopen(self):
        det_replacement = BACKENDS_REPLACE["torch.backends.cudnn.deterministic = True"]
        assert "miopen" in det_replacement.lower()

    def test_import_map_has_pycuda(self):
        assert "import pycuda" in IMPORT_MAP

    def test_import_map_has_tensorrt(self):
        assert "import tensorrt" in IMPORT_MAP
        replacement, _note = IMPORT_MAP["import tensorrt"]
        assert "migraphx" in replacement.lower()


# --- library_map.py ---


class TestLibraryMap:
    def test_has_core_libraries(self):
        lib_map = get_library_map()
        for name in ["cuDNN", "cuBLAS", "cuFFT", "cuRAND", "NCCL", "TensorRT"]:
            assert name in lib_map, f"Core library {name} missing from map"

    def test_cudnn_maps_to_miopen(self):
        lib_map = get_library_map()
        assert lib_map["cuDNN"].rocm_name == "MIOpen"

    def test_nccl_maps_to_rccl(self):
        lib_map = get_library_map()
        assert lib_map["NCCL"].rocm_name == "RCCL"

    def test_tensorrt_maps_to_migraphx(self):
        lib_map = get_library_map()
        assert lib_map["TensorRT"].rocm_name == "MIGraphX"

    def test_no_equivalent_handled(self):
        lib_map = get_library_map()
        assert lib_map["cuDLA"].rocm_name is None

    def test_confidence_range(self):
        for lib in LIBRARY_EQUIVALENTS:
            assert 0.0 <= lib.confidence <= 1.0, f"{lib.cuda_name} has invalid confidence"

    def test_get_library_by_import_match(self):
        result = get_library_by_import("import cudnn")
        assert result is not None
        assert result.rocm_name == "MIOpen"

    def test_get_library_by_import_no_match(self):
        result = get_library_by_import("import numpy")
        assert result is None


# --- optimizations.py ---


class TestOptimizations:
    def test_has_rules(self):
        assert len(OPTIMIZATION_RULES) >= 5

    def test_all_rules_have_required_fields(self):
        for rule in OPTIMIZATION_RULES:
            assert rule.trigger, "Rule missing trigger"
            assert rule.suggestion, "Rule missing suggestion"
            assert rule.category, "Rule missing category"
            assert rule.priority in (1, 2, 3), f"Invalid priority {rule.priority}"

    def test_find_transformers_optimization(self):
        code = "from transformers import AutoModel"
        matches = find_matching_optimizations(code)
        triggers = [m.trigger for m in matches]
        assert "transformers" in triggers

    def test_find_cuda_visible_devices(self):
        code = 'os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"'
        matches = find_matching_optimizations(code)
        triggers = [m.trigger for m in matches]
        assert "CUDA_VISIBLE_DEVICES" in triggers

    def test_find_no_matches(self):
        code = "x = 1 + 2"
        matches = find_matching_optimizations(code)
        assert len(matches) == 0

    def test_results_sorted_by_priority(self):
        code = """
import transformers
import torch
torch.cuda.memory_allocated()
"""
        matches = find_matching_optimizations(code)
        priorities = [m.priority for m in matches]
        assert priorities == sorted(priorities), "Results should be sorted by priority"
