"""Tests for Phase 15 — Knowledge Base Expansion.

Covers: cuda_c_map, TensorRT→MIGraphX, version filtering, analyzer detection
of new patterns, and migrator handling of new mappings.
"""

from __future__ import annotations

import pytest

from knowledge.cuda_c_map import (
    CUB_API_MAP,
    COOPERATIVE_GROUPS_MAP,
    GRAPH_API_MAP,
    THRUST_API_MAP,
    WARP_INTRINSICS_MAP,
    get_all_cuda_c_mappings,
)
from knowledge.cuda_rocm_map import (
    TENSORRT_MIGRAPHX_MAP,
    get_all_mappings,
    _filter_by_rocm_version,
)
from knowledge.optimizations import find_matching_optimizations
from core.analyzer import analyze_source
from core.migrator import migrate


# =====================================================================
# cuda_c_map.py — mapping coverage
# =====================================================================

class TestThrustMappings:
    def test_has_core_algorithms(self):
        for key in ("thrust::sort", "thrust::reduce", "thrust::transform",
                     "thrust::copy", "thrust::fill"):
            assert key in THRUST_API_MAP

    def test_all_api_compatible(self):
        """rocThrust is API-compatible — all thrust:: symbols stay thrust::."""
        for cuda, (hip, _, _) in THRUST_API_MAP.items():
            if cuda.startswith("thrust::"):
                assert hip.startswith("thrust::"), f"{cuda} should map to thrust:: namespace"

    def test_confidence_is_1(self):
        for _, (_, _, conf) in THRUST_API_MAP.items():
            assert conf == 1.0


class TestCubMappings:
    def test_has_device_primitives(self):
        for key in ("cub::DeviceReduce", "cub::DeviceScan", "cub::DeviceRadixSort"):
            assert key in CUB_API_MAP

    def test_has_block_primitives(self):
        for key in ("cub::BlockReduce", "cub::BlockScan"):
            assert key in CUB_API_MAP

    def test_namespace_changes(self):
        """CUB maps from cub:: to hipcub::."""
        for cuda, (hip, _, _) in CUB_API_MAP.items():
            if cuda.startswith("cub::"):
                assert hip.startswith("hipcub::"), f"{cuda} should map to hipcub::"

    def test_header_changes(self):
        assert CUB_API_MAP["<cub/cub.cuh>"][0] == "<hipcub/hipcub.hpp>"

    def test_all_high_confidence(self):
        for _, (_, _, conf) in CUB_API_MAP.items():
            assert conf == 1.0


class TestGraphMappings:
    def test_has_lifecycle_apis(self):
        for key in ("cudaGraphCreate", "cudaGraphDestroy", "cudaGraphLaunch"):
            assert key in GRAPH_API_MAP

    def test_has_stream_capture(self):
        for key in ("cudaStreamBeginCapture", "cudaStreamEndCapture"):
            assert key in GRAPH_API_MAP

    def test_has_node_apis(self):
        for key in ("cudaGraphAddKernelNode", "cudaGraphAddMemcpyNode"):
            assert key in GRAPH_API_MAP

    def test_naming_pattern(self):
        """All cudaGraph* should map to hipGraph*."""
        for cuda, (hip, _, _) in GRAPH_API_MAP.items():
            if cuda.startswith("cudaGraph"):
                assert hip.startswith("hipGraph"), f"{cuda} → {hip}"

    def test_newer_apis_have_lower_confidence(self):
        # Memory allocation nodes are ROCm 6.0+
        _, _, conf = GRAPH_API_MAP["cudaGraphAddMemAllocNode"]
        assert conf < 1.0


class TestWarpIntrinsics:
    def test_has_shuffle_ops(self):
        for key in ("__shfl_sync", "__shfl_down_sync", "__shfl_up_sync", "__shfl_xor_sync"):
            assert key in WARP_INTRINSICS_MAP

    def test_has_ballot(self):
        assert "__ballot_sync" in WARP_INTRINSICS_MAP

    def test_sync_suffix_removed(self):
        """HIP warp intrinsics drop the _sync suffix."""
        for cuda, (hip, _, _) in WARP_INTRINSICS_MAP.items():
            if cuda.endswith("_sync"):
                assert not hip.endswith("_sync"), f"{cuda} → {hip} should drop _sync"

    def test_warp_size_flagged(self):
        hip, notes, conf = WARP_INTRINSICS_MAP["warpSize"]
        assert "64" in notes  # warns about wavefront size difference
        assert conf < 1.0  # needs review


class TestCooperativeGroups:
    def test_has_thread_block(self):
        assert "cooperative_groups::thread_block" in COOPERATIVE_GROUPS_MAP

    def test_has_header(self):
        assert "<cooperative_groups.h>" in COOPERATIVE_GROUPS_MAP

    def test_maps_to_hip_header(self):
        hip, _, _ = COOPERATIVE_GROUPS_MAP["<cooperative_groups.h>"]
        assert "hip" in hip


class TestGetAllCudaCMappings:
    def test_includes_all_submaps(self):
        all_maps = get_all_cuda_c_mappings()
        assert "thrust::sort" in all_maps
        assert "cub::DeviceReduce" in all_maps
        assert "cudaGraphCreate" in all_maps
        assert "__shfl_sync" in all_maps
        assert "cooperative_groups::thread_block" in all_maps

    def test_returns_tuple_format(self):
        for k, v in get_all_cuda_c_mappings().items():
            assert isinstance(v, tuple) and len(v) == 3


# =====================================================================
# TensorRT → MIGraphX
# =====================================================================

class TestTensorRTMappings:
    def test_has_builder(self):
        assert "tensorrt.Builder" in TENSORRT_MIGRAPHX_MAP
        assert "trt.Builder" in TENSORRT_MIGRAPHX_MAP

    def test_has_runtime(self):
        assert "tensorrt.Runtime" in TENSORRT_MIGRAPHX_MAP

    def test_all_low_confidence(self):
        """TensorRT→MIGraphX has different paradigms — should be low confidence."""
        for _, (_, _, conf) in TENSORRT_MIGRAPHX_MAP.items():
            assert conf <= 0.5

    def test_onnx_workflow_mentioned(self):
        _, notes, _ = TENSORRT_MIGRAPHX_MAP["tensorrt.Builder"]
        assert "ONNX" in notes or "onnx" in notes


# =====================================================================
# Version-aware filtering
# =====================================================================

class TestVersionFiltering:
    def test_no_filter_returns_all(self):
        all_maps = get_all_mappings()
        assert "cudaGraphAddMemAllocNode" in all_maps

    def test_filter_excludes_newer(self):
        """ROCm 5.0 should exclude ROCm 6.0+ APIs."""
        filtered = get_all_mappings(rocm_version="5.0")
        assert "cudaGraphAddMemAllocNode" not in filtered  # requires ROCm 6.0+

    def test_filter_includes_older(self):
        """ROCm 6.0 should include ROCm 5.5+ APIs."""
        filtered = get_all_mappings(rocm_version="6.0")
        assert "cudaGraphInstantiateWithFlags" in filtered  # ROCm 5.5+

    def test_filter_keeps_unversioned(self):
        """Mappings without version notes should always be included."""
        filtered = get_all_mappings(rocm_version="5.0")
        assert "cudaMalloc" in filtered
        assert "cudaFree" in filtered

    def test_bad_version_returns_all(self):
        filtered = get_all_mappings(rocm_version="not-a-version")
        assert "cudaMalloc" in filtered

    def test_filter_direct(self):
        mappings = {
            "api_old": ("hip_old", "ROCm 5.0+", 1.0),
            "api_new": ("hip_new", "ROCm 7.0+", 0.9),
            "api_none": ("hip_none", "", 1.0),
        }
        result = _filter_by_rocm_version(mappings, "6.0")
        assert "api_old" in result
        assert "api_new" not in result
        assert "api_none" in result


# =====================================================================
# Analyzer — detection of new patterns
# =====================================================================

class TestAnalyzerNewPatterns:
    def test_detects_cub_namespace(self):
        code = 'kernel = """\ncub::DeviceReduce::Sum(d_in, d_out, n);\n"""'
        report = analyze_source(code)
        symbols = [u.symbol for u in report.usages]
        assert any("cub::DeviceReduce" in s for s in symbols)

    def test_detects_thrust_namespace(self):
        code = 'code = "thrust::sort(d_vec.begin(), d_vec.end());"'
        report = analyze_source(code)
        symbols = [u.symbol for u in report.usages]
        assert any("thrust::sort" in s for s in symbols)

    def test_detects_warp_intrinsics(self):
        code = 'kernel_src = "__shfl_down_sync(0xffffffff, val, offset)"'
        report = analyze_source(code)
        symbols = [u.symbol for u in report.usages]
        assert "__shfl_down_sync" in symbols

    def test_detects_cuda_graph_api(self):
        code = "cudaGraphCreate(&graph, 0)"
        report = analyze_source(code)
        symbols = [u.symbol for u in report.usages]
        assert "cudaGraphCreate" in symbols

    def test_detects_cooperative_groups(self):
        code = 'src = "cooperative_groups::thread_block tb = cooperative_groups::this_thread_block();"'
        report = analyze_source(code)
        symbols = [u.symbol for u in report.usages]
        assert any("cooperative_groups::thread_block" in s for s in symbols)

    def test_detects_tensorrt_builder(self):
        code = "builder = trt.Builder(logger)"
        report = analyze_source(code)
        symbols = [u.symbol for u in report.usages]
        assert "trt.Builder" in symbols

    def test_detects_warp_size(self):
        code = 'kernel = "int lane = threadIdx.x % warpSize;"'
        report = analyze_source(code)
        symbols = [u.symbol for u in report.usages]
        assert "warpSize" in symbols


# =====================================================================
# Migrator — handling new mappings
# =====================================================================

class TestMigratorNewMappings:
    def test_migrates_cuda_graph_api(self):
        code = "cudaGraphCreate(&graph, 0)\ncudaGraphLaunch(exec, stream)\n"
        report = analyze_source(code)
        result = migrate(code, report)
        assert "hipGraphCreate" in result.code
        assert "hipGraphLaunch" in result.code

    def test_cub_flagged_as_remaining(self):
        """CUB symbols in inline strings are detected but may be flagged for LLM."""
        code = 'kernel = "cub::DeviceReduce::Sum(d_in, d_out, n);"\n'
        report = analyze_source(code)
        result = migrate(code, report)
        # Should either be applied or flagged as remaining
        all_symbols = (
            [c.rule for c in result.applied] +
            [r.symbol for r in result.remaining]
        )
        assert any("cub" in s.lower() or "DeviceReduce" in s for s in all_symbols)

    def test_warp_intrinsic_flagged(self):
        """Warp intrinsics should be detected with appropriate confidence."""
        code = '__shfl_down_sync(0xffffffff, val, 1)\n'
        report = analyze_source(code)
        result = migrate(code, report)
        # __shfl_down_sync has confidence 0.9 — should be applied or flagged
        all_items = result.applied + result.remaining
        assert len(all_items) > 0

    def test_version_filtering_excludes_in_migration(self):
        """With ROCm 5.0, newer graph APIs should not be auto-migrated."""
        code = "cudaGraphAddMemAllocNode(&node, graph, deps, numDeps, &params)\n"
        report = analyze_source(code)
        result = migrate(code, report, rocm_version="5.0")
        # Should be flagged as remaining (not in mappings for 5.0)
        remaining_symbols = [r.symbol for r in result.remaining]
        assert "cudaGraphAddMemAllocNode" in remaining_symbols


# =====================================================================
# Optimizations — new rules
# =====================================================================

class TestNewOptimizations:
    def test_cuda_graph_optimization(self):
        code = "cudaGraphCreate(&graph, 0)"
        matches = find_matching_optimizations(code)
        categories = [m.category for m in matches]
        assert "Graphs" in categories

    def test_cub_optimization(self):
        code = 'src = "cub::DeviceReduce::Sum(d_in, d_out, n);"'
        matches = find_matching_optimizations(code)
        categories = [m.category for m in matches]
        assert "Library" in categories

    def test_warp_size_optimization(self):
        code = 'kernel = "int lane = threadIdx.x % warpSize;"'
        matches = find_matching_optimizations(code)
        categories = [m.category for m in matches]
        assert "Architecture" in categories

    def test_warp_size_priority_is_high(self):
        code = "warpSize"
        matches = find_matching_optimizations(code)
        warp_match = [m for m in matches if m.category == "Architecture"]
        assert warp_match and warp_match[0].priority == 1
