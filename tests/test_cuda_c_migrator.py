"""Tests for Phase 10 — CUDA C/C++ Kernel Support.

Covers: cuda_c_migrator (header, type, API, kernel launch, namespace replacements),
inline CUDA C migration in Python strings, analyzer detection of C patterns,
and .cu/.cuh file handling.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.cuda_c_migrator import (
    CudaCMigrationResult,
    migrate_cuda_c_file,
    migrate_inline_cuda_c,
    check_hipify_available,
)
from core.analyzer import analyze_source
from core.file_io import collect_python_files


# =====================================================================
# Header replacement
# =====================================================================

class TestHeaderReplacement:
    def test_cuda_runtime_h(self):
        code = '#include <cuda_runtime.h>\nint main() {}\n'
        result = migrate_cuda_c_file(code)
        assert "hip/hip_runtime.h" in result.code
        assert "cuda_runtime.h" not in result.code

    def test_cudnn_h(self):
        code = '#include "cudnn.h"\n'
        result = migrate_cuda_c_file(code)
        assert "miopen/miopen.h" in result.code

    def test_cublas_h(self):
        code = '#include <cublas_v2.h>\n'
        result = migrate_cuda_c_file(code)
        assert "rocblas/rocblas.h" in result.code

    def test_cub_header(self):
        code = '#include <cub/cub.cuh>\n'
        result = migrate_cuda_c_file(code)
        assert "hipcub/hipcub.hpp" in result.code

    def test_cub_subdir_header(self):
        code = '#include <cub/device/device_reduce.cuh>\n'
        result = migrate_cuda_c_file(code)
        assert "hipcub/device/device_reduce.hpp" in result.code

    def test_non_cuda_header_unchanged(self):
        code = '#include <stdio.h>\n#include <stdlib.h>\n'
        result = migrate_cuda_c_file(code)
        assert result.code == code

    def test_applied_change_logged(self):
        result = migrate_cuda_c_file('#include <cuda_runtime.h>\n')
        assert len(result.applied) >= 1
        assert any("Header" in c.rule for c in result.applied)


# =====================================================================
# Type replacement
# =====================================================================

class TestTypeReplacement:
    def test_cuda_stream_t(self):
        code = "cudaStream_t stream;\n"
        result = migrate_cuda_c_file(code)
        assert "hipStream_t" in result.code

    def test_cuda_event_t(self):
        code = "cudaEvent_t start, stop;\n"
        result = migrate_cuda_c_file(code)
        assert "hipEvent_t" in result.code

    def test_cuda_error_t(self):
        code = "cudaError_t err = cudaMalloc(&ptr, size);\n"
        result = migrate_cuda_c_file(code)
        assert "hipError_t" in result.code

    def test_cuda_device_prop(self):
        code = "cudaDeviceProp prop;\n"
        result = migrate_cuda_c_file(code)
        assert "hipDeviceProp_t" in result.code

    def test_dim3_unchanged(self):
        code = "dim3 grid(16, 16);\ndim3 block(256);\n"
        result = migrate_cuda_c_file(code)
        assert "dim3" in result.code
        # dim3 should NOT be changed — same in HIP
        assert result.code.count("dim3") == 2

    def test_low_confidence_type_flagged(self):
        code = "cudaGraphicsResource_t res;\n"
        result = migrate_cuda_c_file(code)
        assert any("cudaGraphicsResource_t" in r.symbol for r in result.remaining)


# =====================================================================
# Runtime API replacement
# =====================================================================

class TestRuntimeAPIReplacement:
    def test_cuda_malloc(self):
        code = "cudaMalloc(&d_ptr, size);\n"
        result = migrate_cuda_c_file(code)
        assert "hipMalloc" in result.code

    def test_cuda_memcpy(self):
        code = "cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);\n"
        result = migrate_cuda_c_file(code)
        assert "hipMemcpy" in result.code

    def test_cuda_free(self):
        code = "cudaFree(d_ptr);\n"
        result = migrate_cuda_c_file(code)
        assert "hipFree" in result.code

    def test_cuda_device_synchronize(self):
        code = "cudaDeviceSynchronize();\n"
        result = migrate_cuda_c_file(code)
        assert "hipDeviceSynchronize" in result.code

    def test_multiple_apis_in_one_line(self):
        code = "cudaError_t err = cudaMalloc(&ptr, size);\n"
        result = migrate_cuda_c_file(code)
        assert "hipError_t" in result.code
        assert "hipMalloc" in result.code


# =====================================================================
# Kernel launch syntax
# =====================================================================

class TestKernelLaunchConversion:
    def test_basic_launch(self):
        code = "myKernel<<<grid, block>>>(a, b, c);\n"
        result = migrate_cuda_c_file(code)
        assert "hipLaunchKernelGGL" in result.code
        assert "myKernel" in result.code
        assert "<<<" not in result.code

    def test_launch_with_shared_mem(self):
        code = "myKernel<<<grid, block, sharedMem>>>(a, b);\n"
        result = migrate_cuda_c_file(code)
        assert "hipLaunchKernelGGL" in result.code
        assert "sharedMem" in result.code

    def test_launch_with_stream(self):
        code = "myKernel<<<grid, block, 0, stream>>>(a, b);\n"
        result = migrate_cuda_c_file(code)
        assert "hipLaunchKernelGGL" in result.code
        assert "stream" in result.code

    def test_launch_no_args(self):
        code = "initKernel<<<1, 1>>>();\n"
        result = migrate_cuda_c_file(code)
        assert "hipLaunchKernelGGL" in result.code

    def test_launch_logged(self):
        code = "kernel<<<grid, block>>>(x);\n"
        result = migrate_cuda_c_file(code)
        assert any("Kernel launch" in c.rule for c in result.applied)


# =====================================================================
# CUB namespace
# =====================================================================

class TestCubNamespace:
    def test_cub_device_reduce(self):
        code = "cub::DeviceReduce::Sum(d_in, d_out, n);\n"
        result = migrate_cuda_c_file(code)
        assert "hipcub::DeviceReduce" in result.code
        # All bare cub:: should be replaced — only hipcub:: should remain
        import re
        assert not re.search(r"(?<!hip)cub::", result.code)

    def test_cub_block_reduce(self):
        code = "cub::BlockReduce<float, 256> reduce;\n"
        result = migrate_cuda_c_file(code)
        assert "hipcub::BlockReduce" in result.code

    def test_multiple_cub_refs(self):
        code = (
            "cub::DeviceReduce::Sum(d_in, d_out, n);\n"
            "cub::DeviceScan::InclusiveSum(d_in, d_out, n);\n"
        )
        result = migrate_cuda_c_file(code)
        assert result.code.count("hipcub::") == 2
        import re
        assert not re.search(r"(?<!hip)cub::", result.code)


# =====================================================================
# Full .cu file migration
# =====================================================================

class TestFullCuFileMigration:
    SAMPLE_CU = """\
#include <cuda_runtime.h>
#include <cub/cub.cuh>

__global__ void addKernel(float* a, float* b, float* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    dim3 block(256);
    dim3 grid((N + 255) / 256);
    addKernel<<<grid, block>>>(d_a, d_b, d_c, N);

    cudaDeviceSynchronize();
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
"""

    def test_headers_replaced(self):
        result = migrate_cuda_c_file(self.SAMPLE_CU)
        assert "hip/hip_runtime.h" in result.code
        assert "hipcub/hipcub.hpp" in result.code

    def test_apis_replaced(self):
        result = migrate_cuda_c_file(self.SAMPLE_CU)
        assert "hipMalloc" in result.code
        assert "hipDeviceSynchronize" in result.code
        assert "hipFree" in result.code

    def test_kernel_launch_replaced(self):
        result = migrate_cuda_c_file(self.SAMPLE_CU)
        assert "hipLaunchKernelGGL" in result.code
        assert "<<<" not in result.code

    def test_qualifiers_unchanged(self):
        """__global__ stays the same in HIP."""
        result = migrate_cuda_c_file(self.SAMPLE_CU)
        assert "__global__" in result.code

    def test_dim3_unchanged(self):
        result = migrate_cuda_c_file(self.SAMPLE_CU)
        assert "dim3" in result.code

    def test_has_applied_changes(self):
        result = migrate_cuda_c_file(self.SAMPLE_CU)
        assert len(result.applied) >= 6  # headers + APIs + kernel launch

    def test_warnings_for_remaining(self):
        result = migrate_cuda_c_file(self.SAMPLE_CU)
        if result.remaining:
            assert any("hipify" in w.lower() for w in result.warnings)


# =====================================================================
# Inline CUDA C in Python strings
# =====================================================================

class TestInlineCudaCMigration:
    SAMPLE_PYTHON = '''\
from pycuda.compiler import SourceModule

kernel_code = """
#include <cuda_runtime.h>

__global__ void add(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}
"""

mod = SourceModule(kernel_code)
'''

    def test_migrates_header_in_string(self):
        result = migrate_inline_cuda_c(self.SAMPLE_PYTHON)
        assert "hip/hip_runtime.h" in result.code
        assert "cuda_runtime.h" not in result.code

    def test_preserves_python_code(self):
        result = migrate_inline_cuda_c(self.SAMPLE_PYTHON)
        assert "from pycuda.compiler import SourceModule" in result.code
        assert "mod = SourceModule(kernel_code)" in result.code

    def test_skips_non_cuda_strings(self):
        code = '"""This is a docstring about nothing."""\nx = 1\n'
        result = migrate_inline_cuda_c(code)
        assert result.code == code
        assert len(result.applied) == 0

    def test_single_quotes(self):
        code = "kernel = '''\n#include <cuda_runtime.h>\n__global__ void k() {}\n'''\n"
        result = migrate_inline_cuda_c(code)
        assert "hip/hip_runtime.h" in result.code

    def test_applied_changes_logged(self):
        result = migrate_inline_cuda_c(self.SAMPLE_PYTHON)
        assert len(result.applied) >= 1


# =====================================================================
# Analyzer — CUDA C pattern detection
# =====================================================================

class TestAnalyzerCudaCPatterns:
    def test_detects_global_qualifier(self):
        code = 'kernel = "__global__ void myKernel() {}"\n'
        report = analyze_source(code)
        symbols = [u.symbol for u in report.usages]
        assert "__global__" in symbols

    def test_detects_shared_qualifier(self):
        code = 'code = "__shared__ float data[256];"\n'
        report = analyze_source(code)
        symbols = [u.symbol for u in report.usages]
        assert "__shared__" in symbols

    def test_detects_device_qualifier(self):
        code = 'code = "__device__ int helper(int x) { return x; }"\n'
        report = analyze_source(code)
        symbols = [u.symbol for u in report.usages]
        assert "__device__" in symbols

    def test_detects_cuda_stream_t(self):
        code = "cudaStream_t stream;\n"
        report = analyze_source(code)
        symbols = [u.symbol for u in report.usages]
        assert "cudaStream_t" in symbols

    def test_detects_cuda_event_t(self):
        code = "cudaEvent_t ev;\n"
        report = analyze_source(code)
        symbols = [u.symbol for u in report.usages]
        assert "cudaEvent_t" in symbols

    def test_detects_cuda_header_include(self):
        code = '#include <cuda_runtime.h>\n'
        report = analyze_source(code)
        categories = [u.category for u in report.usages]
        assert "cuda_c_header" in categories

    def test_detects_source_module(self):
        code = 'mod = SourceModule(kernel_code)\n'
        report = analyze_source(code)
        categories = [u.category for u in report.usages]
        assert "inline_cuda_c" in categories

    def test_categories_distinct(self):
        """Qualifiers, types, and headers use distinct categories."""
        code = (
            '#include <cuda_runtime.h>\n'
            'cudaStream_t s;\n'
            '__global__ void k() {}\n'
        )
        report = analyze_source(code)
        categories = {u.category for u in report.usages}
        assert "cuda_c_header" in categories
        assert "cuda_c_type" in categories
        assert "cuda_c_qualifier" in categories


# =====================================================================
# File collection with .cu/.cuh
# =====================================================================

class TestFileCollection:
    def test_accepts_cu_file(self, tmp_path: Path):
        cu_file = tmp_path / "kernel.cu"
        cu_file.write_text("__global__ void k() {}\n")
        files = collect_python_files(str(cu_file))
        assert len(files) == 1
        assert files[0].suffix == ".cu"

    def test_accepts_cuh_file(self, tmp_path: Path):
        cuh_file = tmp_path / "header.cuh"
        cuh_file.write_text("#pragma once\n")
        files = collect_python_files(str(cuh_file))
        assert len(files) == 1

    def test_collects_all_types_from_dir(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("import torch\n")
        (tmp_path / "kernel.cu").write_text("__global__ void k() {}\n")
        (tmp_path / "utils.cuh").write_text("#pragma once\n")
        files = collect_python_files(str(tmp_path))
        extensions = {f.suffix for f in files}
        assert extensions == {".py", ".cu", ".cuh"}

    def test_rejects_unsupported_extension(self, tmp_path: Path):
        txt_file = tmp_path / "readme.txt"
        txt_file.write_text("hello\n")
        with pytest.raises(ValueError, match="Expected"):
            collect_python_files(str(txt_file))


# =====================================================================
# Version-aware migration
# =====================================================================

class TestVersionAwareCMigration:
    def test_older_rocm_skips_newer_graph_apis(self):
        code = "cudaGraphAddMemAllocNode(&node, graph, deps, numDeps, &params);\n"
        result = migrate_cuda_c_file(code, rocm_version="5.0")
        # Should be flagged as remaining, not auto-replaced
        remaining_symbols = [r.symbol for r in result.remaining]
        assert "cudaGraphAddMemAllocNode" in remaining_symbols

    def test_newer_rocm_includes_graph_apis(self):
        code = "cudaGraphCreate(&graph, 0);\n"
        result = migrate_cuda_c_file(code, rocm_version="6.0")
        assert "hipGraphCreate" in result.code
