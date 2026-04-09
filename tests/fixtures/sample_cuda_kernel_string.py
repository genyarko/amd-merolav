"""Sample CUDA script: inline CUDA C kernel in a Python string."""

import numpy as np

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except ImportError:
    cuda = None
    SourceModule = None

kernel_code = """
__global__ void vector_add(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
"""

if SourceModule is not None:
    mod = SourceModule(kernel_code)
    vector_add = mod.get_function("vector_add")

    n = 1024
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    c = np.zeros(n, dtype=np.float32)

    vector_add(
        cuda.In(a), cuda.In(b), cuda.Out(c), np.int32(n),
        block=(256, 1, 1), grid=(n // 256, 1),
    )
