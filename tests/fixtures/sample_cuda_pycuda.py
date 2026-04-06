"""Sample CUDA script: pycuda usage with direct API calls."""

import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np

# Allocate memory on device
a = np.random.randn(1024).astype(np.float32)
a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)

# Some kernel work would happen here
# kernel_code = '''<<<grid, block>>>'''

cuda.memcpy_dtoh(a, a_gpu)
a_gpu.free()
