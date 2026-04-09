"""Sample CUDA script: conditional imports and optional CUDA usage."""

import os

try:
    import pycuda.autoinit
    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False

try:
    import tensorrt as trt
    HAS_TRT = True
except ImportError:
    trt = None
    HAS_TRT = False

# Environment variables only set if CUDA is available
if HAS_PYCUDA:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Conditional use
if HAS_TRT:
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING))
