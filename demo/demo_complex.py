"""Complex CUDA demo: custom kernels, pycuda, NVTX profiling, cuDNN handles.

This script contains patterns that require LLM reasoning to migrate —
rule-based substitution alone is insufficient.
"""

import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    import pycuda as _pycuda
    cuda = _pycuda.driver
    from pycuda.compiler import SourceModule
    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False
    cuda = None
    SourceModule = None

try:
    import torch.cuda.nvtx as nvtx
    HAS_NVTX = True
except ImportError:
    HAS_NVTX = False

# --- Environment ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# --- cuDNN tuning ---
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.allow_tf32 = True

# --- Custom CUDA kernel (vector addition) ---
CUDA_KERNEL = """
__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""


def run_custom_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Run a custom CUDA kernel for vector addition."""
    if not HAS_PYCUDA:
        return a + b  # fallback

    mod = SourceModule(CUDA_KERNEL)
    vector_add = mod.get_function("vector_add")

    n = a.numel()
    c = torch.zeros_like(a)

    block_size = 256
    grid_size = (n + block_size - 1) // block_size

    vector_add(
        cuda.In(a.numpy()),
        cuda.In(b.numpy()),
        cuda.Out(c.numpy()),
        n,
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
    )
    return c


class AttentionBlock(nn.Module):
    """Multi-head attention with flash attention hint."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # cuDNN flash attention requires specific memory layout
        torch.backends.cudnn.flash_sdp_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if HAS_NVTX:
            nvtx.range_push("attention_forward")
        out, _ = self.attn(x, x, x)
        if HAS_NVTX:
            nvtx.range_pop()
        return out


def train_loop(model: nn.Module, loader: DataLoader, epochs: int = 3):
    device = torch.device("cuda:0")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        if HAS_NVTX:
            nvtx.range_push(f"epoch_{epoch}")

        for x, y in loader:
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            with autocast():
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Memory stats
        alloc = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        torch.cuda.reset_peak_memory_stats(device)

        print(f"Epoch {epoch+1}: loss={loss.item():.4f} | "
              f"alloc={alloc:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB")

        if HAS_NVTX:
            nvtx.range_pop()

    # Synchronize all streams
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Test custom kernel
    a = torch.randn(1024)
    b = torch.randn(1024)
    c = run_custom_kernel(a, b)
    print(f"Custom kernel output norm: {c.norm():.4f}")

    # Test model training
    model = AttentionBlock(embed_dim=64, num_heads=4)
    dataset = TensorDataset(
        torch.randn(200, 16, 64),
        torch.randn(200, 16, 64),
    )
    loader = DataLoader(dataset, batch_size=16)
    train_loop(model, loader)
