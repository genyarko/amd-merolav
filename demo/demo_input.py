"""Demo CUDA training script: ResNet fine-tuning with multi-GPU, mixed precision, cuDNN.

This is a realistic CUDA script designed to showcase the ROCm migration agent.
It contains 12+ CUDA-specific patterns that need migration.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel

# --- CUDA Environment Setup ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# --- cuDNN Configuration ---
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True


def setup_distributed(rank: int, world_size: int):
    """Initialize distributed training with NCCL backend."""
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)
    print(f"Rank {rank}: GPU {torch.cuda.get_device_name(rank)}")


def cleanup():
    dist.destroy_process_group()


class SimpleHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def train(rank: int, world_size: int, epochs: int = 10):
    setup_distributed(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Load a pretrained model
    backbone = AutoModel.from_pretrained("bert-base-uncased")
    model = nn.Sequential(backbone, SimpleHead(768, 10)).to(device)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scaler = GradScaler()

    # Dummy data
    dataset = TensorDataset(
        torch.randint(0, 30000, (1000, 128)),
        torch.randint(0, 10, (1000,)),
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (input_ids, labels) in enumerate(loader):
            input_ids = input_ids.cuda(rank)
            labels = labels.cuda(rank)

            optimizer.zero_grad()

            with autocast():
                outputs = model(input_ids)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        # Synchronize and log
        torch.cuda.synchronize()
        mem_allocated = torch.cuda.memory_allocated(device) / 1e9
        mem_reserved = torch.cuda.memory_reserved(device) / 1e9

        if rank == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"GPU Mem: {mem_allocated:.2f}GB allocated, "
                  f"{mem_reserved:.2f}GB reserved")

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
