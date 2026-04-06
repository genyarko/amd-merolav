"""Sample CUDA script: multi-GPU with DDP and mixed precision."""

import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import autocast, GradScaler

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    model = nn.Linear(512, 10).to(device)
    model = DistributedDataParallel(model, device_ids=[rank])

    scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters())

    for step in range(100):
        x = torch.randn(64, 512, device=device)
        with autocast():
            out = model(x)
            loss = out.sum()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    dist.destroy_process_group()
