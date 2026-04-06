"""Sample CUDA script: simple single-GPU training loop."""

import os
import torch
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.Linear(10, 2).cuda()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    x = torch.randn(32, 10).cuda()
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print(f"Memory allocated: {torch.cuda.memory_allocated(device)}")
torch.cuda.synchronize()
