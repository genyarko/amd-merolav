"""Sample CUDA script: larger file with many patterns to stress-test the pipeline."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return torch.relu(out)


class DeepModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, num_blocks=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_blocks)]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def setup_distributed(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    dist.destroy_process_group()


def create_dataloaders(batch_size=64):
    # Simulated data
    train_data = torch.randn(1000, 3, 32, 32)
    train_labels = torch.randint(0, 10, (1000,))
    dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()

        with autocast():
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            mem = torch.cuda.memory_allocated(device)
            print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}, GPU mem: {mem / 1e6:.1f}MB")

    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

    model = DeepModel(num_blocks=5, num_classes=10).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scaler = GradScaler()
    loader = create_dataloaders()

    num_epochs = 5
    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, loader, optimizer, scaler, device)
        accuracy = validate(model, loader, device)
        print(f"Epoch {epoch + 1}/{num_epochs} — Loss: {avg_loss:.4f}, Acc: {accuracy:.2%}")

    torch.cuda.synchronize()
    print("Training complete.")


if __name__ == "__main__":
    main()
