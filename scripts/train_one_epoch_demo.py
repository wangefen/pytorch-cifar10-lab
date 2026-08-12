import sys
from pathlib import Path

import torch
from torch import nn

from data import create_data_loaders
from engine import train_one_epoch
from models import SimpleCNN
from scripts.debug_single_batch import select_device

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


def selec_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("CPU")


def main() -> None:
    torch.manual_seed(42)

    device = select_device()

    train_loader,_,_ = (
        create_data_loaders(128, 0.1, 42)
    )

    model = SimpleCNN(10).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    print("开始训练一个 epoch...")
    print("-" * 50)

    metrics = train_one_epoch(
        model = model,
        data_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )
    print("一个 epoch 训练完成：")
    print(
        f"train_loss："
        f"{metrics['loss']:.4f}"
    )
    print(
        f"train_accuracy："
        f"{metrics['accuracy']:.2f}%"
    )

if __name__ == "__main__":
    main()

