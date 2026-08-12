import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from data import create_data_loaders
from engine import train_one_epoch, evaluate
from models import SimpleCNN
from scripts.debug_single_batch import select_device



def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("CPU")


def main() -> None:
    torch.manual_seed(42)

    device = select_device()

    train_loader,val_loader,_ = (
        create_data_loaders(128, 0.1, 42)
    )

    model = SimpleCNN(10).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    print("开始训练一个 epoch...")
    print("-" * 50)

    train_metrics = train_one_epoch(
        model = model,
        data_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    val_metrics = evaluate(
        model = model,
        data_loader = val_loader,
        criterion=criterion,
        device=device,
    )
    print("一个 epoch 训练完成：")
    print("-" * 50)

    print(
        f"train_loss："
        f"{train_metrics['loss']:.4f}"
    )

    print(
        f"train_accuracy："
        f"{train_metrics['accuracy']:.2f}%"
    )

    print(
        f"val_loss："
        f"{val_metrics['loss']:.4f}"
    )

    print(
        f"val_accuracy："
        f"{val_metrics['accuracy']:.2f}%"
    )

if __name__ == "__main__":
    main()

