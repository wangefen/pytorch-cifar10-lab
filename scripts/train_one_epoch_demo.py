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
from utils import seed_everything, load_config


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def main() -> None:
    config = load_config("configs/simple_cnn.yaml")
    seed_everything(
        seed=config["runtime"]["seed"],
        deterministic=config["runtime"]["deterministic"],
        cudnn_benchmark=config["runtime"]["cudnn_benchmark"],
    )

    device = select_device()

    train_loader,val_loader,_ = (
        create_data_loaders(
            batch_size = config["data"]["batch_size"],
            val_ratio = config["data"]["val_ratio"],
            seed = config["runtime"]["seed"],
        )
    )

    model = SimpleCNN(
        num_classes=config["model"]["num_classes"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["optimization"]["learning_rate"],
        momentum=config["optimization"]["momentum"],
    )

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

