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
from utils import (
    CSVHistoryLogger,
    create_logger,
    load_config,
    seed_everything,
)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def main() -> None:
    config = load_config(PROJECT_ROOT /"configs/simple_cnn.yaml")

    output_dir = (
            PROJECT_ROOT
            / config["project"]["output_dir"]
            / config["project"]["run_name"]
    )

    log_path = (
            output_dir
            / "logs"
            / "train.log"
    )

    history_path = (
            output_dir
            / "results"
            / "history.csv"
    )

    logger = create_logger(
        name="train",
        log_path=log_path,
    )

    history_logger = CSVHistoryLogger(
        path=history_path,
        fieldnames=[
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
        ],
    )


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

    epochs = config["optimization"]["epochs"]

    for epoch in range(1, epochs + 1):
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

        logger.info(
            "Epoch %d/%d | "
            "train_loss=%.4f | "
            "train_acc=%.2f%% | "
            "val_loss=%.4f | "
            "val_acc=%.2f%%",
            epoch,
            epochs,
            train_metrics["loss"],
            train_metrics["accuracy"],
            val_metrics["loss"],
            val_metrics["accuracy"],
        )

        history_logger.write(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )


if __name__ == "__main__":
    main()

