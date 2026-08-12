import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from utils.metrics import AverageMeter


def train_one_epoch(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
) -> dict[str, float]:

    model.train()

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()

    for images, targets in data_loader:
        #数据移动到与模型相同的设备。
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        # 2. 前向传播
        logits = model(images)

        loss = criterion(logits, targets)

        loss.backward()

        optimizer.step()

        batch_size = targets.size(0)

        predictions = logits.argmax(dim = 1)

        accuracy = (
            predictions
            .eq(targets)
            .float()
            .mean()
            .item()
            * 100
        )

        # 按真实样本数量累计。
        loss_meter.update(value = loss.item(), n = batch_size)

        accuracy_meter.update(value = accuracy, n=batch_size)

    return {
        "loss": loss_meter.average,
        "accuracy": accuracy_meter.average,
    }