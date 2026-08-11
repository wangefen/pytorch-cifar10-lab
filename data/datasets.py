from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10

from .transforms import build_cifar10_transforms

# 数据集固定在 learning/datasets 下，与进程启动目录无关。
DATA_ROOT = Path(__file__).resolve().parents[1] / "datasets"


def create_data_loaders(
        batch_size: int = 128,
        val_ratio:float = 0.1,
        seed: int = 42,
):
    train_transform, eval_transform =(build_cifar10_transforms())

    full_train_augmented = CIFAR10(
        root=str(DATA_ROOT),
        train=True,
        transform=train_transform,
        download=True,
    )

    full_train_evaluated = CIFAR10(
        root=str(DATA_ROOT),
        train=True,
        transform=eval_transform,
        download=True,
    )

    test_dataset = CIFAR10(
        root=str(DATA_ROOT),
        train=False,
        transform=eval_transform,
        download=True,
    )

    total_size = len(full_train_augmented)

    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(total_size, generator=generator).tolist()

    val_size = int(total_size * val_ratio)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = Subset(full_train_augmented, train_indices)

    val_dataset = Subset(
        full_train_evaluated,
        val_indices,
    )

    #DataLoader 按 batch_size 将 Dataset 组织成一个个 batch，供训练时逐批读取。
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader