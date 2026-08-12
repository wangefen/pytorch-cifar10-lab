import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data import create_data_loaders
from models import SimpleCNN

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')

    return torch.device("cpu")

def main() -> None:
    torch.manual_seed(42)

    device = select_device()

    print(f"使用设备:{device}")
    print("-" * 50)

    # 创建现有的数据流水线。
    train_loader,_,_ = create_data_loaders(128, 0.1, 42)

    #IFAR10 ，这个数据集默认返回image和label对吗
    images, targets = next(iter(train_loader))

    images = images.to(device)
    targets = targets.to(device)

    print("固定 Batch：")
    print(f"image.shape:{images.shape}")
    print(f"targets.shape：{targets.shape}")
    print(f"images.device：{images.device}")
    print(f"targets.device：{targets.device}")
    print("-" * 50)

    # 创建模型并移动到同一设备。
    model = SimpleCNN(10).to(device)

    # 分类任务使用交叉熵损失。
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    model.train()

    for step in range(301):
        # 1. 清理上一轮梯度。
        optimizer.zero_grad(set_to_none=True)

        # 2. 前向传播。
        logits = model(images)

        # 3. 计算分类损失。
        loss = criterion(logits, targets)

        # 4. 反向传播。
        loss.backward()

        # 5. 根据梯度更新模型参数。
        optimizer.step()

        if step % 20 == 0:
            predictions = logits.argmax(dim = 1)

            #预测结果和真实标签逐个比较 → 对的记 1，错的记 0 → 求平均 → 乘 100，得到准确率百分比。
            accuracy = (predictions.eq(targets).float().mean().item()*100)

            print(
                f"step={step:03d}, "
                f"loss={loss.item():.4f}, "
                f"accuracy={accuracy:.2f}%"
            )

    print("-" * 50)
    print("单 Batch 过拟合实验完成。")


if __name__ == "__main__":
    main()