import torch
from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
        nn.Conv2d(3, 32, 3, 1, 1,bias = False),

        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, 3, 1, 1,bias = False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, 3, 1, 1,bias = False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),

         # 每个通道最后总结出一个“这个特征有多明显”的代表值。
         # 把最后的每个 n×n 特征图平均压缩成一个数字
        nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

        self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128, out_features= num_classes),
        )

    def forward(self, x: torch.Tensor,) -> torch.Tensor:
        features = self.features(x)
        logits = self.classifier(features)

        return logits

def main() -> None:
    model = SimpleCNN(10)

    images = torch.randn(8, 3, 32, 32)

    logits = model(images)

    print("模型：")
    print(model)
    print("-" * 50)

    print("Shape 检查：")
    print(f"输入：{images.shape}")
    print(f"输出：{logits.shape}")

    # 统计模型中所有“需要训练”的参数总数量。
    parameter_count = sum(
        # numel() 返回当前参数 Tensor 中一共有多少个数值。
        parameter.numel()
        # 遍历模型中注册的所有参数，例如 Conv、BatchNorm、Linear 的权重和偏置。
        for parameter in model.parameters()
        # 只统计需要计算梯度、会被优化器更新的参数。
        if parameter.requires_grad
    )

    print("-" * 50)
    print(
        f"可训练参数数量："
        f"{parameter_count:,}"
    )

if __name__ == "__main__":
    main()