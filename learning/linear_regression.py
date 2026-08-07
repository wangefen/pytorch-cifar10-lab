import torch
from torch import nn


class LinearRegressionModel(nn.Module):
    """最简单的一元线性回归模型。"""

    def __init__(self) -> None:
        super().__init__()

        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """定义模型的前向传播过程。"""
        return self.linear(x)



def main() -> None:
    # 固定随机种子，使当前环境下重复运行的结果更稳定。
    torch.manual_seed(42)

    x = torch.linspace(-2, 2, 200).unsqueeze(1)

    noise = torch.rand_like(x) * 0.05

    y = 2 * x + 1 + noise

    print("数据形状：")
    print(f"x.shape: {x.shape}")
    print(f"y.shape: {y.shape}")
    print("-" * 50)

    model = LinearRegressionModel()

    print("训练前参数：")
    print(f"weight: {model.linear.weight.item():.4f}")
    print(f"bias: {model.linear.bias.item():.4f}")
    print("-" * 50)

    criterion = nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), 0.05)

    for epoch in range(201):
        optimizer.zero_grad(set_to_none=True)

        prediction = model(x)

        loss = criterion(prediction, y)

        loss.backward()

        optimizer.step()

        if epoch % 20 == 0:
            weight = model.linear.weight.item()
            bias = model.linear.bias.item()

            print(
                f"epoch={epoch:03d}, "
                f"loss={loss.item():.6f}, "
                f"w={weight:.4f}, "
                f"b={bias:.4f}"
            )

    print("-" * 50)
    print("训练完成。")
    print(
        f"最终 weight："
        f"{model.linear.weight.item():.10f}"
    )
    print(
        f"最终 bias："
        f"{model.linear.bias.item():.10f}"
    )


if __name__ == "__main__":
    main()