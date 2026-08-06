import torch


def main() -> None:
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    print("初始张量：")
    print(f"x: {x}")
    print(f"x.raquiers_grad: {x.requires_grad}")
    print(f"反向传播前 x.grad: {x.grad}")
    print("-" * 50)

    y = (x**2).sum()

    print("前向计算结果：")
    print(f"y: {y}")
    print(f"y的普通数值:{y.item()}")
    print(f"y.grad_fn：{y.grad_fn}")   #.grad_fn → 记录张量由什么运算产生
    print("-" * 50)

    # 从 y 开始反向传播，计算 y 对 x 的梯度。
    y.backward()

    print("第一次反向传播：")
    print(f"PyTorch计算的梯度：{x.grad}")
    print(f"理论梯度 2x: {2 * x.detach()}")
    print("-" * 50)

    # 重新建立一个新的计算图：
    # z = 3x1 + 3x2 + 3x3
    z = (3 * x).sum()

    # 注意：这里故意没有清理 x.grad。
    z.backward()

    print("没有清理梯度，再次反向传播：")
    print(f"此时 x.grad：{x.grad}")
    print("第一次梯度：[2, 4, 6]")
    print("第二次梯度：[3, 3, 3]")
    print("累积结果：[5, 7, 9]")
    print("-" * 50)

    # 将已经累积的梯度清零。
    x.grad.zero_()

    print("梯度清零后：")
    print(f"x.grad：{x.grad}")
    print("-" * 50)

    # 再建立一个新的计算图。
    w = (4 * x).sum()
    w.backward()

    print("清零后的第三次反向传播：")
    print(f"x.grad：{x.grad}")
    print("理论梯度：[4, 4, 4]")
    print("-" * 50)

    print("Autograd 练习运行完成。")


if __name__ == "__main__":
    main()