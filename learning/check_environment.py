import platform
import sys

import torch
import torchvision


def main() -> None:
    # 1. 输出当前 Python 版本
    print(f"Python version: {platform.python_version()}")

    # 2. 输出实际执行当前代码的 Python 解释器路径
    print(f"Python executable: {sys.executable}")

    # 3. 输出 PyTorch 和 TorchVision 版本
    print(f"PyTorch version: {torch.__version__}")
    print(f"TorchVision version: {torchvision.__version__}")

    # 4. 检查当前 PyTorch 是否能够使用 CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    # 5. 根据当前环境选择运行设备
    if cuda_available:
        print(f"CUDA version used by PyTorch: {torch.version.cuda}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("MPS available: True")

        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 6. 直接在目标设备上创建两个二维张量
    x = torch.rand(4, 3, device=device)
    y = torch.rand(3, 2, device=device)

    # 7. 矩阵乘法
    z = x @ y

    # 8. 输出张量信息
    print("-" * 50)
    print(f"Selected device: {device}")
    print(f"x.shape: {x.shape}")
    print(f"y.shape: {y.shape}")
    print(f"z.shape: {z.shape}")
    print(f"x.device: {x.device}")
    print(f"y.device: {y.device}")
    print(f"z.device: {z.device}")

    print("-" * 50)
    print("Environment check passed.")


if __name__ == "__main__":
    main()