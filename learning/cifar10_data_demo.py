import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


def main() -> None:
    # 将原始图片转换成 PyTorch Tensor，
    # CIFAR-10 单张图片转换后 shape 为 [3, 32, 32]，
    # 像素值通常从 0~255 缩放到 0~1。
    transform = transforms.ToTensor()

    # 创建 CIFAR-10 训练集对象。
    train_dataset = CIFAR10(
        root="datasets",      # 数据集保存位置
        train=True,          # True=训练集，False=测试集
        transform=transform, # 取图片时自动执行 ToTensor()
        download=True,       # 本地没有数据时自动下载
    )

    print("Dataset 信息：")
    print(f"训练集样本数量：{len(train_dataset)}")
    print("-" * 50)

    # Dataset 每次返回一个样本：(image, label)
    image, label = train_dataset[0]

    print("单个样本：")
    print(f"image.shape：{image.shape}")   # [3, 32, 32]
    print(f"image.dtype：{image.dtype}")   # torch.float32
    print(f"label：{label}")
    print(f"label 类型：{type(label)}")    # 单独取样本时通常是 Python int
    print("-" * 50)

    # DataLoader 负责把 Dataset 中的多个单样本打包成 batch。
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=128,   # 每个 batch 包含 128 个样本
        shuffle=True,     # 每轮训练前打乱样本顺序
        num_workers=0,    # 由主进程加载数据，学习阶段最稳妥
    )

    # iter() 创建 DataLoader 迭代器，
    # next() 取出其中第一个 batch。
    images, labels = next(iter(train_loader))

    print("一个 Batch：")
    # [128, 3, 32, 32]
    # 128张图片，每张图片3通道、32高、32宽。
    print(f"images.shape：{images.shape}")
    print(f"images.dtype：{images.dtype}")

    # 128张图片分别对应128个类别标签，因此 shape=[128]。
    print(f"labels.shape：{labels.shape}")
    print(f"labels.dtype：{labels.dtype}")  # 通常是 torch.int64
    print("-" * 50)

    print("前 10 个标签：")
    print(labels[:10])

    print("-" * 50)
    print("CIFAR-10 数据加载实验完成。")


if __name__ == "__main__":
    main()