from torchvision import transforms


def build_cifar10_transforms():
    # CIFAR-10 是 RGB 三通道，所以 mean/std 分别给 R、G、B 三个通道设置参数。
    # 这里都设为 0.5，配合 Normalize 会把像素大致从 [0, 1] 映射到 [-1, 1]。
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    # Compose：把多个图像预处理操作按照顺序组合起来。
    # 训练集加入随机数据增强，让模型看到更多不同形式的图片。
    train_transform = transforms.Compose(
        [
            # 先在图片四周填充 4 个像素，
            # 再随机裁剪出一张 32×32 图片，相当于制造轻微的位置变化。
            transforms.RandomCrop(
                size=32,
                padding=4,
            ),

            # 以 50% 概率将图片左右翻转。
            transforms.RandomHorizontalFlip(
                p=0.5,
            ),

            # 将图片转换为 Tensor：
            # [H, W, C] → [C, H, W]，像素值通常从 0~255 → 0~1。
            transforms.ToTensor(),

            # 对 RGB 三个通道分别做标准化。
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )

    # 验证/测试阶段不使用随机裁剪、随机翻转，
    # 保证每次评估面对的是相同的原始图片。
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )

    # 同时返回训练集和验证/测试集两套预处理规则。
    return train_transform, eval_transform