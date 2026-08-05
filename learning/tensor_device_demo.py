import torch
from sympy import vectorize
from torch import dtype


def select_device() -> torch.device:
    """根据当前环境选择可用的计算设备。"""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def print_tensor_info(name: str, tensor: torch.Tensor) -> None:
    """统一输出一个张量的基本信息。"""
    print(f"{name} 的内容：")
    print(tensor)
    print(f"{name} 的维度数量：{tensor.ndim}")
    print(f"{name} 的形状：{tensor.shape}")
    print(f"{name} 的数据类型：{tensor.dtype}")
    print(f"{name} 所在设备：{tensor.device}")
    print("-" * 50)


def main() -> None:
    device = select_device()

    number = torch.tensor(7)

    vector = torch.tensor([10, 20, 30])

    matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])

    images = torch.rand(8,3,32,32,dtype=torch.float32)

    labels = torch.randint(low=0, high=10, size=(8,),dtype=torch.int64,)

    # permute() 并没有删除数据，也没有增加数据，只是调整了维度的排列顺序。
    print_tensor_info("nuimber", number)
    print_tensor_info("vector", vector)
    print_tensor_info("matrix",matrix)

    print("原始图像批次: ")
    print(f"image.ndim: {images.ndim}")
    print(f"image.shape: {images.shape}")
    print(f"image.dtype: {images.dtype}")
    print(f"image.device: {images.device}")
    print("-" * 50)

    print("分类标签：")
    print(f"labels：{labels}")
    print(f"labels.ndim：{labels.ndim}")
    print(f"labels.shape：{labels.shape}")
    print(f"labels.dtype：{labels.dtype}")
    print(f"labels.device：{labels.device}")
    print("-" * 50)


    #把 images 的维度顺序从 [批次, 通道, 高度, 宽度] 改成 [批次, 高度, 宽度, 通道]。
    # 新第0维 ← 原第0维
    # 新第1维 ← 原第2维
    # 新第2维 ← 原第3维
    # 新第3维 ← 原第1维
    images_nhwc = images.permute(0, 2, 3, 1)

    print("通道交换顺序: ")
    print(f"image_nchw.shape: {images_nhwc.shape}")
    print(f"image_nhwc.shape: {images.shape}")
    print("-" * 50)


    #reshape(8, 3, 32, 32) 就是把展开的数据重新整理成“8 张、3 通道、32×32”的图片批次。-1表示自动计算剩余维度元素总和
    images_flat = images.reshape(images.shape[0], -1)
    print("图像展开：")
    print(f"展开前：{images.shape}")
    print(f"展开后：{images_flat.shape}")
    print("-" * 50)

    # 将展开后的图片恢复为原来的四维结构。
    images_restored = images_flat.reshape(8, 3, 32, 32)
    print("恢复图像形状：")
    print(f"恢复后：{images_restored.shape}")
    print(
        "恢复前后数值完全相同："
        f"{torch.equal(images, images_restored)}"
    )
    print("-" * 50)

    # 把图片和标签移动到同一个计算设备。
    images = images.to(device)
    labels = labels.to(device)

    print("移动设备后：")
    print(f"选择的设备：{device}")
    print(f"images.device：{images.device}")
    print(f"labels.device：{labels.device}")
    print("-" * 50)

    print("Tensor 与设备练习运行完成。")

if __name__ == "__main__":
    main()