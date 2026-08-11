import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from data import create_data_loaders


def main() -> None:
    train_loader, val_loader, test_loader = (
        create_data_loaders(
            batch_size=128,
            val_ratio=0.1,
            seed=42,
        )
    )

    print("数据集数量：")
    print(
        f"train："
        f"{len(train_loader.dataset)}"
    )
    print(
        f"val："
        f"{len(val_loader.dataset)}"
    )
    print(
        f"test："
        f"{len(test_loader.dataset)}"
    )
    print("-" * 50)

    images, labels = next(
        iter(train_loader)
    )

    print("训练 Batch：")
    print(
        f"images.shape："
        f"{images.shape}"
    )
    print(
        f"labels.shape："
        f"{labels.shape}"
    )
    print(
        f"images.dtype："
        f"{images.dtype}"
    )
    print(
        f"labels.dtype："
        f"{labels.dtype}"
    )


if __name__ == "__main__":
    main()