import csv
import logging
from pathlib import Path
from typing import Any


def create_logger(
    name: str,
    log_path:str | Path,
) -> logging.Logger:
    """同时向终端和文件输出日志。"""
    path = Path(log_path)

    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)
    
    # 防止日志重复向更上层 logger 传播。
    logger.propagate = False

    # 防止多次调用函数后重复添加 handler。
    logger.handlers.clear()

    formatter = logging.Formatter(
    fmt=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 负责往控制台打印日志的对象
    console_handler = logging.StreamHandler()
    # 把这套格式交给控制台输出器使用
    console_handler.setFormatter(formatter)

    # 文件输出
    file_hanler = logging.FileHandler(filename=path, mode="a", encoding="utf-8")

    file_hanler.setFormatter(formatter)

    logger.addHandler(console_handler)

    logger.addHandler(file_hanler)

    return logger

class CSVHistoryLogger:
    """把每个 epoch 的指标写入 CSV。"""
    def __init__(self, path: str|Path, fieldnames: list[str]) -> None:
        self.path = Path(path)

        self.path.parent.mkdir(parents=True,exist_ok=True,)
        self.fieldnames = fieldnames

    def write(
            self,
            row: dict[str, Any],
    ) -> None:
        file_exists = self.path.exists()

        with self.path.open(
                "a",
                newline="",
                encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.fieldnames,
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)