from pathlib import Path
from typing import Any

import yaml


def load_config(
    config_path: str | Path,
) -> dict[str, Any]:
    """读取 YAML 配置文件。"""

    config_path = Path(config_path)

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    return config