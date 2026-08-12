import os
import random

import numpy as np
import torch


def seed_everything(
    seed: int,
    *,
    deterministic: bool = False,
    cudnn_benchmark: bool = True,
) -> None:
    """尽可能固定常见随机源。"""

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault(
            "CUBLAS_WORKSPACE_CONFIG",
            ":4096:8",
        )

        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = (
            cudnn_benchmark
        )