from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """
    Make runs reproducible.

    Notes:
    - This does NOT "make up data"; it only fixes RNG sources for model training.
    - Some GPU ops can still be non-deterministic depending on hardware/driver.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    except Exception:
        # Torch is optional for the non-DL modules
        pass


