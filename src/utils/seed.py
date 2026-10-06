
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed every RNG used by the project."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # cuDNN autotuning picks different algorithms run to run; pin it.
    # On CUDA, use_deterministic_algorithms below also needs
    # CUBLAS_WORKSPACE_CONFIG=:4096:8 set before the CUDA context is
    # created, which is too late to do here -- see scripts/run_v2.py.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
