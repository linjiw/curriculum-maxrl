"""Source-locked Digits estimator-by-sampler factorial.

Thread limits are set before importing NumPy/PyTorch-backed implementation
modules so serial and process-parallel schedules use the same numerical path.
"""

import os

for _variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "1"

from .core import (
    ACTION_BUDGET,
    CELLS,
    CONFIRMATION_SEEDS,
    DEVELOPMENT_LRS,
    DEVELOPMENT_SEEDS,
    GROUP_SIZE,
    GROUPS_PER_STEP,
    N_STEPS,
)

__all__ = [
    "ACTION_BUDGET",
    "CELLS",
    "CONFIRMATION_SEEDS",
    "DEVELOPMENT_LRS",
    "DEVELOPMENT_SEEDS",
    "GROUP_SIZE",
    "GROUPS_PER_STEP",
    "N_STEPS",
]
