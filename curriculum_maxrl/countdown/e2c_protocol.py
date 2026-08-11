"""Frozen, outcome-blind constants for the preregistered E2c experiment."""

from __future__ import annotations

from collections.abc import Iterable


FROZEN_SEEDS = (1, 2, 3)
FROZEN_STEPS = 60
FROZEN_GPU_MEMORY_CEILING_MIB = 4096
FROZEN_MISMATCH_LIMIT = 0.05
FROZEN_DISPLACED_SLOT_LIMIT = 0.25
FROZEN_OPTIMIZER_ROWS = 128
FROZEN_GROUP_SIZE = 16
FROZEN_MINIMUM_GROUPS = 128
FROZEN_MINIMUM_TOKEN_COUNTS = 16
FROZEN_COLLECTOR_SEED = 424242
FROZEN_MAXIMUM_GROUPS = 256
FROZEN_MAXRL_COMMIT = "7197bbb46a2ecd866da52f6b401ff20a34fe9390"
FROZEN_BASE_MODEL_REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"

# These were recorded from the preregistered local assets before reservoir
# collection, E2c training, or E2c held-out generation.
FROZEN_ASSET_SHA256 = {
    "train.parquet":
        "ac0671a2215a806d6c75b359f21697a9cfc8d8f47eef3b236391bd6e9fada91a",
    "test.parquet":
        "95b1456fc3f49bc6f463614fef92900d748a07e4b429fd9383bbcf5edcb4e489",
    "config.json":
        "283834b57c6e55af57e59b007df3bfcaf2f898dbb22fb535a46d224b73acb0cd",
    "model.safetensors":
        "3198bb0f0c8598ec9aa713540e19472ebbe8363702db0d555fb060c679128ff8",
    "tokenizer.json":
        "7d27c493c729a66ecefc837280b05d948b1ed50d130eebdbf911b1b36cf38ed7",
    "training_metrics.json":
        "36085b432f5d3bed12e192648e996de6a10c41a60f9955582cce56b5bd8589f4",
    "countdown_reward.py":
        "99c04d4a4914170a528c67337aec364e7410074c552d9848c714f78c0f9e2312",
    "eval_countdown.py":
        "0f642db64cabff66631b7e9ac88f1f3519651b21bee351051a1190f1a5bf653d",
}


def require_frozen_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    """Reject incomplete, reordered, duplicated, or substituted seed sets."""
    normalized = tuple(int(seed) for seed in seeds)
    if normalized != FROZEN_SEEDS:
        raise ValueError(
            f"E2c requires ordered seeds {FROZEN_SEEDS}, got {normalized}")
    return normalized


def require_frozen_scalar(name: str, actual, expected) -> None:
    """Reject a command-line override of a preregistered scalar."""
    if actual != expected:
        raise ValueError(f"E2c {name} is frozen at {expected}, got {actual}")
