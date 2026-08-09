"""One-time construction of the frozen sklearn Digits split manifest."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedShuffleSplit

try:
    from .core import (
        DATA_MANIFEST_PATH,
        DEV_SIZE,
        TEST_SIZE,
        TRAIN_SIZE,
        assert_pinned_runtime,
        sha256_array,
        write_json,
    )
except ImportError:  # Direct-script execution from the repository root.
    from curriculum_maxrl.digits_factorial.core import (
        DATA_MANIFEST_PATH,
        DEV_SIZE,
        TEST_SIZE,
        TRAIN_SIZE,
        assert_pinned_runtime,
        sha256_array,
        write_json,
    )


TEST_SPLIT_SEED = 20260808
DEV_SPLIT_SEED = 20260809


def class_counts(targets: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    counts = Counter(int(value) for value in targets[indices])
    return {str(label): counts[label] for label in range(10)}


def build_manifest() -> dict[str, object]:
    assert_pinned_runtime()
    bunch = load_digits()
    raw_x = np.asarray(bunch.data)
    y = np.asarray(bunch.target)
    original = np.arange(len(y), dtype=np.int64)

    first = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=TEST_SPLIT_SEED
    )
    remaining_positions, test_positions = next(first.split(raw_x, y))
    remaining = original[np.asarray(remaining_positions, dtype=np.int64)]
    test = original[np.asarray(test_positions, dtype=np.int64)]

    second = StratifiedShuffleSplit(
        n_splits=1, test_size=DEV_SIZE, random_state=DEV_SPLIT_SEED
    )
    train_positions, dev_positions = next(
        second.split(raw_x[remaining], y[remaining])
    )
    train = remaining[np.asarray(train_positions, dtype=np.int64)]
    dev = remaining[np.asarray(dev_positions, dtype=np.int64)]

    expected_shapes = {"train": TRAIN_SIZE, "dev": DEV_SIZE, "test": TEST_SIZE}
    splits = {"train": train, "dev": dev, "test": test}
    for name, expected_size in expected_shapes.items():
        if splits[name].shape != (expected_size,):
            raise AssertionError(f"unexpected {name} split shape: {splits[name].shape}")
    joined = np.concatenate([train, dev, test])
    if not np.array_equal(np.sort(joined), original):
        raise AssertionError("split indices are not a disjoint exhaustive partition")

    return {
        "schema": "curriculum-maxrl/digits-split/v1",
        "dataset": {
            "loader": "sklearn.datasets.load_digits",
            "n_examples": len(y),
            "raw_data_shape": list(raw_x.shape),
            "target_shape": list(y.shape),
            "normalization": "astype(float64) / 16.0",
            "sklearn_version": sklearn.__version__,
        },
        "construction": {
            "stage_1": {
                "algorithm": "StratifiedShuffleSplit(n_splits=1,test_size=360)",
                "random_state": TEST_SPLIT_SEED,
                "output": "sealed test plus remaining 1437",
            },
            "stage_2": {
                "algorithm": "StratifiedShuffleSplit(n_splits=1,test_size=360) on remaining",
                "random_state": DEV_SPLIT_SEED,
                "output": "development 360 plus training 1077",
            },
            "index_semantics": "original load_digits row indices; stored order is authoritative",
            "implicit_regeneration_forbidden": True,
        },
        "array_sha256": {
            "raw_data": sha256_array(raw_x),
            "target": sha256_array(y),
            "normalized_data": sha256_array(raw_x.astype(np.float64) / 16.0),
        },
        "index_sha256": {name: sha256_array(indices) for name, indices in splits.items()},
        "class_counts": {
            "all": class_counts(y, original),
            **{name: class_counts(y, indices) for name, indices in splits.items()},
        },
        "indices": {name: indices.tolist() for name, indices in splits.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DATA_MANIFEST_PATH)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite stored split manifest: {args.output}")
    manifest = build_manifest()
    write_json(args.output, manifest)
    print(args.output)


if __name__ == "__main__":
    main()
