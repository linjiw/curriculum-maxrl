"""Build the immutable 300-course BARN JSONL manifest.

The scalar difficulty is the challenge's published optimal traversal time:
the Dijkstra path length (including fixed start/goal connectors) divided by
the standardized 2 m/s maximum speed.  No navigation outcomes enter it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


START_XY = np.array([-2.25, 3.0], dtype=float)
GOAL_XY = np.array([-2.25, 13.0], dtype=float)
CYLINDER_RADIUS = 0.075


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_coord_to_world(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    row_shift = -CYLINDER_RADIUS - 30 * CYLINDER_RADIUS * 2
    col_shift = CYLINDER_RADIUS + 5.0
    return np.column_stack((
        points[:, 0] * (CYLINDER_RADIUS * 2) + row_shift,
        points[:, 1] * (CYLINDER_RADIUS * 2) + col_shift,
    ))


def optimal_path_length(path: np.ndarray) -> float:
    world = path_coord_to_world(path)
    full = np.vstack((START_XY, world, GOAL_XY))
    return float(np.linalg.norm(np.diff(full, axis=0), axis=1).sum())


def build_records(dataset_root: Path) -> list[dict]:
    root = dataset_root.resolve()
    records = []
    for index in range(300):
        world = root / "world_files" / f"world_{index}.world"
        path_file = root / "path_files" / f"path_{index}.npy"
        metrics_file = root / "metrics_files" / f"metrics_{index}.npy"
        norm_metrics_file = (
            root / "norm_metrics_files" / f"norm_metrics_{index}.npy")
        grid_file = root / "grid_files" / f"grid_{index}.npy"
        missing = [str(path) for path in (
            world, path_file, metrics_file, norm_metrics_file, grid_file)
                   if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"course {index} missing: {missing}")

        path = np.load(path_file, allow_pickle=False)
        metrics = np.load(metrics_file, allow_pickle=False)
        norm_metrics = np.load(norm_metrics_file, allow_pickle=False)
        path_length = optimal_path_length(path)
        records.append({
            "env_id": f"barn-{index:03d}",
            "barn_index": index,
            "difficulty": path_length / 2.0,
            "difficulty_definition": "published_optimal_traversal_time_seconds",
            "optimal_path_length_m": path_length,
            "asset": f"world_files/world_{index}.world",
            "asset_sha256": sha256(world),
            "path_asset": f"path_files/path_{index}.npy",
            "path_sha256": sha256(path_file),
            "grid_asset": f"grid_files/grid_{index}.npy",
            "grid_sha256": sha256(grid_file),
            "published_metrics": [float(x) for x in metrics],
            "published_normalized_metrics": [float(x) for x in norm_metrics],
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = build_records(args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in records))
    print(f"wrote {args.output}: {len(records)} courses; sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
