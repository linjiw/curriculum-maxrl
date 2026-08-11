"""Freeze a difficulty-stratified train/held-out environment-pool split.

Input is JSONL with one object per course:
``{"env_id": "barn-001", "difficulty": 0.42, "asset": "path/or/uri"}``.
The output records the exact IDs, metadata, input SHA-256, and split settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REQUIRED_FIELDS = ("env_id", "difficulty", "asset")


def load_pool(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"line {line_number} missing fields {missing}")
        row = dict(row)
        row["env_id"] = str(row["env_id"])
        row["difficulty"] = float(row["difficulty"])
        row["asset"] = str(row["asset"])
        if not np.isfinite(row["difficulty"]):
            raise ValueError(f"line {line_number} has non-finite difficulty")
        records.append(row)
    ids = [row["env_id"] for row in records]
    if not records:
        raise ValueError("pool is empty")
    if len(ids) != len(set(ids)):
        raise ValueError("env_id values must be unique")
    return records


def stratified_split(records: list[dict], *, holdout_fraction: float,
                     n_strata: int, seed: int) -> tuple[list[str], list[str]]:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must lie strictly between 0 and 1")
    if n_strata < 1:
        raise ValueError("n_strata must be positive")
    ordered = sorted(records, key=lambda row: (row["difficulty"], row["env_id"]))
    strata = np.array_split(np.arange(len(ordered)), min(n_strata, len(ordered)))
    rng = np.random.default_rng(seed)
    heldout = set()
    for stratum in strata:
        indices = np.array(stratum, dtype=int)
        rng.shuffle(indices)
        count = int(round(len(indices) * holdout_fraction))
        if len(indices) > 1:
            count = min(max(count, 1), len(indices) - 1)
        else:
            count = int(holdout_fraction >= 0.5)
        heldout.update(ordered[i]["env_id"] for i in indices[:count])
    train_ids = [row["env_id"] for row in ordered if row["env_id"] not in heldout]
    heldout_ids = [row["env_id"] for row in ordered if row["env_id"] in heldout]
    if not train_ids or not heldout_ids:
        raise ValueError("split produced an empty train or held-out set")
    return train_ids, heldout_ids


def make_manifest(path: Path, *, holdout_fraction: float = 0.2,
                  n_strata: int = 10, seed: int = 20270811) -> dict:
    raw = path.read_bytes()
    records = load_pool(path)
    train_ids, heldout_ids = stratified_split(
        records, holdout_fraction=holdout_fraction,
        n_strata=n_strata, seed=seed)
    by_id = {row["env_id"]: row for row in records}
    return {
        "schema_version": 1,
        "source_path": str(path),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "seed": seed,
        "holdout_fraction_requested": holdout_fraction,
        "n_strata": n_strata,
        "n_total": len(records),
        "n_train": len(train_ids),
        "n_heldout": len(heldout_ids),
        "train_ids": train_ids,
        "heldout_ids": heldout_ids,
        "records": by_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pool", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--n-strata", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20270811)
    args = parser.parse_args()
    manifest = make_manifest(
        args.pool, holdout_fraction=args.holdout_fraction,
        n_strata=args.n_strata, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.output}: {manifest['n_train']} train / "
          f"{manifest['n_heldout']} held out")


if __name__ == "__main__":
    main()

