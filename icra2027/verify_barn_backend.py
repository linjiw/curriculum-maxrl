#!/usr/bin/env python3
"""Run the frozen-course BARN adapter twice and verify fixed-seed identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frontier_rl.adapters.barn_gazebo import (
    BarnGazeboSpace,
    LidarVelocityPolicy,
    load_courses,
)


def digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path("icra2027/barn_manifest.jsonl"))
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/home/robotixx/datasets/barn/BARN_dataset"))
    parser.add_argument("--course-id", default="barn-299")
    parser.add_argument("--seed", type=int, default=20270811)
    parser.add_argument("--domain-id", type=int, default=87)
    parser.add_argument("--master-port", type=int, default=11487)
    parser.add_argument("--runtime-root", type=Path,
                        default=Path("/tmp/icra_barn_backend_verify"))
    parser.add_argument("--output", type=Path,
                        default=Path("icra2027/results/barn_backend_determinism.json"))
    args = parser.parse_args()

    course = load_courses(
        args.manifest, args.dataset_root, [args.course_id])[0]
    runs = []
    for repeat in range(2):
        space = BarnGazeboSpace(
            [course], Path("icra2027/assets/barn_diff_drive.sdf"),
            args.runtime_root / f"repeat_{repeat}", seed=args.seed,
            n_strata=1, domain_id=args.domain_id,
            master_port=args.master_port,
            policy=LidarVelocityPolicy(seed=args.seed))
        successes, sim_steps, episodes = space.evaluate_course(
            course, 1, seed=args.seed)
        episode = episodes[0]
        runs.append({
            "successes": successes,
            "status": episode["status"],
            "sim_steps": sim_steps,
            "sim_seconds": episode["sim_seconds"],
            "planned_clearance_m": episode["planned_clearance_m"],
            "positions_sha256": digest(episode["positions"]),
            "commands_sha256": digest(episode["commands"]),
            "positions_count": len(episode["positions"]),
            "commands_count": len(episode["commands"]),
        })

    deterministic = runs[0] == runs[1]
    receipt = {
        "artifact_type": "barn_backend_determinism_receipt",
        "course_id": course.env_id,
        "course_asset_sha256": course.asset_sha256,
        "seed": args.seed,
        "cpu_only": True,
        "fixed_seed_exact_match": deterministic,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not deterministic:
        raise SystemExit("fixed-seed Gazebo receipts differ")


if __name__ == "__main__":
    main()
