"""Apply the effect-blind Acrobot V2 development launch gates.

This analysis intentionally never reads the runner's treatment contrasts.
It consumes only per-run invariants, within-cell learning/headroom, group
regimes, teacher movement, runtime, and the pre-development source lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = Path(__file__).with_name(
    "acrobot_neural_v2_capacity_development.json"
)
DEFAULT_LOCK = Path(__file__).with_name("ACROBOT_NEURAL_V2_LOCK.json")
DEFAULT_OUTPUT = Path(__file__).with_name(
    "acrobot_neural_v2_development_gates.json"
)
EXPECTED_SEEDS = [11_000, 11_001, 11_002]
WARMUP_TRANSITIONS = 200_000
CHECKPOINT_TRANSITIONS = 1_000_000


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_value_at_or_after(run: dict, coordinate: int) -> tuple[int, float]:
    for x, value in zip(run["x_transitions"], run["mean_pass_curve"]):
        if int(x) >= coordinate:
            return int(x), float(value)
    raise ValueError(
        f"seed {run.get('seed')} has no evaluation at or after {coordinate}"
    )


def expected_capacity(config: dict) -> tuple[int, int]:
    return {
        "shared": (640, 640),
        "disjoint_total_budget": (640, 80),
        "disjoint_active_capacity": (5_120, 640),
    }[config["architecture"]]


def analyze(artifact: dict, source_lock: dict, *, artifact_sha256: str) -> dict:
    cases = artifact["cases"]
    lock_hashes = source_lock["source_sha256"]
    artifact_hashes = artifact["provenance"]["source_sha256"]
    hash_checks = {
        path: {
            "expected": expected,
            "recorded": artifact_hashes.get(path),
            "matches": artifact_hashes.get(path) == expected,
        }
        for path, expected in lock_hashes.items()
    }
    runtime_checks = {
        key: {
            "expected": value,
            "recorded": artifact["provenance"].get(key),
            "matches": str(artifact["provenance"].get(key)) == str(value),
        }
        for key, value in source_lock["runtime"].items()
    }

    per_cell = {}
    all_invariants = (
        artifact.get("artifact_state") == "complete"
        and not artifact.get("run_failures")
        and artifact["protocol"].get("status") == "exploratory"
        and artifact["protocol"].get("explicit_exploratory") is True
        and artifact["protocol"].get("paired_seeds") == EXPECTED_SEEDS
        and all(check["matches"] for check in hash_checks.values())
        and all(check["matches"] for check in runtime_checks.values())
    )
    all_learning_headroom = True
    all_mixed_exposure = True
    all_teacher_moved = True
    all_regimes_complete = True

    for name, record in cases.items():
        runs = record["runs"]
        config = record["config"]
        expected_total, expected_active = expected_capacity(config)
        seeds_match = [run.get("seed") for run in runs] == EXPECTED_SEEDS
        invariant_flags = []
        progress = []
        post_warmup_groups = []
        all_groups = []
        for run in runs:
            invariant_flags.append(
                bool(
                    run.get("numeric_valid")
                    and run.get("accounting_valid")
                    and run.get("verifier_relabel_checks_valid")
                    and run.get("evaluation_cadence_invariant")
                    and run.get("total_parameters") == expected_total
                    and run.get("active_parameters_per_task") == expected_active
                    and run.get("transitions", 0) >= 2_000_000
                )
            )
            x, value = first_value_at_or_after(run, CHECKPOINT_TRANSITIONS)
            improvement = value - float(run["initial_mean_pass"])
            qualifies = improvement >= 0.03 and value < 0.95
            progress.append(
                {
                    "seed": run["seed"],
                    "checkpoint_transitions": x,
                    "mean_pass": value,
                    "initial_mean_pass": float(run["initial_mean_pass"]),
                    "improvement": improvement,
                    "below_0p95": value < 0.95,
                    "qualifies": qualifies,
                }
            )
            all_groups.extend(run["group_diagnostics"])
            post_warmup_groups.extend(
                group
                for group in run["group_diagnostics"]
                if group["transition_start"] >= WARMUP_TRANSITIONS
            )

        qualifying_seeds = sum(item["qualifies"] for item in progress)
        learning_headroom_pass = qualifying_seeds >= 2
        mixed_count = sum(
            group["regime"] == "mixed" for group in post_warmup_groups
        )
        mixed_fraction = (
            mixed_count / len(post_warmup_groups) if post_warmup_groups else 0.0
        )
        mixed_exposure_pass = mixed_count > 0 and mixed_fraction >= 0.10
        regimes = sorted({group["regime"] for group in all_groups})
        all_regimes_pass = regimes == ["all_pass", "dead", "mixed"]
        teacher_tvs = [
            float(group["teacher_tv_from_uniform"])
            for group in post_warmup_groups
        ]
        mean_teacher_tv = (
            sum(teacher_tvs) / len(teacher_tvs) if teacher_tvs else 0.0
        )
        teacher_movement_pass = (
            True if config["sampling"] == "uniform" else mean_teacher_tv > 0.05
        )
        cell_invariants = seeds_match and all(invariant_flags)
        all_invariants = all_invariants and cell_invariants
        all_learning_headroom = all_learning_headroom and learning_headroom_pass
        all_mixed_exposure = all_mixed_exposure and mixed_exposure_pass
        all_teacher_moved = all_teacher_moved and teacher_movement_pass
        all_regimes_complete = all_regimes_complete and all_regimes_pass
        per_cell[name] = {
            "config": config,
            "seeds_match": seeds_match,
            "invariants_pass": cell_invariants,
            "progress_at_first_checkpoint_at_or_after_1m": progress,
            "qualifying_seed_count": qualifying_seeds,
            "learning_and_headroom_pass": learning_headroom_pass,
            "post_warmup_group_count": len(post_warmup_groups),
            "post_warmup_mixed_count": mixed_count,
            "post_warmup_mixed_fraction": mixed_fraction,
            "mixed_exposure_pass": mixed_exposure_pass,
            "complete_run_regimes": regimes,
            "all_three_regimes_pass": all_regimes_pass,
            "post_warmup_mean_teacher_tv_from_uniform": mean_teacher_tv,
            "teacher_movement_pass": teacher_movement_pass,
        }

    observed_dev_wall_seconds = sum(
        float(run["wall_seconds"])
        for record in cases.values()
        for run in record["runs"]
    )
    projected_confirmatory_hours = observed_dev_wall_seconds * 20.0 / 3.0 / 3600.0
    runtime_pass = projected_confirmatory_hours <= 24.0
    gates = {
        "source_runtime_and_run_invariants": all_invariants,
        "every_cell_two_of_three_learn_with_headroom": all_learning_headroom,
        "every_cell_post_warmup_mixed_fraction_at_least_0p10": all_mixed_exposure,
        "every_teacher_cell_mean_tv_above_0p05": all_teacher_moved,
        "every_cell_observes_all_three_k_regimes": all_regimes_complete,
        "projected_20_seed_serial_runtime_at_most_24h": runtime_pass,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "analysis_contract": (
            "effect-blind V2 launch gates only; treatment contrasts are not read"
        ),
        "development_artifact_sha256": artifact_sha256,
        "source_lock": source_lock,
        "source_hash_checks": hash_checks,
        "runtime_version_checks": runtime_checks,
        "per_cell": per_cell,
        "runtime": {
            "observed_development_serial_wall_seconds": observed_dev_wall_seconds,
            "projected_20_seed_six_cell_serial_hours": projected_confirmatory_hours,
            "limit_hours": 24.0,
        },
        "gates": gates,
        "confirmatory_core_authorized": gates["all_pass"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    artifact = json.loads(args.artifact.read_text())
    source_lock = json.loads(args.source_lock.read_text())
    result = analyze(
        artifact,
        source_lock,
        artifact_sha256=sha256_file(args.artifact),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["gates"], indent=2))
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
