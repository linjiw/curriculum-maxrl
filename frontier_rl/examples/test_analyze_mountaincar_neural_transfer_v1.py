"""Adversarial synthetic tests for the independent MountainCar V1r2 verifier."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from frontier_rl.examples import analyze_mountaincar_neural_transfer_v1 as verifier


def _write(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _nested_samples(counts: list[int]) -> list[float]:
    assert len(counts) == verifier.N_TASKS
    assert all(left >= right for left, right in zip(counts, counts[1:]))
    values = [verifier.THRESHOLDS[0] - 0.05] * (
        verifier.EVAL_EPISODES_PER_TASK - counts[0]
    )
    for index in range(verifier.N_TASKS - 1):
        midpoint = (verifier.THRESHOLDS[index] + verifier.THRESHOLDS[index + 1]) / 2.0
        values.extend([midpoint] * (counts[index] - counts[index + 1]))
    values.extend([verifier.THRESHOLDS[-1] + 0.01] * counts[-1])
    assert len(values) == verifier.EVAL_EPISODES_PER_TASK
    return values


def _evaluation(offset: int) -> tuple[list, list, list, list]:
    base = np.asarray([32, 28, 24, 20, 16, 12, 8, 4], dtype=int)
    samples = []
    pass_curve = []
    for checkpoint in range(6):
        addition = round(offset * checkpoint / 5)
        counts = np.minimum(verifier.EVAL_EPISODES_PER_TASK, base + addition).tolist()
        row = _nested_samples(counts)
        checkpoint_samples = [list(row) for _ in range(verifier.N_TASKS)]
        rates = [
            sum(position >= threshold for position in task_row)
            / verifier.EVAL_EPISODES_PER_TASK
            for threshold, task_row in zip(verifier.THRESHOLDS, checkpoint_samples)
        ]
        samples.append(checkpoint_samples)
        pass_curve.append(rates)
    mean_curve = [float(np.mean(row)) for row in pass_curve]
    hardest_curve = [float(row[-1]) for row in pass_curve]
    return samples, pass_curve, mean_curve, hardest_curve


def _rollout_record(episode_seed: int, task: int, success: bool) -> dict:
    threshold = verifier.THRESHOLDS[task]
    if success:
        max_before = threshold - 0.01
        final = threshold + 0.001
    else:
        max_before = threshold - 0.02
        final = threshold - 0.01
    return {
        "episode_seed": int(episode_seed),
        "n_steps": verifier.MAX_EPISODE_STEPS,
        "max_position": max(max_before, final),
        "max_position_before_final": max_before,
        "pre_final_position": max_before,
        "final_position": final,
        "native_terminated": bool(success and task == verifier.N_TASKS - 1),
        "native_truncated": True,
        "native_reward_sum": -float(verifier.MAX_EPISODE_STEPS),
        "success": bool(success),
    }


def _run(condition: dict, seed: int, offset: int) -> dict:
    sampler = verifier._sampler(condition["sampling"], seed)
    episode_rng = np.random.default_rng(seed + verifier.TRAINING_EPISODE_SEED_OFFSET)
    groups = []
    updates = []
    task_groups = [0] * verifier.N_TASKS
    task_successes = [0] * verifier.N_TASKS
    task_transitions = [0] * verifier.N_TASKS
    slot_count = verifier.CAPACITY[condition["architecture"]][1]
    slot_update_calls = [0] * slot_count
    transitions = 0
    optimizer_updates = 0
    current_hash = hashlib.sha256(
        f"{condition['name']}:{seed}:initial".encode("utf-8")
    ).hexdigest()
    regimes = {"dead": 0, "mixed": 0, "all_pass": 0}

    # 157 complete 3,200-transition groups cross 500k at 502,400.
    for index in range(1, 158):
        start = transitions
        parameter_before = current_hash
        task, probabilities = sampler.draw()
        successes = (
            0
            if condition["sampling"] == "hardest_only"
            else (0, 8, 16)[(index - 1) % 3]
        )
        rollouts = [
            _rollout_record(
                int(episode_rng.integers(0, 2**31 - 1)),
                task,
                rollout < successes,
            )
            for rollout in range(verifier.N_ROLLOUTS)
        ]
        count = sum(record["n_steps"] for record in rollouts)
        transitions += count
        sampler.observe(task, successes)
        regime = (
            "dead"
            if successes == 0
            else "all_pass" if successes == verifier.N_ROLLOUTS else "mixed"
        )
        update_source = None
        if regime == "mixed":
            optimizer_updates += 1
            update_source = "requested_live"
            slot = 0 if condition["architecture"] == "shared_h64" else task
            slot_update_calls[slot] += 1
            gradient_norm = 1.0 + task / 10.0
            updates.append(
                {
                    "optimizer_update": optimizer_updates,
                    "after_group": index,
                    "transitions": transitions,
                    "source": "requested_live",
                    "requested_task": task,
                    "credited_task": task,
                    "task_id": task,
                    "slot": slot,
                    "gradient_norm": gradient_norm,
                    "update_norm": verifier.LEARNING_RATE * gradient_norm,
                    "n_trajectories": verifier.N_ROLLOUTS,
                    "n_score_terms": count,
                    "n_weighted_score_terms": count,
                    "weight_l1": 1.0,
                    "mean_policy_entropy": 1.0,
                    "frozen_group_parameters": True,
                    "applied": True,
                }
            )
            current_hash = hashlib.sha256(
                f"{condition['name']}:{seed}:update:{index}".encode("utf-8")
            ).hexdigest()
        groups.append(
            {
                "group": index,
                "transition_start": start,
                "transition_end": transitions,
                "n_transitions": count,
                "task_id": task,
                "success_count": successes,
                "regime": regime,
                "sampling_probabilities": probabilities.tolist(),
                "sampled_task_probability": float(probabilities[task]),
                "optimizer_updates_after_group": optimizer_updates,
                "update_source": update_source,
                "parameter_sha256_before_group": parameter_before,
                "parameter_sha256_after_group": current_hash,
                "rollouts": rollouts,
            }
        )
        regimes[regime] += 1
        task_groups[task] += 1
        task_successes[task] += successes
        task_transitions[task] += count

    x = [0, 100_000, 200_000, 300_000, 400_000, 500_000]
    triggers = [
        0,
        *[
            next(
                group["transition_end"]
                for group in groups
                if group["transition_end"] >= checkpoint
            )
            for checkpoint in x[1:]
        ],
    ]
    groups_by_end = {group["transition_end"]: group for group in groups}
    policy_sources = ["initial"]
    policy_hashes = [groups[0]["parameter_sha256_before_group"]]
    for checkpoint, trigger in zip(x[1:], triggers[1:]):
        exact = checkpoint == trigger
        policy_sources.append("post_exact_group" if exact else "pre_crossing_group")
        policy_hashes.append(
            groups_by_end[trigger][
                (
                    "parameter_sha256_after_group"
                    if exact
                    else "parameter_sha256_before_group"
                )
            ]
        )

    samples, pass_curve, mean_curve, hardest_curve = _evaluation(offset)
    architecture = condition["architecture"]
    hidden, slots, total, active = verifier.CAPACITY[architecture]
    projection = [
        {
            "group": record["group"],
            "transition_end": record["transition_end"],
            "task_id": record["task_id"],
            "success_count": record["success_count"],
            "optimizer_updates_after_group": record["optimizer_updates_after_group"],
        }
        for record in groups
    ]
    uniform = condition["sampling"] == "uniform"
    return {
        "seed": seed,
        "condition": condition["name"],
        "sampling": condition["sampling"],
        "architecture": architecture,
        "hindsight": False,
        "relabel_candidates": 0,
        "relabeled_groups": 0,
        "transition_budget": verifier.TRANSITION_BUDGET,
        "transitions": transitions,
        "transition_at_budget_crossing": transitions,
        "budget_crossing_group": len(groups),
        "complete_group_overshoot": transitions - verifier.TRANSITION_BUDGET,
        "post_budget_alignment_groups": 0,
        "post_budget_alignment_transitions": 0,
        "reached_transition_budget": True,
        "sampled_groups": len(groups),
        "optimizer_updates": optimizer_updates,
        "live_applied_updates": optimizer_updates,
        "zero_gradient_update_attempts": 0,
        "dead_groups": regimes["dead"],
        "mixed_groups": regimes["mixed"],
        "all_pass_groups": regimes["all_pass"],
        "task_groups": task_groups,
        "task_rollouts": [count * verifier.N_ROLLOUTS for count in task_groups],
        "task_successes": task_successes,
        "task_transitions": task_transitions,
        "group_diagnostics": groups,
        "task_sequence": [record["task_id"] for record in groups],
        "registered_uniform_task_schedule_length": (
            verifier.MAX_GROUPS_FOR_BUDGET if uniform else None
        ),
        "registered_uniform_task_schedule_sha256": (
            verifier._uniform_task_schedule_sha256(seed) if uniform else None
        ),
        "update_diagnostics": updates,
        "zero_gradient_diagnostics": [],
        "x_transitions": x,
        "pass_rate_curve": pass_curve,
        "mean_pass_curve": mean_curve,
        "hardest_pass_curve": hardest_curve,
        "evaluation_rng_preserved": [True] * len(x),
        "shared_nested_evaluations": (
            [True] * len(x) if architecture == "shared_h64" else [None] * len(x)
        ),
        "evaluation_trigger_transitions": triggers,
        "evaluation_policy_sources": policy_sources,
        "evaluation_policy_parameter_sha256": policy_hashes,
        "evaluation_max_position_samples": samples,
        "evaluation_episode_seeds": np.random.default_rng(
            seed + verifier.EVALUATION_SEED_BASE
        )
        .integers(
            0,
            2**31 - 1,
            size=verifier.EVAL_EPISODES_PER_TASK,
            dtype=np.int64,
        )
        .astype(int)
        .tolist(),
        "evaluation_action_seeds": np.random.default_rng(
            seed + verifier.EVALUATION_ACTION_SEED_BASE
        )
        .integers(
            0,
            2**31 - 1,
            size=verifier.EVAL_EPISODES_PER_TASK,
            dtype=np.int64,
        )
        .astype(int)
        .tolist(),
        "wall_seconds_at_evaluations": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "auc_mean_pass_fixed_transitions": verifier.fixed_budget_auc(
            mean_curve, x, verifier.TRANSITION_BUDGET
        ),
        "auc_hardest_pass_fixed_transitions": verifier.fixed_budget_auc(
            hardest_curve, x, verifier.TRANSITION_BUDGET
        ),
        "final_mean_pass_at_budget": verifier.fixed_budget_value(
            mean_curve, x, verifier.TRANSITION_BUDGET
        ),
        "final_hardest_pass_at_budget": verifier.fixed_budget_value(
            hardest_curve, x, verifier.TRANSITION_BUDGET
        ),
        "initial_pass_rates": pass_curve[0],
        "actor": {
            "mode": architecture,
            "hidden_size": hidden,
            "n_slots": slots,
            "parameter_count": total,
            "active_parameter_count": active,
            "parameter_sha256": current_hash,
            "parameter_norm": 2.0,
            "update_calls": regimes["mixed"],
            "applied_updates": optimizer_updates,
            "slot_update_calls": slot_update_calls,
        },
        "training_group_trace_sha256": verifier._canonical_hash(projection),
        "wall_seconds": 1.0,
        "numeric_valid": True,
    }


def _artifact(lock_path: Path) -> dict:
    lock = verifier._expected_development_lock_payload()
    _write(lock_path, lock)
    lock_record = {
        "schema": verifier.LOCK_SCHEMA,
        "scope": "development_only",
        "file_sha256": verifier._sha256(lock_path),
        "canonical_sha256": verifier._canonical_hash(lock),
        "payload": copy.deepcopy(lock),
    }
    offsets = {
        "frontier_shared_h64": 8,
        "uniform_shared_h64": 5,
        "hardest_shared_h64": 0,
        "uniform_disjoint_total_h8x8": 3,
        "uniform_disjoint_active_h64x8": 4,
    }
    cases = {}
    for condition in verifier.CONDITIONS:
        runs = [
            _run(condition, seed, offsets[condition["name"]])
            for seed in verifier.DEVELOPMENT_SEEDS
        ]
        cases[condition["name"]] = {
            "config": copy.deepcopy(condition),
            "runs": runs,
            "summary": verifier._summary(runs),
        }
    artifact = {
        "schema": verifier.SCHEMA,
        "study": verifier.STUDY,
        "stage": "development_feasibility",
        "registration_status": "sealed_development_protocol",
        "confirmatory_execution_available": False,
        "created_utc": "2026-07-22T00:00:00+00:00",
        "provenance": {
            "runtime": verifier._runtime(),
            "source_sha256": verifier._source_hashes(),
            "source_lock": lock_record,
        },
        "seed_collision_audit": verifier._expected_seed_audit(),
        "protocol": verifier._expected_protocol(),
        "paired_development_seeds": list(verifier.DEVELOPMENT_SEEDS),
        "confirmatory_seeds_reserved_untouched": list(verifier.CONFIRMATORY_SEEDS),
        "conditions": [copy.deepcopy(value) for value in verifier.CONDITIONS],
        "cases": cases,
        "development_contrasts": verifier._descriptives(cases),
        "development_gates": verifier._gates(cases),
        "artifact_state": "complete",
    }
    assert artifact["development_gates"]["development_feasible"]
    return artifact


@pytest.fixture(scope="module")
def valid_bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("mountaincar_v1r2_bundle")
    lock_path = directory / "development_lock.json"
    return _artifact(lock_path), lock_path


def test_analyzer_is_structurally_independent_and_manifest_is_exact():
    tree = ast.parse(inspect.getsource(verifier))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        f"{node.module}.{alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
        for alias in node.names
    }
    assert not any("run_mountaincar_neural_transfer_v1" in name for name in imported)
    assert not any("mountaincar_neural_transfer_v1_core" in name for name in imported)
    assert verifier.REQUIRED_SOURCE_FILES == (
        "frontier_rl/examples/mountaincar_neural_transfer_v1_core.py",
        "frontier_rl/examples/run_mountaincar_neural_transfer_v1.py",
        "frontier_rl/examples/analyze_mountaincar_neural_transfer_v1.py",
        "frontier_rl/examples/MOUNTAINCAR_NEURAL_TRANSFER_PROTOCOL_V1.md",
        "frontier_rl/examples/test_mountaincar_neural_transfer_v1.py",
        "frontier_rl/examples/test_analyze_mountaincar_neural_transfer_v1.py",
        "frontier_rl/__init__.py",
        "frontier_rl/interfaces.py",
        "frontier_rl/teacher.py",
        "frontier_rl/estimators.py",
        "frontier_rl/trainer.py",
    )
    assert tuple(verifier._source_hashes()) == verifier.REQUIRED_SOURCE_FILES


def test_uniform_schedule_is_outcome_independent_and_hashed_manually():
    seed = verifier.DEVELOPMENT_SEEDS[0]
    rng = np.random.default_rng(seed + 4_000_000)
    probabilities = np.full(8, 1.0 / 8.0)
    manual = [
        int(rng.choice(8, p=probabilities))
        for _ in range(verifier.MAX_GROUPS_FOR_BUDGET)
    ]
    assert list(verifier._uniform_task_sequence(seed)) == manual
    payload = json.dumps(
        manual, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    assert (
        verifier._uniform_task_schedule_sha256(seed)
        == hashlib.sha256(payload).hexdigest()
    )


def test_protocol_makes_hardest_auc_primary_and_mean_auc_supporting():
    protocol = verifier._expected_protocol()
    assert protocol["primary_metric"].startswith("hardest-goal pass AUC")
    assert protocol["primary_estimand"].startswith("paired mean method difference")
    assert protocol["supporting_metric"].startswith("target-uniform mean-pass AUC")
    assert "cannot rescue" in protocol["claim_rule"]
    assert protocol["uniform_task_schedule"]["registered_length"] == 31_250


def test_valid_sealed_artifact_passes_independent_verification(valid_bundle, tmp_path):
    artifact, lock_path = valid_bundle
    artifact_path = tmp_path / "artifact.json"
    _write(artifact_path, artifact)
    report = verifier.verify(artifact_path, lock_path)
    assert report["all_checks_passed"]
    assert report["sealed_development_protocol_verified"]
    assert report["training_rollout_audits_independently_recomputed"]
    assert report["evaluation_samples_and_crns_independently_recomputed"]
    assert report["primary_metric"] == "auc_hardest_pass_fixed_transitions"
    assert report["development_lock_sha256"] == verifier._sha256(lock_path)
    assert report["checked_source_files"] == list(verifier.REQUIRED_SOURCE_FILES)


def test_rank_disagreement_cannot_make_mean_auc_the_primary_result():
    cases = {}
    for condition in verifier.CONDITIONS:
        name = condition["name"]
        hardest = 1.0 if name == "frontier_shared_h64" else 0.0
        mean = 0.0 if name == "frontier_shared_h64" else 1.0
        cases[name] = {
            "runs": [
                {
                    "seed": seed,
                    "auc_hardest_pass_fixed_transitions": hardest,
                    "auc_mean_pass_fixed_transitions": mean,
                }
                for seed in verifier.DEVELOPMENT_SEEDS
            ]
        }
    contrast = verifier._descriptives(cases)["frontier_minus_uniform_shared"]
    assert contrast["primary_metric"] == "auc_hardest_pass_fixed_transitions"
    assert contrast["primary_mean_delta"] == 1.0
    assert contrast["supporting_metric"] == "auc_mean_pass_fixed_transitions"
    assert contrast["supporting_mean_delta"] == -1.0


def test_uniform_common_prefix_allows_unequal_consumed_lengths(valid_bundle):
    artifact, _ = valid_bundle
    cases = copy.deepcopy(artifact["cases"])
    shorter = cases["uniform_disjoint_total_h8x8"]["runs"][0]["task_sequence"]
    cases["uniform_disjoint_total_h8x8"]["runs"][0]["task_sequence"] = shorter[:-7]
    assert verifier._paired_uniform_schedules_valid(cases)
    cases["uniform_disjoint_total_h8x8"]["runs"][0]["task_sequence"][3] = (
        cases["uniform_disjoint_total_h8x8"]["runs"][0]["task_sequence"][3] + 1
    ) % verifier.N_TASKS
    assert not verifier._paired_uniform_schedules_valid(cases)


def _frontier_run(valid_bundle) -> tuple[dict, dict]:
    artifact, _ = valid_bundle
    condition = verifier.CONDITIONS[0]
    return copy.deepcopy(artifact["cases"][condition["name"]]["runs"][0]), condition


@pytest.mark.parametrize(
    "mutation",
    (
        "episode_seed",
        "n_steps",
        "max_position_before_final",
        "pre_final_position",
        "final_position",
        "native_terminated",
        "native_truncated",
        "native_reward_sum",
        "success",
        "group_transitions",
        "group_successes",
        "group_regime",
    ),
)
def test_rollout_audit_and_derived_group_tampering_fail_closed(valid_bundle, mutation):
    run, condition = _frontier_run(valid_bundle)
    group = run["group_diagnostics"][1]  # mixed, so rollout zero is successful
    rollout = group["rollouts"][0]
    if mutation == "episode_seed":
        rollout["episode_seed"] += 1
    elif mutation == "n_steps":
        rollout["n_steps"] -= 1
    elif mutation == "max_position_before_final":
        rollout["max_position_before_final"] = verifier.THRESHOLDS[group["task_id"]]
    elif mutation == "pre_final_position":
        rollout["pre_final_position"] = rollout["max_position_before_final"] + 0.1
    elif mutation == "final_position":
        rollout["final_position"] = verifier.THRESHOLDS[group["task_id"]] - 0.1
    elif mutation == "native_terminated":
        rollout["native_terminated"] = not rollout["native_terminated"]
    elif mutation == "native_truncated":
        rollout["native_truncated"] = False
    elif mutation == "native_reward_sum":
        rollout["native_reward_sum"] += 1.0
    elif mutation == "success":
        rollout["success"] = False
    elif mutation == "group_transitions":
        group["n_transitions"] += 1
    elif mutation == "group_successes":
        group["success_count"] += 1
    elif mutation == "group_regime":
        group["regime"] = "dead"
    with pytest.raises(ValueError):
        verifier._validate_run(run, condition, verifier.DEVELOPMENT_SEEDS[0])


@pytest.mark.parametrize(
    "mutation",
    (
        "episode_seed",
        "action_seed",
        "sample",
        "shared_row",
        "pass_rate",
        "mean_curve",
        "hardest_curve",
        "hardest_auc",
        "mean_auc",
    ),
)
def test_evaluation_sample_crn_curve_and_auc_tampering_fail_closed(
    valid_bundle, mutation
):
    run, condition = _frontier_run(valid_bundle)
    if mutation == "episode_seed":
        run["evaluation_episode_seeds"][0] += 1
    elif mutation == "action_seed":
        run["evaluation_action_seeds"][0] += 1
    elif mutation == "sample":
        run["evaluation_max_position_samples"][2][0][0] = -1.1
    elif mutation == "shared_row":
        run["evaluation_max_position_samples"][2][1][0] += 1e-6
    elif mutation == "pass_rate":
        run["pass_rate_curve"][2][3] += 1.0 / verifier.EVAL_EPISODES_PER_TASK
    elif mutation == "mean_curve":
        run["mean_pass_curve"][2] += 0.01
    elif mutation == "hardest_curve":
        run["hardest_pass_curve"][2] += 0.01
    elif mutation == "hardest_auc":
        run["auc_hardest_pass_fixed_transitions"] += 0.01
    elif mutation == "mean_auc":
        run["auc_mean_pass_fixed_transitions"] += 0.01
    with pytest.raises(ValueError):
        verifier._validate_run(run, condition, verifier.DEVELOPMENT_SEEDS[0])


@pytest.mark.parametrize(
    "mutation",
    ("task_sequence", "schedule_length", "schedule_hash"),
)
def test_uniform_registered_schedule_tampering_fails_closed(valid_bundle, mutation):
    artifact, _ = valid_bundle
    condition = verifier.CONDITIONS[1]
    run = copy.deepcopy(artifact["cases"][condition["name"]]["runs"][0])
    if mutation == "task_sequence":
        run["task_sequence"][0] = (run["task_sequence"][0] + 1) % verifier.N_TASKS
    elif mutation == "schedule_length":
        run["registered_uniform_task_schedule_length"] -= 1
    else:
        run["registered_uniform_task_schedule_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        verifier._validate_run(run, condition, verifier.DEVELOPMENT_SEEDS[0])


def _verify_mutated_artifact(artifact: dict, lock_path: Path, tmp_path: Path) -> None:
    artifact_path = tmp_path / "mutated.json"
    _write(artifact_path, artifact)
    with pytest.raises(ValueError):
        verifier.verify(artifact_path, lock_path)


@pytest.mark.parametrize(
    "mutation",
    (
        "schema",
        "registration_status",
        "artifact_source_missing",
        "artifact_source_extra",
        "embedded_lock_hash",
        "embedded_lock_payload",
        "summary",
        "contrasts",
        "gates",
    ),
)
def test_artifact_lock_manifest_and_aggregate_tampering_fail_closed(
    valid_bundle, tmp_path, mutation
):
    original, lock_path = valid_bundle
    artifact = copy.deepcopy(original)
    if mutation == "schema":
        artifact["schema"] = "wrong"
    elif mutation == "registration_status":
        artifact["registration_status"] = "unsealed_development_design"
    elif mutation == "artifact_source_missing":
        artifact["provenance"]["source_sha256"].pop(
            next(iter(verifier.REQUIRED_SOURCE_FILES))
        )
    elif mutation == "artifact_source_extra":
        artifact["provenance"]["source_sha256"]["extra.py"] = "0" * 64
    elif mutation == "embedded_lock_hash":
        artifact["provenance"]["source_lock"]["file_sha256"] = "0" * 64
    elif mutation == "embedded_lock_payload":
        artifact["provenance"]["source_lock"]["payload"]["scope"] = "confirmatory"
    elif mutation == "summary":
        artifact["cases"]["frontier_shared_h64"]["summary"][
            "auc_hardest_pass_fixed_transitions"
        ]["mean"] += 0.1
    elif mutation == "contrasts":
        artifact["development_contrasts"]["frontier_minus_uniform_shared"][
            "primary_mean_delta"
        ] += 0.1
    elif mutation == "gates":
        artifact["development_gates"]["learning_outcomes_authorize_claims"] = True
    _verify_mutated_artifact(artifact, lock_path, tmp_path)


@pytest.mark.parametrize(
    "mutation",
    ("scope", "runtime", "source_missing", "source_extra", "protocol", "condition"),
)
def test_external_lock_must_exactly_match_live_development_payload(
    valid_bundle, tmp_path, mutation
):
    artifact, original_lock_path = valid_bundle
    lock = json.loads(original_lock_path.read_text(encoding="utf-8"))
    if mutation == "scope":
        lock["scope"] = "confirmatory"
    elif mutation == "runtime":
        lock["runtime"]["python_version"] = "0.0"
    elif mutation == "source_missing":
        lock["source_sha256"].pop(next(iter(verifier.REQUIRED_SOURCE_FILES)))
    elif mutation == "source_extra":
        lock["source_sha256"]["extra.py"] = "0" * 64
    elif mutation == "protocol":
        lock["protocol"]["primary_metric"] = "mean AUC"
    elif mutation == "condition":
        lock["conditions"][0]["sampling"] = "uniform"
    lock_path = tmp_path / "wrong_lock.json"
    artifact_path = tmp_path / "artifact.json"
    _write(lock_path, lock)
    _write(artifact_path, artifact)
    with pytest.raises(ValueError):
        verifier.verify(artifact_path, lock_path)


def test_cli_requires_lock_and_invalid_input_writes_no_report(valid_bundle, tmp_path):
    _, lock_path = valid_bundle
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    output = tmp_path / "must_not_exist.json"
    command = [
        sys.executable,
        "-m",
        "frontier_rl.examples.analyze_mountaincar_neural_transfer_v1",
        str(malformed),
        "--lock",
        str(lock_path),
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=verifier.PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert not output.exists()

    missing_lock = subprocess.run(
        command[:4],
        cwd=verifier.PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing_lock.returncode != 0
    assert not output.exists()
