"""Synthetic-only tests for the strict BARN campaign-cell merger."""

from __future__ import annotations

import copy
import json
from collections import Counter

import pytest

from icra2027 import merge_barn_campaign as merger


HASHES = {
    "manifest_sha256": "1" * 64,
    "split_sha256": "2" * 64,
    "prereg_sha256": "3" * 64,
    "analyzer_sha256": "4" * 64,
    "protocol_sha256": "5" * 64,
    "container_sha256": "6" * 64,
    "source_sha256": "7" * 64,
}


def _protocol(seeds=(1, 2)):
    primary_orders = {}
    ablation_orders = {}
    primary = list(merger.PRIMARY_ARMS)
    for offset, seed in enumerate(seeds):
        primary_orders[str(seed)] = primary[offset % 4:] + primary[:offset % 4]
        ablation_orders[str(seed)] = (
            list(merger.ABLATION_ARMS) if offset % 2 == 0
            else list(reversed(merger.ABLATION_ARMS)))
    return {
        "schema_version": 1,
        "status": "FROZEN",
        "protocol_id": "synthetic-barn-v1",
        "domain": merger.DOMAIN,
        "dataset": {
            "archive_sha256": "8" * 64,
            "manifest_sha256": HASHES["manifest_sha256"],
            "split_sha256": HASHES["split_sha256"],
            "split_seed": 20270811,
            "n_strata": 10,
            "n_train_courses": 240,
            "n_heldout_courses": 60,
        },
        "environment": {
            "container_sha256": HASHES["container_sha256"],
            "cpu_only": True,
            "max_step_size": 0.005,
            "real_time_update_rate": 2000,
            "episode_timeout": 25.0,
        },
        "shared_training": {
            "seeds": list(seeds),
            "tasks_per_step": 2,
            "eval_episodes": 1,
            "training_sim_step_budget": 60,
            "eval_sim_step_interval": 40,
            "max_training_updates": 2,
            "eval_every": 1,
            "teacher_floor": 0.1,
            "teacher_decay": 0.7,
            "teacher_gamma": 1.0,
            "staged_initial_strata": 1,
            "staged_promotion_threshold": 0.7,
            "staged_min_frontier_groups": 5,
        },
        "primary": {
            "evidence_status": merger.PRIMARY_EVIDENCE_STATUS,
            "arms": list(merger.PRIMARY_ARMS),
            "n_rollouts": 2,
            "execution_order_by_seed": primary_orders,
        },
        "ablation": {
            "evidence_status": merger.ABLATION_EVIDENCE_STATUS,
            "arms": list(merger.ABLATION_ARMS),
            "n_values": [2, 4, 8, 16],
            "fresh_cell_names": [
                "ablation_n2", "ablation_n4", "ablation_n16"],
            "n8_source": "primary_ours_uN_and_learnability",
            "execution_order_by_seed": ablation_orders,
        },
        "analysis": {
            "analyzer_sha256": HASHES["analyzer_sha256"],
            "primary_currency": "sim_steps",
        },
        "artifact_requirements": {
            "heldout_course_count": 60,
            "difficulty_bin_count": 10,
            "training_episode_records": True,
            "evaluation_episode_records": True,
            "per_checkpoint_status_counts": True,
            "per_checkpoint_teacher_vector_length": 10,
            "canonical_result_arm_order": True,
        },
        "isolation": {
            "seed_stride": 2,
            "eval_offset": 1,
            "domain_base_by_cell": {
                "primary": 20,
                "ablation_n2": 50,
                "ablation_n4": 80,
                "ablation_n16": 110,
            },
            "master_port_seed_stride": 4,
            "eval_master_port_offset": 1,
            "master_port_base_by_cell": {
                "primary": 13000,
                "ablation_n2": 14000,
                "ablation_n4": 15000,
                "ablation_n16": 16000,
            },
        },
        "retry": {
            "selection": (
                "earliest_submitted_complete_hash_valid_attempt_per_seed"),
            "partial_attempts_retained": True,
            "endpoint_blind_selection": True,
        },
    }


def _strata(first: int, count: int, per_stratum: int):
    rows = []
    for stratum in range(10):
        indices = list(range(
            first + stratum * per_stratum,
            first + (stratum + 1) * per_stratum))
        rows.append({
            "stratum": stratum,
            "n_courses": per_stratum,
            "difficulty_min": float(indices[0] + 1),
            "difficulty_max": float(indices[-1] + 1),
            "course_ids": [f"barn-{index:03d}" for index in indices],
        })
    assert sum(row["n_courses"] for row in rows) == count
    return rows


def _teacher(groups: int):
    visits = [0] * 10
    for group in range(groups):
        visits[group % 10] += 1
    return {
        "posterior_mean": [0.5] * 10,
        "sampling_weights_at_posterior_mean": [0.1] * 10,
        "visits": visits,
    }


def _evaluation(seed: int, arm_index: int, step: int, eval_seed: int,
                teacher: dict):
    heldout_ids = [f"barn-{index:03d}" for index in range(240, 300)]
    per_course = []
    episodes = []
    rates = []
    statuses = Counter()
    for offset, env_id in enumerate(heldout_ids):
        barn_index = 240 + offset
        success = int((barn_index + seed + arm_index + step) % 2 == 0)
        status = "succeeded" if success else "timeout"
        course_seed = merger._derived_seed(0xE7A1C0DE, eval_seed, barn_index)
        rates.append(float(success))
        statuses[status] += 1
        per_course.append({
            "env_id": env_id,
            "barn_index": barn_index,
            "difficulty": float(barn_index + 1),
            "stratum": offset // 6,
            "seed": course_seed,
            "successes": success,
            "episodes": 1,
            "success_rate": float(success),
            "sim_steps": 10,
        })
        episodes.append({
            "env_id": env_id,
            "barn_index": barn_index,
            "seed": course_seed,
            "episode_index": 0,
            "success": success,
            "status": status,
            "sim_steps": 10,
            "sim_seconds": 0.05,
            "planned_clearance_m": 0.4,
        })
    bins = []
    bin_rates = []
    for index in range(10):
        chunk = rates[index * 6:(index + 1) * 6]
        rate = sum(chunk) / 6
        bin_rates.append(rate)
        bins.append({
            "bin": index,
            "n_courses": 6,
            "difficulty_min": float(241 + index * 6),
            "difficulty_max": float(246 + index * 6),
            "mean_success": rate,
        })
    stratum_rates = bin_rates
    pairs = [(0.5, stratum_rates[index]) for index, visits in
             enumerate(teacher["visits"]) if visits >= 1]
    evaluation = {
        "mean_success": sum(rates) / 60,
        "n_tasks": 60,
        "eval_episodes": 60,
        "eval_sim_steps": 600,
        "success@1": sum(rates) / 60,
        "easy_decile_retention": sum(rates[:6]) / 6,
        "teacher/calibration_n": len(pairs),
        "per_task_success": rates,
        "heldout_course_ids": heldout_ids,
        "per_course": per_course,
        "episode_records": episodes,
        "success_by_difficulty_bin": bin_rates,
        "difficulty_bins": bins,
        "status_counts": dict(sorted(statuses.items())),
    }
    if pairs:
        evaluation["teacher/calibration_bias"] = sum(
            estimate - observed for estimate, observed in pairs) / len(pairs)
        evaluation["teacher/calibration_mae"] = sum(
            abs(estimate - observed) for estimate, observed in pairs) / len(pairs)
    return evaluation


def _training_records(final_step: int, n_rollouts: int):
    records = []
    total_groups = final_step * 2
    for group in range(total_groups):
        stratum = group % 10
        course_id = f"barn-{stratum * 24:03d}"
        for group_episode in range(n_rollouts):
            success = int(group_episode % 2 == 0)
            records.append({
                "episode_index": len(records),
                "group_episode_index": group_episode,
                "stratum": stratum,
                "course_id": course_id,
                "difficulty": float(stratum * 24 + 1),
                "success": success,
                "status": "succeeded" if success else "timeout",
                "sim_steps": 10,
                "sim_seconds": 0.05,
                "planned_clearance_m": 0.4,
            })
    return records


def _run(seed: int, arm: str, arm_index: int, n_rollouts: int):
    sim_per_step = 2 * n_rollouts * 10
    final_step = 1 if sim_per_step >= 60 else 2
    records = _training_records(final_step, n_rollouts)
    teacher_seed = merger._derived_seed(0x7EAC4E12, seed)
    eval_seed = merger._derived_seed(0xE7A15EED, seed)
    history = []
    for step in range(final_step + 1):
        prefix = records[:step * 2 * n_rollouts]
        groups = step * 2
        teacher = _teacher(groups)
        status_counts = Counter(row["status"] for row in prefix)
        course_counts = Counter(row["course_id"] for row in prefix)
        stratum_counts = Counter(
            str(prefix[offset]["stratum"])
            for offset in range(0, len(prefix), n_rollouts))
        group_success = [
            sum(row["success"] for row in prefix[offset:offset + n_rollouts])
            for offset in range(0, len(prefix), n_rollouts)]
        all_fail = sum(value == 0 for value in group_success)
        all_pass = sum(value == n_rollouts for value in group_success)
        live = groups - all_fail - all_pass
        history.append({
            "step": step,
            "episodes": len(prefix),
            "sim_steps": sum(row["sim_steps"] for row in prefix),
            "training_wall_seconds": float(step * 3),
            "evaluation_wall_seconds": float(step + 1),
            "dead_group_rate": (all_fail + all_pass) / max(groups, 1),
            "all_fail_group_rate": all_fail / max(groups, 1),
            "all_pass_group_rate": all_pass / max(groups, 1),
            "live_groups": live,
            "dead_groups": all_fail + all_pass,
            "all_fail_groups": all_fail,
            "all_pass_groups": all_pass,
            "relabeled_groups": 0,
            "updates_live": live,
            "updates_relabel": 0,
            "sampled_stratum_counts": dict(sorted(stratum_counts.items())),
            "training_status_counts": dict(sorted(status_counts.items())),
            "training_course_counts": dict(sorted(course_counts.items())),
            "eval": _evaluation(seed, arm_index, step, eval_seed, teacher),
            "teacher": teacher,
        })
    return {
        "arm": arm,
        "seed": seed,
        "teacher_seed": teacher_seed,
        "eval_seed": eval_seed,
        **{
            field: merger._normalized_auc(
                history, currency,
                budget=(60 if currency == "sim_steps" else None))
            for field, currency in merger._AUC_FIELDS.items()
        },
        "final": copy.deepcopy(history[-1]),
        "history": history,
        "training_episode_records": records,
    }


def _artifact(seed: int, protocol: dict, *, campaign_cell="primary",
              attempt="attempt-001", submitted="2026-08-14T00:00:00Z",
              hashes=None):
    hashes = HASHES if hashes is None else hashes
    if campaign_cell == "primary":
        cell = protocol["primary"]
        n_rollouts = cell["n_rollouts"]
    else:
        cell = protocol["ablation"]
        n_rollouts = int(campaign_cell.removeprefix("ablation_n"))
    arms = list(cell["arms"])
    shared = protocol["shared_training"]
    environment = protocol["environment"]
    execution_order = list(cell["execution_order_by_seed"][str(seed)])
    config = {
        "arms": arms,
        "execution_order": execution_order,
        "campaign_cell": campaign_cell,
        "protocol_id": protocol["protocol_id"],
        "seeds": 1,
        "seed_start": seed,
        "seed_list": [seed],
        "steps": shared["max_training_updates"],
        "n_rollouts": n_rollouts,
        "tasks_per_step": shared["tasks_per_step"],
        "eval_every": shared["eval_every"],
        "eval_episodes": shared["eval_episodes"],
        "training_sim_step_budget": shared["training_sim_step_budget"],
        "eval_sim_step_interval": shared["eval_sim_step_interval"],
        "max_training_updates": shared["max_training_updates"],
        "episode_timeout": environment["episode_timeout"],
        "max_step_size": environment["max_step_size"],
        "real_time_update_rate": environment["real_time_update_rate"],
        "hindsight": False,
        "estimator": "maxrl",
        "teacher_gamma": shared["teacher_gamma"],
        "teacher_decay": shared["teacher_decay"],
        "teacher_floor": shared["teacher_floor"],
        "teacher_unit": "frozen_difficulty_stratum",
        "n_strata": 10,
        "difficulty_metadata": (
            "published optimal traversal time seconds; longer is harder"),
        "staged_initial_strata": shared["staged_initial_strata"],
        "staged_promotion_threshold": shared["staged_promotion_threshold"],
        "staged_min_frontier_groups": shared["staged_min_frontier_groups"],
        "n_train_courses": 240,
        "n_heldout_courses": 60,
        "train_strata": _strata(0, 240, 24),
        "heldout_strata": _strata(240, 60, 6),
        "campaign_seed": seed,
        "split_seed": protocol["dataset"]["split_seed"],
        "domain_id": protocol["isolation"]["domain_base_by_cell"][campaign_cell]
        + 2 * seed,
        "eval_domain_id": protocol["isolation"]["domain_base_by_cell"][campaign_cell]
        + 2 * seed + 1,
        "master_port": protocol["isolation"]["master_port_base_by_cell"][
            campaign_cell] + 4 * seed,
        "eval_master_port": protocol["isolation"]["master_port_base_by_cell"][
            campaign_cell] + 4 * seed + 1,
        "runtime_root": f"/runtime/{campaign_cell}/seed-{seed}/{attempt}",
        "smoke": False,
        "engineering_course_id": None,
        "evaluation_partition": "frozen_heldout",
    }
    results = {
        arm: [_run(seed, arm, index, n_rollouts)]
        for index, arm in enumerate(arms)}
    return {
        "schema_version": 1,
        "created_utc": "2026-08-14T03:00:00+00:00",
        "evidence_status": cell["evidence_status"],
        "domain": merger.DOMAIN,
        "execution": {
            "campaign_id": "campaign-1",
            "attempt_id": attempt,
            "submitted_utc": submitted.replace("Z", "+00:00"),
            "slurm_job_id": str(123450 + seed),
            "slurm_array_job_id": "12345",
            "slurm_array_task_id": seed,
        },
        "heldout_protocol": (
            "fixed course-level seeds shared across arms and checkpoints; "
            "isolated evaluation adapter"),
        "provenance": {
            **hashes,
            "manifest_path": f"/job-{seed}/barn_manifest.jsonl",
            "split_path": f"/job-{seed}/barn_split.json",
            "prereg_path": f"/job-{seed}/prereg.md",
            "analyzer_path": f"/job-{seed}/analyze.py",
            "protocol_path": f"/job-{seed}/protocol.json",
            "dataset_root": f"/job-{seed}/dataset",
            "robot_sdf": f"/job-{seed}/robot.sdf",
            "split_bound_manifest_sha256": hashes["manifest_sha256"],
            "robot_sdf_sha256": "9" * 64,
            "asset_hashes_verified": True,
        },
        "config": config,
        "results": results,
    }


def _merge(artifacts, protocol, *, cell="primary", seeds=(1, 2), hashes=None):
    hashes = HASHES if hashes is None else hashes
    return merger._merge_campaign_artifacts_for_preflight(
        artifacts, protocol=protocol, campaign_cell=cell,
        expected_seeds=seeds,
        **{f"expected_{field}": value for field, value in hashes.items()})


def _selection_receipt(paths, artifacts, hashes, *, cell="primary"):
    selected = []
    for path, artifact in sorted(
            zip(paths, artifacts),
            key=lambda pair: pair[1]["config"]["campaign_seed"]):
        execution = artifact["execution"]
        selected.append({
            "seed": artifact["config"]["campaign_seed"],
            "attempt_id": execution["attempt_id"],
            "submitted_utc": execution["submitted_utc"],
            "slurm_array_job_id": execution["slurm_array_job_id"],
            "slurm_array_task_id": execution["slurm_array_task_id"],
            "slurm_job_id": execution["slurm_job_id"],
            "artifact_path": str(path.resolve()),
            "artifact_sha256": merger.sha256_path(path),
        })
    return {
        "schema_version": 1,
        "selection_rule": merger.SELECTION_RULE,
        "outcome_blind": True,
        "campaign_id": "campaign-1",
        "campaign_cell": cell,
        "expected_seed_list": [1, 2, 3, 4, 5],
        "expected_hashes": copy.deepcopy(hashes),
        "ledger_sha256": "8" * 64,
        "selected": selected,
        "excluded": [],
    }


def _production_fixture(tmp_path):
    seeds = (1, 2, 3, 4, 5)
    protocol = _protocol(seeds)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    hashes = {**HASHES, "protocol_sha256": merger.sha256_path(protocol_path)}
    artifacts = [_artifact(seed, protocol, hashes=hashes) for seed in seeds]
    paths = []
    for artifact in artifacts:
        seed = artifact["config"]["campaign_seed"]
        path = tmp_path / f"seed-{seed}.json"
        path.write_text(json.dumps(artifact) + "\n")
        paths.append(path)
    receipt = _selection_receipt(paths, artifacts, hashes)
    receipt_path = tmp_path / "selection.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return protocol, protocol_path, artifacts, paths, hashes, receipt, receipt_path


def test_primary_merge_sorts_and_records_per_seed_execution_order():
    protocol = _protocol()
    merged = _merge([_artifact(2, protocol), _artifact(1, protocol)], protocol)
    assert merged["config"]["seed_list"] == [1, 2]
    assert merged["merge"]["schema_version"] == 1
    assert "execution_order" not in merged["config"]
    assert [row["seed"] for row in merged["results"]["ours_uN"]] == [1, 2]
    assert merged["merge"]["per_seed_execution_order"] == [
        {"seed": seed,
         "execution_order": protocol["primary"]["execution_order_by_seed"][str(seed)]}
        for seed in (1, 2)]
    assert merged["evidence_status"] == merger.PRIMARY_EVIDENCE_STATUS
    assert set(merged["provenance"]) >= set(merger.PROVENANCE_HASH_FIELDS)


@pytest.mark.parametrize("cell,n_rollouts", [
    ("ablation_n2", 2), ("ablation_n4", 4), ("ablation_n16", 16)])
def test_fresh_ablation_cells_have_exact_two_arm_shape(cell, n_rollouts):
    protocol = _protocol()
    artifacts = [_artifact(seed, protocol, campaign_cell=cell) for seed in (1, 2)]
    merged = _merge(artifacts, protocol, cell=cell)
    assert merged["evidence_status"] == merger.ABLATION_EVIDENCE_STATUS
    assert merged["config"]["campaign_cell"] == cell
    assert merged["config"]["n_rollouts"] == n_rollouts
    assert list(merged["results"]) == list(merger.ABLATION_ARMS)


def test_fresh_n8_is_rejected_when_protocol_reuses_primary_n8():
    protocol = _protocol()
    artifact = _artifact(1, protocol, campaign_cell="ablation_n4")
    artifact["config"]["campaign_cell"] = "ablation_n8"
    with pytest.raises(merger.MergeValidationError, match="not declared"):
        _merge([artifact], protocol, cell="ablation_n8", seeds=(1, 2))


def test_hash_status_seed_and_arm_failures_are_closed():
    protocol = _protocol()
    first, second = _artifact(1, protocol), _artifact(2, protocol)
    second["provenance"]["protocol_sha256"] = "a" * 64
    with pytest.raises(merger.MergeValidationError, match="protocol_sha256"):
        _merge([first, second], protocol)

    first, second = _artifact(1, protocol), _artifact(2, protocol)
    del second["results"]["staged"]
    with pytest.raises(merger.MergeValidationError, match="missing/extra arms"):
        _merge([first, second], protocol)

    with pytest.raises(merger.MergeValidationError, match="duplicate"):
        _merge([first, copy.deepcopy(first)], protocol)
    with pytest.raises(merger.MergeValidationError, match="missing=\\[2\\]"):
        _merge([first], protocol)


def test_exact_cell_config_and_execution_order_are_enforced():
    protocol = _protocol()
    artifact = _artifact(1, protocol)
    artifact["config"]["n_rollouts"] = 4
    with pytest.raises(merger.MergeValidationError, match="n_rollouts"):
        _merge([artifact, _artifact(2, protocol)], protocol)

    artifact = _artifact(1, protocol)
    artifact["config"]["execution_order"].reverse()
    with pytest.raises(merger.MergeValidationError, match="execution_order"):
        _merge([artifact, _artifact(2, protocol)], protocol)


def test_execution_array_task_id_is_required_and_equals_seed():
    protocol = _protocol()
    artifact = _artifact(1, protocol)
    del artifact["execution"]["slurm_array_task_id"]
    with pytest.raises(merger.MergeValidationError, match="fields must be exact"):
        _merge([artifact, _artifact(2, protocol)], protocol)

    artifact = _artifact(1, protocol)
    artifact["execution"]["slurm_array_task_id"] = 2
    with pytest.raises(merger.MergeValidationError, match="must equal"):
        _merge([artifact, _artifact(2, protocol)], protocol)


def test_sim_step_auc_is_interpolated_at_exact_budget():
    history = [
        {"sim_steps": 0, "eval": {"mean_success": 0.0}},
        {"sim_steps": 40, "eval": {"mean_success": 1.0}},
        {"sim_steps": 80, "eval": {"mean_success": 1.0}},
    ]
    clipped = merger._normalized_auc(history, "sim_steps", budget=60)
    full = merger._normalized_auc(history, "sim_steps")
    assert clipped != full
    assert clipped == pytest.approx(2.0 / 3.0)
    assert full == pytest.approx(0.75)


@pytest.mark.parametrize("corruption,match", [
    ("panel", "heldout panel/order"),
    ("course_seed", "eval seed"),
    ("episode", "episode-record count"),
    ("status", "status_counts"),
    ("bins", "exactly 10 bins"),
    ("teacher", "teacher vectors"),
    ("training", "training episode arithmetic"),
    ("auc", "internally inconsistent"),
])
def test_strict_panel_records_training_and_auc_consistency(corruption, match):
    protocol = _protocol()
    artifact = _artifact(1, protocol)
    run = artifact["results"]["ours_uN"][0]
    checkpoint = run["history"][0]
    if corruption == "panel":
        checkpoint["eval"]["heldout_course_ids"].reverse()
    elif corruption == "course_seed":
        checkpoint["eval"]["per_course"][0]["seed"] += 1
    elif corruption == "episode":
        checkpoint["eval"]["episode_records"].pop()
    elif corruption == "status":
        checkpoint["eval"]["status_counts"]["timeout"] -= 1
    elif corruption == "bins":
        checkpoint["eval"]["difficulty_bins"].pop()
    elif corruption == "teacher":
        checkpoint["teacher"]["posterior_mean"].pop()
    elif corruption == "training":
        run["history"][1]["episodes"] += 1
    else:
        run["target_uniform_auc_by_sim_step"] += 0.1
    with pytest.raises(merger.MergeValidationError, match=match):
        _merge([artifact, _artifact(2, protocol)], protocol)


@pytest.mark.parametrize("field,collision,match", [
    ("domain_id", 22, "isolation mapping"),
    ("master_port", 13004, "isolation mapping"),
    ("runtime_root", "/runtime/primary/seed-1/attempt-001", "collision"),
])
def test_cross_seed_runtime_isolation_is_enforced(field, collision, match):
    protocol = _protocol()
    first, second = _artifact(1, protocol), _artifact(2, protocol)
    second["config"][field] = collision
    if field == "domain_id":
        second["config"]["eval_domain_id"] = collision + 1
    elif field == "master_port":
        second["config"]["eval_master_port"] = collision + 1
    with pytest.raises(merger.MergeValidationError, match=match):
        _merge([first, second], protocol)


def test_direct_receipt_free_merge_is_rejected():
    protocol = _protocol()
    artifacts = [_artifact(seed, protocol) for seed in (1, 2)]
    with pytest.raises(merger.MergeValidationError, match="receipt is required"):
        merger.merge_campaign_artifacts(
            artifacts, protocol=protocol, campaign_cell="primary",
            expected_seeds=(1, 2),
            **{f"expected_{field}": value
               for field, value in HASHES.items()})


@pytest.mark.parametrize("corruption,match", [
    ("schema", "schema"),
    ("blind", "outcome-blind"),
    ("campaign", "campaign differs"),
    ("cell", "campaign cell"),
    ("seeds", "seed list"),
    ("hashes", "expected hashes differ"),
    ("ledger", "ledger_sha256"),
    ("path", "paths differ"),
    ("artifact_sha", "artifact SHA-256"),
    ("execution", "execution identity"),
])
def test_production_merge_receipt_fails_closed(tmp_path, corruption, match):
    _, protocol_path, _, paths, hashes, receipt, receipt_path = (
        _production_fixture(tmp_path))
    if corruption == "schema":
        receipt["schema_version"] = 2
    elif corruption == "blind":
        receipt["outcome_blind"] = False
    elif corruption == "campaign":
        receipt["campaign_id"] = "different-campaign"
    elif corruption == "cell":
        receipt["campaign_cell"] = "ablation_n2"
    elif corruption == "seeds":
        receipt["expected_seed_list"].reverse()
    elif corruption == "hashes":
        receipt["expected_hashes"]["source_sha256"] = "a" * 64
    elif corruption == "ledger":
        receipt["ledger_sha256"] = None
    elif corruption == "path":
        receipt["selected"][0]["artifact_path"] = str(
            (tmp_path / "cherry-picked.json").resolve())
    elif corruption == "artifact_sha":
        receipt["selected"][0]["artifact_sha256"] = "a" * 64
    else:
        receipt["selected"][0]["attempt_id"] = "cherry-picked-attempt"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    with pytest.raises(merger.MergeValidationError, match=match):
        merger.merge_campaign_files(
            paths, protocol_path=protocol_path,
            selection_receipt_path=receipt_path,
            campaign_cell="primary", expected_seeds=(1, 2, 3, 4, 5),
            **{f"expected_{field}": value for field, value in hashes.items()})


def test_cli_requires_receipt_binds_selection_and_prints_no_endpoint(
        tmp_path, capsys):
    _, protocol_path, _, paths, hashes, _, receipt_path = (
        _production_fixture(tmp_path))
    output = tmp_path / "merged.json"
    argv = [*(str(path) for path in reversed(paths)), "--output", str(output),
            "--protocol", str(protocol_path), "--campaign-cell", "primary",
            "--selection-receipt", str(receipt_path),
            "--expected-seeds", "1,2,3,4,5"]
    for field, value in hashes.items():
        argv.extend(["--expected-" + field.replace("_", "-"), value])
    assert merger.main(argv) == 0
    stdout = capsys.readouterr().out.lower()
    assert "mean_success" not in stdout and "auc" not in stdout
    merged = json.loads(output.read_text())
    assert merged["config"]["seed_list"] == [1, 2, 3, 4, 5]
    selection = merged["merge"]["selection"]
    assert selection == {
        "schema_version": 1,
        "selection_receipt_sha256": merger.sha256_path(receipt_path),
        "ledger_sha256": "8" * 64,
        "rule": merger.SELECTION_RULE,
        "campaign_id": "campaign-1",
        "campaign_cell": "primary",
        "selected": [{
            "seed": seed,
            "attempt_id": "attempt-001",
            "artifact_sha256": merger.sha256_path(paths[seed - 1]),
            "execution": copy.deepcopy(
                json.loads(paths[seed - 1].read_text())["execution"]),
        } for seed in (1, 2, 3, 4, 5)],
    }

    without_receipt = [
        value for index, value in enumerate(argv)
        if value != "--selection-receipt"
        and (index == 0 or argv[index - 1] != "--selection-receipt")]
    with pytest.raises(SystemExit):
        merger.main(without_receipt)
