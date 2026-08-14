"""CPU tests for the ICRA navigation scaffold and frozen analyzer."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from frontier_rl import (LearnabilityTeacher, StagedDifficultyTeacher,
                         allocate_rollouts_greedy)
from frontier_rl.adapters.grid_reach import GridReachSpace
from icra2027 import analyze_campaign as analyzer
from icra2027.freeze_pool_split import load_pool, make_manifest
from icra2027.navigation_campaign import run_campaign


def test_fixed_eval_is_repeatable_and_side_effect_free():
    env = GridReachSpace(radius=4, seed=3)
    before = (env.training_episodes, env.training_sim_steps,
              json.dumps(env.rng.bit_generator.state, sort_keys=True))
    first = env.eval_pass_rates(n=8, seed=99)
    second = env.eval_pass_rates(n=8, seed=99)
    after = (env.training_episodes, env.training_sim_steps,
             json.dumps(env.rng.bit_generator.state, sort_keys=True))
    assert np.array_equal(first, second)
    assert before == after


def test_learnability_and_staged_teachers():
    learnability = LearnabilityTeacher(3, n_rollouts=16, seed=0)
    p = np.array([0.05, 0.5, 0.95])
    assert np.argmax(learnability.utility(p)) == 1

    staged = StagedDifficultyTeacher(
        4, n_rollouts=4, initial_tasks=1, promotion_threshold=0.7,
        min_frontier_groups=2, floor=0.0, seed=0)
    assert staged.active_count == 1
    staged.observe(0, np.ones(4))
    assert staged.active_count == 1
    staged.observe(0, np.ones(4))
    assert staged.active_count == 2
    assert np.allclose(staged.distribution(), [0.5, 0.5, 0.0, 0.0])


def test_water_filling_is_feasible_and_optimal_on_small_case():
    p = np.array([0.1, 0.4, 0.8])
    got = allocate_rollouts_greedy(p, 8, n_min=1, n_max=5)
    assert got.sum() == 8 and got.min() >= 1 and got.max() <= 5

    def objective(counts):
        counts = np.asarray(counts)
        return float(np.sum(1.0 - (1.0 - p) ** counts))

    feasible = [counts for counts in itertools.product(range(1, 6), repeat=3)
                if sum(counts) == 8]
    assert objective(got) >= max(map(objective, feasible)) - 1e-12
    with pytest.raises(ValueError):
        allocate_rollouts_greedy(p, 2, n_min=1, n_max=5)


def test_pool_split_is_deterministic_and_spans_difficulty(tmp_path):
    pool = tmp_path / "pool.jsonl"
    rows = [
        {"env_id": f"course-{i:02d}", "difficulty": i / 19,
         "asset": f"courses/{i:02d}.world"}
        for i in range(20)]
    pool.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert len(load_pool(pool)) == 20
    first = make_manifest(pool, holdout_fraction=0.2, n_strata=5, seed=7)
    second = make_manifest(pool, holdout_fraction=0.2, n_strata=5, seed=7)
    assert first == second
    assert not set(first["train_ids"]) & set(first["heldout_ids"])
    heldout_difficulty = [first["records"][i]["difficulty"]
                          for i in first["heldout_ids"]]
    assert min(heldout_difficulty) < 0.25 and max(heldout_difficulty) > 0.75


def test_smoke_artifact_is_explicitly_non_evidentiary():
    artifact = run_campaign(
        seeds=1, steps=2, radius=3, n_rollouts=4,
        tasks_per_step=2, eval_every=1, eval_episodes=8)
    assert artifact["evidence_status"] == "engineering_smoke_not_paper_evidence"
    assert set(artifact["results"]) == set(analyzer.ARMS)
    for rows in artifact["results"].values():
        assert rows[0]["final"]["episodes"] == 16
        assert len(rows[0]["final"]["eval"]["per_task_success"]) == 3
    ours = artifact["results"]["ours_uN"][0]
    full_budget = ours["history"][-1]["training_wall_seconds"]
    assert abs(analyzer.auc_at_budget(
        ours, "training_wall_seconds", full_budget)
        - ours["target_uniform_auc_by_own_training_wall"]) < 1e-12
    assert analyzer.exact_sign_flip_p(np.array([1.0, 1.0])) == 0.5


def _frozen_protocol(tmp_path: Path) -> tuple[Path, dict, str]:
    protocol = json.loads(Path("icra2027/barn_protocol.json").read_text())
    protocol["status"] = "FROZEN"
    protocol["shared_training"].update({
        "training_sim_step_budget": 1000,
        "eval_sim_step_interval": 500,
        "max_training_updates": 2,
        "eval_every": 1,
    })
    protocol["analysis"]["bootstrap_draws"] = 200
    protocol["analysis"]["analyzer_sha256"] = hashlib.sha256(
        Path(analyzer.__file__).read_bytes()).hexdigest()
    path = tmp_path / "barn_protocol.json"
    payload = (json.dumps(protocol, indent=2) + "\n").encode()
    path.write_bytes(payload)
    return path, protocol, hashlib.sha256(payload).hexdigest()


def _evaluation(training_seed: int, success_count: int) -> dict:
    courses = []
    records = []
    for index in range(60):
        success = int(index < success_count)
        status = "success" if success else "timeout"
        courses.append({
            "env_id": f"barn-{index:03d}",
            "barn_index": index,
            "difficulty": 5.0 + index / 100.0,
            "stratum": index // 6,
            "seed": 10_000 + training_seed * 100 + index,
            "successes": success,
            "episodes": 1,
            "success_rate": float(success),
            "sim_steps": 1,
        })
        records.append({
            "env_id": f"barn-{index:03d}",
            "status": status,
            "success": success,
            "sim_steps": 1,
        })
    bins = []
    for bin_id in range(10):
        selected = courses[bin_id * 6:(bin_id + 1) * 6]
        bins.append({
            "bin": bin_id,
            "n_courses": 6,
            "difficulty_min": selected[0]["difficulty"],
            "difficulty_max": selected[-1]["difficulty"],
            "mean_success": float(np.mean([
                row["success_rate"] for row in selected])),
        })
    mean = success_count / 60.0
    return {
        "mean_success": mean,
        "n_tasks": 60,
        "eval_episodes": 60,
        "eval_sim_steps": 60,
        "success@1": mean,
        "easy_decile_retention": float(np.mean([
            row["success_rate"] for row in courses[:6]])),
        "teacher/calibration_n": 10,
        "teacher/calibration_bias": 0.1,
        "teacher/calibration_mae": 0.1,
        "per_task_success": [row["success_rate"] for row in courses],
        "heldout_course_ids": [row["env_id"] for row in courses],
        "per_course": courses,
        "episode_records": records,
        "success_by_difficulty_bin": [row["mean_success"] for row in bins],
        "difficulty_bins": bins,
        "status_counts": {
            key: count for key, count in {
                "success": success_count,
                "timeout": 60 - success_count,
            }.items() if count
        },
    }


def _run(
    arm: str,
    seed: int,
    sim_steps,
    wall_seconds,
    success_counts,
    *,
    n_rollouts: int = 8,
) -> dict:
    history = []
    for step, (steps, wall, success_count) in enumerate(zip(
            sim_steps, wall_seconds, success_counts)):
        groups = step * 2
        episodes = groups * n_rollouts
        sampled = {
            str(index): 1 for index in range(groups)
        }
        history.append({
            "step": step,
            "episodes": episodes,
            "sim_steps": steps,
            "training_wall_seconds": float(wall),
            "evaluation_wall_seconds": float(step),
            "dead_group_rate": 0.0,
            "all_fail_group_rate": 0.0,
            "all_pass_group_rate": 0.0,
            "live_groups": groups,
            "dead_groups": 0,
            "all_fail_groups": 0,
            "all_pass_groups": 0,
            "relabeled_groups": 0,
            "updates_live": groups,
            "updates_relabel": 0,
            "sampled_stratum_counts": sampled,
            "training_status_counts": ({"success": episodes}
                                       if episodes else {}),
            "training_course_counts": ({"barn-299": episodes}
                                       if episodes else {}),
            "eval": _evaluation(seed, success_count),
        })
    return {
        "arm": arm,
        "seed": seed,
        "teacher_seed": 20_000 + seed,
        "eval_seed": 30_000 + seed,
        "target_uniform_auc_by_episode": 0.0,
        "target_uniform_auc_by_sim_step": 0.0,
        "target_uniform_auc_by_own_training_wall": 0.0,
        "final": copy.deepcopy(history[-1]),
        "history": history,
        "training_episode_records": [
            {"episode_index": index}
            for index in range(history[-1]["episodes"])
        ],
    }


def _strict_artifact(tmp_path: Path) -> tuple[dict, Path]:
    protocol_path, protocol, protocol_sha = _frozen_protocol(tmp_path)
    results = {arm: [] for arm in analyzer.ARMS}
    for seed in analyzer.EXPECTED_SEEDS:
        results["ours_uN"].append(_run(
            "ours_uN", seed, [0, 500, 1000], [0, 900, 1000], [0, 60, 60]))
        for arm in ("uniform", "learnability"):
            results[arm].append(_run(
                arm, seed, [0, 900, 1000], [0, 100, 1000], [0, 60, 60]))
        results["staged"].append(_run(
            "staged", seed, [0, 500, 1000], [0, 100, 1000], [60, 60, 60]))

    shared = protocol["shared_training"]
    primary = protocol["primary"]
    dataset = protocol["dataset"]
    environment = protocol["environment"]
    analyzer_sha = hashlib.sha256(
        Path(analyzer.__file__).read_bytes()).hexdigest()
    artifact = {
        "schema_version": 1,
        "evidence_status": analyzer.FULL_EVIDENCE_STATUS,
        "domain": analyzer.DOMAIN,
        "heldout_protocol": "fixed paired held-out panel",
        "provenance": {
            "manifest_sha256": dataset["manifest_sha256"],
            "split_sha256": dataset["split_sha256"],
            "prereg_sha256": "a" * 64,
            "analyzer_sha256": analyzer_sha,
            "protocol_sha256": protocol_sha,
            "container_sha256": environment["container_sha256"],
            "source_sha256": "b" * 64,
            "split_bound_manifest_sha256": dataset["manifest_sha256"],
            "asset_hashes_verified": True,
        },
        "config": {
            "arms": list(analyzer.ARMS),
            "campaign_cell": "primary",
            "protocol_id": analyzer.PROTOCOL_ID,
            "seeds": 5,
            "seed_start": 1,
            "seed_list": list(analyzer.EXPECTED_SEEDS),
            "n_rollouts": primary["n_rollouts"],
            "tasks_per_step": shared["tasks_per_step"],
            "eval_episodes": shared["eval_episodes"],
            "training_sim_step_budget": shared["training_sim_step_budget"],
            "eval_sim_step_interval": shared["eval_sim_step_interval"],
            "max_training_updates": shared["max_training_updates"],
            "steps": shared["max_training_updates"],
            "eval_every": shared["eval_every"],
            "teacher_floor": shared["teacher_floor"],
            "teacher_decay": shared["teacher_decay"],
            "teacher_gamma": shared["teacher_gamma"],
            "staged_initial_strata": shared["staged_initial_strata"],
            "staged_promotion_threshold": shared["staged_promotion_threshold"],
            "staged_min_frontier_groups": shared["staged_min_frontier_groups"],
            "n_strata": dataset["n_strata"],
            "n_train_courses": dataset["n_train_courses"],
            "n_heldout_courses": dataset["n_heldout_courses"],
            "episode_timeout": environment["episode_timeout"],
            "max_step_size": environment["max_step_size"],
            "real_time_update_rate": environment["real_time_update_rate"],
            "teacher_unit": "frozen_difficulty_stratum",
            "evaluation_partition": "frozen_heldout",
            "smoke": False,
        },
        "results": results,
        "merge": {
            "schema_version": 1,
            "outcome_blind": True,
            "campaign_cell": "primary",
            "expected_seed_list": list(analyzer.EXPECTED_SEEDS),
            "input_artifact_count": 5,
            "per_seed_execution_order": [
                {
                    "seed": seed,
                    "execution_order": primary["execution_order_by_seed"][str(seed)],
                }
                for seed in analyzer.EXPECTED_SEEDS
            ],
            "selected_execution": [
                {
                    "seed": seed,
                    "campaign_id": "campaign-1",
                    "attempt_id": "attempt-001",
                    "submitted_utc": "2026-08-14T00:00:00+00:00",
                    "slurm_job_id": str(1000 + seed),
                    "slurm_array_job_id": "1000",
                    "slurm_array_task_id": seed,
                }
                for seed in analyzer.EXPECTED_SEEDS
            ],
            "per_seed_provenance_paths": [
                {
                    "seed": seed,
                    "manifest_path": f"/frozen/seed-{seed}/manifest.jsonl",
                    "split_path": f"/frozen/seed-{seed}/split.json",
                    "prereg_path": f"/frozen/seed-{seed}/prereg.md",
                    "analyzer_path": f"/frozen/seed-{seed}/analyzer.py",
                    "protocol_path": f"/frozen/seed-{seed}/protocol.json",
                    "dataset_root": "/frozen/dataset",
                    "robot_sdf": "/frozen/robot.sdf",
                }
                for seed in analyzer.EXPECTED_SEEDS
            ],
            "selection": {
                "schema_version": 1,
                "selection_receipt_sha256": "c" * 64,
                "ledger_sha256": "d" * 64,
                "rule": analyzer.SELECTION_RULE,
                "campaign_id": "campaign-1",
                "campaign_cell": "primary",
                "selected": [
                    {
                        "seed": seed,
                        "attempt_id": "attempt-001",
                        "artifact_sha256": f"{seed:x}" * 64,
                        "execution": {
                            "campaign_id": "campaign-1",
                            "attempt_id": "attempt-001",
                            "submitted_utc": "2026-08-14T00:00:00+00:00",
                            "slurm_job_id": str(1000 + seed),
                            "slurm_array_job_id": "1000",
                            "slurm_array_task_id": seed,
                        },
                    }
                    for seed in analyzer.EXPECTED_SEEDS
                ],
            },
        },
    }
    return artifact, protocol_path


def _ablation_artifact(primary: dict, protocol_path: Path, n_rollouts: int) -> dict:
    protocol = json.loads(protocol_path.read_text())
    artifact = copy.deepcopy(primary)
    artifact["evidence_status"] = analyzer.ABLATION_EVIDENCE_STATUS
    artifact["config"].update({
        "arms": ["ours_uN", "learnability"],
        "campaign_cell": f"ablation_n{n_rollouts}",
        "n_rollouts": n_rollouts,
    })
    artifact["merge"]["per_seed_execution_order"] = [
        {
            "seed": seed,
            "execution_order": protocol["ablation"]["execution_order_by_seed"][str(seed)],
        }
        for seed in analyzer.EXPECTED_SEEDS
    ]
    cell = f"ablation_n{n_rollouts}"
    artifact["merge"]["campaign_cell"] = cell
    artifact["merge"]["selection"]["campaign_cell"] = cell
    artifact["results"] = {"ours_uN": [], "learnability": []}
    for seed in analyzer.EXPECTED_SEEDS:
        artifact["results"]["ours_uN"].append(_run(
            "ours_uN", seed, [0, 500, 1000], [0, 700, 1000], [0, 60, 60],
            n_rollouts=n_rollouts,
        ))
        artifact["results"]["learnability"].append(_run(
            "learnability", seed, [0, 800, 1000], [0, 300, 1000], [0, 60, 60],
            n_rollouts=n_rollouts,
        ))
    return artifact


def test_strict_analysis_reports_primary_secondaries_and_named_gate(tmp_path):
    artifact, protocol_path = _strict_artifact(tmp_path)
    report = analyzer.analyze(artifact, protocol_path=protocol_path)
    gate = report["aug24_checkpoint"]
    assert gate["gate_comparators"] == ["uniform", "learnability"]
    assert gate["directional_bar_met"]
    assert gate["decision_ready"]
    assert gate["decision"] == "continue_icra"
    assert report["primary_sim_step_budget"] == 1000
    assert report["input_artifact_hash_basis"] == "canonical_json_in_memory"
    assert report["analyzer_sha256"] == artifact["provenance"]["analyzer_sha256"]
    contrast = report["paired_contrasts"]["ours_uN_minus_uniform"]
    assert contrast["primary_matched_sim_steps"]["mean_delta"] > 0
    assert contrast["descriptive_matched_wall"]["mean_delta"] < 0
    assert (report["paired_contrasts"]["ours_uN_minus_staged"]
            ["primary_matched_sim_steps"]["mean_delta"] < 0)

    ours = report["arms"]["ours_uN"]
    assert ours["primary_auc_at_frozen_sim_steps"]["n"] == 5
    assert len(ours["difficulty_deciles"]) == 10
    assert ours["easy_decile"]["final_retention"]["mean"] == 1.0
    assert ours["posterior_calibration"]["calibration_bias"]["n"] == 5
    accounting = ours["sampling_status_resource_update_accounting"]
    assert accounting["aggregate_sampled_stratum_counts"]["0"] == 5
    assert accounting["aggregate_training_status_counts"]["success"] == 160
    assert len(ours["final_course_outcomes"]) == 60


@pytest.mark.parametrize("mutation,match", [
    ("merge_marker", "outcome-blind merger marker"),
    ("seeds", "exact seeds"),
    ("execution_order", "per_seed_execution_order"),
    ("selection_missing", "merge fields must be exact"),
    ("selection_execution", "execution differs"),
    ("analyzer_sha", "analyzer_sha256"),
    ("secondary", "difficulty_bins"),
])
def test_strict_analysis_fails_closed(tmp_path, mutation, match):
    artifact, protocol_path = _strict_artifact(tmp_path)
    if mutation == "merge_marker":
        artifact["merge"]["outcome_blind"] = False
    elif mutation == "seeds":
        artifact["merge"]["expected_seed_list"] = [1, 2, 3, 4]
    elif mutation == "execution_order":
        artifact["merge"]["per_seed_execution_order"][0]["execution_order"].reverse()
    elif mutation == "selection_missing":
        del artifact["merge"]["selection"]
    elif mutation == "selection_execution":
        artifact["merge"]["selection"]["selected"][0]["execution"][
            "slurm_job_id"] = "9999"
    elif mutation == "analyzer_sha":
        artifact["provenance"]["analyzer_sha256"] = "f" * 64
    elif mutation == "secondary":
        del artifact["results"]["ours_uN"][0]["history"][0]["eval"]["difficulty_bins"]
    with pytest.raises(analyzer.AnalysisValidationError, match=match):
        analyzer.analyze(artifact, protocol_path=protocol_path)


def test_draft_protocol_is_rejected_before_endpoint_analysis(tmp_path):
    artifact, _ = _strict_artifact(tmp_path)
    draft = json.loads(Path("icra2027/barn_protocol.json").read_text())
    draft["status"] = "DRAFT"
    draft_path = tmp_path / "draft.json"
    draft_path.write_text(json.dumps(draft))
    with pytest.raises(analyzer.AnalysisValidationError, match="FROZEN"):
        analyzer.analyze(artifact, protocol_path=draft_path)


def test_analyze_file_records_exact_input_sha(tmp_path):
    artifact, protocol_path = _strict_artifact(tmp_path)
    artifact_path = tmp_path / "matrix.json"
    payload = (json.dumps(artifact, indent=1) + "\n").encode()
    artifact_path.write_bytes(payload)
    report = analyzer.analyze_file(artifact_path, protocol_path=protocol_path)
    assert report["input_artifact_sha256"] == hashlib.sha256(payload).hexdigest()
    assert report["input_artifact_hash_basis"] == "exact_input_file_bytes"


def test_n_ablation_reuses_primary_n8_and_accepts_only_fresh_2_4_16(tmp_path):
    primary, protocol_path = _strict_artifact(tmp_path)
    fresh = [
        _ablation_artifact(primary, protocol_path, n)
        for n in (16, 2, 4)
    ]
    report = analyzer.summarize_n_ablation(
        primary, fresh, protocol_path=protocol_path)
    assert report["n_values"] == [2, 4, 8, 16]
    assert report["fresh_n_values"] == [2, 4, 16]
    cells = {row["n_rollouts"]: row for row in report["cells"]}
    assert cells[8]["source"] == "primary_ours_uN_and_learnability"
    assert cells[8]["strict_merge_receipt"]["selection"][
        "selection_receipt_sha256"] == "c" * 64
    assert cells[2]["strict_merge_receipt"]["campaign_cell"] == "ablation_n2"
    assert cells[8]["input_artifact"]["sha256"] == analyzer._canonical_json_sha256(primary)
    assert cells[2]["ours_uN_minus_learnability"][
        "primary_matched_sim_steps"]["mean_delta"] > 0


def test_n_ablation_rejects_mixed_campaign_ledger(tmp_path):
    primary, protocol_path = _strict_artifact(tmp_path)
    fresh = [
        _ablation_artifact(primary, protocol_path, n)
        for n in (2, 4, 16)
    ]
    fresh[0]["merge"]["selection"]["ledger_sha256"] = "e" * 64
    with pytest.raises(analyzer.AnalysisValidationError,
                       match="selection ledger_sha256 differs from primary"):
        analyzer.summarize_n_ablation(
            primary, fresh, protocol_path=protocol_path)


def test_n_ablation_rejects_fresh_n8_and_incomplete_set(tmp_path):
    primary, protocol_path = _strict_artifact(tmp_path)
    fresh = [_ablation_artifact(primary, protocol_path, n) for n in (2, 4, 16)]
    duplicate_n8 = _ablation_artifact(primary, protocol_path, 8)
    with pytest.raises(analyzer.AnalysisValidationError, match="fresh N=8"):
        analyzer.summarize_n_ablation(
            primary, [*fresh, duplicate_n8], protocol_path=protocol_path)
    with pytest.raises(
            analyzer.AnalysisValidationError, match=r"exactly N=\{2,4,16\}"):
        analyzer.summarize_n_ablation(
            primary, fresh[:2], protocol_path=protocol_path)


def test_atomic_json_output_is_no_overwrite_and_rejects_nan(tmp_path):
    output = tmp_path / "analysis.json"
    analyzer._atomic_write_new_json(output, {"ok": 1})
    assert json.loads(output.read_text()) == {"ok": 1}
    with pytest.raises(FileExistsError):
        analyzer._atomic_write_new_json(output, {"ok": 2})
    assert json.loads(output.read_text()) == {"ok": 1}

    invalid = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="Out of range"):
        analyzer._atomic_write_new_json(invalid, {"bad": float("nan")})
    assert not invalid.exists()
