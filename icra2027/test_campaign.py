"""CPU tests for the ICRA navigation campaign scaffold."""

from __future__ import annotations

import itertools
import json

import numpy as np

from frontier_rl import (LearnabilityTeacher, StagedDifficultyTeacher,
                         allocate_rollouts_greedy)
from frontier_rl.adapters.grid_reach import GridReachSpace
from icra2027.analyze_campaign import analyze, auc_at_budget, exact_sign_flip_p
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
    try:
        allocate_rollouts_greedy(p, 2, n_min=1, n_max=5)
    except ValueError:
        pass
    else:
        raise AssertionError("infeasible budget was not rejected")


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


def test_smoke_artifact_and_analysis_are_explicitly_non_evidentiary():
    artifact = run_campaign(
        seeds=1, steps=2, radius=3, n_rollouts=4,
        tasks_per_step=2, eval_every=1, eval_episodes=8)
    assert artifact["evidence_status"] == "engineering_smoke_not_paper_evidence"
    assert set(artifact["results"]) == {
        "ours_uN", "uniform", "learnability", "staged"}
    for rows in artifact["results"].values():
        assert rows[0]["final"]["episodes"] == 16
        assert len(rows[0]["final"]["eval"]["per_task_success"]) == 3
    report = analyze(artifact)
    assert not report["aug24_checkpoint"]["decision_ready"]
    ours = artifact["results"]["ours_uN"][0]
    full_budget = ours["history"][-1]["training_wall_seconds"]
    assert abs(auc_at_budget(ours, "training_wall_seconds", full_budget)
               - ours["target_uniform_auc_by_own_training_wall"]) < 1e-12
    assert exact_sign_flip_p(np.array([1.0, 1.0])) == 0.5
