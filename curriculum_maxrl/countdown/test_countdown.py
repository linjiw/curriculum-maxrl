import hashlib
import json
import sys
from fractions import Fraction

import numpy as np
import torch

from curriculum_maxrl.countdown.analyze_e2c_endpoints import (
    load_endpoint_result,
    main as analyze_e2c_endpoints_main,
)
from curriculum_maxrl.countdown.analyze_e2c import (
    summarize_displacement,
    validate_seed,
)
from curriculum_maxrl.countdown.countdown_reward import (
    achieved_value,
    compute_score,
    evaluate_equation,
    rewrite_prompt_text,
)
from curriculum_maxrl.countdown.e2c_protocol import (
    FROZEN_SEEDS,
    require_frozen_scalar,
    require_frozen_seeds,
)
from curriculum_maxrl.countdown.preflight_e2c import (
    load_reservoir,
    validate_reservoir,
    validate_schedule_support,
)
from verl_integration.vendored.hindsight import (
    CountdownHindsight,
    DoseMatchedLiveReplay,
    ReplayReservoirCollector,
)


def test_e2c_protocol_rejects_seed_and_scalar_drift():
    assert require_frozen_seeds([1, 2, 3]) == FROZEN_SEEDS
    for seeds in ([1, 2], [1, 3, 2], [1, 2, 4], [1, 2, 3, 3]):
        try:
            require_frozen_seeds(seeds)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted non-preregistered seeds: {seeds}")
    require_frozen_scalar("steps", 60, 60)
    try:
        require_frozen_scalar("steps", 59, 60)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a non-preregistered step count")


def test_e2c_displaced_slot_threshold_preserves_fixed_slot_endpoints():
    report = summarize_displacement([
        {"seed": 1, "scheduled_groups": 10, "displaced_live_slots": 2},
        {"seed": 2, "scheduled_groups": 10, "displaced_live_slots": 3},
        {"seed": 3, "scheduled_groups": 10, "displaced_live_slots": 4},
    ])
    assert report["combined_fraction"] == 0.3
    assert report["within_frozen_threshold"] is False
    assert report["interpretation"] == "fixed_slot_direction_substitution_only"
    assert "does not expose or invalidate endpoints" in report["note"]


def test_e2c_delivery_recomputes_slots_and_token_accounting(tmp_path):
    schedule = tmp_path / "schedule.jsonl"
    schedule.write_text(json.dumps({
        "global_step": 1,
        "accepted_groups": [{
            "dataset_index": 200,
            "response_tokens": 6,
        }],
        "accepted_group_token_counts": [6],
        "hindsight/optimizer_rows_total": 128,
        "hindsight/optimizer_response_tokens_total": 18,
    }) + "\n", encoding="utf-8")
    row = {
        "global_step": 1,
        "replay/groups": 1,
        "replay/reservoir_sources_used": 1,
        "replay/buffer_sources_used": 0,
        "replay/fallback_slots": 0,
        "replay/optimizer_rows_total": 128,
        "replay/target_aux_response_tokens": 6,
        "replay/aux_response_tokens": 6,
        "replay/cumulative_target_aux_response_tokens": 6,
        "replay/cumulative_aux_response_tokens": 6,
        "replay/cumulative_token_delta": 0,
        "replay/cumulative_token_mismatch_fraction": 0.0,
        "replay/optimizer_response_tokens_total": 18,
        "replay/target_optimizer_response_tokens_total": 18,
        "replay/optimizer_response_token_delta": 0,
        "replay/cumulative_target_optimizer_response_tokens": 18,
        "replay/cumulative_optimizer_response_tokens": 18,
        "replay/cumulative_optimizer_token_delta": 0,
        "replay/cumulative_optimizer_response_token_mismatch_fraction": 0.0,
        "replay/displaced_live_slots": 0,
        "replay_groups": [{
            "target_response_tokens": 6,
            "replay_response_tokens": 6,
            "source_dataset_index": 100,
            "source_kind": "reservoir",
            "source_step": None,
            "source_age_steps": None,
            "replaced_dataset_index": 200,
            "replaced_group_status": "dead",
        }],
    }
    audit = tmp_path / "audit.jsonl"
    audit.write_text(json.dumps(row) + "\n", encoding="utf-8")
    report = validate_seed(
        1, audit, schedule, expected_steps=1, mismatch_limit=0.05,
        expected_rows=128)
    assert report["status"] == "pass"

    row["replay_groups"][0]["replaced_dataset_index"] = 201
    audit.write_text(json.dumps(row) + "\n", encoding="utf-8")
    try:
        validate_seed(
            1, audit, schedule, expected_steps=1, mismatch_limit=0.05,
            expected_rows=128)
    except ValueError as error:
        assert "exact B2 slots" in str(error)
    else:
        raise AssertionError("delivery validator accepted the wrong target slot")


def test_exact_verifier_and_relabel_value():
    response = "<think>work</think>\n<answer>((8 / 4) + 3)</answer>"
    gt = {"target": 5, "numbers": [8, 4, 3]}
    assert compute_score(solution_str=response, ground_truth=gt) == 1.0
    assert achieved_value(response, gt["numbers"]) == 5
    assert compute_score(solution_str=response, ground_truth={**gt, "target": 6}) == 0.0


def test_ast_rejects_code_and_wrong_multiset():
    assert evaluate_equation("__import__('os').system('true')") is None
    assert compute_score(solution_str="<answer>8 + 4</answer>", ground_truth={"target": 12, "numbers": [8, 4, 3]}) == 0.0


def test_prompt_rewrite_is_slot_only():
    prompt = "Example 12.5. Using the numbers [3, 4], create an equation that equals 12."
    assert rewrite_prompt_text(prompt, 12, 7) == "Example 12.5. Using the numbers [3, 4], create an equation that equals 7."
    assert rewrite_prompt_text(prompt, 5, 7) is None


def test_hindsight_never_rewrites_response_arithmetic():
    relabeler = CountdownHindsight.__new__(CountdownHindsight)
    response = (
        "<think>The intermediate is 12.5 and 4 * 3 = 12.</think>"
        "<answer>(4 * 3)</answer>"
    )
    assert relabeler._rewrite_response_text(response, 12, 9) == response


def test_dose_matched_replay_uses_b2_slot_and_informative_live_group(tmp_path):
    schedule = tmp_path / "b2.jsonl"
    schedule.write_text(json.dumps({
        "global_step": 1,
        "accepted_group_token_counts": [4],
        "accepted_groups": [{
            "dataset_index": 100,
            "response_tokens": 4,
        }],
        "hindsight/optimizer_rows_total": 8,
        "hindsight/optimizer_response_tokens_total": 26,
    }) + "\n")

    class FakeBatch:
        def __init__(self):
            # Four groups of two rows: dead, informative live, dead, and
            # all-success (which must not be a replay source under MaxRL).
            self.batch = {
                "responses": torch.arange(32).reshape(8, 4),
                "response_mask": torch.tensor([
                    [1, 1, 0, 0], [1, 1, 0, 0],
                    [1, 1, 1, 0], [1, 1, 1, 0],
                    [1, 1, 1, 1], [1, 1, 1, 1],
                    [1, 1, 1, 1], [1, 1, 1, 1],
                ]),
            }
            self.non_tensor_batch = {
                "uid": np.array(["a", "a", "b", "b",
                                 "c", "c", "d", "d"], dtype=object),
                "data_source": np.array(["countdown_tier1"] * 8,
                                        dtype=object),
                "index": np.array([100, 100, 200, 200,
                                   300, 300, 400, 400]),
            }

        def __len__(self):
            return len(self.batch["responses"])

    batch = FakeBatch()
    source_responses = batch.batch["responses"][2:4].clone()
    rewards = torch.zeros(8, 4)
    rewards[3, 2] = 1
    rewards[6, 3] = 1
    rewards[7, 3] = 1
    replay = DoseMatchedLiveReplay(
        str(schedule), seed=7,
        max_cumulative_token_mismatch_fraction=1.0)

    stats = replay.replay_batch(batch, rewards, global_step=1)

    assert stats["replay/groups"] == 1
    assert stats["replay/live_candidates"] == 1
    assert stats["replay/inactive_slots"] == 3
    assert stats["replay/saturated_slots"] == 1
    assert stats["replay/fallback_slots"] == 0
    assert stats["replay/displaced_live_slots"] == 0
    assert stats["replay/target_aux_response_tokens"] == 4
    assert stats["replay/aux_response_tokens"] == 6
    assert torch.equal(batch.batch["responses"][0:2], source_responses)
    assert rewards[0].sum() == 0 and rewards[1].sum() == 1
    assert list(batch.non_tensor_batch["uid"][0:2]) == ["a", "a"]
    assert list(batch.non_tensor_batch["index"][0:2]) == [200, 200]


def test_dose_matched_replay_uses_exact_b2_slot_even_if_now_live(tmp_path):
    schedule = tmp_path / "b2.jsonl"
    schedule.write_text(json.dumps({
        "global_step": 1,
        "accepted_group_token_counts": [6],
        "accepted_groups": [{
            "dataset_index": 200,
            "response_tokens": 6,
        }],
        "hindsight/optimizer_rows_total": 6,
        "hindsight/optimizer_response_tokens_total": 18,
    }) + "\n")

    class FakeBatch:
        def __init__(self):
            self.batch = {
                "responses": torch.arange(18).reshape(6, 3),
                "response_mask": torch.ones(6, 3, dtype=torch.long),
            }
            self.non_tensor_batch = {
                "uid": np.array(["a", "a", "b", "b", "c", "c"],
                                dtype=object),
                "data_source": np.array(["countdown_tier1"] * 6,
                                        dtype=object),
                "index": np.array([100, 100, 200, 200, 300, 300]),
            }

        def __len__(self):
            return len(self.batch["responses"])

    batch = FakeBatch()
    source_a = batch.batch["responses"][0:2].clone()
    source_c = batch.batch["responses"][4:6].clone()
    rewards = torch.zeros(6, 3)
    rewards[1, 2] = 1
    rewards[3, 2] = 1
    rewards[5, 2] = 1
    replay = DoseMatchedLiveReplay(
        str(schedule), seed=9,
        max_cumulative_token_mismatch_fraction=1.0)

    stats = replay.replay_batch(batch, rewards, global_step=1)

    assert stats["replay/groups"] == 1
    assert stats["replay/inactive_slots"] == 0
    assert stats["replay/displaced_live_slots"] == 1
    assert stats["replay/fallback_slots"] == 0
    assert (torch.equal(batch.batch["responses"][2:4], source_a) or
            torch.equal(batch.batch["responses"][2:4], source_c))
    assert list(batch.non_tensor_batch["uid"][2:4]) == ["b", "b"]
    assert set(batch.non_tensor_batch["index"][2:4]) in ({100}, {300})


def test_dose_matched_replay_buffer_survives_batch_with_no_live_group(tmp_path):
    schedule = tmp_path / "b2.jsonl"
    schedule.write_text("\n".join([
        json.dumps({
            "global_step": 1,
            "accepted_group_token_counts": [],
            "accepted_groups": [],
            "hindsight/optimizer_rows_total": 6,
            "hindsight/optimizer_response_tokens_total": 18,
        }),
        json.dumps({
            "global_step": 2,
            "accepted_group_token_counts": [6],
            "accepted_groups": [{
                "dataset_index": 200,
                "response_tokens": 6,
            }],
            "hindsight/optimizer_rows_total": 6,
            "hindsight/optimizer_response_tokens_total": 18,
        }),
    ]) + "\n")

    class FakeBatch:
        def __init__(self):
            self.batch = {
                "responses": torch.arange(18).reshape(6, 3),
                "response_mask": torch.ones(6, 3, dtype=torch.long),
            }
            self.non_tensor_batch = {
                "uid": np.array(["a", "a", "b", "b", "c", "c"],
                                dtype=object),
                "data_source": np.array(["countdown_tier1"] * 6,
                                        dtype=object),
                "index": np.array([100, 100, 200, 200, 300, 300]),
            }

        def __len__(self):
            return len(self.batch["responses"])

    replay = DoseMatchedLiveReplay(
        str(schedule), seed=4, buffer_capacity_groups=4,
        max_cumulative_token_mismatch_fraction=1.0)
    first_batch = FakeBatch()
    first_rewards = torch.zeros(6, 3)
    first_rewards[1, 2] = 1
    stats1 = replay.replay_batch(first_batch, first_rewards, global_step=1)
    assert stats1["replay/buffer_candidates"] == 0
    assert stats1["replay/buffer_groups_after_step"] == 1

    second_batch = FakeBatch()
    expected_source = first_batch.batch["responses"][0:2].clone()
    second_rewards = torch.zeros(6, 3)
    stats2 = replay.replay_batch(second_batch, second_rewards, global_step=2)

    assert stats2["replay/current_live_candidates"] == 0
    assert stats2["replay/buffer_sources_used"] == 1
    assert torch.equal(second_batch.batch["responses"][2:4], expected_source)
    assert second_rewards[3].sum() == 1


def test_e2c_reservoir_supplies_cold_start_without_current_live_group(tmp_path):
    reservoir_path = tmp_path / "reservoir.pt"

    class FakeBatch:
        def __init__(self):
            self.batch = {
                "responses": torch.arange(18).reshape(6, 3),
                "response_mask": torch.ones(6, 3, dtype=torch.long),
            }
            self.non_tensor_batch = {
                "uid": np.array(["a", "a", "b", "b", "c", "c"],
                                dtype=object),
                "data_source": np.array(["countdown_tier1"] * 6,
                                        dtype=object),
                "index": np.array([100, 100, 200, 200, 300, 300]),
                "reward_model": np.array([
                    {"ground_truth": {"target": 5, "numbers": [2, 3]}},
                    {"ground_truth": {"target": 5, "numbers": [2, 3]}},
                    {"ground_truth": {"target": 6, "numbers": [2, 3]}},
                    {"ground_truth": {"target": 6, "numbers": [2, 3]}},
                    {"ground_truth": {"target": 7, "numbers": [2, 3]}},
                    {"ground_truth": {"target": 7, "numbers": [2, 3]}},
                ], dtype=object),
            }

        def __len__(self):
            return len(self.batch["responses"])

    collection_batch = FakeBatch()
    collection_rewards = torch.zeros(6, 3)
    collection_rewards[1, 2] = 1
    collector = ReplayReservoirCollector(
        str(reservoir_path), seed=424242, max_groups=1,
        expected_group_size=2)
    collection_stats = collector.collect_batch(
        collection_batch, collection_rewards, global_step=1)
    assert collection_stats["reservoir/groups_retained"] == 1
    assert reservoir_path.is_file()
    assert (tmp_path / "reservoir.pt.manifest.json").is_file()

    schedule = tmp_path / "b2.jsonl"
    schedule.write_text(json.dumps({
        "global_step": 1,
        "accepted_group_token_counts": [6],
        "accepted_groups": [{
            "dataset_index": 200,
            "response_tokens": 6,
        }],
        "hindsight/optimizer_rows_total": 6,
        "hindsight/optimizer_response_tokens_total": 18,
    }) + "\n")
    digest = DoseMatchedLiveReplay._sha256(reservoir_path)
    replay = DoseMatchedLiveReplay(
        str(schedule), seed=1,
        max_cumulative_token_mismatch_fraction=0.05,
        reservoir_path=str(reservoir_path),
        reservoir_sha256=digest)
    replay_batch = FakeBatch()
    replay_rewards = torch.zeros(6, 3)
    stats = replay.replay_batch(replay_batch, replay_rewards, global_step=1)

    assert stats["replay/current_live_candidates"] == 0
    assert stats["replay/reservoir_candidates"] == 1
    assert stats["replay/reservoir_sources_used"] == 1
    assert stats["replay/buffer_sources_used"] == 0
    assert stats["replay/fallback_slots"] == 0
    assert stats["replay/cumulative_token_mismatch_fraction"] == 0
    assert list(replay_batch.non_tensor_batch["uid"][2:4]) == ["b", "b"]
    assert list(replay_batch.non_tensor_batch["index"][2:4]) == [100, 100]


def test_e2c_reservoir_checksum_is_enforced(tmp_path):
    reservoir = tmp_path / "reservoir.pt"
    torch.save({"format_version": 1, "groups": [{
        "status": "informative",
        "dataset_index": 1,
        "data_source": "countdown_tier1",
        "group_size": 2,
        "tokens": 2,
        "success_rollouts": 1,
        "payload": {
            "batch": {"response_mask": torch.ones(2, 1)},
            "reward": torch.tensor([[0.0], [1.0]]),
            "non_tensor": {},
        },
    }]}, reservoir)
    schedule = tmp_path / "b2.jsonl"
    schedule.write_text(json.dumps({
        "global_step": 1,
        "accepted_groups": [],
        "accepted_group_token_counts": [],
    }) + "\n")
    try:
        DoseMatchedLiveReplay(
            str(schedule), reservoir_path=str(reservoir),
            reservoir_sha256="0" * 64)
    except ValueError as error:
        assert "SHA-256" in str(error)
    else:
        raise AssertionError("reservoir checksum mismatch was accepted")


def test_e2c_preflight_validates_provenance_and_token_support(tmp_path):
    try:
        import pandas as pd
    except ImportError:
        # The recovered RTX-5090 runtime carries the compatible parquet stack;
        # some lightweight system-Python test environments intentionally do not.
        return
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    pd.DataFrame([{
        "extra_info": {"index": 100},
        "data_source": "countdown_tier1",
        "reward_model": {
            "ground_truth": {"target": 5, "numbers": [2, 3]}},
    }]).to_parquet(train_path, index=False)
    pd.DataFrame([{
        "extra_info": {"index": 10_000_000},
        "data_source": "countdown_tier1",
        "reward_model": {
            "ground_truth": {"target": 8, "numbers": [3, 5]}},
    }]).to_parquet(test_path, index=False)

    reservoir_path = tmp_path / "reservoir.pt"
    torch.save({"format_version": 1, "groups": [{
        "status": "informative",
        "dataset_index": 100,
        "data_source": "countdown_tier1",
        "group_size": 2,
        "tokens": 6,
        "success_rollouts": 1,
        "payload": {
            "batch": {"response_mask": torch.ones(2, 3)},
            "reward": torch.tensor([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0]]),
            "non_tensor": {"reward_model": [
                {"ground_truth": {"target": 5, "numbers": [2, 3]}},
                {"ground_truth": {"target": 5, "numbers": [2, 3]}},
            ], "index": [100, 100],
                "data_source": ["countdown_tier1", "countdown_tier1"]},
        },
    }]}, reservoir_path)
    artifact = load_reservoir(reservoir_path)
    report = validate_reservoir(
        artifact, train_path, test_path, minimum_groups=1,
        minimum_token_counts=1, expected_group_size=2)
    assert report["task_overlap_with_test"] == 0
    assert report["distinct_response_token_counts"] == 1

    artifact["groups"][0]["payload"]["non_tensor"][
        "hindsight_target"] = [5, 5]
    try:
        validate_reservoir(
            artifact, train_path, test_path, minimum_groups=1,
            minimum_token_counts=1, expected_group_size=2)
    except ValueError as error:
        assert "relabel metadata" in str(error)
    else:
        raise AssertionError("reservoir preflight accepted relabel metadata")
    del artifact["groups"][0]["payload"]["non_tensor"]["hindsight_target"]

    schedule = tmp_path / "b2.jsonl"
    schedule.write_text(json.dumps({
        "global_step": 1,
        "accepted_group_token_counts": [6],
        "accepted_groups": [{
            "dataset_index": 200,
            "response_tokens": 6,
        }],
        "hindsight/optimizer_rows_total": 128,
        "hindsight/optimizer_response_tokens_total": 18,
    }) + "\n")
    support = validate_schedule_support(
        schedule, reservoir_path, artifact["sha256"], seed=1,
        expected_steps=1, mismatch_limit=0.05)
    assert support["passed"] is True
    assert support["maximum_conditional_aux_token_mismatch_fraction"] == 0


def test_replay_matcher_token_dedup_preserves_reference_selection(tmp_path):
    schedule = tmp_path / "schedule.jsonl"
    schedule.write_text(json.dumps({
        "global_step": 1,
        "accepted_groups": [],
        "accepted_group_token_counts": [],
    }) + "\n")
    replay = DoseMatchedLiveReplay(str(schedule), seed=19)
    replay.cumulative_replay_tokens = 37
    replay.cumulative_optimizer_tokens = 101
    sources = [
        {"uid": f"r:{index}", "dataset_index": index,
         "source_kind": "reservoir", "tokens": tokens}
        for index, tokens in enumerate(
            [500, 500, 610, 610, 725, 810, 810, 940])
    ]
    targets = [
        {"uid": "t:0", "dataset_index": 2, "tokens": 600},
        {"uid": "t:1", "dataset_index": 99, "tokens": 700},
        {"uid": "t:2", "dataset_index": 5, "tokens": 800},
    ]

    def slow_reference():
        rng = np.random.default_rng(
            replay.seed * 1_000_003 + 7 * 101 + 2_100)
        order = list(range(len(sources)))
        rng.shuffle(order)
        states = {0: []}
        for target in targets:
            next_states = {}
            for total, selected in states.items():
                for source_index in order:
                    source = sources[source_index]
                    if source["uid"] == target["uid"]:
                        continue
                    if (source["source_kind"] == "reservoir" and
                            source["dataset_index"] ==
                            target["dataset_index"]):
                        continue
                    new_total = total + source["tokens"]
                    next_states.setdefault(
                        new_total, selected + [source_index])
            states = next_states
        target_current = sum(target["tokens"] for target in targets)

        def objective(total):
            aux_delta = replay.cumulative_replay_tokens + total - 2_100
            optimizer_step = 2_300 - target_current + total
            optimizer_delta = (
                replay.cumulative_optimizer_tokens + optimizer_step - 2_401)
            aux_fraction = abs(aux_delta) / 2_100
            optimizer_fraction = abs(optimizer_delta) / 2_401
            return (max(aux_fraction, optimizer_fraction),
                    aux_fraction + optimizer_fraction,
                    abs(aux_delta), abs(optimizer_delta), total)

        best = min(states, key=objective)
        return [sources[index] for index in states[best]]

    expected = slow_reference()
    actual = replay._select_sources(
        sources, targets,
        target_aux_tokens_after_step=2_100,
        target_optimizer_tokens_after_step=2_401,
        pre_optimizer_tokens=2_300,
        global_step=7,
    )
    assert [source["dataset_index"] for source in actual] == [
        source["dataset_index"] for source in expected]


def _write_e2c_endpoint_fixture(
        tmp_path, run_id="e2c_reservoir_replay_s2_260810",
        evaluation_seed=9002):
    model_path = (tmp_path / "checkpoints" /
                  run_id /
                  "global_step_60" / "actor" / "huggingface")
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    (model_path / "model.safetensors").write_bytes(b"fixture")
    model_files = {}
    for name in ("config.json", "model.safetensors"):
        path = model_path / name
        model_files[name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    result_dir = tmp_path / "results" / run_id
    result_dir.mkdir(parents=True)
    raw_path = result_dir / "eval.jsonl"
    raw_rows = []
    summaries = {}
    for tier_index in range(3):
        tier = f"countdown_tier{tier_index}"
        tier_rewards = []
        for task_index in range(128):
            rewards = [0] * 16
            if task_index % (tier_index + 2) == 0:
                rewards[task_index % 16] = 1
            tier_rewards.append(rewards)
            raw_rows.append({
                "data_source": tier,
                "ground_truth": {
                    "target": tier_index * 1_000 + task_index,
                    "numbers": [tier_index, task_index, 1],
                },
                "rewards": rewards,
            })
        flat = [value for rewards in tier_rewards for value in rewards]
        summaries[tier] = {
            "mean@16": sum(flat) / len(flat),
            "pass@16": sum(any(rewards) for rewards in tier_rewards) / 128,
        }
    raw_path.write_text(
        "".join(json.dumps(row) + "\n" for row in raw_rows),
        encoding="utf-8")
    result_path = result_dir / "eval.json"
    result_path.write_text(json.dumps({
        "raw_outcomes": str(raw_path),
        "model": str(model_path),
        "model_files": model_files,
        "evaluator_sha256":
            "0f642db64cabff66631b7e9ac88f1f3519651b21bee351051a1190f1a5bf653d",
        "reward_sha256":
            "99c04d4a4914170a528c67337aec364e7410074c552d9848c714f78c0f9e2312",
        "data_sha256":
            "95b1456fc3f49bc6f463614fef92900d748a07e4b429fd9383bbcf5edcb4e489",
        "seed": evaluation_seed,
        "k": 16,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_new_tokens": 128,
        "tiers": summaries,
    }), encoding="utf-8")
    return result_path


def test_e2c_endpoint_recomputes_standard_pass16_from_raw_outcomes(tmp_path):
    result_path = _write_e2c_endpoint_fixture(tmp_path)
    endpoint, provenance = load_endpoint_result(result_path)
    assert endpoint["pass@16"] == 43 / 128
    assert endpoint["mean@16"] == 43 / (128 * 16)
    assert "not VERL bootstrap" in provenance["metric_definition"]


def test_e2c_endpoint_rejects_summary_raw_mismatch(tmp_path):
    result_path = _write_e2c_endpoint_fixture(tmp_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["tiers"]["countdown_tier1"]["pass@16"] = 0.75
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_endpoint_result(result_path)
    except ValueError as error:
        assert "does not match raw outcomes" in str(error)
    else:
        raise AssertionError("summary/raw endpoint mismatch was accepted")


def test_e2c_endpoint_matrix_enforces_delivery_seed_and_model_pairing(
        tmp_path, monkeypatch):
    delivery_path = tmp_path / "delivery.json"
    delivery_seeds = [
        {"seed": seed, "status": "pass", "scheduled_groups": 1,
         "displaced_live_slots": 0}
        for seed in (1, 2, 3)]
    delivery_path.write_text(json.dumps({
        "status": "pass",
        "endpoint_evaluation_permitted": True,
        "seeds": delivery_seeds,
        "displaced_slot_diagnostic": summarize_displacement(delivery_seeds),
    }), encoding="utf-8")
    specifications = []
    comparator_receipts = []
    e2c_receipts = []
    for seed in (1, 2, 3):
        for arm in ("b1", "b2", "e2c"):
            run_id = (
                f"e2c_reservoir_replay_s{seed}_260810" if arm == "e2c"
                else f"e2_clean_{arm}_s{seed}_260809")
            result = _write_e2c_endpoint_fixture(
                tmp_path, run_id=run_id, evaluation_seed=10_000 + seed)
            specifications.extend(["--result", f"{seed},{arm},{result}"])
            summary = json.loads(result.read_text(encoding="utf-8"))
            receipt = {
                "run_id": run_id,
                "status": "complete",
                "checkpoint_fingerprint": summary["model_files"],
            }
            (e2c_receipts if arm == "e2c" else comparator_receipts).append(
                receipt)
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(json.dumps({
        "audit_kind": "outcome_blind_e2c_launch_readiness",
        "integrity_status": "pass",
        "delivery_gate": {"status": "pass"},
        "heldout_artifacts_inspected": False,
        "comparators": comparator_receipts,
        "e2c_runs": e2c_receipts,
    }), encoding="utf-8")
    output = tmp_path / "endpoints.json"
    monkeypatch.setattr(sys, "argv", [
        "analyze_e2c_endpoints", "--delivery", str(delivery_path),
        "--readiness", str(readiness_path),
        "--output", str(output), *specifications])

    analyze_e2c_endpoints_main()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "complete_descriptive_n3"
    assert len(report["seed_results"]) == 3
