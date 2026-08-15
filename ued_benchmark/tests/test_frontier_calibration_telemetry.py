"""Adversarial, synthetic, CPU-only tests for the calibration DRAFT."""

from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "ued_benchmark/analysis"
sys.path.insert(0, str(ANALYSIS))

import frontier_calibration_telemetry as telemetry  # noqa: E402


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


def _analyzer_sha() -> str:
    return telemetry.sha256(Path(telemetry.__file__).resolve())


def _frontier_event(
    *,
    event_index: int,
    cycle: int,
    group_index: int,
    chain_label: str,
    sequence: int,
    source: str,
    successes: int,
    pre_successes: int,
    pre_trials: int,
    post_successes: int,
    post_trials: int,
    disposition: str,
    slot_pre: tuple[int, int] | None,
    slot_post: tuple[int, int] | None,
    accepted: bool,
    persisted: bool,
    N: int = 8,
    trials: int | None = None,
    snapshot_suffix: str = "",
    score_source_snapshot_id: str | None = None,
    level_sha256: str | None = None,
    runner_branch: str | None = None,
    pre_update_count: int = 0,
) -> dict[str, object]:
    actual_trials = N if trials is None else trials
    target = (
        telemetry.realized_activity(successes, N)
        if actual_trials == N
        else None
    )
    post_score = (
        telemetry.expected_activity(post_successes, post_trials, N)
        if persisted
        else None
    )
    if source == "replay" and score_source_snapshot_id is None:
        score_source_snapshot_id = _hash(
            f"snapshot:{chain_label}:{sequence - 1}:{cycle - 1}:"
        )
    branch = runner_branch or ("replay" if source == "replay" else "new")
    update_delta = 1 if branch == "replay" else 0
    return {
        "schema": 1,
        "protocol_id": telemetry.PROTOCOL_ID,
        "run_id": "calibration-s101-frontier",
        "event_index": event_index,
        "training_seed": 101,
        "arm": "frontier",
        "N": N,
        "student_index": 0,
        "outer_cycle": cycle,
        "within_cycle_group_index": group_index,
        "runner_branch": branch,
        "pre_upstream_n_iters": cycle,
        "post_upstream_n_iters": cycle + 1,
        "pre_upstream_n_updates": pre_update_count,
        "post_upstream_n_updates": pre_update_count + update_delta,
        "pre_upstream_n_grad_updates": pre_update_count,
        "post_upstream_n_grad_updates": pre_update_count + update_delta,
        "pre_optimizer_step_applications": pre_update_count * 5,
        "post_optimizer_step_applications": (pre_update_count + update_delta) * 5,
        "snapshot_id": _hash(
            f"snapshot:{chain_label}:{sequence}:{cycle}:{snapshot_suffix}"
        ),
        "level_chain_id": _hash(f"chain:{chain_label}"),
        "level_sha256": level_sha256 or _hash(f"level:{chain_label}"),
        "posterior_snapshot_sequence": sequence,
        "selection_source": source,
        "current_successes": successes,
        "current_trials": actual_trials,
        "realized_activity": target,
        "pre_successes": pre_successes,
        "pre_trials": pre_trials,
        "prior_alpha": 1.0,
        "prior_beta": 1.0,
        "pre_score": telemetry.expected_activity(pre_successes, pre_trials, N),
        "pre_score_semantics":
            "posterior_expected_activity_before_current_batch",
        "pre_score_source_snapshot_id": (
            score_source_snapshot_id if source == "replay" else None
        ),
        "post_score": post_score,
        "post_score_semantics": (
            "posterior_expected_activity_after_current_batch"
            if persisted
            else "not_persisted"
        ),
        "posterior_evidence_accepted": accepted,
        "posterior_persisted_after_snapshot": persisted,
        "post_successes": post_successes,
        "post_trials": post_trials,
        "slot_index_pre": None if slot_pre is None else slot_pre[0],
        "slot_generation_pre": None if slot_pre is None else slot_pre[1],
        "slot_index_post": None if slot_post is None else slot_post[0],
        "slot_generation_post": None if slot_post is None else slot_post[1],
        "disposition": disposition,
    }


def _maxmc_event(
    *,
    event_index: int,
    cycle: int,
    group_index: int,
    chain_label: str,
    sequence: int,
    source: str,
    successes: int,
    disposition: str,
    slot_pre: tuple[int, int] | None,
    slot_post: tuple[int, int] | None,
    pre_score: float | None,
    post_score: float | None,
    persisted: bool,
    N: int = 8,
    score_source_snapshot_id: str | None = None,
    runner_branch: str | None = None,
    pre_update_count: int = 0,
) -> dict[str, object]:
    if source == "replay" and score_source_snapshot_id is None:
        score_source_snapshot_id = _hash(
            f"maxmc-snapshot:{chain_label}:{sequence - 1}:{cycle - 1}"
        )
    branch = runner_branch or ("replay" if source == "replay" else "new")
    update_delta = 1 if branch == "replay" else 0
    return {
        "schema": 1,
        "protocol_id": telemetry.PROTOCOL_ID,
        "run_id": "calibration-s101-maxmc",
        "event_index": event_index,
        "training_seed": 101,
        "arm": "maxmc",
        "N": N,
        "student_index": 0,
        "outer_cycle": cycle,
        "within_cycle_group_index": group_index,
        "runner_branch": branch,
        "pre_upstream_n_iters": cycle,
        "post_upstream_n_iters": cycle + 1,
        "pre_upstream_n_updates": pre_update_count,
        "post_upstream_n_updates": pre_update_count + update_delta,
        "pre_upstream_n_grad_updates": pre_update_count,
        "post_upstream_n_grad_updates": pre_update_count + update_delta,
        "pre_optimizer_step_applications": pre_update_count * 5,
        "post_optimizer_step_applications": (pre_update_count + update_delta) * 5,
        "snapshot_id": _hash(f"maxmc-snapshot:{chain_label}:{sequence}:{cycle}"),
        "level_chain_id": _hash(f"maxmc-chain:{chain_label}"),
        "level_sha256": _hash(f"maxmc-level:{chain_label}"),
        "posterior_snapshot_sequence": sequence,
        "selection_source": source,
        "current_successes": successes,
        "current_trials": N,
        "realized_activity": telemetry.realized_activity(successes, N),
        "pre_successes": None,
        "pre_trials": None,
        "prior_alpha": None,
        "prior_beta": None,
        "pre_score": pre_score,
        "pre_score_semantics": (
            "stored_maxmc_before_current_group"
            if source == "replay"
            else "unavailable_new_candidate"
        ),
        "pre_score_source_snapshot_id": (
            score_source_snapshot_id if source == "replay" else None
        ),
        "post_score": post_score if persisted else None,
        "post_score_semantics": (
            "stored_maxmc_after_current_batch" if persisted else "not_persisted"
        ),
        "posterior_evidence_accepted": None,
        "posterior_persisted_after_snapshot": persisted,
        "post_successes": None,
        "post_trials": None,
        "slot_index_pre": None if slot_pre is None else slot_pre[0],
        "slot_generation_pre": None if slot_pre is None else slot_pre[1],
        "slot_index_post": None if slot_post is None else slot_post[0],
        "slot_generation_post": None if slot_post is None else slot_post[1],
        "disposition": disposition,
    }


def _frontier_ledger() -> list[dict[str, object]]:
    first_successes = [0, 1, 4, 8]
    second_successes = [1, 2, 3, 7]
    records: list[dict[str, object]] = []
    for index, successes in enumerate(first_successes):
        records.append(_frontier_event(
            event_index=index,
            cycle=0,
            group_index=index,
            chain_label=str(index),
            sequence=0,
            source="new",
            successes=successes,
            pre_successes=0,
            pre_trials=0,
            post_successes=successes,
            post_trials=8,
            disposition="inserted",
            slot_pre=None,
            slot_post=(index, 0),
            accepted=True,
            persisted=True,
        ))
    for index, successes in enumerate(second_successes):
        previous = first_successes[index]
        records.append(_frontier_event(
            event_index=4 + index,
            cycle=1,
            group_index=index,
            chain_label=str(index),
            sequence=1,
            source="replay",
            successes=successes,
            pre_successes=previous,
            pre_trials=8,
            post_successes=previous + successes,
            post_trials=16,
            disposition="updated",
            slot_pre=(index, 0),
            slot_post=(index, 0),
            accepted=True,
            persisted=True,
        ))
    return records


def _maxmc_ledger() -> list[dict[str, object]]:
    first_successes = [0, 1, 4, 8]
    second_successes = [0, 2, 5, 7]
    initial_scores = [0.1, 0.2, 0.3, 0.4]
    updated_scores = [0.15, 0.25, 0.35, 0.45]
    records: list[dict[str, object]] = []
    for index, (successes, post_score) in enumerate(
        zip(first_successes, initial_scores)
    ):
        records.append(_maxmc_event(
            event_index=index,
            cycle=0,
            group_index=index,
            chain_label=str(index),
            sequence=0,
            source="new",
            successes=successes,
            disposition="inserted",
            slot_pre=None,
            slot_post=(index, 0),
            pre_score=None,
            post_score=post_score,
            persisted=True,
        ))
    for index, (successes, pre_score, post_score) in enumerate(
        zip(second_successes, initial_scores, updated_scores)
    ):
        records.append(_maxmc_event(
            event_index=4 + index,
            cycle=1,
            group_index=index,
            chain_label=str(index),
            sequence=1,
            source="replay",
            successes=successes,
            disposition="updated",
            slot_pre=(index, 0),
            slot_post=(index, 0),
            pre_score=pre_score,
            post_score=post_score,
            persisted=True,
        ))
    return records


def _maxmc_ledger_two_updates() -> list[dict[str, object]]:
    """Three cycles: all-new warm-up followed by two replay updates."""
    records = _maxmc_ledger()
    for index, (successes, pre_score, post_score) in enumerate(zip(
        (1, 3, 4, 6),
        (0.15, 0.25, 0.35, 0.45),
        (0.2, 0.3, 0.4, 0.5),
    )):
        records.append(_maxmc_event(
            event_index=8 + index,
            cycle=2,
            group_index=index,
            chain_label=str(index),
            sequence=2,
            source="replay",
            successes=successes,
            disposition="updated",
            slot_pre=(index, 0),
            slot_post=(index, 0),
            pre_score=pre_score,
            post_score=post_score,
            persisted=True,
            pre_update_count=1,
        ))
    return records


def _frontier_ledger_with_extra_exploration_cycle() -> list[dict[str, object]]:
    """Add a purchased new-level cycle that robust PLR does not train on."""
    base = _frontier_ledger()
    records = base[:4]
    for group_index, successes in enumerate((1, 2, 5, 7)):
        records.append(_frontier_event(
            event_index=4 + group_index,
            cycle=1,
            group_index=group_index,
            chain_label=f"extra-{group_index}",
            sequence=0,
            source="new",
            successes=successes,
            pre_successes=0,
            pre_trials=0,
            post_successes=0,
            post_trials=0,
            disposition="valid_not_persisted",
            slot_pre=None,
            slot_post=None,
            accepted=False,
            persisted=False,
        ))
    for group_index, replay in enumerate(base[4:]):
        shifted = copy.deepcopy(replay)
        shifted["event_index"] = 8 + group_index
        shifted["outer_cycle"] = 2
        shifted["snapshot_id"] = _hash(
            f"snapshot:{group_index}:1:2:"
        )
        shifted["pre_upstream_n_iters"] = 2
        shifted["post_upstream_n_iters"] = 3
        records.append(shifted)
    return records


def _artifact(path: Path, contents: str) -> dict[str, str]:
    path.write_text(contents, encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": telemetry.sha256(path)}


def _write_campaign(
    root: Path,
    *,
    budget_overrides: dict[str, int] | None = None,
) -> tuple[Path, str, dict[str, object]]:
    assets = root / "campaign-assets"
    assets.mkdir()
    common: dict[str, object] = {}
    for name in sorted(telemetry.COMMON_ARTIFACT_KEYS):
        common[name] = _artifact(assets / name, f"frozen {name}\n")
    frontier_config = _artifact(assets / "frontier-config", "frontier config\n")
    maxmc_config = _artifact(assets / "maxmc-config", "maxmc config\n")
    budget = {
        "N": 8,
        "n_eval": 8,
        "n_parallel": 4,
        "n_rollout_steps": 256,
        "target_student_ppo_updates": 1,
        "max_outer_cycles": 4,
        "ppo_epochs": 5,
        "ppo_minibatches": 1,
    }
    budget.update(budget_overrides or {})
    campaign: dict[str, object] = {
        "schema": 1,
        "campaign_id": "calibration-campaign-s101",
        "protocol_id": telemetry.PROTOCOL_ID,
        "purpose": telemetry.PURPOSE,
        "status": "DRAFT_FROZEN_OUTCOME_BLIND_NOT_RUN_AUTHORIZATION",
        "frozen_before_endpoint_access": True,
        "production_authorized": False,
        "endpoint_access_authorized": False,
        "paper_evidence": False,
        "protocol": {
            "path": str(telemetry.PROTOCOL_PATH.resolve()),
            "sha256": telemetry.PROTOCOL_SHA256,
        },
        "analyzer": {
            "path": str(Path(telemetry.__file__).resolve()),
            "sha256": _analyzer_sha(),
        },
        "common_artifacts": common,
        "runner_semantics": {
            "use_robust_plr": True,
            "use_mutations": False,
            "cycle_source_policy": "uniform_new_or_replay_cycle",
        },
        "arms": {
            "frontier": {
                "run_id": "calibration-s101-frontier",
                "training_seed": 101,
                "config": frontier_config,
            },
            "maxmc": {
                "run_id": "calibration-s101-maxmc",
                "training_seed": 101,
                "config": maxmc_config,
            },
        },
        "budget": budget,
    }
    path = root / "expected-campaign-contract.json"
    path.write_text(_canonical(campaign), encoding="utf-8")
    return path.resolve(), telemetry.sha256(path), campaign


def _receipt(
    records: list[dict[str, object]],
    campaign: dict[str, object],
    campaign_sha256: str,
    *,
    counter_overrides: dict[str, int] | None = None,
    student_updates: int | None = None,
) -> dict[str, object]:
    first = records[0]
    cycles = sorted({int(record["outer_cycle"]) for record in records})
    repeated = sum(
        max(0, count - 1)
        for count in Counter(
            (record["outer_cycle"], record["level_sha256"])
            for record in records
        ).values()
    )
    new_counts = Counter(
        (record["outer_cycle"], record["level_sha256"])
        for record in records
        if record["selection_source"] in {"new", "mutation"}
    )
    counters = {
        "duplicate_event_id_count": 0,
        "duplicate_new_group_count": sum(
            max(0, count - 1) for count in new_counts.values()
        ),
        "partial_group_count": sum(
            record["current_trials"] != record["N"] for record in records
        ),
        "nonfinite_record_count": 0,
        "repeated_level_same_batch_count": repeated,
    }
    counters.update(counter_overrides or {})
    budget = campaign["budget"]
    terminal_event = records[-1]
    realized_updates = (
        int(terminal_event["post_upstream_n_updates"])
        if student_updates is None
        else student_updates
    )
    realized_cycles = int(terminal_event["post_upstream_n_iters"])
    common = campaign["common_artifacts"]
    arm_spec = campaign["arms"][first["arm"]]
    return {
        "schema": 1,
        "protocol_id": telemetry.PROTOCOL_ID,
        "purpose": telemetry.PURPOSE,
        "status": "complete",
        "campaign_id": campaign["campaign_id"],
        "campaign_contract_sha256": campaign_sha256,
        "run_id": first["run_id"],
        "arm": first["arm"],
        "training_seed": first["training_seed"],
        "N": budget["N"],
        "n_eval": budget["n_eval"],
        "n_parallel": budget["n_parallel"],
        "n_rollout_steps": budget["n_rollout_steps"],
        "upstream_n_iters": realized_cycles,
        "student_ppo_updates": realized_updates,
        "upstream_n_updates": realized_updates,
        "upstream_n_grad_updates": realized_updates,
        "ppo_epochs": budget["ppo_epochs"],
        "ppo_minibatches": budget["ppo_minibatches"],
        "optimizer_step_applications": (
            realized_updates * budget["ppo_epochs"] * budget["ppo_minibatches"]
        ),
        "student_training_transition_count": (
            realized_cycles
            * budget["n_parallel"]
            * budget["n_eval"]
            * budget["n_rollout_steps"]
        ),
        "telemetry_records": len(records),
        "attempted_group_count": len(records),
        "complete_group_count": sum(
            record["current_trials"] == record["N"] for record in records
        ),
        "outer_cycle_count": realized_cycles,
        "terminal_outer_cycle": cycles[-1],
        "from_last_checkpoint": False,
        "closed_before_analysis": True,
        "endpoint_class": "calibration_telemetry_engineering_draft",
        "production_authorized": False,
        "endpoint_accessed": False,
        "paper_evidence": False,
        "provenance": {
            "base_commit": telemetry.BASE_COMMIT,
            "base_tree": telemetry.BASE_TREE,
            "v4_contract_sha256": telemetry.V4_CONTRACT_SHA256,
            "protocol_sha256": telemetry.PROTOCOL_SHA256,
            "analyzer_sha256": _analyzer_sha(),
            "campaign_contract_sha256": campaign_sha256,
            "config_sha256": arm_spec["config"]["sha256"],
            **{
                f"{name}_sha256": common[name]["sha256"]
                for name in telemetry.COMMON_ARTIFACT_KEYS
            },
        },
        "integrity_counters": counters,
    }


def _write_package(
    root: Path,
    records: list[dict[str, object]],
    campaign: dict[str, object],
    campaign_sha256: str,
    receipt: dict[str, object] | None = None,
) -> str:
    receipt_value = (
        _receipt(records, campaign, campaign_sha256)
        if receipt is None
        else receipt
    )
    (root / "telemetry-events.jsonl").write_text(
        "".join(_canonical(record) for record in records), encoding="utf-8"
    )
    (root / "telemetry-receipt.json").write_text(
        _canonical(receipt_value), encoding="utf-8"
    )
    manifest = "".join(
        f"{telemetry.sha256(root / name)}  {name}\n"
        for name in sorted(telemetry.PACKAGE_PAYLOADS)
    )
    manifest_path = root / "telemetry-SHA256SUMS"
    manifest_path.write_text(manifest, encoding="utf-8")
    digest = telemetry.sha256(manifest_path)
    complete = {
        "schema": 1,
        "status": "complete",
        "run_id": receipt_value["run_id"],
        "arm": receipt_value["arm"],
        "sha256sums_sha256": digest,
        "file_count": len(telemetry.PACKAGE_PAYLOADS),
    }
    (root / "telemetry-COMPLETE").write_text(
        _canonical(complete), encoding="utf-8"
    )
    return digest


def _validate_closed(
    package: Path,
    package_sha256: str,
    campaign_path: Path,
    campaign_sha256: str,
) -> dict[str, object]:
    return telemetry.validate_package(
        package,
        package_sha256,
        campaign_path,
        campaign_sha256,
        _analyzer_sha(),
    )


class CalibrationMathTest(unittest.TestCase):
    def test_conditional_identity_matches_exact_binomial_enumeration(self) -> None:
        for N in (2, 4, 8):
            for p in (0.0, 0.01, 0.2, 0.5, 0.9, 1.0):
                with self.subTest(N=N, p=p):
                    self.assertAlmostEqual(
                        telemetry.enumerated_conditional_expectation(p, N),
                        telemetry.activity_at_probability(p, N),
                        places=14,
                    )

    def test_posterior_predictive_matches_beta_binomial_enumeration(self) -> None:
        for N in (2, 4, 8):
            for successes, trials in ((0, 0), (0, 3), (2, 3), (7, 10)):
                a = successes + 1.0
                b = trials - successes + 1.0
                expected = 0.0
                for K in range(N + 1):
                    log_probability = (
                        math.lgamma(N + 1)
                        - math.lgamma(K + 1)
                        - math.lgamma(N - K + 1)
                        + math.lgamma(a + K)
                        + math.lgamma(b + N - K)
                        - math.lgamma(a + b + N)
                        - math.lgamma(a)
                        - math.lgamma(b)
                        + math.lgamma(a + b)
                    )
                    expected += (
                        math.exp(log_probability)
                        * telemetry.realized_activity(K, N)
                    )
                self.assertAlmostEqual(
                    telemetry.expected_activity(successes, trials, N),
                    expected,
                    places=13,
                )

    def test_realized_activity_endpoints_and_interior(self) -> None:
        self.assertEqual(telemetry.realized_activity(0, 8), 0.0)
        self.assertEqual(telemetry.realized_activity(8, 8), 0.0)
        self.assertEqual(telemetry.realized_activity(1, 8), 0.875)
        self.assertEqual(telemetry.realized_activity(4, 8), 0.5)


class CalibrationEventContractTest(unittest.TestCase):
    def test_preflight_freezes_permissions_bins_prior_and_full_lineage(self) -> None:
        result = telemetry.repository_preflight(_analyzer_sha())
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["protected_artifact_count"], 25)
        self.assertFalse(result["production_authorized"])
        self.assertFalse(result["endpoint_accessed"])
        self.assertFalse(result["paper_evidence"])
        protocol = telemetry.load_json(telemetry.PROTOCOL_PATH, "test protocol")
        self.assertFalse(protocol["production_driver_authorized"])
        self.assertFalse(protocol["endpoint_access_authorized"])
        self.assertFalse(protocol["paper_evidence"])
        self.assertIn(
            "adaptively purchased group distribution",
            protocol["estimand"]["population_scope"],
        )
        self.assertEqual(
            protocol["estimand"]["predeclared_prior"],
            {"alpha": 1.0, "beta": 1.0},
        )
        self.assertEqual(
            tuple(protocol["calibration_analysis"]["frozen_bin_edges"]),
            telemetry.BIN_EDGES,
        )

    def test_preflight_rejects_external_analyzer_and_lineage_drift(self) -> None:
        with self.assertRaisesRegex(telemetry.TelemetryError, "external.*analyzer"):
            telemetry.repository_preflight(_hash("wrong analyzer"))
        path = next(iter(telemetry.PROTECTED_HASHES))
        with mock.patch.dict(
            telemetry.PROTECTED_HASHES, {path: _hash("wrong protected byte")}
        ):
            with self.assertRaisesRegex(telemetry.TelemetryError, "protected artifact"):
                telemetry.repository_preflight(_analyzer_sha())

    def test_frontier_summary_uses_adaptively_purchased_complete_groups(self) -> None:
        records = _frontier_ledger()
        records.append(_frontier_event(
            event_index=8,
            cycle=2,
            group_index=0,
            chain_label="candidate",
            sequence=0,
            source="new",
            successes=2,
            pre_successes=0,
            pre_trials=0,
            post_successes=0,
            post_trials=0,
            disposition="valid_not_persisted",
            slot_pre=None,
            slot_post=None,
            accepted=False,
            persisted=False,
            pre_update_count=1,
        ))
        result = telemetry.analyze_events(records)
        summary = result["calibration"][
            "all_complete_adaptively_purchased_groups"
        ]
        self.assertEqual(summary["count"], 9)
        self.assertEqual(result["delivery"]["count"], 9)
        self.assertEqual(sum(row["count"] for row in summary["bins"]), 9)
        self.assertEqual(result["independent_unit"], "training_seed")
        self.assertFalse(result["group_level_inference_authorized"])

    def test_new_begins_with_predeclared_prior_and_mutation_is_forbidden(self) -> None:
        record = _frontier_event(
            event_index=0,
            cycle=0,
            group_index=0,
            chain_label="new-prior",
            sequence=0,
            source="new",
            successes=3,
            pre_successes=0,
            pre_trials=0,
            post_successes=3,
            post_trials=8,
            disposition="inserted",
            slot_pre=None,
            slot_post=(2, 0),
            accepted=True,
            persisted=True,
        )
        telemetry.validate_events([record])
        leaked = copy.deepcopy(record)
        leaked["pre_successes"] = 1
        leaked["pre_trials"] = 1
        leaked["pre_score"] = telemetry.expected_activity(1, 1, 8)
        leaked["post_successes"] = 4
        leaked["post_trials"] = 9
        leaked["post_score"] = telemetry.expected_activity(4, 9, 8)
        with self.assertRaisesRegex(telemetry.TelemetryError, "predeclared prior"):
            telemetry.validate_events([leaked])

        mutation = copy.deepcopy(record)
        mutation["selection_source"] = "mutation"
        with self.assertRaisesRegex(
            telemetry.TelemetryError, "mutation telemetry is forbidden"
        ):
            telemetry.validate_events([mutation])

    def test_cycle_clock_fields_are_sibling_invariant(self) -> None:
        records = _frontier_ledger()
        records[5]["post_upstream_n_iters"] = 3
        with self.assertRaisesRegex(
            telemetry.TelemetryError, "within-cycle counter/branch drift"
        ):
            telemetry.validate_events(records)

    def test_cycle_clock_rejects_nonunit_iter_update_and_optimizer_deltas(self) -> None:
        cases = (
            (
                "n_iters",
                lambda record: record.__setitem__("post_upstream_n_iters", 3),
                "n_iters did not advance exactly once",
            ),
            (
                "update_delta",
                lambda record: (
                    record.__setitem__("post_upstream_n_updates", 2),
                    record.__setitem__("post_upstream_n_grad_updates", 2),
                    record.__setitem__("post_optimizer_step_applications", 10),
                ),
                "update delta is not zero or one",
            ),
            (
                "optimizer_delta",
                lambda record: record.__setitem__(
                    "post_optimizer_step_applications", 4
                ),
                "optimizer cumulative counter",
            ),
        )
        for label, mutation, message in cases:
            with self.subTest(label=label):
                records = _frontier_ledger()
                for record in records[4:]:
                    mutation(record)
                with self.assertRaisesRegex(telemetry.TelemetryError, message):
                    telemetry.validate_events(records)

    def test_all_new_cycle_cannot_claim_a_student_update(self) -> None:
        records = _frontier_ledger()
        for record in records[:4]:
            record["post_upstream_n_updates"] = 1
            record["post_upstream_n_grad_updates"] = 1
            record["post_optimizer_step_applications"] = 5
        with self.assertRaisesRegex(
            telemetry.TelemetryError, "all-new runner branch falsely claims"
        ):
            telemetry.validate_events(records)

    def test_cycle_source_must_be_uniform_and_match_actual_runner_branch(self) -> None:
        records = _frontier_ledger()
        mixed = records[1]
        mixed["selection_source"] = "replay"
        mixed["pre_score_source_snapshot_id"] = _hash("orphan prior snapshot")
        mixed["slot_index_pre"] = mixed["slot_index_post"]
        mixed["slot_generation_pre"] = mixed["slot_generation_post"]
        mixed["disposition"] = "updated"
        with self.assertRaisesRegex(
            telemetry.TelemetryError, "cycle-uniform all-new"
        ):
            telemetry.validate_events(records)

    def test_cycle_counter_ledger_must_be_continuous(self) -> None:
        records = _frontier_ledger()
        for record in records[4:]:
            record["pre_upstream_n_iters"] = 0
            record["post_upstream_n_iters"] = 1
        with self.assertRaisesRegex(
            telemetry.TelemetryError, "counter ledger is discontinuous"
        ):
            telemetry.validate_events(records)

    def test_recorded_target_and_nonfinite_json_are_rejected(self) -> None:
        records = _frontier_ledger()
        records[0]["realized_activity"] = 0.5
        with self.assertRaisesRegex(telemetry.TelemetryError, "realized activity"):
            telemetry.validate_events(records)
        with self.assertRaisesRegex(telemetry.TelemetryError, "nonfinite"):
            telemetry.parse_json('{"pre_score":NaN}', "synthetic nonfinite")

    def test_malformed_and_overflow_event_values_fail_as_telemetry(self) -> None:
        self.assertFalse(telemetry._is_finite_number(10 ** 10000))
        corruptions = (
            ("unhashable_N", "N", []),
            ("counter_overflow", "pre_trials", telemetry.MAX_COUNTER + 1),
            ("numeric_overflow", "pre_score", 10 ** 10000),
        )
        for label, field, value in corruptions:
            with self.subTest(label=label):
                records = _frontier_ledger()
                records[0][field] = value
                with self.assertRaises(telemetry.TelemetryError):
                    telemetry.validate_events(records)

    def test_frozen_bin_edges_use_left_closed_membership(self) -> None:
        self.assertEqual(telemetry._bin_index(0.0), 0)
        self.assertEqual(telemetry._bin_index(0.049999), 0)
        self.assertEqual(telemetry._bin_index(0.05), 1)
        self.assertEqual(telemetry._bin_index(1.0), 19)

    def test_current_group_posterior_leakage_fails_both_gates(self) -> None:
        records = _frontier_ledger()
        leaked_score = copy.deepcopy(records)
        event = leaked_score[4]
        event["pre_score"] = telemetry.expected_activity(
            event["pre_successes"] + event["current_successes"],
            event["pre_trials"] + event["current_trials"],
            event["N"],
        )
        with self.assertRaisesRegex(telemetry.TelemetryError, "current or drifted"):
            telemetry.validate_events(leaked_score)

        leaked_counts = copy.deepcopy(records)
        event = leaked_counts[4]
        event["pre_successes"] += event["current_successes"]
        event["pre_trials"] += event["current_trials"]
        event["pre_score"] = telemetry.expected_activity(
            event["pre_successes"], event["pre_trials"], event["N"]
        )
        event["post_successes"] = event["pre_successes"] + event["current_successes"]
        event["post_trials"] = event["pre_trials"] + event["current_trials"]
        event["post_score"] = telemetry.expected_activity(
            event["post_successes"], event["post_trials"], event["N"]
        )
        with self.assertRaisesRegex(
            telemetry.TelemetryError, "prior stored post-score|continuity"
        ):
            telemetry.validate_events(leaked_counts)

    def _sibling_ledger(self) -> list[dict[str, object]]:
        first = _frontier_event(
            event_index=0,
            cycle=0,
            group_index=0,
            chain_label="shared",
            sequence=0,
            source="new",
            successes=2,
            pre_successes=0,
            pre_trials=0,
            post_successes=2,
            post_trials=8,
            disposition="inserted",
            slot_pre=None,
            slot_post=(0, 0),
            accepted=True,
            persisted=True,
        )
        common = dict(
            cycle=1,
            chain_label="shared",
            sequence=1,
            source="replay",
            pre_successes=2,
            pre_trials=8,
            post_successes=5,
            post_trials=24,
            disposition="updated",
            slot_pre=(0, 0),
            slot_post=(0, 0),
            accepted=True,
            persisted=True,
            snapshot_suffix="same",
        )
        return [
            first,
            _frontier_event(
                event_index=1, group_index=0, successes=1, **common
            ),
            _frontier_event(
                event_index=2, group_index=1, successes=2, **common
            ),
        ]

    def test_siblings_share_one_prebatch_snapshot_and_aggregate_afterward(self) -> None:
        records = self._sibling_ledger()
        telemetry.validate_events(records)
        drifted = copy.deepcopy(records)
        drifted[2]["pre_successes"] = 3
        drifted[2]["pre_trials"] = 16
        drifted[2]["pre_score"] = telemetry.expected_activity(3, 16, 8)
        with self.assertRaisesRegex(telemetry.TelemetryError, "concurrent snapshot"):
            telemetry.validate_events(drifted)

    def test_two_snapshots_for_same_chain_and_cycle_are_rejected(self) -> None:
        records = self._sibling_ledger()
        records[2]["snapshot_id"] = _hash("second snapshot in same batch")
        records[2]["posterior_snapshot_sequence"] = 2
        with self.assertRaisesRegex(telemetry.TelemetryError, "concurrent snapshot"):
            telemetry.validate_events(records)

    def test_snapshot_sequence_must_advance_strictly_across_cycles(self) -> None:
        records = _frontier_ledger()
        records[4]["posterior_snapshot_sequence"] = 2
        with self.assertRaisesRegex(telemetry.TelemetryError, "snapshot sequence"):
            telemetry.validate_events(records)

    def _eviction_ledger(self) -> list[dict[str, object]]:
        first_a = _frontier_event(
            event_index=0, cycle=0, group_index=0,
            chain_label="evicted-a", sequence=0, source="new", successes=2,
            pre_successes=0, pre_trials=0, post_successes=2, post_trials=8,
            disposition="inserted", slot_pre=None, slot_post=(0, 0),
            accepted=True, persisted=True,
        )
        final_a = _frontier_event(
            event_index=1, cycle=1, group_index=0,
            chain_label="evicted-a", sequence=1, source="replay", successes=1,
            pre_successes=2, pre_trials=8, post_successes=3, post_trials=16,
            disposition="updated", slot_pre=(0, 0),
            slot_post=(0, 0), accepted=True, persisted=True,
        )
        first_b = _frontier_event(
            event_index=2, cycle=2, group_index=0,
            chain_label="replacement-b", sequence=0, source="new", successes=4,
            pre_successes=0, pre_trials=0, post_successes=4, post_trials=8,
            disposition="inserted", slot_pre=None, slot_post=(0, 1),
            accepted=True, persisted=True, pre_update_count=1,
        )
        replay_b = _frontier_event(
            event_index=3, cycle=3, group_index=0,
            chain_label="replacement-b", sequence=1, source="replay", successes=3,
            pre_successes=4, pre_trials=8, post_successes=7, post_trials=16,
            disposition="updated", slot_pre=(0, 1), slot_post=(0, 1),
            accepted=True, persisted=True, pre_update_count=1,
        )
        return [first_a, final_a, first_b, replay_b]

    def test_eviction_requires_overwriter_and_generation_advances(self) -> None:
        telemetry.validate_events(self._eviction_ledger())
        no_overwriter = [copy.deepcopy(self._eviction_ledger()[0])]
        no_overwriter[0]["disposition"] = "inserted_then_evicted"
        no_overwriter[0]["posterior_persisted_after_snapshot"] = False
        no_overwriter[0]["post_score"] = None
        no_overwriter[0]["post_score_semantics"] = "not_persisted"
        with self.assertRaisesRegex(telemetry.TelemetryError, "persistence"):
            telemetry.validate_events(no_overwriter)

        skipped_generation = self._eviction_ledger()
        skipped_generation[2]["slot_generation_post"] = 2
        skipped_generation[3]["slot_generation_pre"] = 2
        skipped_generation[3]["slot_generation_post"] = 2
        with self.assertRaisesRegex(telemetry.TelemetryError, "generation"):
            telemetry.validate_events(skipped_generation)

    def test_cross_cycle_eviction_does_not_retroactively_relabel_replay(self) -> None:
        records = self._eviction_ledger()
        telemetry.validate_events(records)
        self.assertEqual(records[1]["disposition"], "updated")
        self.assertTrue(records[1]["posterior_persisted_after_snapshot"])
        self.assertEqual(records[2]["disposition"], "inserted")

        retroactively_relabelled = copy.deepcopy(records)
        replay = retroactively_relabelled[1]
        replay["disposition"] = "updated_then_evicted"
        replay["posterior_persisted_after_snapshot"] = False
        replay["post_score"] = None
        replay["post_score_semantics"] = "not_persisted"
        with self.assertRaisesRegex(telemetry.TelemetryError, "disposition drift"):
            telemetry.validate_events(retroactively_relabelled)

    def test_evicted_chain_may_not_reappear(self) -> None:
        records = self._eviction_ledger()
        records.append(_frontier_event(
            event_index=4, cycle=4, group_index=0,
            chain_label="evicted-a", sequence=2, source="replay", successes=2,
            pre_successes=3, pre_trials=16, post_successes=5, post_trials=24,
            disposition="updated", slot_pre=(0, 0), slot_post=(0, 0),
            accepted=True, persisted=True,
            score_source_snapshot_id=records[1]["snapshot_id"],
            pre_update_count=2,
        ))
        with self.assertRaisesRegex(telemetry.TelemetryError, "non-live|reappeared"):
            telemetry.validate_events(records)

    def test_inserted_then_evicted_requires_canonical_later_insertion(self) -> None:
        first = _frontier_event(
            event_index=0, cycle=0, group_index=0,
            chain_label="temporary", sequence=0, source="new", successes=2,
            pre_successes=0, pre_trials=0, post_successes=2, post_trials=8,
            disposition="inserted_then_evicted", slot_pre=None, slot_post=(0, 0),
            accepted=True, persisted=False,
        )
        second = _frontier_event(
            event_index=1, cycle=0, group_index=1,
            chain_label="winner", sequence=0, source="new", successes=3,
            pre_successes=0, pre_trials=0, post_successes=3, post_trials=8,
            disposition="inserted", slot_pre=None, slot_post=(0, 1),
            accepted=True, persisted=True,
        )
        telemetry.validate_events([first, second])
        with self.assertRaisesRegex(telemetry.TelemetryError, "persistence"):
            telemetry.validate_events([first])

    def test_source_disposition_state_machine_rejects_new_updated_then_evicted(self) -> None:
        record = copy.deepcopy(_frontier_ledger()[0])
        record["disposition"] = "updated_then_evicted"
        record["posterior_persisted_after_snapshot"] = False
        record["post_score"] = None
        record["post_score_semantics"] = "not_persisted"
        with self.assertRaisesRegex(telemetry.TelemetryError, "disposition drift"):
            telemetry.validate_events([record])

    def test_one_live_generation_per_slot_is_derived_not_claimed(self) -> None:
        first = copy.deepcopy(_frontier_ledger()[0])
        second = _frontier_event(
            event_index=1, cycle=0, group_index=1,
            chain_label="same-slot", sequence=0, source="new", successes=3,
            pre_successes=0, pre_trials=0, post_successes=3, post_trials=8,
            disposition="inserted", slot_pre=None, slot_post=(0, 1),
            accepted=True, persisted=True,
        )
        with self.assertRaisesRegex(telemetry.TelemetryError, "persistence"):
            telemetry.validate_events([first, second])

    def test_level_sha_is_immutable_and_unique_across_chains(self) -> None:
        changed = _frontier_ledger()
        changed[4]["level_sha256"] = _hash("changed level")
        with self.assertRaisesRegex(telemetry.TelemetryError, "changed within"):
            telemetry.validate_events(changed)

        duplicate = _frontier_ledger()
        duplicate[1]["level_sha256"] = duplicate[0]["level_sha256"]
        with self.assertRaisesRegex(telemetry.TelemetryError, "distinct level chains"):
            telemetry.validate_events(duplicate)

    def test_input_order_is_preserved_and_must_be_monotonic(self) -> None:
        records = _frontier_ledger()
        records[0], records[1] = records[1], records[0]
        for index, record in enumerate(records):
            record["event_index"] = index
        with self.assertRaisesRegex(telemetry.TelemetryError, "strictly ordered"):
            telemetry.validate_events(records)

    def test_partial_groups_are_ledgered_but_not_calibrated(self) -> None:
        record = _frontier_event(
            event_index=0, cycle=0, group_index=0,
            chain_label="partial", sequence=0, source="new", successes=1,
            trials=3, pre_successes=0, pre_trials=0,
            post_successes=0, post_trials=0,
            disposition="incomplete_rejected", slot_pre=None, slot_post=None,
            accepted=False, persisted=False,
        )
        result = telemetry.analyze_events([record])
        self.assertEqual(result["complete_group_count"], 0)
        self.assertEqual(
            result["calibration"]["all_complete_adaptively_purchased_groups"]["count"],
            0,
        )

    def test_maxmc_prior_score_provenance_is_continuous_and_outcome_blind(self) -> None:
        records = _maxmc_ledger()
        telemetry.validate_events(records)
        wrong_source = copy.deepcopy(records)
        wrong_source[4]["pre_score_source_snapshot_id"] = _hash("current snapshot")
        with self.assertRaisesRegex(telemetry.TelemetryError, "preceding snapshot"):
            telemetry.validate_events(wrong_source)
        wrong_score = copy.deepcopy(records)
        wrong_score[4]["pre_score"] = 0.99
        with self.assertRaisesRegex(telemetry.TelemetryError, "prior stored post-score"):
            telemetry.validate_events(wrong_score)
        current_group_call = copy.deepcopy(records)
        current_group_call[0]["pre_score"] = 0.4
        with self.assertRaisesRegex(telemetry.TelemetryError, "must be null"):
            telemetry.validate_events(current_group_call)

    def test_unknown_maxmc_current_group_score_field_is_rejected(self) -> None:
        records = _maxmc_ledger()
        records[4]["current_group_maxmc_score"] = 999.0
        with self.assertRaisesRegex(telemetry.TelemetryError, "keys drift"):
            telemetry.validate_events(records)

    def test_duplicate_new_label_cannot_replace_canonical_derivation(self) -> None:
        record = copy.deepcopy(_frontier_ledger()[0])
        record["disposition"] = "duplicate_new_rejected"
        record["slot_index_post"] = None
        record["slot_generation_post"] = None
        record["posterior_evidence_accepted"] = False
        record["posterior_persisted_after_snapshot"] = False
        record["post_successes"] = 0
        record["post_trials"] = 0
        record["post_score"] = None
        record["post_score_semantics"] = "not_persisted"
        with self.assertRaisesRegex(
            telemetry.TelemetryError, "canonical identity derivation"
        ):
            telemetry.validate_events([record])


class CalibrationCampaignAndPackageTest(unittest.TestCase):
    def test_campaign_loads_and_hashes_every_referenced_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path, digest, campaign = _write_campaign(root)
            validated = telemetry.validate_campaign_contract(
                path, digest, _analyzer_sha()
            )
            self.assertEqual(validated["campaign_id"], campaign["campaign_id"])

    def test_campaign_rejects_external_digest_and_analyzer_expectation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path, digest, _ = _write_campaign(root)
            with self.assertRaisesRegex(telemetry.TelemetryError, "campaign-contract"):
                telemetry.validate_campaign_contract(
                    path, _hash("wrong campaign"), _analyzer_sha()
                )
            with self.assertRaisesRegex(telemetry.TelemetryError, "external expectation"):
                telemetry.validate_campaign_contract(
                    path, digest, _hash("wrong analyzer")
                )

    def test_each_common_campaign_artifact_tamper_is_rejected(self) -> None:
        for name in sorted(telemetry.COMMON_ARTIFACT_KEYS):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                path, digest, campaign = _write_campaign(root)
                artifact_path = Path(campaign["common_artifacts"][name]["path"])
                artifact_path.write_text("tampered\n", encoding="utf-8")
                with self.assertRaisesRegex(telemetry.TelemetryError, "byte hash drift"):
                    telemetry.validate_campaign_contract(
                        path, digest, _analyzer_sha()
                    )

    def test_each_arm_config_byte_tamper_is_rejected(self) -> None:
        for arm in sorted(telemetry.ALLOWED_ARMS):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                path, digest, campaign = _write_campaign(root)
                config = Path(campaign["arms"][arm]["config"]["path"])
                config.write_text("tampered config\n", encoding="utf-8")
                with self.assertRaisesRegex(telemetry.TelemetryError, "byte hash drift"):
                    telemetry.validate_campaign_contract(
                        path, digest, _analyzer_sha()
                    )

    def test_campaign_artifact_roles_cannot_alias_one_path(self) -> None:
        for label, mutation, message in (
            (
                "common_roles",
                lambda campaign: campaign["common_artifacts"].__setitem__(
                    "telemetry_writer",
                    copy.deepcopy(campaign["common_artifacts"]["training_driver"]),
                ),
                "common artifact role-path collision",
            ),
            (
                "arm_configs",
                lambda campaign: campaign["arms"]["maxmc"].__setitem__(
                    "config", copy.deepcopy(campaign["arms"]["frontier"]["config"])
                ),
                "config role-path collision",
            ),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                path, _, campaign = _write_campaign(root)
                mutation(campaign)
                path.write_text(_canonical(campaign), encoding="utf-8")
                digest = telemetry.sha256(path)
                with self.assertRaisesRegex(telemetry.TelemetryError, message):
                    telemetry.validate_campaign_contract(
                        path, digest, _analyzer_sha()
                    )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path, _, campaign = _write_campaign(root)
            writer = Path(campaign["common_artifacts"]["telemetry_writer"]["path"])
            campaign["common_artifacts"]["telemetry_writer"]["path"] = str(
                writer.parent / ".." / writer.parent.name / writer.name
            )
            path.write_text(_canonical(campaign), encoding="utf-8")
            digest = telemetry.sha256(path)
            with self.assertRaisesRegex(telemetry.TelemetryError, "path must be canonical"):
                telemetry.validate_campaign_contract(
                    path, digest, _analyzer_sha()
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path, _, campaign = _write_campaign(root)
            driver = Path(
                campaign["common_artifacts"]["training_driver"]["path"]
            )
            hardlink = driver.parent / "telemetry-writer-hardlink"
            os.link(driver, hardlink)
            campaign["common_artifacts"]["telemetry_writer"] = {
                "path": str(hardlink.resolve()),
                "sha256": telemetry.sha256(hardlink),
            }
            path.write_text(_canonical(campaign), encoding="utf-8")
            digest = telemetry.sha256(path)
            with self.assertRaisesRegex(
                telemetry.TelemetryError, "artifact hardlink alias"
            ):
                telemetry.validate_campaign_contract(
                    path, digest, _analyzer_sha()
                )

    def test_malformed_and_overflow_campaign_counters_fail_as_telemetry(self) -> None:
        for label, value in (
            ("wrong_type", "4"),
            ("bounded_overflow", telemetry.MAX_COUNTER + 1),
            ("derived_product_overflow", telemetry.MAX_COUNTER),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                path, _, campaign = _write_campaign(root)
                campaign["budget"]["max_outer_cycles"] = value
                path.write_text(_canonical(campaign), encoding="utf-8")
                digest = telemetry.sha256(path)
                with self.assertRaises(telemetry.TelemetryError):
                    telemetry.validate_campaign_contract(
                        path, digest, _analyzer_sha()
                    )

    def test_campaign_rejects_shape_target_and_cycle_cap_drift(self) -> None:
        corruptions = {
            "exact_N": {"n_eval": 4},
            "stream_layout": {"n_parallel": 3},
            "rollout": {"n_rollout_steps": 128},
            "zero_target": {"target_student_ppo_updates": 0},
            "zero_cap": {"max_outer_cycles": 0},
            "target_above_cap": {
                "target_student_ppo_updates": 5,
                "max_outer_cycles": 4,
            },
            "target_equals_cap_without_warmup": {
                "target_student_ppo_updates": 4,
                "max_outer_cycles": 4,
            },
            "zero_epochs": {"ppo_epochs": 0},
            "zero_minibatches": {"ppo_minibatches": 0},
        }
        for label, overrides in corruptions.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                path, digest, _ = _write_campaign(
                    root, budget_overrides=overrides
                )
                with self.assertRaises(telemetry.TelemetryError):
                    telemetry.validate_campaign_contract(
                        path, digest, _analyzer_sha()
                    )

    def test_warmup_zero_update_progress_is_valid_but_not_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            _, _, campaign = _write_campaign(root)
            counters = {
                "outer_cycle_count": 1,
                "upstream_n_iters": 1,
                "student_ppo_updates": 0,
                "upstream_n_updates": 0,
                "upstream_n_grad_updates": 0,
                "optimizer_step_applications": 0,
                "student_training_transition_count": 8192,
            }
            telemetry._validate_realized_counters(
                counters,
                campaign["budget"],
                terminal=False,
                label="hostile warmup fixture",
            )
            with self.assertRaisesRegex(
                telemetry.TelemetryError, "terminal update target"
            ):
                telemetry._validate_realized_counters(
                    counters,
                    campaign["budget"],
                    terminal=True,
                    label="hostile warmup fixture",
                )

    def test_zero_update_closed_package_fails_before_event_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            campaign_path, campaign_sha, campaign = _write_campaign(root)
            records = _frontier_ledger()
            receipt = _receipt(
                records, campaign, campaign_sha, student_updates=0
            )
            package = root / "zero-update-package"
            package.mkdir()
            package_sha = _write_package(
                package, records, campaign, campaign_sha, receipt
            )
            with self.assertRaisesRegex(
                telemetry.TelemetryError, "terminal update target"
            ):
                _validate_closed(
                    package, package_sha, campaign_path, campaign_sha
                )

    def test_terminal_cycle_must_be_the_target_replay_update(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            campaign_path, campaign_sha, campaign = _write_campaign(root)
            records = _frontier_ledger()[:4]
            for group_index, successes in enumerate((1, 2, 5, 7)):
                records.append(_frontier_event(
                    event_index=4 + group_index,
                    cycle=1,
                    group_index=group_index,
                    chain_label=f"false-terminal-{group_index}",
                    sequence=0,
                    source="new",
                    successes=successes,
                    pre_successes=0,
                    pre_trials=0,
                    post_successes=0,
                    post_trials=0,
                    disposition="valid_not_persisted",
                    slot_pre=None,
                    slot_post=None,
                    accepted=False,
                    persisted=False,
                ))
            receipt = _receipt(
                records, campaign, campaign_sha, student_updates=1
            )
            package = root / "all-new-terminal-package"
            package.mkdir()
            package_sha = _write_package(
                package, records, campaign, campaign_sha, receipt
            )
            with self.assertRaisesRegex(
                telemetry.TelemetryError, "terminal cycle is not.*replay update"
            ):
                _validate_closed(
                    package, package_sha, campaign_path, campaign_sha
                )

    def test_post_target_cycle_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            campaign_path, campaign_sha, campaign = _write_campaign(root)
            records = _frontier_ledger()
            for group_index, successes in enumerate((1, 2, 5, 7)):
                records.append(_frontier_event(
                    event_index=8 + group_index,
                    cycle=2,
                    group_index=group_index,
                    chain_label=f"post-target-{group_index}",
                    sequence=0,
                    source="new",
                    successes=successes,
                    pre_successes=0,
                    pre_trials=0,
                    post_successes=0,
                    post_trials=0,
                    disposition="valid_not_persisted",
                    slot_pre=None,
                    slot_post=None,
                    accepted=False,
                    persisted=False,
                    pre_update_count=1,
                ))
            package = root / "post-target-package"
            package.mkdir()
            package_sha = _write_package(
                package, records, campaign, campaign_sha
            )
            with self.assertRaisesRegex(
                telemetry.TelemetryError, "preterminal cycle already reached"
            ):
                _validate_closed(
                    package, package_sha, campaign_path, campaign_sha
                )

    def test_closed_package_validates_before_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            campaign_path, campaign_sha, campaign = _write_campaign(root)
            package = root / "frontier-package"
            package.mkdir()
            package_sha = _write_package(
                package, _frontier_ledger(), campaign, campaign_sha
            )
            result = _validate_closed(
                package, package_sha, campaign_path, campaign_sha
            )
        self.assertEqual(result["record_count"], 8)
        self.assertEqual(result["complete_group_count"], 8)
        self.assertEqual(result["independent_unit"], "training_seed")
        self.assertTrue(result["package_validated"])
        self.assertEqual(result["realized_exposure"]["outer_cycle_count"], 2)
        self.assertEqual(result["realized_exposure"]["upstream_n_iters"], 2)
        self.assertEqual(result["realized_exposure"]["student_ppo_updates"], 1)
        self.assertEqual(
            result["realized_exposure"]["student_training_transition_count"],
            16384,
        )
        self.assertFalse(result["endpoint_accessed"])
        self.assertFalse(result["paper_evidence"])

    def test_payload_tamper_fails_outer_hash_closure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            campaign_path, campaign_sha, campaign = _write_campaign(root)
            package = root / "package"
            package.mkdir()
            package_sha = _write_package(
                package, _frontier_ledger(), campaign, campaign_sha
            )
            with (package / "telemetry-events.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(telemetry.TelemetryError, "payload hash drift"):
                _validate_closed(package, package_sha, campaign_path, campaign_sha)

    def test_receipt_rejects_nonzero_integrity_and_analyzer_drift(self) -> None:
        for label, mutation, message in (
            (
                "partial",
                lambda receipt: receipt["integrity_counters"].__setitem__(
                    "partial_group_count", 1
                ),
                "partial groups violate",
            ),
            (
                "analyzer",
                lambda receipt: receipt["provenance"].__setitem__(
                    "analyzer_sha256", _hash("wrong analyzer")
                ),
                "analyzer hash drift",
            ),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                campaign_path, campaign_sha, campaign = _write_campaign(root)
                records = _frontier_ledger()
                receipt = _receipt(records, campaign, campaign_sha)
                mutation(receipt)
                package = root / "package"
                package.mkdir()
                package_sha = _write_package(
                    package, records, campaign, campaign_sha, receipt
                )
                with self.assertRaisesRegex(telemetry.TelemetryError, message):
                    _validate_closed(
                        package, package_sha, campaign_path, campaign_sha
                    )

    def test_receipt_rejects_realized_counter_and_campaign_projection_drift(self) -> None:
        cases = (
            ("updates_above_cycles", lambda r: r.__setitem__("student_ppo_updates", 3)),
            ("target_overshoot", lambda r: r.__setitem__("student_ppo_updates", 2)),
            ("n_iters", lambda r: r.__setitem__("upstream_n_iters", 1)),
            (
                "cycles_above_cap",
                lambda r: (
                    r.__setitem__("outer_cycle_count", 5),
                    r.__setitem__("upstream_n_iters", 5),
                    r.__setitem__("terminal_outer_cycle", 4),
                    r.__setitem__("student_training_transition_count", 40960),
                ),
            ),
            ("upstream_updates", lambda r: r.__setitem__("upstream_n_updates", 0)),
            (
                "upstream_grad_updates",
                lambda r: r.__setitem__("upstream_n_grad_updates", 0),
            ),
            (
                "optimizer_applications",
                lambda r: r.__setitem__("optimizer_step_applications", 4),
            ),
            (
                "cycle_based_optimizer_bug",
                lambda r: r.__setitem__("optimizer_step_applications", 10),
            ),
            (
                "transition_count",
                lambda r: r.__setitem__(
                    "student_training_transition_count", 8192
                ),
            ),
            ("ppo_epochs", lambda r: r.__setitem__("ppo_epochs", 4)),
            ("omitted_group", lambda r: r.__setitem__("attempted_group_count", 7)),
            (
                "source_bundle_manifest",
                lambda r: r["provenance"].__setitem__(
                    "source_bundle_manifest_sha256", _hash("wrong source")
                ),
            ),
            (
                "campaign_contract",
                lambda r: r["provenance"].__setitem__(
                    "campaign_contract_sha256", _hash("wrong campaign")
                ),
            ),
        )
        for label, mutate in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                campaign_path, campaign_sha, campaign = _write_campaign(root)
                records = _frontier_ledger()
                receipt = _receipt(records, campaign, campaign_sha)
                mutate(receipt)
                package = root / "package"
                package.mkdir()
                package_sha = _write_package(
                    package, records, campaign, campaign_sha, receipt
                )
                with self.assertRaises(telemetry.TelemetryError):
                    _validate_closed(
                        package, package_sha, campaign_path, campaign_sha
                    )

    def test_duplicate_new_counter_is_derived_from_canonical_identity(self) -> None:
        records = _frontier_ledger()
        records[1]["level_sha256"] = records[0]["level_sha256"]
        with self.assertRaisesRegex(telemetry.TelemetryError, "distinct level chains"):
            telemetry.validate_events(records)

    def test_matched_comparator_allows_unequal_exposure_only_under_same_target_cap(self) -> None:
        frontier_open = telemetry.analyze_events(_frontier_ledger())
        maxmc_open = telemetry.analyze_events(_maxmc_ledger())
        self.assertIsNone(maxmc_open["calibration"])
        self.assertFalse(maxmc_open["calibration_authorized"])
        with self.assertRaisesRegex(
            telemetry.TelemetryError, "immutable validated result"
        ):
            telemetry.compare_matched_runs(frontier_open, maxmc_open)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            campaign_path, campaign_sha, campaign = _write_campaign(root)
            frontier_package = root / "frontier"
            maxmc_package = root / "maxmc"
            frontier_package.mkdir()
            maxmc_package.mkdir()
            frontier_sha = _write_package(
                frontier_package,
                _frontier_ledger_with_extra_exploration_cycle(),
                campaign,
                campaign_sha,
            )
            maxmc_sha = _write_package(
                maxmc_package, _maxmc_ledger(), campaign, campaign_sha
            )
            frontier = _validate_closed(
                frontier_package, frontier_sha, campaign_path, campaign_sha
            )
            maxmc = _validate_closed(
                maxmc_package, maxmc_sha, campaign_path, campaign_sha
            )
            comparison = telemetry.compare_matched_runs(frontier, maxmc)
            with self.assertRaises(TypeError):
                maxmc["matched_run_contract"]["max_outer_cycles"] = 5
            with self.assertRaises(AttributeError):
                maxmc._data = {}
            with self.assertRaisesRegex(
                telemetry.TelemetryError, "immutable validated result"
            ):
                telemetry.compare_matched_runs(frontier, dict(maxmc))

            drift_cases = (
                (
                    "max_outer_cycles",
                    {"max_outer_cycles": 5},
                    _maxmc_ledger(),
                ),
                (
                    "target_student_ppo_updates",
                    {"target_student_ppo_updates": 2},
                    _maxmc_ledger_two_updates(),
                ),
            )
            for field, budget_overrides, records in drift_cases:
                with self.subTest(matched_contract_drift=field):
                    drift_root = root / f"drift-{field}"
                    drift_root.mkdir()
                    drift_path, drift_campaign_sha, drift_campaign = _write_campaign(
                        drift_root, budget_overrides=budget_overrides
                    )
                    drift_package = drift_root / "maxmc"
                    drift_package.mkdir()
                    drift_package_sha = _write_package(
                        drift_package,
                        records,
                        drift_campaign,
                        drift_campaign_sha,
                    )
                    drifted = _validate_closed(
                        drift_package,
                        drift_package_sha,
                        drift_path,
                        drift_campaign_sha,
                    )
                    with self.assertRaisesRegex(
                        telemetry.TelemetryError,
                        f"matched run-contract drift: {field}",
                    ):
                        telemetry.compare_matched_runs(frontier, drifted)

            forged_payload = telemetry._thaw_json(maxmc)
            forged_payload["delivery"]["count"] += 1
            forged = telemetry.ValidatedPackageResult(
                forged_payload,
                maxmc._validation_request,
                telemetry._VALIDATED_RESULT_TOKEN,
            )
            with self.assertRaisesRegex(
                telemetry.TelemetryError,
                "does not equal its revalidated closed package",
            ):
                telemetry.compare_matched_runs(frontier, forged)

            with (maxmc_package / "telemetry-events.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(
                telemetry.TelemetryError, "payload hash drift"
            ):
                telemetry.compare_matched_runs(frontier, maxmc)
        self.assertEqual(comparison["independent_unit"], "training_seed")
        self.assertEqual(
            comparison["paired_exposure"]["frontier_minus_maxmc"][
                "outer_cycle_count"
            ],
            1,
        )
        self.assertEqual(
            comparison["paired_exposure"]["frontier_minus_maxmc"][
                "attempted_group_count"
            ],
            4,
        )
        self.assertEqual(
            comparison["paired_exposure"]["frontier_minus_maxmc"][
                "student_training_transition_count"
            ],
            8192,
        )
        self.assertEqual(
            comparison["paired_exposure"]["frontier_minus_maxmc"][
                "student_ppo_updates"
            ],
            0,
        )
        self.assertFalse(comparison["maxmc_proper_calibration_authorized"])
        self.assertFalse(comparison["matched_level_causal_effect"])
        self.assertFalse(comparison["performance_or_ood_claim_authorized"])


if __name__ == "__main__":
    unittest.main()
