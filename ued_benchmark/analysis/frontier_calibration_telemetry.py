#!/usr/bin/env python3
"""Fail-closed analysis for the Frontier calibration-telemetry DRAFT.

The module is deliberately independent of JAX and of every frozen minimax
overlay.  It validates a future, separately instrumented JSONL sidecar and
recomputes both the pre-group posterior-predictive activity and the realized
group target.  No checkpoint is opened and no training or evaluation endpoint
is accessed.

MaxMC is handled only as a matched delivery/discrimination comparator.  Its
raw score is not a probability forecast and is never assigned a calibration
error.
"""

from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping as ABCMapping
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    ROOT / "ued_benchmark/analysis/frontier_calibration_telemetry_v1_draft.json"
)
PROTOCOL_ID = "frontier-calibration-telemetry-v1-draft"
PURPOSE = (
    "outcome_blind_online_posterior_predictive_calibration_and_"
    "group_matched_maxmc_diagnostics_only"
)
PROTOCOL_SHA256 = "4053c52052ade233224903b0c989d9f39b1a626762209da93c4432428c430004"
BASE_COMMIT = "d053054c5290a04c1c4cd8b55704d999cad73e30"
BASE_TREE = "b0cace1fc54984e21a842f12d15d0b899e33d270"
V4_CONTRACT_SHA256 = (
    "3d5f3827a82a4f713314091289196a1c2909dd5d7c4c96dd532052c5706e832b"
)
PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0
ALLOWED_N = frozenset({2, 4, 8})
ALLOWED_ARMS = frozenset({"frontier", "maxmc"})
ALLOWED_SOURCES = frozenset({"new", "replay", "mutation"})
ALLOWED_DISPOSITIONS = frozenset({
    "inserted",
    "inserted_then_evicted",
    "updated",
    "valid_not_persisted",
    "duplicate_new_rejected",
    "incomplete_rejected",
})
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")
MANIFEST_LINE_RE = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9_.-]+)$")
PRE_SCORE_ATOL = 5e-7
TARGET_ATOL = 1e-12
MAX_COUNTER = 2**63 - 1
PPO_EPOCHS = 5
PPO_MINIBATCHES = 1
BIN_EDGES = tuple(index / 20.0 for index in range(21))
PACKAGE_PAYLOADS = frozenset({
    "telemetry-events.jsonl",
    "telemetry-receipt.json",
})
PACKAGE_FILES = PACKAGE_PAYLOADS | frozenset({
    "telemetry-SHA256SUMS",
    "telemetry-COMPLETE",
})

PROTECTED_HASHES = {
    "ued_benchmark/UPSTREAM_PIN.json":
        "375ff36d64a98dd72f9b94f8bf7e63ae2cb6ec99571de37c7a8d483a936401d7",
    "ued_benchmark/OVERLAY_CONTRACT.json":
        "5868d346ba00e43225424f053ded3dea056b8f76ce4bfdaa1a107a4955cb9000",
    "ued_benchmark/OVERLAY_CONTRACT_V4.json": V4_CONTRACT_SHA256,
    "ued_benchmark/OVERLAY_LINEAGE.json":
        "784e2fd1f545d49c8d10c3f3aeda37aae51fa00127e2c14578702e275bfb6971",
    "ued_benchmark/scripts/apply_minimax_overlay.py":
        "ddd3569b86adb703c8c7141fe7f2dae7a49c2c6b08e326edd61c3e3da7a345f7",
    "ued_benchmark/scripts/apply_minimax_overlay_v4.py":
        "c2e5eb3dac02b86723ece485cd348832f1636198c781bae82c1d99df0167590b",
    "ued_benchmark/overlay/minimax/util/rl/frontier_activity.py":
        "63726251813bd9fafc2722409c4a2942c6ae2728327870797df47d01504738ca",
    "ued_benchmark/overlay/minimax/util/rl/tie_aware_rank.py":
        "1b9db20d05edd3212346e84d14606af91ae443c0665945a7b679ade161560244",
    "ued_benchmark/analysis/development_protocol_v1.json":
        "9d0ccbeaf83564958c5374e6e68793aa644013b1e9f6b889a91da69c99a720ba",
    "ued_benchmark/analysis/development_protocol_v2_tie_aware_draft.json":
        "1e4bd62be2412fa5291fde9d2c8750f30ed2e9c9f43afcda93d8ab552e4a3269",
    "ued_benchmark/analysis/development_protocol_v3_n_factorial_tie_aware_draft.json":
        "81a57668d3cfdf595f13710df6152a437b8c4640791fbeeed2ef8c9e9486f26f",
    "ued_benchmark/analysis/n_factorial_tie_aware_v4_draft_manifest.json":
        "58e1ffd9c7e3d80992971b331c540d6c8976c9cd4082391fae92de0df4fd417f",
    "ued_benchmark/configs/maze_frontier_exact_grouped_n8.json":
        "b49168142a9d5a5d8edce88634975ac52d7615dca8bad9fff1cbcaf29ec43508",
    "ued_benchmark/configs/maze_maxmc_group_matched_4x8_b500.json":
        "6ec2083745ccc585383170f0a14f464397614a4365ba644e5c9e7e4ef422d943",
    "ued_benchmark/configs/maze_frontier_exact_grouped_n8_tie_aware_v4.json":
        "0e1b1907b319e42437d91ef4b19fef9ea39183a68e49069a17e337d7f78147f2",
    "ued_benchmark/configs/maze_maxmc_group_matched_4x8_b500_tie_aware_v4.json":
        "a3cc3ddf387a3bb7cf3d9759c3f13e6a74f2c7e32de311ac344d54ef5e703ec6",
    "ued_benchmark/configs/maze_maxmc_upstream_official_reference_32x1_b4000.json":
        "a5b8b87799bce31564959b3e8b55cfdaba658b31b8135fa35e3b96704d65185b",
    "ued_benchmark/configs/maze_maxmc_v4_stable_rank_compat_32x1_b4000.json":
        "99def895587a85e2ad060c356bf53041cf1fb6d1140304496451748cce207c92",
    "ued_benchmark/configs/maze_frontier_posterior_bridge_n8_neval1.json":
        "581369156855cf58718a686ea849df71410253b09561aaf83a68a6353151c883",
    "ued_benchmark/configs/maze_frontier_factorial_n2_16x2_b2000_tie_aware_v4.json":
        "2e443515d3876ad8c8a632d9cc21f2a92288adf971cde0b2c4751679eed32791",
    "ued_benchmark/configs/maze_maxmc_factorial_n2_16x2_b2000_tie_aware_v4.json":
        "81e2af766e588896c3013de23f22906446e27e18661e2dfc6b6f2cc4e284f1b3",
    "ued_benchmark/configs/maze_frontier_factorial_n4_8x4_b1000_tie_aware_v4.json":
        "181ca0210ad988a699d408827b15941ef0a6b9c1588f4abd8c12dc1e6cc706b5",
    "ued_benchmark/configs/maze_maxmc_factorial_n4_8x4_b1000_tie_aware_v4.json":
        "9033de1f79ee7f64ac980ebe28e90542e3cfe93dddf63374c20e1e52def824fb",
    "ued_benchmark/configs/maze_frontier_factorial_n8_4x8_b500_tie_aware_v4.json":
        "5cdaf48da9b6e3f2ab9dd0b9dd8c94eb7e49fe07d7744fc46b7b5f735b3a436d",
    "ued_benchmark/configs/maze_maxmc_factorial_n8_4x8_b500_tie_aware_v4.json":
        "105c6695baf86b894d65c6756fc5647d560c84daaef35ee3d3859c1eb9f68090",
}

EVENT_KEYS = frozenset({
    "schema",
    "protocol_id",
    "run_id",
    "event_index",
    "training_seed",
    "arm",
    "N",
    "student_index",
    "outer_cycle",
    "within_cycle_group_index",
    "runner_branch",
    "pre_upstream_n_iters",
    "post_upstream_n_iters",
    "pre_upstream_n_updates",
    "post_upstream_n_updates",
    "pre_upstream_n_grad_updates",
    "post_upstream_n_grad_updates",
    "pre_optimizer_step_applications",
    "post_optimizer_step_applications",
    "snapshot_id",
    "level_chain_id",
    "level_sha256",
    "posterior_snapshot_sequence",
    "selection_source",
    "current_successes",
    "current_trials",
    "realized_activity",
    "pre_successes",
    "pre_trials",
    "prior_alpha",
    "prior_beta",
    "pre_score",
    "pre_score_semantics",
    "pre_score_source_snapshot_id",
    "post_score",
    "post_score_semantics",
    "posterior_evidence_accepted",
    "posterior_persisted_after_snapshot",
    "post_successes",
    "post_trials",
    "slot_index_pre",
    "slot_generation_pre",
    "slot_index_post",
    "slot_generation_post",
    "disposition",
})

RECEIPT_KEYS = frozenset({
    "schema",
    "protocol_id",
    "purpose",
    "status",
    "campaign_id",
    "campaign_contract_sha256",
    "run_id",
    "arm",
    "training_seed",
    "N",
    "n_eval",
    "n_parallel",
    "n_rollout_steps",
    "upstream_n_iters",
    "student_ppo_updates",
    "upstream_n_updates",
    "upstream_n_grad_updates",
    "ppo_epochs",
    "ppo_minibatches",
    "optimizer_step_applications",
    "student_training_transition_count",
    "telemetry_records",
    "attempted_group_count",
    "complete_group_count",
    "outer_cycle_count",
    "terminal_outer_cycle",
    "from_last_checkpoint",
    "closed_before_analysis",
    "endpoint_class",
    "production_authorized",
    "endpoint_accessed",
    "paper_evidence",
    "provenance",
    "integrity_counters",
})

PROVENANCE_KEYS = frozenset({
    "base_commit",
    "base_tree",
    "v4_contract_sha256",
    "protocol_sha256",
    "analyzer_sha256",
    "campaign_contract_sha256",
    "config_sha256",
    "source_bundle_manifest_sha256",
    "applied_overlay_manifest_sha256",
    "telemetry_overlay_sha256",
    "telemetry_writer_sha256",
    "training_driver_sha256",
    "environment_manifest_sha256",
    "scheduler_script_sha256",
})

COUNTER_KEYS = frozenset({
    "duplicate_event_id_count",
    "duplicate_new_group_count",
    "partial_group_count",
    "nonfinite_record_count",
    "repeated_level_same_batch_count",
})

CAMPAIGN_KEYS = frozenset({
    "schema",
    "campaign_id",
    "protocol_id",
    "purpose",
    "status",
    "frozen_before_endpoint_access",
    "production_authorized",
    "endpoint_access_authorized",
    "paper_evidence",
    "protocol",
    "analyzer",
    "common_artifacts",
    "runner_semantics",
    "arms",
    "budget",
})
ARTIFACT_KEYS = frozenset({"path", "sha256"})
COMMON_ARTIFACT_KEYS = frozenset({
    "source_bundle_manifest",
    "applied_overlay_manifest",
    "telemetry_overlay",
    "telemetry_writer",
    "training_driver",
    "environment_manifest",
    "scheduler_script",
})
CAMPAIGN_ARM_KEYS = frozenset({"run_id", "training_seed", "config"})
RUNNER_SEMANTICS_KEYS = frozenset({
    "use_robust_plr",
    "use_mutations",
    "cycle_source_policy",
})
CAMPAIGN_BUDGET_KEYS = frozenset({
    "N",
    "n_eval",
    "n_parallel",
    "n_rollout_steps",
    "target_student_ppo_updates",
    "max_outer_cycles",
    "ppo_epochs",
    "ppo_minibatches",
})

REALIZED_COUNTER_KEYS = frozenset({
    "outer_cycle_count",
    "upstream_n_iters",
    "student_ppo_updates",
    "upstream_n_updates",
    "upstream_n_grad_updates",
    "optimizer_step_applications",
    "student_training_transition_count",
})


class TelemetryError(RuntimeError):
    """Raised when telemetry violates the frozen DRAFT contract."""


_VALIDATED_RESULT_TOKEN = object()


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


class ValidatedPackageResult(ABCMapping[str, Any]):
    """Immutable result bound to the exact request used for validation."""

    __slots__ = ("_data", "_validation_request", "_seal")

    def __init__(
        self,
        value: Mapping[str, Any],
        validation_request: Mapping[str, Any],
        token: object,
    ) -> None:
        require(token is _VALIDATED_RESULT_TOKEN,
                "validated package results may only be created by the validator")
        require(
            isinstance(validation_request, Mapping)
            and set(validation_request) == {
                "package_root",
                "expected_sha256sums_sha256",
                "campaign_contract_path",
                "expected_campaign_contract_sha256",
                "expected_analyzer_sha256",
            },
            "validated package request shape drift",
        )
        frozen = _freeze_json(value)
        frozen_request = _freeze_json(dict(validation_request))
        serialized = json.dumps(
            {
                "result": _thaw_json(frozen),
                "validation_request": _thaw_json(frozen_request),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        object.__setattr__(self, "_data", frozen)
        object.__setattr__(self, "_validation_request", frozen_request)
        object.__setattr__(self, "_seal", hashlib.sha256(serialized).hexdigest())

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("validated package results are immutable")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def _seal_is_valid(self) -> bool:
        serialized = json.dumps(
            {
                "result": _thaw_json(self._data),
                "validation_request": _thaw_json(self._validation_request),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        return hashlib.sha256(serialized).hexdigest() == self._seal


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TelemetryError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _is_counter(value: Any) -> bool:
    return _is_int(value) and 0 <= value <= MAX_COUNTER


def _bounded_product(values: Iterable[Any], label: str) -> int:
    """Multiply counters while keeping every derived value in int64 range."""
    product = 1
    for value in values:
        require(_is_counter(value), f"{label} factor must be a bounded counter")
        if value != 0:
            require(product <= MAX_COUNTER // value,
                    f"{label} exceeds the bounded-counter range")
        product *= value
    return product


def _require_exact_keys(
    value: Mapping[str, Any], expected: Iterable[str], label: str
) -> None:
    expected_set = set(expected)
    actual = set(value)
    require(
        actual == expected_set,
        f"{label} keys drift: expected {sorted(expected_set)}, got {sorted(actual)}",
    )


def sha256(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"unsafe or missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise TelemetryError(f"nonfinite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(text: str, label: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise TelemetryError(f"invalid JSON in {label}") from exc


def load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"unsafe or missing {label}: {path}")
    try:
        value = parse_json(path.read_text(encoding="utf-8"), label)
    except OSError as exc:
        raise TelemetryError(f"cannot read {label}: {path}") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def repository_preflight(expected_analyzer_sha256: str) -> dict[str, Any]:
    """Validate only local DRAFT metadata and frozen dependency bytes."""
    _require_hash(expected_analyzer_sha256, "externally expected analyzer digest")
    actual_analyzer_sha256 = sha256(Path(__file__).resolve())
    require(actual_analyzer_sha256 == expected_analyzer_sha256,
            "externally expected analyzer hash drift")
    require(sha256(PROTOCOL_PATH) == PROTOCOL_SHA256, "protocol hash drift")
    protocol = load_json(PROTOCOL_PATH, "calibration protocol")
    require(protocol.get("schema") == 1, "protocol schema drift")
    require(protocol.get("protocol_id") == PROTOCOL_ID, "protocol identity drift")
    require(protocol.get("purpose") == PURPOSE, "protocol purpose drift")
    require(protocol.get("production_driver_authorized") is False,
            "production authorization drift")
    require(protocol.get("endpoint_access_authorized") is False,
            "endpoint authorization drift")
    require(protocol.get("paper_evidence") is False, "paper-evidence drift")
    require(
        tuple(protocol["calibration_analysis"]["frozen_bin_edges"]) == BIN_EDGES,
        "calibration-bin drift",
    )
    prior = protocol["estimand"]["predeclared_prior"]
    require(
        prior == {"alpha": PRIOR_ALPHA, "beta": PRIOR_BETA},
        "predeclared prior drift",
    )
    require(protocol["estimand"]["independent_unit"] == "training_seed",
            "protocol independent-unit drift")
    dependencies = protocol["frozen_dependencies"]
    require(isinstance(dependencies, Mapping),
            "protocol frozen dependencies must be an object")
    _require_exact_keys(
        dependencies,
        {"base_commit", "base_tree", "protected_artifacts"},
        "protocol frozen dependencies",
    )
    require(dependencies["base_commit"] == BASE_COMMIT,
            "protocol base commit drift")
    require(dependencies["base_tree"] == BASE_TREE,
            "protocol base tree drift")
    protected_entries = dependencies["protected_artifacts"]
    require(isinstance(protected_entries, list),
            "protocol protected artifacts must be a list")
    declared: dict[str, str] = {}
    for index, entry in enumerate(protected_entries):
        require(isinstance(entry, Mapping),
                f"protocol protected artifact {index} must be an object")
        _require_exact_keys(entry, ARTIFACT_KEYS,
                            f"protocol protected artifact {index}")
        require(isinstance(entry["path"], str) and entry["path"] not in declared,
                "protocol protected artifact path collision")
        declared[entry["path"]] = _require_hash(
            entry["sha256"], f"protocol protected artifact {index} digest"
        )
    require(declared == PROTECTED_HASHES,
            "protocol/full-lineage protected artifact declaration drift")
    for relative, expected in PROTECTED_HASHES.items():
        require(sha256(ROOT / relative) == expected,
                f"protected artifact hash drift: {relative}")
    return {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "analyzer_sha256": actual_analyzer_sha256,
        "protected_artifact_count": len(PROTECTED_HASHES),
        "production_authorized": False,
        "endpoint_accessed": False,
        "paper_evidence": False,
    }


def _load_hashed_artifact(
    entry: Mapping[str, Any], label: str, expected_path: Path | None = None
) -> tuple[Path, str]:
    require(isinstance(entry, Mapping), f"{label} artifact must be an object")
    _require_exact_keys(entry, ARTIFACT_KEYS, f"{label} artifact")
    require(isinstance(entry["path"], str) and entry["path"],
            f"{label} artifact path must be nonempty")
    path = Path(entry["path"])
    require(path.is_absolute(), f"{label} artifact path must be absolute")
    try:
        canonical_path = path.resolve(strict=True)
    except OSError as exc:
        raise TelemetryError(f"cannot resolve {label} artifact path") from exc
    require(path == canonical_path,
            f"{label} artifact path must be canonical")
    require(path.is_file() and not path.is_symlink(),
            f"unsafe or missing {label} artifact: {path}")
    if expected_path is not None:
        require(path == expected_path.resolve(), f"{label} artifact path drift")
    expected = _require_hash(entry["sha256"], f"{label} artifact digest")
    actual = sha256(path)
    require(actual == expected, f"{label} artifact byte hash drift")
    return canonical_path, actual


def _require_distinct_artifact_roles(role_paths: Mapping[str, Path]) -> None:
    """Reject lexical, canonical, and hardlink aliases across artifact roles."""
    require(len(set(role_paths.values())) == len(role_paths),
            "campaign artifact role-path collision")
    identity_owner: dict[tuple[int, int], str] = {}
    for role, path in role_paths.items():
        try:
            stat = path.stat()
        except OSError as exc:
            raise TelemetryError(f"cannot stat campaign artifact role: {role}") from exc
        identity = (stat.st_dev, stat.st_ino)
        previous = identity_owner.setdefault(identity, role)
        require(previous == role,
                f"campaign artifact hardlink alias: {previous} and {role}")


def _validate_campaign_budget(budget: Mapping[str, Any], label: str) -> None:
    require(isinstance(budget, Mapping), f"{label} budget must be an object")
    _require_exact_keys(budget, CAMPAIGN_BUDGET_KEYS, f"{label} budget")
    for field in CAMPAIGN_BUDGET_KEYS:
        require(_is_counter(budget[field]) and budget[field] > 0,
                f"{label} budget {field} must be a positive integer")
    require(budget["N"] in ALLOWED_N, f"{label} budget N drift")
    require(budget["n_eval"] == budget["N"],
            f"{label} budget exact-N drift")
    require(_bounded_product(
        (budget["n_parallel"], budget["N"]),
        f"{label} stream layout",
    ) == 32,
            f"{label} budget stream-layout drift")
    require(budget["n_rollout_steps"] == 256,
            f"{label} budget rollout-length drift")
    require(
        (budget["ppo_epochs"], budget["ppo_minibatches"])
        == (PPO_EPOCHS, PPO_MINIBATCHES),
        f"{label} PPO optimizer-shape drift",
    )
    require(
        budget["target_student_ppo_updates"] < budget["max_outer_cycles"],
        f"{label} requires one all-new warm-up cycle plus the update target",
    )
    _bounded_product(
        (
            budget["target_student_ppo_updates"],
            budget["ppo_epochs"],
            budget["ppo_minibatches"],
        ),
        f"{label} target optimizer applications",
    )
    _bounded_product(
        (
            budget["max_outer_cycles"],
            budget["n_parallel"],
            budget["n_eval"],
            budget["n_rollout_steps"],
        ),
        f"{label} maximum transition count",
    )
    _bounded_product(
        (budget["max_outer_cycles"], budget["n_parallel"]),
        f"{label} maximum attempted-group count",
    )


def _validate_realized_counters(
    values: Mapping[str, Any],
    campaign_budget: Mapping[str, Any],
    *,
    terminal: bool,
    label: str,
) -> None:
    """Reconcile realized robust-PLR counters without equating cycles to updates."""
    require(isinstance(values, Mapping), f"{label} counters must be an object")
    _require_exact_keys(values, REALIZED_COUNTER_KEYS, f"{label} counters")
    for field in REALIZED_COUNTER_KEYS:
        require(_is_counter(values[field]),
                f"{label} {field} must be a nonnegative integer")
    cycles = values["outer_cycle_count"]
    updates = values["student_ppo_updates"]
    target = campaign_budget["target_student_ppo_updates"]
    cap = campaign_budget["max_outer_cycles"]
    require(cycles > 0, f"{label} must contain at least one purchased-group cycle")
    require(updates < cycles <= cap,
            f"{label} must satisfy updates < cycles <= cap after warm-up")
    require(updates <= target, f"{label} update target was overshot")
    if terminal:
        require(0 < updates == target,
                f"{label} terminal update target was not reached")
    require(values["upstream_n_iters"] == cycles,
            f"{label} outer_cycle_count/upstream n_iters drift")
    require(values["upstream_n_updates"] == updates,
            f"{label} student/upstream n_updates drift")
    require(values["upstream_n_grad_updates"] == updates,
            f"{label} student/upstream n_grad_updates drift")
    expected_optimizer_applications = _bounded_product(
        (
            updates,
            campaign_budget["ppo_epochs"],
            campaign_budget["ppo_minibatches"],
        ),
        f"{label} optimizer applications",
    )
    require(
        values["optimizer_step_applications"]
        == expected_optimizer_applications,
        f"{label} optimizer-step equation drift",
    )
    expected_transitions = _bounded_product(
        (
            cycles,
            campaign_budget["n_parallel"],
            campaign_budget["n_eval"],
            campaign_budget["n_rollout_steps"],
        ),
        f"{label} transition count",
    )
    require(
        values["student_training_transition_count"]
        == expected_transitions,
        f"{label} transition equation drift",
    )


def validate_campaign_contract(
    path: Path,
    expected_campaign_sha256: str,
    expected_analyzer_sha256: str,
) -> dict[str, Any]:
    """Load and byte-verify an externally frozen outcome-blind campaign."""
    require(path.is_absolute(), "campaign-contract path must be absolute")
    _require_hash(expected_campaign_sha256, "expected campaign-contract digest")
    require(sha256(path) == expected_campaign_sha256,
            "externally expected campaign-contract hash drift")
    campaign = load_json(path, "campaign contract")
    _require_exact_keys(campaign, CAMPAIGN_KEYS, "campaign contract")
    require(campaign["schema"] == 1, "campaign schema drift")
    require(
        isinstance(campaign["campaign_id"], str)
        and RUN_ID_RE.fullmatch(campaign["campaign_id"]) is not None,
        "campaign identity drift",
    )
    require(campaign["protocol_id"] == PROTOCOL_ID,
            "campaign protocol identity drift")
    require(campaign["purpose"] == PURPOSE, "campaign purpose drift")
    require(campaign["status"] == "DRAFT_FROZEN_OUTCOME_BLIND_NOT_RUN_AUTHORIZATION",
            "campaign status drift")
    require(campaign["frozen_before_endpoint_access"] is True,
            "campaign was not frozen before endpoint access")
    for field in (
        "production_authorized", "endpoint_access_authorized", "paper_evidence"
    ):
        require(campaign[field] is False, f"campaign {field} must be false")

    protocol_path, protocol_sha = _load_hashed_artifact(
        campaign["protocol"], "protocol", PROTOCOL_PATH
    )
    require(protocol_sha == PROTOCOL_SHA256, "campaign protocol digest drift")
    analyzer_path, analyzer_sha = _load_hashed_artifact(
        campaign["analyzer"], "analyzer", Path(__file__).resolve()
    )
    _require_hash(expected_analyzer_sha256, "externally expected analyzer digest")
    require(analyzer_sha == expected_analyzer_sha256,
            "campaign analyzer does not match external expectation")
    require(protocol_path != analyzer_path,
            "campaign protocol/analyzer role-path collision")

    runner_semantics = campaign["runner_semantics"]
    require(isinstance(runner_semantics, Mapping),
            "campaign runner semantics must be an object")
    _require_exact_keys(
        runner_semantics, RUNNER_SEMANTICS_KEYS, "campaign runner semantics"
    )
    require(runner_semantics == {
        "use_robust_plr": True,
        "use_mutations": False,
        "cycle_source_policy": "uniform_new_or_replay_cycle",
    }, "campaign robust-PLR runner semantics drift")

    common = campaign["common_artifacts"]
    require(isinstance(common, Mapping), "campaign common artifacts must be an object")
    _require_exact_keys(common, COMMON_ARTIFACT_KEYS, "campaign common artifacts")
    common_hashes: dict[str, str] = {}
    common_paths: dict[str, Path] = {}
    for name in sorted(COMMON_ARTIFACT_KEYS):
        common_paths[name], common_hashes[name] = _load_hashed_artifact(
            common[name], f"campaign {name}"
        )
    require(len(set(common_paths.values())) == len(common_paths),
            "campaign common artifact role-path collision")
    require(
        not set(common_paths.values()) & {protocol_path, analyzer_path},
        "campaign common artifact aliases protocol/analyzer path",
    )
    require(
        common_hashes["telemetry_overlay"]
        not in set(PROTECTED_HASHES.values()) | {PROTOCOL_SHA256},
        "campaign telemetry overlay is not separately versioned",
    )

    arms = campaign["arms"]
    require(isinstance(arms, Mapping) and set(arms) == ALLOWED_ARMS,
            "campaign arms drift")
    run_ids: set[str] = set()
    seeds: set[int] = set()
    config_hashes: set[str] = set()
    config_paths: dict[str, Path] = {}
    for arm in sorted(ALLOWED_ARMS):
        spec = arms[arm]
        require(isinstance(spec, Mapping), f"campaign {arm} arm must be an object")
        _require_exact_keys(spec, CAMPAIGN_ARM_KEYS, f"campaign {arm} arm")
        require(
            isinstance(spec["run_id"], str)
            and RUN_ID_RE.fullmatch(spec["run_id"]) is not None,
            f"campaign {arm} run_id drift",
        )
        require(spec["run_id"] not in run_ids, "campaign run_id collision")
        run_ids.add(spec["run_id"])
        require(_is_counter(spec["training_seed"]),
                f"campaign {arm} training seed drift")
        seeds.add(spec["training_seed"])
        config_path, config_hash = _load_hashed_artifact(
            spec["config"], f"campaign {arm} config"
        )
        require(config_path not in set(config_paths.values()),
                "campaign arm config role-path collision")
        require(
            config_path not in set(common_paths.values()) | {protocol_path, analyzer_path},
            f"campaign {arm} config aliases another artifact role",
        )
        config_paths[arm] = config_path
        require(config_hash not in config_hashes, "campaign arm config collision")
        config_hashes.add(config_hash)
    require(len(seeds) == 1, "campaign arms are not paired by training seed")
    _require_distinct_artifact_roles({
        "protocol": protocol_path,
        "analyzer": analyzer_path,
        **{f"common:{name}": path for name, path in common_paths.items()},
        **{f"config:{arm}": path for arm, path in config_paths.items()},
    })
    _validate_campaign_budget(campaign["budget"], "campaign")
    return campaign


def expected_activity(
    successes: int,
    trials: int,
    N: int,
    alpha: float = PRIOR_ALPHA,
    beta: float = PRIOR_BETA,
) -> float:
    """Return E[u_N(p)] for the pre-group Beta posterior in float64."""
    require(_is_counter(successes) and _is_counter(trials),
            "posterior counts must be bounded nonnegative integers")
    require(0 <= successes <= trials, "posterior counts violate 0 <= successes <= trials")
    require(_is_int(N) and N >= 2, "N must be an integer >= 2")
    require(_is_finite_number(alpha) and float(alpha) > 0.0,
            "alpha must be finite and positive")
    require(_is_finite_number(beta) and float(beta) > 0.0,
            "beta must be finite and positive")
    a = float(successes) + float(alpha)
    b = float(trials - successes) + float(beta)
    failure_moment = 1.0
    for offset in range(N):
        failure_moment *= (b + offset) / (a + b + offset)
    value = 1.0 - failure_moment - a / (a + b)
    require(math.isfinite(value), "expected activity is nonfinite")
    require(-1e-15 <= value <= 1.0 + 1e-15, "expected activity is out of range")
    return min(1.0, max(0.0, value))


def activity_at_probability(p: float, N: int) -> float:
    """Return u_N(p)=1-p-(1-p)^N."""
    require(_is_finite_number(p) and 0.0 <= float(p) <= 1.0,
            "p must be finite and in [0,1]")
    require(_is_int(N) and N >= 2, "N must be an integer >= 2")
    p_float = float(p)
    return 1.0 - p_float - (1.0 - p_float) ** N


def realized_activity(successes: int, N: int) -> float:
    """Return m_N(K) for an exact-N group."""
    require(_is_int(successes), "current successes must be an integer")
    require(_is_int(N) and N >= 2, "N must be an integer >= 2")
    require(0 <= successes <= N, "current successes violate 0 <= K <= N")
    if successes == 0 or successes == N:
        return 0.0
    return (N - successes) / N


def enumerated_conditional_expectation(p: float, N: int) -> float:
    """Independently enumerate E[m_N(K)|p] for tests and audits."""
    require(_is_finite_number(p) and 0.0 <= float(p) <= 1.0,
            "p must be finite and in [0,1]")
    require(_is_int(N) and N >= 2, "N must be an integer >= 2")
    p_float = float(p)
    total = 0.0
    for successes in range(N + 1):
        probability = (
            math.comb(N, successes)
            * p_float ** successes
            * (1.0 - p_float) ** (N - successes)
        )
        total += probability * realized_activity(successes, N)
    return total


def _require_hash(value: Any, label: str) -> str:
    require(isinstance(value, str) and HASH_RE.fullmatch(value) is not None,
            f"{label} must be a lowercase SHA-256")
    return value


def _nullable_nonnegative_int(value: Any, label: str) -> None:
    require(value is None or _is_counter(value),
            f"{label} must be null or a nonnegative integer")


def _validate_common_event(record: Mapping[str, Any], row: int) -> None:
    label = f"event row {row}"
    _require_exact_keys(record, EVENT_KEYS, label)
    require(record["schema"] == 1, f"{label} schema drift")
    require(record["protocol_id"] == PROTOCOL_ID, f"{label} protocol drift")
    require(
        isinstance(record["run_id"], str)
        and RUN_ID_RE.fullmatch(record["run_id"]) is not None,
        f"{label} invalid run_id",
    )
    for field in (
        "event_index",
        "training_seed",
        "student_index",
        "outer_cycle",
        "within_cycle_group_index",
        "pre_upstream_n_iters",
        "post_upstream_n_iters",
        "pre_upstream_n_updates",
        "post_upstream_n_updates",
        "pre_upstream_n_grad_updates",
        "post_upstream_n_grad_updates",
        "pre_optimizer_step_applications",
        "post_optimizer_step_applications",
        "posterior_snapshot_sequence",
        "current_successes",
        "current_trials",
    ):
        require(_is_counter(record[field]),
                f"{label} {field} must be a nonnegative integer")
    require(record["student_index"] == 0, f"{label} student index drift")
    require(isinstance(record["arm"], str) and record["arm"] in ALLOWED_ARMS,
            f"{label} arm drift")
    require(_is_int(record["N"]) and record["N"] in ALLOWED_N,
            f"{label} N drift")
    require(isinstance(record["selection_source"], str)
            and record["selection_source"] in ALLOWED_SOURCES,
            f"{label} selection source drift")
    require(isinstance(record["runner_branch"], str)
            and record["runner_branch"] in {"new", "replay"},
            f"{label} runner branch drift")
    require(isinstance(record["disposition"], str)
            and record["disposition"] in ALLOWED_DISPOSITIONS,
            f"{label} disposition drift")
    for field in ("snapshot_id", "level_chain_id", "level_sha256"):
        _require_hash(record[field], f"{label} {field}")
    if record["selection_source"] == "replay":
        _require_hash(
            record["pre_score_source_snapshot_id"],
            f"{label} pre_score_source_snapshot_id",
        )
    else:
        require(record["pre_score_source_snapshot_id"] is None,
                f"{label} new/mutation score source must be null")
    require(record["current_trials"] <= record["N"],
            f"{label} current trials exceed N")
    require(record["current_successes"] <= record["current_trials"],
            f"{label} current successes exceed trials")
    expected_target = (
        realized_activity(record["current_successes"], record["N"])
        if record["current_trials"] == record["N"]
        else None
    )
    if expected_target is None:
        require(record["realized_activity"] is None,
                f"{label} partial group must have null realized activity")
        require(
            record["disposition"]
            == "incomplete_rejected",
                f"{label} partial group disposition drift")
    else:
        require(_is_finite_number(record["realized_activity"]),
                f"{label} complete target must be finite")
        require(
            abs(float(record["realized_activity"]) - expected_target) <= TARGET_ATOL,
            f"{label} realized activity drift",
        )
        require(
            record["disposition"]
            != "incomplete_rejected",
                f"{label} complete group cannot be incomplete-rejected")

    for field in (
        "slot_index_pre",
        "slot_generation_pre",
        "slot_index_post",
        "slot_generation_post",
    ):
        _nullable_nonnegative_int(record[field], f"{label} {field}")
    require((record["slot_index_pre"] is None) ==
            (record["slot_generation_pre"] is None),
            f"{label} partial pre-slot identity")
    require((record["slot_index_post"] is None) ==
            (record["slot_generation_post"] is None),
            f"{label} partial post-slot identity")

    source = record["selection_source"]
    disposition = record["disposition"]
    require(isinstance(record["posterior_persisted_after_snapshot"], bool),
            f"{label} persisted flag must be Boolean for both arms")
    if source == "replay":
        require(record["slot_index_pre"] is not None,
                f"{label} replay must bind a pre-slot")
        require(
            disposition in {
                "updated", "incomplete_rejected",
            },
            f"{label} replay disposition drift")
        require(
            (record["slot_index_post"], record["slot_generation_post"])
            == (record["slot_index_pre"], record["slot_generation_pre"]),
            f"{label} replay slot identity changed",
        )
    else:
        require(record["slot_index_pre"] is None,
                f"{label} new/mutation group cannot have a pre-slot")
        require(
            disposition in {
                "inserted", "inserted_then_evicted",
                "valid_not_persisted", "duplicate_new_rejected",
                "incomplete_rejected",
            },
            f"{label} new/mutation disposition drift",
        )
        if disposition in {"inserted", "inserted_then_evicted"}:
            require(record["slot_index_post"] is not None,
                    f"{label} insertion must bind a post-slot")
        else:
            require(record["slot_index_post"] is None,
                    f"{label} non-insertion cannot bind a post-slot")
    expected_persisted = (
        disposition in {"inserted", "updated"}
        or (source == "replay" and disposition == "incomplete_rejected")
    )
    require(record["posterior_persisted_after_snapshot"] == expected_persisted,
            f"{label} source/disposition persistence drift")


def _validate_frontier_event(record: Mapping[str, Any], row: int) -> None:
    label = f"event row {row}"
    for field in ("pre_successes", "pre_trials", "post_successes", "post_trials"):
        require(_is_counter(record[field]),
                f"{label} {field} must be a nonnegative integer")
    require(record["pre_successes"] <= record["pre_trials"],
            f"{label} invalid pre counts")
    require(record["post_successes"] <= record["post_trials"],
            f"{label} invalid post counts")
    require(record["prior_alpha"] == PRIOR_ALPHA,
            f"{label} prior alpha drift")
    require(record["prior_beta"] == PRIOR_BETA,
            f"{label} prior beta drift")
    require(record["pre_score_semantics"] ==
            "posterior_expected_activity_before_current_batch",
            f"{label} Frontier score clock drift")
    require(_is_finite_number(record["pre_score"]),
            f"{label} Frontier pre-score must be finite")
    recomputed = expected_activity(
        record["pre_successes"],
        record["pre_trials"],
        record["N"],
        record["prior_alpha"],
        record["prior_beta"],
    )
    require(abs(float(record["pre_score"]) - recomputed) <= PRE_SCORE_ATOL,
            f"{label} Frontier pre-score used current or drifted counts")
    require(isinstance(record["posterior_evidence_accepted"], bool),
            f"{label} evidence flag must be Boolean")
    accepted = record["posterior_evidence_accepted"]
    disposition = record["disposition"]
    require(
        accepted == (
            disposition in {
                "inserted", "inserted_then_evicted",
                "updated",
            }
        ),
        f"{label} evidence/disposition mismatch",
    )
    require(not accepted or record["current_trials"] == record["N"],
            f"{label} incomplete evidence was accepted")
    if record["posterior_persisted_after_snapshot"]:
        require(record["post_score_semantics"] ==
                "posterior_expected_activity_after_current_batch",
                f"{label} Frontier post-score semantics drift")
        require(_is_finite_number(record["post_score"]),
                f"{label} persisted Frontier post-score must be finite")
        recomputed_post = expected_activity(
            record["post_successes"],
            record["post_trials"],
            record["N"],
            record["prior_alpha"],
            record["prior_beta"],
        )
        require(abs(float(record["post_score"]) - recomputed_post) <= PRE_SCORE_ATOL,
                f"{label} Frontier post-score drift")
    else:
        require(record["post_score"] is None,
                f"{label} nonpersisted Frontier post-score must be null")
        require(record["post_score_semantics"] == "not_persisted",
                f"{label} nonpersisted Frontier score semantics drift")


def _validate_maxmc_event(record: Mapping[str, Any], row: int) -> None:
    label = f"event row {row}"
    for field in (
        "pre_successes",
        "pre_trials",
        "prior_alpha",
        "prior_beta",
        "posterior_evidence_accepted",
        "post_successes",
        "post_trials",
    ):
        require(record[field] is None, f"{label} MaxMC {field} must be null")
    if record["selection_source"] == "replay":
        require(record["pre_score_semantics"] == "stored_maxmc_before_current_group",
                f"{label} MaxMC replay score clock drift")
        require(_is_finite_number(record["pre_score"]),
                f"{label} MaxMC replay pre-score must be finite")
    else:
        require(record["pre_score_semantics"] == "unavailable_new_candidate",
                f"{label} MaxMC new/mutation score semantics drift")
        require(record["pre_score"] is None,
                f"{label} MaxMC new/mutation pre-score must be null")
    if record["posterior_persisted_after_snapshot"]:
        require(record["post_score_semantics"] ==
                "stored_maxmc_after_current_batch",
                f"{label} MaxMC post-score semantics drift")
        require(_is_finite_number(record["post_score"]),
                f"{label} persisted MaxMC post-score must be finite")
    else:
        require(record["post_score"] is None,
                f"{label} nonpersisted MaxMC post-score must be null")
        require(record["post_score_semantics"] == "not_persisted",
                f"{label} nonpersisted MaxMC score semantics drift")


def _validate_cycle_update_ledger(records: Sequence[Mapping[str, Any]]) -> None:
    """Bind every purchased cycle to the pinned robust-PLR update branch."""
    cycle_groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        cycle_groups[record["outer_cycle"]].append(record)
    clock_fields = (
        "runner_branch",
        "pre_upstream_n_iters",
        "post_upstream_n_iters",
        "pre_upstream_n_updates",
        "post_upstream_n_updates",
        "pre_upstream_n_grad_updates",
        "post_upstream_n_grad_updates",
        "pre_optimizer_step_applications",
        "post_optimizer_step_applications",
    )
    previous: Mapping[str, Any] | None = None
    for cycle in sorted(cycle_groups):
        group = cycle_groups[cycle]
        first = group[0]
        for record in group[1:]:
            for field in clock_fields:
                require(record[field] == first[field],
                        f"within-cycle counter/branch drift: {field}")
        sources = {record["selection_source"] for record in group}
        require("mutation" not in sources,
                "mutation telemetry is forbidden without frozen driver semantics")
        pre_iters = first["pre_upstream_n_iters"]
        post_iters = first["post_upstream_n_iters"]
        pre_updates = first["pre_upstream_n_updates"]
        post_updates = first["post_upstream_n_updates"]
        pre_grad = first["pre_upstream_n_grad_updates"]
        post_grad = first["post_upstream_n_grad_updates"]
        pre_optimizer = first["pre_optimizer_step_applications"]
        post_optimizer = first["post_optimizer_step_applications"]
        require(post_iters == pre_iters + 1,
                "upstream n_iters did not advance exactly once per cycle")
        update_delta = post_updates - pre_updates
        require(update_delta in {0, 1},
                "student update delta is not zero or one")
        require(pre_grad == pre_updates and post_grad == post_updates,
                "upstream n_updates/n_grad_updates cycle clock drift")
        require(post_grad - pre_grad == update_delta,
                "gradient-counter delta disagrees with update delta")
        expected_pre_optimizer = _bounded_product(
            (pre_updates, PPO_EPOCHS, PPO_MINIBATCHES),
            "pre-cycle optimizer applications",
        )
        expected_post_optimizer = _bounded_product(
            (post_updates, PPO_EPOCHS, PPO_MINIBATCHES),
            "post-cycle optimizer applications",
        )
        expected_optimizer_delta = _bounded_product(
            (update_delta, PPO_EPOCHS, PPO_MINIBATCHES),
            "cycle optimizer delta",
        )
        require(
            pre_optimizer == expected_pre_optimizer
            and post_optimizer == expected_post_optimizer,
            "optimizer cumulative counter disagrees with update clock",
        )
        require(
            post_optimizer - pre_optimizer
            == expected_optimizer_delta,
            "optimizer delta disagrees with update delta",
        )
        if first["runner_branch"] == "replay":
            require(sources == {"replay"},
                    "replay runner branch is not cycle-uniform replay")
            require(update_delta == 1,
                    "replay runner branch did not apply one student update")
        else:
            require(sources == {"new"},
                    "new runner branch is not cycle-uniform all-new")
            require(update_delta == 0,
                    "all-new runner branch falsely claims a student update")
        if previous is None:
            require(
                (pre_iters, pre_updates, pre_grad, pre_optimizer) == (0, 0, 0, 0),
                "cycle counter ledger did not begin from zero",
            )
        else:
            require(
                (
                    pre_iters,
                    pre_updates,
                    pre_grad,
                    pre_optimizer,
                )
                == (
                    previous["post_upstream_n_iters"],
                    previous["post_upstream_n_updates"],
                    previous["post_upstream_n_grad_updates"],
                    previous["post_optimizer_step_applications"],
                ),
                "cycle counter ledger is discontinuous",
            )
        previous = first


def _validate_snapshot_chains(records: Sequence[Mapping[str, Any]]) -> None:
    # A snapshot is identified by logical level and outer cycle, never by a
    # caller-selected sequence number. This makes two snapshots in one batch a
    # structural error instead of two apparently valid chain steps.
    snapshots: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        snapshots[(record["level_chain_id"], record["outer_cycle"])].append(record)

    snapshot_id_owner: dict[str, tuple[str, int]] = {}
    slot_generation_owner: dict[tuple[int, int], str] = {}
    chains: dict[str, list[tuple[int, list[Mapping[str, Any]]]]] = defaultdict(list)
    level_owner: dict[str, str] = {}
    for key, group in snapshots.items():
        chain_id, cycle = key
        first = group[0]
        invariant_fields = (
            "run_id",
            "arm",
            "training_seed",
            "N",
            "student_index",
            "outer_cycle",
            "snapshot_id",
            "level_chain_id",
            "level_sha256",
            "posterior_snapshot_sequence",
            "selection_source",
            "pre_successes",
            "pre_trials",
            "prior_alpha",
            "prior_beta",
            "pre_score",
            "pre_score_semantics",
            "pre_score_source_snapshot_id",
            "post_score",
            "post_score_semantics",
            "posterior_evidence_accepted",
            "posterior_persisted_after_snapshot",
            "post_successes",
            "post_trials",
            "slot_index_pre",
            "slot_generation_pre",
            "slot_index_post",
            "slot_generation_post",
            "disposition",
        )
        for record in group[1:]:
            for field in invariant_fields:
                require(record[field] == first[field],
                        f"concurrent snapshot field drift: {field}")
        owner = snapshot_id_owner.setdefault(first["snapshot_id"], key)
        require(owner == key, "snapshot_id reused across logical snapshots")
        chains[chain_id].append((cycle, group))
        canonical_owner = level_owner.setdefault(first["level_sha256"], chain_id)
        require(canonical_owner == chain_id,
                "distinct level chains share one canonical level_sha256")
        if first["selection_source"] in {"new", "mutation"}:
            require(len(group) == 1,
                    "new/mutation snapshot contains duplicate canonical groups")

        if first["arm"] == "frontier":
            accepted_successes = sum(
                record["current_successes"]
                for record in group
                if record["posterior_evidence_accepted"]
            )
            accepted_trials = sum(
                record["current_trials"]
                for record in group
                if record["posterior_evidence_accepted"]
            )
            require(
                first["post_successes"]
                == first["pre_successes"] + accepted_successes,
                "snapshot post-success count does not aggregate accepted siblings",
            )
            require(
                first["post_trials"] == first["pre_trials"] + accepted_trials,
                "snapshot post-trial count does not aggregate accepted siblings",
            )

        post_slot = (first["slot_index_post"], first["slot_generation_post"])
        if post_slot[0] is not None:
            typed_slot = (int(post_slot[0]), int(post_slot[1]))
            prior_owner = slot_generation_owner.setdefault(typed_slot, chain_id)
            require(prior_owner == chain_id,
                    "slot-index/generation pair reused by another level chain")

    # Duplicate-new is derived from canonical identity, source, and cycle. A
    # disposition label is an integrity check and is never the source of truth.
    new_identity_counts = Counter(
        (record["outer_cycle"], record["level_sha256"])
        for record in records
        if record["selection_source"] in {"new", "mutation"}
    )
    derived_duplicate_new = sum(
        max(0, count - 1) for count in new_identity_counts.values()
    )
    labeled_duplicate_new = sum(
        record["disposition"] == "duplicate_new_rejected" for record in records
    )
    require(labeled_duplicate_new == derived_duplicate_new,
            "duplicate-new label disagrees with canonical identity derivation")
    require(derived_duplicate_new == 0,
            "duplicate-new canonical level groups are forbidden")

    for chain_id, cycle_groups in chains.items():
        ordered = sorted(cycle_groups, key=lambda item: item[0])
        cycles = [cycle for cycle, _ in ordered]
        require(all(left < right for left, right in zip(cycles, cycles[1:])),
                "level-chain snapshot cycles do not strictly advance")
        sequences = [
            group[0]["posterior_snapshot_sequence"] for _, group in ordered
        ]
        require(sequences == list(range(len(ordered))),
                "posterior snapshot sequence is not zero-based contiguous")
        first_record = ordered[0][1][0]
        require(first_record["selection_source"] in {"new", "mutation"},
                "level chain did not begin as new or mutation")
        if first_record["arm"] == "frontier":
            require(
                (first_record["pre_successes"], first_record["pre_trials"])
                == (0, 0),
                "new/mutation chain did not use the predeclared prior",
            )

        immutable_level_sha = first_record["level_sha256"]
        previous: Mapping[str, Any] | None = None
        for _, group in ordered:
            current = group[0]
            require(current["level_sha256"] == immutable_level_sha,
                    "level_sha256 changed within one level chain")
            if previous is not None:
                require(previous["posterior_persisted_after_snapshot"] is True,
                        "evicted or nonpersisted chain reappeared")
                require(current["selection_source"] == "replay",
                        "continued level chain is not replay-sourced")
                require(
                    current["pre_score_source_snapshot_id"]
                    == previous["snapshot_id"],
                    "pre-group score does not name the preceding snapshot",
                )
                require(
                    (current["slot_index_pre"], current["slot_generation_pre"])
                    == (previous["slot_index_post"], previous["slot_generation_post"]),
                    "level-chain slot continuity drift",
                )
                require(
                    abs(float(current["pre_score"]) - float(previous["post_score"]))
                    <= (PRE_SCORE_ATOL if current["arm"] == "frontier" else 0.0),
                    "pre-group score does not equal prior stored post-score",
                )
                if current["arm"] == "frontier":
                    require(
                        (current["pre_successes"], current["pre_trials"])
                        == (previous["post_successes"], previous["post_trials"]),
                        "current-group leakage or posterior-chain discontinuity",
                    )
            previous = current

    # Validate a single live generation per slot. Replay identities are checked
    # against the pre-batch live map before any insertion in that cycle. Every
    # insertion is applied in canonical group order, including an insertion that
    # a later candidate evicts in the same cycle. Final persistence/disposition
    # is therefore derived from the lifecycle rather than trusted as a label.
    representatives = {
        key: group[0] for key, group in snapshots.items()
    }
    live_slots: dict[int, tuple[int, str]] = {}
    live_chain_slot: dict[str, tuple[int, int]] = {}
    last_generation: dict[int, int] = {}
    all_cycles = sorted({record["outer_cycle"] for record in records})
    for cycle in all_cycles:
        current = [
            record for (chain_id, snapshot_cycle), record in representatives.items()
            if snapshot_cycle == cycle
        ]
        replays = sorted(
            (record for record in current if record["selection_source"] == "replay"),
            key=lambda record: record["within_cycle_group_index"],
        )
        candidates = sorted(
            (record for record in current if record["selection_source"] != "replay"),
            key=lambda record: record["within_cycle_group_index"],
        )
        for record in replays:
            slot = int(record["slot_index_pre"])
            generation = int(record["slot_generation_pre"])
            require(live_slots.get(slot) == (generation, record["level_chain_id"]),
                    "replay references a non-live slot generation")
            require(live_chain_slot.get(record["level_chain_id"])
                    == (slot, generation),
                    "replay chain has no unique live slot generation")
        for record in candidates:
            if record["slot_index_post"] is None:
                continue
            slot = int(record["slot_index_post"])
            generation = int(record["slot_generation_post"])
            expected_generation = last_generation.get(slot, -1) + 1
            require(generation == expected_generation,
                    "new insertion slot generation did not advance exactly once")
            last_generation[slot] = generation
            prior_live = live_slots.pop(slot, None)
            if prior_live is not None:
                prior_generation, prior_chain = prior_live
                require(live_chain_slot.pop(prior_chain) == (slot, prior_generation),
                        "slot eviction did not remove one live chain")
            require(record["level_chain_id"] not in live_chain_slot,
                    "one level chain occupies multiple live slots")
            live_slots[slot] = (generation, record["level_chain_id"])
            live_chain_slot[record["level_chain_id"]] = (slot, generation)
        for record in current:
            chain_id = record["level_chain_id"]
            if record["slot_index_post"] is None:
                derived_persisted = False
            else:
                slot = int(record["slot_index_post"])
                generation = int(record["slot_generation_post"])
                derived_persisted = (
                    live_slots.get(slot) == (generation, chain_id)
                    and live_chain_slot.get(chain_id) == (slot, generation)
                )
            require(
                record["posterior_persisted_after_snapshot"] is derived_persisted,
                "snapshot persistence disagrees with live-slot lifecycle",
            )
            source = record["selection_source"]
            complete = record["current_trials"] == record["N"]
            if source == "replay":
                require(derived_persisted,
                        "uniform replay cycle unexpectedly evicted a replay")
                expected_disposition = (
                    "updated" if complete else "incomplete_rejected"
                )
            elif record["slot_index_post"] is not None:
                expected_disposition = (
                    "inserted" if derived_persisted else "inserted_then_evicted"
                )
            elif complete:
                # duplicate_new_rejected is allowed by the row schema so that
                # corrupt writer output can be diagnosed, but the independently
                # derived duplicate-new clean gate below always rejects it.
                expected_disposition = record["disposition"]
                require(
                    expected_disposition
                    in {"valid_not_persisted", "duplicate_new_rejected"},
                    "complete non-insertion disposition drift",
                )
            else:
                expected_disposition = "incomplete_rejected"
            require(record["disposition"] == expected_disposition,
                    "source/disposition disagrees with live-slot lifecycle")


def validate_events(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate an ordered run ledger and attach analyzer-derived quantities."""
    require(isinstance(records, Sequence) and len(records) > 0,
            "telemetry ledger must be a nonempty sequence")
    copied: list[dict[str, Any]] = []
    for row, raw in enumerate(records, start=1):
        require(isinstance(raw, Mapping), f"event row {row} must be an object")
        record = dict(raw)
        _validate_common_event(record, row)
        if record["arm"] == "frontier":
            _validate_frontier_event(record, row)
        else:
            _validate_maxmc_event(record, row)
        copied.append(record)

    homogeneous_fields = ("run_id", "training_seed", "arm", "N", "student_index")
    for field in homogeneous_fields:
        values = {record[field] for record in copied}
        require(len(values) == 1, f"run ledger is heterogeneous in {field}")
    require([record["event_index"] for record in copied] == list(range(len(copied))),
            "event_index is not zero-based, contiguous, and file ordered")
    order_keys = [
        (record["outer_cycle"], record["within_cycle_group_index"])
        for record in copied
    ]
    require(order_keys == sorted(order_keys) and len(order_keys) == len(set(order_keys)),
            "input rows are not strictly ordered by cycle/group index")

    cycle_groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in copied:
        cycle_groups[record["outer_cycle"]].append(record)
    cycles = sorted(cycle_groups)
    require(cycles == list(range(len(cycles))),
            "outer_cycle is not zero-based contiguous")
    for cycle, group in cycle_groups.items():
        indices = [record["within_cycle_group_index"] for record in group]
        require(indices == list(range(len(group))),
                f"within-cycle indices are not ordered and contiguous at cycle {cycle}")

    _validate_cycle_update_ledger(copied)
    _validate_snapshot_chains(copied)
    enriched: list[dict[str, Any]] = []
    for record in copied:
        value = dict(record)
        if record["current_trials"] == record["N"]:
            value["_target"] = realized_activity(
                record["current_successes"], record["N"]
            )
        else:
            value["_target"] = None
        if record["arm"] == "frontier":
            value["_prediction"] = expected_activity(
                record["pre_successes"],
                record["pre_trials"],
                record["N"],
                record["prior_alpha"],
                record["prior_beta"],
            )
        else:
            value["_prediction"] = record["pre_score"]
        enriched.append(value)
    return enriched


def load_events(path: Path) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"unsafe or missing JSONL: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise TelemetryError(f"cannot read telemetry JSONL: {path}") from exc
    require(lines and all(line.strip() for line in lines),
            "telemetry JSONL is empty or contains blank lines")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        value = parse_json(line, f"telemetry JSONL line {index}")
        require(isinstance(value, dict), f"telemetry line {index} must be an object")
        records.append(value)
    return records


def _bin_index(prediction: float) -> int:
    require(_is_finite_number(prediction),
            "calibration prediction must be a finite number")
    prediction = float(prediction)
    require(0.0 <= prediction <= 1.0,
            "calibration prediction outside [0,1]")
    if prediction == 1.0:
        return len(BIN_EDGES) - 2
    index = bisect.bisect_right(BIN_EDGES, prediction) - 1
    require(0 <= index < len(BIN_EDGES) - 1, "calibration bin lookup failed")
    return index


def _calibration_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [
        record for record in records
        if record["arm"] == "frontier" and record["_target"] is not None
    ]
    bins: list[list[Mapping[str, Any]]] = [
        [] for _ in range(len(BIN_EDGES) - 1)
    ]
    for record in eligible:
        bins[_bin_index(float(record["_prediction"]))].append(record)
    bin_rows: list[dict[str, Any]] = []
    for index, group in enumerate(bins):
        predictions = [float(record["_prediction"]) for record in group]
        targets = [float(record["_target"]) for record in group]
        bin_rows.append({
            "lower": BIN_EDGES[index],
            "upper": BIN_EDGES[index + 1],
            "upper_inclusive": index == len(bins) - 1,
            "count": len(group),
            "mean_prediction": statistics.fmean(predictions) if group else None,
            "mean_target": statistics.fmean(targets) if group else None,
            "absolute_gap": (
                abs(statistics.fmean(targets) - statistics.fmean(predictions))
                if group else None
            ),
        })
    if not eligible:
        return {
            "count": 0,
            "mean_prediction": None,
            "mean_target": None,
            "signed_bias": None,
            "mean_absolute_error": None,
            "mean_squared_error": None,
            "root_mean_squared_error": None,
            "fixed_bin_ece": None,
            "fixed_bin_mce": None,
            "bins": bin_rows,
        }
    predictions = [float(record["_prediction"]) for record in eligible]
    targets = [float(record["_target"]) for record in eligible]
    errors = [target - prediction for target, prediction in zip(targets, predictions)]
    mse = statistics.fmean(error * error for error in errors)
    nonempty_gaps = [row["absolute_gap"] for row in bin_rows if row["count"]]
    ece = sum(
        row["count"] / len(eligible) * float(row["absolute_gap"])
        for row in bin_rows if row["count"]
    )
    return {
        "count": len(eligible),
        "mean_prediction": statistics.fmean(predictions),
        "mean_target": statistics.fmean(targets),
        "signed_bias": statistics.fmean(errors),
        "mean_absolute_error": statistics.fmean(abs(error) for error in errors),
        "mean_squared_error": mse,
        "root_mean_squared_error": math.sqrt(mse),
        "fixed_bin_ece": ece,
        "fixed_bin_mce": max(nonempty_gaps),
        "bins": bin_rows,
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = ((start + 1) + end) / 2.0
        for offset in range(start, end):
            ranks[order[offset]] = average
        start = end
    return ranks


def spearman(values: Sequence[float], targets: Sequence[float]) -> float | None:
    require(len(values) == len(targets), "Spearman inputs have unequal lengths")
    if len(values) < 2:
        return None
    x = _average_ranks(values)
    y = _average_ranks(targets)
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    centered_x = [value - mean_x for value in x]
    centered_y = [value - mean_y for value in y]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0.0:
        return None
    return sum(a * b for a, b in zip(centered_x, centered_y)) / denominator


def _delivery_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [record for record in records if record["_target"] is not None]
    if not eligible:
        return {
            "count": 0,
            "mean_realized_activity": None,
            "mixed_group_rate": None,
            "mean_success_fraction": None,
        }
    return {
        "count": len(eligible),
        "mean_realized_activity": statistics.fmean(
            float(record["_target"]) for record in eligible
        ),
        "mixed_group_rate": statistics.fmean(
            0.0 < record["current_successes"] < record["N"]
            for record in eligible
        ),
        "mean_success_fraction": statistics.fmean(
            record["current_successes"] / record["N"] for record in eligible
        ),
    }


def analyze_events(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate and summarize one run without accessing an endpoint."""
    validated = validate_events(records)
    first = validated[0]
    arm = first["arm"]
    complete = [record for record in validated if record["_target"] is not None]
    replay_scored = [
        record for record in complete
        if record["selection_source"] == "replay"
        and record["_prediction"] is not None
    ]
    score_values = [float(record["_prediction"]) for record in replay_scored]
    targets = [float(record["_target"]) for record in replay_scored]
    calibration = None
    if arm == "frontier":
        calibration = {
            "all_complete_adaptively_purchased_groups":
                _calibration_summary(validated),
            "new_or_mutation": _calibration_summary([
                record for record in validated
                if record["selection_source"] in {"new", "mutation"}
            ]),
            "replay": _calibration_summary([
                record for record in validated
                if record["selection_source"] == "replay"
            ]),
        }
    per_cycle = Counter(record["outer_cycle"] for record in validated)
    return {
        "schema": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "DRAFT_ENGINEERING_DIAGNOSTIC_ONLY",
        "run_id": first["run_id"],
        "arm": arm,
        "training_seed": first["training_seed"],
        "independent_unit": "training_seed",
        "N": first["N"],
        "population_scope": "adaptively_purchased_groups_not_all_buffer_or_tasks",
        "record_count": len(validated),
        "complete_group_count": len(complete),
        "outer_cycles": sorted(per_cycle),
        "attempted_groups_per_cycle": {
            str(cycle): per_cycle[cycle] for cycle in sorted(per_cycle)
        },
        "selection_source_counts": dict(sorted(Counter(
            record["selection_source"] for record in validated
        ).items())),
        "disposition_counts": dict(sorted(Counter(
            record["disposition"] for record in validated
        ).items())),
        "delivery": _delivery_summary(validated),
        "pre_group_discrimination": {
            "population": "complete replay groups with finite pre-group score",
            "count": len(replay_scored),
            "spearman_score_vs_realized_activity": spearman(score_values, targets),
            "maxmc_is_probability_forecast": False if arm == "maxmc" else None,
        },
        "calibration": calibration,
        "calibration_authorized": arm == "frontier",
        "group_level_inference_authorized": False,
        "production_authorized": False,
        "endpoint_accessed": False,
        "paper_evidence": False,
    }


def _revalidate_comparator_input(
    value: Mapping[str, Any], label: str
) -> ValidatedPackageResult:
    """Reopen the exact externally digest-bound package behind a result."""
    require(type(value) is ValidatedPackageResult,
            f"{label} comparator input is not an immutable validated result")
    require(value._seal_is_valid(),
            f"{label} immutable validated-result seal drift")
    request = value._validation_request
    require(isinstance(request, Mapping),
            f"{label} validation request is missing")
    expected_keys = {
        "package_root",
        "expected_sha256sums_sha256",
        "campaign_contract_path",
        "expected_campaign_contract_sha256",
        "expected_analyzer_sha256",
    }
    require(set(request) == expected_keys,
            f"{label} validation request shape drift")
    for field in expected_keys:
        require(isinstance(request[field], str) and request[field],
                f"{label} validation request field drift: {field}")
    fresh = validate_package(
        Path(request["package_root"]),
        request["expected_sha256sums_sha256"],
        Path(request["campaign_contract_path"]),
        request["expected_campaign_contract_sha256"],
        request["expected_analyzer_sha256"],
    )
    require(_thaw_json(fresh) == _thaw_json(value),
            f"{label} result does not equal its revalidated closed package")
    return fresh


def compare_matched_runs(
    frontier_result: Mapping[str, Any], maxmc_result: Mapping[str, Any]
) -> dict[str, Any]:
    """Revalidate and compare exact closed, group-matched run packages."""
    frontier_result = _revalidate_comparator_input(frontier_result, "Frontier")
    maxmc_result = _revalidate_comparator_input(maxmc_result, "MaxMC")
    require(frontier_result.get("arm") == "frontier", "first result is not Frontier")
    require(maxmc_result.get("arm") == "maxmc", "second result is not MaxMC")
    require(frontier_result.get("package_validated") is True,
            "Frontier validated-result seal drift")
    require(maxmc_result.get("package_validated") is True,
            "MaxMC validated-result seal drift")
    for field in ("training_seed", "N"):
        require(frontier_result.get(field) == maxmc_result.get(field),
                f"matched comparator drift: {field}")
    common_contract_fields = {
        "campaign_id",
        "campaign_contract_sha256",
        "source_bundle_manifest_sha256",
        "applied_overlay_manifest_sha256",
        "telemetry_overlay_sha256",
        "telemetry_writer_sha256",
        "training_driver_sha256",
        "environment_manifest_sha256",
        "scheduler_script_sha256",
        "N",
        "n_eval",
        "n_parallel",
        "n_rollout_steps",
        "target_student_ppo_updates",
        "max_outer_cycles",
        "ppo_epochs",
        "ppo_minibatches",
    }
    frontier_contract = frontier_result.get("matched_run_contract")
    maxmc_contract = maxmc_result.get("matched_run_contract")
    require(isinstance(frontier_contract, Mapping),
            "Frontier matched-run contract is missing")
    require(isinstance(maxmc_contract, Mapping),
            "MaxMC matched-run contract is missing")
    require(set(frontier_contract) == common_contract_fields | {"config_sha256"},
            "Frontier matched-run contract shape drift")
    require(set(maxmc_contract) == common_contract_fields | {"config_sha256"},
            "MaxMC matched-run contract shape drift")
    # Compare the scientific budget first so a target/cap drift is reported as
    # such even though changing a frozen campaign also changes its file digest.
    for field in ("target_student_ppo_updates", "max_outer_cycles"):
        require(frontier_contract[field] == maxmc_contract[field],
                f"matched run-contract drift: {field}")
    for field in sorted(
        common_contract_fields
        - {"target_student_ppo_updates", "max_outer_cycles"}
    ):
        require(frontier_contract[field] == maxmc_contract[field],
                f"matched run-contract drift: {field}")
    require(
        frontier_contract["config_sha256"] != maxmc_contract["config_sha256"],
        "Frontier and MaxMC comparator configs must be distinct and campaign-bound",
    )
    exposure_fields = {
        "outer_cycle_count",
        "upstream_n_iters",
        "student_ppo_updates",
        "upstream_n_updates",
        "upstream_n_grad_updates",
        "optimizer_step_applications",
        "student_training_transition_count",
        "attempted_group_count",
        "complete_group_count",
    }
    frontier_exposure = frontier_result.get("realized_exposure")
    maxmc_exposure = maxmc_result.get("realized_exposure")
    require(
        isinstance(frontier_exposure, Mapping)
        and set(frontier_exposure) == exposure_fields,
        "Frontier realized exposure shape drift",
    )
    require(
        isinstance(maxmc_exposure, Mapping)
        and set(maxmc_exposure) == exposure_fields,
        "MaxMC realized exposure shape drift",
    )
    matched_budget = {
        field: frontier_contract[field] for field in CAMPAIGN_BUDGET_KEYS
    }
    _validate_campaign_budget(matched_budget, "matched comparator")
    for label, exposure in (
        ("Frontier", frontier_exposure), ("MaxMC", maxmc_exposure)
    ):
        _validate_realized_counters(
            {field: exposure[field] for field in REALIZED_COUNTER_KEYS},
            matched_budget,
            terminal=True,
            label=f"{label} matched exposure",
        )
        expected_attempted_groups = _bounded_product(
            (exposure["outer_cycle_count"], matched_budget["n_parallel"]),
            f"{label} matched attempted-group count",
        )
        require(
            _is_int(exposure["attempted_group_count"])
            and exposure["attempted_group_count"]
            == expected_attempted_groups,
            f"{label} matched attempted-group exposure drift",
        )
        require(
            _is_int(exposure["complete_group_count"])
            and 0 <= exposure["complete_group_count"]
            <= exposure["attempted_group_count"],
            f"{label} matched complete-group exposure drift",
        )
    exposure_deltas = {
        field: int(frontier_exposure[field]) - int(maxmc_exposure[field])
        for field in sorted(exposure_fields)
    }
    frontier_delivery = frontier_result["delivery"]
    maxmc_delivery = maxmc_result["delivery"]
    deltas: dict[str, float | None] = {}
    for field in (
        "mean_realized_activity",
        "mixed_group_rate",
        "mean_success_fraction",
    ):
        left = frontier_delivery[field]
        right = maxmc_delivery[field]
        deltas[field] = None if left is None else float(left) - float(right)
    return {
        "schema": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "DRAFT_MATCHED_DELIVERY_DIAGNOSTIC_ONLY",
        "training_seed": frontier_result["training_seed"],
        "independent_unit": "training_seed",
        "N": frontier_result["N"],
        "paired_exposure": {
            "frontier": dict(frontier_exposure),
            "maxmc": dict(maxmc_exposure),
            "frontier_minus_maxmc": exposure_deltas,
        },
        "frontier_minus_maxmc_delivery": deltas,
        "frontier_pre_group_spearman": frontier_result[
            "pre_group_discrimination"
        ]["spearman_score_vs_realized_activity"],
        "maxmc_pre_group_spearman": maxmc_result[
            "pre_group_discrimination"
        ]["spearman_score_vs_realized_activity"],
        "maxmc_proper_calibration_authorized": False,
        "matched_level_causal_effect": False,
        "performance_or_ood_claim_authorized": False,
        "production_authorized": False,
        "paper_evidence": False,
    }


def _validate_receipt_static(
    receipt: Mapping[str, Any],
    campaign: Mapping[str, Any],
    expected_campaign_sha256: str,
    expected_analyzer_sha256: str,
) -> None:
    """Validate receipt identity/provenance before parsing numeric events."""
    _require_exact_keys(receipt, RECEIPT_KEYS, "telemetry receipt")
    require(receipt["schema"] == 1, "receipt schema drift")
    require(receipt["protocol_id"] == PROTOCOL_ID, "receipt protocol drift")
    require(receipt["purpose"] == PURPOSE, "receipt purpose drift")
    require(receipt["status"] == "complete", "receipt is not complete")
    require(receipt["campaign_id"] == campaign["campaign_id"],
            "receipt campaign identity drift")
    require(receipt["campaign_contract_sha256"] == expected_campaign_sha256,
            "receipt campaign-contract hash drift")
    require(receipt["endpoint_class"] == "calibration_telemetry_engineering_draft",
            "receipt endpoint-class drift")
    for field in ("production_authorized", "endpoint_accessed", "paper_evidence"):
        require(receipt[field] is False, f"receipt {field} must be false")
    require(receipt["from_last_checkpoint"] is False,
            "resumed telemetry run is forbidden")
    require(receipt["closed_before_analysis"] is True,
            "telemetry was not closed before analysis")
    require(
        isinstance(receipt["run_id"], str)
        and RUN_ID_RE.fullmatch(receipt["run_id"]) is not None,
        "receipt run_id drift",
    )
    require(isinstance(receipt["arm"], str)
            and receipt["arm"] in ALLOWED_ARMS,
            "receipt arm drift")
    for field in (
        "training_seed",
        "N",
        "n_eval",
        "n_parallel",
        "n_rollout_steps",
        "upstream_n_iters",
        "student_ppo_updates",
        "upstream_n_updates",
        "upstream_n_grad_updates",
        "ppo_epochs",
        "ppo_minibatches",
        "optimizer_step_applications",
        "student_training_transition_count",
        "telemetry_records",
        "attempted_group_count",
        "complete_group_count",
        "outer_cycle_count",
        "terminal_outer_cycle",
    ):
        require(_is_counter(receipt[field]),
                f"receipt {field} must be a nonnegative integer")
    campaign_budget = campaign["budget"]
    for field in (
        "N", "n_eval", "n_parallel", "n_rollout_steps",
        "ppo_epochs", "ppo_minibatches",
    ):
        require(receipt[field] == campaign_budget[field],
                f"receipt {field} does not match frozen campaign")
    _validate_realized_counters(
        {field: receipt[field] for field in REALIZED_COUNTER_KEYS},
        campaign_budget,
        terminal=True,
        label="receipt realized",
    )
    require(receipt["terminal_outer_cycle"] == receipt["outer_cycle_count"] - 1,
            "receipt terminal/cycle equation drift")
    require(receipt["telemetry_records"] > 0,
            "zero-record closed telemetry package is forbidden")
    require(receipt["attempted_group_count"] > 0,
            "zero-attempt closed telemetry package is forbidden")
    require(0 <= receipt["complete_group_count"] <= receipt["attempted_group_count"],
            "receipt complete-group count is impossible")

    arm_spec = campaign["arms"][receipt["arm"]]
    require(receipt["run_id"] == arm_spec["run_id"],
            "receipt run_id does not match campaign arm")
    require(receipt["training_seed"] == arm_spec["training_seed"],
            "receipt training seed does not match campaign arm")
    provenance = receipt["provenance"]
    require(isinstance(provenance, Mapping), "receipt provenance must be an object")
    _require_exact_keys(provenance, PROVENANCE_KEYS, "receipt provenance")
    require(provenance["base_commit"] == BASE_COMMIT, "receipt base commit drift")
    require(provenance["base_tree"] == BASE_TREE, "receipt base tree drift")
    require(provenance["v4_contract_sha256"] == V4_CONTRACT_SHA256,
            "receipt v4 dependency drift")
    require(provenance["protocol_sha256"] == PROTOCOL_SHA256,
            "receipt protocol hash drift")
    require(provenance["protocol_sha256"] == campaign["protocol"]["sha256"],
            "receipt/campaign protocol hash drift")
    require(provenance["analyzer_sha256"] == expected_analyzer_sha256,
            "receipt analyzer hash drift")
    require(provenance["analyzer_sha256"] == campaign["analyzer"]["sha256"],
            "receipt/campaign analyzer hash drift")
    require(provenance["campaign_contract_sha256"] == expected_campaign_sha256,
            "receipt provenance campaign-contract hash drift")
    require(provenance["config_sha256"] == arm_spec["config"]["sha256"],
            "receipt config hash does not match campaign arm")
    for name in COMMON_ARTIFACT_KEYS:
        provenance_field = f"{name}_sha256"
        require(
            provenance[provenance_field]
            == campaign["common_artifacts"][name]["sha256"],
            f"receipt {name} hash does not match campaign",
        )
    for field, value in provenance.items():
        if field not in {"base_commit", "base_tree"}:
            _require_hash(value, f"receipt provenance {field}")
    frozen_hashes = set(PROTECTED_HASHES.values()) | {PROTOCOL_SHA256}
    require(provenance["telemetry_overlay_sha256"] not in frozen_hashes,
            "telemetry overlay is not separately versioned from frozen artifacts")

    counters = receipt["integrity_counters"]
    require(isinstance(counters, Mapping), "integrity counters must be an object")
    _require_exact_keys(counters, COUNTER_KEYS, "integrity counters")
    for field, value in counters.items():
        require(_is_counter(value),
                f"integrity counter {field} must be nonnegative integer")
    require(counters["duplicate_event_id_count"] == 0,
            "duplicate event IDs are forbidden")
    require(counters["duplicate_new_group_count"] == 0,
            "duplicate-new groups violate the clean telemetry gate")
    require(counters["partial_group_count"] == 0,
            "partial groups violate the clean telemetry gate")
    require(counters["nonfinite_record_count"] == 0,
            "nonfinite records violate the clean telemetry gate")


def _validate_receipt(
    receipt: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    campaign: Mapping[str, Any],
    expected_campaign_sha256: str,
    expected_analyzer_sha256: str,
) -> None:
    _validate_receipt_static(
        receipt, campaign, expected_campaign_sha256, expected_analyzer_sha256
    )
    first = records[0]
    for field in ("run_id", "arm", "training_seed", "N"):
        require(receipt[field] == first[field], f"receipt {field} drift")
    cycles = sorted({record["outer_cycle"] for record in records})
    complete_count = sum(record["current_trials"] == record["N"] for record in records)
    require(receipt["telemetry_records"] == len(records), "receipt record-count drift")
    require(receipt["attempted_group_count"] == len(records),
            "receipt attempted-group count drift")
    require(receipt["complete_group_count"] == complete_count,
            "receipt complete-group count drift")
    require(receipt["outer_cycle_count"] == len(cycles),
            "receipt outer-cycle count drift")
    require(receipt["upstream_n_iters"] == len(cycles),
            "receipt upstream n_iters/event-cycle drift")
    require(receipt["terminal_outer_cycle"] == cycles[-1],
            "receipt terminal-cycle drift")
    counts_by_cycle = Counter(record["outer_cycle"] for record in records)
    require(all(count == receipt["n_parallel"] for count in counts_by_cycle.values()),
            "attempted-group ledger is incomplete within an outer cycle")
    expected_attempted_groups = _bounded_product(
        (receipt["outer_cycle_count"], receipt["n_parallel"]),
        "receipt attempted-group count",
    )
    require(
        receipt["attempted_group_count"] == expected_attempted_groups,
        "purchased-group ledger omits an outer-cycle group",
    )
    expected_transitions = _bounded_product(
        (
            len(cycles),
            receipt["n_parallel"],
            receipt["N"],
            receipt["n_rollout_steps"],
        ),
        "receipt event-ledger transition count",
    )
    require(receipt["student_training_transition_count"] == expected_transitions,
            "receipt training-transition accounting drift")

    cycle_representatives = [
        next(record for record in records if record["outer_cycle"] == cycle)
        for cycle in cycles
    ]
    target_updates = campaign["budget"]["target_student_ppo_updates"]
    require(
        all(
            record["post_upstream_n_updates"] < target_updates
            for record in cycle_representatives[:-1]
        ),
        "a preterminal cycle already reached the update target",
    )
    terminal = cycle_representatives[-1]
    require(
        terminal["runner_branch"] == "replay"
        and terminal["pre_upstream_n_updates"] == target_updates - 1
        and terminal["post_upstream_n_updates"] == target_updates,
        "terminal cycle is not the target-1 to target replay update",
    )
    require(
        receipt["outer_cycle_count"] == terminal["post_upstream_n_iters"]
        and receipt["upstream_n_iters"] == terminal["post_upstream_n_iters"]
        and receipt["student_ppo_updates"]
            == terminal["post_upstream_n_updates"]
        and receipt["upstream_n_updates"]
            == terminal["post_upstream_n_updates"]
        and receipt["upstream_n_grad_updates"]
            == terminal["post_upstream_n_grad_updates"]
        and receipt["optimizer_step_applications"]
            == terminal["post_optimizer_step_applications"],
        "receipt totals do not equal terminal cycle post counters",
    )

    counters = receipt["integrity_counters"]
    new_identity_counts = Counter(
        (record["outer_cycle"], record["level_sha256"])
        for record in records
        if record["selection_source"] in {"new", "mutation"}
    )
    derived_duplicate_new = sum(
        max(0, count - 1) for count in new_identity_counts.values()
    )
    labeled_duplicate_new = sum(
        record["disposition"] == "duplicate_new_rejected" for record in records
    )
    derived_partial = len(records) - complete_count
    repeated = sum(
        max(0, count - 1)
        for count in Counter(
            (record["outer_cycle"], record["level_sha256"]) for record in records
        ).values()
    )
    require(labeled_duplicate_new == derived_duplicate_new,
            "duplicate-new disposition does not match canonical derivation")
    require(counters["duplicate_new_group_count"] == derived_duplicate_new == 0,
            "duplicate-new groups violate the clean telemetry gate")
    require(counters["partial_group_count"] == derived_partial == 0,
            "partial groups violate the clean telemetry gate")
    require(counters["repeated_level_same_batch_count"] == repeated,
            "repeated-level same-batch counter drift")


def validate_package(
    root: Path,
    expected_sha256sums_sha256: str,
    campaign_contract_path: Path,
    expected_campaign_contract_sha256: str,
    expected_analyzer_sha256: str,
) -> ValidatedPackageResult:
    """Validate a closed telemetry package, then compute its diagnostics."""
    repository_preflight(expected_analyzer_sha256)
    campaign = validate_campaign_contract(
        campaign_contract_path,
        expected_campaign_contract_sha256,
        expected_analyzer_sha256,
    )
    require(root.is_absolute(), "package path must be absolute")
    require(root.is_dir() and not root.is_symlink(), f"unsafe package root: {root}")
    actual_names = {path.name for path in root.iterdir()}
    require(actual_names == PACKAGE_FILES,
            f"package file closure drift: {sorted(actual_names)}")
    _require_hash(expected_sha256sums_sha256, "expected SHA256SUMS digest")
    manifest_path = root / "telemetry-SHA256SUMS"
    require(sha256(manifest_path) == expected_sha256sums_sha256,
            "telemetry SHA256SUMS digest mismatch")
    listed: dict[str, str] = {}
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise TelemetryError("cannot read telemetry SHA256SUMS") from exc
    for line in lines:
        match = MANIFEST_LINE_RE.fullmatch(line)
        require(match is not None, "unsafe telemetry SHA256SUMS line")
        digest, name = match.groups()
        require(name not in listed, "duplicate telemetry manifest entry")
        listed[name] = digest
    require(set(listed) == PACKAGE_PAYLOADS, "telemetry payload closure drift")
    for name, expected in listed.items():
        require(sha256(root / name) == expected, f"telemetry payload hash drift: {name}")

    receipt = load_json(root / "telemetry-receipt.json", "telemetry receipt")
    _validate_receipt_static(
        receipt,
        campaign,
        expected_campaign_contract_sha256,
        expected_analyzer_sha256,
    )
    complete = load_json(root / "telemetry-COMPLETE", "telemetry COMPLETE")
    _require_exact_keys(
        complete,
        {"schema", "status", "run_id", "arm", "sha256sums_sha256", "file_count"},
        "telemetry COMPLETE",
    )
    require(complete == {
        "schema": 1,
        "status": "complete",
        "run_id": receipt.get("run_id"),
        "arm": receipt.get("arm"),
        "sha256sums_sha256": expected_sha256sums_sha256,
        "file_count": len(PACKAGE_PAYLOADS),
    }, "telemetry COMPLETE binding drift")

    raw_records = load_events(root / "telemetry-events.jsonl")
    validated = validate_events(raw_records)
    _validate_receipt(
        receipt,
        validated,
        campaign,
        expected_campaign_contract_sha256,
        expected_analyzer_sha256,
    )
    # Analyze the exact closed records, not the internal validation copies that
    # carry underscore-prefixed derived values.
    result = analyze_events(raw_records)
    result["package_sha256sums_sha256"] = expected_sha256sums_sha256
    result["receipt_sha256"] = sha256(root / "telemetry-receipt.json")
    result["events_sha256"] = sha256(root / "telemetry-events.jsonl")
    result["analyzer_sha256"] = sha256(Path(__file__).resolve())
    result["protocol_sha256"] = PROTOCOL_SHA256
    result["campaign_id"] = campaign["campaign_id"]
    result["campaign_contract_sha256"] = expected_campaign_contract_sha256
    provenance = receipt["provenance"]
    result["package_validated"] = True
    result["matched_run_contract"] = {
        "campaign_id": campaign["campaign_id"],
        "campaign_contract_sha256": provenance["campaign_contract_sha256"],
        "config_sha256": provenance["config_sha256"],
        "source_bundle_manifest_sha256":
            provenance["source_bundle_manifest_sha256"],
        "applied_overlay_manifest_sha256":
            provenance["applied_overlay_manifest_sha256"],
        "telemetry_overlay_sha256": provenance["telemetry_overlay_sha256"],
        "telemetry_writer_sha256": provenance["telemetry_writer_sha256"],
        "training_driver_sha256": provenance["training_driver_sha256"],
        "environment_manifest_sha256": provenance["environment_manifest_sha256"],
        "scheduler_script_sha256": provenance["scheduler_script_sha256"],
        "N": receipt["N"],
        "n_eval": receipt["n_eval"],
        "n_parallel": receipt["n_parallel"],
        "n_rollout_steps": receipt["n_rollout_steps"],
        "target_student_ppo_updates":
            campaign["budget"]["target_student_ppo_updates"],
        "max_outer_cycles": campaign["budget"]["max_outer_cycles"],
        "ppo_epochs": receipt["ppo_epochs"],
        "ppo_minibatches": receipt["ppo_minibatches"],
    }
    result["realized_exposure"] = {
        "outer_cycle_count": receipt["outer_cycle_count"],
        "upstream_n_iters": receipt["upstream_n_iters"],
        "student_ppo_updates": receipt["student_ppo_updates"],
        "upstream_n_updates": receipt["upstream_n_updates"],
        "upstream_n_grad_updates": receipt["upstream_n_grad_updates"],
        "optimizer_step_applications": receipt["optimizer_step_applications"],
        "student_training_transition_count":
            receipt["student_training_transition_count"],
        "attempted_group_count": receipt["attempted_group_count"],
        "complete_group_count": receipt["complete_group_count"],
    }
    validation_request = {
        "package_root": str(root.resolve()),
        "expected_sha256sums_sha256": expected_sha256sums_sha256,
        "campaign_contract_path": str(campaign_contract_path.resolve()),
        "expected_campaign_contract_sha256": expected_campaign_contract_sha256,
        "expected_analyzer_sha256": expected_analyzer_sha256,
    }
    return ValidatedPackageResult(
        result, validation_request, _VALIDATED_RESULT_TOKEN
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _thaw_json(value),
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--package", type=Path)
    parser.add_argument("--expected-sha256sums-sha256")
    parser.add_argument("--campaign-contract", type=Path)
    parser.add_argument("--expected-campaign-contract-sha256")
    parser.add_argument("--expected-analyzer-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    require(args.expected_analyzer_sha256 is not None,
            "--expected-analyzer-sha256 is required")
    if args.preflight:
        require(
            args.package is None
            and args.expected_sha256sums_sha256 is None
            and args.campaign_contract is None
            and args.expected_campaign_contract_sha256 is None,
                "--preflight does not accept a package")
        result = repository_preflight(args.expected_analyzer_sha256)
    else:
        require(args.package is not None, "--package is required")
        require(args.expected_sha256sums_sha256 is not None,
                "--expected-sha256sums-sha256 is required")
        require(args.campaign_contract is not None,
                "--campaign-contract is required")
        require(args.expected_campaign_contract_sha256 is not None,
                "--expected-campaign-contract-sha256 is required")
        result = validate_package(
            args.package.resolve(),
            args.expected_sha256sums_sha256,
            args.campaign_contract.resolve(),
            args.expected_campaign_contract_sha256,
            args.expected_analyzer_sha256,
        )
    payload = _canonical_json(result)
    if args.output is None:
        print(payload, end="")
    else:
        output = args.output.resolve()
        require(output.parent.is_dir() and not output.parent.is_symlink(),
                "unsafe output parent")
        require(not output.exists() and not output.is_symlink(),
                "refusing to overwrite analysis output")
        output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TelemetryError as exc:
        raise SystemExit(f"calibration telemetry validation failed: {exc}")
