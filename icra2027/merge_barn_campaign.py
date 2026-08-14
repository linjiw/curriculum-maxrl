"""Strict, outcome-blind merger for per-seed BARN evidence artifacts.

The merger treats the frozen machine-readable protocol as authoritative.  It
validates identities, hashes, the complete campaign cell, execution order,
budget accounting, and every held-out/training record before copying rows into
seed order.  Validation recomputes only internal consistency checks; no result
or endpoint is printed and no comparison between outcomes is made here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DOMAIN = "barn_gazebo_cpu_navigation"
PRIMARY_ARMS = ("ours_uN", "uniform", "learnability", "staged")
ABLATION_ARMS = ("ours_uN", "learnability")
ARM_NAMES = PRIMARY_ARMS  # Backwards-compatible public name.
PRIMARY_EVIDENCE_STATUS = "full_barn_campaign"
ABLATION_EVIDENCE_STATUS = "full_barn_n_ablation"
FULL_EVIDENCE_STATUS = PRIMARY_EVIDENCE_STATUS
CAMPAIGN_CELLS = (
    "primary", "ablation_n2", "ablation_n4", "ablation_n8", "ablation_n16")
EVALUATION_STATUSES = frozenset({"succeeded", "collided", "timeout"})

PROVENANCE_HASH_FIELDS = (
    "manifest_sha256",
    "split_sha256",
    "prereg_sha256",
    "analyzer_sha256",
    "protocol_sha256",
    "container_sha256",
    "source_sha256",
)

_SEED_CONFIG_FIELDS = {
    "seeds", "seed_start", "seed_list", "campaign_seed", "execution_order",
    "domain_id", "eval_domain_id", "master_port", "eval_master_port",
    "runtime_root",
}
_RUNTIME_CONFIG_FIELDS = (
    "domain_id", "eval_domain_id", "master_port", "eval_master_port",
    "runtime_root",
)
_PATH_PROVENANCE_FIELDS = {
    "manifest_path", "split_path", "prereg_path", "analyzer_path",
    "protocol_path", "dataset_root", "robot_sdf",
}
_AUC_FIELDS = {
    "target_uniform_auc_by_episode": "episodes",
    "target_uniform_auc_by_sim_step": "sim_steps",
    "target_uniform_auc_by_own_training_wall": "training_wall_seconds",
}
SELECTION_RULE = "earliest_submitted_complete_hash_valid_attempt_per_seed"
_SELECTION_RECEIPT_FIELDS = {
    "schema_version", "selection_rule", "outcome_blind", "campaign_id",
    "campaign_cell", "expected_seed_list", "expected_hashes",
    "ledger_sha256", "selected", "excluded",
}
_SELECTED_RECEIPT_FIELDS = {
    "seed", "attempt_id", "submitted_utc", "slurm_array_job_id",
    "slurm_array_task_id", "slurm_job_id", "artifact_path",
    "artifact_sha256",
}
_EXCLUDED_RECEIPT_FIELDS = {
    "seed", "attempt_id", "submitted_utc", "artifact_path",
    "artifact_complete", "reason",
}


class MergeValidationError(ValueError):
    """An artifact cannot safely enter the frozen evidence matrix."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(value: object, *, field: str, minimum: int = 0) -> int:
    if not _is_int(value) or value < minimum:
        raise MergeValidationError(
            f"{field} must be an integer >= {minimum}")
    return int(value)


def _require_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MergeValidationError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise MergeValidationError(f"{field} must be finite")
    return number


def _require_rate(value: object, *, field: str) -> float:
    rate = _require_number(value, field=field)
    if not 0.0 <= rate <= 1.0:
        raise MergeValidationError(f"{field} must be in [0, 1]")
    return rate


def _same_number(actual: object, expected: float, *, field: str) -> None:
    number = _require_number(actual, field=field)
    if not math.isclose(number, float(expected), rel_tol=1e-10, abs_tol=1e-12):
        raise MergeValidationError(f"{field} is internally inconsistent")


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise MergeValidationError(f"{field} must be a SHA-256 string")
    digest = value.lower()
    if (len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)):
        raise MergeValidationError(
            f"{field} must be a 64-character SHA-256 digest")
    return digest


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_expected_seeds(value: str) -> tuple[int, ...]:
    if not isinstance(value, str) or not value.strip():
        raise MergeValidationError("expected seed list must not be empty")
    pieces = value.split(",")
    if any(not piece.strip() for piece in pieces):
        raise MergeValidationError(
            "expected seed list contains an empty comma-separated item")
    seeds: list[int] = []
    for piece in pieces:
        try:
            seed = int(piece.strip(), 10)
        except ValueError as error:
            raise MergeValidationError(
                f"invalid expected seed {piece.strip()!r}") from error
        if seed < 0:
            raise MergeValidationError("expected seeds must be non-negative")
        seeds.append(seed)
    return _canonical_expected_seeds(seeds)


def _canonical_expected_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    if not seeds:
        raise MergeValidationError("expected seed list must not be empty")
    normalized = [
        _require_int(seed, field="expected seed") for seed in seeds]
    if len(normalized) != len(set(normalized)):
        raise MergeValidationError("expected seed list contains duplicates")
    return tuple(sorted(normalized))


def _json_constant_error(token: str) -> None:
    raise ValueError(f"non-finite JSON number {token}")


def load_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(), parse_constant=_json_constant_error)
    except (json.JSONDecodeError, ValueError) as error:
        raise MergeValidationError(f"invalid JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise MergeValidationError(f"{path} must contain a JSON object")
    return value


_load_json = load_json  # Compatibility for the attempt selector and old callers.


def _expected_hashes(**values: str) -> dict[str, str]:
    missing = sorted(set(PROVENANCE_HASH_FIELDS) - set(values))
    extra = sorted(set(values) - set(PROVENANCE_HASH_FIELDS))
    if missing or extra:
        raise MergeValidationError(
            f"expected hash fields differ: missing={missing}, extra={extra}")
    return {
        field: _require_sha256(values[field], field=f"expected {field}")
        for field in PROVENANCE_HASH_FIELDS
    }


def _protocol_contract(
    protocol: Mapping[str, Any],
    *,
    campaign_cell: str,
    expected_seeds: Sequence[int],
    expected_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Validate the frozen protocol and return one exact cell contract."""

    if not isinstance(protocol, dict) or protocol.get("schema_version") != 1:
        raise MergeValidationError("unsupported machine protocol schema")
    if protocol.get("status") != "FROZEN":
        raise MergeValidationError("machine protocol must be marked FROZEN")
    if protocol.get("domain") != DOMAIN:
        raise MergeValidationError("machine protocol has the wrong domain")
    protocol_id = protocol.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise MergeValidationError("machine protocol is missing protocol_id")
    if campaign_cell not in CAMPAIGN_CELLS:
        raise MergeValidationError(f"unknown campaign cell {campaign_cell!r}")

    dataset = protocol.get("dataset")
    environment = protocol.get("environment")
    shared = protocol.get("shared_training")
    if not all(isinstance(item, dict)
               for item in (dataset, environment, shared)):
        raise MergeValidationError(
            "machine protocol is missing dataset/environment/shared_training")
    exact_dataset = {
        "n_strata": 10, "n_train_courses": 240, "n_heldout_courses": 60}
    for field, expected in exact_dataset.items():
        if dataset.get(field) != expected:
            raise MergeValidationError(
                f"protocol dataset {field} must equal {expected}")
    for field in ("manifest_sha256", "split_sha256"):
        actual = _require_sha256(
            dataset.get(field), field=f"protocol dataset {field}")
        if not hmac.compare_digest(actual, expected_hashes[field]):
            raise MergeValidationError(
                f"protocol dataset {field} differs from expected hash")
    protocol_container = _require_sha256(
        environment.get("container_sha256"),
        field="protocol environment container_sha256")
    if not hmac.compare_digest(
            protocol_container, expected_hashes["container_sha256"]):
        raise MergeValidationError("protocol container hash differs")
    if environment.get("cpu_only") is not True:
        raise MergeValidationError("BARN evidence protocol must be CPU-only")
    analysis = protocol.get("analysis")
    if not isinstance(analysis, dict):
        raise MergeValidationError("machine protocol is missing analysis settings")
    analyzer_hash = _require_sha256(
        analysis.get("analyzer_sha256"),
        field="protocol analysis analyzer_sha256")
    if not hmac.compare_digest(
            analyzer_hash, expected_hashes["analyzer_sha256"]):
        raise MergeValidationError("protocol analyzer hash differs")
    requirements = protocol.get("artifact_requirements")
    exact_requirements = {
        "heldout_course_count": 60,
        "difficulty_bin_count": 10,
        "training_episode_records": True,
        "evaluation_episode_records": True,
        "per_checkpoint_status_counts": True,
        "per_checkpoint_teacher_vector_length": 10,
        "canonical_result_arm_order": True,
    }
    if not isinstance(requirements, dict) or any(
            requirements.get(field) != expected
            for field, expected in exact_requirements.items()):
        raise MergeValidationError("protocol artifact requirements differ")
    retry = protocol.get("retry")
    if (not isinstance(retry, dict)
            or retry.get("selection") !=
            "earliest_submitted_complete_hash_valid_attempt_per_seed"
            or retry.get("partial_attempts_retained") is not True
            or retry.get("endpoint_blind_selection") is not True):
        raise MergeValidationError("protocol retry rule differs")

    canonical_seeds = _canonical_expected_seeds(expected_seeds)
    protocol_seeds = shared.get("seeds")
    if (not isinstance(protocol_seeds, list)
            or tuple(protocol_seeds) != canonical_seeds):
        raise MergeValidationError(
            "expected seed list must exactly equal machine protocol seeds")

    if campaign_cell == "primary":
        cell = protocol.get("primary")
        canonical_arms = PRIMARY_ARMS
        status = PRIMARY_EVIDENCE_STATUS
        if not isinstance(cell, dict):
            raise MergeValidationError("protocol is missing primary cell")
        n_rollouts = cell.get("n_rollouts")
        if tuple(cell.get("arms", ())) != canonical_arms:
            raise MergeValidationError("protocol primary arms are not canonical")
        if cell.get("evidence_status") != status:
            raise MergeValidationError("protocol primary evidence_status differs")
    else:
        cell = protocol.get("ablation")
        match = re.fullmatch(r"ablation_n(2|4|8|16)", campaign_cell)
        if not isinstance(cell, dict) or match is None:
            raise MergeValidationError(f"protocol is missing {campaign_cell}")
        declared_cells = cell.get("fresh_cell_names")
        n_values = cell.get("n_values")
        n_rollouts = int(match.group(1))
        if (n_values != [2, 4, 8, 16]
                or cell.get("n8_source") !=
                "primary_ours_uN_and_learnability"
                or not isinstance(declared_cells, list)
                or campaign_cell not in declared_cells
                or n_rollouts not in n_values):
            raise MergeValidationError(
                f"campaign cell {campaign_cell!r} is not declared in protocol")
        canonical_arms = ABLATION_ARMS
        status = ABLATION_EVIDENCE_STATUS
        if tuple(cell.get("arms", ())) != canonical_arms:
            raise MergeValidationError("protocol ablation arms are not canonical")
        if cell.get("evidence_status") != status:
            raise MergeValidationError("protocol ablation evidence_status differs")
    _require_int(n_rollouts, field="protocol n_rollouts", minimum=2)

    order_map = cell.get("execution_order_by_seed")
    expected_keys = {str(seed) for seed in canonical_seeds}
    if not isinstance(order_map, dict) or set(order_map) != expected_keys:
        raise MergeValidationError(
            "protocol execution_order_by_seed must cover seeds exactly")
    normalized_orders: dict[str, list[str]] = {}
    for seed in canonical_seeds:
        order = order_map[str(seed)]
        if (not isinstance(order, list) or len(order) != len(canonical_arms)
                or len(set(order)) != len(order)
                or set(order) != set(canonical_arms)):
            raise MergeValidationError(
                f"protocol has invalid execution order for seed {seed}")
        normalized_orders[str(seed)] = list(order)

    expected_config = {
        "arms": list(canonical_arms),
        "campaign_cell": campaign_cell,
        "protocol_id": protocol_id,
        "steps": shared.get("max_training_updates"),
        "max_training_updates": shared.get("max_training_updates"),
        "n_rollouts": n_rollouts,
        "tasks_per_step": shared.get("tasks_per_step"),
        "eval_every": shared.get("eval_every"),
        "eval_episodes": shared.get("eval_episodes"),
        "training_sim_step_budget": shared.get("training_sim_step_budget"),
        "eval_sim_step_interval": shared.get("eval_sim_step_interval"),
        "episode_timeout": environment.get("episode_timeout"),
        "max_step_size": environment.get("max_step_size"),
        "real_time_update_rate": environment.get("real_time_update_rate"),
        "hindsight": False,
        "estimator": "maxrl",
        "teacher_gamma": shared.get("teacher_gamma"),
        "teacher_decay": shared.get("teacher_decay"),
        "teacher_floor": shared.get("teacher_floor"),
        "teacher_unit": "frozen_difficulty_stratum",
        "n_strata": dataset.get("n_strata"),
        "difficulty_metadata": (
            "published optimal traversal time seconds; longer is harder"),
        "staged_initial_strata": shared.get("staged_initial_strata"),
        "staged_promotion_threshold": shared.get("staged_promotion_threshold"),
        "staged_min_frontier_groups": shared.get("staged_min_frontier_groups"),
        "n_train_courses": dataset.get("n_train_courses"),
        "n_heldout_courses": dataset.get("n_heldout_courses"),
        "split_seed": dataset.get("split_seed"),
        "smoke": False,
        "engineering_course_id": None,
        "evaluation_partition": "frozen_heldout",
    }
    missing = sorted(
        field for field, value in expected_config.items()
        if value is None and field != "engineering_course_id")
    if missing:
        raise MergeValidationError(
            f"machine protocol contract is missing {missing}")
    for field in (
            "steps", "max_training_updates", "n_rollouts", "tasks_per_step",
            "eval_every", "eval_episodes", "training_sim_step_budget",
            "eval_sim_step_interval", "real_time_update_rate", "n_strata",
            "n_train_courses", "n_heldout_courses", "split_seed"):
        minimum = 0 if field == "split_seed" else 1
        _require_int(expected_config[field], field=f"protocol {field}",
                     minimum=minimum)
    isolation = protocol.get("isolation")
    if not isinstance(isolation, dict):
        raise MergeValidationError("machine protocol is missing isolation settings")
    seed_stride = _require_int(
        isolation.get("seed_stride"), field="protocol isolation seed_stride",
        minimum=1)
    eval_offset = _require_int(
        isolation.get("eval_offset"), field="protocol isolation eval_offset",
        minimum=1)
    port_stride = _require_int(
        isolation.get("master_port_seed_stride"),
        field="protocol isolation master_port_seed_stride", minimum=1)
    port_offset = _require_int(
        isolation.get("eval_master_port_offset"),
        field="protocol isolation eval_master_port_offset", minimum=1)
    domain_bases = isolation.get("domain_base_by_cell")
    port_bases = isolation.get("master_port_base_by_cell")
    if (not isinstance(domain_bases, dict) or campaign_cell not in domain_bases
            or not isinstance(port_bases, dict) or campaign_cell not in port_bases):
        raise MergeValidationError(
            f"protocol isolation does not declare {campaign_cell}")
    domain_base = _require_int(
        domain_bases[campaign_cell], field="protocol cell domain base")
    port_base = _require_int(
        port_bases[campaign_cell], field="protocol cell master port base",
        minimum=1024)
    expected_runtime_by_seed = {}
    for seed in canonical_seeds:
        domain_id = domain_base + seed_stride * seed
        master_port = port_base + port_stride * seed
        if domain_id + eval_offset > 232 or master_port + port_offset > 65535:
            raise MergeValidationError("protocol isolation values exceed valid bounds")
        expected_runtime_by_seed[str(seed)] = {
            "domain_id": domain_id,
            "eval_domain_id": domain_id + eval_offset,
            "master_port": master_port,
            "eval_master_port": master_port + port_offset,
        }
    return {
        "campaign_cell": campaign_cell,
        "arms": canonical_arms,
        "evidence_status": status,
        "seeds": canonical_seeds,
        "execution_order_by_seed": normalized_orders,
        "expected_config": expected_config,
        "n_strata": 10,
        "n_train_courses": 240,
        "n_heldout_courses": 60,
        "expected_runtime_by_seed": expected_runtime_by_seed,
    }


def _derived_seed(namespace: int, seed: int, component: int = 0) -> int:
    state = np.random.SeedSequence([
        int(namespace) & 0xFFFFFFFF,
        int(seed) & 0xFFFFFFFF,
        (int(seed) >> 32) & 0xFFFFFFFF,
        int(component) & 0xFFFFFFFF,
    ]).generate_state(1, dtype=np.uint32)[0]
    return int(state % np.uint32(2**31 - 1))


def _normalize_execution(execution: object, *, label: str) -> dict[str, Any]:
    if not isinstance(execution, dict):
        raise MergeValidationError(f"{label} is missing execution identity")
    required = {
        "campaign_id", "attempt_id", "submitted_utc", "slurm_job_id",
        "slurm_array_job_id", "slurm_array_task_id"}
    if set(execution) != required:
        raise MergeValidationError(
            f"{label} execution identity fields must be exact")
    for field in ("campaign_id", "attempt_id"):
        value = execution[field]
        if (not isinstance(value, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", value)
                is None):
            raise MergeValidationError(f"{label} has invalid {field}")
    for field in ("slurm_job_id", "slurm_array_job_id"):
        value = execution[field]
        if (not isinstance(value, str)
                or re.fullmatch(r"[0-9]+(?:_[0-9]+)?", value) is None):
            raise MergeValidationError(f"{label} has invalid {field}")
    _require_int(
        execution["slurm_array_task_id"],
        field=f"{label} slurm_array_task_id")
    value = execution["submitted_utc"]
    if not isinstance(value, str):
        raise MergeValidationError(f"{label} submitted_utc must be a string")
    try:
        submitted = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise MergeValidationError(
            f"{label} submitted_utc must be ISO-8601") from error
    if submitted.tzinfo is None or submitted.utcoffset() is None:
        raise MergeValidationError(
            f"{label} submitted_utc needs an explicit offset")
    normalized = dict(execution)
    normalized["submitted_utc"] = submitted.astimezone(timezone.utc).isoformat()
    return normalized


def _extract_seed(config: Mapping[str, Any], *, label: str) -> int:
    seed_list = config.get("seed_list")
    if (not isinstance(seed_list, list) or len(seed_list) != 1):
        raise MergeValidationError(
            f"{label} config seed_list must contain exactly one seed")
    seed = _require_int(seed_list[0], field=f"{label} seed")
    for field, expected in {
            "seeds": 1, "seed_start": seed, "campaign_seed": seed}.items():
        if config.get(field) != expected:
            raise MergeValidationError(
                f"{label} config {field} does not identify seed {seed}")
    return seed


def _validate_runtime_config(
    config: Mapping[str, Any], *, seed: int, label: str
) -> dict[str, Any]:
    runtime: dict[str, Any] = {}
    for field in _RUNTIME_CONFIG_FIELDS:
        if field not in config:
            raise MergeValidationError(f"{label} config is missing {field}")
        value = config[field]
        if field == "runtime_root":
            if (not isinstance(value, str) or not value
                    or not Path(value).is_absolute()):
                raise MergeValidationError(
                    f"{label} runtime_root must be a non-empty absolute path")
        else:
            value = _require_int(value, field=f"{label} config {field}")
        runtime[field] = copy.deepcopy(value)
    if not 0 <= runtime["domain_id"] <= 231:
        raise MergeValidationError(f"{label} seed {seed} ROS domain is out of range")
    if runtime["eval_domain_id"] != runtime["domain_id"] + 1:
        raise MergeValidationError(
            f"{label} seed {seed} eval_domain_id must equal domain_id + 1")
    if not 1024 <= runtime["master_port"] <= 65534:
        raise MergeValidationError(f"{label} seed {seed} master port is out of range")
    if runtime["eval_master_port"] != runtime["master_port"] + 1:
        raise MergeValidationError(
            f"{label} seed {seed} eval_master_port must equal master_port + 1")
    return runtime


def _validate_strata(
    config: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    panels: dict[str, list[str]] = {}
    ranges: dict[str, list[tuple[float, float]]] = {}
    for kind, total, per_stratum in (
            ("train", 240, 24), ("heldout", 60, 6)):
        rows = config.get(f"{kind}_strata")
        if not isinstance(rows, list) or len(rows) != 10:
            raise MergeValidationError(
                f"{label} config {kind}_strata must contain 10 strata")
        panel: list[str] = []
        difficulty_ranges: list[tuple[float, float]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or row.get("stratum") != index:
                raise MergeValidationError(
                    f"{label} {kind} stratum {index} is malformed")
            if row.get("n_courses") != per_stratum:
                raise MergeValidationError(
                    f"{label} {kind} stratum {index} has wrong course count")
            course_ids = row.get("course_ids")
            if (not isinstance(course_ids, list) or len(course_ids) != per_stratum
                    or len(set(course_ids)) != len(course_ids)
                    or not all(isinstance(item, str) and
                               re.fullmatch(r"barn-[0-2][0-9]{2}", item)
                               for item in course_ids)):
                raise MergeValidationError(
                    f"{label} {kind} stratum {index} has invalid course IDs")
            low = _require_number(
                row.get("difficulty_min"),
                field=f"{label} {kind} stratum {index} difficulty_min")
            high = _require_number(
                row.get("difficulty_max"),
                field=f"{label} {kind} stratum {index} difficulty_max")
            if low < 0.0 or high < low:
                raise MergeValidationError(
                    f"{label} {kind} stratum {index} has invalid difficulty range")
            if difficulty_ranges and low < difficulty_ranges[-1][1]:
                raise MergeValidationError(
                    f"{label} {kind} difficulty strata are not ordered")
            difficulty_ranges.append((low, high))
            panel.extend(course_ids)
        if len(panel) != total or len(set(panel)) != total:
            raise MergeValidationError(
                f"{label} {kind} panel is not an exact {total}-course panel")
        panels[kind] = panel
        ranges[kind] = difficulty_ranges
    if set(panels["train"]) & set(panels["heldout"]):
        raise MergeValidationError(f"{label} train and heldout panels overlap")
    expected_ids = {f"barn-{index:03d}" for index in range(300)}
    if set(panels["train"]) | set(panels["heldout"]) != expected_ids:
        raise MergeValidationError(
            f"{label} train/heldout panels do not cover BARN exactly")
    return {"train_ids": panels["train"],
            "heldout_ids": panels["heldout"],
            "heldout_ranges": ranges["heldout"]}


def _require_count_dict(
    value: object, *, field: str, expected: Counter[str]
) -> None:
    if not isinstance(value, dict) or any(
            not isinstance(key, str) or not _is_int(count) or count < 0
            for key, count in value.items()):
        raise MergeValidationError(f"{field} must be a string-to-count object")
    normalized = {key: count for key, count in sorted(expected.items()) if count}
    if value != normalized:
        raise MergeValidationError(f"{field} does not match episode records")


def _validate_teacher(
    teacher: object, *, n_strata: int, groups: int, label: str
) -> tuple[list[float], list[int]]:
    if not isinstance(teacher, dict):
        raise MergeValidationError(f"{label} is missing teacher diagnostics")
    posterior = teacher.get("posterior_mean")
    weights = teacher.get("sampling_weights_at_posterior_mean")
    visits = teacher.get("visits")
    if not all(isinstance(vector, list) and len(vector) == n_strata
               for vector in (posterior, weights, visits)):
        raise MergeValidationError(
            f"{label} teacher vectors must have length {n_strata}")
    posterior_values = [
        _require_rate(value, field=f"{label} teacher posterior")
        for value in posterior]
    weight_values = [
        _require_rate(value, field=f"{label} teacher weight")
        for value in weights]
    visit_values = [
        _require_int(value, field=f"{label} teacher visit") for value in visits]
    if not math.isclose(sum(weight_values), 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise MergeValidationError(f"{label} teacher weights do not sum to one")
    if sum(visit_values) != groups:
        raise MergeValidationError(
            f"{label} teacher visits do not match training groups")
    return posterior_values, visit_values


def _validate_eval(
    evaluation: object,
    *,
    panel: Mapping[str, Any],
    eval_seed: int,
    eval_episodes: int,
    posterior: Sequence[float],
    visits: Sequence[int],
    label: str,
) -> None:
    if not isinstance(evaluation, dict):
        raise MergeValidationError(f"{label} is missing eval")
    heldout_ids = panel["heldout_ids"]
    if evaluation.get("heldout_course_ids") != heldout_ids:
        raise MergeValidationError(f"{label} heldout panel/order differs")
    per_course = evaluation.get("per_course")
    episode_records = evaluation.get("episode_records")
    if not isinstance(per_course, list) or len(per_course) != 60:
        raise MergeValidationError(f"{label} must contain 60 per-course rows")
    expected_episode_count = 60 * eval_episodes
    if (not isinstance(episode_records, list)
            or len(episode_records) != expected_episode_count):
        raise MergeValidationError(
            f"{label} has the wrong heldout episode-record count")

    rates: list[float] = []
    difficulties: list[float] = []
    status_counter: Counter[str] = Counter()
    total_success = 0
    total_sim_steps = 0
    record_offset = 0
    stratum_successes = [0] * 10
    stratum_episodes = [0] * 10
    for index, (env_id, course) in enumerate(zip(heldout_ids, per_course)):
        if not isinstance(course, dict) or course.get("env_id") != env_id:
            raise MergeValidationError(f"{label} per-course order differs")
        barn_index = int(env_id.removeprefix("barn-"))
        stratum = index // 6
        if course.get("barn_index") != barn_index or course.get("stratum") != stratum:
            raise MergeValidationError(f"{label} has invalid course identity")
        difficulty = _require_number(
            course.get("difficulty"), field=f"{label} {env_id} difficulty")
        if difficulties and difficulty < difficulties[-1]:
            raise MergeValidationError(f"{label} course difficulty order differs")
        low, high = panel["heldout_ranges"][stratum]
        if not low <= difficulty <= high:
            raise MergeValidationError(
                f"{label} {env_id} falls outside its difficulty stratum")
        difficulties.append(difficulty)
        expected_seed = _derived_seed(0xE7A1C0DE, eval_seed, barn_index)
        if course.get("seed") != expected_seed:
            raise MergeValidationError(f"{label} {env_id} eval seed differs")
        if course.get("episodes") != eval_episodes:
            raise MergeValidationError(f"{label} {env_id} episode count differs")
        successes = _require_int(
            course.get("successes"), field=f"{label} {env_id} successes")
        if successes > eval_episodes:
            raise MergeValidationError(f"{label} {env_id} has excess successes")
        sim_steps = _require_int(
            course.get("sim_steps"), field=f"{label} {env_id} sim_steps")
        expected_records = episode_records[
            record_offset:record_offset + eval_episodes]
        record_offset += eval_episodes
        record_success = 0
        record_steps = 0
        for episode_index, record in enumerate(expected_records):
            if not isinstance(record, dict):
                raise MergeValidationError(f"{label} episode row is malformed")
            expected_identity = {
                "env_id": env_id, "barn_index": barn_index,
                "seed": expected_seed, "episode_index": episode_index}
            if any(record.get(field) != value
                   for field, value in expected_identity.items()):
                raise MergeValidationError(f"{label} episode identity differs")
            success = _require_int(
                record.get("success"), field=f"{label} episode success")
            if success not in (0, 1):
                raise MergeValidationError(f"{label} episode success is not binary")
            status = record.get("status")
            if status not in EVALUATION_STATUSES:
                raise MergeValidationError(f"{label} episode status is invalid")
            if success != int(status == "succeeded"):
                raise MergeValidationError(
                    f"{label} episode success/status disagree")
            steps = _require_int(
                record.get("sim_steps"), field=f"{label} episode sim_steps")
            seconds = _require_number(
                record.get("sim_seconds"), field=f"{label} episode sim_seconds")
            if seconds < 0.0:
                raise MergeValidationError(f"{label} episode sim_seconds is negative")
            clearance = record.get("planned_clearance_m")
            if clearance is not None and _require_number(
                    clearance, field=f"{label} episode clearance") < 0.0:
                raise MergeValidationError(f"{label} episode clearance is negative")
            record_success += success
            record_steps += steps
            status_counter[status] += 1
        if successes != record_success or sim_steps != record_steps:
            raise MergeValidationError(
                f"{label} {env_id} aggregate differs from episode records")
        rate = successes / eval_episodes
        _same_number(course.get("success_rate"), rate,
                     field=f"{label} {env_id} success_rate")
        rates.append(rate)
        total_success += successes
        total_sim_steps += sim_steps
        stratum_successes[stratum] += successes
        stratum_episodes[stratum] += eval_episodes

    for stratum, (low, high) in enumerate(panel["heldout_ranges"]):
        chunk = difficulties[stratum * 6:(stratum + 1) * 6]
        if not (math.isclose(chunk[0], low, rel_tol=0.0, abs_tol=1e-12)
                and math.isclose(chunk[-1], high, rel_tol=0.0, abs_tol=1e-12)):
            raise MergeValidationError(
                f"{label} difficulty range endpoints differ")
    if evaluation.get("per_task_success") != rates:
        raise MergeValidationError(f"{label} per_task_success differs")
    mean_success = total_success / expected_episode_count
    _same_number(evaluation.get("mean_success"), mean_success,
                 field=f"{label} mean_success")
    if evaluation.get("n_tasks") != 60:
        raise MergeValidationError(f"{label} n_tasks must equal 60")
    if evaluation.get("eval_episodes") != expected_episode_count:
        raise MergeValidationError(f"{label} eval_episodes total differs")
    if evaluation.get("eval_sim_steps") != total_sim_steps:
        raise MergeValidationError(f"{label} eval_sim_steps differs")
    _same_number(evaluation.get("success@1"), mean_success,
                 field=f"{label} success@1")
    _same_number(evaluation.get("easy_decile_retention"), sum(rates[:6]) / 6,
                 field=f"{label} easy_decile_retention")
    _require_count_dict(
        evaluation.get("status_counts"), field=f"{label} status_counts",
        expected=status_counter)

    bins = evaluation.get("difficulty_bins")
    bin_success = evaluation.get("success_by_difficulty_bin")
    if (not isinstance(bins, list) or len(bins) != 10
            or not isinstance(bin_success, list) or len(bin_success) != 10):
        raise MergeValidationError(f"{label} must contain exactly 10 bins")
    for bin_index, row in enumerate(bins):
        if not isinstance(row, dict) or row.get("bin") != bin_index:
            raise MergeValidationError(f"{label} difficulty bin identity differs")
        chunk_difficulty = difficulties[bin_index * 6:(bin_index + 1) * 6]
        chunk_rates = rates[bin_index * 6:(bin_index + 1) * 6]
        if row.get("n_courses") != 6:
            raise MergeValidationError(f"{label} difficulty bin size differs")
        _same_number(row.get("difficulty_min"), chunk_difficulty[0],
                     field=f"{label} bin difficulty_min")
        _same_number(row.get("difficulty_max"), chunk_difficulty[-1],
                     field=f"{label} bin difficulty_max")
        bin_rate = sum(chunk_rates) / 6
        _same_number(row.get("mean_success"), bin_rate,
                     field=f"{label} bin mean_success")
        _same_number(bin_success[bin_index], bin_rate,
                     field=f"{label} success_by_difficulty_bin")

    observed_rates = [
        stratum_successes[index] / stratum_episodes[index]
        for index in range(10)]
    pairs = [(posterior[index], observed_rates[index])
             for index in range(10) if visits[index] >= 1]
    if evaluation.get("teacher/calibration_n") != len(pairs):
        raise MergeValidationError(f"{label} calibration_n differs")
    if pairs:
        bias = sum(estimate - observed for estimate, observed in pairs) / len(pairs)
        mae = sum(abs(estimate - observed) for estimate, observed in pairs) / len(pairs)
        _same_number(evaluation.get("teacher/calibration_bias"), bias,
                     field=f"{label} calibration_bias")
        _same_number(evaluation.get("teacher/calibration_mae"), mae,
                     field=f"{label} calibration_mae")
    elif ("teacher/calibration_bias" in evaluation
          or "teacher/calibration_mae" in evaluation):
        raise MergeValidationError(f"{label} has calibration values with no pairs")


def _validate_training_records(
    records: object,
    *,
    train_ids: set[str],
    n_rollouts: int,
    label: str,
) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise MergeValidationError(f"{label} training_episode_records is missing")
    for index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("episode_index") != index:
            raise MergeValidationError(f"{label} training episode order differs")
        if record.get("group_episode_index") != index % n_rollouts:
            raise MergeValidationError(f"{label} group episode order differs")
        stratum = _require_int(
            record.get("stratum"), field=f"{label} training stratum")
        if stratum >= 10:
            raise MergeValidationError(f"{label} training stratum is invalid")
        if record.get("course_id") not in train_ids:
            raise MergeValidationError(f"{label} training course is not in panel")
        difficulty = _require_number(
            record.get("difficulty"), field=f"{label} training difficulty")
        if difficulty < 0.0:
            raise MergeValidationError(f"{label} training difficulty is negative")
        success = _require_int(
            record.get("success"), field=f"{label} training success")
        if success not in (0, 1):
            raise MergeValidationError(f"{label} training success is not binary")
        status = record.get("status")
        if status not in EVALUATION_STATUSES or success != int(status == "succeeded"):
            raise MergeValidationError(
                f"{label} training success/status disagree")
        _require_int(record.get("sim_steps"), field=f"{label} training sim_steps")
        seconds = _require_number(
            record.get("sim_seconds"), field=f"{label} training sim_seconds")
        if seconds < 0.0:
            raise MergeValidationError(f"{label} training sim_seconds is negative")
        clearance = record.get("planned_clearance_m")
        if clearance is not None and _require_number(
                clearance, field=f"{label} training clearance") < 0.0:
            raise MergeValidationError(f"{label} training clearance is negative")
    if len(records) % n_rollouts:
        raise MergeValidationError(
            f"{label} training records do not form complete rollout groups")
    for offset in range(0, len(records), n_rollouts):
        group = records[offset:offset + n_rollouts]
        if len({record["stratum"] for record in group}) != 1:
            raise MergeValidationError(f"{label} training group mixes strata")
    return records


def _normalized_auc(
    history: Sequence[Mapping[str, Any]],
    currency: str,
    *,
    budget: float | None = None,
) -> float:
    x = np.asarray([row[currency] for row in history], dtype=float)
    y = np.asarray(
        [row["eval"]["mean_success"] for row in history], dtype=float)
    if len(x) == 0:
        raise MergeValidationError("cannot validate AUC from empty history")
    if np.any(np.diff(x) < 0.0):
        raise MergeValidationError(f"non-monotone {currency} history")
    if budget is not None:
        budget = float(budget)
        if budget <= 0.0 or x[-1] < budget:
            raise MergeValidationError(
                f"{currency} history does not reach requested AUC budget")
        keep = x < budget
        boundary_y = float(np.interp(budget, x, y))
        x = np.concatenate((x[keep], [budget]))
        y = np.concatenate((y[keep], [boundary_y]))
    if x[-1] <= 0.0:
        return float(y[-1])
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(y, x) / x[-1])


def _validate_history(
    run: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    config: Mapping[str, Any],
    panel: Mapping[str, Any],
    eval_seed: int,
    records: Sequence[dict[str, Any]],
    label: str,
) -> None:
    history = run.get("history")
    if (not isinstance(history, list) or not history
            or not all(isinstance(row, dict) for row in history)):
        raise MergeValidationError(f"{label} {arm} history is empty or malformed")
    steps = [row.get("step") for row in history]
    max_updates = config["max_training_updates"]
    if (not all(_is_int(step) for step in steps) or steps[0] != 0
            or any(later <= earlier for earlier, later in zip(steps, steps[1:]))
            or steps[-1] > max_updates):
        raise MergeValidationError(f"{label} {arm} checkpoint steps are incomplete")
    budget = config["training_sim_step_budget"]
    interval = config["eval_sim_step_interval"]
    n_rollouts = config["n_rollouts"]
    tasks_per_step = config["tasks_per_step"]
    previous_currencies = {
        "episodes": -1.0, "sim_steps": -1.0,
        "training_wall_seconds": -1.0}
    sim_steps: list[int] = []
    for checkpoint_index, row in enumerate(history):
        step = row["step"]
        expected_episodes = step * tasks_per_step * n_rollouts
        if row.get("episodes") != expected_episodes:
            raise MergeValidationError(
                f"{label} {arm} training episode arithmetic differs")
        prefix = records[:expected_episodes]
        expected_sim_steps = sum(record["sim_steps"] for record in prefix)
        if row.get("sim_steps") != expected_sim_steps:
            raise MergeValidationError(
                f"{label} {arm} training sim-step accounting differs")
        for currency in previous_currencies:
            current = _require_number(
                row.get(currency), field=f"{label} {arm} {currency}")
            if current < 0.0 or current < previous_currencies[currency]:
                raise MergeValidationError(
                    f"{label} {arm} has non-monotone {currency}")
            previous_currencies[currency] = current
        sim_steps.append(expected_sim_steps)

        groups = step * tasks_per_step
        posterior, visits = _validate_teacher(
            row.get("teacher"), n_strata=10, groups=groups,
            label=f"{label} {arm} checkpoint {checkpoint_index}")
        prefix_status = Counter(record["status"] for record in prefix)
        prefix_courses = Counter(record["course_id"] for record in prefix)
        _require_count_dict(
            row.get("training_status_counts"),
            field=f"{label} {arm} training_status_counts",
            expected=prefix_status)
        _require_count_dict(
            row.get("training_course_counts"),
            field=f"{label} {arm} training_course_counts",
            expected=prefix_courses)
        group_rows = [prefix[offset:offset + n_rollouts]
                      for offset in range(0, len(prefix), n_rollouts)]
        stratum_counts = Counter(str(group[0]["stratum"]) for group in group_rows)
        _require_count_dict(
            row.get("sampled_stratum_counts"),
            field=f"{label} {arm} sampled_stratum_counts",
            expected=stratum_counts)
        group_successes = [sum(record["success"] for record in group)
                           for group in group_rows]
        all_fail = sum(value == 0 for value in group_successes)
        all_pass = sum(value == n_rollouts for value in group_successes)
        live = groups - all_fail - all_pass
        expected_counts = {
            "all_fail_groups": all_fail,
            "all_pass_groups": all_pass,
            "live_groups": live,
            "dead_groups": all_fail + all_pass,
            "relabeled_groups": 0,
            "updates_live": live,
            "updates_relabel": 0,
        }
        for field, expected in expected_counts.items():
            if row.get(field) != expected:
                raise MergeValidationError(
                    f"{label} {arm} {field} differs from training records")
        _same_number(
            row.get("dead_group_rate"), (all_fail + all_pass) / max(groups, 1),
            field=f"{label} {arm} dead_group_rate")
        _same_number(
            row.get("all_fail_group_rate"), all_fail / max(groups, 1),
            field=f"{label} {arm} all_fail_group_rate")
        _same_number(
            row.get("all_pass_group_rate"), all_pass / max(groups, 1),
            field=f"{label} {arm} all_pass_group_rate")
        _validate_eval(
            row.get("eval"), panel=panel, eval_seed=eval_seed,
            eval_episodes=config["eval_episodes"], posterior=posterior,
            visits=visits,
            label=f"{label} {arm} checkpoint {checkpoint_index}")

    if sim_steps[0] != 0 or sim_steps[-1] < budget:
        raise MergeValidationError(
            f"{label} {arm} simulator-step-budget history is incomplete")
    if any(value >= budget for value in sim_steps[:-1]):
        raise MergeValidationError(
            f"{label} {arm} contains post-budget checkpoints")
    next_checkpoint = interval
    for value in sim_steps[1:]:
        if value < min(next_checkpoint, budget):
            raise MergeValidationError(
                f"{label} {arm} simulator-step checkpoint is premature")
        while next_checkpoint <= value:
            next_checkpoint += interval
    final = run.get("final")
    if not isinstance(final, dict) or final != history[-1]:
        raise MergeValidationError(
            f"{label} {arm} final does not equal history[-1]")
    if len(records) != final["episodes"]:
        raise MergeValidationError(
            f"{label} {arm} final episode total differs from records")
    for field, currency in _AUC_FIELDS.items():
        expected = _normalized_auc(
            history, currency,
            budget=(budget if currency == "sim_steps" else None))
        _same_number(run.get(field), expected, field=f"{label} {arm} {field}")


def _validate_artifact(
    artifact: object,
    *,
    expected_hashes: Mapping[str, str],
    contract: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    if not isinstance(artifact, dict) or artifact.get("schema_version") != 1:
        raise MergeValidationError(f"{label} has unsupported artifact schema")
    if artifact.get("evidence_status") != contract["evidence_status"]:
        raise MergeValidationError(f"{label} evidence_status differs from cell")
    if artifact.get("domain") != DOMAIN:
        raise MergeValidationError(f"{label} is not BARN evidence")
    heldout_protocol = artifact.get("heldout_protocol")
    if not isinstance(heldout_protocol, str) or "fixed course-level seeds" not in heldout_protocol:
        raise MergeValidationError(f"{label} heldout_protocol is invalid")

    provenance = artifact.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("asset_hashes_verified") is not True:
        raise MergeValidationError(f"{label} lacks verified provenance")
    stable_provenance: dict[str, Any] = {"asset_hashes_verified": True}
    for field, expected in expected_hashes.items():
        actual = _require_sha256(
            provenance.get(field), field=f"{label} provenance {field}")
        if not hmac.compare_digest(actual, expected):
            raise MergeValidationError(
                f"{label} provenance {field} differs from expected hash")
        stable_provenance[field] = actual
    bound_manifest = _require_sha256(
        provenance.get("split_bound_manifest_sha256"),
        field=f"{label} split_bound_manifest_sha256")
    if not hmac.compare_digest(bound_manifest, expected_hashes["manifest_sha256"]):
        raise MergeValidationError(f"{label} split is not bound to manifest")
    stable_provenance["split_bound_manifest_sha256"] = bound_manifest
    provenance_paths = {
        field: copy.deepcopy(provenance[field])
        for field in _PATH_PROVENANCE_FIELDS if field in provenance}
    for field, value in provenance.items():
        if field.endswith("_sha256") and field not in stable_provenance:
            stable_provenance[field] = _require_sha256(
                value, field=f"{label} provenance {field}")
        elif field not in stable_provenance and field not in _PATH_PROVENANCE_FIELDS:
            stable_provenance[field] = copy.deepcopy(value)

    config = artifact.get("config")
    if not isinstance(config, dict):
        raise MergeValidationError(f"{label} is missing config")
    seed = _extract_seed(config, label=label)
    if seed not in contract["seeds"]:
        raise MergeValidationError(f"{label} seed is not declared by protocol")
    for field, expected in contract["expected_config"].items():
        if config.get(field) != expected:
            raise MergeValidationError(
                f"{label} config {field} differs from campaign cell")
    expected_order = contract["execution_order_by_seed"][str(seed)]
    if config.get("execution_order") != expected_order:
        raise MergeValidationError(
            f"{label} execution_order differs from protocol seed mapping")
    runtime = _validate_runtime_config(config, seed=seed, label=label)
    expected_runtime = contract["expected_runtime_by_seed"][str(seed)]
    for field, expected in expected_runtime.items():
        if runtime[field] != expected:
            raise MergeValidationError(
                f"{label} config {field} differs from protocol isolation mapping")
    panel = _validate_strata(config, label=label)
    execution = _normalize_execution(artifact.get("execution"), label=label)
    if execution["slurm_array_task_id"] != seed:
        raise MergeValidationError(
            f"{label} slurm_array_task_id must equal the campaign seed")

    results = artifact.get("results")
    if (not isinstance(results, dict)
            or list(results) != list(contract["arms"])):
        actual = set(results) if isinstance(results, dict) else set()
        expected = set(contract["arms"])
        raise MergeValidationError(
            f"{label} has missing/extra arms or noncanonical order: "
            f"missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    runs: dict[str, dict[str, Any]] = {}
    expected_teacher_seed = _derived_seed(0x7EAC4E12, seed)
    expected_eval_seed = _derived_seed(0xE7A15EED, seed)
    train_id_set = set(panel["train_ids"])
    for arm in contract["arms"]:
        arm_rows = results[arm]
        if not isinstance(arm_rows, list) or len(arm_rows) != 1:
            raise MergeValidationError(
                f"{label} arm {arm} must contain exactly one row")
        run = arm_rows[0]
        if (not isinstance(run, dict) or run.get("arm") != arm
                or run.get("seed") != seed):
            raise MergeValidationError(f"{label} arm {arm} identity differs")
        if run.get("teacher_seed") != expected_teacher_seed:
            raise MergeValidationError(f"{label} arm {arm} teacher seed differs")
        if run.get("eval_seed") != expected_eval_seed:
            raise MergeValidationError(f"{label} arm {arm} eval seed differs")
        records = _validate_training_records(
            run.get("training_episode_records"), train_ids=train_id_set,
            n_rollouts=config["n_rollouts"], label=f"{label} arm {arm}")
        _validate_history(
            run, arm=arm, seed=seed, config=config, panel=panel,
            eval_seed=expected_eval_seed, records=records, label=label)
        runs[arm] = copy.deepcopy(run)

    frozen_config = {
        key: copy.deepcopy(value) for key, value in config.items()
        if key not in _SEED_CONFIG_FIELDS
    }
    return {
        "seed": seed,
        "runs": runs,
        "runtime": runtime,
        "execution_order": list(expected_order),
        "execution": execution,
        "frozen_config": frozen_config,
        "stable_provenance": stable_provenance,
        "provenance_paths": provenance_paths,
        "heldout_protocol": heldout_protocol,
    }


def prepare_validation(
    *,
    protocol: Mapping[str, Any],
    campaign_cell: str,
    expected_seeds: Sequence[int],
    expected_manifest_sha256: str,
    expected_split_sha256: str,
    expected_prereg_sha256: str,
    expected_analyzer_sha256: str,
    expected_protocol_sha256: str,
    expected_container_sha256: str,
    expected_source_sha256: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    hashes = _expected_hashes(
        manifest_sha256=expected_manifest_sha256,
        split_sha256=expected_split_sha256,
        prereg_sha256=expected_prereg_sha256,
        analyzer_sha256=expected_analyzer_sha256,
        protocol_sha256=expected_protocol_sha256,
        container_sha256=expected_container_sha256,
        source_sha256=expected_source_sha256,
    )
    contract = _protocol_contract(
        protocol, campaign_cell=campaign_cell,
        expected_seeds=expected_seeds, expected_hashes=hashes)
    return hashes, contract


def validate_campaign_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_hashes: Mapping[str, str],
    contract: Mapping[str, Any],
    label: str = "artifact",
) -> dict[str, Any]:
    return _validate_artifact(
        artifact, expected_hashes=expected_hashes, contract=contract, label=label)


def _validate_cross_seed_isolation(validated: Sequence[Mapping[str, Any]]) -> None:
    domains: dict[int, int] = {}
    ports: dict[int, int] = {}
    roots: dict[str, int] = {}
    for item in validated:
        seed = item["seed"]
        runtime = item["runtime"]
        for domain in (runtime["domain_id"], runtime["eval_domain_id"]):
            if domain in domains:
                raise MergeValidationError(
                    f"ROS domain collision between seeds {domains[domain]} and {seed}")
            domains[domain] = seed
        for port in (runtime["master_port"], runtime["eval_master_port"]):
            if port in ports:
                raise MergeValidationError(
                    f"Gazebo port collision between seeds {ports[port]} and {seed}")
            ports[port] = seed
        root = str(Path(runtime["runtime_root"]).resolve())
        if root in roots:
            raise MergeValidationError(
                f"runtime_root collision between seeds {roots[root]} and {seed}")
        roots[root] = seed


def _merge_campaign_artifacts_for_preflight(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
    campaign_cell: str,
    expected_seeds: Sequence[int],
    expected_manifest_sha256: str,
    expected_split_sha256: str,
    expected_prereg_sha256: str,
    expected_analyzer_sha256: str,
    expected_protocol_sha256: str,
    expected_container_sha256: str,
    expected_source_sha256: str,
) -> dict[str, Any]:
    """Internal structural merge used only by tests and selector preflight."""

    canonical_seeds = _canonical_expected_seeds(expected_seeds)
    hashes, contract = prepare_validation(
        protocol=protocol, campaign_cell=campaign_cell,
        expected_seeds=canonical_seeds,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_split_sha256=expected_split_sha256,
        expected_prereg_sha256=expected_prereg_sha256,
        expected_analyzer_sha256=expected_analyzer_sha256,
        expected_protocol_sha256=expected_protocol_sha256,
        expected_container_sha256=expected_container_sha256,
        expected_source_sha256=expected_source_sha256,
    )
    if not artifacts:
        raise MergeValidationError("no per-seed artifacts were provided")
    validated = [
        _validate_artifact(
            artifact, expected_hashes=hashes, contract=contract,
            label=f"artifact[{index}]")
        for index, artifact in enumerate(artifacts)
    ]
    actual_seeds = [item["seed"] for item in validated]
    duplicates = sorted(seed for seed, count in Counter(actual_seeds).items()
                        if count > 1)
    if duplicates:
        raise MergeValidationError(f"duplicate per-seed artifacts: {duplicates}")
    missing = sorted(set(canonical_seeds) - set(actual_seeds))
    extra = sorted(set(actual_seeds) - set(canonical_seeds))
    if missing or extra:
        raise MergeValidationError(
            f"seed matrix differs: missing={missing}, extra={extra}")
    validated.sort(key=lambda item: item["seed"])
    _validate_cross_seed_isolation(validated)

    reference = validated[0]
    campaign_ids = {item["execution"]["campaign_id"] for item in validated}
    if len(campaign_ids) != 1:
        raise MergeValidationError("execution campaign_id differs across seeds")
    for item in validated[1:]:
        if item["stable_provenance"] != reference["stable_provenance"]:
            raise MergeValidationError(
                f"provenance hash/semantic mismatch for seed {item['seed']}")
        if item["frozen_config"] != reference["frozen_config"]:
            raise MergeValidationError(
                f"frozen config mismatch for seed {item['seed']}")
        if item["heldout_protocol"] != reference["heldout_protocol"]:
            raise MergeValidationError(
                f"heldout_protocol mismatch for seed {item['seed']}")

    merged_config = copy.deepcopy(reference["frozen_config"])
    merged_config.update({
        "seeds": len(canonical_seeds),
        "seed_start": canonical_seeds[0],
        "seed_list": list(canonical_seeds),
        "per_seed_runtime": [
            {"seed": item["seed"], **copy.deepcopy(item["runtime"])}
            for item in validated],
    })
    merged_results = {
        arm: [copy.deepcopy(item["runs"][arm]) for item in validated]
        for arm in contract["arms"]}
    per_seed_execution_order = [
        {"seed": item["seed"],
         "execution_order": copy.deepcopy(item["execution_order"])}
        for item in validated]
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": contract["evidence_status"],
        "domain": DOMAIN,
        "heldout_protocol": reference["heldout_protocol"],
        "provenance": copy.deepcopy(reference["stable_provenance"]),
        "config": merged_config,
        "results": merged_results,
        "merge": {
            "schema_version": 1,
            "outcome_blind": True,
            "campaign_cell": campaign_cell,
            "expected_seed_list": list(canonical_seeds),
            "input_artifact_count": len(validated),
            "per_seed_execution_order": per_seed_execution_order,
            "selected_execution": [
                {"seed": item["seed"], **copy.deepcopy(item["execution"])}
                for item in validated],
            "per_seed_provenance_paths": [
                {"seed": item["seed"], **copy.deepcopy(item["provenance_paths"])}
                for item in validated],
        },
    }


def _canonical_receipt_path(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise MergeValidationError(f"{field} must be a non-empty path string")
    path = Path(value)
    resolved = path.resolve()
    if not path.is_absolute() or str(resolved) != value:
        raise MergeValidationError(f"{field} must be a canonical absolute path")
    return resolved


def _validate_receipt_exclusions(
    excluded: object, *, canonical_seeds: Sequence[int]
) -> None:
    if not isinstance(excluded, list):
        raise MergeValidationError("selection receipt excluded must be a list")
    identities: set[tuple[int, str]] = set()
    for index, row in enumerate(excluded):
        label = f"selection receipt excluded[{index}]"
        if not isinstance(row, dict) or set(row) != _EXCLUDED_RECEIPT_FIELDS:
            raise MergeValidationError(f"{label} fields are not exact")
        seed = _require_int(row["seed"], field=f"{label} seed")
        if seed not in canonical_seeds:
            raise MergeValidationError(f"{label} seed is outside the campaign")
        attempt_id = row["attempt_id"]
        if (not isinstance(attempt_id, str)
                or re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", attempt_id) is None):
            raise MergeValidationError(f"{label} has invalid attempt_id")
        identity = (seed, attempt_id)
        if identity in identities:
            raise MergeValidationError(
                "selection receipt contains duplicate excluded attempts")
        identities.add(identity)
        submitted = row["submitted_utc"]
        if not isinstance(submitted, str):
            raise MergeValidationError(f"{label} submitted_utc is invalid")
        try:
            parsed = datetime.fromisoformat(submitted.replace("Z", "+00:00"))
        except ValueError as error:
            raise MergeValidationError(
                f"{label} submitted_utc is invalid") from error
        if (parsed.tzinfo is None or parsed.utcoffset() is None
                or parsed.astimezone(timezone.utc).isoformat() != submitted):
            raise MergeValidationError(
                f"{label} submitted_utc is not normalized UTC")
        _canonical_receipt_path(row["artifact_path"], field=f"{label} path")
        complete = row["artifact_complete"]
        reason = row["reason"]
        if (not isinstance(complete, bool)
                or (complete and reason != "later_complete_attempt")
                or (not complete and reason != "incomplete_no_artifact")):
            raise MergeValidationError(f"{label} completion reason differs")


def _validate_selection_receipt(
    receipt: object,
    *,
    selection_receipt_sha256: str,
    artifact_paths: Sequence[Path],
    artifacts: Sequence[Mapping[str, Any]],
    merged: Mapping[str, Any],
    expected_hashes: Mapping[str, str],
    campaign_cell: str,
    canonical_seeds: Sequence[int],
) -> dict[str, Any]:
    """Bind a selector receipt to the exact five files entering a merge."""

    receipt_digest = _require_sha256(
        selection_receipt_sha256, field="selection receipt SHA-256")
    if (not isinstance(receipt, dict)
            or set(receipt) != _SELECTION_RECEIPT_FIELDS
            or receipt.get("schema_version") != 1):
        raise MergeValidationError("unsupported selection receipt schema")
    if receipt.get("outcome_blind") is not True:
        raise MergeValidationError("selection receipt is not outcome-blind")
    if receipt.get("selection_rule") != SELECTION_RULE:
        raise MergeValidationError("selection receipt rule differs")
    if receipt.get("campaign_cell") != campaign_cell:
        raise MergeValidationError("selection receipt campaign cell differs")
    if receipt.get("expected_seed_list") != list(canonical_seeds):
        raise MergeValidationError("selection receipt seed list differs")
    receipt_hashes = receipt.get("expected_hashes")
    if (not isinstance(receipt_hashes, dict)
            or set(receipt_hashes) != set(PROVENANCE_HASH_FIELDS)):
        raise MergeValidationError(
            "selection receipt expected hashes are not exact")
    normalized_hashes = {
        field: _require_sha256(
            receipt_hashes[field],
            field=f"selection receipt expected {field}")
        for field in PROVENANCE_HASH_FIELDS}
    if normalized_hashes != dict(expected_hashes):
        raise MergeValidationError("selection receipt expected hashes differ")
    ledger_digest = _require_sha256(
        receipt.get("ledger_sha256"), field="selection receipt ledger_sha256")

    campaign_id = receipt.get("campaign_id")
    execution_rows = merged["merge"]["selected_execution"]
    execution_by_seed = {
        row["seed"]: {key: copy.deepcopy(value) for key, value in row.items()
                      if key != "seed"}
        for row in execution_rows}
    artifact_campaign_ids = {
        execution["campaign_id"] for execution in execution_by_seed.values()}
    if (not isinstance(campaign_id, str)
            or artifact_campaign_ids != {campaign_id}):
        raise MergeValidationError("selection receipt campaign differs")

    resolved_inputs = [Path(path).resolve() for path in artifact_paths]
    if (len(resolved_inputs) != 5 or len(artifacts) != 5
            or len(set(resolved_inputs)) != 5):
        raise MergeValidationError(
            "production merge requires exactly five unique input artifacts")
    artifact_by_path = dict(zip(resolved_inputs, artifacts))
    selected = receipt.get("selected")
    if not isinstance(selected, list) or len(selected) != 5:
        raise MergeValidationError(
            "selection receipt must select exactly five artifacts")
    selected_seeds: list[int] = []
    selected_paths: list[Path] = []
    canonical_selected: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        label = f"selection receipt selected[{index}]"
        if not isinstance(row, dict) or set(row) != _SELECTED_RECEIPT_FIELDS:
            raise MergeValidationError(f"{label} fields are not exact")
        seed = _require_int(row["seed"], field=f"{label} seed")
        selected_seeds.append(seed)
        path = _canonical_receipt_path(
            row["artifact_path"], field=f"{label} artifact_path")
        selected_paths.append(path)
        artifact_digest = _require_sha256(
            row["artifact_sha256"], field=f"{label} artifact_sha256")
        if path not in artifact_by_path:
            raise MergeValidationError(
                "selection receipt paths differ from merge inputs")
        if not path.is_file() or not hmac.compare_digest(
                sha256_path(path), artifact_digest):
            raise MergeValidationError(
                f"{label} artifact SHA-256 differs from input file")
        artifact = artifact_by_path[path]
        config = artifact.get("config")
        if not isinstance(config, dict) or _extract_seed(
                config, label=label) != seed:
            raise MergeValidationError(
                f"{label} path is bound to the wrong seed artifact")
        receipt_execution = {
            "campaign_id": campaign_id,
            "attempt_id": row["attempt_id"],
            "submitted_utc": row["submitted_utc"],
            "slurm_job_id": row["slurm_job_id"],
            "slurm_array_job_id": row["slurm_array_job_id"],
            "slurm_array_task_id": row["slurm_array_task_id"],
        }
        normalized_execution = _normalize_execution(
            receipt_execution, label=label)
        if (normalized_execution != receipt_execution
                or normalized_execution != execution_by_seed.get(seed)):
            raise MergeValidationError(
                f"{label} execution identity differs from input artifact")
        canonical_selected.append({
            "seed": seed,
            "attempt_id": normalized_execution["attempt_id"],
            "artifact_sha256": artifact_digest,
            "execution": copy.deepcopy(normalized_execution),
        })
    if selected_seeds != list(canonical_seeds):
        raise MergeValidationError(
            "selection receipt selected rows are not in exact seed order")
    if len(set(selected_paths)) != 5 or set(selected_paths) != set(resolved_inputs):
        raise MergeValidationError(
            "selection receipt paths differ from merge inputs")
    _validate_receipt_exclusions(
        receipt.get("excluded"), canonical_seeds=canonical_seeds)
    return {
        "schema_version": 1,
        "selection_receipt_sha256": receipt_digest,
        "ledger_sha256": ledger_digest,
        "rule": SELECTION_RULE,
        "campaign_id": campaign_id,
        "campaign_cell": campaign_cell,
        "selected": canonical_selected,
    }


def merge_campaign_artifacts(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Reject receipt-free in-memory merges that could cherry-pick outcomes."""

    del args, kwargs
    raise MergeValidationError(
        "direct merge is forbidden; a selector receipt is required")


merge_artifacts = merge_campaign_artifacts


def merge_campaign_files(
    paths: Sequence[Path],
    *,
    protocol_path: Path,
    selection_receipt_path: Path,
    campaign_cell: str,
    expected_seeds: Sequence[int],
    **hashes: str,
) -> dict[str, Any]:
    expected_protocol = _require_sha256(
        hashes.get("expected_protocol_sha256"),
        field="expected protocol_sha256")
    actual_protocol = sha256_path(protocol_path)
    if not hmac.compare_digest(actual_protocol, expected_protocol):
        raise MergeValidationError("machine protocol file hash differs")
    canonical_seeds = _canonical_expected_seeds(expected_seeds)
    if canonical_seeds != (1, 2, 3, 4, 5):
        raise MergeValidationError(
            "production merge requires exact seeds 1,2,3,4,5")
    resolved_paths = [Path(path).resolve() for path in paths]
    artifacts = [load_json(path) for path in resolved_paths]
    merged = _merge_campaign_artifacts_for_preflight(
        artifacts,
        protocol=load_json(protocol_path), campaign_cell=campaign_cell,
        expected_seeds=canonical_seeds, **hashes)
    expected_hashes = _expected_hashes(**{
        field: hashes.get(f"expected_{field}")
        for field in PROVENANCE_HASH_FIELDS})
    selection = _validate_selection_receipt(
        load_json(selection_receipt_path),
        selection_receipt_sha256=sha256_path(selection_receipt_path),
        artifact_paths=resolved_paths, artifacts=artifacts, merged=merged,
        expected_hashes=expected_hashes, campaign_cell=campaign_cell,
        canonical_seeds=canonical_seeds)
    merged["merge"]["selection"] = selection
    return merged


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


_atomic_write_json = atomic_write_json


def add_hash_arguments(parser: argparse.ArgumentParser) -> None:
    for field in PROVENANCE_HASH_FIELDS:
        parser.add_argument("--expected-" + field.replace("_", "-"),
                            required=True)


def hash_arguments(args: argparse.Namespace) -> dict[str, str]:
    return {f"expected_{field}": getattr(args, f"expected_{field}")
            for field in PROVENANCE_HASH_FIELDS}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict outcome-blind merge of one frozen BARN cell")
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--campaign-cell", choices=CAMPAIGN_CELLS, required=True)
    parser.add_argument(
        "--expected-seeds", "--expected-seed-list", dest="expected_seeds",
        required=True, help="required comma-separated frozen seed list")
    add_hash_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        expected_seeds = parse_expected_seeds(args.expected_seeds)
    except MergeValidationError as error:
        parser.error(str(error))
    resolved_inputs = {path.resolve() for path in args.artifacts}
    resolved_inputs.update({
        args.protocol.resolve(), args.selection_receipt.resolve()})
    if args.output.resolve() in resolved_inputs:
        parser.error("output path must not overwrite an input artifact")
    merged = merge_campaign_files(
        args.artifacts, protocol_path=args.protocol,
        selection_receipt_path=args.selection_receipt,
        campaign_cell=args.campaign_cell, expected_seeds=expected_seeds,
        **hash_arguments(args))
    atomic_write_json(args.output, merged)
    print(
        f"wrote {args.output}: cell={args.campaign_cell} "
        f"seeds={len(expected_seeds)} arms={len(merged['results'])} "
        f"status={merged['evidence_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
