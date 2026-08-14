"""Frozen, fail-closed analysis for the primary ICRA BARN campaign.

Only the outcome-blind merger's complete five-seed primary artifact is
accepted.  Engineering smokes and per-seed files deliberately have no path
through this module.  The analyzer verifies its own bytes, the adjacent frozen
machine protocol, and every frozen provenance binding before deriving any
endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import itertools
import json
import math
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ARMS = ("ours_uN", "uniform", "learnability", "staged")
COMPARATORS = ARMS[1:]
GATE_COMPARATORS = ("uniform", "learnability")
EXPECTED_SEEDS = (1, 2, 3, 4, 5)
FULL_EVIDENCE_STATUS = "full_barn_campaign"
ABLATION_EVIDENCE_STATUS = "full_barn_n_ablation"
DOMAIN = "barn_gazebo_cpu_navigation"
PROTOCOL_ID = "icra2027-barn-v1"
DEFAULT_PROTOCOL = Path(__file__).with_name("barn_protocol.json")
FRESH_ABLATION_N = (2, 4, 16)
SELECTION_RULE = "earliest_submitted_complete_hash_valid_attempt_per_seed"

MERGE_FIELDS = {
    "schema_version", "outcome_blind", "campaign_cell",
    "expected_seed_list", "input_artifact_count",
    "per_seed_execution_order", "selected_execution",
    "per_seed_provenance_paths", "selection",
}
SELECTION_FIELDS = {
    "schema_version", "selection_receipt_sha256", "ledger_sha256", "rule",
    "campaign_id", "campaign_cell", "selected",
}
SELECTION_ROW_FIELDS = {"seed", "attempt_id", "artifact_sha256", "execution"}
EXECUTION_FIELDS = {
    "campaign_id", "attempt_id", "submitted_utc", "slurm_job_id",
    "slurm_array_job_id", "slurm_array_task_id",
}
PROVENANCE_PATH_FIELDS = {
    "manifest_path", "split_path", "prereg_path", "analyzer_path",
    "protocol_path", "dataset_root", "robot_sdf",
}

FROZEN_PROVENANCE_HASHES = (
    "manifest_sha256",
    "split_sha256",
    "prereg_sha256",
    "analyzer_sha256",
    "protocol_sha256",
    "container_sha256",
    "source_sha256",
)
RESOURCE_FIELDS = (
    "episodes",
    "sim_steps",
    "training_wall_seconds",
    "evaluation_wall_seconds",
    "live_groups",
    "dead_groups",
    "relabeled_groups",
    "all_fail_groups",
    "all_pass_groups",
)
UPDATE_FIELDS = ("updates_live", "updates_relabel")


class AnalysisValidationError(ValueError):
    """Raised before analysis when a frozen evidence contract is violated."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _require_object(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise AnalysisValidationError(f"{field} must be an object")
    return value


def _require_list(value: object, *, field: str) -> list:
    if not isinstance(value, list):
        raise AnalysisValidationError(f"{field} must be a list")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise AnalysisValidationError(f"{field} must be a SHA-256 string")
    digest = value.lower()
    if (len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)):
        raise AnalysisValidationError(
            f"{field} must be a 64-character SHA-256 digest")
    return digest


def _require_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisValidationError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise AnalysisValidationError(f"{field} must be finite")
    return number


def _require_int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AnalysisValidationError(
            f"{field} must be an integer >= {minimum}")
    return int(value)


def _validate_execution_identity(
    value: object, *, field: str, seed: int,
) -> dict[str, Any]:
    execution = _require_object(value, field=field)
    if set(execution) != EXECUTION_FIELDS:
        raise AnalysisValidationError(f"{field} fields must be exact")
    normalized = dict(execution)
    for name in ("campaign_id", "attempt_id"):
        candidate = execution.get(name)
        if (not isinstance(candidate, str)
                or re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", candidate) is None):
            raise AnalysisValidationError(f"{field}.{name} is invalid")
    submitted = execution.get("submitted_utc")
    if not isinstance(submitted, str):
        raise AnalysisValidationError(f"{field}.submitted_utc is invalid")
    try:
        parsed = datetime.fromisoformat(submitted.replace("Z", "+00:00"))
    except ValueError as error:
        raise AnalysisValidationError(
            f"{field}.submitted_utc is invalid") from error
    if (parsed.tzinfo is None or parsed.utcoffset() is None
            or parsed.astimezone(timezone.utc).isoformat() != submitted):
        raise AnalysisValidationError(
            f"{field}.submitted_utc is not normalized UTC")
    for name in ("slurm_job_id", "slurm_array_job_id"):
        candidate = execution.get(name)
        if not isinstance(candidate, str) or re.fullmatch(r"[0-9]+", candidate) is None:
            raise AnalysisValidationError(f"{field}.{name} is invalid")
    task = _require_int(
        execution.get("slurm_array_task_id"),
        field=f"{field}.slurm_array_task_id", minimum=1)
    if task != seed:
        raise AnalysisValidationError(
            f"{field}.slurm_array_task_id must equal seed")
    return normalized


def _validate_selection_binding(
    merge: Mapping[str, Any], *, campaign_cell: str,
) -> Mapping[str, Any]:
    """Require the frozen selector receipt to survive intact through merge."""

    if set(merge) != MERGE_FIELDS:
        raise AnalysisValidationError("merge fields must be exact")
    if merge.get("campaign_cell") != campaign_cell:
        raise AnalysisValidationError("merge campaign_cell mismatch")

    execution_rows = _require_list(
        merge.get("selected_execution"), field="merge.selected_execution")
    if len(execution_rows) != len(EXPECTED_SEEDS):
        raise AnalysisValidationError(
            "merge.selected_execution must contain exact five seeds")
    execution_by_seed: dict[int, dict[str, Any]] = {}
    for index, value in enumerate(execution_rows):
        field = f"merge.selected_execution[{index}]"
        row = _require_object(value, field=field)
        if set(row) != {"seed", *EXECUTION_FIELDS}:
            raise AnalysisValidationError(f"{field} fields must be exact")
        seed = _require_int(row.get("seed"), field=f"{field}.seed", minimum=1)
        if seed != EXPECTED_SEEDS[index]:
            raise AnalysisValidationError(
                "merge.selected_execution is not in exact seed order")
        execution_by_seed[seed] = _validate_execution_identity(
            {key: row[key] for key in EXECUTION_FIELDS}, field=field, seed=seed)

    path_rows = _require_list(
        merge.get("per_seed_provenance_paths"),
        field="merge.per_seed_provenance_paths")
    if len(path_rows) != len(EXPECTED_SEEDS):
        raise AnalysisValidationError(
            "merge.per_seed_provenance_paths must contain exact five seeds")
    for index, value in enumerate(path_rows):
        field = f"merge.per_seed_provenance_paths[{index}]"
        row = _require_object(value, field=field)
        if set(row) != {"seed", *PROVENANCE_PATH_FIELDS}:
            raise AnalysisValidationError(f"{field} fields must be exact")
        if row.get("seed") != EXPECTED_SEEDS[index]:
            raise AnalysisValidationError(
                "merge.per_seed_provenance_paths is not in exact seed order")
        if any(key != "seed" and (not isinstance(item, str) or not item)
               for key, item in row.items()):
            raise AnalysisValidationError(f"{field} contains an invalid path")

    selection = _require_object(merge.get("selection"), field="merge.selection")
    if set(selection) != SELECTION_FIELDS or selection.get("schema_version") != 1:
        raise AnalysisValidationError("merge.selection fields/schema must be exact")
    _require_sha256(
        selection.get("selection_receipt_sha256"),
        field="merge.selection.selection_receipt_sha256")
    _require_sha256(
        selection.get("ledger_sha256"), field="merge.selection.ledger_sha256")
    if selection.get("rule") != SELECTION_RULE:
        raise AnalysisValidationError("merge.selection rule mismatch")
    if selection.get("campaign_cell") != campaign_cell:
        raise AnalysisValidationError("merge.selection campaign_cell mismatch")
    campaign_id = selection.get("campaign_id")
    if (not isinstance(campaign_id, str)
            or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", campaign_id) is None):
        raise AnalysisValidationError("merge.selection campaign_id is invalid")

    selected = _require_list(selection.get("selected"), field="merge.selection.selected")
    if len(selected) != len(EXPECTED_SEEDS):
        raise AnalysisValidationError(
            "merge.selection.selected must contain exact five seeds")
    for index, value in enumerate(selected):
        field = f"merge.selection.selected[{index}]"
        row = _require_object(value, field=field)
        if set(row) != SELECTION_ROW_FIELDS:
            raise AnalysisValidationError(f"{field} fields must be exact")
        seed = _require_int(row.get("seed"), field=f"{field}.seed", minimum=1)
        if seed != EXPECTED_SEEDS[index]:
            raise AnalysisValidationError(
                "merge.selection.selected is not in exact seed order")
        _require_sha256(row.get("artifact_sha256"), field=f"{field}.artifact_sha256")
        execution = _validate_execution_identity(
            row.get("execution"), field=f"{field}.execution", seed=seed)
        if row.get("attempt_id") != execution["attempt_id"]:
            raise AnalysisValidationError(
                f"{field}.attempt_id differs from execution identity")
        if execution["campaign_id"] != campaign_id:
            raise AnalysisValidationError(
                f"{field} campaign differs from selection campaign")
        if execution != execution_by_seed.get(seed):
            raise AnalysisValidationError(
                f"{field} execution differs from merge.selected_execution")
    return selection


def _require_probability(value: object, *, field: str) -> float:
    number = _require_number(value, field=field)
    if not 0.0 <= number <= 1.0:
        raise AnalysisValidationError(f"{field} must be in [0, 1]")
    return number


def _json_load_bytes(payload: bytes, *, field: str) -> dict:
    try:
        value = json.loads(
            payload,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {token}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise AnalysisValidationError(f"invalid JSON in {field}: {error}") from error
    return dict(_require_object(value, field=field))


def exact_sign_flip_p(differences: np.ndarray) -> float | None:
    """Exact two-sided paired randomization p-value."""

    differences = np.asarray(differences, dtype=float)
    if not np.all(np.isfinite(differences)):
        raise AnalysisValidationError("sign-flip differences must be finite")
    if len(differences) < 2:
        return None
    observed = abs(float(differences.mean()))
    exceed = 0
    total = 2 ** len(differences)
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        statistic = abs(float(np.mean(differences * np.asarray(signs))))
        exceed += statistic >= observed - 1e-15
    return exceed / total


def paired_bootstrap_ci(
    differences: np.ndarray,
    *,
    draws: int = 20_000,
    seed: int = 20270811,
) -> list[float | None]:
    differences = np.asarray(differences, dtype=float)
    if not np.all(np.isfinite(differences)):
        raise AnalysisValidationError("bootstrap differences must be finite")
    if len(differences) < 2:
        return [None, None]
    if draws < 1:
        raise AnalysisValidationError("bootstrap draws must be positive")
    rng = np.random.default_rng(seed)
    sample = rng.choice(
        differences, size=(draws, len(differences)), replace=True,
    ).mean(axis=1)
    return [float(value) for value in np.quantile(sample, [0.025, 0.975])]


def _curve_auc_at_budget(
    run: Mapping[str, Any],
    currency: str,
    budget: float,
    value_at: Callable[[Mapping[str, Any]], float],
) -> float:
    history = _require_list(run.get("history"), field="run.history")
    x = np.asarray([
        _require_number(row.get(currency), field=f"history.{currency}")
        for row in history
    ], dtype=float)
    y = np.asarray([value_at(row) for row in history], dtype=float)
    if len(x) == 0 or len(x) != len(y):
        raise AnalysisValidationError("AUC history is empty or inconsistent")
    if not np.all(np.isfinite(y)):
        raise AnalysisValidationError("AUC values must be finite")
    if x[0] != 0.0 or np.any(np.diff(x) < 0.0):
        raise AnalysisValidationError(
            f"{currency} history must start at zero and be monotone")
    if budget <= 0.0 or budget > x[-1] + 1e-12:
        raise AnalysisValidationError(
            f"run ends at {x[-1]} {currency}, below analysis budget {budget}")
    keep = x < budget
    clipped_x = np.append(x[keep], budget)
    clipped_y = np.append(y[keep], np.interp(budget, x, y))
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(clipped_y, clipped_x) / budget)


def auc_at_budget(run: dict, currency: str, budget: float) -> float:
    """Linearly interpolate target-uniform success to one common budget."""

    return _curve_auc_at_budget(
        run,
        currency,
        budget,
        lambda row: _require_probability(
            _require_object(row.get("eval"), field="history.eval").get(
                "mean_success"),
            field="history.eval.mean_success",
        ),
    )


def _seed_map(artifact: Mapping[str, Any], arm: str) -> dict[int, dict]:
    results = _require_object(artifact.get("results"), field="results")
    rows = _require_list(results.get(arm), field=f"results.{arm}")
    mapped: dict[int, dict] = {}
    for index, value in enumerate(rows):
        row = dict(_require_object(value, field=f"results.{arm}[{index}]"))
        seed = _require_int(row.get("seed"), field=f"results.{arm}[{index}].seed")
        if seed in mapped:
            raise AnalysisValidationError(f"results.{arm} has duplicate seed {seed}")
        mapped[seed] = row
    return mapped


def _value_summary(values: Mapping[int, float]) -> dict:
    ordered = [(seed, float(values[seed])) for seed in sorted(values)]
    array = np.asarray([value for _, value in ordered], dtype=float)
    if not np.all(np.isfinite(array)):
        raise AnalysisValidationError("summary values must be finite")
    return {
        "n": len(ordered),
        "mean": float(array.mean()) if len(array) else None,
        "sample_sd": float(array.std(ddof=1)) if len(array) > 1 else None,
        "per_seed": [
            {"seed": seed, "value": value} for seed, value in ordered
        ],
    }


def _nullable_value_summary(values: Mapping[int, float | None]) -> dict:
    available = {
        seed: value for seed, value in values.items() if value is not None
    }
    report = _value_summary(available)
    report["missing_seeds"] = sorted(set(values) - set(available))
    report["per_seed"] = [
        {"seed": seed, "value": values[seed]} for seed in sorted(values)
    ]
    return report


def _sum_maps(rows: Sequence[Mapping[str, int]]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for row in rows:
        total.update({str(key): int(value) for key, value in row.items()})
    return dict(sorted(total.items()))


def _count_map(value: object, *, field: str) -> dict[str, int]:
    row = _require_object(value, field=field)
    output = {}
    for key, count in row.items():
        output[str(key)] = _require_int(count, field=f"{field}.{key}")
    return dict(sorted(output.items()))


def _load_frozen_protocol(path: Path) -> tuple[dict, str]:
    path = Path(path)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise AnalysisValidationError(
            f"cannot read machine protocol {path}: {error}") from error
    protocol = _json_load_bytes(payload, field=str(path))
    if protocol.get("schema_version") != 1:
        raise AnalysisValidationError("unsupported machine protocol schema")
    if protocol.get("status") != "FROZEN":
        raise AnalysisValidationError(
            "analysis requires the machine protocol status to be FROZEN")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise AnalysisValidationError("machine protocol_id mismatch")
    if protocol.get("domain") != DOMAIN:
        raise AnalysisValidationError("machine protocol domain mismatch")

    shared = _require_object(protocol.get("shared_training"), field="protocol.shared_training")
    primary = _require_object(protocol.get("primary"), field="protocol.primary")
    analysis = _require_object(protocol.get("analysis"), field="protocol.analysis")
    if shared.get("seeds") != list(EXPECTED_SEEDS):
        raise AnalysisValidationError(
            "frozen primary protocol must use exact seeds [1, 2, 3, 4, 5]")
    if primary.get("arms") != list(ARMS):
        raise AnalysisValidationError("frozen primary protocol arm order mismatch")
    if primary.get("evidence_status") != FULL_EVIDENCE_STATUS:
        raise AnalysisValidationError("frozen primary evidence_status mismatch")
    if analysis.get("primary_currency") != "sim_steps":
        raise AnalysisValidationError("primary currency must be sim_steps")
    if analysis.get("descriptive_currency") != "training_wall_seconds":
        raise AnalysisValidationError(
            "descriptive currency must be training_wall_seconds")
    if analysis.get("gate_comparators") != list(GATE_COMPARATORS):
        raise AnalysisValidationError("frozen gate comparator contract mismatch")
    expected_analyzer_sha = _require_sha256(
        analysis.get("analyzer_sha256"),
        field="protocol.analysis.analyzer_sha256")
    executing_analyzer_sha = _sha256_file(Path(__file__).resolve())
    if not hmac.compare_digest(expected_analyzer_sha, executing_analyzer_sha):
        raise AnalysisValidationError(
            "frozen protocol analyzer_sha256 differs from executing analyzer")
    _require_int(analysis.get("bootstrap_draws"), field="protocol.analysis.bootstrap_draws", minimum=1)
    _require_int(analysis.get("bootstrap_seed"), field="protocol.analysis.bootstrap_seed")
    return protocol, _sha256_bytes(payload)


def _require_config_value(config: Mapping[str, Any], field: str, expected: Any) -> None:
    if config.get(field) != expected:
        raise AnalysisValidationError(
            f"config.{field} mismatch: expected {expected!r}, "
            f"got {config.get(field)!r}")


def _validate_contract(
    artifact: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    protocol_sha256: str,
    analyzer_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], float, int, int]:
    if artifact.get("schema_version") != 1:
        raise AnalysisValidationError("unsupported artifact schema_version")
    if artifact.get("evidence_status") != FULL_EVIDENCE_STATUS:
        raise AnalysisValidationError(
            "only a full_barn_campaign artifact may be analyzed")
    if artifact.get("domain") != DOMAIN:
        raise AnalysisValidationError("artifact domain mismatch")
    if not isinstance(artifact.get("heldout_protocol"), str) or not artifact.get(
            "heldout_protocol"):
        raise AnalysisValidationError("artifact is missing heldout_protocol")

    merge = _require_object(artifact.get("merge"), field="merge")
    if merge.get("schema_version") != 1 or merge.get("outcome_blind") is not True:
        raise AnalysisValidationError(
            "artifact lacks the strict outcome-blind merger marker")
    if merge.get("expected_seed_list") != list(EXPECTED_SEEDS):
        raise AnalysisValidationError(
            "merge.expected_seed_list must be exact seeds [1, 2, 3, 4, 5]")
    if merge.get("input_artifact_count") != len(EXPECTED_SEEDS):
        raise AnalysisValidationError("merge.input_artifact_count must be 5")
    _validate_selection_binding(merge, campaign_cell="primary")

    shared = _require_object(protocol.get("shared_training"), field="protocol.shared_training")
    primary = _require_object(protocol.get("primary"), field="protocol.primary")
    dataset = _require_object(protocol.get("dataset"), field="protocol.dataset")
    environment = _require_object(protocol.get("environment"), field="protocol.environment")
    analysis = _require_object(protocol.get("analysis"), field="protocol.analysis")
    expected_orders = primary.get("execution_order_by_seed")
    expected_order_rows = [
        {
            "seed": seed,
            "execution_order": expected_orders.get(str(seed)),
        }
        for seed in EXPECTED_SEEDS
    ] if isinstance(expected_orders, dict) else None
    for row in expected_order_rows or []:
        if (not isinstance(row["execution_order"], list)
                or sorted(row["execution_order"]) != sorted(ARMS)):
            raise AnalysisValidationError(
                f"invalid protocol execution order for seed {row['seed']}")
    if merge.get("per_seed_execution_order") != expected_order_rows:
        raise AnalysisValidationError(
            "merge.per_seed_execution_order is missing or disagrees with "
            "the frozen protocol")

    config = _require_object(artifact.get("config"), field="config")
    if "execution_order" in config:
        raise AnalysisValidationError(
            "merged config must not collapse seed-specific execution_order")
    expected_config = {
        "arms": list(ARMS),
        "campaign_cell": "primary",
        "protocol_id": PROTOCOL_ID,
        "seeds": len(EXPECTED_SEEDS),
        "seed_start": EXPECTED_SEEDS[0],
        "seed_list": list(EXPECTED_SEEDS),
        "n_rollouts": primary.get("n_rollouts"),
        "tasks_per_step": shared.get("tasks_per_step"),
        "eval_episodes": shared.get("eval_episodes"),
        "training_sim_step_budget": shared.get("training_sim_step_budget"),
        "eval_sim_step_interval": shared.get("eval_sim_step_interval"),
        "max_training_updates": shared.get("max_training_updates"),
        "steps": shared.get("max_training_updates"),
        "eval_every": shared.get("eval_every"),
        "teacher_floor": shared.get("teacher_floor"),
        "teacher_decay": shared.get("teacher_decay"),
        "teacher_gamma": shared.get("teacher_gamma"),
        "staged_initial_strata": shared.get("staged_initial_strata"),
        "staged_promotion_threshold": shared.get("staged_promotion_threshold"),
        "staged_min_frontier_groups": shared.get("staged_min_frontier_groups"),
        "n_strata": dataset.get("n_strata"),
        "n_train_courses": dataset.get("n_train_courses"),
        "n_heldout_courses": dataset.get("n_heldout_courses"),
        "episode_timeout": environment.get("episode_timeout"),
        "max_step_size": environment.get("max_step_size"),
        "real_time_update_rate": environment.get("real_time_update_rate"),
        "teacher_unit": "frozen_difficulty_stratum",
        "evaluation_partition": "frozen_heldout",
        "smoke": False,
    }
    for field, expected in expected_config.items():
        _require_config_value(config, field, expected)
    budget = _require_number(
        config.get("training_sim_step_budget"),
        field="config.training_sim_step_budget",
    )
    if budget <= 0.0:
        raise AnalysisValidationError("primary simulator-step budget must be positive")

    provenance = _require_object(artifact.get("provenance"), field="provenance")
    frozen = {
        field: _require_sha256(provenance.get(field), field=f"provenance.{field}")
        for field in FROZEN_PROVENANCE_HASHES
    }
    for field, value in provenance.items():
        if str(field).endswith("_sha256"):
            _require_sha256(value, field=f"provenance.{field}")
    if provenance.get("asset_hashes_verified") is not True:
        raise AnalysisValidationError(
            "provenance.asset_hashes_verified must be true")
    expected_provenance = {
        "manifest_sha256": dataset.get("manifest_sha256"),
        "split_sha256": dataset.get("split_sha256"),
        "container_sha256": environment.get("container_sha256"),
        "protocol_sha256": protocol_sha256,
        "analyzer_sha256": analyzer_sha256,
    }
    for field, expected in expected_provenance.items():
        expected_digest = _require_sha256(expected, field=f"protocol {field}")
        if not hmac.compare_digest(frozen[field], expected_digest):
            raise AnalysisValidationError(
                f"provenance.{field} does not match the executing frozen input")
    bound_manifest = _require_sha256(
        provenance.get("split_bound_manifest_sha256"),
        field="provenance.split_bound_manifest_sha256",
    )
    if not hmac.compare_digest(bound_manifest, frozen["manifest_sha256"]):
        raise AnalysisValidationError("split is not bound to the frozen manifest")

    results = _require_object(artifact.get("results"), field="results")
    if set(results) != set(ARMS):
        raise AnalysisValidationError("results must contain exactly four frozen arms")
    for arm in ARMS:
        mapped = _seed_map(artifact, arm)
        if tuple(sorted(mapped)) != EXPECTED_SEEDS:
            raise AnalysisValidationError(
                f"results.{arm} must contain exact seeds {list(EXPECTED_SEEDS)}")
        for seed, run in mapped.items():
            if run.get("arm") != arm:
                raise AnalysisValidationError(
                    f"results.{arm} seed {seed} has a mismatched arm label")

    draws = _require_int(
        analysis.get("bootstrap_draws"),
        field="protocol.analysis.bootstrap_draws",
        minimum=1,
    )
    bootstrap_seed = _require_int(
        analysis.get("bootstrap_seed"), field="protocol.analysis.bootstrap_seed")
    return config, provenance, budget, draws, bootstrap_seed


def _validate_eval(
    evaluation: Mapping[str, Any],
    *,
    field: str,
    n_heldout: int,
    n_strata: int,
) -> tuple:
    mean_success = _require_probability(
        evaluation.get("mean_success"), field=f"{field}.mean_success")
    eval_episodes = _require_int(
        evaluation.get("eval_episodes"), field=f"{field}.eval_episodes")
    eval_sim_steps = _require_int(
        evaluation.get("eval_sim_steps"), field=f"{field}.eval_sim_steps")
    if eval_episodes < n_heldout:
        raise AnalysisValidationError(f"{field} has too few held-out episodes")

    courses = _require_list(evaluation.get("per_course"), field=f"{field}.per_course")
    if len(courses) != n_heldout:
        raise AnalysisValidationError(
            f"{field}.per_course must contain {n_heldout} courses")
    course_signature = []
    successes = 0
    course_episodes = 0
    course_sim_steps = 0
    rates = []
    previous_order = None
    for index, value in enumerate(courses):
        course = _require_object(value, field=f"{field}.per_course[{index}]")
        env_id = course.get("env_id")
        if not isinstance(env_id, str) or not env_id:
            raise AnalysisValidationError(
                f"{field}.per_course[{index}].env_id must be non-empty")
        difficulty = _require_number(
            course.get("difficulty"), field=f"{field}.per_course[{index}].difficulty")
        order = (difficulty, env_id)
        if previous_order is not None and order < previous_order:
            raise AnalysisValidationError(
                f"{field}.per_course is not in frozen difficulty order")
        previous_order = order
        barn_index = _require_int(
            course.get("barn_index"),
            field=f"{field}.per_course[{index}].barn_index",
        )
        evaluation_seed = _require_int(
            course.get("seed"), field=f"{field}.per_course[{index}].seed")
        stratum = _require_int(
            course.get("stratum"), field=f"{field}.per_course[{index}].stratum")
        if stratum >= n_strata:
            raise AnalysisValidationError(f"{field} has an invalid course stratum")
        episodes = _require_int(
            course.get("episodes"), field=f"{field}.per_course[{index}].episodes", minimum=1)
        count = _require_int(
            course.get("successes"), field=f"{field}.per_course[{index}].successes")
        if count > episodes:
            raise AnalysisValidationError(f"{field} has successes > episodes")
        rate = _require_probability(
            course.get("success_rate"), field=f"{field}.per_course[{index}].success_rate")
        if not math.isclose(rate, count / episodes, abs_tol=1e-12):
            raise AnalysisValidationError(f"{field} has inconsistent course success_rate")
        sim_steps = _require_int(
            course.get("sim_steps"), field=f"{field}.per_course[{index}].sim_steps")
        successes += count
        course_episodes += episodes
        course_sim_steps += sim_steps
        rates.append(rate)
        course_signature.append((
            env_id, difficulty, barn_index, stratum, evaluation_seed, episodes,
        ))
    if course_episodes != eval_episodes or course_sim_steps != eval_sim_steps:
        raise AnalysisValidationError(f"{field} course resource totals disagree")
    if not math.isclose(mean_success, successes / eval_episodes, abs_tol=1e-12):
        raise AnalysisValidationError(f"{field}.mean_success disagrees with courses")

    course_ids = evaluation.get("heldout_course_ids")
    if course_ids != [row[0] for row in course_signature]:
        raise AnalysisValidationError(f"{field}.heldout_course_ids mismatch")
    per_task = _require_list(
        evaluation.get("per_task_success"), field=f"{field}.per_task_success")
    if len(per_task) != n_heldout or any(
        not math.isclose(
            _require_probability(value, field=f"{field}.per_task_success"),
            rates[index],
            abs_tol=1e-12,
        )
        for index, value in enumerate(per_task)
    ):
        raise AnalysisValidationError(f"{field}.per_task_success mismatch")

    bins = _require_list(
        evaluation.get("difficulty_bins"), field=f"{field}.difficulty_bins")
    if len(bins) != 10:
        raise AnalysisValidationError(f"{field} must contain ten difficulty deciles")
    bin_signature = []
    split_indices = np.array_split(np.arange(n_heldout), 10)
    for bin_id, (value, indices) in enumerate(zip(bins, split_indices)):
        row = _require_object(value, field=f"{field}.difficulty_bins[{bin_id}]")
        if row.get("bin") != bin_id or row.get("n_courses") != len(indices):
            raise AnalysisValidationError(f"{field} has invalid difficulty-bin metadata")
        low = _require_number(row.get("difficulty_min"), field=f"{field}.difficulty_bins[{bin_id}].min")
        high = _require_number(row.get("difficulty_max"), field=f"{field}.difficulty_bins[{bin_id}].max")
        observed = _require_probability(
            row.get("mean_success"), field=f"{field}.difficulty_bins[{bin_id}].mean_success")
        expected = float(np.mean(np.asarray(rates)[indices]))
        if (not math.isclose(low, course_signature[int(indices[0])][1], abs_tol=1e-12)
                or not math.isclose(high, course_signature[int(indices[-1])][1], abs_tol=1e-12)
                or not math.isclose(observed, expected, abs_tol=1e-12)):
            raise AnalysisValidationError(f"{field} difficulty-bin values mismatch")
        bin_signature.append((bin_id, len(indices), low, high))
    success_by_bin = evaluation.get("success_by_difficulty_bin")
    if success_by_bin != [row["mean_success"] for row in bins]:
        raise AnalysisValidationError(f"{field}.success_by_difficulty_bin mismatch")

    easy_count = max(1, math.ceil(n_heldout / 10))
    easy = _require_probability(
        evaluation.get("easy_decile_retention"),
        field=f"{field}.easy_decile_retention",
    )
    if not math.isclose(easy, float(np.mean(rates[:easy_count])), abs_tol=1e-12):
        raise AnalysisValidationError(f"{field}.easy_decile_retention mismatch")

    statuses = _count_map(evaluation.get("status_counts"), field=f"{field}.status_counts")
    if sum(statuses.values()) != eval_episodes:
        raise AnalysisValidationError(f"{field}.status_counts total mismatch")
    records = _require_list(
        evaluation.get("episode_records"), field=f"{field}.episode_records")
    if len(records) != eval_episodes:
        raise AnalysisValidationError(f"{field}.episode_records count mismatch")
    record_statuses = Counter()
    record_successes = 0
    record_sim_steps = 0
    for index, value in enumerate(records):
        record = _require_object(value, field=f"{field}.episode_records[{index}]")
        status = record.get("status")
        if not isinstance(status, str):
            raise AnalysisValidationError(f"{field} episode status must be a string")
        record_statuses[status] += 1
        record_successes += int(_require_probability(
            record.get("success"),
            field=f"{field}.episode_records[{index}].success"))
        record_sim_steps += _require_int(
            record.get("sim_steps"),
            field=f"{field}.episode_records[{index}].sim_steps")
    if dict(sorted(record_statuses.items())) != statuses:
        raise AnalysisValidationError(f"{field} episode status receipt mismatch")
    if record_successes != successes or record_sim_steps != eval_sim_steps:
        raise AnalysisValidationError(f"{field} episode outcome/resource receipt mismatch")

    calibration_n = _require_int(
        evaluation.get("teacher/calibration_n"),
        field=f"{field}.teacher/calibration_n",
    )
    for key in ("teacher/calibration_bias", "teacher/calibration_mae"):
        if calibration_n == 0:
            if key in evaluation:
                _require_number(evaluation[key], field=f"{field}.{key}")
        else:
            _require_number(evaluation.get(key), field=f"{field}.{key}")
    return tuple(course_signature), tuple(bin_signature)


def _validate_run(
    run: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    config: Mapping[str, Any],
    budget: float,
) -> tuple:
    history = _require_list(run.get("history"), field=f"{arm}[{seed}].history")
    if len(history) < 2:
        raise AnalysisValidationError(f"{arm} seed {seed} needs at least two checkpoints")
    if run.get("final") != history[-1]:
        raise AnalysisValidationError(f"{arm} seed {seed} final != history[-1]")
    n_heldout = _require_int(config.get("n_heldout_courses"), field="config.n_heldout_courses", minimum=1)
    n_strata = _require_int(config.get("n_strata"), field="config.n_strata", minimum=1)
    n_rollouts = _require_int(config.get("n_rollouts"), field="config.n_rollouts", minimum=2)
    tasks_per_step = _require_int(config.get("tasks_per_step"), field="config.tasks_per_step", minimum=1)
    previous = {"step": -1.0, "episodes": -1.0, "sim_steps": -1.0,
                "training_wall_seconds": -1.0, "evaluation_wall_seconds": -1.0}
    eval_signature = None
    for index, value in enumerate(history):
        row = _require_object(value, field=f"{arm}[{seed}].history[{index}]")
        for key in previous:
            if key in {"step", "episodes", "sim_steps"}:
                current = float(_require_int(
                    row.get(key), field=f"{arm}[{seed}].history[{index}].{key}"))
            else:
                current = _require_number(
                    row.get(key), field=f"{arm}[{seed}].history[{index}].{key}")
            if current < 0.0 or current < previous[key]:
                raise AnalysisValidationError(
                    f"{arm} seed {seed} has non-monotone {key}")
            previous[key] = current
        if index == 0 and any(
            _require_number(row.get(key), field=key) != 0.0
            for key in ("step", "episodes", "sim_steps", "training_wall_seconds")
        ):
            raise AnalysisValidationError(f"{arm} seed {seed} must start at zero training budget")
        for key in ("dead_group_rate", "all_fail_group_rate", "all_pass_group_rate"):
            _require_probability(row.get(key), field=f"{arm}[{seed}].history[{index}].{key}")
        for key in RESOURCE_FIELDS[4:] + UPDATE_FIELDS:
            _require_int(row.get(key), field=f"{arm}[{seed}].history[{index}].{key}")
        sampled = _count_map(
            row.get("sampled_stratum_counts"),
            field=f"{arm}[{seed}].history[{index}].sampled_stratum_counts",
        )
        if any(not key.isdigit() or int(key) >= n_strata for key in sampled):
            raise AnalysisValidationError(f"{arm} seed {seed} has invalid sampled stratum")
        training_status = _count_map(
            row.get("training_status_counts"),
            field=f"{arm}[{seed}].history[{index}].training_status_counts",
        )
        training_course = _count_map(
            row.get("training_course_counts"),
            field=f"{arm}[{seed}].history[{index}].training_course_counts",
        )
        episodes = int(row["episodes"])
        step = int(row["step"])
        groups = int(row["live_groups"]) + int(row["dead_groups"])
        if (sum(sampled.values()) != groups
                or groups != step * tasks_per_step
                or episodes != groups * n_rollouts
                or sum(training_status.values()) != episodes
                or sum(training_course.values()) != episodes
                or int(row["dead_groups"]) != (
                    int(row["all_fail_groups"]) + int(row["all_pass_groups"]))
                or int(row["updates_live"]) != int(row["live_groups"])
                or int(row["updates_relabel"]) != int(row["relabeled_groups"])):
            raise AnalysisValidationError(
                f"{arm} seed {seed} has inconsistent group/update accounting")
        evaluation = _require_object(
            row.get("eval"), field=f"{arm}[{seed}].history[{index}].eval")
        signature = _validate_eval(
            evaluation,
            field=f"{arm}[{seed}].history[{index}].eval",
            n_heldout=n_heldout,
            n_strata=n_strata,
        )
        if eval_signature is None:
            eval_signature = signature
        elif signature != eval_signature:
            raise AnalysisValidationError(
                f"{arm} seed {seed} held-out panel drifted across checkpoints")
    if float(history[-1]["sim_steps"]) < budget:
        raise AnalysisValidationError(f"{arm} seed {seed} did not reach the frozen budget")
    if any(float(row["sim_steps"]) >= budget for row in history[:-1]):
        raise AnalysisValidationError(
            f"{arm} seed {seed} has checkpoints after reaching the frozen budget")
    records = _require_list(
        run.get("training_episode_records"),
        field=f"{arm}[{seed}].training_episode_records",
    )
    if len(records) != int(history[-1]["episodes"]):
        raise AnalysisValidationError(
            f"{arm} seed {seed} training episode receipt count mismatch")
    return eval_signature


def _paired_summary(
    ours: dict[int, dict],
    other: dict[int, dict],
    currency: str,
    *,
    fixed_budget: float | None = None,
    bootstrap_draws: int = 20_000,
    bootstrap_seed: int = 20270811,
) -> dict:
    if tuple(sorted(ours)) != EXPECTED_SEEDS or tuple(sorted(other)) != EXPECTED_SEEDS:
        raise AnalysisValidationError("paired contrast requires exact five seeds")
    delta = []
    common_budgets = []
    for seed in EXPECTED_SEEDS:
        available = min(
            float(ours[seed]["history"][-1][currency]),
            float(other[seed]["history"][-1][currency]),
        )
        budget = available if fixed_budget is None else float(fixed_budget)
        if budget <= 0.0 or budget > available + 1e-12:
            raise AnalysisValidationError(
                f"seed {seed} ends at {available} {currency}, below budget {budget}")
        common_budgets.append(budget)
        delta.append(
            auc_at_budget(ours[seed], currency, budget)
            - auc_at_budget(other[seed], currency, budget)
        )
    differences = np.asarray(delta, dtype=float)
    return {
        "currency": currency,
        "paired_seeds": list(EXPECTED_SEEDS),
        "common_budget_per_seed": common_budgets,
        "n": len(differences),
        "mean_delta": float(differences.mean()),
        "positive": int(np.sum(differences > 0)),
        "ties": int(np.sum(differences == 0)),
        "paired_bootstrap_95_ci": paired_bootstrap_ci(
            differences, draws=bootstrap_draws, seed=bootstrap_seed,
        ),
        "exact_two_sided_sign_flip_p": exact_sign_flip_p(differences),
        "per_seed_delta": [
            {"seed": seed, "value": float(value)}
            for seed, value in zip(EXPECTED_SEEDS, differences)
        ],
    }


def _evaluation_scalar(run: Mapping[str, Any], key: str) -> float:
    evaluation = _require_object(run["final"].get("eval"), field="final.eval")
    return _require_number(evaluation.get(key), field=f"final.eval.{key}")


def _arm_report(rows: Mapping[int, dict], *, budget: float) -> dict:
    primary = {
        seed: auc_at_budget(rows[seed], "sim_steps", budget)
        for seed in EXPECTED_SEEDS
    }
    final_success = {
        seed: _evaluation_scalar(rows[seed], "mean_success")
        for seed in EXPECTED_SEEDS
    }
    easy_final = {
        seed: _evaluation_scalar(rows[seed], "easy_decile_retention")
        for seed in EXPECTED_SEEDS
    }
    easy_auc = {
        seed: _curve_auc_at_budget(
            rows[seed], "sim_steps", budget,
            lambda row: _require_probability(
                _require_object(row.get("eval"), field="history.eval").get(
                    "easy_decile_retention"),
                field="history.eval.easy_decile_retention",
            ),
        )
        for seed in EXPECTED_SEEDS
    }

    difficulty_deciles = []
    for bin_id in range(10):
        def bin_success(history_row: Mapping[str, Any], index: int = bin_id) -> float:
            evaluation = _require_object(history_row.get("eval"), field="history.eval")
            bins = _require_list(evaluation.get("difficulty_bins"), field="eval.difficulty_bins")
            return _require_probability(
                _require_object(bins[index], field=f"difficulty_bins[{index}]").get(
                    "mean_success"),
                field=f"difficulty_bins[{index}].mean_success",
            )

        first_bin = rows[EXPECTED_SEEDS[0]]["final"]["eval"]["difficulty_bins"][bin_id]
        difficulty_deciles.append({
            "decile": bin_id,
            "n_courses": first_bin["n_courses"],
            "difficulty_min": first_bin["difficulty_min"],
            "difficulty_max": first_bin["difficulty_max"],
            "primary_auc_at_frozen_sim_steps": _value_summary({
                seed: _curve_auc_at_budget(
                    rows[seed], "sim_steps", budget, bin_success,
                )
                for seed in EXPECTED_SEEDS
            }),
            "final_mean_success": _value_summary({
                seed: bin_success(rows[seed]["final"])
                for seed in EXPECTED_SEEDS
            }),
        })

    final_eval = {
        seed: _require_object(rows[seed]["final"].get("eval"), field="final.eval")
        for seed in EXPECTED_SEEDS
    }
    coverage_keys = sorted(
        set.intersection(*[
            {key for key in evaluation if key.startswith("success@")}
            for evaluation in final_eval.values()
        ])
    )
    final_report = {
        "mean_success": _value_summary(final_success),
        "coverage": {
            key: _value_summary({
                seed: _require_probability(final_eval[seed][key], field=key)
                for seed in EXPECTED_SEEDS
            })
            for key in coverage_keys
        },
        "easy_decile_retention": _value_summary(easy_final),
    }

    calibration = {
        "n_strata": _value_summary({
            seed: _require_int(
                final_eval[seed].get("teacher/calibration_n"),
                field="teacher/calibration_n",
            )
            for seed in EXPECTED_SEEDS
        }),
    }
    for metric in ("teacher/calibration_bias", "teacher/calibration_mae"):
        values: dict[int, float | None] = {}
        for seed in EXPECTED_SEEDS:
            n_calibrated = final_eval[seed]["teacher/calibration_n"]
            values[seed] = (
                _require_number(final_eval[seed].get(metric), field=metric)
                if n_calibrated > 0 else None
            )
        calibration[metric.removeprefix("teacher/")] = _nullable_value_summary(values)

    per_seed_accounting = []
    sampled_maps = []
    course_maps = []
    training_status_maps = []
    evaluation_status_maps = []
    for seed in EXPECTED_SEEDS:
        final = rows[seed]["final"]
        sampled = _count_map(final["sampled_stratum_counts"], field="sampled_stratum_counts")
        courses = _count_map(final["training_course_counts"], field="training_course_counts")
        train_status = _count_map(final["training_status_counts"], field="training_status_counts")
        eval_status = _count_map(final_eval[seed]["status_counts"], field="eval.status_counts")
        sampled_maps.append(sampled)
        course_maps.append(courses)
        training_status_maps.append(train_status)
        evaluation_status_maps.append(eval_status)
        per_seed_accounting.append({
            "seed": seed,
            "sampled_stratum_counts": sampled,
            "training_course_counts": courses,
            "training_status_counts": train_status,
            "evaluation_status_counts": eval_status,
            "resources": {
                field: _require_number(final.get(field), field=field)
                for field in RESOURCE_FIELDS
            },
            "updates": {
                field: _require_int(final.get(field), field=field)
                for field in UPDATE_FIELDS
            },
            "evaluation_resources": {
                "eval_episodes": _require_int(
                    final_eval[seed].get("eval_episodes"), field="eval_episodes"),
                "eval_sim_steps": _require_int(
                    final_eval[seed].get("eval_sim_steps"), field="eval_sim_steps"),
            },
        })
    resource_summaries = {
        field: _value_summary({
            row["seed"]: row["resources"][field]
            for row in per_seed_accounting
        })
        for field in RESOURCE_FIELDS
    }
    update_summaries = {
        field: _value_summary({
            row["seed"]: row["updates"][field]
            for row in per_seed_accounting
        })
        for field in UPDATE_FIELDS
    }

    course_outcomes = []
    reference_courses = final_eval[EXPECTED_SEEDS[0]]["per_course"]
    for index, reference in enumerate(reference_courses):
        per_seed_rates = {
            seed: _require_probability(
                final_eval[seed]["per_course"][index]["success_rate"],
                field="per_course.success_rate",
            )
            for seed in EXPECTED_SEEDS
        }
        course_outcomes.append({
            "env_id": reference["env_id"],
            "difficulty": reference["difficulty"],
            "stratum": reference["stratum"],
            "final_success": _value_summary(per_seed_rates),
            "total_eval_episodes": sum(
                int(final_eval[seed]["per_course"][index]["episodes"])
                for seed in EXPECTED_SEEDS
            ),
            "total_eval_sim_steps": sum(
                int(final_eval[seed]["per_course"][index]["sim_steps"])
                for seed in EXPECTED_SEEDS
            ),
        })

    return {
        "n": len(EXPECTED_SEEDS),
        "primary_auc_at_frozen_sim_steps": {
            "budget": budget,
            **_value_summary(primary),
        },
        "final": final_report,
        "difficulty_deciles": difficulty_deciles,
        "easy_decile": {
            "auc_at_frozen_sim_steps": _value_summary(easy_auc),
            "final_retention": _value_summary(easy_final),
        },
        "group_outcomes": {
            "dead_group_rate": _value_summary({
                seed: _require_probability(
                    rows[seed]["final"]["dead_group_rate"],
                    field="dead_group_rate",
                ) for seed in EXPECTED_SEEDS
            }),
            "all_fail_group_rate": _value_summary({
                seed: _require_probability(
                    rows[seed]["final"]["all_fail_group_rate"],
                    field="all_fail_group_rate",
                ) for seed in EXPECTED_SEEDS
            }),
            "all_pass_group_rate": _value_summary({
                seed: _require_probability(
                    rows[seed]["final"]["all_pass_group_rate"],
                    field="all_pass_group_rate",
                ) for seed in EXPECTED_SEEDS
            }),
        },
        "posterior_calibration": calibration,
        "sampling_status_resource_update_accounting": {
            "per_seed": per_seed_accounting,
            "aggregate_sampled_stratum_counts": _sum_maps(sampled_maps),
            "aggregate_training_course_counts": _sum_maps(course_maps),
            "aggregate_training_status_counts": _sum_maps(training_status_maps),
            "aggregate_evaluation_status_counts": _sum_maps(evaluation_status_maps),
            "resource_summaries": resource_summaries,
            "update_summaries": update_summaries,
        },
        "final_course_outcomes": course_outcomes,
    }


def _panel_metadata_without_eval_seed(signature: tuple) -> tuple:
    courses, bins = signature
    return (
        tuple((env_id, difficulty, barn_index, stratum, episodes)
              for (env_id, difficulty, barn_index, stratum, _eval_seed,
                   episodes) in courses),
        bins,
    )


def _validate_ablation_artifact(
    artifact: Mapping[str, Any],
    *,
    n_rollouts: int,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    analyzer_sha256: str,
) -> tuple[dict[str, dict[int, dict]], Mapping[str, Any], float]:
    """Validate one fresh, strict two-arm N-ablation merged artifact."""

    if n_rollouts not in FRESH_ABLATION_N:
        if n_rollouts == 8:
            raise AnalysisValidationError(
                "a fresh N=8 ablation source is forbidden; reuse primary N=8")
        raise AnalysisValidationError(f"unexpected fresh N={n_rollouts} ablation")
    if artifact.get("schema_version") != 1:
        raise AnalysisValidationError("unsupported ablation artifact schema_version")
    if artifact.get("evidence_status") != ABLATION_EVIDENCE_STATUS:
        raise AnalysisValidationError(
            "fresh ablation artifact must have full_barn_n_ablation status")
    if artifact.get("domain") != DOMAIN:
        raise AnalysisValidationError("ablation artifact domain mismatch")
    if not isinstance(artifact.get("heldout_protocol"), str) or not artifact.get(
            "heldout_protocol"):
        raise AnalysisValidationError("ablation artifact is missing heldout_protocol")

    ablation = _require_object(protocol.get("ablation"), field="protocol.ablation")
    shared = _require_object(protocol.get("shared_training"), field="protocol.shared_training")
    dataset = _require_object(protocol.get("dataset"), field="protocol.dataset")
    environment = _require_object(protocol.get("environment"), field="protocol.environment")
    if ablation.get("arms") != ["ours_uN", "learnability"]:
        raise AnalysisValidationError("protocol ablation arm contract mismatch")
    if ablation.get("n_values") != [2, 4, 8, 16]:
        raise AnalysisValidationError("protocol ablation N values mismatch")
    if ablation.get("fresh_cell_names") != [
            "ablation_n2", "ablation_n4", "ablation_n16"]:
        raise AnalysisValidationError("protocol fresh ablation cells mismatch")
    if ablation.get("n8_source") != "primary_ours_uN_and_learnability":
        raise AnalysisValidationError("protocol N=8 reuse contract mismatch")

    cell = f"ablation_n{n_rollouts}"
    merge = _require_object(artifact.get("merge"), field=f"{cell}.merge")
    if merge.get("schema_version") != 1 or merge.get("outcome_blind") is not True:
        raise AnalysisValidationError(
            f"{cell} lacks the strict outcome-blind merger marker")
    if (merge.get("expected_seed_list") != list(EXPECTED_SEEDS)
            or merge.get("input_artifact_count") != len(EXPECTED_SEEDS)):
        raise AnalysisValidationError(f"{cell} merge does not contain exact five seeds")
    _validate_selection_binding(merge, campaign_cell=cell)
    execution_orders = ablation.get("execution_order_by_seed")
    expected_order_rows = [
        {
            "seed": seed,
            "execution_order": execution_orders.get(str(seed)),
        }
        for seed in EXPECTED_SEEDS
    ] if isinstance(execution_orders, dict) else None
    for row in expected_order_rows or []:
        if (not isinstance(row["execution_order"], list)
                or sorted(row["execution_order"]) != ["learnability", "ours_uN"]):
            raise AnalysisValidationError(
                f"invalid protocol ablation execution order for seed {row['seed']}")
    if merge.get("per_seed_execution_order") != expected_order_rows:
        raise AnalysisValidationError(
            f"{cell} per-seed execution-order receipt mismatch")

    config = _require_object(artifact.get("config"), field=f"{cell}.config")
    if "execution_order" in config:
        raise AnalysisValidationError(
            f"{cell} merged config must not collapse execution_order")
    expected_config = {
        "arms": ["ours_uN", "learnability"],
        "campaign_cell": cell,
        "protocol_id": PROTOCOL_ID,
        "seeds": 5,
        "seed_start": 1,
        "seed_list": list(EXPECTED_SEEDS),
        "n_rollouts": n_rollouts,
        "tasks_per_step": shared.get("tasks_per_step"),
        "eval_episodes": shared.get("eval_episodes"),
        "training_sim_step_budget": shared.get("training_sim_step_budget"),
        "eval_sim_step_interval": shared.get("eval_sim_step_interval"),
        "max_training_updates": shared.get("max_training_updates"),
        "steps": shared.get("max_training_updates"),
        "eval_every": shared.get("eval_every"),
        "teacher_floor": shared.get("teacher_floor"),
        "teacher_decay": shared.get("teacher_decay"),
        "teacher_gamma": shared.get("teacher_gamma"),
        "staged_initial_strata": shared.get("staged_initial_strata"),
        "staged_promotion_threshold": shared.get("staged_promotion_threshold"),
        "staged_min_frontier_groups": shared.get("staged_min_frontier_groups"),
        "n_strata": dataset.get("n_strata"),
        "n_train_courses": dataset.get("n_train_courses"),
        "n_heldout_courses": dataset.get("n_heldout_courses"),
        "episode_timeout": environment.get("episode_timeout"),
        "max_step_size": environment.get("max_step_size"),
        "real_time_update_rate": environment.get("real_time_update_rate"),
        "teacher_unit": "frozen_difficulty_stratum",
        "evaluation_partition": "frozen_heldout",
        "smoke": False,
    }
    for field, expected in expected_config.items():
        _require_config_value(config, field, expected)
    budget = _require_number(
        config.get("training_sim_step_budget"),
        field=f"{cell}.config.training_sim_step_budget",
    )
    if budget <= 0.0:
        raise AnalysisValidationError(f"{cell} simulator-step budget must be positive")

    provenance = _require_object(
        artifact.get("provenance"), field=f"{cell}.provenance")
    frozen = {
        field: _require_sha256(
            provenance.get(field), field=f"{cell}.provenance.{field}")
        for field in FROZEN_PROVENANCE_HASHES
    }
    for field, value in provenance.items():
        if str(field).endswith("_sha256"):
            _require_sha256(value, field=f"{cell}.provenance.{field}")
    expected_provenance = {
        "manifest_sha256": dataset.get("manifest_sha256"),
        "split_sha256": dataset.get("split_sha256"),
        "container_sha256": environment.get("container_sha256"),
        "protocol_sha256": protocol_sha256,
        "analyzer_sha256": analyzer_sha256,
    }
    for field, expected in expected_provenance.items():
        expected_digest = _require_sha256(expected, field=f"protocol {field}")
        if not hmac.compare_digest(frozen[field], expected_digest):
            raise AnalysisValidationError(f"{cell} provenance.{field} mismatch")
    if provenance.get("asset_hashes_verified") is not True:
        raise AnalysisValidationError(f"{cell} asset hashes are not verified")
    bound_manifest = _require_sha256(
        provenance.get("split_bound_manifest_sha256"),
        field=f"{cell}.provenance.split_bound_manifest_sha256",
    )
    if not hmac.compare_digest(bound_manifest, frozen["manifest_sha256"]):
        raise AnalysisValidationError(f"{cell} split/manifest binding mismatch")

    results = _require_object(artifact.get("results"), field=f"{cell}.results")
    if set(results) != {"ours_uN", "learnability"}:
        raise AnalysisValidationError(
            f"{cell} results must contain exactly ours_uN and learnability")
    seed_maps = {
        arm: _seed_map(artifact, arm) for arm in ("ours_uN", "learnability")
    }
    signatures = {}
    for arm, mapped in seed_maps.items():
        if tuple(sorted(mapped)) != EXPECTED_SEEDS:
            raise AnalysisValidationError(f"{cell}.{arm} must contain exact five seeds")
        for seed in EXPECTED_SEEDS:
            if mapped[seed].get("arm") != arm:
                raise AnalysisValidationError(f"{cell}.{arm} seed/arm label mismatch")
            signatures[(arm, seed)] = _validate_run(
                mapped[seed], arm=f"{cell}.{arm}", seed=seed,
                config=config, budget=budget,
            )
    for seed in EXPECTED_SEEDS:
        if signatures[("ours_uN", seed)] != signatures[("learnability", seed)]:
            raise AnalysisValidationError(
                f"{cell} held-out panel is not paired for seed {seed}")
    panel_metadata = _panel_metadata_without_eval_seed(
        signatures[("ours_uN", EXPECTED_SEEDS[0])])
    for signature in signatures.values():
        if _panel_metadata_without_eval_seed(signature) != panel_metadata:
            raise AnalysisValidationError(
                f"{cell} held-out course/difficulty panel drifted across seeds")
    return seed_maps, provenance, budget


def summarize_n_ablation(
    primary_artifact: dict,
    fresh_artifacts: Sequence[dict],
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    primary_input_sha256: str | None = None,
    fresh_input_sha256: Mapping[str, str] | None = None,
) -> dict:
    """Summarize N={2,4,8,16}, reusing primary N=8 by construction.

    Exactly three fresh strict merged cells are accepted: N=2, N=4, and
    N=16.  Supplying a fresh N=8 artifact is an error, which prevents the
    protocol's shared N=8 evidence from being duplicated or cherry-picked.
    """

    protocol, protocol_sha = _load_frozen_protocol(Path(protocol_path))
    analyzer_sha = _sha256_file(Path(__file__).resolve())
    primary_report = analyze(
        primary_artifact,
        input_artifact_sha256=primary_input_sha256,
        protocol_path=protocol_path,
    )
    if not isinstance(fresh_artifacts, Sequence) or isinstance(
            fresh_artifacts, (str, bytes)):
        raise AnalysisValidationError("fresh ablation artifacts must be a sequence")

    by_n: dict[int, dict] = {}
    provenance_by_n: dict[int, Mapping[str, Any]] = {}
    budget_by_n: dict[int, float] = {}
    merge_receipts_by_n: dict[int, Mapping[str, Any]] = {
        8: primary_report["strict_merge_receipt"],
    }
    input_receipts = {}
    supplied_hashes = dict(fresh_input_sha256 or {})
    for index, artifact in enumerate(fresh_artifacts):
        artifact = dict(_require_object(artifact, field=f"fresh_artifacts[{index}]"))
        config = _require_object(
            artifact.get("config"), field=f"fresh_artifacts[{index}].config")
        cell = config.get("campaign_cell")
        if cell == "ablation_n8" or (
                cell == "primary" and config.get("n_rollouts") == 8):
            raise AnalysisValidationError(
                "a fresh N=8 ablation source is forbidden; reuse primary N=8")
        if not isinstance(cell, str) or not cell.startswith("ablation_n"):
            raise AnalysisValidationError(
                f"fresh artifact {index} has invalid campaign_cell {cell!r}")
        try:
            n_rollouts = int(cell.removeprefix("ablation_n"))
        except ValueError as error:
            raise AnalysisValidationError(
                f"fresh artifact {index} has invalid campaign_cell {cell!r}") from error
        if n_rollouts in by_n:
            raise AnalysisValidationError(f"duplicate fresh N={n_rollouts} ablation")
        seed_maps, provenance, budget = _validate_ablation_artifact(
            artifact,
            n_rollouts=n_rollouts,
            protocol=protocol,
            protocol_sha256=protocol_sha,
            analyzer_sha256=analyzer_sha,
        )
        by_n[n_rollouts] = seed_maps
        provenance_by_n[n_rollouts] = provenance
        budget_by_n[n_rollouts] = budget
        merge_receipts_by_n[n_rollouts] = _require_object(
            artifact.get("merge"), field=f"{cell}.merge")
        supplied = supplied_hashes.get(cell)
        input_receipts[cell] = {
            "sha256": (_require_sha256(supplied, field=f"{cell} input SHA-256")
                       if supplied is not None
                       else _canonical_json_sha256(artifact)),
            "hash_basis": ("exact_input_file_bytes" if supplied is not None
                           else "canonical_json_in_memory"),
        }
    if set(by_n) != set(FRESH_ABLATION_N):
        raise AnalysisValidationError(
            "fresh ablation set must be exactly N={2,4,16}; "
            f"got {sorted(by_n)}")

    primary_provenance = primary_report["frozen_provenance"]
    for n_rollouts, provenance in provenance_by_n.items():
        for field in FROZEN_PROVENANCE_HASHES:
            if not hmac.compare_digest(
                    str(provenance[field]), str(primary_provenance[field])):
                raise AnalysisValidationError(
                    f"N={n_rollouts} provenance.{field} differs from primary")

    primary_selection = _require_object(
        primary_report["strict_merge_receipt"].get("selection"),
        field="primary merge.selection")
    for n_rollouts in FRESH_ABLATION_N:
        selection = _require_object(
            merge_receipts_by_n[n_rollouts].get("selection"),
            field=f"N={n_rollouts} merge.selection")
        for field in ("campaign_id", "ledger_sha256", "rule"):
            if selection.get(field) != primary_selection.get(field):
                raise AnalysisValidationError(
                    f"N={n_rollouts} selection {field} differs from primary")

    primary_maps = {
        arm: _seed_map(primary_artifact, arm)
        for arm in ("ours_uN", "learnability")
    }
    by_n[8] = primary_maps
    budget_by_n[8] = float(primary_report["primary_sim_step_budget"])
    analysis_contract = _require_object(
        protocol.get("analysis"), field="protocol.analysis")
    draws = _require_int(
        analysis_contract.get("bootstrap_draws"),
        field="protocol.analysis.bootstrap_draws", minimum=1)
    bootstrap_seed = _require_int(
        analysis_contract.get("bootstrap_seed"),
        field="protocol.analysis.bootstrap_seed")

    cells = []
    for n_rollouts in (2, 4, 8, 16):
        maps = by_n[n_rollouts]
        budget = budget_by_n[n_rollouts]
        primary_values = {
            arm: {
                seed: auc_at_budget(maps[arm][seed], "sim_steps", budget)
                for seed in EXPECTED_SEEDS
            }
            for arm in ("ours_uN", "learnability")
        }
        cells.append({
            "n_rollouts": n_rollouts,
            "source": ("primary_ours_uN_and_learnability" if n_rollouts == 8
                       else f"fresh_ablation_n{n_rollouts}"),
            "input_artifact": (
                {
                    "sha256": primary_report["input_artifact_sha256"],
                    "hash_basis": primary_report["input_artifact_hash_basis"],
                }
                if n_rollouts == 8 else input_receipts[f"ablation_n{n_rollouts}"]
            ),
            "sim_step_budget": budget,
            "strict_merge_receipt": dict(merge_receipts_by_n[n_rollouts]),
            "arms": {
                arm: {
                    "primary_auc_at_frozen_sim_steps": _value_summary(
                        primary_values[arm]),
                    "final_mean_success": _value_summary({
                        seed: _evaluation_scalar(maps[arm][seed], "mean_success")
                        for seed in EXPECTED_SEEDS
                    }),
                }
                for arm in ("ours_uN", "learnability")
            },
            "ours_uN_minus_learnability": {
                "primary_matched_sim_steps": _paired_summary(
                    maps["ours_uN"], maps["learnability"], "sim_steps",
                    fixed_budget=budget,
                    bootstrap_draws=draws,
                    bootstrap_seed=bootstrap_seed,
                ),
                "descriptive_matched_wall": _paired_summary(
                    maps["ours_uN"], maps["learnability"],
                    "training_wall_seconds",
                    bootstrap_draws=draws,
                    bootstrap_seed=bootstrap_seed,
                ),
            },
        })
    report = {
        "analysis_schema_version": 2,
        "analysis_kind": "barn_n_ablation",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_sha,
        "analyzer_sha256": analyzer_sha,
        "n_values": [2, 4, 8, 16],
        "fresh_n_values": list(FRESH_ABLATION_N),
        "n8_source": "primary_ours_uN_and_learnability",
        "frozen_provenance": primary_provenance,
        "cells": cells,
    }
    json.dumps(report, allow_nan=False)
    return report


def analyze(
    artifact: dict,
    *,
    input_artifact_sha256: str | None = None,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> dict:
    """Analyze one strict merged primary artifact.

    ``input_artifact_sha256`` should be the exact file-byte digest.  In-memory
    callers that omit it receive an explicitly labelled canonical-JSON digest;
    the command-line path always records the exact input bytes.
    """

    protocol, protocol_sha = _load_frozen_protocol(Path(protocol_path))
    analyzer_path = Path(__file__).resolve()
    analyzer_sha = _sha256_file(analyzer_path)
    config, provenance, budget, draws, bootstrap_seed = _validate_contract(
        artifact,
        protocol,
        protocol_sha256=protocol_sha,
        analyzer_sha256=analyzer_sha,
    )
    input_hash_basis = "exact_input_file_bytes"
    if input_artifact_sha256 is None:
        input_artifact_sha256 = _canonical_json_sha256(artifact)
        input_hash_basis = "canonical_json_in_memory"
    input_artifact_sha256 = _require_sha256(
        input_artifact_sha256, field="input_artifact_sha256")

    panel_signatures = {}
    for arm in ARMS:
        mapped = _seed_map(artifact, arm)
        for seed in EXPECTED_SEEDS:
            panel_signatures[(arm, seed)] = _validate_run(
                mapped[seed], arm=arm, seed=seed, config=config, budget=budget,
            )
    for seed in EXPECTED_SEEDS:
        reference = panel_signatures[(ARMS[0], seed)]
        if any(panel_signatures[(arm, seed)] != reference for arm in ARMS[1:]):
            raise AnalysisValidationError(
                f"held-out panel metadata is not paired across arms for seed {seed}")
    panel_metadata = _panel_metadata_without_eval_seed(
        panel_signatures[(ARMS[0], EXPECTED_SEEDS[0])])
    for arm in ARMS:
        for seed in EXPECTED_SEEDS:
            if (_panel_metadata_without_eval_seed(panel_signatures[(arm, seed)])
                    != panel_metadata):
                raise AnalysisValidationError(
                    "held-out course/difficulty panel drifted across seeds")

    seed_maps = {arm: _seed_map(artifact, arm) for arm in ARMS}
    arms = {
        arm: _arm_report(seed_maps[arm], budget=budget) for arm in ARMS
    }
    contrasts = {}
    ours = seed_maps["ours_uN"]
    for comparator in COMPARATORS:
        contrasts[f"ours_uN_minus_{comparator}"] = {
            "primary_matched_sim_steps": _paired_summary(
                ours,
                seed_maps[comparator],
                "sim_steps",
                fixed_budget=budget,
                bootstrap_draws=draws,
                bootstrap_seed=bootstrap_seed,
            ),
            "descriptive_matched_wall": _paired_summary(
                ours,
                seed_maps[comparator],
                "training_wall_seconds",
                bootstrap_draws=draws,
                bootstrap_seed=bootstrap_seed,
            ),
        }

    primary_rows = [
        contrasts[f"ours_uN_minus_{comparator}"]["primary_matched_sim_steps"]
        for comparator in GATE_COMPARATORS
    ]
    directional = all(row["mean_delta"] >= 0.0 for row in primary_rows)
    frozen_provenance = {
        key: provenance[key] for key in sorted(provenance)
    }
    report = {
        "analysis_schema_version": 2,
        "input_artifact_sha256": input_artifact_sha256,
        "input_artifact_hash_basis": input_hash_basis,
        "input_evidence_status": artifact["evidence_status"],
        "analyzer_sha256": analyzer_sha,
        "protocol_sha256": protocol_sha,
        "protocol_id": PROTOCOL_ID,
        "frozen_provenance": frozen_provenance,
        "strict_merge_receipt": dict(artifact["merge"]),
        "primary_metric": (
            "target-uniform mean-success AUC at frozen simulator steps"),
        "primary_sim_step_budget": budget,
        "descriptive_wall_metric": (
            "target-uniform mean-success AUC at paired common training wall time"),
        "arms": arms,
        "paired_contrasts": contrasts,
        "aug24_checkpoint": {
            "minimum_seed_requirement_met": True,
            "directional_bar_met": directional,
            "gate_comparators": list(GATE_COMPARATORS),
            "staged_is_report_only": True,
            "decision_ready": True,
            "decision": "continue_icra" if directional else "pivot_to_ral",
            "note": (
                "The gate uses only the frozen transition-budget contrasts "
                "against uniform and learnability; staged is reported only."),
        },
    }
    # Enforce standards-compliant JSON before a caller can write the report.
    json.dumps(report, allow_nan=False)
    return report


def analyze_file(path: Path, *, protocol_path: Path = DEFAULT_PROTOCOL) -> dict:
    """Read and analyze an artifact while binding the report to exact bytes."""

    path = Path(path)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise AnalysisValidationError(f"cannot read artifact {path}: {error}") from error
    artifact = _json_load_bytes(payload, field=str(path))
    return analyze(
        artifact,
        input_artifact_sha256=_sha256_bytes(payload),
        protocol_path=protocol_path,
    )


def summarize_n_ablation_files(
    primary_path: Path,
    fresh_paths: Sequence[Path],
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> dict:
    """File-bound wrapper for the strict N-ablation summarizer."""

    primary_payload = Path(primary_path).read_bytes()
    primary = _json_load_bytes(primary_payload, field=str(primary_path))
    fresh = []
    hashes = {}
    for path in fresh_paths:
        payload = Path(path).read_bytes()
        artifact = _json_load_bytes(payload, field=str(path))
        config = _require_object(artifact.get("config"), field=f"{path}.config")
        cell = config.get("campaign_cell")
        if not isinstance(cell, str):
            raise AnalysisValidationError(f"{path} is missing config.campaign_cell")
        if cell in hashes:
            raise AnalysisValidationError(f"duplicate file input for {cell}")
        hashes[cell] = _sha256_bytes(payload)
        fresh.append(artifact)
    return summarize_n_ablation(
        primary,
        fresh,
        protocol_path=protocol_path,
        primary_input_sha256=_sha256_bytes(primary_payload),
        fresh_input_sha256=hashes,
    )


def _atomic_write_new_json(path: Path, value: object) -> None:
    """Atomically create ``path`` without ever replacing an existing file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o644)
        # A same-filesystem hard link is an atomic create-if-absent operation.
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze one strict five-seed merged BARN primary artifact")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--ablation-artifact",
        action="append",
        type=Path,
        default=[],
        help=("fresh strict merged ablation artifact; provide exactly N=2, "
              "N=4, and N=16 to summarize the N sweep"),
    )
    args = parser.parse_args(argv)
    if args.ablation_artifact:
        report = summarize_n_ablation_files(
            args.artifact, args.ablation_artifact)
        default_suffix = "_n_ablation_analysis.json"
    else:
        report = analyze_file(args.artifact)
        default_suffix = "_analysis.json"
    output = args.output or args.artifact.with_name(
        args.artifact.stem + default_suffix)
    _atomic_write_new_json(output, report)
    if report.get("analysis_kind") == "barn_n_ablation":
        print(f"wrote {output}: n_values=2,4,8,16 seeds={len(EXPECTED_SEEDS)}")
    else:
        print(
            f"wrote {output}: input_sha256={report['input_artifact_sha256']} "
            f"seeds={len(EXPECTED_SEEDS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
