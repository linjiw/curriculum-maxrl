"""Source-lock, analysis-binding, and execution-authorization checks."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .core import (
    AUTHORIZATION_SCHEMA,
    DEVELOPMENT_LRS,
    ENGINEERING_AUDIT_SCHEMA,
    EXPECTED_RUNTIME,
    LOCK_SCHEMA,
    LR_SELECTION_SCHEMA,
    PROJECT_ROOT,
    SOURCE_LOCK_PATH,
    THREAD_ENVIRONMENT_VARIABLES,
    assert_pinned_runtime,
    frozen_schedule,
    sha256_file,
    strict_json_load,
)


LOCKED_RELATIVE_PATHS = (
    "curriculum_maxrl/__init__.py",
    "curriculum_maxrl/estimators.py",
    "curriculum_maxrl/digits_factorial/PROTOCOL.md",
    "curriculum_maxrl/digits_factorial/__init__.py",
    "curriculum_maxrl/digits_factorial/analyze.py",
    "curriculum_maxrl/digits_factorial/core.py",
    "curriculum_maxrl/digits_factorial/digits_split_manifest.json",
    "curriculum_maxrl/digits_factorial/freeze_lock.py",
    "curriculum_maxrl/digits_factorial/locking.py",
    "curriculum_maxrl/digits_factorial/prepare_data.py",
    "curriculum_maxrl/digits_factorial/pyproject.toml",
    "curriculum_maxrl/digits_factorial/runner.py",
    "curriculum_maxrl/digits_factorial/tests/test_adversarial.py",
    "curriculum_maxrl/digits_factorial/tests/test_analyze.py",
    "curriculum_maxrl/digits_factorial/tests/test_core.py",
    "curriculum_maxrl/digits_factorial/tests/test_runner.py",
    "curriculum_maxrl/digits_factorial/tests/test_verifier.py",
    "curriculum_maxrl/digits_factorial/uv.lock",
    "curriculum_maxrl/digits_factorial/verify_portable.py",
)


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_project_file(relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"{label} is not a canonical project-relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise ValueError(f"{label} is not a canonical project-relative path")
    path = PROJECT_ROOT.joinpath(*pure.parts).resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes the project root") from error
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {relative}")
    return path


def live_source_manifest(root: Path = PROJECT_ROOT) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in LOCKED_RELATIVE_PATHS:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"locked source file is absent: {relative}")
        result[relative] = sha256_file(path)
    return result


def load_and_verify_source_lock(
    lock_path: Path = SOURCE_LOCK_PATH,
    *,
    check_runtime: bool = True,
    source_root: Path = PROJECT_ROOT,
) -> tuple[dict[str, Any], str]:
    lock = strict_json_load(lock_path)
    exact_keys = {
        "schema",
        "status",
        "sealed_utc",
        "purpose",
        "public_preexecution_commit",
        "public_preexecution_commit_disclosure",
        "protocol_relative_path",
        "data_manifest_relative_path",
        "data_manifest_sha256",
        "expected_runtime",
        "schedule",
        "source_sha256",
    }
    if set(lock) != exact_keys:
        raise ValueError("source lock has an unexpected field set")
    if lock.get("schema") != LOCK_SCHEMA:
        raise ValueError("source-lock schema mismatch")
    if lock.get("status") != "frozen_before_lr_development_or_confirmation":
        raise ValueError("source lock is not in the frozen pre-execution state")
    if not isinstance(lock.get("sealed_utc"), str) or not lock["sealed_utc"]:
        raise ValueError("source lock lacks a sealing timestamp")
    if lock.get("purpose") != (
        "Pre-development/pre-confirmation source, data, schedule, and runtime "
        "lock for the Digits exact-probability estimator-by-sampler factorial."
    ):
        raise ValueError("source-lock purpose differs from the frozen declaration")
    if lock.get("public_preexecution_commit") is not None or lock.get(
        "public_preexecution_commit_disclosure"
    ) != "No public pre-execution commit existed when this local lock was sealed.":
        raise ValueError("source-lock public-preexecution disclosure differs")
    if lock.get("protocol_relative_path") != "curriculum_maxrl/digits_factorial/PROTOCOL.md":
        raise ValueError("source-lock protocol path differs")
    data_relative = "curriculum_maxrl/digits_factorial/digits_split_manifest.json"
    if lock.get("data_manifest_relative_path") != data_relative:
        raise ValueError("source-lock data-manifest path differs")
    data_path = source_root / data_relative
    if not data_path.is_file() or lock.get("data_manifest_sha256") != sha256_file(data_path):
        raise ValueError("source-lock data-manifest digest differs")
    if lock.get("schedule") != frozen_schedule():
        raise ValueError("source-lock schedule differs from executable schedule")
    if lock.get("expected_runtime") != EXPECTED_RUNTIME:
        raise ValueError("source lock records an unexpected pinned runtime")
    expected_manifest = lock.get("source_sha256")
    if not isinstance(expected_manifest, dict) or set(expected_manifest) != set(
        LOCKED_RELATIVE_PATHS
    ):
        raise ValueError("source lock does not contain the exact source path set")
    if not all(_valid_sha256(value) for value in expected_manifest.values()):
        raise ValueError("source lock contains an invalid source digest")
    observed_manifest = live_source_manifest(source_root)
    if observed_manifest != expected_manifest:
        mismatches = sorted(
            path
            for path in set(observed_manifest) | set(expected_manifest)
            if observed_manifest.get(path) != expected_manifest.get(path)
        )
        raise ValueError(f"live source differs from lock: {mismatches}")
    if check_runtime:
        assert_pinned_runtime()
    return lock, sha256_file(lock_path)


def authorization_payload_digest(payload_without_digest: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload_without_digest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_review_record(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"passed", "review_sha256"}:
        raise ValueError("independent pre-seal review record has wrong schema")
    if value["passed"] is not True or not _valid_sha256(value["review_sha256"]):
        raise ValueError("independent pre-seal review is not valid and passing")


def _validate_root_record(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"authorized", "authorized_utc"}:
        raise ValueError("root authorization record has wrong schema")
    if value["authorized"] is not True or not isinstance(value["authorized_utc"], str):
        raise ValueError("root execution authorization is not valid")


def validate_engineering_audit_binding(
    binding: object, *, lock_sha256: str
) -> dict[str, Any]:
    if not isinstance(binding, dict) or set(binding) != {"relative_path", "sha256", "passed"}:
        raise ValueError("zero-LR engineering audit binding has wrong schema")
    if binding["passed"] is not True or not _valid_sha256(binding["sha256"]):
        raise ValueError("zero-LR engineering audit binding is not passing")
    path = _safe_project_file(binding["relative_path"], label="engineering audit")
    if sha256_file(path) != binding["sha256"]:
        raise ValueError("zero-LR engineering audit SHA mismatch")
    audit = strict_json_load(path)
    required = {
        "schema",
        "canonical_evidence",
        "passed",
        "source_lock_sha256",
        "formula_audit",
        "zero_lr_sampler_pair_checks",
        "shared_initialization",
        "shared_rng_tapes",
        "worker_execution",
        "validated_run_sha256",
    }
    if set(audit) != required or audit.get("schema") != ENGINEERING_AUDIT_SCHEMA:
        raise ValueError("zero-LR engineering audit has wrong schema")
    if audit.get("canonical_evidence") is not False or audit.get("passed") is not True:
        raise ValueError("zero-LR engineering audit is not passing/non-evidentiary")
    if audit.get("source_lock_sha256") != lock_sha256:
        raise ValueError("zero-LR engineering audit is bound to another source lock")
    if not isinstance(audit["formula_audit"], dict) or audit["formula_audit"].get(
        "passed"
    ) is not True:
        raise ValueError("zero-LR engineering formula audit failed")
    pair_checks = audit["zero_lr_sampler_pair_checks"]
    if not isinstance(pair_checks, dict) or set(pair_checks) != {
        "uniform",
        "p1mp",
        "u8",
    } or any(
        not isinstance(record, dict)
        or set(record)
        != {
            "model_eval_trajectories_identical",
            "selected_examples_identical",
            "actions_identical",
        }
        or any(value is not True for value in record.values())
        for record in pair_checks.values()
    ):
        raise ValueError("zero-LR paired common-random-number audit failed")
    if audit["shared_initialization"] is not True or audit["shared_rng_tapes"] is not True:
        raise ValueError("zero-LR initialization/tape audit failed")
    worker = audit["worker_execution"]
    if not isinstance(worker, dict) or set(worker) != {
        "thread_provenance",
        "orchestration_provenance",
        "serial_parallel_scientific_files_byte_identical",
        "timing_files_excluded_as_unbound_metadata",
        "compared_file_sha256",
    }:
        raise ValueError("zero-LR worker audit schema differs")
    if worker["serial_parallel_scientific_files_byte_identical"] is not True:
        raise ValueError("zero-LR serial/parallel artifacts differ")
    thread = worker["thread_provenance"]
    expected_thread = {
        "device": "cpu",
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "deterministic_algorithms": True,
        "thread_environment": {
            variable: "1" for variable in THREAD_ENVIRONMENT_VARIABLES
        },
    }
    if thread != expected_thread:
        raise ValueError("zero-LR worker thread provenance failed")
    orchestration = worker["orchestration_provenance"]
    if not isinstance(orchestration, dict) or set(orchestration) != {
        "serial",
        "parallel",
    }:
        raise ValueError("zero-LR orchestration provenance schema differs")
    if orchestration["serial"] != {
        "worker_mode": "serial",
        "requested_workers": 1,
    }:
        raise ValueError("zero-LR serial orchestration provenance failed")
    parallel = orchestration["parallel"]
    if (
        not isinstance(parallel, dict)
        or set(parallel) != {"worker_mode", "requested_workers"}
        or parallel["worker_mode"] != "process_pool_worker"
        or type(parallel["requested_workers"]) is not int
        or parallel["requested_workers"] < 2
    ):
        raise ValueError("zero-LR parallel orchestration provenance failed")
    run_hashes = audit["validated_run_sha256"]
    if not isinstance(run_hashes, dict) or set(run_hashes) != {"serial", "parallel"}:
        raise ValueError("zero-LR run SHA manifest differs")
    expected_cells = {
        "practical_maxrl__uniform",
        "practical_maxrl__p1mp",
        "practical_maxrl__u8",
        "rloo__uniform",
        "rloo__p1mp",
        "rloo__u8",
    }
    if any(not isinstance(values, dict) or set(values) != expected_cells for values in run_hashes.values()):
        raise ValueError("zero-LR run SHA manifest lacks registered cells")
    if any(
        not _valid_sha256(digest)
        for values in run_hashes.values()
        for digest in values.values()
    ):
        raise ValueError("zero-LR run SHA manifest contains an invalid digest")
    if run_hashes["serial"] != run_hashes["parallel"]:
        raise ValueError("zero-LR serial/parallel summary SHA manifests differ")
    compared = worker["compared_file_sha256"]
    if not isinstance(compared, dict) or set(compared) != expected_cells:
        raise ValueError("zero-LR compared-file manifest lacks registered cells")
    if any(
        not isinstance(files, dict)
        or not files
        or any(not isinstance(name, str) or not _valid_sha256(digest) for name, digest in files.items())
        for files in compared.values()
    ):
        raise ValueError("zero-LR compared-file manifest contains invalid entries")
    return audit


def validate_lr_selection_document(
    payload: Mapping[str, Any], *, lock_sha256: str
) -> None:
    required = {
        "schema",
        "status",
        "all_development_gates_passed",
        "source_lock_sha256",
        "development_authorization",
        "zero_lr_engineering_audit",
        "development_run_manifest",
        "selection_metric",
        "tie_break",
        "estimator_rate_scores",
        "common_rate_scores",
        "selected_learning_rates_by_estimator",
        "selected_common_learning_rate",
        "gates",
    }
    if set(payload) != required:
        raise ValueError("LR-selection artifact has an unexpected field set")
    if payload.get("schema") != LR_SELECTION_SCHEMA:
        raise ValueError("LR-selection schema mismatch")
    if payload.get("status") != "frozen_after_development_before_test_materialization":
        raise ValueError("LR-selection artifact is not frozen")
    if payload.get("source_lock_sha256") != lock_sha256:
        raise ValueError("LR selection is bound to another source lock")
    if payload.get("all_development_gates_passed") is not True:
        raise ValueError("LR selection records failed development gates")
    if payload.get("selection_metric") != "development C8 normalized action-budget AUC":
        raise ValueError("LR-selection metric differs from the frozen metric")
    if payload.get("tie_break") != "smaller learning rate on literal exact score equality":
        raise ValueError("LR-selection tie-break differs from the frozen rule")
    rates = payload.get("selected_learning_rates_by_estimator")
    if not isinstance(rates, dict) or set(rates) != {"practical_maxrl", "rloo"}:
        raise ValueError("LR selection lacks the exact estimator rate set")
    allowed = set(DEVELOPMENT_LRS)
    if any(type(value) is not float or value not in allowed for value in rates.values()):
        raise ValueError("selected estimator LR is outside the frozen grid")
    common = payload.get("selected_common_learning_rate")
    if type(common) is not float or common not in allowed:
        raise ValueError("selected common LR is outside the frozen grid")
    expected_rate_keys = {f"{rate:g}" for rate in DEVELOPMENT_LRS}
    estimator_scores = payload.get("estimator_rate_scores")
    if not isinstance(estimator_scores, dict) or set(estimator_scores) != {
        "practical_maxrl",
        "rloo",
    }:
        raise ValueError("estimator LR score table has wrong schema")
    for scores in estimator_scores.values():
        if not isinstance(scores, dict) or set(scores) != expected_rate_keys:
            raise ValueError("estimator LR score grid is incomplete")
        if any(type(value) is not float or not math.isfinite(value) for value in scores.values()):
            raise ValueError("estimator LR score table contains non-finite values")
    common_scores = payload.get("common_rate_scores")
    if not isinstance(common_scores, dict) or set(common_scores) != expected_rate_keys:
        raise ValueError("common LR score grid is incomplete")
    if any(
        type(value) is not float or not math.isfinite(value)
        for value in common_scores.values()
    ):
        raise ValueError("common LR score table contains non-finite values")
    for estimator in ("practical_maxrl", "rloo"):
        scores = estimator_scores[estimator]
        best = max(scores.values())
        expected_selected = min(
            rate for rate in DEVELOPMENT_LRS if scores[f"{rate:g}"] == best
        )
        if float(rates[estimator]) != expected_selected:
            raise ValueError("selected estimator LR is inconsistent with exact score maximum")
    best_common = max(common_scores.values())
    expected_common = min(
        rate for rate in DEVELOPMENT_LRS if common_scores[f"{rate:g}"] == best_common
    )
    if float(common) != expected_common:
        raise ValueError("selected common LR is inconsistent with exact score maximum")
    run_manifest = payload.get("development_run_manifest")
    if not isinstance(run_manifest, dict) or len(run_manifest) != 120:
        raise ValueError("development run manifest must contain exactly 120 summaries")
    expected_run_paths = {
        f"lr_{rate:g}/seed_{seed}/{estimator}__{sampler}/summary.json"
        for rate in DEVELOPMENT_LRS
        for seed in range(31000, 31004)
        for estimator in ("practical_maxrl", "rloo")
        for sampler in ("uniform", "p1mp", "u8")
    }
    if set(run_manifest) != expected_run_paths:
        raise ValueError("development run manifest paths differ from the frozen schedule")
    if any(
        not isinstance(path, str)
        or not path.endswith("/summary.json")
        or not _valid_sha256(digest)
        for path, digest in run_manifest.items()
    ):
        raise ValueError("development run manifest contains an invalid path or SHA")
    gates = payload.get("gates")
    required_gates = {
        "all_120_runs_complete_finite_and_valid",
        "stored_split_and_hashes_valid",
        "formula_and_mass_audit_passed",
        "zero_lr_engineering_audit_bound_and_passing",
        "cross_cell_initialization_and_tapes_identical",
        "thread_provenance_valid",
        "exact_budgets_and_checkpoints",
        "valid_learning_rate_each_estimator",
        "uniform_arms_median_dev_c8_improvement",
        "uniform_arms_median_dev_c8_improvement_at_least_0p02",
        "sealed_test_outcomes_absent_from_development",
    }
    if not isinstance(gates, dict) or set(gates) != required_gates:
        raise ValueError("LR-selection gate set is not exact")
    boolean_gates = required_gates - {"uniform_arms_median_dev_c8_improvement"}
    if any(gates[name] is not True for name in boolean_gates):
        raise ValueError("one or more required development gates failed")
    median = gates["uniform_arms_median_dev_c8_improvement"]
    if type(median) is not float or not math.isfinite(median) or median < 0.02:
        raise ValueError("uniform-arm development improvement gate failed")
    audit_binding = payload.get("zero_lr_engineering_audit")
    validate_engineering_audit_binding(audit_binding, lock_sha256=lock_sha256)
    development_authorization = payload.get("development_authorization")
    if not isinstance(development_authorization, dict) or set(development_authorization) != {
        "relative_path",
        "sha256",
    }:
        raise ValueError("development authorization binding has wrong schema")
    if not _valid_sha256(development_authorization["sha256"]):
        raise ValueError("development authorization binding has invalid SHA")
    authorization_path = _safe_project_file(
        development_authorization["relative_path"], label="development authorization"
    )
    if sha256_file(authorization_path) != development_authorization["sha256"]:
        raise ValueError("development authorization binding SHA mismatch")
    authorization = verify_execution_authorization(
        authorization_path, phase="development", lock_sha256=lock_sha256
    )
    if authorization["zero_lr_engineering_audit"] != audit_binding:
        raise ValueError("LR selection and development authorization bind different audits")


def verify_execution_authorization(
    authorization_path: Path,
    *,
    phase: str,
    lock_sha256: str,
    lr_selection_path: Path | None = None,
) -> dict[str, Any]:
    authorization = strict_json_load(authorization_path)
    common = {
        "schema",
        "authorized_phase",
        "source_lock_sha256",
        "independent_preseal_review",
        "root_execution_authorization",
        "authorization_digest",
    }
    required = (
        common | {"zero_lr_engineering_audit"}
        if phase == "development"
        else common | {"lr_selection"}
    )
    if set(authorization) != required:
        raise ValueError("execution authorization has an unexpected field set")
    if authorization.get("schema") != AUTHORIZATION_SCHEMA:
        raise ValueError("execution-authorization schema mismatch")
    if authorization.get("authorized_phase") != phase:
        raise ValueError(f"authorization does not permit exact phase: {phase}")
    if authorization.get("source_lock_sha256") != lock_sha256:
        raise ValueError("authorization is bound to a different source lock")
    _validate_review_record(authorization.get("independent_preseal_review"))
    _validate_root_record(authorization.get("root_execution_authorization"))
    without_digest = {key: value for key, value in authorization.items() if key != "authorization_digest"}
    if authorization.get("authorization_digest") != authorization_payload_digest(without_digest):
        raise ValueError("execution authorization digest mismatch")
    if phase == "development":
        validate_engineering_audit_binding(
            authorization["zero_lr_engineering_audit"], lock_sha256=lock_sha256
        )
    else:
        if lr_selection_path is None:
            raise ValueError("confirmation authorization requires LR selection path")
        binding = authorization["lr_selection"]
        if not isinstance(binding, dict) or set(binding) != {
            "relative_path",
            "sha256",
            "all_development_gates_passed",
        }:
            raise ValueError("confirmation LR-selection binding has wrong schema")
        if binding["all_development_gates_passed"] is not True:
            raise ValueError("confirmation authorization binds failed development gates")
        bound_path = _safe_project_file(binding["relative_path"], label="LR selection")
        if bound_path.resolve() != lr_selection_path.resolve():
            raise ValueError("provided LR-selection path differs from authorized path")
        if not _valid_sha256(binding["sha256"]) or sha256_file(bound_path) != binding["sha256"]:
            raise ValueError("authorized LR-selection SHA mismatch")
        selection = strict_json_load(bound_path)
        validate_lr_selection_document(selection, lock_sha256=lock_sha256)
    return authorization
