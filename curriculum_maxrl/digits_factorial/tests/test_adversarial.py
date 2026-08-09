from __future__ import annotations

import copy
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import torch

from curriculum_maxrl.digits_factorial import analyze, locking
from curriculum_maxrl.digits_factorial.analyze import (
    _analyze_confirmation_collected,
    _collect_confirmation,
    shared_complete_confirmation_seeds,
    validate_run,
)
from curriculum_maxrl.digits_factorial.core import (
    CELLS,
    CONFIRMATION_SEEDS,
    DEVELOPMENT_LRS,
    DEVELOPMENT_SEEDS,
    Cell,
    sha256_array,
    sha256_file,
    strict_json_load,
    torch_state_sha256,
    write_json,
)
from curriculum_maxrl.digits_factorial.runner import (
    RunSpec,
    _expected_confirmation_lr,
    _validate_confirmation_lr,
    run_one,
)


@pytest.fixture(scope="module")
def base_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("digits-adversarial") / "engineering" / "base"
    return run_one(
        RunSpec(
            phase="engineering",
            cell=Cell("practical_maxrl", "uniform"),
            seed=33000,
            learning_rate=0.0,
            steps=2,
            include_test=False,
            output_dir=root,
        )
    )


def clone_run(base_run: Path, tmp_path: Path) -> Path:
    destination = tmp_path / "engineering" / "run"
    shutil.copytree(base_run.parent, destination)
    return destination / "summary.json"


def mutate_summary(path: Path, mutator: Callable[[dict[str, Any]], None]) -> None:
    payload = strict_json_load(path)
    mutator(payload)
    write_json(path, payload)


def mutate_ledger(
    summary_path: Path, mutator: Callable[[dict[str, np.ndarray]], None]
) -> None:
    ledger_path = summary_path.parent / "ledger.npz"
    with np.load(ledger_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    mutator(arrays)
    np.savez_compressed(ledger_path, **arrays)
    payload = strict_json_load(summary_path)
    payload["ledger"]["sha256"] = sha256_file(ledger_path)
    payload["ledger"]["array_sha256"] = {
        name: sha256_array(array) for name, array in sorted(arrays.items())
    }
    write_json(summary_path, payload)


def mutate_checkpoint(
    summary_path: Path,
    step: int,
    mutator: Callable[[dict[str, Any]], None],
) -> None:
    checkpoint_path = summary_path.parent / f"checkpoint_step{step:04d}.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    mutator(payload)
    torch.save(payload, checkpoint_path)
    summary = strict_json_load(summary_path)
    record = next(item for item in summary["recovery_checkpoints"] if item["step"] == step)
    record["sha256"] = sha256_file(checkpoint_path)
    if isinstance(payload.get("model_state_dict"), dict):
        try:
            record["model_state_sha256"] = analyze._state_dict_sha256(
                payload["model_state_dict"]
            )
        except Exception:
            pass
    if isinstance(payload.get("optimizer_state_dict"), dict):
        try:
            record["optimizer_state_sha256"] = torch_state_sha256(
                payload["optimizer_state_dict"]
            )
        except Exception:
            pass
    if isinstance(payload.get("torch_cpu_rng_state"), torch.Tensor):
        record["torch_rng_state_sha256"] = torch_state_sha256(
            payload["torch_cpu_rng_state"]
        )
    write_json(summary_path, summary)


@pytest.mark.parametrize("field", ["extra", "sealed_test_outcomes"])
def test_extra_or_sealed_test_summary_field_is_rejected(
    base_run: Path, tmp_path: Path, field: str
) -> None:
    summary = clone_run(base_run, tmp_path)
    mutate_summary(summary, lambda payload: payload.__setitem__(field, {}))
    with pytest.raises(ValueError, match="field set"):
        validate_run(summary)


def test_duplicate_and_nonfinite_json_are_rejected(base_run: Path, tmp_path: Path) -> None:
    summary = clone_run(base_run, tmp_path)
    text = summary.read_text()
    summary.write_text(text.replace('{\n  "accounting"', '{\n  "schema": "duplicate",\n  "accounting"', 1))
    with pytest.raises(ValueError, match="duplicate JSON key"):
        validate_run(summary)

    summary = clone_run(base_run, tmp_path / "overflow")
    timing = summary.parent / "timing.json"
    timing.write_text(
        '{"schema":"curriculum-maxrl/digits-factorial-unbound-timing/v1",'
        '"non_evidentiary_unbound_metadata":true,"wall_seconds":1e309}\n'
    )
    with pytest.raises(ValueError, match="non-finite"):
        validate_run(summary)


def test_extra_and_duplicate_ledger_members_are_rejected(
    base_run: Path, tmp_path: Path
) -> None:
    summary = clone_run(base_run, tmp_path)
    mutate_ledger(summary, lambda arrays: arrays.__setitem__("extra", np.zeros(1)))
    with pytest.raises(ValueError, match="member set"):
        validate_run(summary)

    summary = clone_run(base_run, tmp_path / "duplicate")
    ledger = summary.parent / "ledger.npz"
    with zipfile.ZipFile(ledger, "r") as archive:
        member = archive.read("loss.npy")
    with zipfile.ZipFile(ledger, "a") as archive:
        archive.writestr("loss.npy", member)
    mutate_summary(summary, lambda payload: payload["ledger"].__setitem__("sha256", sha256_file(ledger)))
    with pytest.raises(ValueError, match="duplicate"):
        validate_run(summary)


@pytest.mark.parametrize(
    ("name", "mutation", "message"),
    [
        (
            "wrong_dtype",
            lambda arrays: arrays.__setitem__(
                "selected_train_positions", arrays["selected_train_positions"].astype(np.int64)
            ),
            "shape/dtype",
        ),
        (
            "wrong_shape",
            lambda arrays: arrays.__setitem__("loss", arrays["loss"][:1]),
            "shape/dtype",
        ),
        (
            "sealed_test_array",
            lambda arrays: arrays.__setitem__(
                "eval_probabilities_test", np.zeros((1, 360, 10), dtype=np.float64)
            ),
            "member set",
        ),
        (
            "p_outside",
            lambda arrays: arrays["p_train_by_step"].__setitem__((0, 0), 1.1),
            "outside",
        ),
        (
            "negative_probability",
            lambda arrays: arrays["selected_action_probabilities"].__setitem__((0, 0, 0), -0.1),
            "simplex",
        ),
        (
            "probability_row_sum",
            lambda arrays: arrays["selected_action_probabilities"].__setitem__(
                (0, 0, 0), arrays["selected_action_probabilities"][0, 0, 0] + 1e-4
            ),
            "simplex",
        ),
        (
            "loss_tamper",
            lambda arrays: arrays["loss"].__setitem__(0, arrays["loss"][0] + 0.01),
            "loss/gradient",
        ),
        (
            "gradient_tamper",
            lambda arrays: arrays["gradient_norm"].__setitem__(0, arrays["gradient_norm"][0] + 0.01),
            "loss/gradient",
        ),
        (
            "evaluation_tamper",
            lambda arrays: arrays["eval_probabilities_dev"].__setitem__(
                (0, 0, slice(None)), np.roll(arrays["eval_probabilities_dev"][0, 0], 1)
            ),
            "evaluation probabilities",
        ),
    ],
)
def test_ledger_dtype_shape_probability_and_replay_tampering_is_rejected(
    base_run: Path,
    tmp_path: Path,
    name: str,
    mutation: Callable[[dict[str, np.ndarray]], None],
    message: str,
) -> None:
    summary = clone_run(base_run, tmp_path / name)
    mutate_ledger(summary, mutation)
    with pytest.raises(ValueError, match=message):
        validate_run(summary)


def test_selected_correct_probability_must_be_tied_to_full_pool_p(
    base_run: Path, tmp_path: Path
) -> None:
    summary = clone_run(base_run, tmp_path)

    def mutation(arrays: dict[str, np.ndarray]) -> None:
        probs = arrays["selected_action_probabilities"]
        probs[0, 0, 0] += 1e-8
        probs[0, 0, 1] -= 1e-8

    mutate_ledger(summary, mutation)
    with pytest.raises(ValueError, match="selected action probabilities|tied"):
        validate_run(summary)


def test_initialization_and_ledger_paths_fail_closed(base_run: Path, tmp_path: Path) -> None:
    summary = clone_run(base_run, tmp_path)
    mutate_summary(
        summary,
        lambda payload: payload.__setitem__("initial_model_state_sha256", "0" * 64),
    )
    with pytest.raises(ValueError, match="initialization"):
        validate_run(summary)

    summary = clone_run(base_run, tmp_path / "path")
    mutate_summary(
        summary,
        lambda payload: payload["ledger"].__setitem__("relative_path", "../ledger.npz"),
    )
    with pytest.raises(ValueError, match="filename"):
        validate_run(summary)


@pytest.mark.parametrize(
    ("name", "step", "mutation", "message"),
    [
        ("extra_key", 0, lambda payload: payload.__setitem__("extra", 1), "field set"),
        ("missing_key", 0, lambda payload: payload.pop("torch_cpu_rng_state"), "field set"),
        (
            "python_rng_schema",
            0,
            lambda payload: payload.__setitem__(
                "python_random_state", (2, payload["python_random_state"][1], None)
            ),
            "Python RNG schema",
        ),
        (
            "wrong_model_shape",
            0,
            lambda payload: payload["model_state_dict"].__setitem__(
                "linear1.bias", payload["model_state_dict"]["linear1.bias"][:63]
            ),
            "shape/dtype",
        ),
        (
            "wrong_model_dtype",
            0,
            lambda payload: payload["model_state_dict"].__setitem__(
                "linear1.bias", payload["model_state_dict"]["linear1.bias"].float()
            ),
            "shape/dtype",
        ),
        (
            "nonfinite_model",
            0,
            lambda payload: payload["model_state_dict"]["linear1.bias"].__setitem__(0, float("inf")),
            "non-finite",
        ),
        (
            "optimizer_hyperparameter",
            2,
            lambda payload: payload["optimizer_state_dict"]["param_groups"][0].__setitem__(
                "momentum", 0.8
            ),
            "hyperparameters",
        ),
        (
            "optimizer_buffer_dtype",
            2,
            lambda payload: payload["optimizer_state_dict"]["state"][0].__setitem__(
                "momentum_buffer",
                payload["optimizer_state_dict"]["state"][0]["momentum_buffer"].float(),
            ),
            "buffer shape/dtype",
        ),
        (
            "optimizer_buffer_missing",
            2,
            lambda payload: payload["optimizer_state_dict"]["state"][0].pop(
                "momentum_buffer"
            ),
            "buffer schema",
        ),
        (
            "valid_model_change",
            2,
            lambda payload: payload["model_state_dict"]["linear1.bias"].__setitem__(
                0, payload["model_state_dict"]["linear1.bias"][0] + 1e-5
            ),
            "differs from replay",
        ),
    ],
)
def test_checkpoint_schema_state_and_replay_tampering_is_rejected(
    base_run: Path,
    tmp_path: Path,
    name: str,
    step: int,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    summary = clone_run(base_run, tmp_path / name)
    mutate_checkpoint(summary, step, mutation)
    with pytest.raises((ValueError, RuntimeError), match=message):
        validate_run(summary)


def test_checkpoint_path_traversal_and_coexisting_extra_file_are_rejected(
    base_run: Path, tmp_path: Path
) -> None:
    summary = clone_run(base_run, tmp_path)
    mutate_summary(
        summary,
        lambda payload: payload["recovery_checkpoints"][0].__setitem__(
            "relative_path", "../../checkpoint_step0000.pt"
        ),
    )
    with pytest.raises(ValueError, match="filename"):
        validate_run(summary)

    summary = clone_run(base_run, tmp_path / "extra")
    (summary.parent / "failure.json").write_text("{}\n")
    with pytest.raises(ValueError, match="extra/missing"):
        validate_run(summary)


def test_checkpoint_loader_always_uses_weights_only_true(
    base_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = torch.load
    calls: list[object] = []

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs.get("weights_only"))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", wrapped)
    validate_run(base_run)
    assert calls and set(calls) == {True}


def test_exact_scalar_types_and_worker_provenance_fail_closed(
    base_run: Path, tmp_path: Path
) -> None:
    summary = clone_run(base_run, tmp_path / "bool_steps")
    mutate_summary(summary, lambda payload: payload.__setitem__("steps", True))
    with pytest.raises(ValueError, match="exact JSON integer"):
        validate_run(summary)

    summary = clone_run(base_run, tmp_path / "worker_count")
    timing_path = summary.parent / "timing.json"
    timing = strict_json_load(timing_path)
    timing["requested_workers"] = 2
    write_json(timing_path, timing)
    with pytest.raises(ValueError, match="direct/serial worker provenance"):
        validate_run(summary)


def _authorization_payload(phase: str, extra: dict[str, Any]) -> dict[str, Any]:
    _, lock_sha = locking.load_and_verify_source_lock()
    payload: dict[str, Any] = {
        "schema": "curriculum-maxrl/digits-factorial-execution-authorization/v2",
        "authorized_phase": phase,
        "source_lock_sha256": lock_sha,
        "independent_preseal_review": {"passed": True, "review_sha256": "1" * 64},
        "root_execution_authorization": {
            "authorized": True,
            "authorized_utc": "2026-08-08T00:00:00+00:00",
        },
        **extra,
    }
    payload["authorization_digest"] = locking.authorization_payload_digest(payload)
    return payload


def test_development_authorization_requires_bound_passing_engineering_audit(
    tmp_path: Path,
) -> None:
    _, lock_sha = locking.load_and_verify_source_lock()
    auth = tmp_path / "dev_auth.json"
    payload = _authorization_payload(
        "development",
        {
            "zero_lr_engineering_audit": {
                "relative_path": "curriculum_maxrl/digits_factorial/engineering/missing.json",
                "sha256": "0" * 64,
                "passed": True,
            }
        },
    )
    write_json(auth, payload)
    with pytest.raises(ValueError, match="does not exist|SHA"):
        locking.verify_execution_authorization(
            auth, phase="development", lock_sha256=lock_sha
        )
    with pytest.raises(ValueError, match="field set"):
        locking.verify_execution_authorization(
            auth, phase="confirmation_tuned", lock_sha256=lock_sha
        )


def _failed_lr_selection(lock_sha: str) -> dict[str, Any]:
    rates = {key: 0.5 for key in ("0.03", "0.1", "0.3", "1", "3")}
    gates = {
        "all_120_runs_complete_finite_and_valid": True,
        "stored_split_and_hashes_valid": True,
        "formula_and_mass_audit_passed": True,
        "zero_lr_engineering_audit_bound_and_passing": True,
        "cross_cell_initialization_and_tapes_identical": True,
        "thread_provenance_valid": True,
        "exact_budgets_and_checkpoints": True,
        "valid_learning_rate_each_estimator": True,
        "uniform_arms_median_dev_c8_improvement": 0.03,
        "uniform_arms_median_dev_c8_improvement_at_least_0p02": False,
        "sealed_test_outcomes_absent_from_development": True,
    }
    return {
        "schema": "curriculum-maxrl/digits-factorial-lr-selection/v2",
        "status": "frozen_after_development_before_test_materialization",
        "all_development_gates_passed": True,
        "source_lock_sha256": lock_sha,
        "development_authorization": {"relative_path": "x", "sha256": "2" * 64},
        "zero_lr_engineering_audit": {"relative_path": "x", "sha256": "3" * 64, "passed": True},
        "development_run_manifest": {
            f"lr_{rate:g}/seed_{seed}/{estimator}__{sampler}/summary.json": "5" * 64
            for rate in DEVELOPMENT_LRS
            for seed in DEVELOPMENT_SEEDS
            for estimator, sampler in CELLS
        },
        "selection_metric": "development C8 normalized action-budget AUC",
        "tie_break": "smaller learning rate on literal exact score equality",
        "estimator_rate_scores": {"practical_maxrl": rates, "rloo": rates},
        "common_rate_scores": rates,
        "selected_learning_rates_by_estimator": {"practical_maxrl": 0.03, "rloo": 0.03},
        "selected_common_learning_rate": 0.03,
        "gates": gates,
    }


def test_failed_nested_lr_gate_blocks_confirmation_and_selection_swap_sha(
    tmp_path: Path,
) -> None:
    _, lock_sha = locking.load_and_verify_source_lock()
    selection = _failed_lr_selection(lock_sha)
    with pytest.raises(ValueError, match="required development gates failed"):
        locking.validate_lr_selection_document(selection, lock_sha256=lock_sha)

    auth = tmp_path / "confirm_auth.json"
    payload = _authorization_payload(
        "confirmation_tuned",
        {
            "lr_selection": {
                "relative_path": "curriculum_maxrl/digits_factorial/engineering/missing_selection.json",
                "sha256": "4" * 64,
                "all_development_gates_passed": True,
            }
        },
    )
    write_json(auth, payload)
    with pytest.raises(ValueError, match="does not exist|SHA"):
        locking.verify_execution_authorization(
            auth,
            phase="confirmation_tuned",
            lock_sha256=lock_sha,
            lr_selection_path=tmp_path / "swapped.json",
        )


def test_tuned_and_common_rates_are_taken_exactly_from_frozen_selection() -> None:
    selection = {
        "selected_learning_rates_by_estimator": {
            "practical_maxrl": 0.03,
            "rloo": 0.3,
        },
        "selected_common_learning_rate": 0.1,
    }
    assert (
        _expected_confirmation_lr(
            "confirmation_tuned", Cell("practical_maxrl", "u8"), selection
        )
        == 0.03
    )
    assert (
        _expected_confirmation_lr(
            "confirmation_tuned", Cell("rloo", "p1mp"), selection
        )
        == 0.3
    )
    assert (
        _expected_confirmation_lr(
            "confirmation_common", Cell("rloo", "uniform"), selection
        )
        == 0.1
    )
    with pytest.raises(ValueError, match="confirmation LR"):
        _validate_confirmation_lr(
            "confirmation_tuned",
            Cell("practical_maxrl", "u8"),
            0.1,
            selection,
        )
    with pytest.raises(ValueError, match="confirmation LR"):
        _validate_confirmation_lr(
            "confirmation_common", Cell("rloo", "uniform"), 0.3, selection
        )


def test_lr_selection_rejects_boolean_rate_and_nonliteral_tie_rule() -> None:
    _, lock_sha = locking.load_and_verify_source_lock()
    selection = _failed_lr_selection(lock_sha)
    selection["selected_learning_rates_by_estimator"]["practical_maxrl"] = True
    with pytest.raises(ValueError, match="outside the frozen grid"):
        locking.validate_lr_selection_document(selection, lock_sha256=lock_sha)

    selection = _failed_lr_selection(lock_sha)
    selection["tie_break"] = "smaller rate within tolerance"
    with pytest.raises(ValueError, match="tie-break"):
        locking.validate_lr_selection_document(selection, lock_sha256=lock_sha)


def _synthetic_confirmation_map(seeds: list[int], *, identical_q: bool) -> dict[int, dict[str, dict[str, Any]]]:
    result: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in seeds:
        result[seed] = {}
        for estimator, sampler in CELLS:
            if sampler == "uniform":
                outcome = 0.4
            elif (estimator, sampler) in {
                ("practical_maxrl", "u8"),
                ("rloo", "p1mp"),
            }:
                outcome = 0.8
            else:
                outcome = 0.5
            q = np.full((2, 3), 1.0 / 3.0)
            if not identical_q and sampler == "u8":
                q = np.asarray([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]])
            result[seed][Cell(estimator, sampler).name] = {
                "primary_c8_auc": outcome,
                "q_by_step": q,
                "initial_model_state_sha256": f"{seed:064x}"[-64:],
                "rng_tape_sha256": {"task": str(seed), "action": str(seed)},
            }
    return result


def test_treatment_delivery_gate_is_required_for_primary_support() -> None:
    seeds = list(CONFIRMATION_SEEDS[:20])
    result = _analyze_confirmation_collected(
        Path("synthetic"),
        "confirmation_tuned",
        _synthetic_confirmation_map(seeds, identical_q=True),
        [],
    )
    assert result["contrasts"]["interaction"]["mean"] > 0.01
    assert result["treatment_delivery"]["passed"] is False
    assert result["primary_supported"] is False


def test_tuned_common_sensitivity_uses_shared_complete_block_intersection() -> None:
    tuned_seeds = list(CONFIRMATION_SEEDS[:-1])
    common_seeds = list(CONFIRMATION_SEEDS[1:])
    tuned = _synthetic_confirmation_map(tuned_seeds, identical_q=False)
    common = _synthetic_confirmation_map(common_seeds, identical_q=False)
    shared = shared_complete_confirmation_seeds(tuned, common)
    assert shared == list(CONFIRMATION_SEEDS[1:-1])
    tuned_report = _analyze_confirmation_collected(
        Path("tuned"), "confirmation_tuned", tuned, [], seed_subset=shared
    )
    common_report = _analyze_confirmation_collected(
        Path("common"), "confirmation_common", common, [], seed_subset=shared
    )
    assert tuned_report["complete_block_seeds"] == common_report["complete_block_seeds"]
    assert len(shared) == 22


@pytest.mark.parametrize("extra", ["seed_99999", "unknown_file"])
def test_extra_confirmation_result_directories_or_files_are_rejected(
    tmp_path: Path, extra: str
) -> None:
    root = tmp_path / "confirmation"
    root.mkdir()
    if extra.startswith("seed_"):
        (root / extra).mkdir()
    else:
        (root / extra).write_text("x")
    with pytest.raises(ValueError, match="extra"):
        _collect_confirmation(root, "confirmation_tuned")


def test_extra_confirmation_cell_nested_directory_and_summary_failure_pair_are_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "extra_cell"
    seed = root / f"seed_{CONFIRMATION_SEEDS[0]}"
    (seed / "unknown_cell").mkdir(parents=True)
    with pytest.raises(ValueError, match="extra"):
        _collect_confirmation(root, "confirmation_tuned")

    root = tmp_path / "nested"
    cell = root / f"seed_{CONFIRMATION_SEEDS[0]}" / Cell(*CELLS[0]).name
    (cell / "nested").mkdir(parents=True)
    with pytest.raises(ValueError, match="nested"):
        _collect_confirmation(root, "confirmation_tuned")

    root = tmp_path / "coexisting"
    cell = root / f"seed_{CONFIRMATION_SEEDS[0]}" / Cell(*CELLS[0]).name
    cell.mkdir(parents=True)
    (cell / "summary.json").write_text("{}\n")
    (cell / "failure.json").write_text("{}\n")
    with pytest.raises(ValueError, match="both summary and failure"):
        _collect_confirmation(root, "confirmation_tuned")


@pytest.mark.parametrize(
    ("identity_field", "wrong_value"),
    [
        ("seed", CONFIRMATION_SEEDS[1]),
        ("cell", Cell(*CELLS[1]).name),
    ],
)
def test_confirmation_artifact_must_match_seed_and_cell_directory_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_field: str,
    wrong_value: object,
) -> None:
    seed = CONFIRMATION_SEEDS[0]
    cell = Cell(*CELLS[0]).name
    summary = tmp_path / f"seed_{seed}" / cell / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text("{}\n")
    item = {"phase": "confirmation_tuned", "seed": seed, "cell": cell}
    item[identity_field] = wrong_value
    monkeypatch.setattr(analyze, "validate_run", lambda *args, **kwargs: item)
    by_seed, failures = _collect_confirmation(tmp_path, "confirmation_tuned")
    assert by_seed == {}
    assert any("identity differs" in failure["reason"] for failure in failures)


def test_confirmation_rejects_duplicated_run_under_another_seed_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = Cell(*CELLS[0]).name
    first_seed, second_seed = CONFIRMATION_SEEDS[:2]
    for seed in (first_seed, second_seed):
        summary = tmp_path / f"seed_{seed}" / cell / "summary.json"
        summary.parent.mkdir(parents=True)
        summary.write_text("{}\n")
    duplicated_identity = {
        "phase": "confirmation_tuned",
        "seed": first_seed,
        "cell": cell,
    }
    monkeypatch.setattr(
        analyze, "validate_run", lambda *args, **kwargs: duplicated_identity
    )
    by_seed, failures = _collect_confirmation(tmp_path, "confirmation_tuned")
    assert set(by_seed) == {first_seed}
    assert second_seed not in by_seed
    assert any(
        failure["seed"] == second_seed and "identity differs" in failure["reason"]
        for failure in failures
    )


def test_confirmation_cell_without_summary_has_exact_file_schema(
    tmp_path: Path,
) -> None:
    seed = CONFIRMATION_SEEDS[0]
    cell = tmp_path / f"seed_{seed}" / Cell(*CELLS[0]).name
    cell.mkdir(parents=True)
    (cell / "unexpected.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="without summary/failure"):
        _collect_confirmation(tmp_path, "confirmation_tuned")

    root = tmp_path / "failure_with_extra"
    cell = root / f"seed_{seed}" / Cell(*CELLS[0]).name
    cell.mkdir(parents=True)
    (cell / "failure.json").write_text("{}\n")
    (cell / "unexpected.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="extra artifacts"):
        _collect_confirmation(root, "confirmation_tuned")


def test_source_lock_covers_canonical_estimator_dependencies() -> None:
    assert "curriculum_maxrl/__init__.py" in locking.LOCKED_RELATIVE_PATHS
    assert "curriculum_maxrl/estimators.py" in locking.LOCKED_RELATIVE_PATHS


def test_source_lock_rejects_omitted_canonical_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        locking,
        "LOCKED_RELATIVE_PATHS",
        tuple(
            path
            for path in locking.LOCKED_RELATIVE_PATHS
            if path != "curriculum_maxrl/estimators.py"
        ),
    )
    with pytest.raises(ValueError, match="exact source path set"):
        locking.load_and_verify_source_lock()
