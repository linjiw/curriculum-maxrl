#!/usr/bin/env python3
"""Atomically assemble one permanently non-evidence v4 engineering package.

This Phase-B tool is intentionally structural.  It verifies the closed
training and evaluation components, but it never parses evaluation episode or
aggregate values and never imports the preregistered production analyzer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import evaluate_matched_terminal_v4 as evaluation
import run_matched_terminal_v4 as training


HASH_RE = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_RE = re.compile(r"^([0-9a-f]{64})  (.+)$")
PACKAGE_TOP_LEVEL = {
    "INPUT_CLOSURE.json",
    "components-COMPLETE.json",
    "components-SHA256SUMS",
    "campaign-manifest.json",
    "run-context.json",
    "scheduler.json",
    "run-manifest.json",
    "training-plr-replay-snapshot.json",
    "training-output",
    "training-sidecar",
    "evaluation-package",
    "evaluation-integrity.json",
    "phase-b-receipts",
    "SHA256SUMS",
    "COMPLETE",
}


class AssemblyError(RuntimeError):
    """Raised when an engineering package cannot close exactly."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssemblyError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_existing(path: Path, *, directory: bool, label: str) -> Path:
    require(path.is_absolute() and ".." not in path.parts, f"{label} must be canonical absolute")
    resolved = path.resolve(strict=True)
    require(resolved == path, f"{label} is noncanonical or symbolic")
    require((path.is_dir() if directory else path.is_file()) and not path.is_symlink(), f"unsafe {label}")
    return path


def canonical_new_output(path: Path, protected: Sequence[Path]) -> Path:
    require(path.is_absolute() and ".." not in path.parts, "output must be canonical absolute")
    require(path.name not in {"", ".", ".."}, "unsafe output basename")
    require(not path.exists() and not path.is_symlink(), "output already exists")
    parent = canonical_existing(path.parent, directory=True, label="output parent")
    canonical = parent / path.name
    require(canonical == path, "output path is noncanonical")
    for existing in protected:
        require(
            not canonical.is_relative_to(existing)
            and not existing.is_relative_to(canonical),
            "output/protected-input overlap",
        )
    return canonical


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"unsafe {label}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AssemblyError(f"invalid {label}: {path}") from exc
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    require(path.parent.is_dir() and not path.parent.is_symlink(), "unsafe output parent")
    require(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_text(path: Path, value: str) -> None:
    require(path.parent.is_dir() and not path.parent.is_symlink(), "unsafe output parent")
    require(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_relative(name: str) -> PurePosixPath:
    relative = PurePosixPath(name)
    require(
        name != ""
        and not relative.is_absolute()
        and ".." not in relative.parts
        and "." not in relative.parts,
        f"unsafe manifest path: {name!r}",
    )
    return relative


def _all_files(root: Path, excluded: set[str]) -> set[str]:
    require(root.is_dir() and not root.is_symlink(), f"unsafe directory: {root}")
    names: set[str] = set()
    for path in root.rglob("*"):
        require(not path.is_symlink(), f"symbolic link forbidden: {path}")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            if name not in excluded:
                names.add(name)
        else:
            require(path.is_dir(), f"non-file entry forbidden: {path}")
    return names


def validate_manifest_tree(
    root: Path,
    manifest_name: str,
    complete_name: str,
) -> tuple[str, dict[str, Any]]:
    manifest_path = root / manifest_name
    complete_path = root / complete_name
    require(manifest_path.is_file() and not manifest_path.is_symlink(), "manifest missing")
    require(complete_path.is_file() and not complete_path.is_symlink(), "completion marker missing")
    listed: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        match = MANIFEST_RE.fullmatch(line)
        require(match is not None, "unsafe manifest line")
        digest, encoded = match.groups()
        relative = _safe_relative(encoded).as_posix()
        require(relative not in listed, f"duplicate manifest path: {relative}")
        listed[relative] = digest
    actual = _all_files(root, {manifest_name, complete_name})
    if set(listed) != actual:
        missing = sorted(actual - set(listed))
        extra = sorted(set(listed) - actual)
        raise AssemblyError(
            f"manifest file closure drift: unlisted={missing} missing={extra}"
        )
    for relative, expected in listed.items():
        require(sha256(root / relative) == expected, f"payload hash drift: {relative}")
    return sha256(manifest_path), load_json(complete_path, "completion marker")


def _validate_context_campaign(
    root: Path,
    protocol_path: Path,
    *,
    local_test_mode: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    context_path = root / "run-context.json"
    campaign_path = root / "campaign-manifest.json"
    context = load_json(context_path, "run context")
    campaign = load_json(campaign_path, "campaign manifest")
    protocol, protocol_sha = training.load_protocol(protocol_path)
    require(context.get("protocol_id") == training.PROTOCOL_ID, "context protocol drift")
    require(context.get("purpose") == training.PURPOSE, "context purpose drift")
    require(context.get("arm") in training.ARMS, "context arm drift")
    if local_test_mode:
        require(context.get("job_id") == "local-test", "local context job drift")
        expected_run_id = f"engineering-{context['arm']}-s{context['training_seed']}"
    else:
        require(context.get("job_id") and str(context["job_id"]).isdigit(), "context job drift")
        expected_run_id = f"engineering-slurm-{context['job_id']}-{context['arm']}-s{context['training_seed']}"
    require(context.get("run_id") == expected_run_id, "context run identity drift")
    require(context.get("campaign_manifest_sha256") == sha256(campaign_path), "campaign binding drift")
    require(set(campaign) == training.CAMPAIGN_KEYS, "campaign keys drift")
    require(campaign.get("protocol_id") == training.PROTOCOL_ID, "campaign protocol drift")
    require(campaign.get("purpose") == training.PURPOSE, "campaign purpose drift")
    require(campaign.get("protocol_sha256") == protocol_sha, "campaign protocol hash drift")
    require(campaign.get("frozen_before_endpoint_access") is True, "campaign not frozen")
    submissions = campaign.get("submissions")
    require(isinstance(submissions, list) and len(submissions) == 1, "engineering campaign size drift")
    submission = submissions[0]
    for field in ("run_id", "arm", "training_seed", "job_id"):
        require(submission.get(field) == context.get(field), f"campaign/context drift: {field}")
    require(submission.get("attempt") == 1, "retry attempt forbidden")
    require(submission.get("evaluation_seed") == 100000 + context["training_seed"], "evaluation seed drift")

    provenance = campaign.get("provenance")
    require(isinstance(provenance, dict), "campaign provenance missing")
    require(set(provenance) == training.CAMPAIGN_PROVENANCE_KEYS, "campaign provenance keys drift")
    expected_context = {
        key: provenance[key]
        for key in ({"base_commit", "base_tree", "overlay_contract_sha256"} | training.HASH_KEYS)
    }
    require(context.get("provenance") == expected_context, "context provenance projection drift")
    require(provenance.get("base_commit") == training.BASE_COMMIT, "base commit drift")
    require(provenance.get("base_tree") == training.BASE_TREE, "base tree drift")
    require(
        provenance.get("overlay_contract_sha256") == training.OVERLAY_CONTRACT_SHA256,
        "overlay contract drift",
    )
    for key, value in provenance.items():
        if key not in {"base_commit", "base_tree"}:
            require(isinstance(value, str) and HASH_RE.fullmatch(value) is not None, f"bad hash: {key}")
    here = Path(__file__).resolve().parent
    require(provenance["training_driver_sha256"] == sha256(here / "run_matched_terminal_v4.py"), "training driver drift")
    require(provenance["evaluation_driver_sha256"] == sha256(here / "evaluate_matched_terminal_v4.py"), "evaluation driver drift")
    require(provenance["assembler_driver_sha256"] == sha256(Path(__file__).resolve()), "assembler driver drift")
    analyzer = Path(__file__).resolve().parents[1] / "analysis" / "preregistered_dev_analysis.py"
    require(campaign.get("analyzer_sha256") == sha256(analyzer), "protected analyzer binding drift")
    return context, campaign, protocol, protocol_sha


def _validate_replay_sidecar(root: Path, context: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    sidecar = root / "training-sidecar"
    try:
        manifest_sha = training.validate_training_sidecar(
            sidecar, str(context["run_id"]), str(context["arm"])
        )
    except training.DriverError as exc:
        raise AssemblyError(str(exc)) from exc
    receipt = load_json(sidecar / "training-receipt.json", "training receipt")
    snapshot_path = sidecar / "plr-replay-snapshot.json"
    snapshot = load_json(snapshot_path, "PLR replay snapshot")
    require(receipt.get("paper_evidence") is False, "training evidence label drift")
    require(receipt.get("endpoint_class") == "bounded_engineering_test", "training endpoint class drift")
    require(receipt.get("resumed") is False, "resume forbidden")
    require(receipt.get("n_updates") == 1, "training update count drift")
    require(receipt.get("upstream_n_grad_updates") == 1, "gradient update count drift")
    require(receipt.get("arm") == context["arm"], "training arm drift")
    require(
        receipt.get("plr_snapshot")
        == {"path": "plr-replay-snapshot.json", "sha256": sha256(snapshot_path)},
        "training receipt/snapshot binding drift",
    )
    require(
        snapshot.get("schema") == 1
        and snapshot.get("status") == "completed"
        and snapshot.get("protocol_id") == training.PROTOCOL_ID
        and snapshot.get("purpose") == training.PURPOSE
        and snapshot.get("paper_evidence") is False,
        "snapshot protocol/evidence drift",
    )
    require(
        snapshot.get("run_id") == context["run_id"]
        and snapshot.get("arm") == context["arm"]
        and snapshot.get("training_seed") == context["training_seed"],
        "snapshot run/arm/seed drift",
    )
    require(snapshot.get("kind") == "tie_aware_plr_buffer_safe_snapshot", "snapshot kind drift")
    replay = receipt.get("integrity", {}).get("terminal", {}).get("replay_integrity")
    checkpoint_replay = receipt.get("integrity", {}).get("checkpoint_round_trip", {}).get("replay_integrity")
    require(isinstance(replay, dict) and replay == checkpoint_replay, "terminal/checkpoint replay drift")
    try:
        training.validate_replay_integrity({"replay_integrity": replay})
    except training.DriverError as exc:
        raise AssemblyError(str(exc)) from exc
    require(replay.get("tie_aware_score_ranks") is True, "tie-aware replay disabled")
    require(replay.get("nonfinite_filled_score_count") == 0, "nonfinite filled score")
    require(replay.get("nonfinite_score_rejection_count") == 0, "nonfinite score rejection")
    require(snapshot.get("checkpoint_sha256") == receipt.get("terminal_checkpoint", {}).get("sha256"), "snapshot checkpoint drift")
    require(snapshot.get("sampling_diagnostics") == replay, "snapshot replay diagnostics drift")
    distribution = snapshot.get("replay_distribution")
    require(isinstance(distribution, dict), "snapshot replay distribution missing")
    require(distribution.get("tie_aware_score_ranks") is True, "snapshot tie-aware flag drift")
    require(distribution.get("score_normalization_order") == "canonical_ascending_unnormalized_mass", "snapshot normalization order drift")
    temperature = distribution.get("temperature")
    require(isinstance(temperature, (int, float)) and not isinstance(temperature, bool), "snapshot temperature type drift")
    require(math.isfinite(float(temperature)) and float(temperature) > 0.0, "snapshot temperature invalid")
    filled = snapshot.get("filled_count")
    buffer_size = snapshot.get("buffer_size")
    require(isinstance(filled, int) and isinstance(buffer_size, int) and 1 <= filled <= buffer_size, "snapshot filled count drift")
    blocks = distribution.get("tie_block_sizes_descending_score_order")
    require(
        isinstance(blocks, list)
        and blocks
        and all(isinstance(size, int) and size > 0 for size in blocks)
        and sum(blocks) == filled
        and distribution.get("distinct_filled_score_count") == len(blocks),
        "snapshot tie-block closure drift",
    )
    for key in ("score_effective_support", "replay_effective_support"):
        value = distribution.get(key)
        require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and 1.0 <= float(value) <= filled + training.FLOAT32_DIAGNOSTIC_TOLERANCE,
            f"snapshot {key} drift",
        )
    require(
        distribution.get("tie_equality") == "exact filled-score equality; +0 and -0 tie",
        "snapshot tie equality drift",
    )
    require(
        snapshot.get("sampling_semantics")
        == {
            "replacement": "with_replacement",
            "force_unique_effect": "buffer-update deduplication only; no replay resampling",
            "distinct_identity": "replay buffer slot index",
            "last_counts_persist_until_next_actual_replay_batch": True,
        },
        "snapshot sampling semantics drift",
    )
    require(replay.get("force_unique_resamples_replay") is False, "force_unique replay semantics drift")
    require(replay.get("replay_group_draw_count") == 4 and replay.get("last_replay_group_count") == 4, "replay draw count drift")
    require(
        replay.get("replay_distinct_group_count") + replay.get("replay_duplicate_group_count") == 4
        and replay.get("last_replay_distinct_group_count") + replay.get("last_replay_duplicate_group_count") == 4,
        "replay distinct/duplicate accounting drift",
    )
    slots = snapshot.get("slots")
    require(isinstance(slots, list) and len(slots) == filled, "snapshot slots drift")
    posterior_keys = {
        "success_count", "trial_count", "analytic_expected_activity_score",
        "mean_plugin_score", "jensen_gap", "stored_score_abs_error",
    }
    for slot in slots:
        require(isinstance(slot, dict), "snapshot slot type drift")
        for key in ("stored_score", "normalized_score_probability", "normalized_replay_probability"):
            require(key in slot and math.isfinite(float(slot[key])), f"snapshot slot field drift: {key}")
        if context["arm"] == "frontier":
            require(posterior_keys <= set(slot), "Frontier posterior snapshot missing")
        else:
            require(posterior_keys.isdisjoint(slot), "MaxMC posterior snapshot contamination")
    require(
        (snapshot.get("stored_score_validation") is None)
        is (context["arm"] == "maxmc"),
        "snapshot arm-specific validation drift",
    )
    return manifest_sha, receipt


def _validate_archived_prerequisites(
    closure: Mapping[str, Any], frontier_config_sha256: str
) -> None:
    prerequisites = closure.get("prerequisites")
    require(
        isinstance(prerequisites, dict)
        and set(prerequisites) == {"import", "one_update"},
        "input closure prerequisite keys drift",
    )
    for rung, updates in (("import", 0), ("one_update", 1)):
        prerequisite = prerequisites[rung]
        require(
            isinstance(prerequisite, dict)
            and set(prerequisite) == {
                "result_dir", "manifest_sha256", "complete_sha256",
                "bundle_manifest_sha256", "actual_student_updates",
                "paper_evidence", "analyzer_eligible", "archived_provenance",
            },
            f"{rung} prerequisite fields drift",
        )
        require(
            isinstance(prerequisite["result_dir"], str)
            and prerequisite["result_dir"]
            and prerequisite["bundle_manifest_sha256"]
            == closure["bundle_manifest_sha256"]
            and prerequisite["actual_student_updates"] == updates
            and prerequisite["paper_evidence"] is False
            and prerequisite["analyzer_eligible"] is False,
            f"{rung} prerequisite semantics drift",
        )
        archive = prerequisite["archived_provenance"]
        require(
            isinstance(archive, dict)
            and set(archive) == {"schema", "files"}
            and archive["schema"] == 1,
            f"{rung} archived provenance schema drift",
        )
        files = archive["files"]
        require(
            isinstance(files, dict)
            and set(files) == {"receipt.tsv", "SHA256SUMS", "COMPLETE"},
            f"{rung} archived provenance files drift",
        )
        texts: dict[str, str] = {}
        for name, record in files.items():
            require(
                isinstance(record, dict)
                and set(record) == {"encoding", "text", "sha256"}
                and record.get("encoding") == "utf-8"
                and isinstance(record.get("text"), str)
                and HASH_RE.fullmatch(str(record.get("sha256"))) is not None
                and hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
                == record["sha256"],
                f"{rung} archived {name} byte drift",
            )
            texts[name] = record["text"]
        require(
            files["SHA256SUMS"]["sha256"] == prerequisite["manifest_sha256"]
            and files["COMPLETE"]["sha256"] == prerequisite["complete_sha256"],
            f"{rung} archived closure hash drift",
        )
        listed: dict[str, str] = {}
        for line in texts["SHA256SUMS"].splitlines():
            match = MANIFEST_RE.fullmatch(line)
            require(match is not None, f"{rung} archived manifest row drift")
            digest, raw_name = match.groups()
            name = raw_name.removeprefix("./")
            relative = _safe_relative(name)
            require(
                relative.as_posix() not in listed,
                f"{rung} archived manifest duplicate path",
            )
            listed[relative.as_posix()] = digest
        require(
            listed.get("receipt.tsv") == files["receipt.tsv"]["sha256"],
            f"{rung} archived receipt/manifest binding drift",
        )
        lines = texts["receipt.tsv"].splitlines()
        require(
            lines and lines[0] == "field\tvalue",
            f"{rung} archived receipt header drift",
        )
        receipt: dict[str, str] = {}
        for line in lines[1:]:
            fields = line.split("\t", 1)
            require(
                len(fields) == 2
                and fields[0]
                and fields[1]
                and fields[0] not in receipt,
                f"{rung} archived receipt row drift",
            )
            receipt[fields[0]] = fields[1]
        result_path = PurePosixPath(prerequisite["result_dir"])
        require(
            result_path.is_absolute()
            and all(part not in {"", ".", ".."} for part in result_path.parts)
            and receipt.get("job_id") == result_path.name
            and receipt.get("result_dir") == prerequisite["result_dir"]
            and receipt.get("bundle_manifest_sha256")
            == closure["bundle_manifest_sha256"]
            and receipt.get("applied_overlay_manifest_sha256")
            == closure["applied_overlay_manifest_sha256"],
            f"{rung} archived receipt provenance drift",
        )
        if rung == "import":
            require(
                receipt.get("training_endpoint") == "false"
                and re.fullmatch(
                    r"complete\t[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\n",
                    texts["COMPLETE"],
                ) is not None,
                "import archived completion semantics drift",
            )
        else:
            require(
                receipt.get("endpoint_class")
                == "bounded_engineering_one_update"
                and receipt.get("actual_student_updates") == "1"
                and receipt.get("paper_evidence") == "false"
                and receipt.get("config_sha256") == frontier_config_sha256,
                "one_update archived receipt semantics drift",
            )
            try:
                complete = json.loads(
                    texts["COMPLETE"], object_pairs_hook=_unique_pairs
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise AssemblyError(
                    "invalid one_update archived completion"
                ) from exc
            require(
                isinstance(complete, dict)
                and set(complete) == {
                    "complete_schema", "artifact_type", "job_id",
                    "paper_evidence", "actual_ppo_updates", "n_grad_updates",
                    "ppo_epochs", "ppo_minibatches",
                    "optimizer_step_applications", "resource_accounting_source",
                    "external_accounting_authority", "terminal_sacct_included",
                    "input_closure_sha256", "sha256sums_sha256",
                }
                and complete.get("complete_schema") == 2
                and complete.get("artifact_type")
                == "frontier_exact_grouped_one_update_engineering"
                and complete.get("paper_evidence") is False
                and complete.get("job_id") == receipt.get("job_id")
                and complete.get("input_closure_sha256")
                == receipt.get("input_closure_sha256")
                and complete.get("actual_ppo_updates")
                == complete.get("n_grad_updates") == 1
                and complete.get("terminal_sacct_included") is False
                and complete.get("sha256sums_sha256")
                == prerequisite["manifest_sha256"],
                "one_update archived completion semantics drift",
            )


def _validate_components(
    root: Path,
    protocol_path: Path,
    expected_manifest_sha: str,
    *,
    local_test_mode: bool,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    require(HASH_RE.fullmatch(expected_manifest_sha) is not None, "malformed component manifest hash")
    manifest_sha, complete = validate_manifest_tree(root, "SHA256SUMS", "COMPONENTS_COMPLETE.json")
    require(manifest_sha == expected_manifest_sha, "component manifest hash mismatch")
    required_complete = {
        "schema", "status", "paper_evidence", "analyzer_eligible", "endpoint_class",
        "job_id", "run_id", "arm", "bundle_manifest_sha256", "input_closure_sha256",
        "campaign_manifest_sha256", "run_context_sha256", "training_sidecar_manifest_sha256",
        "evaluation_package_manifest_sha256", "evaluation_integrity_sha256",
        "actual_student_updates",
        "actual_external_evaluation", "raw_evaluation_records", "terminal_sacct_included",
        "phase_b_required", "config_sha256", "student_training_transitions",
        "optimizer_step_applications", "outer_cycles", "training_wall_seconds",
        "primary_evaluation_max_transitions",
    }
    require(set(complete) == required_complete, "component completion keys drift")
    require(complete["schema"] == 1 and complete["status"] == "complete", "components incomplete")
    require(complete["paper_evidence"] is False and complete["analyzer_eligible"] is False, "component evidence label drift")
    require(complete["endpoint_class"] == "bounded_engineering_terminal_chain_components_v4", "component class drift")
    require(complete["actual_student_updates"] == 1, "component update count drift")
    require(complete["actual_external_evaluation"] is True, "external evaluation missing")
    require(complete["raw_evaluation_records"] == 30, "evaluation record-count receipt drift")
    require(complete["outer_cycles"] == 2, "outer-cycle receipt drift")
    require(
        complete["student_training_transitions"]
        == (128 if local_test_mode else 16_384),
        "training transition receipt drift",
    )
    require(
        complete["optimizer_step_applications"]
        == (1 if local_test_mode else 5),
        "optimizer application receipt drift",
    )
    require(
        isinstance(complete["training_wall_seconds"], (int, float))
        and not isinstance(complete["training_wall_seconds"], bool)
        and math.isfinite(float(complete["training_wall_seconds"]))
        and float(complete["training_wall_seconds"]) > 0.0,
        "training wall-time receipt drift",
    )
    require(complete["primary_evaluation_max_transitions"] == 13_500, "evaluation transition receipt drift")
    require(complete["terminal_sacct_included"] is False and complete["phase_b_required"] is True, "Phase-B boundary drift")
    context, campaign, protocol, _protocol_sha = _validate_context_campaign(
        root, protocol_path, local_test_mode=local_test_mode
    )
    for key in ("job_id", "run_id", "arm"):
        require(complete[key] == context[key], f"component/context drift: {key}")
    require(complete["campaign_manifest_sha256"] == sha256(root / "campaign-manifest.json"), "campaign hash drift")
    require(complete["run_context_sha256"] == sha256(root / "run-context.json"), "context hash drift")
    input_closure_sha = sha256(root / "INPUT_CLOSURE.json")
    require(complete["input_closure_sha256"] == input_closure_sha, "input closure drift")
    input_closure = load_json(root / "INPUT_CLOSURE.json", "input closure")
    require(
        input_closure.get("schema") == 1
        and input_closure.get("status") == "frozen_before_phase_a"
        and input_closure.get("purpose") == training.PURPOSE
        and input_closure.get("endpoint_class")
        == "bounded_engineering_terminal_chain_components_v4",
        "input closure identity drift",
    )
    for key in (
        "paper_evidence", "analyzer_eligible", "endpoint_access_authorized",
        "production_authorized", "cost100_implemented", "from_last_checkpoint",
        "periodic_checkpoint_used",
    ):
        require(input_closure.get(key) is False, f"input closure enables {key}")
    require(
        input_closure.get("max_student_updates") == 1
        and input_closure.get("attempt") == 1
        and input_closure.get("archive_interval") == 0
        and input_closure.get("no_requeue") is True,
        "input closure budget/resume drift",
    )
    require(
        input_closure.get("arm") == context["arm"]
        and input_closure.get("job_id") == context["job_id"]
        and input_closure.get("training_seed") == context["training_seed"],
        "input closure execution identity drift",
    )
    provenance = context["provenance"]
    campaign_provenance = campaign["provenance"]
    for closure_key, provenance_key in (
        ("bundle_manifest_sha256", "bundle_manifest_sha256"),
        ("overlay_manifest_sha256", "overlay_manifest_sha256"),
        ("applied_overlay_manifest_sha256", "applied_overlay_manifest_sha256"),
        ("environment_manifest_sha256", "environment_manifest_sha256"),
        ("sbatch_sha256", "sbatch_sha256"),
        ("training_driver_sha256", "training_driver_sha256"),
        ("evaluation_driver_sha256", "evaluation_driver_sha256"),
    ):
        require(
            input_closure.get(closure_key) == provenance[provenance_key],
            f"input closure/context drift: {closure_key}",
        )
    # The protocol hash belongs to the campaign rather than run-context's
    # projection; validate both explicitly without broadening HASH_KEYS.
    require(
        input_closure.get("assembler_sha256")
        == campaign_provenance["assembler_driver_sha256"]
        and input_closure.get("protocol_sha256") == campaign["protocol_sha256"]
        and input_closure.get("source_commit") == training.BASE_COMMIT
        and input_closure.get("source_tree") == training.BASE_TREE
        and input_closure.get("config_sha256") == complete["config_sha256"],
        "input closure static binding drift",
    )
    expected_config = protocol["arms"][context["arm"]]["config_sha256"]
    require(complete["config_sha256"] == expected_config, "arm config hash drift")
    _validate_archived_prerequisites(
        input_closure, protocol["arms"]["frontier"]["config_sha256"]
    )
    sidecar_sha, _receipt = _validate_replay_sidecar(root, context)
    require(complete["training_sidecar_manifest_sha256"] == sidecar_sha, "sidecar manifest drift")
    try:
        evaluation_sha = evaluation.validate_package(
            root / "evaluation-package", str(context["run_id"])
        )
    except evaluation.EvaluationError as exc:
        raise AssemblyError(str(exc)) from exc
    require(complete["evaluation_package_manifest_sha256"] == evaluation_sha, "evaluation manifest drift")
    evaluation_integrity_path = root / "evaluation-integrity.json"
    require(
        complete["evaluation_integrity_sha256"] == sha256(evaluation_integrity_path),
        "evaluation integrity hash drift",
    )
    evaluation_integrity = load_json(evaluation_integrity_path, "evaluation integrity")
    require(
        evaluation_integrity
        == {
            "schema": 1,
            "status": "complete",
            "paper_evidence": False,
            "analyzer_eligible": False,
            "performance_values_copied": False,
            "run_id": context["run_id"],
            "arm": context["arm"],
            "training_seed": context["training_seed"],
            "evaluation_seed": 100000 + context["training_seed"],
            "synthetic_test_mode": False,
            "raw_record_count": 30,
            "primary_max_transitions": 13_500,
            "evaluation_receipt_sha256": sha256(
                root / "evaluation-package" / "evaluation-receipt.json"
            ),
            "evaluation_package_manifest_sha256": evaluation_sha,
        },
        "evaluation integrity semantics drift",
    )
    # Evaluation payload values remain sealed: this assembler hashes JSONL,
    # CSV, and the evaluator receipt but parses only this value-free integrity
    # projection and evaluation COMPLETE.
    return complete, context, manifest_sha, input_closure_sha


def _validate_scheduler(
    path: Path,
    context: Mapping[str, Any],
    *,
    local_test_mode: bool,
    components_manifest_sha256: str,
) -> dict[str, Any]:
    scheduler = load_json(path, "scheduler receipt")
    required = {
        "schema", "job_id", "job_name", "arm", "state", "exit_code",
        "partition", "qos", "gpu_profile", "gpu_count", "cpus", "memory",
        "elapsed_raw", "max_rss_bytes", "resource_rows", "terminal_sacct_sha256",
        "submission_receipt_sha256", "fetch_receipt_sha256",
        "terminal_sacct_included", "fetched_after_terminal", "restarts",
        "array_job", "phase_b_mode", "components_manifest_sha256",
        "bundle_manifest_sha256", "sbatch_sha256", "submit_line_sha256",
        "submission_export_mode", "phase_b_receipts_manifest_sha256",
    }
    require(set(scheduler) == required, "scheduler receipt keys drift")
    require(scheduler["schema"] == 1, "scheduler schema drift")
    require(scheduler["job_id"] == context["job_id"], "scheduler job drift")
    require(scheduler["arm"] == context["arm"], "scheduler arm drift")
    require(scheduler["job_name"] == "ued-v4-terminal", "scheduler name drift")
    require(scheduler["state"] == "COMPLETED" and scheduler["exit_code"] == "0:0", "job not cleanly complete")
    if local_test_mode:
        require(scheduler["partition"] == "local" and scheduler["qos"] == "local", "local scheduler queue drift")
        require(scheduler["gpu_profile"] == "local-cpu" and scheduler["gpu_count"] == 1, "local device receipt drift")
    else:
        require(scheduler["partition"] == "gpuq" and scheduler["qos"] == "gpu", "scheduler queue drift")
        require(scheduler["gpu_profile"] == "1g.10gb" and scheduler["gpu_count"] == 1, "MIG allocation drift")
    require(scheduler["cpus"] == 2 and scheduler["memory"] == "15G", "host allocation drift")
    require(isinstance(scheduler["elapsed_raw"], int) and scheduler["elapsed_raw"] >= 0, "elapsed time invalid")
    require(
        isinstance(scheduler["max_rss_bytes"], int)
        and scheduler["max_rss_bytes"] > 0,
        "MaxRSS byte receipt drift",
    )
    rows = scheduler["resource_rows"]
    require(isinstance(rows, list) and rows, "resource rows missing")
    seen_rows: set[str] = set()
    parsed_sizes: list[int] = []
    tres_seen = False
    for row in rows:
        require(
            isinstance(row, dict)
            and set(row) == {"job_id", "max_rss", "tres_usage_in_max"}
            and isinstance(row["job_id"], str)
            and (
                row["job_id"] == str(context["job_id"])
                or row["job_id"].startswith(f"{context['job_id']}.")
            )
            and row["job_id"] not in seen_rows
            and isinstance(row["max_rss"], str)
            and isinstance(row["tres_usage_in_max"], str),
            "resource row drift",
        )
        seen_rows.add(row["job_id"])
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTP]?)", row["max_rss"])
        require(match is not None, "resource MaxRSS format drift")
        exponent = {"": 0, "K": 1, "M": 2, "G": 3, "T": 4, "P": 5}[
            match.group(2)
        ]
        parsed_sizes.append(int(float(match.group(1)) * (1024 ** exponent)))
        tres_seen = tres_seen or bool(row["tres_usage_in_max"])
    require(
        parsed_sizes
        and max(parsed_sizes) == scheduler["max_rss_bytes"]
        and tres_seen,
        "resource accounting summary drift",
    )
    if not local_test_mode:
        require(
            f"{context['job_id']}.batch" in seen_rows
            and f"{context['job_id']}.extern" in seen_rows,
            "batch/extern resource rows missing",
        )
    require(scheduler["terminal_sacct_included"] is True, "terminal sacct missing")
    require(scheduler["fetched_after_terminal"] is True, "preterminal fetch forbidden")
    require(scheduler["restarts"] == 0, "restarted job forbidden")
    require(scheduler["array_job"] is False, "array job forbidden")
    require(scheduler["phase_b_mode"] == ("local_fixture" if local_test_mode else "post_terminal_local"), "Phase-B mode drift")
    require(scheduler["components_manifest_sha256"] == components_manifest_sha256, "scheduler/component binding drift")
    require(scheduler["bundle_manifest_sha256"] == context["provenance"]["bundle_manifest_sha256"], "scheduler/bundle binding drift")
    require(scheduler["sbatch_sha256"] == context["provenance"]["sbatch_sha256"], "scheduler/sbatch binding drift")
    require(
        scheduler["submission_export_mode"]
        == "explicit_assignments_no_all_or_none",
        "scheduler export mode drift",
    )
    for key in ("terminal_sacct_sha256", "submission_receipt_sha256", "fetch_receipt_sha256"):
        require(HASH_RE.fullmatch(str(scheduler[key])) is not None, f"bad scheduler hash: {key}")
    require(
        HASH_RE.fullmatch(str(scheduler["phase_b_receipts_manifest_sha256"]))
        is not None,
        "bad Phase-B receipt archive hash",
    )
    return scheduler


def _validate_phase_b_receipts(
    root: Path,
    expected_manifest_sha256: str,
    scheduler: Mapping[str, Any],
) -> str:
    require(
        {entry.name for entry in root.iterdir()}
        == {"terminal.tsv", "submission.tsv", "fetch.tsv", "SHA256SUMS", "COMPLETE"},
        "Phase-B receipt archive closure drift",
    )
    manifest_sha, complete = validate_manifest_tree(root, "SHA256SUMS", "COMPLETE")
    require(manifest_sha == expected_manifest_sha256, "Phase-B receipt archive hash drift")
    require(
        complete
        == {
            "schema": 1,
            "status": "complete",
            "sha256sums_sha256": manifest_sha,
            "file_count": 3,
        },
        "Phase-B receipt archive completion drift",
    )
    require(
        sha256(root / "terminal.tsv") == scheduler["terminal_sacct_sha256"]
        and sha256(root / "submission.tsv") == scheduler["submission_receipt_sha256"]
        and sha256(root / "fetch.tsv") == scheduler["fetch_receipt_sha256"]
        and scheduler["phase_b_receipts_manifest_sha256"] == manifest_sha,
        "Phase-B receipt/scheduler binding drift",
    )
    return manifest_sha


def _copy_closed_tree(source: Path, destination: Path) -> None:
    require(source.is_dir() and not source.is_symlink(), f"unsafe source tree: {source}")
    _all_files(source, set())
    shutil.copytree(source, destination, symlinks=False)


def _write_root_closure(root: Path, run_id: str, arm: str) -> str:
    payloads = sorted(_all_files(root, {"SHA256SUMS", "COMPLETE"}))
    atomic_text(
        root / "SHA256SUMS",
        "".join(f"{sha256(root / name)}  {name}\n" for name in payloads),
    )
    manifest_sha = sha256(root / "SHA256SUMS")
    atomic_json(
        root / "COMPLETE",
        {
            "schema": 1,
            "status": "complete",
            "paper_evidence": False,
            "analyzer_eligible": False,
            "production_analyzer_invoked": False,
            "endpoint_class": "bounded_engineering_terminal_package_v4",
            "run_id": run_id,
            "arm": arm,
            "sha256sums_sha256": manifest_sha,
            "file_count": len(payloads),
        },
    )
    return manifest_sha


def validate_package(root: Path) -> str:
    require(root.is_dir() and not root.is_symlink(), "assembled package missing")
    require({entry.name for entry in root.iterdir()} == PACKAGE_TOP_LEVEL, "assembled top-level closure drift")
    manifest_sha, complete = validate_manifest_tree(root, "SHA256SUMS", "COMPLETE")
    require(
        set(complete)
        == {
            "schema", "status", "paper_evidence", "analyzer_eligible",
            "production_analyzer_invoked", "endpoint_class", "run_id",
            "arm", "sha256sums_sha256", "file_count",
        },
        "assembled completion keys drift",
    )
    require(
        complete.get("schema") == 1
        and complete.get("status") == "complete"
        and complete.get("paper_evidence") is False
        and complete.get("analyzer_eligible") is False
        and complete.get("production_analyzer_invoked") is False
        and complete.get("endpoint_class") == "bounded_engineering_terminal_package_v4"
        and complete.get("sha256sums_sha256") == manifest_sha,
        "assembled completion drift",
    )
    manifest = load_json(root / "run-manifest.json", "run manifest")
    require(
        set(manifest)
        == {
            "schema", "status", "protocol_id", "purpose", "paper_evidence",
            "analyzer_eligible", "production_analyzer_invoked",
            "performance_values_inspected", "endpoint_class", "run_id", "arm",
            "training_seed", "job_id", "actual_student_updates",
            "actual_external_evaluation", "raw_evaluation_records_sealed",
            "student_training_transitions", "optimizer_step_applications",
            "outer_cycles", "training_wall_seconds",
            "primary_evaluation_max_transitions", "bundle_manifest_sha256",
            "input_closure_sha256", "components_manifest_sha256",
            "scheduler_receipt_sha256", "config_sha256",
            "evaluation_integrity_sha256", "phase_b_receipts_manifest_sha256",
        },
        "assembled run-manifest keys drift",
    )
    require(
        complete["run_id"] == manifest.get("run_id")
        and complete["arm"] == manifest.get("arm"),
        "assembled completion identity drift",
    )
    require(
        complete["file_count"]
        == len(_all_files(root, {"SHA256SUMS", "COMPLETE"})),
        "assembled completion file-count drift",
    )
    require(manifest.get("paper_evidence") is False, "assembled evidence label drift")
    require(manifest.get("analyzer_eligible") is False, "assembled analyzer label drift")
    require(manifest.get("production_analyzer_invoked") is False, "analyzer invocation drift")
    require(manifest.get("performance_values_inspected") is False, "structural-only guarantee drift")
    require(
        manifest.get("schema") == 1
        and manifest.get("status") == "complete"
        and manifest.get("protocol_id") == training.PROTOCOL_ID
        and manifest.get("purpose") == training.PURPOSE
        and manifest.get("endpoint_class")
        == "bounded_engineering_terminal_package_v4"
        and manifest.get("training_seed") == 101
        and manifest.get("actual_student_updates") == 1
        and manifest.get("actual_external_evaluation") is True
        and manifest.get("raw_evaluation_records_sealed") == 30
        and manifest.get("outer_cycles") == 2
        and manifest.get("primary_evaluation_max_transitions") == 13_500,
        "assembled run-manifest semantics drift",
    )
    for key in (
        "bundle_manifest_sha256", "input_closure_sha256",
        "components_manifest_sha256", "scheduler_receipt_sha256",
        "config_sha256", "evaluation_integrity_sha256",
        "phase_b_receipts_manifest_sha256",
    ):
        require(HASH_RE.fullmatch(str(manifest[key])) is not None, f"bad run-manifest hash: {key}")
    context = load_json(root / "run-context.json", "packaged run context")
    require(
        context.get("run_id") == manifest["run_id"]
        and context.get("arm") == manifest["arm"]
        and context.get("training_seed") == manifest["training_seed"]
        and context.get("job_id") == manifest["job_id"],
        "packaged context/run-manifest identity drift",
    )
    components_manifest_sha = sha256(root / "components-SHA256SUMS")
    scheduler_path = root / "scheduler.json"
    input_closure = load_json(root / "INPUT_CLOSURE.json", "packaged input closure")
    require(
        manifest["input_closure_sha256"] == sha256(root / "INPUT_CLOSURE.json")
        and manifest["components_manifest_sha256"] == components_manifest_sha
        and manifest["scheduler_receipt_sha256"] == sha256(scheduler_path)
        and manifest["evaluation_integrity_sha256"]
        == sha256(root / "evaluation-integrity.json")
        and manifest["config_sha256"] == input_closure.get("config_sha256")
        and manifest["bundle_manifest_sha256"]
        == input_closure.get("bundle_manifest_sha256"),
        "packaged run-manifest payload binding drift",
    )
    scheduler_projection = load_json(scheduler_path, "packaged scheduler")
    require(
        scheduler_projection.get("phase_b_mode")
        in {"local_fixture", "post_terminal_local"},
        "packaged scheduler Phase-B mode drift",
    )
    scheduler = _validate_scheduler(
        scheduler_path,
        context,
        local_test_mode=scheduler_projection["phase_b_mode"] == "local_fixture",
        components_manifest_sha256=components_manifest_sha,
    )
    receipt_manifest_sha = _validate_phase_b_receipts(
        root / "phase-b-receipts",
        manifest["phase_b_receipts_manifest_sha256"],
        scheduler,
    )
    require(
        receipt_manifest_sha == scheduler["phase_b_receipts_manifest_sha256"],
        "packaged Phase-B receipt/run-manifest drift",
    )
    source_snapshot = root / "training-sidecar" / "plr-replay-snapshot.json"
    flat_snapshot = root / "training-plr-replay-snapshot.json"
    require(
        sha256(source_snapshot) == sha256(flat_snapshot),
        "flat/source PLR snapshot drift",
    )
    training_receipt = load_json(
        root / "training-sidecar" / "training-receipt.json",
        "packaged training receipt",
    )
    require(
        training_receipt.get("plr_snapshot")
        == {
            "path": "plr-replay-snapshot.json",
            "sha256": sha256(source_snapshot),
        },
        "packaged PLR snapshot receipt binding drift",
    )
    snapshot = load_json(flat_snapshot, "flat PLR replay snapshot")
    require(
        snapshot.get("run_id") == manifest["run_id"]
        and snapshot.get("arm") == manifest["arm"]
        and snapshot.get("training_seed") == manifest["training_seed"]
        and snapshot.get("protocol_id") == training.PROTOCOL_ID
        and snapshot.get("purpose") == training.PURPOSE
        and snapshot.get("paper_evidence") is False,
        "packaged PLR snapshot identity drift",
    )
    return manifest_sha


def assemble(cli: argparse.Namespace) -> tuple[Path, str]:
    driver = Path(__file__).resolve()
    require(HASH_RE.fullmatch(cli.expected_assembler_sha256 or "") is not None, "bad expected assembler hash")
    require(sha256(driver) == cli.expected_assembler_sha256, "assembler hash mismatch")
    components = canonical_existing(
        cli.components_dir, directory=True, label="components"
    )
    protocol = canonical_existing(cli.protocol, directory=False, label="protocol")
    scheduler_receipt = canonical_existing(
        cli.scheduler_receipt, directory=False, label="scheduler receipt"
    )
    phase_b_receipts = canonical_existing(
        cli.phase_b_receipts_dir, directory=True, label="Phase-B receipt archive"
    )
    complete, context, components_sha, input_closure_sha = _validate_components(
        components, protocol, cli.expected_components_manifest_sha256,
        local_test_mode=cli.local_test_mode,
    )
    require(sha256(scheduler_receipt) == cli.expected_scheduler_receipt_sha256, "scheduler receipt hash mismatch")
    scheduler = _validate_scheduler(
        scheduler_receipt, context,
        local_test_mode=cli.local_test_mode,
        components_manifest_sha256=components_sha,
    )
    phase_b_receipts_sha = _validate_phase_b_receipts(
        phase_b_receipts,
        cli.expected_phase_b_receipts_manifest_sha256,
        scheduler,
    )
    cli.output_dir = canonical_new_output(
        cli.output_dir, [components, phase_b_receipts]
    )
    temporary = Path(tempfile.mkdtemp(prefix=f".{cli.output_dir.name}.", dir=cli.output_dir.parent))
    try:
        for directory in ("training-output", "training-sidecar", "evaluation-package"):
            _copy_closed_tree(components / directory, temporary / directory)
        _copy_closed_tree(phase_b_receipts, temporary / "phase-b-receipts")
        for name in ("INPUT_CLOSURE.json", "campaign-manifest.json", "run-context.json"):
            shutil.copy2(components / name, temporary / name, follow_symlinks=False)
        shutil.copy2(
            components / "evaluation-integrity.json",
            temporary / "evaluation-integrity.json",
            follow_symlinks=False,
        )
        shutil.copy2(
            components / "training-sidecar" / "plr-replay-snapshot.json",
            temporary / "training-plr-replay-snapshot.json",
            follow_symlinks=False,
        )
        shutil.copy2(components / "SHA256SUMS", temporary / "components-SHA256SUMS", follow_symlinks=False)
        shutil.copy2(components / "COMPONENTS_COMPLETE.json", temporary / "components-COMPLETE.json", follow_symlinks=False)
        atomic_json(temporary / "scheduler.json", scheduler)
        atomic_json(
            temporary / "run-manifest.json",
            {
                "schema": 1,
                "status": "complete",
                "protocol_id": training.PROTOCOL_ID,
                "purpose": training.PURPOSE,
                "paper_evidence": False,
                "analyzer_eligible": False,
                "production_analyzer_invoked": False,
                "performance_values_inspected": False,
                "endpoint_class": "bounded_engineering_terminal_package_v4",
                "run_id": context["run_id"],
                "arm": context["arm"],
                "training_seed": context["training_seed"],
                "job_id": context["job_id"],
                "actual_student_updates": 1,
                "actual_external_evaluation": True,
                "raw_evaluation_records_sealed": complete["raw_evaluation_records"],
                "student_training_transitions": complete["student_training_transitions"],
                "optimizer_step_applications": complete["optimizer_step_applications"],
                "outer_cycles": complete["outer_cycles"],
                "training_wall_seconds": complete["training_wall_seconds"],
                "primary_evaluation_max_transitions": complete["primary_evaluation_max_transitions"],
                "bundle_manifest_sha256": complete["bundle_manifest_sha256"],
                "input_closure_sha256": input_closure_sha,
                "components_manifest_sha256": components_sha,
                "scheduler_receipt_sha256": cli.expected_scheduler_receipt_sha256,
                "config_sha256": complete["config_sha256"],
                "evaluation_integrity_sha256": complete["evaluation_integrity_sha256"],
                "phase_b_receipts_manifest_sha256": phase_b_receipts_sha,
            },
        )
        manifest_sha = _write_root_closure(temporary, context["run_id"], context["arm"])
        validate_package(temporary)
        os.replace(temporary, cli.output_dir)
        require(validate_package(cli.output_dir) == manifest_sha, "post-publish manifest drift")
        return cli.output_dir, manifest_sha
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components-dir", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--expected-components-manifest-sha256")
    parser.add_argument("--scheduler-receipt", type=Path)
    parser.add_argument("--expected-scheduler-receipt-sha256")
    parser.add_argument("--expected-assembler-sha256")
    parser.add_argument("--phase-b-receipts-dir", type=Path)
    parser.add_argument("--expected-phase-b-receipts-manifest-sha256")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-only", type=Path)
    parser.add_argument("--local-test-mode", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    cli = parse_cli(argv)
    try:
        if cli.validate_only is not None:
            require(
                all(
                    value is None
                    for value in (
                        cli.components_dir,
                        cli.protocol,
                        cli.expected_components_manifest_sha256,
                        cli.scheduler_receipt,
                        cli.expected_scheduler_receipt_sha256,
                        cli.expected_assembler_sha256,
                        cli.phase_b_receipts_dir,
                        cli.expected_phase_b_receipts_manifest_sha256,
                        cli.output_dir,
                    )
                ) and not cli.local_test_mode,
                "validate-only cannot be combined with assembly arguments",
            )
            digest = validate_package(
                canonical_existing(
                    cli.validate_only, directory=True, label="validation package"
                )
            )
            print(f"V4_ENGINEERING_PACKAGE_VALID manifest={digest}")
            return 0
        require(
            all(
                value is not None
                for value in (
                    cli.components_dir,
                    cli.protocol,
                    cli.expected_components_manifest_sha256,
                    cli.scheduler_receipt,
                    cli.expected_scheduler_receipt_sha256,
                    cli.expected_assembler_sha256,
                    cli.phase_b_receipts_dir,
                    cli.expected_phase_b_receipts_manifest_sha256,
                    cli.output_dir,
                )
            ),
            "assembly arguments are incomplete",
        )
        output, digest = assemble(cli)
    except (AssemblyError, training.DriverError, evaluation.EvaluationError, OSError, ValueError, KeyError) as exc:
        print(f"V4_ENGINEERING_ASSEMBLY_REFUSED: {exc}", file=os.sys.stderr)
        return 1
    print(
        "V4_ENGINEERING_ASSEMBLY_COMPLETE "
        f"manifest={digest} result={output} analyzer_eligible=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
