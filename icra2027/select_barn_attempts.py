"""Outcome-blind retry selection from the normalized BARN submission ledger.

Selection uses submission time and structural/hash validity only.  It never
reads an outcome to rank attempts and never prints endpoint-bearing content.
"""

from __future__ import annotations

import argparse
import copy
import hmac
import re
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from icra2027 import merge_barn_campaign as merger
except ModuleNotFoundError:  # Support direct execution from a staged source tree.
    import merge_barn_campaign as merger


LEDGER_FIELDS = {
    "campaign_id",
    "campaign_cell",
    "attempt_id",
    "seed",
    "submitted_utc",
    "slurm_array_job_id",
    "slurm_array_task_id",
    "slurm_job_id",
    "artifact_path",
    "artifact_complete",
    "artifact_sha256",
    "expected_hashes",
}

# N=8 is supplied by the primary cell's ours_uN/learnability arms.  There is
# deliberately no separately submitted ablation_n8 campaign cell.
PRODUCTION_CAMPAIGN_CELLS = (
    "primary", "ablation_n2", "ablation_n4", "ablation_n16")
PRODUCTION_SEEDS = (1, 2, 3, 4, 5)


class AttemptSelectionError(merger.MergeValidationError):
    """The submission ledger cannot support a blind retry selection."""


def _normalize_row(
    row: object, *, ledger_base: Path, index: int
) -> dict[str, Any]:
    label = f"ledger submission[{index}]"
    if not isinstance(row, dict) or set(row) != LEDGER_FIELDS:
        raise AttemptSelectionError(f"{label} fields are not normalized")
    seed = merger._require_int(row["seed"], field=f"{label} seed")
    task_id = merger._require_int(
        row["slurm_array_task_id"], field=f"{label} slurm_array_task_id")
    if task_id != seed:
        raise AttemptSelectionError(
            f"{label} array task ID must equal the campaign seed")
    if row["campaign_cell"] not in merger.CAMPAIGN_CELLS:
        raise AttemptSelectionError(f"{label} has an unknown campaign cell")
    execution = {
        "campaign_id": row["campaign_id"],
        "attempt_id": row["attempt_id"],
        "submitted_utc": row["submitted_utc"],
        "slurm_job_id": (row["slurm_job_id"]
                         if row["slurm_job_id"] is not None
                         else f"{row['slurm_array_job_id']}_{task_id}"),
        "slurm_array_job_id": row["slurm_array_job_id"],
        "slurm_array_task_id": task_id,
    }
    try:
        normalized_execution = merger._normalize_execution(
            execution, label=label)
    except merger.MergeValidationError as error:
        raise AttemptSelectionError(str(error)) from error
    if row["slurm_job_id"] is not None:
        normalized_job_id: str | None = normalized_execution["slurm_job_id"]
    else:
        normalized_job_id = None
    path_value = row["artifact_path"]
    if not isinstance(path_value, str) or not path_value:
        raise AttemptSelectionError(f"{label} artifact_path must be non-empty")
    artifact_path = Path(path_value)
    if not artifact_path.is_absolute():
        artifact_path = ledger_base / artifact_path
    artifact_path = artifact_path.resolve()
    complete = row["artifact_complete"]
    if not isinstance(complete, bool):
        raise AttemptSelectionError(f"{label} artifact_complete must be boolean")
    if complete:
        artifact_sha = merger._require_sha256(
            row["artifact_sha256"], field=f"{label} artifact_sha256")
    else:
        if row["artifact_sha256"] is not None:
            raise AttemptSelectionError(
                f"{label} incomplete row must have null artifact_sha256")
        artifact_sha = None
    expected_hashes = row["expected_hashes"]
    if (not isinstance(expected_hashes, dict)
            or set(expected_hashes) != set(merger.PROVENANCE_HASH_FIELDS)):
        raise AttemptSelectionError(
            f"{label} expected_hashes must contain exactly seven bindings")
    normalized_hashes = {
        field: merger._require_sha256(
            expected_hashes[field], field=f"{label} expected_hashes {field}")
        for field in merger.PROVENANCE_HASH_FIELDS}
    return {
        "campaign_id": normalized_execution["campaign_id"],
        "campaign_cell": row["campaign_cell"],
        "attempt_id": normalized_execution["attempt_id"],
        "seed": seed,
        "submitted_utc": normalized_execution["submitted_utc"],
        "slurm_array_job_id": normalized_execution["slurm_array_job_id"],
        "slurm_array_task_id": task_id,
        "slurm_job_id": normalized_job_id,
        "artifact_path": artifact_path,
        "artifact_complete": complete,
        "artifact_sha256": artifact_sha,
        "expected_hashes": normalized_hashes,
    }


def normalize_ledger(
    ledger: object, *, ledger_base: Path
) -> list[dict[str, Any]]:
    if (not isinstance(ledger, dict) or set(ledger) != {
            "schema_version", "submissions"}
            or ledger.get("schema_version") != 1
            or not isinstance(ledger.get("submissions"), list)):
        raise AttemptSelectionError("unsupported normalized submission ledger")
    rows = [
        _normalize_row(row, ledger_base=ledger_base, index=index)
        for index, row in enumerate(ledger["submissions"])
    ]
    identities = [
        (row["campaign_id"], row["campaign_cell"], row["attempt_id"], row["seed"])
        for row in rows]
    duplicates = [identity for identity, count in Counter(identities).items()
                  if count > 1]
    if duplicates:
        raise AttemptSelectionError(
            "submission ledger contains duplicate campaign/cell/attempt/seed rows")
    paths = [row["artifact_path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise AttemptSelectionError("submission ledger reuses an artifact path")
    return rows


def validate_finalized_campaign_closure(
    *,
    ledger: Mapping[str, Any],
    ledger_base: Path,
    ledger_path: Path,
    ledger_sha256: str,
    campaign_id: str,
    expected_seeds: Sequence[int],
    expected_hashes: Mapping[str, str],
) -> None:
    """Fail closed unless a production ledger proves full-campaign closure.

    This preflight hashes every declared complete artifact but deliberately
    decodes none of them.  It finishes before ``select_attempts`` opens any
    target-cell artifact as JSON.
    """

    canonical_seeds = merger._canonical_expected_seeds(expected_seeds)
    if canonical_seeds != PRODUCTION_SEEDS:
        raise AttemptSelectionError(
            "production selection requires the frozen seed list 1,2,3,4,5")

    ledger_path = Path(ledger_path)
    if not ledger_path.is_absolute():
        raise AttemptSelectionError(
            "production selection requires an absolute finalized ledger path")
    try:
        resolved_ledger_path = ledger_path.resolve(strict=True)
    except OSError as error:
        raise AttemptSelectionError(
            "production finalized ledger path cannot be resolved") from error
    if (resolved_ledger_path != ledger_path or ledger_path.is_symlink()
            or not ledger_path.is_file()
            or ledger_path.parent.name != "finalized_ledgers"):
        raise AttemptSelectionError(
            "production finalized ledger path is not canonical")
    campaign_root = ledger_path.parent.parent
    if campaign_root.name != campaign_id:
        raise AttemptSelectionError(
            "finalized ledger campaign root differs from the requested campaign")

    normalized_ledger_sha256 = merger._require_sha256(
        ledger_sha256, field="submission ledger SHA-256")
    expected_name = (
        f"SUBMISSION_LEDGER.finalized-{normalized_ledger_sha256}.json")
    if Path(ledger_path).name != expected_name:
        raise AttemptSelectionError(
            "production selection requires the canonical content-addressed "
            f"finalized ledger filename {expected_name}")

    if set(expected_hashes) != set(merger.PROVENANCE_HASH_FIELDS):
        raise AttemptSelectionError(
            "selector expected_hashes must contain exactly seven bindings")
    normalized_hashes = {
        field: merger._require_sha256(
            expected_hashes[field], field=f"selector expected {field}")
        for field in merger.PROVENANCE_HASH_FIELDS}

    rows = normalize_ledger(ledger, ledger_base=Path(ledger_base).resolve())
    if not rows:
        raise AttemptSelectionError("finalized campaign ledger is empty")

    campaign_ids = {row["campaign_id"] for row in rows}
    if campaign_ids != {campaign_id}:
        raise AttemptSelectionError(
            "finalized ledger campaign ID differs from the requested campaign")

    actual_cells = {row["campaign_cell"] for row in rows}
    expected_cells = set(PRODUCTION_CAMPAIGN_CELLS)
    if actual_cells != expected_cells:
        raise AttemptSelectionError(
            "finalized ledger campaign-cell closure differs: "
            f"missing={sorted(expected_cells - actual_cells)}, "
            f"extra={sorted(actual_cells - expected_cells)}")

    attempt_groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    array_owners: dict[str, tuple[str, str, str]] = {}
    for row in rows:
        if row["expected_hashes"] != normalized_hashes:
            raise AttemptSelectionError(
                "finalized ledger expected_hashes differ across the campaign "
                "or from selector CLI bindings")
        job_id = row["slurm_job_id"]
        if not isinstance(job_id, str) or re.fullmatch(r"[0-9]+", job_id) is None:
            raise AttemptSelectionError(
                "every finalized ledger row must have a numeric slurm_job_id")
        array_id = row["slurm_array_job_id"]
        if re.fullmatch(r"[0-9]+", array_id) is None:
            raise AttemptSelectionError(
                "every finalized ledger row must have a numeric "
                "slurm_array_job_id")
        group_key = (
            row["campaign_cell"], row["attempt_id"], array_id)
        group = attempt_groups.setdefault(group_key, {
            "seeds": set(),
            "submitted_utc": row["submitted_utc"],
            "expected_hashes": row["expected_hashes"],
        })
        if (group["submitted_utc"] != row["submitted_utc"]
                or group["expected_hashes"] != row["expected_hashes"]):
            raise AttemptSelectionError(
                "one finalized attempt group has submission-time or "
                "expected-hash drift")
        group["seeds"].add(row["seed"])
        prior_owner = array_owners.setdefault(array_id, group_key)
        if prior_owner != group_key:
            raise AttemptSelectionError(
                "one Slurm array ID is reused across finalized attempt groups")

    for group in attempt_groups.values():
        if group["seeds"] != set(PRODUCTION_SEEDS):
            raise AttemptSelectionError(
                "a finalized campaign attempt group must contain exactly "
                "seeds 1,2,3,4,5")

    expected_coverage = set(product(PRODUCTION_CAMPAIGN_CELLS, PRODUCTION_SEEDS))
    actual_coverage = {
        (row["campaign_cell"], row["seed"]) for row in rows}
    if actual_coverage != expected_coverage:
        missing = sorted(expected_coverage - actual_coverage)
        extra = sorted(actual_coverage - expected_coverage)
        raise AttemptSelectionError(
            "finalized ledger cell/seed coverage differs: "
            f"missing={missing}, extra={extra}")

    complete_coverage = {
        (row["campaign_cell"], row["seed"])
        for row in rows if row["artifact_complete"]}
    missing_complete = sorted(expected_coverage - complete_coverage)
    if missing_complete:
        raise AttemptSelectionError(
            "finalized ledger lacks a complete artifact attempt for "
            f"cell/seed pairs: {missing_complete}")

    # Authenticate the complete four-cell artifact closure without decoding any
    # artifact JSON.  Target-cell JSON parsing happens only after this returns.
    for original, row in zip(ledger["submissions"], rows, strict=True):
        expected_artifact = (
            campaign_root / "cells" / row["campaign_cell"] / "attempts"
            / row["attempt_id"] / f"seed-{row['seed']}" / "results"
            / f"seed-{row['seed']}.json")
        if original["artifact_path"] != str(expected_artifact):
            raise AttemptSelectionError(
                "finalized ledger artifact path is not canonical for its "
                "campaign/cell/attempt/seed identity")
        if (row["artifact_path"] != expected_artifact
                or expected_artifact.resolve() != expected_artifact):
            raise AttemptSelectionError(
                "finalized ledger artifact path or ancestry is symbolic or "
                "noncanonical")
        if row["artifact_complete"]:
            if (not expected_artifact.is_file()
                    or expected_artifact.is_symlink()):
                raise AttemptSelectionError(
                    "finalized ledger complete artifact is missing or symbolic")
            actual_artifact_sha256 = merger.sha256_path(expected_artifact)
            if not hmac.compare_digest(
                    actual_artifact_sha256, row["artifact_sha256"]):
                raise AttemptSelectionError(
                    "finalized ledger complete artifact hash differs")
        elif expected_artifact.exists() or expected_artifact.is_symlink():
            raise AttemptSelectionError(
                "finalized ledger incomplete artifact path must be absent")


def _execution_matches(row: Mapping[str, Any], item: Mapping[str, Any]) -> bool:
    execution = item["execution"]
    if (execution["campaign_id"] != row["campaign_id"]
            or execution["attempt_id"] != row["attempt_id"]
            or execution["submitted_utc"] != row["submitted_utc"]
            or execution["slurm_array_job_id"] != row["slurm_array_job_id"]
            or execution["slurm_array_task_id"]
            != row["slurm_array_task_id"]):
        return False
    if row["slurm_job_id"] is not None:
        return execution["slurm_job_id"] == row["slurm_job_id"]
    return True


def select_attempts(
    *,
    ledger: Mapping[str, Any],
    ledger_base: Path,
    artifact_paths: Sequence[Path],
    campaign_id: str,
    campaign_cell: str,
    protocol: Mapping[str, Any],
    expected_seeds: Sequence[int],
    expected_manifest_sha256: str,
    expected_split_sha256: str,
    expected_prereg_sha256: str,
    expected_analyzer_sha256: str,
    expected_protocol_sha256: str,
    expected_container_sha256: str,
    expected_source_sha256: str,
    ledger_sha256: str,
) -> dict[str, Any]:
    """Select the earliest-submitted complete valid attempt for every seed."""

    canonical_seeds = merger._canonical_expected_seeds(expected_seeds)
    normalized_ledger_sha256 = merger._require_sha256(
        ledger_sha256, field="submission ledger SHA-256")
    hashes, contract = merger.prepare_validation(
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
    rows = normalize_ledger(ledger, ledger_base=Path(ledger_base).resolve())
    target = [row for row in rows
              if row["campaign_id"] == campaign_id
              and row["campaign_cell"] == campaign_cell]
    if not target:
        raise AttemptSelectionError("ledger has no rows for campaign/cell")
    for row in target:
        if row["expected_hashes"] != hashes:
            raise AttemptSelectionError(
                "ledger expected_hashes differ from selector CLI bindings")
    extra_seeds = sorted({row["seed"] for row in target} - set(canonical_seeds))
    missing_submissions = sorted(
        set(canonical_seeds) - {row["seed"] for row in target})
    if extra_seeds or missing_submissions:
        raise AttemptSelectionError(
            f"ledger seed coverage differs: missing={missing_submissions}, "
            f"extra={extra_seeds}")

    supplied = [Path(path).resolve() for path in artifact_paths]
    if len(supplied) != len(set(supplied)):
        raise AttemptSelectionError("duplicate supplied artifact paths")
    complete_rows = [row for row in target if row["artifact_complete"]]
    complete_paths = {row["artifact_path"] for row in complete_rows}
    supplied_paths = set(supplied)
    omitted = sorted(str(path) for path in complete_paths - supplied_paths)
    unknown = sorted(str(path) for path in supplied_paths - complete_paths)
    if omitted or unknown:
        raise AttemptSelectionError(
            f"supplied artifact closure differs: omitted={omitted}, unknown={unknown}")
    for row in target:
        if not row["artifact_complete"] and row["artifact_path"].is_file():
            raise AttemptSelectionError(
                "ledger marks an existing artifact as incomplete")

    artifacts_by_path = {path: merger.load_json(path) for path in supplied}
    validated_by_path: dict[Path, dict[str, Any]] = {}
    for row in complete_rows:
        actual_hash = merger.sha256_path(row["artifact_path"])
        if not hmac.compare_digest(actual_hash, row["artifact_sha256"]):
            raise AttemptSelectionError(
                f"artifact hash differs for {row['artifact_path']}")
        item = merger.validate_campaign_artifact(
            artifacts_by_path[row["artifact_path"]],
            expected_hashes=hashes, contract=contract,
            label=f"attempt {row['attempt_id']} seed {row['seed']}")
        if item["seed"] != row["seed"] or not _execution_matches(row, item):
            raise AttemptSelectionError(
                f"artifact execution identity differs for {row['artifact_path']}")
        validated_by_path[row["artifact_path"]] = item

    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in complete_rows:
        by_seed[row["seed"]].append(row)
    selected_rows: list[dict[str, Any]] = []
    for seed in canonical_seeds:
        candidates = sorted(
            by_seed.get(seed, []),
            key=lambda row: (row["submitted_utc"], row["attempt_id"]))
        if not candidates:
            raise AttemptSelectionError(
                f"seed {seed} has no complete hash-valid attempt")
        earliest_time = candidates[0]["submitted_utc"]
        if sum(row["submitted_utc"] == earliest_time for row in candidates) > 1:
            raise AttemptSelectionError(
                f"seed {seed} has an ambiguous earliest submission time")
        selected_rows.append(candidates[0])

    # Reuse the merger as the final structural and cross-seed isolation check.
    selected_artifacts = [
        artifacts_by_path[row["artifact_path"]] for row in selected_rows]
    merger._merge_campaign_artifacts_for_preflight(
        selected_artifacts, protocol=protocol, campaign_cell=campaign_cell,
        expected_seeds=canonical_seeds,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_split_sha256=expected_split_sha256,
        expected_prereg_sha256=expected_prereg_sha256,
        expected_analyzer_sha256=expected_analyzer_sha256,
        expected_protocol_sha256=expected_protocol_sha256,
        expected_container_sha256=expected_container_sha256,
        expected_source_sha256=expected_source_sha256,
    )

    selected_paths = {row["artifact_path"] for row in selected_rows}
    excluded: list[dict[str, Any]] = []
    for row in sorted(target, key=lambda item: (
            item["seed"], item["submitted_utc"], item["attempt_id"])):
        if row["artifact_path"] in selected_paths:
            continue
        excluded.append({
            "seed": row["seed"],
            "attempt_id": row["attempt_id"],
            "submitted_utc": row["submitted_utc"],
            "artifact_path": str(row["artifact_path"]),
            "artifact_complete": row["artifact_complete"],
            "reason": ("later_complete_attempt" if row["artifact_complete"]
                       else "incomplete_no_artifact"),
        })
    return {
        "schema_version": 1,
        "selection_rule": merger.SELECTION_RULE,
        "outcome_blind": True,
        "campaign_id": campaign_id,
        "campaign_cell": campaign_cell,
        "expected_seed_list": list(canonical_seeds),
        "expected_hashes": copy.deepcopy(hashes),
        "ledger_sha256": normalized_ledger_sha256,
        "selected": [{
            "seed": row["seed"],
            "attempt_id": row["attempt_id"],
            "submitted_utc": row["submitted_utc"],
            "slurm_array_job_id": row["slurm_array_job_id"],
            "slurm_array_task_id": row["slurm_array_task_id"],
            "slurm_job_id": validated_by_path[
                row["artifact_path"]]["execution"]["slurm_job_id"],
            "artifact_path": str(row["artifact_path"]),
            "artifact_sha256": row["artifact_sha256"],
        } for row in selected_rows],
        "excluded": excluded,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Outcome-blind BARN retry selection from submission ledger")
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--campaign-cell", choices=PRODUCTION_CAMPAIGN_CELLS,
                        required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--expected-seeds", "--expected-seed-list", dest="expected_seeds",
        required=True)
    parser.add_argument("--output", type=Path, required=True)
    merger.add_hash_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        expected_seeds = merger.parse_expected_seeds(args.expected_seeds)
    except merger.MergeValidationError as error:
        parser.error(str(error))
    expected_protocol = merger._require_sha256(
        args.expected_protocol_sha256, field="expected protocol_sha256")
    if not hmac.compare_digest(
            merger.sha256_path(args.protocol), expected_protocol):
        raise AttemptSelectionError("machine protocol file hash differs")
    ledger_sha256 = merger.sha256_path(args.ledger)
    ledger = merger.load_json(args.ledger)
    cli_hashes = {
        field: getattr(args, f"expected_{field}")
        for field in merger.PROVENANCE_HASH_FIELDS}
    validate_finalized_campaign_closure(
        ledger=ledger, ledger_base=args.ledger.resolve().parent,
        ledger_path=args.ledger, ledger_sha256=ledger_sha256,
        campaign_id=args.campaign_id, expected_seeds=expected_seeds,
        expected_hashes=cli_hashes)
    receipt = select_attempts(
        ledger=ledger, ledger_base=args.ledger.resolve().parent,
        artifact_paths=args.artifacts, campaign_id=args.campaign_id,
        campaign_cell=args.campaign_cell,
        protocol=merger.load_json(args.protocol), expected_seeds=expected_seeds,
        ledger_sha256=ledger_sha256,
        **merger.hash_arguments(args))
    protected = {path.resolve() for path in args.artifacts}
    protected.update({args.ledger.resolve(), args.protocol.resolve()})
    if args.output.resolve() in protected:
        parser.error("output path must not overwrite an input")
    merger.atomic_write_json(args.output, receipt)
    print(
        f"wrote {args.output}: campaign={args.campaign_id} "
        f"cell={args.campaign_cell} selected={len(receipt['selected'])} "
        f"excluded={len(receipt['excluded'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
