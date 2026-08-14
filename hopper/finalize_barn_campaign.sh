#!/usr/bin/env bash
# Outcome-blind all-cell BARN postprocessing transaction.
#
# Usage:
#   finalize_barn_campaign.sh CAMPAIGN_ID SOURCE_BUNDLE SOURCE_SHA LEDGER_SHA
#
# The finalized ledger must cover primary plus fresh N={2,4,16}. The remote
# transaction runs four selectors, four receipt-bound mergers, and both frozen
# analyses before atomically publishing or fetching anything.
set -euo pipefail
umask 077

REMOTE_MODE=false
if [[ ${1:-} == --remote ]]; then
  REMOTE_MODE=true
  shift
fi

if [[ "$REMOTE_MODE" == true ]]; then
  if (( $# != 7 )); then
    echo "internal remote campaign finalizer argument mismatch" >&2
    exit 2
  fi
  readonly CAMPAIGN_ID=$1
  readonly SOURCE_BUNDLE=$2
  readonly SOURCE_SHA256=$3
  readonly LEDGER_SHA256=$4
  readonly WRAPPER_SHA256=$5
  readonly SCRATCH=$6
  readonly APPTAINER_BIN=$7

  python3 - "$CAMPAIGN_ID" "$SOURCE_BUNDLE" "$SOURCE_SHA256" \
    "$LEDGER_SHA256" "$WRAPPER_SHA256" "$SCRATCH" "$APPTAINER_BIN" <<'PY'
from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile


class Refusal(RuntimeError):
    pass


def safe_excepthook(error_type, error, traceback) -> None:
    del traceback
    if issubclass(error_type, Refusal):
        print(f"BARN campaign sealing refused: {error}", file=sys.stderr)
    else:
        print(f"BARN campaign sealing failed closed: {error_type.__name__}",
              file=sys.stderr)


sys.excepthook = safe_excepthook


CELLS = ("primary", "ablation_n2", "ablation_n4", "ablation_n16")
HASH_FIELDS = (
    "manifest_sha256", "split_sha256", "prereg_sha256",
    "analyzer_sha256", "protocol_sha256", "container_sha256",
    "source_sha256",
)
LEDGER_FIELDS = {
    "campaign_id", "campaign_cell", "attempt_id", "seed", "submitted_utc",
    "slurm_job_id", "slurm_array_job_id", "slurm_array_task_id",
    "artifact_path", "artifact_complete", "artifact_sha256",
    "expected_hashes",
}
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
SAFE_SHA = re.compile(r"[0-9a-f]{64}\Z")
SAFE_JOB = re.compile(r"[0-9]+\Z")
MANIFEST_LINE = re.compile(r"([0-9A-Fa-f]{64}) ([ *])(.+)\Z")


def refuse(condition: bool, message: str) -> None:
    if not condition:
        raise Refusal(message)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                Refusal(f"non-finite JSON constant {token} is forbidden")),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Refusal(f"invalid JSON syntax in {path.name}") from error
    refuse(isinstance(value, dict), f"{path.name} must contain an object")
    return value


def require_canonical_directory(path: Path, *, label: str) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError as error:
        raise Refusal(f"{label} is missing") from error
    refuse(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
           f"{label} is non-directory or symbolic")
    refuse(path.resolve(strict=True) == path,
           f"{label} has a symbolic or non-canonical ancestor")


def require_regular(path: Path, *, label: str) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError as error:
        raise Refusal(f"{label} is missing") from error
    refuse(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode),
           f"{label} is non-regular or symbolic")
    refuse(path.resolve(strict=True) == path,
           f"{label} has a symbolic or non-canonical ancestor")


def verify_manifest(root: Path, *, expected_digest: str | None = None,
                    require_complete: bool = False) -> tuple[str, int]:
    require_canonical_directory(root, label="checksum package")
    manifest = root / "SHA256SUMS"
    require_regular(manifest, label="SHA256SUMS")
    digest = sha256_path(manifest)
    if expected_digest is not None:
        refuse(digest == expected_digest, "SHA256SUMS digest mismatch")
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = MANIFEST_LINE.fullmatch(line)
        refuse(match is not None, "unsafe SHA256SUMS format")
        name = PurePosixPath(match.group(3))
        refuse(not name.is_absolute() and ".." not in name.parts,
               "unsafe SHA256SUMS path")
        normalized = name.as_posix()
        while normalized.startswith("./"):
            normalized = normalized[2:]
        refuse(bool(normalized) and normalized not in listed,
               "duplicate or empty SHA256SUMS path")
        refuse(normalized not in {"SHA256SUMS", "COMPLETE"},
               "SHA256SUMS lists a finalization control file")
        require_regular(root / normalized, label="manifest-listed file")
        listed.add(normalized)
    actual = set()
    for candidate in root.rglob("*"):
        refuse(not candidate.is_symlink(), "checksum package contains a symlink")
        if candidate.is_file():
            relative = candidate.relative_to(root).as_posix()
            if relative not in {"SHA256SUMS", "COMPLETE"}:
                actual.add(relative)
        else:
            refuse(candidate.is_dir(), "checksum package contains a special file")
    refuse(actual == listed, "SHA256SUMS does not close over package files")
    check = subprocess.run(
        ["sha256sum", "-c", "--strict", "--quiet", "SHA256SUMS"],
        cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    refuse(check.returncode == 0, "strict SHA256SUMS verification failed")
    if require_complete:
        require_regular(root / "COMPLETE", label="COMPLETE")
    return digest, len(listed)


def parse_complete(path: Path) -> dict[str, str]:
    require_regular(path, label="COMPLETE")
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        refuse(len(fields) == 2 and fields[0] and fields[0] not in rows,
               "malformed COMPLETE metadata")
        rows[fields[0]] = fields[1]
    return rows


def atomic_publish_directory(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    refuse(renameat2 is not None,
           "renameat2 unavailable; refusing non-atomic publish")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p,
                          ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100,
                       os.fsencode(destination), 1)
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise Refusal("sealed campaign destination appeared concurrently")
        raise OSError(error, os.strerror(error), destination)


def run_tool(label: str, script: Path, arguments: list[str]) -> None:
    python_path = f"{source_bundle}:{source_bundle / 'curriculum_maxrl'}"
    command = [
        str(apptainer), "exec", "--cleanenv", "--bind",
        f"{scratch}:{scratch}", str(sif), "env", "CUDA_VISIBLE_DEVICES=",
        "PYTHONDONTWRITEBYTECODE=1", f"PYTHONPATH={python_path}",
        "python3", str(script), *arguments,
    ]
    result = subprocess.run(
        command, cwd=source_bundle, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    refuse(result.returncode == 0, f"frozen {label} failed closed")


def existing_transaction(output_root: Path, *, ledger_sha: str,
                         source_sha: str, wrapper_sha: str):
    matches = []
    if not output_root.exists():
        return None
    require_canonical_directory(output_root, label="sealed campaign root")
    for package in output_root.glob("campaign-*"):
        require_canonical_directory(package, label="sealed campaign package")
        complete = parse_complete(package / "COMPLETE")
        if (complete.get("ledger_sha256") == ledger_sha
                and complete.get("source_sha256") == source_sha
                and complete.get("wrapper_sha256") == wrapper_sha):
            expected = complete.get("sha256sums_sha256")
            refuse(isinstance(expected, str)
                   and SAFE_SHA.fullmatch(expected) is not None,
                   "existing COMPLETE has invalid manifest digest")
            digest, count = verify_manifest(
                package, expected_digest=expected, require_complete=True)
            refuse(package.name == f"campaign-{digest}",
                   "existing campaign package is not content addressed")
            matches.append((package, digest, count))
    refuse(len(matches) <= 1,
           "multiple sealed packages bind the same finalized ledger")
    return matches[0] if matches else None


campaign_id, source_arg, source_sha, ledger_sha, wrapper_sha, scratch_arg, apptainer_arg = sys.argv[1:]
refuse(SAFE_ID.fullmatch(campaign_id) is not None, "unsafe campaign ID")
for value, label in ((source_sha, "source"), (ledger_sha, "ledger"),
                     (wrapper_sha, "wrapper")):
    refuse(SAFE_SHA.fullmatch(value) is not None, f"invalid {label} SHA-256")
source_bundle = Path(source_arg)
scratch = Path(scratch_arg)
apptainer = Path(apptainer_arg)
refuse(source_bundle.is_absolute() and scratch.is_absolute(),
       "source and scratch paths must be absolute")
refuse(source_bundle == scratch / "maxrl" / "bundles" / "barn_source"
       / source_sha[:20], "source bundle path is not content addressed")
require_canonical_directory(source_bundle, label="source bundle")
require_regular(apptainer, label="pinned Apptainer")
source_manifest = source_bundle / "SHA256SUMS"
verify_manifest(source_bundle, expected_digest=source_sha)
state = load_json(source_bundle / "SOURCE_STATE.json")
refuse(state.get("mode") == "evidence"
       and state.get("worktree_dirty") is False
       and state.get("relevant_paths_match_head") is True,
       "source bundle is not frozen evidence source")
state_files = state.get("files")
refuse(isinstance(state_files, list) and state_files,
       "source state lacks its explicit file closure")
state_paths = set()
for index, row in enumerate(state_files):
    refuse(isinstance(row, dict)
           and isinstance(row.get("path"), str)
           and isinstance(row.get("worktree_mode"), str)
           and isinstance(row.get("worktree_sha256"), str),
           f"source state row {index} is malformed")
    relative = PurePosixPath(row["path"])
    refuse(not relative.is_absolute() and ".." not in relative.parts
           and relative.as_posix() not in state_paths,
           "source state contains an unsafe or duplicate path")
    state_paths.add(relative.as_posix())
    path = source_bundle / relative.as_posix()
    require_regular(path, label="source-state file")
    mode = format(stat.S_IMODE(os.lstat(path).st_mode), "o")
    refuse(mode == row["worktree_mode"]
           and sha256_path(path) == row["worktree_sha256"],
           "source-state file hash/mode evidence mismatch")
wrapper_path = source_bundle / "hopper" / "finalize_barn_campaign.sh"
require_regular(wrapper_path, label="bundled campaign finalizer")
refuse(sha256_path(wrapper_path) == wrapper_sha,
       "local and bundled campaign finalizer differ")

sif = scratch / "ros2-gazebo-classic.sif"
require_regular(sif, label="pinned CPU container")
campaign_root = scratch / "maxrl" / "barn" / "campaigns" / campaign_id
require_canonical_directory(campaign_root, label="campaign root")
ledger_path = (campaign_root / "finalized_ledgers"
               / f"SUBMISSION_LEDGER.finalized-{ledger_sha}.json")
require_regular(ledger_path, label="finalized campaign ledger")
refuse(sha256_path(ledger_path) == ledger_sha,
       "finalized campaign ledger digest mismatch")
ledger = load_json(ledger_path)
refuse(set(ledger) == {"schema_version", "submissions"}
       and ledger.get("schema_version") == 1
       and isinstance(ledger.get("submissions"), list)
       and ledger["submissions"],
       "finalized campaign ledger schema mismatch")

expected_hashes = None
rows_by_cell = {cell: [] for cell in CELLS}
identities = set()
for index, row in enumerate(ledger["submissions"]):
    label = f"finalized ledger row {index}"
    refuse(isinstance(row, dict) and set(row) == LEDGER_FIELDS,
           f"{label} fields are not exact")
    refuse(row["campaign_id"] == campaign_id and row["campaign_cell"] in CELLS,
           f"{label} campaign identity mismatch")
    refuse(isinstance(row["seed"], int) and not isinstance(row["seed"], bool)
           and row["seed"] in range(1, 6)
           and row["slurm_array_task_id"] == row["seed"],
           f"{label} seed/task mismatch")
    identity = (row["campaign_cell"], row["attempt_id"], row["seed"])
    refuse(identity not in identities, "duplicate campaign/cell/attempt/seed row")
    identities.add(identity)
    hashes = row["expected_hashes"]
    refuse(isinstance(hashes, dict) and set(hashes) == set(HASH_FIELDS)
           and all(isinstance(hashes[field], str)
                   and SAFE_SHA.fullmatch(hashes[field]) is not None
                   for field in HASH_FIELDS),
           f"{label} expected hashes are invalid")
    if expected_hashes is None:
        expected_hashes = hashes
    else:
        refuse(hashes == expected_hashes,
               "expected hashes drift across finalized ledger")
    artifact = Path(row["artifact_path"])
    expected_artifact = (
        campaign_root / "cells" / row["campaign_cell"] / "attempts"
        / row["attempt_id"] / f"seed-{row['seed']}" / "results"
        / f"seed-{row['seed']}.json")
    refuse(artifact == expected_artifact and artifact.is_absolute(),
           f"{label} artifact path mismatch")
    refuse(isinstance(row["artifact_complete"], bool),
           f"{label} completeness is invalid")
    if row["artifact_complete"]:
        require_regular(artifact, label="complete evidence artifact")
        refuse(isinstance(row["artifact_sha256"], str)
               and sha256_path(artifact) == row["artifact_sha256"],
               f"{label} artifact digest mismatch")
        refuse(isinstance(row["slurm_job_id"], str)
               and SAFE_JOB.fullmatch(row["slurm_job_id"]) is not None,
               f"{label} element job identity missing")
    else:
        refuse(row["artifact_sha256"] is None and not artifact.exists()
               and not artifact.is_symlink(),
               f"{label} incomplete row exposes an artifact")
    rows_by_cell[row["campaign_cell"]].append(row)

refuse(set(rows_by_cell) == set(CELLS)
       and all({row["seed"] for row in rows_by_cell[cell]} == set(range(1, 6))
               for cell in CELLS),
       "finalized ledger lacks exact four-cell seed coverage")
refuse(expected_hashes is not None
       and expected_hashes["source_sha256"] == source_sha,
       "source bundle differs from finalized ledger")
refuse(sha256_path(sif) == expected_hashes["container_sha256"],
       "container differs from finalized ledger")

selector = source_bundle / "icra2027" / "select_barn_attempts.py"
merger = source_bundle / "icra2027" / "merge_barn_campaign.py"
analyzer = source_bundle / "icra2027" / "analyze_campaign.py"
protocol = source_bundle / "icra2027" / "barn_protocol.json"
for path, label in ((selector, "selector"), (merger, "merger"),
                    (analyzer, "analyzer"), (protocol, "protocol")):
    require_regular(path, label=f"frozen {label}")
refuse(sha256_path(analyzer) == expected_hashes["analyzer_sha256"],
       "frozen analyzer digest mismatch")
refuse(sha256_path(protocol) == expected_hashes["protocol_sha256"],
       "frozen protocol digest mismatch")

output_root = campaign_root / "sealed_campaigns"
output_root.mkdir(mode=0o750, parents=True, exist_ok=True)
require_canonical_directory(output_root, label="sealed campaign root")
existing = existing_transaction(
    output_root, ledger_sha=ledger_sha, source_sha=source_sha,
    wrapper_sha=wrapper_sha)
if existing is not None:
    package, digest, count = existing
    print(f"SEALED_CAMPAIGN_PATH={package}")
    print(f"SEALED_CAMPAIGN_SHA256SUMS_SHA256={digest}")
    print(f"SEALED_CAMPAIGN_FILE_COUNT={count}")
    raise SystemExit(0)

lock_root = campaign_root / ".postprocess_locks"
lock_root.mkdir(mode=0o750, exist_ok=True)
require_canonical_directory(lock_root, label="postprocess lock root")
lock = lock_root / f"ledger-{ledger_sha}-source-{source_sha}"
try:
    lock.mkdir(mode=0o700)
except FileExistsError as error:
    raise Refusal("campaign postprocessing transaction is already active") from error

stage = None
try:
    existing = existing_transaction(
        output_root, ledger_sha=ledger_sha, source_sha=source_sha,
        wrapper_sha=wrapper_sha)
    if existing is not None:
        package, digest, count = existing
        print(f"SEALED_CAMPAIGN_PATH={package}")
        print(f"SEALED_CAMPAIGN_SHA256SUMS_SHA256={digest}")
        print(f"SEALED_CAMPAIGN_FILE_COUNT={count}")
        raise SystemExit(0)

    stage = Path(tempfile.mkdtemp(prefix=".campaign-stage-", dir=output_root))
    selection_dir = stage / "selection"
    merged_dir = stage / "merged"
    reports_dir = stage / "reports"
    selection_dir.mkdir()
    merged_dir.mkdir()
    reports_dir.mkdir()
    hash_arguments = []
    for field in HASH_FIELDS:
        hash_arguments.extend([
            "--expected-" + field.replace("_", "-"),
            expected_hashes[field],
        ])

    selected_by_cell = {}
    receipt_paths = {}
    merged_paths = {}
    for cell in CELLS:
        complete_paths = [
            Path(row["artifact_path"]) for row in rows_by_cell[cell]
            if row["artifact_complete"]]
        refuse(bool(complete_paths), f"{cell} has no complete attempts")
        receipt = selection_dir / f"{cell}.json"
        run_tool(
            f"selector for {cell}", selector,
            [*(str(path) for path in complete_paths),
             "--ledger", str(ledger_path), "--campaign-id", campaign_id,
             "--campaign-cell", cell, "--protocol", str(protocol),
             "--expected-seeds", "1,2,3,4,5", "--output", str(receipt),
             *hash_arguments])
        receipt_document = load_json(receipt)
        selected = receipt_document.get("selected")
        refuse(receipt_document.get("outcome_blind") is True
               and receipt_document.get("campaign_id") == campaign_id
               and receipt_document.get("campaign_cell") == cell
               and receipt_document.get("ledger_sha256") == ledger_sha
               and isinstance(selected, list) and len(selected) == 5
               and [row.get("seed") for row in selected] == [1, 2, 3, 4, 5],
               f"selector receipt shape mismatch for {cell}")
        selected_paths = [Path(row["artifact_path"]) for row in selected]
        refuse(all(path.is_absolute() and path.resolve(strict=True) == path
                   for path in selected_paths),
               f"selector returned a non-canonical path for {cell}")
        selected_by_cell[cell] = selected_paths
        receipt_paths[cell] = receipt

        merged = merged_dir / f"{cell}.json"
        run_tool(
            f"selection-bound merger for {cell}", merger,
            [*(str(path) for path in selected_paths),
             "--selection-receipt", str(receipt),
             "--protocol", str(protocol), "--campaign-cell", cell,
             "--expected-seeds", "1,2,3,4,5", "--output", str(merged),
             *hash_arguments])
        merged_document = load_json(merged)
        selection_binding = merged_document.get("merge", {}).get("selection", {})
        refuse(selection_binding.get("selection_receipt_sha256")
               == sha256_path(receipt)
               and selection_binding.get("ledger_sha256") == ledger_sha
               and selection_binding.get("campaign_id") == campaign_id
               and selection_binding.get("campaign_cell") == cell,
               f"merged selection binding mismatch for {cell}")
        merged_paths[cell] = merged

    primary_report = reports_dir / "primary_gate.json"
    run_tool(
        "primary frozen analyzer", analyzer,
        [str(merged_paths["primary"]), "--output", str(primary_report)])
    ablation_report = reports_dir / "n_ablation.json"
    run_tool(
        "N-ablation frozen analyzer", analyzer,
        [str(merged_paths["primary"]),
         "--ablation-artifact", str(merged_paths["ablation_n2"]),
         "--ablation-artifact", str(merged_paths["ablation_n4"]),
         "--ablation-artifact", str(merged_paths["ablation_n16"]),
         "--output", str(ablation_report)])

    primary_document = load_json(primary_report)
    refuse(primary_document.get("input_artifact_sha256")
           == sha256_path(merged_paths["primary"])
           and primary_document.get("analyzer_sha256")
           == expected_hashes["analyzer_sha256"]
           and primary_document.get("protocol_sha256")
           == expected_hashes["protocol_sha256"],
           "primary analysis input binding mismatch")
    ablation_document = load_json(ablation_report)
    cells = ablation_document.get("cells")
    refuse(ablation_document.get("analysis_kind") == "barn_n_ablation"
           and ablation_document.get("analyzer_sha256")
           == expected_hashes["analyzer_sha256"]
           and ablation_document.get("protocol_sha256")
           == expected_hashes["protocol_sha256"]
           and isinstance(cells, list) and len(cells) == 4,
           "N-ablation analysis shape/binding mismatch")
    input_hash_by_n = {
        row.get("n_rollouts"): row.get("input_artifact", {}).get("sha256")
        for row in cells if isinstance(row, dict)}
    refuse(input_hash_by_n == {
        2: sha256_path(merged_paths["ablation_n2"]),
        4: sha256_path(merged_paths["ablation_n4"]),
        8: sha256_path(merged_paths["primary"]),
        16: sha256_path(merged_paths["ablation_n16"]),
    }, "N-ablation report input binding mismatch")

    metadata = {
        "schema_version": 1,
        "artifact_type": "barn_four_cell_selection_merge_analysis_package",
        "outcome_blind_pipeline": True,
        "campaign_id": campaign_id,
        "campaign_cells": list(CELLS),
        "expected_seed_list": [1, 2, 3, 4, 5],
        "finalized_ledger_sha256": ledger_sha,
        "source_sha256": source_sha,
        "wrapper_sha256": wrapper_sha,
        "expected_hashes": expected_hashes,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cell_artifacts": {
            cell: {
                "selection_receipt": f"selection/{cell}.json",
                "selection_receipt_sha256": sha256_path(receipt_paths[cell]),
                "merged_artifact": f"merged/{cell}.json",
                "merged_artifact_sha256": sha256_path(merged_paths[cell]),
            }
            for cell in CELLS
        },
        "analysis_reports": {
            "primary_gate": {
                "path": "reports/primary_gate.json",
                "sha256": sha256_path(primary_report),
            },
            "n_ablation": {
                "path": "reports/n_ablation.json",
                "sha256": sha256_path(ablation_report),
            },
        },
    }
    (stage / "PACKAGE_METADATA.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    files = sorted(
        path for path in stage.rglob("*") if path.is_file()
        and path.name not in {"SHA256SUMS", "COMPLETE"})
    with (stage / "SHA256SUMS").open("w", encoding="utf-8") as stream:
        for path in files:
            stream.write(
                f"{sha256_path(path)}  ./{path.relative_to(stage).as_posix()}\n")
    manifest_sha, file_count = verify_manifest(stage)
    complete_rows = (
        ("artifact_type", "barn_four_cell_campaign_complete"),
        ("campaign_id", campaign_id),
        ("campaign_cells", ",".join(CELLS)),
        ("ledger_sha256", ledger_sha),
        ("source_sha256", source_sha),
        ("wrapper_sha256", wrapper_sha),
        ("sha256sums_sha256", manifest_sha),
        ("completed_utc", datetime.now(timezone.utc).isoformat()),
    )
    (stage / "COMPLETE").write_text(
        "".join(f"{key}\t{value}\n" for key, value in complete_rows),
        encoding="utf-8")
    for path in stage.rglob("*"):
        if path.is_file():
            path.chmod(0o440)
    for path in sorted(
            (path for path in stage.rglob("*") if path.is_dir()), reverse=True):
        path.chmod(0o550)
    stage.chmod(0o550)
    destination = output_root / f"campaign-{manifest_sha}"
    atomic_publish_directory(stage, destination)
    stage = None
    print(f"SEALED_CAMPAIGN_PATH={destination}")
    print(f"SEALED_CAMPAIGN_SHA256SUMS_SHA256={manifest_sha}")
    print(f"SEALED_CAMPAIGN_FILE_COUNT={file_count}")
finally:
    if stage is not None and stage.exists():
        for path in stage.rglob("*"):
            if path.is_dir():
                path.chmod(0o700)
            elif path.is_file():
                path.chmod(0o600)
        stage.chmod(0o700)
        shutil.rmtree(stage)
    lock.rmdir()
PY
  exit
fi

if (( $# != 4 )); then
  echo "usage: $0 CAMPAIGN_ID SOURCE_BUNDLE SOURCE_SHA LEDGER_SHA" >&2
  exit 2
fi
readonly CAMPAIGN_ID=$1
readonly SOURCE_BUNDLE=$2
readonly SOURCE_SHA256=$3
readonly LEDGER_SHA256=$4
readonly HOST="${HOPPER_HOST:-lwang44@hopper.orc.gmu.edu}"
readonly SCRATCH="${HOPPER_SCRATCH:-/scratch/lwang44}"
readonly HERE="$(cd "$(dirname "$0")" && pwd)"
readonly ROOT="$(cd "$HERE/.." && pwd)"
readonly LOCAL_PACKAGE_DIR="${FINALIZE_BARN_LOCAL_PACKAGE_DIR:-$ROOT/autoresearch/iterate-260814-0047/sealed_campaigns}"
readonly WRAPPER_SHA256="$(sha256sum -- "$0" | awk '{print $1}')"
readonly FINALIZE_SBATCH="$HERE/sbatch/barn_finalize_cpu.sbatch"
[[ -f "$FINALIZE_SBATCH" && ! -L "$FINALIZE_SBATCH" ]] || {
  echo "local finalize sbatch is missing or symbolic" >&2
  exit 2
}
readonly FINALIZE_SBATCH_SHA256="$(sha256sum -- "$FINALIZE_SBATCH" | awk '{print $1}')"
readonly POLL_INTERVAL="${FINALIZE_BARN_POLL_INTERVAL:-30}"
readonly WAIT_SECONDS="${FINALIZE_BARN_WAIT_SECONDS:-3600}"

[[ "$CAMPAIGN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]] || {
  echo "unsafe CAMPAIGN_ID" >&2; exit 2;
}
[[ "$HOST" =~ ^([A-Za-z0-9._-]+@)?[A-Za-z0-9._-]+$ ]] || {
  echo "unsafe HOPPER_HOST" >&2; exit 2;
}
[[ "$SCRATCH" =~ ^/scratch/[A-Za-z0-9._-]+$ ]] || {
  echo "HOPPER_SCRATCH must be canonical below /scratch" >&2; exit 2;
}
[[ "$SOURCE_BUNDLE" =~ ^/scratch/[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ ]] || {
  echo "source bundle must be a canonical path below HOPPER_SCRATCH" >&2; exit 2;
}
for digest in "$SOURCE_SHA256" "$LEDGER_SHA256" "$WRAPPER_SHA256" \
              "$FINALIZE_SBATCH_SHA256"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || {
    echo "invalid SHA-256 argument" >&2; exit 2;
  }
done
[[ "$SOURCE_BUNDLE" == "$SCRATCH/maxrl/bundles/barn_source/${SOURCE_SHA256:0:20}" ]] || {
  echo "source bundle path is not content addressed" >&2; exit 2;
}
[[ "$POLL_INTERVAL" =~ ^[1-9][0-9]*$ && "$WAIT_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "finalizer polling limits must be positive integers" >&2; exit 2;
}

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 \
          -o ServerAliveCountMax=3)
readonly RSYNC_RSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3"

# Verify the complete immutable source closure and both executing local files
# before submitting the bundled CPU-only postprocessing job directly to Slurm.
if ! submit_output=$(ssh "${SSH_OPTS[@]}" "$HOST" bash -s -- \
  "$CAMPAIGN_ID" "$SOURCE_BUNDLE" "$SOURCE_SHA256" "$LEDGER_SHA256" \
  "$WRAPPER_SHA256" "$FINALIZE_SBATCH_SHA256" "$SCRATCH" <<'REMOTE'
set -euo pipefail
umask 077
campaign=$1
source_bundle=$2
source_sha=$3
ledger_sha=$4
wrapper_sha=$5
sbatch_sha=$6
scratch=$7
[[ -d "$source_bundle" && ! -L "$source_bundle" \
   && "$(readlink -f -- "$source_bundle")" == "$source_bundle" ]] || exit 2
[[ -f "$source_bundle/SHA256SUMS" \
   && ! -L "$source_bundle/SHA256SUMS" ]] || exit 2
[[ "$(sha256sum -- "$source_bundle/SHA256SUMS" | awk '{print $1}')" \
   == "$source_sha" ]] || exit 2
(cd "$source_bundle" && sha256sum -c --strict --quiet SHA256SUMS)
python3 - "$source_bundle/SOURCE_STATE.json" <<'PY'
import json
import sys
state = json.load(open(sys.argv[1], encoding="utf-8"))
if (state.get("mode") != "evidence"
        or state.get("worktree_dirty") is not False
        or state.get("relevant_paths_match_head") is not True):
    raise SystemExit("source bundle is not a frozen evidence closure")
PY
bundled_wrapper="$source_bundle/hopper/finalize_barn_campaign.sh"
bundled_sbatch="$source_bundle/hopper/sbatch/barn_finalize_cpu.sbatch"
for path in "$bundled_wrapper" "$bundled_sbatch"; do
  [[ -f "$path" && ! -L "$path" ]] || exit 2
done
[[ "$(sha256sum -- "$bundled_wrapper" | awk '{print $1}')" \
   == "$wrapper_sha" ]] || exit 2
[[ "$(sha256sum -- "$bundled_sbatch" | awk '{print $1}')" \
   == "$sbatch_sha" ]] || exit 2
mkdir -p -- "$scratch/maxrl/barn/logs"
record=$(sbatch --parsable \
  --export="ALL,BARN_CAMPAIGN_ID=$campaign,BARN_SOURCE_BUNDLE_DIR=$source_bundle,BARN_SOURCE_SHA256=$source_sha,BARN_FINALIZED_LEDGER_SHA256=$ledger_sha,BARN_FINALIZE_WRAPPER_SHA256=$wrapper_sha,BARN_FINALIZE_SBATCH_SHA256=$sbatch_sha,BARN_SCRATCH_ROOT=$scratch" \
  "$bundled_sbatch")
job_id=${record%%;*}
[[ "$job_id" =~ ^[0-9]+$ ]] || exit 2
printf 'FINALIZE_JOB_ID=%s\n' "$job_id"
REMOTE
); then
  echo "source-bound CPU finalizer preflight/submission failed closed" >&2
  exit 2
fi
JOB_ID=$(printf '%s\n' "$submit_output" \
  | awk -F= '$1 == "FINALIZE_JOB_ID" {print $2}')
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || {
  echo "postprocess submission succeeded but job ID could not be parsed" >&2
  exit 2
}
readonly JOB_ID

elapsed=0
while true; do
  accounting=$(ssh "${SSH_OPTS[@]}" "$HOST" bash -s -- "$JOB_ID" <<'REMOTE'
set -euo pipefail
sacct -j "$1" -X -n -P --format=JobID,State,ExitCode \
  | awk -F'|' -v wanted="$1" '$1 == wanted {print $2 "|" $3}'
REMOTE
)
  records=$(printf '%s\n' "$accounting" | awk 'NF {n++} END {print n+0}')
  (( records <= 1 )) || {
    echo "Slurm accounting returned duplicate finalizer rows" >&2; exit 2;
  }
  if (( records == 1 )); then
    state=${accounting%%|*}
    exit_code=${accounting#*|}
    state=${state%%+}
    case "$state" in
      BOOT_FAIL|CANCELLED|COMPLETED|DEADLINE|FAILED|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|REVOKED|SPECIAL_EXIT|TIMEOUT)
        [[ "$state" == COMPLETED && "$exit_code" == 0:0 ]] || {
          printf 'CPU finalizer job %s ended state=%s exit=%s\n' \
            "$JOB_ID" "$state" "$exit_code" >&2
          exit 2
        }
        break
        ;;
    esac
  fi
  (( elapsed < WAIT_SECONDS )) || {
    printf 'timed out waiting for CPU finalizer job %s\n' "$JOB_ID" >&2
    exit 2
  }
  sleep "$POLL_INTERVAL"
  elapsed=$((elapsed + POLL_INTERVAL))
done

readonly REMOTE_LOG="$SCRATCH/maxrl/barn/logs/barn-finalize-safe-log_${JOB_ID}.out"
remote_output=$(ssh "${SSH_OPTS[@]}" "$HOST" bash -s -- \
  "$REMOTE_LOG" <<'REMOTE'
set -euo pipefail
[[ -f "$1" && ! -L "$1" ]] || exit 2
python3 - "$1" <<'PY'
from pathlib import Path
import sys
allowed = {
    "SEALED_CAMPAIGN_PATH",
    "SEALED_CAMPAIGN_SHA256SUMS_SHA256",
    "SEALED_CAMPAIGN_FILE_COUNT",
}
found = {key: [] for key in allowed}
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    key, separator, value = line.partition("=")
    if separator and key in found:
        found[key].append(value)
for key in sorted(found):
    if len(found[key]) != 1:
        raise SystemExit("safe finalizer sentinel closure mismatch")
    print(f"{key}={found[key][0]}")
PY
REMOTE
)

sentinel_value() {
  local key=$1 count value
  count=$(printf '%s\n' "$remote_output" \
    | awk -F= -v wanted="$key" '$1 == wanted {n++} END {print n+0}')
  (( count == 1 )) || {
    printf 'remote campaign finalizer returned %s %s records\n' "$count" "$key" >&2
    exit 2
  }
  value=$(printf '%s\n' "$remote_output" \
    | awk -F= -v wanted="$key" '$1 == wanted {print substr($0, length($1) + 2)}')
  printf '%s\n' "$value"
}

readonly REMOTE_PACKAGE="$(sentinel_value SEALED_CAMPAIGN_PATH)"
readonly MANIFEST_SHA256="$(sentinel_value SEALED_CAMPAIGN_SHA256SUMS_SHA256)"
readonly FILE_COUNT="$(sentinel_value SEALED_CAMPAIGN_FILE_COUNT)"
readonly EXPECTED_REMOTE_PACKAGE="$SCRATCH/maxrl/barn/campaigns/$CAMPAIGN_ID/sealed_campaigns/campaign-$MANIFEST_SHA256"
[[ "$MANIFEST_SHA256" =~ ^[0-9a-f]{64}$
   && "$FILE_COUNT" =~ ^[0-9]+$
   && "$REMOTE_PACKAGE" == "$EXPECTED_REMOTE_PACKAGE" ]] || {
  echo "remote campaign finalizer returned invalid package metadata" >&2
  exit 2
}

mkdir -p -- "$LOCAL_PACKAGE_DIR"
[[ -d "$LOCAL_PACKAGE_DIR" && ! -L "$LOCAL_PACKAGE_DIR" ]] || {
  echo "local package directory must not be symbolic" >&2; exit 2;
}
readonly LOCAL_PACKAGE="$LOCAL_PACKAGE_DIR/$CAMPAIGN_ID.sealed-$MANIFEST_SHA256"
[[ ! -e "$LOCAL_PACKAGE" && ! -L "$LOCAL_PACKAGE" ]] || {
  printf 'refusing to overwrite prior sealed campaign: %s\n' "$LOCAL_PACKAGE" >&2
  exit 2
}
local_stage=$(mktemp -d "$LOCAL_PACKAGE_DIR/.$CAMPAIGN_ID.sealed.XXXXXX")
cleanup() {
  if [[ -n "${local_stage:-}" && -d "$local_stage" ]]; then
    chmod -R u+w "$local_stage" 2>/dev/null || true
    rm -rf -- "$local_stage"
  fi
}
trap cleanup EXIT
rsync -az --safe-links --protect-args -e "$RSYNC_RSH" \
  "$HOST:$REMOTE_PACKAGE/" "$local_stage/"

python3 - "$local_stage" "$CAMPAIGN_ID" "$LEDGER_SHA256" \
  "$SOURCE_SHA256" "$WRAPPER_SHA256" "$MANIFEST_SHA256" "$FILE_COUNT" <<'PY'
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

root = Path(sys.argv[1])
campaign, ledger_sha, source_sha, wrapper_sha, expected_manifest, count = sys.argv[2:]
manifest = root / "SHA256SUMS"
complete = root / "COMPLETE"
if (not manifest.is_file() or manifest.is_symlink()
        or not complete.is_file() or complete.is_symlink()):
    raise SystemExit("fetched package lacks regular sealing controls")
digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
if digest != expected_manifest:
    raise SystemExit("fetched SHA256SUMS digest mismatch")
pattern = re.compile(r"([0-9A-Fa-f]{64}) ([ *])(.+)\Z")
listed = set()
for line in manifest.read_text().splitlines():
    match = pattern.fullmatch(line)
    if match is None:
        raise SystemExit("unsafe fetched SHA256SUMS format")
    path = PurePosixPath(match.group(3))
    if path.is_absolute() or ".." in path.parts:
        raise SystemExit("unsafe fetched SHA256SUMS path")
    name = path.as_posix()
    while name.startswith("./"):
        name = name[2:]
    if not name or name in listed or name in {"SHA256SUMS", "COMPLETE"}:
        raise SystemExit("invalid fetched SHA256SUMS closure")
    target = root / name
    if not target.is_file() or target.is_symlink():
        raise SystemExit("fetched manifest target is missing or symbolic")
    listed.add(name)
actual = set()
for path in root.rglob("*"):
    if path.is_symlink():
        raise SystemExit("fetched package contains a symlink")
    if path.is_file() and path.name not in {"SHA256SUMS", "COMPLETE"}:
        actual.add(path.relative_to(root).as_posix())
    elif not path.is_file() and not path.is_dir():
        raise SystemExit("fetched package contains a special file")
if actual != listed or len(listed) != int(count):
    raise SystemExit("fetched package file closure mismatch")
check = subprocess.run(
    ["sha256sum", "-c", "--strict", "--quiet", "SHA256SUMS"],
    cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if check.returncode != 0:
    raise SystemExit("fetched package checksum verification failed")
rows = {}
for line in complete.read_text().splitlines():
    fields = line.split("\t")
    if len(fields) != 2 or fields[0] in rows:
        raise SystemExit("malformed fetched COMPLETE")
    rows[fields[0]] = fields[1]
expected = {
    "artifact_type": "barn_four_cell_campaign_complete",
    "campaign_id": campaign,
    "campaign_cells": "primary,ablation_n2,ablation_n4,ablation_n16",
    "ledger_sha256": ledger_sha,
    "source_sha256": source_sha,
    "wrapper_sha256": wrapper_sha,
    "sha256sums_sha256": expected_manifest,
}
if any(rows.get(field) != value for field, value in expected.items()):
    raise SystemExit("fetched COMPLETE identity/hash binding mismatch")
if set(rows) != set(expected) | {"completed_utc"}:
    raise SystemExit("fetched COMPLETE fields are not exact")
PY

python3 - "$local_stage" "$LOCAL_PACKAGE" <<'PY'
import ctypes
import errno
import os
import sys

source, destination = map(os.fsencode, sys.argv[1:3])
libc = ctypes.CDLL(None, use_errno=True)
renameat2 = getattr(libc, "renameat2", None)
if renameat2 is None:
    raise SystemExit("renameat2 unavailable; refusing non-atomic local publish")
renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p,
                      ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
renameat2.restype = ctypes.c_int
if renameat2(-100, source, -100, destination, 1) != 0:
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise SystemExit("sealed campaign destination appeared during fetch")
    raise OSError(error, os.strerror(error), os.fsdecode(destination))
PY
local_stage=""
trap - EXIT
printf 'BARN_CAMPAIGN_SEALED\tcampaign=%s\tcells=4\tseeds_per_cell=5\tfiles=%s\tsha256sums_sha256=%s\tremote=%s\tlocal=%s\n' \
  "$CAMPAIGN_ID" "$FILE_COUNT" "$MANIFEST_SHA256" "$REMOTE_PACKAGE" \
  "$LOCAL_PACKAGE"
