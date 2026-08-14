#!/usr/bin/env bash
# Finalize one BARN campaign submission ledger after every recorded Slurm
# array task is terminal. Scientific endpoints are never inspected or printed.
#
# Usage:
#   finalize_barn_ledger.sh CAMPAIGN_ID SOURCE_BUNDLE SOURCE_SHA
#
# The immutable pre-submission ledger is expected at
# autoresearch/iterate-260814-0047/submission_ledgers/CAMPAIGN_ID.json. A
# content-addressed finalized snapshot is published beside the remote campaign
# and fetched into that local directory without overwriting an earlier copy.
set -euo pipefail
umask 077

if (( $# != 3 )); then
  echo "usage: $0 CAMPAIGN_ID SOURCE_BUNDLE SOURCE_SHA" >&2
  exit 2
fi

readonly CAMPAIGN_ID=$1
readonly SOURCE_BUNDLE=$2
readonly SOURCE_SHA256=$3
[[ "$CAMPAIGN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]] || {
  echo "unsafe CAMPAIGN_ID" >&2
  exit 2
}

readonly HERE="$(cd "$(dirname "$0")" && pwd)"
readonly ROOT="$(cd "$HERE/.." && pwd)"
readonly HOST="${HOPPER_HOST:-lwang44@hopper.orc.gmu.edu}"
readonly SCRATCH="${HOPPER_SCRATCH:-/scratch/lwang44}"
readonly LOCAL_LEDGER_DIR="${FINALIZE_BARN_LOCAL_LEDGER_DIR:-$ROOT/autoresearch/iterate-260814-0047/submission_ledgers}"
readonly LOCAL_INPUT="$LOCAL_LEDGER_DIR/$CAMPAIGN_ID.json"
readonly REMOTE_CAMPAIGN_ROOT="$SCRATCH/maxrl/barn/campaigns/$CAMPAIGN_ID"
readonly REMOTE_INPUT="$REMOTE_CAMPAIGN_ROOT/SUBMISSION_LEDGER.json"
readonly FINALIZER_SHA256="$(sha256sum -- "$0" | awk '{print $1}')"

[[ "$HOST" =~ ^([A-Za-z0-9._-]+@)?[A-Za-z0-9._-]+$ ]] || {
  echo "unsafe HOPPER_HOST" >&2
  exit 2
}
[[ "$SCRATCH" =~ ^/scratch/[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ ]] || {
  echo "HOPPER_SCRATCH must be a canonical path below /scratch" >&2
  exit 2
}
[[ "$SOURCE_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$SOURCE_BUNDLE" == \
      "$SCRATCH/maxrl/bundles/barn_source/${SOURCE_SHA256:0:20}" ]] || {
  echo "source bundle is not content addressed by a valid SHA-256" >&2
  exit 2
}
[[ -f "$0" && ! -L "$0" ]] || {
  echo "executing ledger finalizer is missing or symbolic" >&2
  exit 2
}
[[ -f "$LOCAL_INPUT" && ! -L "$LOCAL_INPUT" ]] || {
  printf 'missing regular local submission ledger: %s\n' "$LOCAL_INPUT" >&2
  exit 2
}
mkdir -p -- "$LOCAL_LEDGER_DIR"
[[ -d "$LOCAL_LEDGER_DIR" && ! -L "$LOCAL_LEDGER_DIR" ]] || {
  echo "local ledger directory must not be symbolic" >&2
  exit 2
}

readonly INPUT_SHA256="$(sha256sum -- "$LOCAL_INPUT" | awk '{print $1}')"
[[ "$INPUT_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "could not hash local submission ledger" >&2
  exit 2
}

SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=15
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=3
)
readonly RSYNC_RSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3"

# The inline remote program below is source-bound by requiring this executing
# script to be byte-identical to the copy in the verified immutable bundle.
if ! ssh "${SSH_OPTS[@]}" "$HOST" bash -s -- \
  "$SOURCE_BUNDLE" "$SOURCE_SHA256" "$FINALIZER_SHA256" <<'REMOTE'
set -euo pipefail
bundle=$1
source_sha=$2
finalizer_sha=$3
[[ -d "$bundle" && ! -L "$bundle" \
   && "$(readlink -f -- "$bundle")" == "$bundle" ]] || exit 2
[[ -f "$bundle/SHA256SUMS" && ! -L "$bundle/SHA256SUMS" \
   && "$(sha256sum -- "$bundle/SHA256SUMS" | awk '{print $1}')" \
      == "$source_sha" ]] || exit 2
(cd "$bundle" && sha256sum -c --strict --quiet SHA256SUMS)
test -z "$(find "$bundle" -type l -print -quit)"
python3 - "$bundle/SOURCE_STATE.json" "$bundle" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
state_path, root = Path(sys.argv[1]), Path(sys.argv[2])
state = json.loads(state_path.read_text(encoding="utf-8"))
if (state.get("mode") != "evidence"
        or state.get("worktree_dirty") is not False
        or state.get("relevant_paths_match_head") is not True):
    raise SystemExit("source bundle is not a frozen evidence closure")
for row in state.get("files", []):
    path = root / row["path"]
    info = os.lstat(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    mode = format(stat.S_IMODE(info.st_mode), "o")
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or digest != row["worktree_sha256"]
            or mode != row["worktree_mode"]):
        raise SystemExit("source-state file evidence mismatch")
PY
bundled="$bundle/hopper/finalize_barn_ledger.sh"
[[ -f "$bundled" && ! -L "$bundled" \
   && "$(sha256sum -- "$bundled" | awk '{print $1}')" \
      == "$finalizer_sha" ]] || exit 2
REMOTE
then
  echo "source-bound ledger finalizer preflight failed closed" >&2
  exit 2
fi

# This remote program reads only ledger fields and the identity, provenance,
# and structural config/result keys of complete artifacts. It captures all
# checksum and scheduler command output and emits completion metadata only.
remote_output=$(ssh "${SSH_OPTS[@]}" "$HOST" bash -s -- \
  "$REMOTE_INPUT" "$REMOTE_CAMPAIGN_ROOT" "$CAMPAIGN_ID" \
  "$INPUT_SHA256" "$SOURCE_SHA256" <<'REMOTE'
set -euo pipefail
umask 077
python3 - "$1" "$2" "$3" "$4" "$5" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone


class Refusal(RuntimeError):
    pass


LEDGER_FIELDS = {
    "campaign_id",
    "campaign_cell",
    "attempt_id",
    "seed",
    "submitted_utc",
    "slurm_job_id",
    "slurm_array_job_id",
    "slurm_array_task_id",
    "artifact_path",
    "artifact_complete",
    "artifact_sha256",
    "expected_hashes",
}
HASH_FIELDS = (
    "manifest_sha256",
    "split_sha256",
    "prereg_sha256",
    "analyzer_sha256",
    "protocol_sha256",
    "container_sha256",
    "source_sha256",
)
CELLS = {
    "primary": (8, ("ours_uN", "uniform", "learnability", "staged"),
                "full_barn_campaign"),
    "ablation_n2": (2, ("ours_uN", "learnability"),
                    "full_barn_n_ablation"),
    "ablation_n4": (4, ("ours_uN", "learnability"),
                    "full_barn_n_ablation"),
    "ablation_n16": (16, ("ours_uN", "learnability"),
                     "full_barn_n_ablation"),
}
REQUIRED_CAMPAIGN_CELLS = {
    "primary", "ablation_n2", "ablation_n4", "ablation_n16"}
TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
SAFE_JOB = re.compile(r"[0-9]+(?:_[0-9]+)?\Z")
SAFE_SHA = re.compile(r"[0-9a-f]{64}\Z")
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


def require_canonical_directory(path: Path, *, label: str) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError as error:
        raise Refusal(f"{label} is missing") from error
    refuse(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
           f"{label} is non-directory or symbolic")
    refuse(path.resolve(strict=True) == path,
           f"{label} has a symbolic or non-canonical ancestor")


def require_existing_ancestor_canonical(path: Path, *, label: str) -> None:
    candidate = path.parent
    while not candidate.exists() and not candidate.is_symlink():
        refuse(candidate != candidate.parent,
               f"{label} has no existing absolute ancestor")
        candidate = candidate.parent
    require_canonical_directory(candidate, label=label)


def reject_constant(value: str):
    raise Refusal(f"non-finite JSON constant {value} is forbidden")


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Refusal("duplicate JSON object key is forbidden")
        result[key] = value
    return result


def load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=reject_duplicates,
                parse_constant=reject_constant,
            )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Refusal(f"invalid JSON syntax in {path.name}") from error


def canonical_utc(value, *, label: str) -> str:
    refuse(isinstance(value, str), f"{label} must be a timestamp string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise Refusal(f"{label} is not ISO-8601") from error
    refuse(
        parsed.tzinfo is not None and parsed.utcoffset() is not None,
        f"{label} needs an explicit UTC offset",
    )
    return parsed.astimezone(timezone.utc).isoformat()


def require_int(value, *, label: str, minimum: int | None = None) -> int:
    refuse(isinstance(value, int) and not isinstance(value, bool),
           f"{label} must be an integer")
    if minimum is not None:
        refuse(value >= minimum, f"{label} is below its minimum")
    return value


def validate_manifest(block: Path, artifact: Path, seed: int) -> None:
    manifest = block / "SHA256SUMS"
    complete = block / "COMPLETE"
    for path, label in ((manifest, "SHA256SUMS"), (complete, "COMPLETE"),
                        (artifact, "artifact")):
        refuse(path.is_file() and not path.is_symlink(),
               f"complete seed block lacks regular {label}")

    listed: set[str] = set()
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except UnicodeError as error:
        raise Refusal("SHA256SUMS is not UTF-8") from error
    refuse(bool(lines), "SHA256SUMS is empty")
    for line in lines:
        match = MANIFEST_LINE.fullmatch(line)
        refuse(match is not None, "SHA256SUMS is not strict GNU format")
        raw_name = match.group(3)
        name = PurePosixPath(raw_name)
        refuse(not name.is_absolute() and ".." not in name.parts,
               "SHA256SUMS contains an unsafe path")
        normalized = name.as_posix()
        while normalized.startswith("./"):
            normalized = normalized[2:]
        refuse(bool(normalized) and normalized not in listed,
               "SHA256SUMS contains an empty or duplicate path")
        refuse(normalized not in {"SHA256SUMS", "COMPLETE"}
               and not normalized.startswith("tmp/"),
               "SHA256SUMS lists a forbidden finalization path")
        target = block / normalized
        refuse(target.is_file() and not target.is_symlink(),
               "SHA256SUMS lists a missing, non-regular, or symbolic file")
        listed.add(normalized)

    artifact_rel = f"results/seed-{seed}.json"
    refuse(artifact_rel in listed, "SHA256SUMS omits the evidence artifact")
    actual_files: set[str] = set()
    for candidate in block.rglob("*"):
        refuse(not candidate.is_symlink(), "complete seed block contains a symlink")
        if candidate.is_file():
            relative = candidate.relative_to(block).as_posix()
            if relative not in {"SHA256SUMS", "COMPLETE"}:
                actual_files.add(relative)
    refuse(actual_files == listed,
           "SHA256SUMS does not close over every finalized block file")

    check = subprocess.run(
        ["sha256sum", "-c", "--strict", "--quiet", "SHA256SUMS"],
        cwd=block,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    refuse(check.returncode == 0, "strict SHA256SUMS verification failed")

    complete_rows: dict[str, str] = {}
    for line in complete.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        refuse(len(fields) == 2 and fields[0] and fields[0] not in complete_rows,
               "COMPLETE has malformed or duplicate fields")
        complete_rows[fields[0]] = fields[1]
    refuse(set(complete_rows) == {
        "artifact_type", "campaign_id", "campaign_cell", "attempt_id",
        "seed", "completed_utc", "sha256sums_sha256"},
        "COMPLETE fields are not exact")
    refuse(complete_rows["artifact_type"] == "barn_evidence_seed_complete",
           "COMPLETE artifact type mismatch")
    canonical_utc(complete_rows["completed_utc"], label="completed_utc")
    refuse(complete_rows["sha256sums_sha256"] == sha256_path(manifest),
           "COMPLETE does not bind SHA256SUMS")


def scheduler_rows(array_job_id: str):
    command = [
        "sacct", "-j", array_job_id, "-X", "-n", "-P",
        "--format=JobID,JobIDRaw,State,ExitCode",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    refuse(result.returncode == 0, "Slurm accounting query failed")
    parsed = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if fields and fields[-1] == "":
            fields.pop()
        if len(fields) != 4:
            continue
        array_element, raw_job, raw_state, exit_code = fields
        match = re.fullmatch(re.escape(array_job_id) + r"_([0-9]+)",
                             array_element)
        if match is None:
            continue
        task = int(match.group(1))
        refuse(task not in parsed, "Slurm accounting has duplicate array tasks")
        state = raw_state.split("+", 1)[0].split(maxsplit=1)[0]
        refuse(state in TERMINAL_STATES,
               "at least one recorded Slurm task is not terminal")
        refuse(re.fullmatch(r"[0-9]+", raw_job) is not None,
               "Slurm accounting returned an invalid element job ID")
        if state == "COMPLETED":
            refuse(exit_code == "0:0",
                   "Slurm marked a task COMPLETED with nonzero exit status")
        parsed[task] = {
            "slurm_job_id": raw_job,
            "state": state,
        }
    return parsed


def validate_artifact_shape(
    artifact: object,
    *,
    row: dict,
    actual_job_id: str,
) -> None:
    refuse(isinstance(artifact, dict) and artifact.get("schema_version") == 1,
           "artifact schema is not version 1")
    n_rollouts, arms, evidence_status = CELLS[row["campaign_cell"]]
    refuse(artifact.get("domain") == "barn_gazebo_cpu_navigation",
           "artifact is not BARN navigation")
    refuse(artifact.get("evidence_status") == evidence_status,
           "artifact evidence status differs from campaign cell")

    execution = artifact.get("execution")
    refuse(isinstance(execution, dict) and set(execution) == {
        "campaign_id", "attempt_id", "submitted_utc", "slurm_job_id",
        "slurm_array_job_id", "slurm_array_task_id"},
        "artifact execution identity fields are not exact")
    refuse(execution["campaign_id"] == row["campaign_id"]
           and execution["attempt_id"] == row["attempt_id"],
           "artifact campaign/attempt identity mismatch")
    refuse(canonical_utc(execution["submitted_utc"], label="artifact submitted_utc")
           == canonical_utc(row["submitted_utc"], label="ledger submitted_utc"),
           "artifact submission time mismatch")
    refuse(execution["slurm_job_id"] == actual_job_id
           and execution["slurm_array_job_id"] == row["slurm_array_job_id"]
           and execution["slurm_array_task_id"] == row["slurm_array_task_id"],
           "artifact scheduler identity mismatch")

    provenance = artifact.get("provenance")
    refuse(isinstance(provenance, dict)
           and provenance.get("asset_hashes_verified") is True,
           "artifact lacks verified provenance")
    for field in HASH_FIELDS:
        refuse(provenance.get(field) == row["expected_hashes"][field],
               f"artifact provenance {field} mismatch")
    refuse(provenance.get("split_bound_manifest_sha256")
           == row["expected_hashes"]["manifest_sha256"],
           "artifact split/manifest binding mismatch")

    config = artifact.get("config")
    refuse(isinstance(config, dict), "artifact config is missing")
    seed = row["seed"]
    refuse(config.get("campaign_cell") == row["campaign_cell"],
           "artifact campaign cell mismatch")
    refuse(config.get("seed_list") == [seed]
           and config.get("seeds") == 1
           and config.get("seed_start") == seed
           and config.get("campaign_seed") == seed,
           "artifact seed config mismatch")
    refuse(config.get("arms") == list(arms), "artifact arm config mismatch")
    execution_order = config.get("execution_order")
    refuse(isinstance(execution_order, list)
           and len(execution_order) == len(arms)
           and set(execution_order) == set(arms),
           "artifact execution-order shape mismatch")
    refuse(config.get("n_rollouts") == n_rollouts,
           "artifact rollout config mismatch")
    for field in (
        "steps", "max_training_updates", "tasks_per_step", "eval_every",
        "eval_episodes", "training_sim_step_budget",
        "eval_sim_step_interval", "n_strata", "n_train_courses",
        "n_heldout_courses",
    ):
        require_int(config.get(field), label=f"artifact config {field}", minimum=1)
    results = artifact.get("results")
    refuse(isinstance(results, dict) and set(results) == set(arms),
           "artifact result-key shape mismatch")


def publish_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    refuse(not path.parent.is_symlink(), "finalized-ledger directory is symbolic")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=".ledger-finalize-", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o440)
        try:
            os.link(temporary, path)
        except FileExistsError:
            refuse(path.is_file() and not path.is_symlink()
                   and sha256_path(path) == hashlib.sha256(payload).hexdigest(),
                   "content-addressed finalized ledger already exists with drift")
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    input_path = Path(sys.argv[1])
    campaign_root = Path(sys.argv[2])
    campaign_id = sys.argv[3]
    expected_input_sha = sys.argv[4]
    expected_source_sha = sys.argv[5]
    refuse(SAFE_ID.fullmatch(campaign_id) is not None, "unsafe campaign ID")
    refuse(SAFE_SHA.fullmatch(expected_input_sha) is not None,
           "invalid expected input digest")
    refuse(SAFE_SHA.fullmatch(expected_source_sha) is not None,
           "invalid expected source digest")
    refuse(input_path == campaign_root / "SUBMISSION_LEDGER.json",
           "remote ledger path is not canonical")
    refuse(campaign_root.is_absolute(), "campaign root must be absolute")
    require_canonical_directory(campaign_root, label="campaign root")
    refuse(input_path.is_file() and not input_path.is_symlink(),
           "remote submission ledger is missing or symbolic")
    refuse(sha256_path(input_path) == expected_input_sha,
           "remote and local pre-submission ledger digests differ")

    ledger = load_json(input_path)
    refuse(isinstance(ledger, dict)
           and set(ledger) == {"schema_version", "submissions"}
           and ledger.get("schema_version") == 1
           and isinstance(ledger.get("submissions"), list)
           and ledger["submissions"],
           "submission ledger schema is unsupported or empty")

    identities = set()
    artifact_paths = set()
    expected_blocks = set()
    expected_attempt_dirs = set()
    attempt_groups = {}
    array_groups = {}
    normalized_rows = []
    campaign_hashes = None
    for index, original in enumerate(ledger["submissions"]):
        label = f"submission row {index}"
        refuse(isinstance(original, dict) and set(original) == LEDGER_FIELDS,
               f"{label} fields are not exact")
        row = dict(original)
        refuse(row["campaign_id"] == campaign_id,
               f"{label} belongs to another campaign")
        refuse(row["campaign_cell"] in CELLS, f"{label} has unknown cell")
        refuse(isinstance(row["attempt_id"], str)
               and SAFE_ID.fullmatch(row["attempt_id"]) is not None,
               f"{label} has unsafe attempt ID")
        seed = require_int(row["seed"], label=f"{label} seed", minimum=1)
        task = require_int(
            row["slurm_array_task_id"], label=f"{label} array task", minimum=1)
        refuse(seed in range(1, 6) and task == seed,
               f"{label} seed/task identity mismatch")
        submitted = canonical_utc(row["submitted_utc"],
                                  label=f"{label} submitted_utc")
        array_job = row["slurm_array_job_id"]
        refuse(isinstance(array_job, str) and re.fullmatch(r"[0-9]+", array_job),
               f"{label} has invalid array job ID")
        existing_job = row["slurm_job_id"]
        refuse(existing_job is None
               or (isinstance(existing_job, str)
                   and SAFE_JOB.fullmatch(existing_job) is not None),
               f"{label} has invalid nullable element job ID")
        refuse(isinstance(row["artifact_complete"], bool),
               f"{label} completeness is not boolean")
        existing_artifact_sha = row["artifact_sha256"]
        if row["artifact_complete"]:
            refuse(isinstance(existing_artifact_sha, str)
                   and SAFE_SHA.fullmatch(existing_artifact_sha) is not None,
                   f"{label} has invalid artifact digest")
        else:
            refuse(existing_artifact_sha is None,
                   f"{label} incomplete artifact digest must be null")
        hashes = row["expected_hashes"]
        refuse(isinstance(hashes, dict) and set(hashes) == set(HASH_FIELDS),
               f"{label} expected hashes are not exact")
        for field in HASH_FIELDS:
            refuse(isinstance(hashes[field], str)
                   and SAFE_SHA.fullmatch(hashes[field]) is not None,
                   f"{label} has invalid expected {field}")
        if campaign_hashes is None:
            campaign_hashes = hashes
        else:
            refuse(hashes == campaign_hashes,
                   "expected hashes drift across campaign submissions")

        expected_artifact = (
            campaign_root / "cells" / row["campaign_cell"] / "attempts"
            / row["attempt_id"] / f"seed-{seed}" / "results"
            / f"seed-{seed}.json"
        )
        refuse(isinstance(row["artifact_path"], str)
               and row["artifact_path"] == str(expected_artifact),
               f"{label} artifact path does not match its identity")
        require_existing_ancestor_canonical(
            expected_artifact, label=f"{label} artifact ancestry")
        identity = (campaign_id, row["campaign_cell"], row["attempt_id"], seed)
        refuse(identity not in identities, "duplicate campaign attempt/seed row")
        refuse(row["artifact_path"] not in artifact_paths,
               "submission ledger reuses an artifact path")
        identities.add(identity)
        artifact_paths.add(row["artifact_path"])
        block = expected_artifact.parent.parent
        expected_blocks.add(block)
        expected_attempt_dirs.add(block.parent)

        attempt_key = (row["campaign_cell"], row["attempt_id"], array_job)
        attempt = attempt_groups.setdefault(attempt_key, {
            "seeds": set(), "submitted": submitted, "hashes": hashes})
        refuse(attempt["submitted"] == submitted and attempt["hashes"] == hashes,
               "one attempt has submission-time or expected-hash drift")
        attempt["seeds"].add(seed)
        prior_key = array_groups.setdefault(array_job, attempt_key)
        refuse(prior_key == attempt_key,
               "one Slurm array ID is reused across campaign attempts")
        normalized_rows.append(row)

    for attempt in attempt_groups.values():
        refuse(attempt["seeds"] == set(range(1, 6)),
               "a recorded campaign attempt omits or adds a seed")
    recorded_cells = {row["campaign_cell"] for row in normalized_rows}
    refuse(recorded_cells == REQUIRED_CAMPAIGN_CELLS,
           "campaign ledger must cover exactly the four preregistered cells")
    refuse(campaign_hashes is not None
           and campaign_hashes["source_sha256"] == expected_source_sha,
           "submission ledger source hash differs from finalizer source bundle")

    cells_root = campaign_root / "cells"
    if cells_root.exists():
        refuse(cells_root.is_dir() and not cells_root.is_symlink(),
               "campaign cells path is not a regular directory")
        for attempt_dir in cells_root.glob("*/attempts/*"):
            require_canonical_directory(
                attempt_dir, label="campaign attempt path")
            refuse(attempt_dir in expected_attempt_dirs,
                   "remote campaign contains an unknown/omitted attempt")
        for block in cells_root.glob("*/attempts/*/seed-*"):
            require_canonical_directory(block, label="campaign seed block")
            refuse(block in expected_blocks,
                   "remote campaign contains an unknown/omitted seed block")

    accounting = {}
    for array_job, attempt_key in array_groups.items():
        rows = scheduler_rows(array_job)
        expected_tasks = attempt_groups[attempt_key]["seeds"]
        refuse(set(rows) == expected_tasks,
               "Slurm array task closure differs from the submission ledger")
        for task, state in rows.items():
            accounting[(array_job, task)] = state

    completed = 0
    for row in normalized_rows:
        seed = row["seed"]
        artifact = Path(row["artifact_path"])
        block = artifact.parent.parent
        scheduler = accounting[(row["slurm_array_job_id"], seed)]
        actual_job_id = scheduler["slurm_job_id"]
        if row["slurm_job_id"] is not None:
            refuse(row["slurm_job_id"] == actual_job_id,
                   "recorded element job ID differs from Slurm accounting")
        row["slurm_job_id"] = actual_job_id

        scheduler_complete = scheduler["state"] == "COMPLETED"
        has_complete = (block / "COMPLETE").exists()
        has_artifact = artifact.exists()
        if not scheduler_complete:
            refuse(not block.exists() and not block.is_symlink()
                   and not has_complete and not has_artifact,
                   "failed Slurm task exposes a canonical partial/final seed block")
            row["artifact_complete"] = False
            row["artifact_sha256"] = None
            continue

        require_canonical_directory(block, label="completed Slurm seed block")
        validate_manifest(block, artifact, seed)
        complete_fields = {}
        for line in (block / "COMPLETE").read_text(encoding="utf-8").splitlines():
            key, value = line.split("\t", 1)
            complete_fields[key] = value
        refuse(complete_fields["campaign_id"] == row["campaign_id"]
               and complete_fields["campaign_cell"] == row["campaign_cell"]
               and complete_fields["attempt_id"] == row["attempt_id"]
               and complete_fields["seed"] == str(seed),
               "COMPLETE identity differs from ledger")
        artifact_document = load_json(artifact)
        validate_artifact_shape(
            artifact_document, row=row, actual_job_id=actual_job_id)
        artifact_sha = sha256_path(artifact)
        if row["artifact_complete"]:
            refuse(row["artifact_sha256"] == artifact_sha,
                   "pre-existing finalized artifact digest drifted")
        row["artifact_complete"] = True
        row["artifact_sha256"] = artifact_sha
        completed += 1

    finalized = {"schema_version": 1, "submissions": normalized_rows}
    payload = (json.dumps(
        finalized, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    output = campaign_root / "finalized_ledgers" / (
        f"SUBMISSION_LEDGER.finalized-{digest}.json")
    publish_no_clobber(output, payload)
    print(f"FINALIZED_LEDGER_PATH={output}")
    print(f"FINALIZED_LEDGER_SHA256={digest}")
    print(f"FINALIZED_LEDGER_SUBMISSIONS={len(normalized_rows)}")
    print(f"FINALIZED_LEDGER_COMPLETE={completed}")
    print(f"FINALIZED_LEDGER_INCOMPLETE={len(normalized_rows) - completed}")


try:
    main()
except Refusal as error:
    print(f"BARN ledger finalization refused: {error}", file=sys.stderr)
    raise SystemExit(2)
except (OSError, subprocess.SubprocessError) as error:
    print(f"BARN ledger finalization failed closed: {type(error).__name__}: {error}",
          file=sys.stderr)
    raise SystemExit(2)
PY
REMOTE
)

sentinel_value() {
  local key=$1
  local count value
  count=$(printf '%s\n' "$remote_output" | awk -F= -v wanted="$key" '$1 == wanted {n++} END {print n+0}')
  (( count == 1 )) || {
    printf 'remote finalizer returned %s %s records\n' "$count" "$key" >&2
    exit 2
  }
  value=$(printf '%s\n' "$remote_output" \
    | awk -F= -v wanted="$key" '$1 == wanted {print substr($0, length($1) + 2)}')
  printf '%s\n' "$value"
}

readonly REMOTE_FINAL="$(sentinel_value FINALIZED_LEDGER_PATH)"
readonly FINAL_SHA256="$(sentinel_value FINALIZED_LEDGER_SHA256)"
readonly SUBMISSION_COUNT="$(sentinel_value FINALIZED_LEDGER_SUBMISSIONS)"
readonly COMPLETE_COUNT="$(sentinel_value FINALIZED_LEDGER_COMPLETE)"
readonly INCOMPLETE_COUNT="$(sentinel_value FINALIZED_LEDGER_INCOMPLETE)"
[[ "$FINAL_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "remote finalizer returned an invalid digest" >&2
  exit 2
}
readonly EXPECTED_REMOTE_FINAL="$REMOTE_CAMPAIGN_ROOT/finalized_ledgers/SUBMISSION_LEDGER.finalized-$FINAL_SHA256.json"
[[ "$REMOTE_FINAL" == "$EXPECTED_REMOTE_FINAL" ]] || {
  echo "remote finalizer returned a non-canonical finalized path" >&2
  exit 2
}
for count in "$SUBMISSION_COUNT" "$COMPLETE_COUNT" "$INCOMPLETE_COUNT"; do
  [[ "$count" =~ ^[0-9]+$ ]] || {
    echo "remote finalizer returned an invalid count" >&2
    exit 2
  }
done
(( COMPLETE_COUNT + INCOMPLETE_COUNT == SUBMISSION_COUNT )) || {
  echo "remote finalizer returned inconsistent counts" >&2
  exit 2
}

readonly LOCAL_FINAL="$LOCAL_LEDGER_DIR/$CAMPAIGN_ID.finalized-$FINAL_SHA256.json"
[[ ! -e "$LOCAL_FINAL" && ! -L "$LOCAL_FINAL" ]] || {
  printf 'refusing to overwrite prior finalized ledger: %s\n' "$LOCAL_FINAL" >&2
  exit 2
}
local_partial=$(mktemp "$LOCAL_LEDGER_DIR/.$CAMPAIGN_ID.finalized.XXXXXX")
cleanup() {
  if [[ -n "${local_partial:-}" && -f "$local_partial" ]]; then
    rm -f -- "$local_partial"
  fi
}
trap cleanup EXIT

rsync -az --safe-links --protect-args -e "$RSYNC_RSH" \
  "$HOST:$REMOTE_FINAL" "$local_partial"
[[ -f "$local_partial" && ! -L "$local_partial" ]] || {
  echo "finalized ledger transfer did not produce a regular file" >&2
  exit 2
}
[[ "$(sha256sum -- "$local_partial" | awk '{print $1}')" == "$FINAL_SHA256" ]] || {
  echo "finalized ledger checksum differs after transfer" >&2
  exit 2
}

# Recheck only the fixed ledger envelope locally before publishing the fetched
# copy. The ledger itself contains no experiment endpoints.
python3 - "$local_partial" "$SUBMISSION_COUNT" "$COMPLETE_COUNT" <<'PY'
import json
import sys

fields = {
    "campaign_id", "campaign_cell", "attempt_id", "seed", "submitted_utc",
    "slurm_job_id", "slurm_array_job_id", "slurm_array_task_id",
    "artifact_path", "artifact_complete", "artifact_sha256",
    "expected_hashes",
}
with open(sys.argv[1], encoding="utf-8") as stream:
    ledger = json.load(stream)
if (not isinstance(ledger, dict)
        or set(ledger) != {"schema_version", "submissions"}
        or ledger.get("schema_version") != 1
        or not isinstance(ledger.get("submissions"), list)):
    raise SystemExit("fetched finalized ledger envelope mismatch")
if len(ledger["submissions"]) != int(sys.argv[2]):
    raise SystemExit("fetched finalized ledger row count mismatch")
if any(not isinstance(row, dict) or set(row) != fields
       for row in ledger["submissions"]):
    raise SystemExit("fetched finalized ledger row schema mismatch")
if sum(row["artifact_complete"] is True for row in ledger["submissions"]) != int(sys.argv[3]):
    raise SystemExit("fetched finalized ledger completeness count mismatch")
PY

chmod 0440 "$local_partial"
if ! ln -- "$local_partial" "$LOCAL_FINAL"; then
  echo "finalized ledger destination appeared during transfer; refusing overwrite" >&2
  exit 2
fi
rm -f -- "$local_partial"
local_partial=""
trap - EXIT

printf 'BARN_LEDGER_FINALIZED\tcampaign=%s\tsubmissions=%s\tcomplete=%s\tincomplete=%s\tsha256=%s\tremote=%s\tlocal=%s\n' \
  "$CAMPAIGN_ID" "$SUBMISSION_COUNT" "$COMPLETE_COUNT" "$INCOMPLETE_COUNT" \
  "$FINAL_SHA256" "$REMOTE_FINAL" "$LOCAL_FINAL"
