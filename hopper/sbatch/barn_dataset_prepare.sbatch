#!/usr/bin/env bash
#SBATCH --job-name=barn-data-prepare
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/maxrl/barn/logs/%x_%j.out

# CPU-only compute-node preparation of the official BARN archive.  The input
# archive and source closure must already be content-addressed under scratch.
# This job never downloads data and never runs on a login node.  It extracts
# into a fresh directory, regenerates the 300-course manifest using the staged
# source, verifies every manifest-bound asset, seals receipts, and performs an
# atomic no-replace rename to the canonical data path.
set -euo pipefail
umask 027

readonly EXPECTED_ARCHIVE_SHA256="5ad443412f6f2f38b6d0e1d330c9a820ab48e566553197459005e751711fe320"
readonly EXPECTED_CONTAINER_SHA256="cd6620e33c0822f7d6a03c6de6ea9dd4304f0927e8d7997c003560f5b4781be0"

required=(
  BARN_SOURCE_BUNDLE_DIR
  BARN_SOURCE_SHA256
  BARN_DATASET_ARCHIVE
  BARN_DATASET_ARCHIVE_SHA256
  SLURM_JOB_ID
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || {
    printf 'missing required variable: %s\n' "$name" >&2
    exit 2
  }
done
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || {
  echo "invalid SLURM_JOB_ID" >&2
  exit 2
}
[[ "$BARN_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "invalid BARN_SOURCE_SHA256" >&2
  exit 2
}
[[ "$BARN_DATASET_ARCHIVE_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "invalid BARN_DATASET_ARCHIVE_SHA256" >&2
  exit 2
}
[[ "$BARN_DATASET_ARCHIVE_SHA256" == "$EXPECTED_ARCHIVE_SHA256" ]] || {
  echo "caller archive digest is not the frozen official BARN digest" >&2
  exit 2
}

RUN_USER=$(id -un)
readonly RUN_USER
readonly SCRATCH_ROOT="/scratch/$RUN_USER"
readonly APPTAINER_BIN="/opt/sw/other/apps/apptainer/1.4.1/bin/apptainer"
readonly SIF="$SCRATCH_ROOT/ros2-gazebo-classic.sif"
readonly ARCHIVE_PREFIX="${EXPECTED_ARCHIVE_SHA256:0:20}"
readonly EXPECTED_ARCHIVE_PATH="$SCRATCH_ROOT/maxrl/bundles/barn_dataset/$ARCHIVE_PREFIX/BARN_dataset.zip"
readonly DATASET_PARENT="$SCRATCH_ROOT/maxrl/barn/datasets"
readonly FINAL_PACKAGE="$DATASET_PARENT/$EXPECTED_ARCHIVE_SHA256"

[[ -x "$APPTAINER_BIN" ]] || {
  echo "pinned Apptainer executable unavailable" >&2
  exit 2
}
[[ -f "$SIF" && ! -L "$SIF" ]] || {
  echo "pinned CPU SIF missing or symbolic" >&2
  exit 2
}
[[ "$(sha256sum "$SIF" | awk '{print $1}')" == "$EXPECTED_CONTAINER_SHA256" ]] || {
  echo "pinned CPU SIF hash mismatch" >&2
  exit 2
}

[[ -d "$BARN_SOURCE_BUNDLE_DIR" && ! -L "$BARN_SOURCE_BUNDLE_DIR" ]] || {
  echo "source bundle missing or symbolic" >&2
  exit 2
}
SOURCE_BUNDLE=$(readlink -f -- "$BARN_SOURCE_BUNDLE_DIR")
readonly SOURCE_BUNDLE
case "$SOURCE_BUNDLE" in
  "$SCRATCH_ROOT"/*) ;;
  *) echo "source bundle outside scratch" >&2; exit 2 ;;
esac
[[ "$SOURCE_BUNDLE" == "$BARN_SOURCE_BUNDLE_DIR" ]] || {
  echo "source bundle path must be canonical" >&2
  exit 2
}
readonly SOURCE_MANIFEST="$SOURCE_BUNDLE/SHA256SUMS"
[[ -f "$SOURCE_MANIFEST" && ! -L "$SOURCE_MANIFEST" ]] || {
  echo "source SHA256SUMS missing or symbolic" >&2
  exit 2
}
[[ "$(sha256sum "$SOURCE_MANIFEST" | awk '{print $1}')" == "$BARN_SOURCE_SHA256" ]] || {
  echo "source bundle manifest hash mismatch" >&2
  exit 2
}

# Reject absolute, parent-traversing, duplicate, or symbolic entries before
# handing the trusted-format manifest to sha256sum.
python3 - "$SOURCE_MANIFEST" "$SOURCE_BUNDLE" <<'PY'
from pathlib import Path, PurePosixPath
import sys

manifest = Path(sys.argv[1])
root = Path(sys.argv[2]).resolve()
seen = set()
for line_number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
    if not raw or raw.startswith("\\") or len(raw) < 67:
        raise SystemExit(f"unsafe source manifest line {line_number}")
    digest, marker, name = raw[:64], raw[64:66], raw[66:]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise SystemExit(f"invalid digest on source manifest line {line_number}")
    if marker not in ("  ", " *"):
        raise SystemExit(f"invalid checksum marker on source manifest line {line_number}")
    relative = PurePosixPath(name)
    if (not name or "\\" in name or relative.is_absolute()
            or any(part in ("", ".", "..") for part in relative.parts)):
        raise SystemExit(f"unsafe path on source manifest line {line_number}")
    if name in seen:
        raise SystemExit(f"duplicate path on source manifest line {line_number}")
    seen.add(name)
    target = root.joinpath(*relative.parts)
    if target.is_symlink() or not target.is_file():
        raise SystemExit(f"missing or symbolic source entry: {name}")
    try:
        target.resolve().relative_to(root)
    except ValueError as error:
        raise SystemExit(f"source entry escapes bundle: {name}") from error
if not seen:
    raise SystemExit("empty source manifest")
PY
(cd "$SOURCE_BUNDLE" && sha256sum -c --strict SHA256SUMS >/dev/null)

readonly BUNDLED_SBATCH="$SOURCE_BUNDLE/hopper/sbatch/barn_dataset_prepare.sbatch"
[[ -f "$BUNDLED_SBATCH" && ! -L "$BUNDLED_SBATCH" ]] || {
  echo "bundled dataset-prepare sbatch is missing or symbolic" >&2
  exit 2
}
RUNNING_SBATCH=$(readlink -f -- "$0")
readonly RUNNING_SBATCH
[[ -f "$RUNNING_SBATCH" && ! -L "$RUNNING_SBATCH" \
   && "$(sha256sum "$RUNNING_SBATCH" | awk '{print $1}')" == \
      "$(sha256sum "$BUNDLED_SBATCH" | awk '{print $1}')" ]] || {
  echo "running dataset-prepare sbatch differs from immutable source closure" >&2
  exit 2
}

source_required=(
  icra2027/barn_manifest.jsonl
  icra2027/barn_split.json
  icra2027/build_barn_manifest.py
  icra2027/barn_campaign.py
  icra2027/barn_protocol.json
)
for relative in "${source_required[@]}"; do
  [[ -f "$SOURCE_BUNDLE/$relative" && ! -L "$SOURCE_BUNDLE/$relative" ]] || {
    printf 'required source file missing or symbolic: %s\n' "$relative" >&2
    exit 2
  }
done

[[ -f "$BARN_DATASET_ARCHIVE" && ! -L "$BARN_DATASET_ARCHIVE" ]] || {
  echo "dataset archive missing or symbolic" >&2
  exit 2
}
ARCHIVE=$(readlink -f -- "$BARN_DATASET_ARCHIVE")
readonly ARCHIVE
[[ "$ARCHIVE" == "$BARN_DATASET_ARCHIVE" ]] || {
  echo "dataset archive path must be canonical" >&2
  exit 2
}
[[ "$ARCHIVE" == "$EXPECTED_ARCHIVE_PATH" ]] || {
  echo "dataset archive is not at the canonical content-addressed path" >&2
  exit 2
}
[[ "$(sha256sum "$ARCHIVE" | awk '{print $1}')" == "$EXPECTED_ARCHIVE_SHA256" ]] || {
  echo "official BARN archive hash mismatch" >&2
  exit 2
}

mkdir -p -- "$DATASET_PARENT"
[[ -d "$DATASET_PARENT" && ! -L "$DATASET_PARENT" \
   && "$(readlink -f -- "$DATASET_PARENT")" == "$DATASET_PARENT" ]] || {
  echo "dataset parent is symbolic or non-canonical" >&2
  exit 2
}
if ! mkdir -- "$FINAL_PACKAGE"; then
  printf 'refusing to overwrite or reuse prepared dataset: %s\n' \
    "$FINAL_PACKAGE" >&2
  exit 2
fi
readonly STAGE_DIR="$FINAL_PACKAGE"
WORK_ACTIVE=1
cleanup() {
  if [[ "${WORK_ACTIVE:-0}" == 1 && -n "${STAGE_DIR:-}" \
        && "$STAGE_DIR" == "$FINAL_PACKAGE" \
        && -d "$STAGE_DIR" && ! -L "$STAGE_DIR" ]]; then
    chmod -R u+w -- "$STAGE_DIR" 2>/dev/null || true
    rm -rf -- "$STAGE_DIR"
  fi
}
trap cleanup EXIT INT TERM
mkdir -- "$STAGE_DIR/meta"
cp -- "$SOURCE_MANIFEST" "$STAGE_DIR/meta/SOURCE_SHA256SUMS"

# The exact archive digest is fixed above.  Validate member paths/types before
# extraction, then regenerate and compare the canonical source manifest.  The
# campaign verifier independently checks all manifest-bound world/path/grid
# hashes and the frozen split binding.
TIMEFORMAT=$'real_seconds\t%R\nuser_seconds\t%U\nsystem_seconds\t%S'
{
  time "$APPTAINER_BIN" exec --cleanenv \
  --bind "$SCRATCH_ROOT:$SCRATCH_ROOT" "$SIF" bash -s -- \
  "$ARCHIVE" "$STAGE_DIR" "$SOURCE_BUNDLE" \
  2> "$STAGE_DIR/meta/runtime-stderr.txt" <<'CONTAINER'
set -euo pipefail
archive="$1"
stage="$2"
bundle="$3"

python3 - "$archive" "$stage" <<'PY'
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import shutil
import stat
import sys

archive = Path(sys.argv[1])
destination = Path(sys.argv[2])
seen = set()
with ZipFile(archive) as handle:
    for index, info in enumerate(handle.infolist(), 1):
        name = info.filename
        relative = PurePosixPath(name)
        if (not name or "\\" in name or relative.is_absolute()
                or any(part in ("", ".", "..") for part in relative.parts)
                or relative.parts[0] != "BARN_dataset"):
            raise SystemExit(f"unsafe zip member {index}: {name!r}")
        normalized = relative.as_posix().rstrip("/")
        if normalized in seen:
            raise SystemExit(f"duplicate zip member {index}: {name!r}")
        seen.add(normalized)
        if info.flag_bits & 1:
            raise SystemExit(f"encrypted zip member {index}: {name!r}")
        mode = (info.external_attr >> 16) & 0o170000
        if mode not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise SystemExit(f"non-regular zip member {index}: {name!r}")
        target = destination.joinpath(*relative.parts)
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=False)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with handle.open(info, "r") as source, target.open("xb") as sink:
            shutil.copyfileobj(source, sink, length=1024 * 1024)

root = destination / "BARN_dataset"
if not root.is_dir():
    raise SystemExit("archive did not contain BARN_dataset root")
if any(path.is_symlink() for path in root.rglob("*")):
    raise SystemExit("symbolic link found after extraction")
PY

python3 "$bundle/icra2027/build_barn_manifest.py" \
  "$stage/BARN_dataset" --output "$stage/meta/rebuilt_barn_manifest.jsonl" \
  > "$stage/meta/manifest-build.txt"
cmp -- "$bundle/icra2027/barn_manifest.jsonl" \
  "$stage/meta/rebuilt_barn_manifest.jsonl"

PYTHONPATH="$bundle:$bundle/curriculum_maxrl" python3 - \
  "$stage/BARN_dataset" "$bundle/icra2027/barn_manifest.jsonl" \
  "$bundle/icra2027/barn_split.json" "$stage/meta/dataset-verification.json" <<'PY'
import json
from pathlib import Path
import sys

from icra2027.barn_campaign import load_frozen_inputs, verify_dataset_assets

dataset = Path(sys.argv[1])
manifest = Path(sys.argv[2])
split_path = Path(sys.argv[3])
output = Path(sys.argv[4])
split, rows = load_frozen_inputs(manifest, split_path)
if len(rows) != 300 or len({row["env_id"] for row in rows}) != 300:
    raise SystemExit("frozen manifest is not exactly 300 unique courses")
if sorted(row["barn_index"] for row in rows) != list(range(300)):
    raise SystemExit("frozen manifest BARN indices are not exactly 0..299")
verify_dataset_assets(dataset, rows)
checks = sum(
    2 + int("grid_asset" in row and "grid_sha256" in row)
    for row in rows
)
output.write_text(json.dumps({
    "artifact_type": "barn_dataset_verification",
    "course_count": len(rows),
    "unique_course_count": len({row["env_id"] for row in rows}),
    "manifest_bound_asset_hashes_verified": checks,
    "train_course_count": len(split["train_ids"]),
    "heldout_course_count": len(split["heldout_ids"]),
    "outcomes_read": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
CONTAINER
} 2> "$STAGE_DIR/meta/resource-time.txt"

export BARN_PREP_RECEIPT="$STAGE_DIR/PREPARE_RECEIPT.json"
export BARN_PREP_SOURCE="$SOURCE_BUNDLE"
export BARN_PREP_ARCHIVE="$ARCHIVE"
export BARN_PREP_FINAL="$FINAL_PACKAGE"
export BARN_PREP_SIF="$SIF"
export BARN_PREP_SOURCE_SHA="$BARN_SOURCE_SHA256"
export BARN_PREP_ARCHIVE_SHA="$EXPECTED_ARCHIVE_SHA256"
export BARN_PREP_CONTAINER_SHA="$EXPECTED_CONTAINER_SHA256"
export BARN_PREP_MANIFEST_SHA
BARN_PREP_MANIFEST_SHA=$(sha256sum "$SOURCE_BUNDLE/icra2027/barn_manifest.jsonl" | awk '{print $1}')
export BARN_PREP_SPLIT_SHA
BARN_PREP_SPLIT_SHA=$(sha256sum "$SOURCE_BUNDLE/icra2027/barn_split.json" | awk '{print $1}')
python3 - <<'PY'
from datetime import datetime, timezone
import json
import os
import platform
from pathlib import Path

receipt = {
    "artifact_type": "barn_hopper_dataset_prepare_receipt",
    "evidence_status": "immutable_dataset_preparation_not_paper_evidence",
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "job_id": int(os.environ["SLURM_JOB_ID"]),
    "node": platform.node(),
    "cpu_only": True,
    "source_bundle": os.environ["BARN_PREP_SOURCE"],
    "source_manifest_sha256": os.environ["BARN_PREP_SOURCE_SHA"],
    "archive_path": os.environ["BARN_PREP_ARCHIVE"],
    "archive_sha256": os.environ["BARN_PREP_ARCHIVE_SHA"],
    "container_path": os.environ["BARN_PREP_SIF"],
    "container_sha256": os.environ["BARN_PREP_CONTAINER_SHA"],
    "barn_manifest_sha256": os.environ["BARN_PREP_MANIFEST_SHA"],
    "barn_split_sha256": os.environ["BARN_PREP_SPLIT_SHA"],
    "canonical_package": os.environ["BARN_PREP_FINAL"],
    "dataset_subdirectory": "BARN_dataset",
    "archive_extracted_on_compute": True,
    "atomic_finalize": "mkdir_claim_plus_complete_last_hardlink",
    "paper_endpoint_emitted": False,
}
Path(os.environ["BARN_PREP_RECEIPT"]).write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

(
  cd "$STAGE_DIR"
  find . -type f ! -name SHA256SUMS ! -name COMPLETE -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c --strict SHA256SUMS >/dev/null
)
# The canonical directory was atomically claimed before extraction. Seal every
# payload first, then publish COMPLETE as the sole visibility gate using an
# exclusive hard link (supported by Hopper scratch even though directory
# renameat2(RENAME_NOREPLACE) is not).
find "$STAGE_DIR" -mindepth 1 -type f -exec chmod 0444 {} +
find "$STAGE_DIR" -mindepth 1 -type d -exec chmod 0555 {} +
python3 - "$STAGE_DIR" "$SLURM_JOB_ID" <<'PY'
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
job_id = sys.argv[2]
manifest_digest = __import__("hashlib").sha256(
    (root / "SHA256SUMS").read_bytes()).hexdigest()
temporary = root / f".COMPLETE.stage-{job_id}"
destination = root / "COMPLETE"
payload = (
    "artifact_type\tbarn_hopper_dataset_complete\n"
    f"completed_utc\t{datetime.now(timezone.utc).isoformat()}\n"
    f"sha256sums_sha256\t{manifest_digest}\n"
)
with temporary.open("x", encoding="utf-8") as stream:
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
temporary.chmod(0o444)
os.link(temporary, destination)
temporary.unlink()
root.chmod(0o555)
PY
WORK_ACTIVE=0
trap - EXIT INT TERM

[[ -f "$FINAL_PACKAGE/COMPLETE" && -f "$FINAL_PACKAGE/SHA256SUMS" ]] || {
  echo "completion-last finalization did not publish sealed controls" >&2
  exit 2
}
printf 'BARN_DATASET_PREPARE_PASS\t%s\n' "$FINAL_PACKAGE"
