#!/usr/bin/env bash
#SBATCH --job-name=barn-evidence-safe-log
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=1-12:00:00
#SBATCH --output=/scratch/%u/maxrl/barn/logs/%x_%A_%a.out
#SBATCH --array=1-5%5

# One immutable BARN evidence cell/seed per array task, CPU-only.  The machine
# protocol selects either the four-arm primary cell or one two-arm N-ablation
# cell.  Scientific endpoints are written below the sealed attempt tree and
# are never printed; stdout contains only provenance/completion metadata.
set -euo pipefail
umask 027
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=""

readonly EXPECTED_ARCHIVE_SHA256="5ad443412f6f2f38b6d0e1d330c9a820ab48e566553197459005e751711fe320"
readonly EXPECTED_CONTAINER_SHA256="cd6620e33c0822f7d6a03c6de6ea9dd4304f0927e8d7997c003560f5b4781be0"

required=(
  BARN_SOURCE_BUNDLE_DIR
  BARN_SOURCE_SHA256
  BARN_DATASET_ARCHIVE_SHA256
  BARN_CONTAINER_SHA256
  BARN_MANIFEST_SHA256
  BARN_SPLIT_SHA256
  BARN_PREREG_SHA256
  BARN_ANALYZER_SHA256
  BARN_PROTOCOL_SHA256
  BARN_CAMPAIGN_ID
  BARN_CAMPAIGN_CELL
  BARN_ATTEMPT_ID
  BARN_SUBMITTED_UTC
  SLURM_JOB_ID
  SLURM_ARRAY_JOB_ID
  SLURM_ARRAY_TASK_ID
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || {
    printf 'missing required variable: %s\n' "$name" >&2
    exit 2
  }
done

for name in BARN_SOURCE_SHA256 BARN_DATASET_ARCHIVE_SHA256 \
            BARN_CONTAINER_SHA256 BARN_MANIFEST_SHA256 BARN_SPLIT_SHA256 \
            BARN_PREREG_SHA256 BARN_ANALYZER_SHA256 BARN_PROTOCOL_SHA256; do
  [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || {
    printf 'invalid SHA-256 variable: %s\n' "$name" >&2
    exit 2
  }
done
[[ "$BARN_DATASET_ARCHIVE_SHA256" == "$EXPECTED_ARCHIVE_SHA256" ]] || {
  echo "archive hash differs from frozen official BARN archive" >&2
  exit 2
}
[[ "$BARN_CONTAINER_SHA256" == "$EXPECTED_CONTAINER_SHA256" ]] || {
  echo "container hash differs from frozen Hopper image" >&2
  exit 2
}
[[ "$BARN_CAMPAIGN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]] || {
  echo "unsafe BARN_CAMPAIGN_ID" >&2
  exit 2
}
[[ "$BARN_ATTEMPT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]] || {
  echo "unsafe BARN_ATTEMPT_ID" >&2
  exit 2
}
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[1-5]$ ]] || {
  echo "evidence seed must be exactly one of 1..5" >&2
  exit 2
}
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && "$SLURM_ARRAY_JOB_ID" =~ ^[0-9]+$ ]] || {
  echo "invalid Slurm job identity" >&2
  exit 2
}
python3 - "$BARN_SUBMITTED_UTC" <<'PY'
from datetime import datetime
import sys
value = sys.argv[1]
parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
if parsed.tzinfo is None or parsed.utcoffset() is None:
    raise SystemExit("BARN_SUBMITTED_UTC must include an explicit UTC offset")
PY

case "$BARN_CAMPAIGN_CELL" in
  primary)
    readonly CELL_INDEX=0
    readonly N_ROLLOUTS=8
    readonly ARMS="ours_uN,uniform,learnability,staged"
    ;;
  ablation_n2)
    readonly CELL_INDEX=1 N_ROLLOUTS=2 ARMS="ours_uN,learnability"
    ;;
  ablation_n4)
    readonly CELL_INDEX=2 N_ROLLOUTS=4 ARMS="ours_uN,learnability"
    ;;
  ablation_n16)
    readonly CELL_INDEX=3 N_ROLLOUTS=16 ARMS="ours_uN,learnability"
    ;;
  *) echo "invalid BARN_CAMPAIGN_CELL" >&2; exit 2 ;;
esac

readonly SEED="$SLURM_ARRAY_TASK_ID"
readonly RUN_USER="$(id -un)"
readonly SCRATCH_ROOT="/scratch/$RUN_USER"
readonly APPTAINER_BIN="/opt/sw/other/apps/apptainer/1.4.1/bin/apptainer"
readonly SIF="$SCRATCH_ROOT/ros2-gazebo-classic.sif"
readonly DATASET_PACKAGE="$SCRATCH_ROOT/maxrl/barn/datasets/$EXPECTED_ARCHIVE_SHA256"
readonly DATASET_ROOT="$DATASET_PACKAGE/BARN_dataset"
readonly CAMPAIGN_ROOT="$SCRATCH_ROOT/maxrl/barn/campaigns/$BARN_CAMPAIGN_ID"
readonly ATTEMPT_ROOT="$CAMPAIGN_ROOT/cells/$BARN_CAMPAIGN_CELL/attempts/$BARN_ATTEMPT_ID"
readonly FINAL_BLOCK="$ATTEMPT_ROOT/seed-$SEED"
readonly WORK_BLOCK="$ATTEMPT_ROOT/.seed-$SEED.stage-$SLURM_JOB_ID"
readonly BLOCK="$WORK_BLOCK"
readonly DOMAIN_ID=$((20 + CELL_INDEX * 30 + SEED * 2))
readonly MASTER_PORT=$((13000 + CELL_INDEX * 1000 + SEED * 4))

[[ "$DOMAIN_ID" -ge 0 && $((DOMAIN_ID + 1)) -le 232 ]] || {
  echo "computed ROS domain pair is out of range" >&2
  exit 2
}
[[ "$MASTER_PORT" -ge 1024 && $((MASTER_PORT + 1)) -le 65535 ]] || {
  echo "computed Gazebo port pair is out of range" >&2
  exit 2
}
[[ -x "$APPTAINER_BIN" ]] || {
  echo "pinned Apptainer executable unavailable" >&2
  exit 2
}
[[ -f "$SIF" && ! -L "$SIF" ]] || {
  echo "pinned CPU SIF missing or symbolic" >&2
  exit 2
}
[[ "$(sha256sum "$SIF" | awk '{print $1}')" == "$BARN_CONTAINER_SHA256" ]] || {
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
readonly SOURCE_STATE="$SOURCE_BUNDLE/SOURCE_STATE.json"
[[ -f "$SOURCE_MANIFEST" && ! -L "$SOURCE_MANIFEST" \
   && -f "$SOURCE_STATE" && ! -L "$SOURCE_STATE" ]] || {
  echo "source provenance files missing or symbolic" >&2
  exit 2
}
[[ "$(sha256sum "$SOURCE_MANIFEST" | awk '{print $1}')" == "$BARN_SOURCE_SHA256" ]] || {
  echo "source manifest hash mismatch" >&2
  exit 2
}
(cd "$SOURCE_BUNDLE" && sha256sum -c --strict SHA256SUMS >/dev/null)
python3 - "$SOURCE_STATE" "$SOURCE_BUNDLE" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
state = json.load(open(sys.argv[1], encoding="utf-8"))
if state.get("mode") != "evidence":
    raise SystemExit("source bundle was not staged in evidence mode")
if state.get("worktree_dirty") is not False:
    raise SystemExit("evidence source contains dirty bundled paths")
if state.get("relevant_paths_match_head") is not True:
    raise SystemExit("evidence source paths do not match the recorded commit")
files = state.get("files")
if not isinstance(files, list) or not files:
    raise SystemExit("evidence source state lacks its file closure")
root = Path(sys.argv[2])
seen = set()
for row in files:
    relative = Path(row["path"])
    if relative.is_absolute() or ".." in relative.parts or str(relative) in seen:
        raise SystemExit("unsafe or duplicate source-state path")
    seen.add(str(relative))
    path = root / relative
    info = os.lstat(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    mode = format(stat.S_IMODE(info.st_mode), "o")
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or digest != row.get("worktree_sha256")
            or mode != row.get("worktree_mode")):
        raise SystemExit("source-state file hash/mode evidence mismatch")
PY

# Slurm executes a staged copy of this script.  Require that copy to be
# byte-identical to the version included in the source manifest whose digest
# is the frozen source binding.
readonly BUNDLED_SBATCH="$SOURCE_BUNDLE/hopper/sbatch/barn_seed_cpu.sbatch"
[[ -f "$BUNDLED_SBATCH" && ! -L "$BUNDLED_SBATCH" ]] || {
  echo "bundled evidence sbatch is missing or symbolic" >&2
  exit 2
}
RUNNING_SBATCH=$(readlink -f -- "$0")
readonly RUNNING_SBATCH
[[ -f "$RUNNING_SBATCH" && ! -L "$RUNNING_SBATCH" ]] || {
  echo "running evidence sbatch is missing or symbolic" >&2
  exit 2
}
[[ "$(sha256sum "$RUNNING_SBATCH" | awk '{print $1}')" == \
   "$(sha256sum "$BUNDLED_SBATCH" | awk '{print $1}')" ]] || {
  echo "running evidence sbatch differs from immutable source closure" >&2
  exit 2
}

readonly MANIFEST="$SOURCE_BUNDLE/icra2027/barn_manifest.jsonl"
readonly SPLIT="$SOURCE_BUNDLE/icra2027/barn_split.json"
readonly PREREG="$SOURCE_BUNDLE/icra2027/prereg_icra.md"
readonly ANALYZER="$SOURCE_BUNDLE/icra2027/analyze_campaign.py"
readonly PROTOCOL="$SOURCE_BUNDLE/icra2027/barn_protocol.json"
readonly ROBOT_SDF="$SOURCE_BUNDLE/icra2027/assets/barn_diff_drive.sdf"
for path in "$MANIFEST" "$SPLIT" "$PREREG" "$ANALYZER" "$PROTOCOL" \
            "$ROBOT_SDF" "$SOURCE_BUNDLE/icra2027/build_gazebo_stepper.sh"; do
  [[ -f "$path" && ! -L "$path" ]] || {
    printf 'missing or symbolic evidence input: %s\n' "$path" >&2
    exit 2
  }
done
check_hash() {
  local path=$1 expected=$2 label=$3 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || {
    printf '%s hash mismatch\n' "$label" >&2
    exit 2
  }
}
check_hash "$MANIFEST" "$BARN_MANIFEST_SHA256" manifest
check_hash "$SPLIT" "$BARN_SPLIT_SHA256" split
check_hash "$PREREG" "$BARN_PREREG_SHA256" preregistration
check_hash "$ANALYZER" "$BARN_ANALYZER_SHA256" analyzer
check_hash "$PROTOCOL" "$BARN_PROTOCOL_SHA256" protocol
grep -Eq '^\*\*Status:\*\* FROZEN[[:blank:]]*$' "$PREREG" || {
  echo "preregistration is not FROZEN" >&2
  exit 2
}
python3 - "$PROTOCOL" "$BARN_CAMPAIGN_CELL" "$SEED" <<'PY'
import json
import sys
protocol = json.load(open(sys.argv[1], encoding="utf-8"))
cell, seed = sys.argv[2], int(sys.argv[3])
if protocol.get("status") != "FROZEN":
    raise SystemExit("machine protocol is not FROZEN")
if seed not in protocol.get("shared_training", {}).get("seeds", []):
    raise SystemExit("seed is absent from frozen protocol")
if cell == "primary":
    selected = protocol.get("primary", {})
else:
    selected = protocol.get("ablation", {})
    if cell not in selected.get("fresh_cell_names", []):
        raise SystemExit("ablation cell is absent from frozen protocol")
order = selected.get("execution_order_by_seed", {}).get(str(seed))
if not isinstance(order, list) or set(order) != set(selected.get("arms", [])):
    raise SystemExit("missing/invalid seed-specific execution order")
PY

[[ -d "$DATASET_PACKAGE" && ! -L "$DATASET_PACKAGE" \
   && -d "$DATASET_ROOT" && ! -L "$DATASET_ROOT" \
   && -f "$DATASET_PACKAGE/PREPARE_RECEIPT.json" \
   && -f "$DATASET_PACKAGE/SHA256SUMS" \
   && -f "$DATASET_PACKAGE/COMPLETE" ]] || {
  echo "complete immutable prepared dataset unavailable" >&2
  exit 2
}
(cd "$DATASET_PACKAGE" && sha256sum -c --strict SHA256SUMS >/dev/null)
python3 - "$DATASET_PACKAGE/PREPARE_RECEIPT.json" \
  "$EXPECTED_ARCHIVE_SHA256" "$BARN_MANIFEST_SHA256" \
  "$BARN_SPLIT_SHA256" "$BARN_CONTAINER_SHA256" <<'PY'
import json
import sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "archive_sha256": sys.argv[2],
    "barn_manifest_sha256": sys.argv[3],
    "barn_split_sha256": sys.argv[4],
    "container_sha256": sys.argv[5],
}
for field, value in expected.items():
    if receipt.get(field) != value:
        raise SystemExit(f"prepared dataset {field} mismatch")
PY

# Create every evidence-tree ancestor one component at a time, rejecting
# symbolic links and non-canonical resolution.  Concurrent array tasks may
# race to create shared parents, so a losing mkdir is accepted only after the
# same no-follow checks pass.
python3 - "$SCRATCH_ROOT" "$CAMPAIGN_ROOT" "$BARN_CAMPAIGN_CELL" \
  "$BARN_ATTEMPT_ID" <<'PY'
import os
from pathlib import Path
import stat
import sys

scratch = Path(sys.argv[1])
campaign_root = Path(sys.argv[2])
cell, attempt = sys.argv[3:]
expected_campaign = scratch / "maxrl" / "barn" / "campaigns" / campaign_root.name
if campaign_root != expected_campaign:
    raise SystemExit("campaign root is not canonical")
paths = [
    scratch,
    scratch / "maxrl",
    scratch / "maxrl" / "barn",
    scratch / "maxrl" / "barn" / "campaigns",
    campaign_root,
    campaign_root / "cells",
    campaign_root / "cells" / cell,
    campaign_root / "cells" / cell / "attempts",
    campaign_root / "cells" / cell / "attempts" / attempt,
]
for path in paths:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        try:
            os.mkdir(path, 0o750)
        except FileExistsError:
            pass
        info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise SystemExit(f"unsafe evidence-tree component: {path}")
    if path.resolve(strict=True) != path:
        raise SystemExit(f"non-canonical evidence-tree component: {path}")
PY

[[ ! -e "$FINAL_BLOCK" && ! -L "$FINAL_BLOCK" ]] || {
  printf 'refusing to overwrite evidence block: %s\n' "$FINAL_BLOCK" >&2
  exit 2
}
[[ ! -e "$WORK_BLOCK" && ! -L "$WORK_BLOCK" ]] || {
  printf 'refusing to overwrite evidence work block: %s\n' "$WORK_BLOCK" >&2
  exit 2
}
mkdir -- "$BLOCK"
mkdir -- "$BLOCK/bin" "$BLOCK/meta" "$BLOCK/results" "$BLOCK/runtime" "$BLOCK/tmp"
cp -- "$SOURCE_MANIFEST" "$BLOCK/meta/SOURCE_SHA256SUMS"
cp -- "$SOURCE_STATE" "$BLOCK/meta/SOURCE_STATE.json"
cp -- "$PREREG" "$BLOCK/meta/prereg_icra.md"
cp -- "$PROTOCOL" "$BLOCK/meta/barn_protocol.json"

{
  printf 'key\tvalue\n'
  printf 'campaign_id\t%s\n' "$BARN_CAMPAIGN_ID"
  printf 'campaign_cell\t%s\n' "$BARN_CAMPAIGN_CELL"
  printf 'attempt_id\t%s\n' "$BARN_ATTEMPT_ID"
  printf 'seed\t%s\n' "$SEED"
  printf 'submitted_utc\t%s\n' "$BARN_SUBMITTED_UTC"
  printf 'started_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'slurm_job_id\t%s\n' "$SLURM_JOB_ID"
  printf 'slurm_array_job_id\t%s\n' "$SLURM_ARRAY_JOB_ID"
  printf 'node\t%s\n' "$(hostname)"
  printf 'cpu_only\ttrue\n'
  printf 'cuda_visible_devices\t%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  printf 'source_sha256\t%s\n' "$BARN_SOURCE_SHA256"
  printf 'manifest_sha256\t%s\n' "$BARN_MANIFEST_SHA256"
  printf 'split_sha256\t%s\n' "$BARN_SPLIT_SHA256"
  printf 'prereg_sha256\t%s\n' "$BARN_PREREG_SHA256"
  printf 'analyzer_sha256\t%s\n' "$BARN_ANALYZER_SHA256"
  printf 'protocol_sha256\t%s\n' "$BARN_PROTOCOL_SHA256"
  printf 'container_sha256\t%s\n' "$BARN_CONTAINER_SHA256"
  printf 'domain_id\t%s\n' "$DOMAIN_ID"
  printf 'master_port\t%s\n' "$MASTER_PORT"
} > "$BLOCK/meta/job.tsv"

readonly PARTIAL="$BLOCK/tmp/seed-$SEED.json"
readonly FINAL="$BLOCK/results/seed-$SEED.json"
TIMEFORMAT=$'real_seconds\t%R\nuser_seconds\t%U\nsystem_seconds\t%S'
{
  time "$APPTAINER_BIN" exec --cleanenv \
  --bind "$SCRATCH_ROOT:$SCRATCH_ROOT" "$SIF" bash -s -- \
  "$SOURCE_BUNDLE" "$DATASET_ROOT" "$BLOCK" "$PARTIAL" "$SEED" \
  "$DOMAIN_ID" "$MASTER_PORT" "$ARMS" "$N_ROLLOUTS" \
  "$BARN_CAMPAIGN_ID" "$BARN_CAMPAIGN_CELL" "$BARN_ATTEMPT_ID" \
  "$BARN_SUBMITTED_UTC" "$SLURM_JOB_ID" "$SLURM_ARRAY_JOB_ID" \
  "$BARN_MANIFEST_SHA256" "$BARN_SPLIT_SHA256" "$BARN_PREREG_SHA256" \
  "$BARN_ANALYZER_SHA256" "$BARN_PROTOCOL_SHA256" \
  "$BARN_CONTAINER_SHA256" "$BARN_SOURCE_SHA256" \
  2> "$BLOCK/meta/runtime-stderr.txt" <<'CONTAINER'
set -eo pipefail
source_root="$1"; dataset_root="$2"; block="$3"; output="$4"; seed="$5"
domain_id="$6"; master_port="$7"; arms="$8"; n_rollouts="$9"
shift 9
campaign_id="$1"; campaign_cell="$2"; attempt_id="$3"; submitted_utc="$4"
slurm_job_id="$5"; slurm_array_job_id="$6"; manifest_sha="$7"; split_sha="$8"
prereg_sha="$9"; shift 9
analyzer_sha="$1"; protocol_sha="$2"; container_sha="$3"; source_sha="$4"

set +u
source /opt/ros/humble/setup.bash
set -u
export CUDA_VISIBLE_DEVICES=""
export LIBGL_ALWAYS_SOFTWARE=1
export PYTHONPATH="$source_root:$source_root/curriculum_maxrl${PYTHONPATH:+:$PYTHONPATH}"
cd "$source_root"
bash icra2027/build_gazebo_stepper.sh "$block/bin" \
  > "$block/meta/build.txt" 2>&1
test -x "$block/bin/gazebo_stepper"
test -f "$block/bin/libbarn_exact_step.so"
sha256sum "$block/bin/gazebo_stepper" "$block/bin/libbarn_exact_step.so" \
  > "$block/meta/binaries.sha256"

python3 -m icra2027.barn_campaign \
  --backend barn_gazebo \
  --dataset-root "$dataset_root" \
  --manifest icra2027/barn_manifest.jsonl \
  --split icra2027/barn_split.json \
  --robot-sdf icra2027/assets/barn_diff_drive.sdf \
  --stepper-path "$block/bin/gazebo_stepper" \
  --prereg icra2027/prereg_icra.md \
  --analyzer icra2027/analyze_campaign.py \
  --protocol icra2027/barn_protocol.json \
  --campaign-cell "$campaign_cell" \
  --seed "$seed" \
  --output "$output" \
  --runtime-root "$block/runtime" \
  --arms "$arms" \
  --steps 200 \
  --n-rollouts "$n_rollouts" \
  --tasks-per-step 2 \
  --eval-every 10 \
  --eval-episodes 1 \
  --training-sim-step-budget 1000000 \
  --eval-sim-step-interval 200000 \
  --episode-timeout 25.0 \
  --max-step-size 0.005 \
  --real-time-update-rate 2000 \
  --teacher-floor 0.1 \
  --teacher-decay 0.7 \
  --teacher-gamma 1.0 \
  --staged-initial-strata 1 \
  --staged-promotion-threshold 0.7 \
  --staged-min-frontier-groups 5 \
  --domain-id "$domain_id" \
  --master-port "$master_port" \
  --expected-manifest-sha256 "$manifest_sha" \
  --expected-split-sha256 "$split_sha" \
  --expected-prereg-sha256 "$prereg_sha" \
  --expected-analyzer-sha256 "$analyzer_sha" \
  --expected-protocol-sha256 "$protocol_sha" \
  --container-sha256 "$container_sha" \
  --source-sha256 "$source_sha" \
  --campaign-id "$campaign_id" \
  --attempt-id "$attempt_id" \
  --submitted-utc "$submitted_utc" \
  --slurm-job-id "$slurm_job_id" \
  --slurm-array-job-id "$slurm_array_job_id" \
  --slurm-array-task-id "$seed" \
  > "$block/meta/runner.txt" 2>&1
CONTAINER
} 2> "$BLOCK/meta/resource-time.txt"

# This validation reads identity/shape only and never prints endpoint values.
python3 - "$PARTIAL" "$BARN_CAMPAIGN_CELL" "$SEED" \
  "$BARN_CAMPAIGN_ID" "$BARN_ATTEMPT_ID" "$BARN_PROTOCOL_SHA256" <<'PY'
import json
import sys
from pathlib import Path
artifact = json.loads(Path(sys.argv[1]).read_text())
cell, seed, campaign, attempt, protocol_sha = (
    sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6])
expected_status = "full_barn_campaign" if cell == "primary" else "full_barn_n_ablation"
if artifact.get("evidence_status") != expected_status:
    raise SystemExit("unsafe evidence status")
if artifact.get("config", {}).get("campaign_cell") != cell:
    raise SystemExit("campaign cell mismatch")
if artifact.get("config", {}).get("seed_list") != [seed]:
    raise SystemExit("seed identity mismatch")
if artifact.get("provenance", {}).get("protocol_sha256") != protocol_sha:
    raise SystemExit("protocol provenance mismatch")
execution = artifact.get("execution", {})
if execution.get("campaign_id") != campaign or execution.get("attempt_id") != attempt:
    raise SystemExit("attempt identity mismatch")
PY
mv -- "$PARTIAL" "$FINAL"

printf 'finished_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >> "$BLOCK/meta/job.tsv"
(
  cd "$BLOCK"
  find . -type f ! -path './tmp/*' ! -name SHA256SUMS ! -name COMPLETE -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c --strict SHA256SUMS >/dev/null
)
{
  printf 'artifact_type\tbarn_evidence_seed_complete\n'
  printf 'campaign_id\t%s\n' "$BARN_CAMPAIGN_ID"
  printf 'campaign_cell\t%s\n' "$BARN_CAMPAIGN_CELL"
  printf 'attempt_id\t%s\n' "$BARN_ATTEMPT_ID"
  printf 'seed\t%s\n' "$SEED"
  printf 'completed_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'sha256sums_sha256\t%s\n' \
    "$(sha256sum "$BLOCK/SHA256SUMS" | awk '{print $1}')"
} > "$BLOCK/COMPLETE"
rm -rf -- "$BLOCK/tmp"
find "$BLOCK" -type f -exec chmod 0440 {} +
find "$BLOCK" -type d -exec chmod 0550 {} +

# Publish the entire checksum-closed seed block atomically and without
# replacement.  This is intentionally the final command: a failed job can
# retain its hidden work block without ever exposing a canonical partial
# artifact, while a successful Slurm task can only expose a complete block.
python3 - "$BLOCK" "$FINAL_BLOCK" <<'PY'
import ctypes
import errno
import os
import sys

source, destination = map(os.fsencode, sys.argv[1:3])
libc = ctypes.CDLL(None, use_errno=True)
renameat2 = getattr(libc, "renameat2", None)
if renameat2 is None:
    raise SystemExit("renameat2 unavailable; refusing non-atomic fallback")
renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p,
                      ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
renameat2.restype = ctypes.c_int
AT_FDCWD = -100
RENAME_NOREPLACE = 1
if renameat2(AT_FDCWD, source, AT_FDCWD, destination,
             RENAME_NOREPLACE) != 0:
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise SystemExit("evidence block destination already exists")
    raise OSError(error, os.strerror(error), os.fsdecode(destination))
PY
