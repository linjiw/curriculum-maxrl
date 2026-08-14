#!/usr/bin/env bash
#SBATCH --job-name=barn-train-smoke
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/maxrl/barn/logs/%x_%j.out

# ENGINEERING SMOKE — NEVER PAPER EVIDENCE.  This CPU-only job builds the
# exact-step Gazebo helper/plugin from a job-local copy of the pinned source,
# then executes one bounded real policy update on barn-299.  The selected
# course is required to belong to the frozen TRAIN partition and is reused by
# the isolated smoke evaluator; no held-out course is loaded.  The detailed
# campaign artifact stays in disposable job-local work.  Only redacted
# provenance/completion and resource-timing receipts are published.
set -euo pipefail
umask 027
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=""

readonly EXPECTED_ARCHIVE_SHA256="5ad443412f6f2f38b6d0e1d330c9a820ab48e566553197459005e751711fe320"
readonly EXPECTED_CONTAINER_SHA256="cd6620e33c0822f7d6a03c6de6ea9dd4304f0927e8d7997c003560f5b4781be0"
readonly ALLOWED_ENGINEERING_COURSE="barn-299"

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
ENGINEERING_COURSE="${BARN_TRAIN_COURSE:-$ALLOWED_ENGINEERING_COURSE}"
readonly ENGINEERING_COURSE
[[ "$ENGINEERING_COURSE" == "$ALLOWED_ENGINEERING_COURSE" ]] || {
  echo "this prospective smoke permits only frozen TRAIN course barn-299" >&2
  exit 2
}

RUN_USER=$(id -un)
readonly RUN_USER
readonly SCRATCH_ROOT="/scratch/$RUN_USER"
readonly APPTAINER_BIN="/opt/sw/other/apps/apptainer/1.4.1/bin/apptainer"
readonly SIF="$SCRATCH_ROOT/ros2-gazebo-classic.sif"
readonly ARCHIVE_PREFIX="${EXPECTED_ARCHIVE_SHA256:0:20}"
readonly EXPECTED_ARCHIVE_PATH="$SCRATCH_ROOT/maxrl/bundles/barn_dataset/$ARCHIVE_PREFIX/BARN_dataset.zip"
readonly DATASET_PACKAGE="$SCRATCH_ROOT/maxrl/barn/datasets/$EXPECTED_ARCHIVE_SHA256"
readonly DATASET_ROOT="$DATASET_PACKAGE/BARN_dataset"
readonly RESULT_ROOT="$SCRATCH_ROOT/maxrl/tests/barn-training-smoke"
readonly RESULT_DIR="$RESULT_ROOT/$SLURM_JOB_ID"
readonly WORK_PARENT="$SCRATCH_ROOT/maxrl/barn/tmp"

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
# Do not let a checksum entry escape the staged bundle before strict
# verification follows it.  The staging helper emits ordinary sha256sum
# records and does not permit symbolic source entries.
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

readonly BUNDLED_SBATCH="$SOURCE_BUNDLE/hopper/sbatch/barn_training_smoke.sbatch"
[[ -f "$BUNDLED_SBATCH" && ! -L "$BUNDLED_SBATCH" ]] || {
  echo "bundled training-smoke sbatch is missing or symbolic" >&2
  exit 2
}
RUNNING_SBATCH=$(readlink -f -- "$0")
readonly RUNNING_SBATCH
[[ -f "$RUNNING_SBATCH" && ! -L "$RUNNING_SBATCH" \
   && "$(sha256sum "$RUNNING_SBATCH" | awk '{print $1}')" == \
      "$(sha256sum "$BUNDLED_SBATCH" | awk '{print $1}')" ]] || {
  echo "running training-smoke sbatch differs from immutable source closure" >&2
  exit 2
}

source_required=(
  icra2027/barn_campaign.py
  icra2027/verify_barn_smoke_package.py
  icra2027/barn_manifest.jsonl
  icra2027/barn_split.json
  icra2027/analyze_campaign.py
  icra2027/prereg_icra.md
  icra2027/barn_protocol.json
  icra2027/build_gazebo_stepper.sh
  icra2027/gazebo_stepper.cpp
  icra2027/barn_exact_step_plugin.cpp
  icra2027/assets/barn_diff_drive.sdf
  frontier_rl/adapters/barn_gazebo.py
)
for relative in "${source_required[@]}"; do
  [[ -f "$SOURCE_BUNDLE/$relative" && ! -L "$SOURCE_BUNDLE/$relative" ]] || {
    printf 'required source file missing or symbolic: %s\n' "$relative" >&2
    exit 2
  }
done

[[ "$BARN_DATASET_ARCHIVE" == "$EXPECTED_ARCHIVE_PATH" ]] || {
  echo "dataset archive is not at its canonical content-addressed path" >&2
  exit 2
}

[[ -d "$DATASET_PACKAGE" && ! -L "$DATASET_PACKAGE" \
   && -d "$DATASET_ROOT" && ! -L "$DATASET_ROOT" \
   && -f "$DATASET_PACKAGE/PREPARE_RECEIPT.json" \
   && -f "$DATASET_PACKAGE/SHA256SUMS" \
   && -f "$DATASET_PACKAGE/COMPLETE" ]] || {
  echo "complete immutable prepared dataset is unavailable" >&2
  exit 2
}

# Dataset preparation already verified every archive member and published a
# checksum-closed immutable package.  This training-only smoke must not reread
# the prospective held-out filesystem, so authenticate only package controls
# and the selected train row's declarations here.  The runner below is the sole
# code that opens and hashes that selected train row's real assets.
python3 "$SOURCE_BUNDLE/icra2027/verify_barn_smoke_package.py" \
  --package "$DATASET_PACKAGE" \
  --manifest "$SOURCE_BUNDLE/icra2027/barn_manifest.jsonl" \
  --course-id "$ENGINEERING_COURSE"

export BARN_SMOKE_PREP_RECEIPT="$DATASET_PACKAGE/PREPARE_RECEIPT.json"
export BARN_SMOKE_SOURCE_SHA="$BARN_SOURCE_SHA256"
export BARN_SMOKE_ARCHIVE_SHA="$EXPECTED_ARCHIVE_SHA256"
export BARN_SMOKE_CONTAINER_SHA="$EXPECTED_CONTAINER_SHA256"
export BARN_SMOKE_MANIFEST_SHA
BARN_SMOKE_MANIFEST_SHA=$(sha256sum "$SOURCE_BUNDLE/icra2027/barn_manifest.jsonl" | awk '{print $1}')
export BARN_SMOKE_SPLIT_SHA
BARN_SMOKE_SPLIT_SHA=$(sha256sum "$SOURCE_BUNDLE/icra2027/barn_split.json" | awk '{print $1}')
export BARN_SMOKE_COURSE="$ENGINEERING_COURSE"
export BARN_SMOKE_SPLIT="$SOURCE_BUNDLE/icra2027/barn_split.json"
python3 - <<'PY'
import json
import os
from pathlib import Path

receipt = json.loads(Path(os.environ["BARN_SMOKE_PREP_RECEIPT"]).read_text())
expected = {
    "archive_sha256": os.environ["BARN_SMOKE_ARCHIVE_SHA"],
    "container_sha256": os.environ["BARN_SMOKE_CONTAINER_SHA"],
    "barn_manifest_sha256": os.environ["BARN_SMOKE_MANIFEST_SHA"],
    "barn_split_sha256": os.environ["BARN_SMOKE_SPLIT_SHA"],
}
for field, value in expected.items():
    if receipt.get(field) != value:
        raise SystemExit(f"prepared dataset/{field} mismatch")
split = json.loads(Path(os.environ["BARN_SMOKE_SPLIT"]).read_text())
course = os.environ["BARN_SMOKE_COURSE"]
if course not in split.get("train_ids", []):
    raise SystemExit("engineering course is not in frozen TRAIN partition")
if course in split.get("heldout_ids", []):
    raise SystemExit("engineering course appears in frozen heldout partition")
PY

mkdir -p -- "$RESULT_ROOT" "$WORK_PARENT"
[[ ! -e "$RESULT_DIR" && ! -L "$RESULT_DIR" ]] || {
  printf 'refusing to overwrite smoke output: %s\n' "$RESULT_DIR" >&2
  exit 2
}
mkdir -- "$RESULT_DIR"
WORK_DIR=$(mktemp -d "$WORK_PARENT/.job-${SLURM_JOB_ID}.XXXXXXXX")
readonly WORK_DIR
readonly JOB_SOURCE="$WORK_DIR/source"
readonly CAMPAIGN_ARTIFACT="$WORK_DIR/internal-smoke-artifact.json"
readonly RUNTIME_DIR="$WORK_DIR/runtime"
WORK_ACTIVE=1
cleanup() {
  if [[ "${WORK_ACTIVE:-0}" == 1 && -n "${WORK_DIR:-}" \
        && "$WORK_DIR" == "$WORK_PARENT/.job-${SLURM_JOB_ID}."* \
        && -d "$WORK_DIR" ]]; then
    chmod -R u+w -- "$WORK_DIR" 2>/dev/null || true
    rm -rf -- "$WORK_DIR"
  fi
}
trap cleanup EXIT INT TERM
mkdir -- "$JOB_SOURCE"
cp -a -- "$SOURCE_BUNDLE/." "$JOB_SOURCE/"
[[ "$(sha256sum "$JOB_SOURCE/SHA256SUMS" | awk '{print $1}')" == "$BARN_SOURCE_SHA256" ]] || {
  echo "job-local source manifest hash mismatch" >&2
  exit 2
}
(cd "$JOB_SOURCE" && sha256sum -c --strict SHA256SUMS >/dev/null)
chmod u+w -- "$JOB_SOURCE" "$JOB_SOURCE/icra2027"

MANIFEST_SHA=$(sha256sum "$JOB_SOURCE/icra2027/barn_manifest.jsonl" | awk '{print $1}')
SPLIT_SHA=$(sha256sum "$JOB_SOURCE/icra2027/barn_split.json" | awk '{print $1}')
ANALYZER_SHA=$(sha256sum "$JOB_SOURCE/icra2027/analyze_campaign.py" | awk '{print $1}')
PREREG_SHA=$(sha256sum "$JOB_SOURCE/icra2027/prereg_icra.md" | awk '{print $1}')
PROTOCOL_SHA=$(sha256sum "$JOB_SOURCE/icra2027/barn_protocol.json" | awk '{print $1}')
readonly MANIFEST_SHA SPLIT_SHA ANALYZER_SHA PREREG_SHA PROTOCOL_SHA
SMOKE_SEED="${BARN_SMOKE_SEED:-20270811}"
[[ "$SMOKE_SEED" =~ ^[0-9]+$ ]] || {
  echo "BARN_SMOKE_SEED must be a non-negative integer" >&2
  exit 2
}
DOMAIN_ID=$((70 + SLURM_JOB_ID % 50))
MASTER_PORT=$((12000 + SLURM_JOB_ID % 1000))
readonly SMOKE_SEED DOMAIN_ID MASTER_PORT

started_epoch=$(date +%s)
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
TIMEFORMAT=$'real_seconds\t%R\nuser_seconds\t%U\nsystem_seconds\t%S'
{
  time "$APPTAINER_BIN" exec --cleanenv \
  --bind "$SCRATCH_ROOT:$SCRATCH_ROOT" "$SIF" bash -s -- \
  "$JOB_SOURCE" "$DATASET_ROOT" "$CAMPAIGN_ARTIFACT" "$RUNTIME_DIR" \
  "$ENGINEERING_COURSE" "$SMOKE_SEED" "$DOMAIN_ID" "$MASTER_PORT" \
  "$MANIFEST_SHA" "$SPLIT_SHA" "$PREREG_SHA" "$ANALYZER_SHA" \
  "$PROTOCOL_SHA" "$EXPECTED_CONTAINER_SHA256" "$BARN_SOURCE_SHA256" \
  "$RESULT_DIR/build.txt" "$RESULT_DIR/runner.txt" \
  2> "$RESULT_DIR/runtime-stderr.txt" <<'CONTAINER'
set -eo pipefail
source_root="$1"
dataset_root="$2"
artifact="$3"
runtime_root="$4"
course="$5"
seed="$6"
domain_id="$7"
master_port="$8"
manifest_sha="$9"
shift 9
split_sha="$1"
prereg_sha="$2"
analyzer_sha="$3"
protocol_sha="$4"
container_sha="$5"
source_sha="$6"
build_log="$7"
runner_log="$8"

# ROS Humble setup reads unset optional variables.  Enable nounset only after
# loading it, then make accelerator absence explicit inside the clean env.
set +u
source /opt/ros/humble/setup.bash
set -u
export CUDA_VISIBLE_DEVICES=""
export LIBGL_ALWAYS_SOFTWARE=1
export PYTHONPATH="$source_root:$source_root/curriculum_maxrl${PYTHONPATH:+:$PYTHONPATH}"
cd "$source_root"
bash icra2027/build_gazebo_stepper.sh > "$build_log" 2>&1
test -x icra2027/bin/gazebo_stepper
test -f icra2027/bin/libbarn_exact_step.so

python3 -m icra2027.barn_campaign \
  --backend barn_gazebo \
  --dataset-root "$dataset_root" \
  --manifest icra2027/barn_manifest.jsonl \
  --split icra2027/barn_split.json \
  --robot-sdf icra2027/assets/barn_diff_drive.sdf \
  --stepper-path icra2027/bin/gazebo_stepper \
  --prereg icra2027/prereg_icra.md \
  --analyzer icra2027/analyze_campaign.py \
  --protocol icra2027/barn_protocol.json \
  --seed "$seed" \
  --output "$artifact" \
  --runtime-root "$runtime_root" \
  --smoke \
  --engineering-course-id "$course" \
  --smoke-update-budget \
  --arms ours_uN \
  --steps 1 \
  --n-rollouts 8 \
  --tasks-per-step 2 \
  --eval-every 1 \
  --eval-episodes 1 \
  --domain-id "$domain_id" \
  --master-port "$master_port" \
  --expected-manifest-sha256 "$manifest_sha" \
  --expected-split-sha256 "$split_sha" \
  --expected-prereg-sha256 "$prereg_sha" \
  --expected-analyzer-sha256 "$analyzer_sha" \
  --expected-protocol-sha256 "$protocol_sha" \
  --container-sha256 "$container_sha" \
  --source-sha256 "$source_sha" > "$runner_log" 2>&1
CONTAINER
} 2> "$RESULT_DIR/resource-time.txt"
finished_epoch=$(date +%s)
finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Validate completion internally, but deliberately do not publish success,
# reward, AUC, trajectory, or any other metric-bearing endpoint field.
export BARN_SMOKE_ARTIFACT="$CAMPAIGN_ARTIFACT"
export BARN_SMOKE_RECEIPT="$RESULT_DIR/ENGINEERING_RECEIPT.json"
export BARN_SMOKE_SOURCE="$SOURCE_BUNDLE"
export BARN_SMOKE_DATASET="$DATASET_ROOT"
export BARN_SMOKE_SIF="$SIF"
export BARN_SMOKE_MANIFEST_SHA="$MANIFEST_SHA"
export BARN_SMOKE_SPLIT_SHA="$SPLIT_SHA"
export BARN_SMOKE_PREREG_SHA="$PREREG_SHA"
export BARN_SMOKE_ANALYZER_SHA="$ANALYZER_SHA"
export BARN_SMOKE_PROTOCOL_SHA="$PROTOCOL_SHA"
export BARN_SMOKE_CONTAINER_SHA="$EXPECTED_CONTAINER_SHA256"
export BARN_SMOKE_STARTED="$started_utc"
export BARN_SMOKE_FINISHED="$finished_utc"
export BARN_SMOKE_ELAPSED="$((finished_epoch - started_epoch))"
export BARN_SMOKE_DOMAIN_ID="$DOMAIN_ID"
export BARN_SMOKE_MASTER_PORT="$MASTER_PORT"
export BARN_SMOKE_STEPPER="$JOB_SOURCE/icra2027/bin/gazebo_stepper"
export BARN_SMOKE_PLUGIN="$JOB_SOURCE/icra2027/bin/libbarn_exact_step.so"
python3 - <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path

artifact = json.loads(Path(os.environ["BARN_SMOKE_ARTIFACT"]).read_text())
course = os.environ["BARN_SMOKE_COURSE"]
if artifact.get("evidence_status") != "engineering_smoke_not_paper_evidence":
    raise SystemExit("smoke artifact has an unsafe evidence status")
config = artifact.get("config", {})
if config.get("arms") != ["ours_uN"] or config.get("steps") != 1:
    raise SystemExit("smoke did not use the one-arm/one-update contract")
if config.get("n_rollouts") != 8 or config.get("tasks_per_step") != 2:
    raise SystemExit("smoke rollout shape mismatch")
if (config.get("engineering_course_id") != course
        or config.get("evaluation_partition")
        != "training_course_engineering_smoke"):
    raise SystemExit("smoke did not retain train-only course isolation")
runs = artifact.get("results", {}).get("ours_uN", [])
if len(runs) != 1 or [row.get("step") for row in runs[0].get("history", [])] != [0, 1]:
    raise SystemExit("smoke did not finish exactly one update")
final = runs[0].get("final", {})
training_sim_steps = final.get("sim_steps")
training_episodes = final.get("episodes")
training_wall_seconds = final.get("training_wall_seconds")
evaluation_wall_seconds = final.get("evaluation_wall_seconds")
if (not isinstance(training_sim_steps, int) or training_sim_steps <= 0
        or training_episodes != 16
        or not isinstance(training_wall_seconds, (int, float))
        or training_wall_seconds <= 0
        or not isinstance(evaluation_wall_seconds, (int, float))
        or evaluation_wall_seconds <= 0):
    raise SystemExit("smoke resource counters are incomplete")

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

receipt = {
    "artifact_type": "barn_hopper_training_smoke_receipt",
    "evidence_status": "engineering_smoke_not_paper_evidence",
    "started_utc": os.environ["BARN_SMOKE_STARTED"],
    "finished_utc": os.environ["BARN_SMOKE_FINISHED"],
    "elapsed_seconds": int(os.environ["BARN_SMOKE_ELAPSED"]),
    "job_id": int(os.environ["SLURM_JOB_ID"]),
    "node": platform.node(),
    "cpu_only": True,
    "source_bundle": os.environ["BARN_SMOKE_SOURCE"],
    "source_manifest_sha256": os.environ["BARN_SMOKE_SOURCE_SHA"],
    "dataset_root": os.environ["BARN_SMOKE_DATASET"],
    "archive_sha256": os.environ["BARN_SMOKE_ARCHIVE_SHA"],
    "container_path": os.environ["BARN_SMOKE_SIF"],
    "container_sha256": os.environ["BARN_SMOKE_CONTAINER_SHA"],
    "barn_manifest_sha256": os.environ["BARN_SMOKE_MANIFEST_SHA"],
    "barn_split_sha256": os.environ["BARN_SMOKE_SPLIT_SHA"],
    "preregistration_sha256": os.environ["BARN_SMOKE_PREREG_SHA"],
    "analyzer_sha256": os.environ["BARN_SMOKE_ANALYZER_SHA"],
    "protocol_sha256": os.environ["BARN_SMOKE_PROTOCOL_SHA"],
    "engineering_course_id": course,
    "course_partition": "train",
    "heldout_courses_read": False,
    "arm": "ours_uN",
    "rollouts_per_group": 8,
    "tasks_per_update": 2,
    "completed_updates": 1,
    "training_sim_steps": training_sim_steps,
    "training_episodes": training_episodes,
    "training_wall_seconds": float(training_wall_seconds),
    "evaluation_wall_seconds": float(evaluation_wall_seconds),
    "resource_counters_only": True,
    "exact_step_build": {
        "gazebo_stepper_sha256": digest(os.environ["BARN_SMOKE_STEPPER"]),
        "plugin_sha256": digest(os.environ["BARN_SMOKE_PLUGIN"]),
    },
    "domain_id": int(os.environ["BARN_SMOKE_DOMAIN_ID"]),
    "master_port": int(os.environ["BARN_SMOKE_MASTER_PORT"]),
    "paper_endpoint_emitted": False,
    "internal_metric_artifact_retained": False,
}
Path(os.environ["BARN_SMOKE_RECEIPT"]).write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

(
  cd "$RESULT_DIR"
  find . -type f ! -name SHA256SUMS ! -name COMPLETE -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c --strict SHA256SUMS >/dev/null
)
cleanup
WORK_ACTIVE=0
trap - EXIT INT TERM
printf 'complete\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RESULT_DIR/COMPLETE"
printf 'BARN_TRAINING_SMOKE_PASS\t%s\n' "$RESULT_DIR"
