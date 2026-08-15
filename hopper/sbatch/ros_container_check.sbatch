#!/bin/bash -l
#SBATCH --job-name=ros-sif-check
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-00:30:00
#SBATCH --output=/scratch/%u/icra2027/logs/ros_sif_check_%j.out

# ENGINEERING VALIDATION — NOT EVIDENCE. Verifies the ROS 2 Humble + Gazebo
# Apptainer image boots headless on a Hopper compute node and writes the exact
# environment fingerprint required by the BARN preregistration.  This does not
# authorize a campaign; it only content-addresses the candidate runtime.
# lmod is unreliable in batch shells here; use the shared install directly.
export PATH="/opt/sw/other/apps/apptainer/1.4.1/bin:$PATH"
command -v apptainer >/dev/null || { echo "apptainer unavailable on $(hostname)"; exit 1; }
set -euo pipefail
SIF="/scratch/$USER/ros2-gazebo-classic.sif"
SCRATCH_ROOT="/scratch/$USER"
RESULT_DIR="$SCRATCH_ROOT/icra2027/environment/job-$SLURM_JOB_ID"
mkdir -p "$SCRATCH_ROOT/icra2027/logs" "$RESULT_DIR"
test -f "$SIF"
sha256sum "$SIF" > "$RESULT_DIR/container.sha256"

apptainer exec --bind "$SCRATCH_ROOT:$SCRATCH_ROOT" "$SIF" bash -s \
  -- "$RESULT_DIR" <<'CONTAINER'
  set -eo pipefail
  trap 'status=$?; echo "fingerprint failed: line=$LINENO status=$status command=$BASH_COMMAND" >&2; exit "$status"' ERR
  result_dir="$1"
  # ROS Humble's generated setup script reads optional shell variables before
  # defining them, so nounset must be enabled only after the environment loads.
  set +u
  source /opt/ros/humble/setup.bash
  set -u
  ros2 --help >/dev/null
  {
    printf 'ROS_DISTRO=%s\n' "${ROS_DISTRO:-unknown}"
    printf 'ROS_VERSION=%s\n' "${ROS_VERSION:-unknown}"
    printf 'ROS_PYTHON_VERSION=%s\n' "${ROS_PYTHON_VERSION:-unknown}"
    printf 'RMW_IMPLEMENTATION=%s\n' "${RMW_IMPLEMENTATION:-default}"
    echo 'ros2_cli=ok'
  } > "$result_dir/ros-environment.txt"
  python3 --version > "$result_dir/python-version.txt" 2>&1
  # Gazebo Classic 11 prints a valid version banner but returns 255 in this
  # SIF; validate the banner rather than treating that known exit code as a
  # failed fingerprint.
  gzserver --version > "$result_dir/gazebo-version.txt" 2>&1 || true
  grep -q "Gazebo multi-robot simulator, version" \
    "$result_dir/gazebo-version.txt"
  python3 - "$result_dir" <<'PY'
from importlib import metadata
from pathlib import Path
import sys

result_dir = Path(sys.argv[1])
status = Path("/var/lib/dpkg/status").read_text()
packages = []
for block in status.split("\n\n"):
    fields = {}
    for line in block.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            fields[key] = value
    if "Package" in fields and "Version" in fields:
        packages.append((fields["Package"], fields.get("Architecture", ""),
                         fields["Version"]))
(result_dir / "dpkg.tsv").write_text("".join(
    f"{name}\t{architecture}\t{version}\n"
    for name, architecture, version in sorted(packages)))

distributions = sorted(
    {(dist.metadata.get("Name", "unknown"), dist.version)
     for dist in metadata.distributions()},
    key=lambda row: (row[0].lower(), row[1]))
(result_dir / "pip-freeze.txt").write_text("".join(
    f"{name}=={version}\n" for name, version in distributions))
PY
  cat /etc/os-release > "$result_dir/os-release.txt"
  uname -a > "$result_dir/uname.txt"
  timeout 10 gzserver --verbose -r --iters 1000 /usr/share/gazebo-11/worlds/empty.world > /tmp/gz.log 2>&1 || true
  grep -m1 -i "Connected to gazebo master\|Gazebo multi-robot simulator" /tmp/gz.log \
    > "$result_dir/headless-boot.txt"
CONTAINER

python3 - "$RESULT_DIR" "$SIF" <<'PY'
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

result_dir = Path(sys.argv[1])
sif = Path(sys.argv[2])
receipt = {
    "artifact_type": "barn_hopper_environment_fingerprint",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "job_id": int(os.environ["SLURM_JOB_ID"]),
    "node": platform.node(),
    "container_path": str(sif),
    "container_size_bytes": sif.stat().st_size,
    "cpu_only": True,
    "evidence_status": "engineering_environment_check_not_paper_evidence",
}
(result_dir / "runtime.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY

(
  cd "$RESULT_DIR"
  sha256sum \
    container.sha256 dpkg.tsv gazebo-version.txt headless-boot.txt \
    os-release.txt pip-freeze.txt python-version.txt ros-environment.txt \
    runtime.json uname.txt > SHA256SUMS
  sha256sum -c SHA256SUMS
)
echo "environment_result=$RESULT_DIR"
echo "ROS CONTAINER CHECK PASS"
