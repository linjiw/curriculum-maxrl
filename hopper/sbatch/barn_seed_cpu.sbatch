#!/bin/bash
#SBATCH --job-name=barn-seed
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/%u/icra2027/logs/barn_seed%a_%j.out
#SBATCH --array=1-5

# One ICRA campaign seed (all four sampler arms) per array index, CPU-only,
# inside the ROS 2 Humble + Gazebo Apptainer container.
#
# PREREQUISITES (one-time, on the login node):
#   1. Build the container (definition mirrors the lab machine's stack):
#        apptainer build $SCRATCH/ros2-gazebo.sif docker://osrf/ros:humble-desktop
#      then verify `apptainer exec ros2-gazebo.sif gzserver --version` matches
#      the lab Gazebo major version recorded in the prereg environment
#      fingerprint. Any mismatch must be recorded in the prereg BEFORE launch.
#   2. rsync the repo + BARN dataset root to /scratch/$USER/:
#        rsync -av --exclude .git /home/robotixx/curriculum-maxrl hopper:/scratch/$USER/
#        rsync -av <BARN_DATASET_ROOT> hopper:/scratch/$USER/barn_dataset/
#   3. The frozen prereg + split + analyzer commit must already exist; this
#      script only runs seeds, it never edits protocol.
#   4. TEMPLATE: the python invocation below assumes flags the campaign runner
#      may not have yet — align it with the finalized runner CLI (codex owns
#      that interface) before first submission.
#
# GOVERNANCE: transition-matched accounting is the primary budget on Hopper by
# the prereg's shared-hardware clause; wall-clock is descriptive only.
set -euo pipefail

SCRATCH="/scratch/$USER"
SEED="${SLURM_ARRAY_TASK_ID}"
SIF="$SCRATCH/ros2-gazebo.sif"
REPO="$SCRATCH/curriculum-maxrl"
mkdir -p "$SCRATCH/icra2027/logs"

apptainer exec --bind "$SCRATCH:$SCRATCH" "$SIF" bash -c "
  source /opt/ros/humble/setup.bash
  export PYTHONPATH=$REPO/curriculum_maxrl:$REPO
  export ROS_DOMAIN_ID=\$((100 + $SEED))
  cd $REPO
  python3 -m icra2027.navigation_campaign \
    --backend barn_gazebo \
    --dataset-root $SCRATCH/barn_dataset \
    --split icra2027/barn_split.json \
    --seed $SEED \
    --out $SCRATCH/icra2027/results/full_seed$SEED
"
echo "DONE seed $SEED"
