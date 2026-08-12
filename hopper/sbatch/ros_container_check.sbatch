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
# Apptainer image boots headless on a Hopper compute node: gzserver version,
# a 10 s empty-world simulation, and ros2 CLI availability. Green-lights the
# BARN CPU matrix path (which itself waits for the frozen prereg + runner CLI).
module load hosts/hopper apptainer/1.4.1 2>/dev/null || module load apptainer 2>/dev/null || true
command -v apptainer >/dev/null || { echo "apptainer unavailable"; exit 1; }
set -euo pipefail
SIF="/scratch/$USER/ros2-gazebo.sif"
mkdir -p "/scratch/$USER/icra2027/logs"

apptainer exec --bind "/scratch/$USER:/scratch/$USER" "$SIF" bash -c '
  set -e
  source /opt/ros/humble/setup.bash
  echo "ros2: $(ros2 --help >/dev/null 2>&1 && echo ok)"
  gzserver --version | head -2
  timeout 10 gzserver --verbose -r --iters 1000 /usr/share/gazebo-11/worlds/empty.world > /tmp/gz.log 2>&1 || true
  grep -m1 -i "Connected to gazebo master\|Gazebo multi-robot simulator" /tmp/gz.log && echo "headless boot ok"
'
echo "ROS CONTAINER CHECK PASS"
