#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${1:-$repo_root/icra2027/bin}"
mkdir -p "$output_dir"
g++ "$repo_root/icra2027/gazebo_stepper.cpp" \
  -O2 -Wall -Wextra -Wpedantic \
  $(pkg-config --cflags --libs gazebo) \
  -o "$output_dir/gazebo_stepper"
g++ "$repo_root/icra2027/barn_exact_step_plugin.cpp" \
  -std=c++17 -O2 -Wall -Wextra -Wpedantic -fPIC -shared \
  $(pkg-config --cflags --libs gazebo) \
  -o "$output_dir/libbarn_exact_step.so"
echo "built $output_dir/gazebo_stepper"
echo "built $output_dir/libbarn_exact_step.so"
