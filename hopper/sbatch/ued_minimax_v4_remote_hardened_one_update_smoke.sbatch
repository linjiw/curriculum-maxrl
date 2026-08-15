#!/usr/bin/bash
#SBATCH --job-name=ued-v4h-one-update
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --export=NIL
#SBATCH --output=/scratch/%u/maxrl/tests/logs/%x_%j.out
set -euo pipefail
readonly SELF="$(readlink -f -- "$0")"
[[ $# == 3 && "$1" == --ued-input-envelope=* \
   && "$2" == --ued-bundle-dir=* && "$3" == --ued-submitted-sbatch=* ]] || {
  echo "exact v4h spool bootstrap arguments required" >&2; exit 2;
}
readonly BUNDLE=${2#--ued-bundle-dir=}
readonly SUBMITTED=${3#--ued-submitted-sbatch=}
[[ "$BUNDLE" == /* && "$BUNDLE" != *'/../'* && -d "$BUNDLE" && ! -L "$BUNDLE" \
   && "$(readlink -f -- "$BUNDLE")" == "$BUNDLE" ]] || { echo "unsafe bundle bootstrap" >&2; exit 2; }
[[ "$SUBMITTED" == "$BUNDLE/hopper/sbatch/ued_minimax_v4_remote_hardened_one_update_smoke.sbatch" \
   && -f "$SUBMITTED" && ! -L "$SUBMITTED" \
   && "$(readlink -f -- "$SUBMITTED")" == "$SUBMITTED" ]] || { echo "unsafe submitted sbatch bootstrap" >&2; exit 2; }
[[ -f "$SELF" && ! -L "$0" \
   && "$(sha256sum "$SELF" | awk '{print $1}')" == "$(sha256sum "$SUBMITTED" | awk '{print $1}')" ]] || {
  echo "Slurm spool sbatch byte drift" >&2; exit 2;
}
exec /usr/bin/bash "$BUNDLE/hopper/run_ued_minimax_v4_remote_hardened.sh" \
  one_update "$SUBMITTED" "$1"
