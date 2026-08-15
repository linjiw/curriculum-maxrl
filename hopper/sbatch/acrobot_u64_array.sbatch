#!/bin/bash -l
#SBATCH --job-name=acrobot-u64
#SBATCH --account=xiao
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --array=0-19%20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/lwang44/acrobot_u64/logs/acrobot-u64_%A_%a.out
#SBATCH --error=/scratch/lwang44/acrobot_u64/logs/acrobot-u64_%A_%a.out
#
# Confirmatory campaign for the frozen Acrobot U64 preregistration.
#
# One array task per LOGICAL SEED, running all four arms inside a single
# process on a single node.  This is deliberate: every primary estimand is a
# within-seed paired contrast, so pairing must not straddle nodes.  See
# AMENDMENT_2026-08-15_runtime_scope.md.
#
# Requires (set by hopper/submit_acrobot_u64.sh via --export):
#   ACROBOT_U64_BUNDLE_DIR     content-addressed staged bundle
#   ACROBOT_U64_BUNDLE_SHA256  expected manifest digest
set -euo pipefail

echo "=== acrobot-u64 task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} ==="
echo "host      : $(hostname -f)"
echo "started   : $(date -Is)"
echo "cpu       : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"

: "${ACROBOT_U64_BUNDLE_DIR:?bundle dir not exported}"
: "${ACROBOT_U64_BUNDLE_SHA256:?bundle digest not exported}"

BUNDLE="$ACROBOT_U64_BUNDLE_DIR"
[[ -d "$BUNDLE" ]] || { echo "missing bundle: $BUNDLE" >&2; exit 1; }

# --- fail closed on the exact staged bytes ---------------------------------
ACTUAL=$(cd "$BUNDLE" && find . -type f -not -name MANIFEST.sha256 -print0 \
  | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
if [[ "$ACTUAL" != "$ACROBOT_U64_BUNDLE_SHA256" ]]; then
  echo "BUNDLE DIGEST MISMATCH" >&2
  echo "  expected $ACROBOT_U64_BUNDLE_SHA256" >&2
  echo "  actual   $ACTUAL" >&2
  exit 1
fi
echo "bundle    : $BUNDLE"
echo "digest    : $ACTUAL (verified)"

SEED=$(( 20000 + SLURM_ARRAY_TASK_ID ))
if (( SEED < 20000 || SEED > 20019 )); then
  echo "seed $SEED is outside the frozen confirmatory block 20000-20019" >&2
  exit 2
fi
echo "seed      : $SEED"

RESULTS="/scratch/lwang44/acrobot_u64/results/${SLURM_ARRAY_JOB_ID}"
mkdir -p "$RESULTS"

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/scratch/lwang44/uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch/lwang44/uv-python
export UV_OFFLINE=1              # the cache was pre-warmed on the login node
# Keep the run single-threaded so 20 concurrent tasks do not oversubscribe and
# so BLAS threading cannot introduce run-to-run nondeterminism.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1

cd "$BUNDLE"

echo "--- vendored V2 source lock ---"
uv run --python 3.12.13 --with numpy==2.5.1 \
   --with 'gymnasium[classic-control]==1.3.0' \
   python verify_vendor_lock.py | tail -3

echo "--- arm sanity ---"
uv run --python 3.12.13 --with numpy==2.5.1 \
   --with 'gymnasium[classic-control]==1.3.0' \
   python check_arms_differ.py | tail -3

echo "--- confirmatory: all four arms, seed $SEED ---"
/usr/bin/time -f 'WALL %e s | MAXRSS %M KB' \
  uv run --python 3.12.13 --with numpy==2.5.1 \
     --with 'gymnasium[classic-control]==1.3.0' \
     python run_u64_tournament.py \
       --all-arms --seed "$SEED" --mode confirmatory --outdir "$RESULTS"

echo "--- produced ---"
ls -l "$RESULTS" | grep -- "-${SEED}.json" || true
COUNT=$(ls "$RESULTS" | grep -c -- "-${SEED}.json" || true)
if [[ "$COUNT" != "4" ]]; then
  echo "expected 4 arm files for seed $SEED, found $COUNT" >&2
  exit 1
fi

echo "finished  : $(date -Is)"
echo "ACROBOT_U64_TASK_COMPLETE seed=$SEED"
