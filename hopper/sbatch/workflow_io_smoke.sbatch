#!/usr/bin/env bash
#SBATCH --job-name=maxrl-io-smoke
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/%u/maxrl/tests/logs/%x_%j.out

# CPU-only smoke test for the Hopper submission and result-retrieval path.
# Submit through hopper/hopper.sh so the stdout parent exists before Slurm opens it.
set -euo pipefail
umask 077

: "${USER:?USER is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

test_root="/scratch/$USER/maxrl/tests"
result_dir="$test_root/results/$SLURM_JOB_ID"

mkdir -p -- "$test_root/logs" "$(dirname "$result_dir")"
if [[ -e "$result_dir" || -L "$result_dir" ]]; then
  printf 'refusing to overwrite existing smoke result: %s\n' "$result_dir" >&2
  exit 2
fi
mkdir -- "$result_dir"

utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
host=$(hostname)
printf 'key\tvalue\n' > "$result_dir/receipt.tsv"
printf 'job_id\t%s\n' "$SLURM_JOB_ID" >> "$result_dir/receipt.tsv"
printf 'utc\t%s\n' "$utc" >> "$result_dir/receipt.tsv"
printf 'user\t%s\n' "$USER" >> "$result_dir/receipt.tsv"
printf 'host\t%s\n' "$host" >> "$result_dir/receipt.tsv"
printf 'partition\t%s\n' "${SLURM_JOB_PARTITION:-unknown}" >> "$result_dir/receipt.tsv"
printf 'cpus_per_task\t%s\n' "${SLURM_CPUS_PER_TASK:-unknown}" >> "$result_dir/receipt.tsv"
printf 'submit_dir\t%s\n' "${SLURM_SUBMIT_DIR:-unknown}" >> "$result_dir/receipt.tsv"
printf 'bash\t%s\n' "$BASH_VERSION" >> "$result_dir/receipt.tsv"

printf 'Hopper workflow I/O smoke payload for job %s\n' "$SLURM_JOB_ID" \
  > "$result_dir/payload.txt"
(
  cd "$result_dir"
  sha256sum receipt.tsv payload.txt > SHA256SUMS
  sha256sum -c --strict SHA256SUMS
)

# COMPLETE is written last. Its presence means receipt creation and checksum
# verification both succeeded; transfer integrity is additionally checked by fetch.
printf 'complete\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$result_dir/COMPLETE"
printf 'WORKFLOW_IO_SMOKE_PASS\t%s\n' "$result_dir"
