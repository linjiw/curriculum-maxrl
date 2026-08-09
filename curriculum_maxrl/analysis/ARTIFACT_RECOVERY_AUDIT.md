# Raw-artifact recovery audit

Completed 2026-08-08, read-only.  No missing central raw artifact was
recoverable from this Mac.

## Search coverage

The audit inspected:

- every local and remote Git ref, reflog, both stashes, alternate worktrees,
  and every tree reachable from them;
- all unreachable Git objects: 310 commits, 997 trees, and 1,911 blobs in
  total, including all 173 unreachable objects and all 78 unreachable blobs;
- all eight local Git LFS objects, which are known Acrobot/MountainCar JSON
  artifacts;
- the main checkout, paper-revision worktree, sibling clones, Downloads,
  Documents, temporary directories, Codex caches, and plausible Ray,
  checkpoint, and dataset-cache paths;
- exact filename and content-signature matches across the home directory;
- local Time Machine/APFS snapshots and mounted backup volumes; and
- the public `tajwarfahim/maxrl` repository and its advertised remote refs.

No files were copied, altered, or deleted during this audit.

## Missing maze evidence

Still absent:

- all 24
  `fact250_{uniform,frontier_un}_{maxrl,grpo}_s{6..11}.jsonl` logs;
- all six `seed{6..11}_steps600_sft_warmstart.pt` files; and
- unpublished execution commit `9f7dd2e`.

The frozen multiverse correctly detects 0/24 logs.  The surviving factorial
artifact contains scalar cell/block summaries, not the checkpoint trajectories
at steps `-1,0,25,...,250`.  Those summaries cannot identify alternative AUC
integration conventions, warm-up cutoffs, early/mid/full windows, or
leave-one-checkpoint-out variants.  No resampling of the scalar summaries can
reconstruct those trajectories honestly.

## Missing Countdown evidence

Still absent:

- B1/B2 seed 1--3 task identities and all 16 verifier outcomes per task at
  step 60;
- SFT and evaluation manifests and compatible checkpoints; and
- execution commits `79473b2`, `2c95170`, and `ecdc461`.

The three-seed scoreboard is non-invertible.  Aggregate means and the VERL
bootstrap best@16 proxy do not determine which task produced which 16-bit
outcome vector.  They therefore cannot recover standard pass@16, the clean
101-task tier-0 reanalysis, task/bootstrap uncertainty, or paired task/seed
comparisons.

## Remaining external avenues

The only credible remaining source is the original Amazon Linux execution
machine or a backup of it:

- checkout `/home/ec2-user/work/curriculumrl/maxrl` and unpublished maze
  commit `9f7dd2e`;
- Ray sessions under `/tmp/ray/session_2026-07-28_*`,
  `/tmp/ray/session_2026-07-29_*`, and `/tmp/ray/session_2026-07-30_*`;
- checkpoints under `/home/ec2-user/ckpt/countdown_a10g/`; and
- attached EBS volumes/snapshots, an AMI, unpublished Git fork, S3, W&B, or
  another experiment-tracking upload.

If none survives, the missing analyses are irrecoverable and require new
runs.  The current maze/Countdown summaries must remain explicitly labeled
aggregate evidence rather than raw-artifact reproduction.
