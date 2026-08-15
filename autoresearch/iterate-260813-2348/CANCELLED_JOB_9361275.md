# Canceled Hopper engineering job 9361275

**Submitted:** 2026-08-13 14:25:30 America/New_York

**Canceled:** 2026-08-13 23:53 America/New_York, before allocation

**Final accounting:** `CANCELLED by 1224577940`, elapsed `00:00:00`
**Purpose:** MAZE-SCORE engineering smoke, seed 99; never paper evidence

## Scheduler receipt

- Account/QOS/partition: `xiao` / `gpu` / `gpuq`
- Request: one `3g.40gb` MIG, 8 CPUs, 60 GB, 40 minutes
- Command: `/scratch/lwang44/sbatch/maze_smoke.sbatch`
- Stdout target:
  `/scratch/lwang44/maze_score/logs/mazescore_s99_9361275.out`
- Sbatch SHA-256:
  `f332316e6608c760be1ac38aafa2177eb1672983eb6dcd25fa5b8c33c6270f59`
- Remote `train.py` SHA-256:
  `c69cbf931a9c6d50b835dff7f851129a066bd9b2afde77542fa1a5a98ba2735b`

## Outcome-blind cancellation reason

The job was pending for priority with a predicted start on 2026-08-14. A
clean-layout reproduction found that the script copied only
`curriculum_maxrl/maze_gpu/*.py`, while `train.py` imports the parent
`curriculum_maxrl/estimators.py`. The resulting staged layout raises
`ModuleNotFoundError: estimators`. The script also used `cp -n ... || true`,
which could retain stale source or hide a failed copy, and the Slurm stdout
parent did not exist before submission. It was canceled before consuming GPU
time. No experiment endpoint was produced or inspected.

This job is a failed infrastructure attempt, not a scientific run. It must not
be retried from the same mutable staging directory or entered in the paper run
registry.
