# Goal: strengthen the paper through a verified Hopper evidence loop

**Started:** 2026-08-13 23:48 America/New_York

**Iteration bound:** 25 build/verify/keep decisions

**Remote account:** `lwang44@hopper.orc.gmu.edu`

**Primary paper:** `paper/body_iclr.tex`

## Objective

Verify and make unambiguous the method-to-code contract for the
estimator-derived score, establish a safe and reproducible GMU Hopper workflow,
run and retrieve an engineering smoke job, and make MAZE-SCORE launch-ready
without inspecting or launching any evidence-bearing endpoint before its
protocol and exact selected code path are frozen.

## Primary metric

The evidence loop is ready only if all of the following are true:

1. Every implementation selected by future evidence computes exactly
   `u_N(p) = 1 - (1-p)^N - p` for configured rollout count `N`; in particular,
   MAZE-SCORE's `frontier_un` dispatch is verified rather than inferred.
2. Cross-implementation tests include `u_2=p(1-p)` and distinguish the exact
   path from the retained historical `u_(N+1)` implementation.
3. The MAZE-SCORE source bundle, analysis, environment, and Slurm request are
   content-addressed before its preregistration freezes.
4. A Hopper smoke job reaches terminal `COMPLETED` with exit code 0, records the
   assigned GPU/runtime/source hashes, and its logs are fetched back into this
   iteration directory.
5. A one-command status/fetch path can recover a run after the initiating shell
   exits; no result exists only in `/scratch`.

## Secondary scientific metric

The candidate design now uses 30 paired blocks (seeds 20--49), subject to an
outcome-blind cost freeze after the full-arm engineering smoke. Its primary is
`u_32 - u_2` time-integrated held-out coverage AUC. Support requires mean at
least +.005, bootstrap lower bound above zero, and Holm-adjusted exact
sign-flip `p < .05`; practical exclusion requires the interval upper bound
below +.005, with every other nonsupport result labeled inconclusive. This
iteration does not inspect that endpoint or select the count from new outcomes.

## Safety and scope gates

- E2c never runs on Hopper; its local runtime and GPU gate remain frozen.
- Do not submit candidate MAZE-SCORE seeds 20--49 until the verified exact
  source path, smoke receipts, analyzer, environment, retry policy, and
  preregistration are frozen together.
- Engineering smoke seed 99 is outside 20--49 and never enters paper inference.
- Do not cancel, reprioritize, or alter other users' jobs.
- Do not modify a frozen preregistration except through a dated, outcome-blind
  amendment.
- Do not inspect partial MAZE-SCORE outcomes.
- Do not git-push, publish, or deploy paper artifacts in this iteration.
- Preserve all pre-existing dirty-worktree files and unrelated user changes.

## Official Hopper rules adopted

- Submit jobs from a login node with `sbatch`; compute workloads run under Slurm.
- GPU jobs use `--partition=gpuq --qos=gpu` and request the smallest suitable
  device.
- `3g.40gb` jobs request at most 8 CPUs and 60 GB RAM; `1g.10gb` smoke jobs use
  2 CPUs and 15 GB RAM.
- Use `/scratch/lwang44` for execution, but fetch evidence immediately because
  scratch is not backed up and files older than 90 days are purged.
- Track queue state with `squeue`, terminal/accounting state with `sacct`, and
  GPU utilization with `nvidia-smi` on the assigned compute node.
- Keep file counts modest and bundle small artifacts where practical.

## Planned decision sequence

1. Audit official documentation, SSH access, current queue, storage, modules,
   environment, and existing remote files.
2. Audit and harden the formula/dispatch contract locally with tests.
3. Harden `hopper.sh`, setup checks, Slurm receipts, and deterministic retrieval.
4. Stage only the smoke/source bundle needed for validation.
5. Submit a bounded smoke job, wait for terminal status, fetch and verify it.
6. Freeze or block MAZE-SCORE based on the predeclared readiness metric.
7. Update `STATUS.md`, `RESULTS.md`, `handoff.json`, and the design plan.
