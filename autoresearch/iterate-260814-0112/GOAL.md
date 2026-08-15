# Goal: source-faithful UED benchmark and FrontierRL improvement loop

**Started:** 2026-08-14 01:12 America/New_York

**Iteration bound:** 25 keep/discard decisions

**Compute:** local RTX 5090 for fast engineering; GMU Hopper for isolated,
receipt-backed replication and parallel seeds

## Objective

Reproduce the strongest relevant Prioritized Level Replay and `minimax`
navigation baselines, integrate rollout-aware coefficient activity without
changing their student, environment, budget, or evaluator, and determine
whether FrontierRL improves held-out navigation performance under matched
environment interactions and wall-clock reporting.

## Candidate primary benchmark

Use `minimax` AMaze because it is the maintained, accelerated implementation
of the MiniGrid UED navigation benchmark and contains DR, PAIRED, PLR, robust
PLR, parallel PLR, and ACCEL in one evaluator. Freeze the exact upstream
revision, environment, official training config, held-out levels, interaction
budget, update accounting, seeds, and metric before any confirmatory run.

## Success contract

1. Upstream DR and PLR commands complete from a pinned environment and produce
   source-format logs/checkpoints/evaluation output.
2. A FrontierRL arm changes only the level priority/acquisition score; PPO,
   architecture, rollout budget, replay probability, staleness term, and
   evaluator remain matched unless separately ablated.
3. Engineering improvements are selected using fixed smoke/development seeds;
   confirmatory seeds and held-out endpoints remain sealed until the protocol
   is frozen.
4. A claim of beating a baseline requires paired multi-seed held-out results at
   the same environment-step budget, uncertainty intervals, multiplicity-aware
   inference, and compute/resource accounting. A single fast seed is only an
   engineering result.
5. Every retained run has source, environment, arguments, seed, job/device,
   stdout, checkpoint, raw evaluation, and checksum receipts saved outside
   ephemeral scratch.

## Safety and scope

- Do not push, publish, or modify upstream repositories.
- Do not inspect partial confirmatory endpoints.
- Do not disturb unrelated local GPU users or other Hopper jobs.
- Do not submit expensive Hopper arrays until local source-faithful smoke and
  one bounded Hopper import/runtime smoke pass.
- Preserve all existing dirty-worktree and concurrent BARN changes.
- Respect upstream licenses; the archived Level Replay code is
  non-commercial, while `minimax` is Apache-2.0.
