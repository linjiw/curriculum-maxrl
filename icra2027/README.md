# ICRA 2027 navigation campaign

This directory is the outcome-blind scaffold for the robotics track proposed
in `claude-fable-plan.md`.

## Domain decision

Use BARN-style mobile-robot navigation as the centerpiece, with the lab's
Jackal stack as the intended real-robot validation.  Goal-conditioned
hindsight is an optional arm inside this domain.  Isaac Lab terrain curricula
remain the fallback only if the BARN backend cannot complete one end-to-end
seed by August 17.

The BARN/Jackal packages and course assets are not present in this repository.
No result from `navigation_campaign.py` is a substitute: that script runs the
same comparison protocol on a small goal-conditioned grid adapter solely to
validate plumbing.

## What is ready

- estimator-derived, uniform, `p(1-p)` learnability, and staged-difficulty
  teachers with shared posterior bookkeeping;
- fixed held-out evaluation streams that do not mutate training randomness;
- per-difficulty success, easy retention, dead-group rate, episodes, simulator
  steps, and training wall time in one JSON artifact;
- exact greedy water-filling for a fixed rollout budget;
- a deterministic, difficulty-stratified environment-pool split generator;
- paired bootstrap and exact sign-flip analysis with an explicit five-seed
  and full-domain decision gate.

## Commands

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q icra2027/test_campaign.py

python3 -m icra2027.navigation_campaign \
  --seeds 1 --steps 60 --n-rollouts 16 \
  --output icra2027/results/navigation_smoke.json

python3 -m icra2027.analyze_campaign \
  icra2027/results/navigation_smoke.json

python3 -m icra2027.freeze_pool_split barn_pool.jsonl \
  --output icra2027/barn_split.json --seed 20270811
```

Before a full run, replace the grid adapter with a backend that consumes the
frozen BARN split and supplies binary success plus actual simulator step
counts.  Set `evidence_status` to `full_barn_campaign` only for that backend;
the analyzer refuses to issue an August 24 decision from a smoke artifact.

