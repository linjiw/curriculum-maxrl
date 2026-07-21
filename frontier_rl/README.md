# frontier_rl — the curriculum-MaxRL schedule as a reusable framework

The validated training algorithm (see `../REPORT.md`), packaged so it can be
applied to gym environments, robotics simulators, or LLM prompt sets without
touching the core. numpy-only; no torch/gym dependency in the core.

## The algorithm (one screen)

```
teacher:   Beta(α,β) posterior per task (decay 0.7) → Thompson sample p̃
           → utility u(p̃) = (1−(1−p̃)^N) − p̃    [= MaxRL's expected advantage
             mass, proved exact; peak at p* ≈ ln N/N]
           → sample tasks ∝ u^γ  (γ≈4 if tasks share skills, 1 if independent)
           → mixed with a 10% uniform floor

estimator: MaxRL success-conditioned advantages  w_i = r_i/K − 1/N
           (K=0 groups dropped — this is what makes the teacher necessary)

hindsight: dead groups are relabeled by the ENV to the sub-goal actually
           achieved and trained as successes of that easier task
           (exact ML gradient where the env's relabel is exact — proved +
           measured; breaks the information ceiling of any pure sampler)
```

## Plugging in your environment

Implement `TaskSpace` (three methods) and `Policy` (one method) from
`interfaces.py`:

```python
class MyEnv:                                  # TaskSpace
    n_tasks: int                              # discrete task/goal bins
    def rollout_group(task_id, n) -> GroupResult   # N episodes, binary rewards
    def relabel(group) -> (task', rewards') | (task', rewards', trajs') | None

class MyPolicy:                               # Policy
    def update(task_id, trajectories, weights)     # one weighted PG step

trainer = FrontierTrainer(MyEnv(), MyPolicy(),
                          TrainerConfig(n_rollouts=16, hindsight=True))
trainer.train(steps=500)
```

**The two hindsight contracts** (from Proposition 6; violating either turns
the exact relabeled gradient into noise):

1. **Exactness** — a relabeled success must be a *true* success of the
   relabeled task under the env's own verifier.
2. **Conditioning** — if trajectories embed the goal (goal-relative features,
   `desired_goal` obs, goal tokens in a prompt), return rewritten
   trajectories with the achieved goal substituted. We measured the cost of
   skipping this on the gridworld: hindsight *hurts* without the rewrite
   (AUC 0.600 < teacher-only 0.658) and gives the best result with it
   (0.703). This is HER's observation-rewrite, surfaced as an interface
   contract.

## Adapters included

| adapter | what it shows | result (AUC, uniform → teacher → +hindsight) |
|---|---|---|
| `skill_chain` | regression anchor vs the validated testbed | 0.650 → 0.728 → **0.890** (matches REPORT.md) |
| `grid_reach` | goal-conditioned robotics pattern (goal bins = distance rings, relabel = reached cell, REINFORCE tabular policy) | 0.592 → 0.658 → **0.703** |
| `gym_goal` | gymnasium GoalEnv skeleton: where reset/step/is_success go, how to bin continuous goals, HER-style relabel via `achieved_goal` | (skeleton — bring your env) |

Run them:

```bash
python3 frontier_rl/test_framework.py            # unit tests
python3 frontier_rl/examples/run_skill_chain.py  # regression anchor (~2 min)
python3 frontier_rl/examples/run_grid_reach.py   # robotics-style demo (~3 min)
```

## Mapping to robotics / gym in practice

- **Task bins**: pick the axis your curriculum should walk (goal distance,
  obstacle count, object mass...). ~8–30 bins is plenty; the posterior needs
  a few groups per bin to localize the frontier.
- **Binary success**: use the env's own success predicate. Shaped rewards
  can coexist in your policy update; the *teacher* should only see binary
  outcomes (that is what the advantage-mass math assumes).
- **relabel**: gymnasium GoalEnvs give you `achieved_goal` for free — map it
  to its bin and rewrite `desired_goal` in the stored observations
  (contract 2). For non-goal envs with no meaningful relabel, return `None`;
  you keep the teacher benefits and lose only the hindsight term.
- **Group size N**: the teacher's band targets p ≈ ln N/N. N=16 targets
  ~17% success tasks; raise N to push the curriculum toward harder bins.
- **On/off-policy**: the schedule is estimator-agnostic at the interface
  level, but its guarantees are for the MaxRL weights; if you swap in a PPO
  update keep the weights as advantages and stay near-on-policy.

## What this does NOT do

- Continuous task spaces without binning (subclass the teacher with a
  parametric density — ALP-GMM-style — if you need it).
- Replace your RL optimizer: `Policy.update` is yours; this package decides
  *what to train on and with what advantage weights*, not how to descend.
- GRPO-style std-normalized advantages under a curriculum — measured to
  amplify coverage collapse (REPORT.md F2). Use the MaxRL weights.
