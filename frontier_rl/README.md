# frontier_rl — the curriculum-MaxRL schedule as a reusable framework

The validated training algorithm (see `../REPORT.md`), packaged so it can be
applied to gym environments, robotics simulators, or LLM prompt sets without
touching the core. numpy-only; no torch/gym dependency in the core.

## The algorithm (one screen)

```
teacher:   discounted Beta pseudo-counts per task → Thompson-style draw p̃
           → utility u(p̃) = (1−(1−p̃)^N) − p̃    [= MaxRL's expected advantage
             mass divided by two; peak at p* ≈ ln N/N]
           → sample tasks ∝ u^γ  (γ=1 conservative; γ=4 is a tested
             shared-skill hypothesis, not a theorem)
           → mixed with a 10% uniform floor

estimator: MaxRL success-conditioned advantages  w_i = r_i/K − 1/N
           (K=0 groups dropped; effective objective order N−1)

hindsight: dead groups are relabeled by the ENV to the sub-goal actually
           achieved and trained as successes of that easier task
           (verifier-valid auxiliary update; exact gradient equality needs
           the trajectory-law condition in Proposition 6)
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

**Two necessary semantic contracts** (from Proposition 6; they are not
sufficient for unbiasedness):

1. **Verifier validity** — a relabeled success must be a *true* success of the
   relabeled task under the env's own verifier.
2. **Conditioning** — if trajectories embed the goal (goal-relative features,
   `desired_goal` obs, goal tokens in a prompt), return rewritten
   trajectories with the achieved goal substituted. The corrected ten-seed
   gridworld study exercises this valid-rewrite path (teacher AUC 0.652,
   teacher+hindsight 0.702); it does not include a corrected no-rewrite arm.
   The rewrite requirement follows from the estimator's conditioning, not
   from an archived smoke-test number.

For centered practical weights, equality to a fresh order-(N−1) update follows
from equality of the full relabeled/fresh joint laws. For success-only
hindsight, the corresponding success-conditional marginal-law match guarantees
exact ML. More generally, equality of the relevant update moment is necessary
and sufficient. Verifier correctness and goal rewriting alone establish none
of these distributional conditions.

## Adapters included

| adapter | what it shows | result (uniform → teacher → +hindsight) |
|---|---|---|
| `skill_chain` | 12-seed component ablation; checkpoint mean includes step zero | 0.660 uniform/no-HS → 0.781 γ=4/no-HS → **0.886** γ=4+centered, scale 1 |
| `grid_reach` | goal-conditioned pattern; concrete-goal verifier relabel | 0.583 → 0.652 → **0.702** (10-seed corrected study) |
| `gym_classic` | Gymnasium MountainCar/CartPole dynamics with custom nested binary tasks | corrected transition-matched MountainCar study below; historical CartPole result pending rerun |
| `gym_goal` | Gymnasium GoalEnv skeleton; requires an environment-specific verifier-backed relabel callback | (skeleton — bring your env) |

The adapter uses official Gymnasium dynamics and modern reset/step semantics,
but evaluates custom binary task predicates rather than standard episodic
return. MountainCar's sparse flag success is the external-dynamics twin of
the frontier-heavy regime; its positional curriculum ("reach x ≥ x*, walking
x* from valley to flag") is the pattern being tested.

### MountainCar case study: opening the sparse flag through transfer

The corrected study matches methods by **500,000 environment transitions**,
uses evaluation that preserves training RNG state, distinguishes all-fail
from all-pass groups, and commits the shared-policy implementation and raw
per-seed curves. It uses ten paired seeds, 64 evaluation episodes for each of
ten custom thresholds, and the complete final rollout group (so actual counts
are at least 500k). Values are mean ± sample SD:

| config | mean-pass AUC | final mean pass | final FLAG pass |
|---|---:|---:|---:|
| flag-only, shared policy | 0.024 ± 0.006 | 0.024 ± 0.006 | **0.000 ± 0.000** |
| uniform curriculum, shared | 0.389 ± 0.071 | 0.684 ± 0.094 | 0.058 ± 0.079 |
| exact mass γ=1, shared | 0.414 ± 0.081 | 0.758 ± 0.127 | 0.208 ± 0.266 |
| legacy `u_{N+1}` γ=1, shared | 0.414 ± 0.078 | 0.745 ± 0.121 | 0.175 ± 0.274 |
| learnability γ=1, shared | 0.411 ± 0.037 | 0.697 ± 0.088 | 0.080 ± 0.143 |
| exact mass γ=4, shared | 0.530 ± 0.059 | 0.928 ± 0.056 | 0.664 ± 0.232 |
| exact γ=4 + centered hindsight, shared | 0.720 ± 0.029 | 0.969 ± 0.013 | 0.842 ± 0.062 |
| **exact γ=4 + success-only hindsight, shared** | **0.727 ± 0.023** | **0.970 ± 0.014** | **0.848 ± 0.058** |
| centered hindsight, **per-bin parameters** | 0.229 ± 0.031 | 0.284 ± 0.028 | **0.000 ± 0.000** |

Paired bootstrap AUC deltas are +0.141 [0.076, 0.202] for exact γ=4
versus uniform, +0.116 [0.060, 0.172] for γ=4 versus γ=1, +0.191
[0.155, 0.231] for centered hindsight, and +0.197 [0.160, 0.238]
for success-only hindsight. All four remain supported by exact paired
sign-flip tests after Holm correction across the nine AUC contrasts.
By contrast, exact γ=1 is not separated from uniform, legacy `u_{N+1}`, or
learnability, and centered versus success-only is not separated after that
correction. Neither relabeling update is claimed unbiased under arbitrary
goal selection. Two practical conclusions:

1. **This aligned nested-threshold curriculum needs a transfer channel.**
   A task-agnostic shared policy transfers here; ten disjoint parameter tables
   do not at this budget. This negative control is not capacity/data matched
   and is not a universal proof that every curriculum must omit task identity.
2. With sharing in place, concentration and auxiliary hindsight matter more
   than the small fixed-N distinction among the three γ=1 scores. The
   corrected matched-transition result is flag-only 0.000 → exact γ=4
   0.664 → full stack 0.848 mean flag pass; the per-bin control remains
   0.000.

These are custom binary-threshold metrics on official `MountainCar-v0`
dynamics, not the environment's standard episodic return. Full confidence
intervals and the multiple-comparison table are in
`curriculum_maxrl/VALIDATION.md`.

Run them:

```bash
python3 frontier_rl/test_framework.py                 # unit tests
python3 frontier_rl/examples/run_skill_chain.py       # five-seed regression anchor
python3 -m frontier_rl.examples.run_skill_chain_ablation # retained 12-seed factorial
python3 frontier_rl/examples/run_grid_reach.py        # robotics-style demo (~3 min)
python3 frontier_rl/examples/run_gym_benchmark.py     # gymnasium benchmark (~10 min, pip install gymnasium)
python3 frontier_rl/examples/run_mountaincar_shared.py # corrected matched-transition study
```

## Mapping to robotics / gym in practice

- **Task bins**: pick the axis your curriculum should walk (goal distance,
  obstacle count, object mass...). ~8–30 bins is plenty; the pseudo-count model needs
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

## Design: one schedule, five execution shapes

The algorithm is deliberately factored so each piece can be swapped to match
the training regime without touching the others — the flexibility is the
design, not an accident:

| training regime | teacher variant | evidence stream | hindsight | validated on |
|---|---|---|---|---|
| episodic groups, fixed task pool (RLVR/LLM prompts) | `FrontierTeacher` (Beta rows, Thompson) | group (task, K of N) | dense relabel via `TaskSpace.relabel` | skill chain, maze GPU, verl integration |
| episodic groups, procedural tasks | `StreamingFrontierTeacher` (kernel posterior) | (difficulty, K of N) | same | continuous-goal reach |
| goal-conditioned control (gym/robotics) | `FrontierTeacher` over goal bins | group | relabel + conditioning rewrite | gridworld, MountainCar, CartPole |
| massively parallel sim (IsaacLab, 4096 envs) | `FrontierBinTeacher` (vectorized, evidence-scaled decay, deterministic optimism) | per-reset Bernoulli stream | statistics-half only (occupancy credit) | adapter + unit tests; SONIC design doc |
| dense-reward PPO | any of the above with `utility="learnability"` | termination flag as verifier | usually skip (dense reward is the partial credit) | SONIC_RESPONSE.md analysis |

The swap points and what fixes each choice:

- **utility** — `advmass` when a real group size N exists (the band is then
  *derived*, peak ≈ ln N/N); `learnability` when evidence is a reset/hazard
  stream with no N (SONIC Q2).
- **posterior** — Beta rows for fixed pools; kernel over a difficulty axis
  for procedural sources; vectorized arrays with half-life-in-episode-
  equivalents decay when throughput varies by orders of magnitude (Q4).
- **optimism** — Thompson when stochasticity is fine; `mean + k·std` under
  determinism guardrails (Q3). Both validated equivalent when a floor exists.
- **γ** — 4 on tight chains (compounding), 1 everywhere else (measured,
  including the negative transfer on broad pools).
- **hindsight** — full trajectory relabel where the env verifies exactly and
  conditioning can be rewritten; statistics-only credit where it can't
  (on-policy PPO); off where dense reward already carries partial credit.

## IsaacLab / massively-parallel sim adapter

`adapters/isaaclab_curriculum.py` provides `FrontierBinTeacher`, mapping the
teacher onto IsaacLab's ManagerBasedRLEnv pattern (verified against a
production humanoid-tracking fork): task bins live in the *command manager*,
success lives in the *termination manager*, and the teacher consumes the
**reset stream** — every episode reset is one Bernoulli observation
(bin, terminated-early?). No groups needed; no isaaclab import required (the
curriculum-term wrapper imports it lazily), so the module unit-tests on CPU.

```python
teacher = FrontierBinTeacher(n_bins=n_motion_bins, utility="learnability",
                             decay_half_life=2048)   # episode-equivalents
# termination hook (each step or on resets):
teacher.observe_resets(bin_of_env[reset_ids], terminated_early[reset_ids])
# command hook (assigning tasks to reset envs):
new_bins = teacher.sample_bins(len(reset_ids))
```

Key adaptations for the parallel-sim regime, each traced to the SONIC
analysis: evidence-scaled decay (half-life invariant to env count — exact:
10 successes age to 5.0 after one half-life of events), deterministic
optimism bonus, learnability default, and a `max_prob` tripwire instead of a
shaping cap. See `SONIC_RESPONSE.md` for the full design rationale including
the closed-loop threshold-curriculum stability rules.

## Streaming / procedural task sources

`streaming.py` provides `StreamingFrontierTeacher` for sources with **no
fixed task pool** (every task fresh: generated mazes, sampled goals,
synthetic problems with a difficulty parameter d ∈ [0,1]). It replaces the
per-task Beta rows with a kernel (Nadaraya-Watson) pass-rate posterior over
the difficulty axis + Thompson sampling on a difficulty grid, with optional
isotonic projection when d orders pass rates. Validated on a continuous-goal
reach task (5 seeds, matched budgets): streaming **matches the discrete-bin
teacher exactly** (AUC 0.684 vs 0.684; final 0.922 vs 0.919; uniform 0.648)
— you lose nothing by dropping the pool assumption. Use it when your
task generator has a difficulty dial; use bins when you have a fixed
prompt set.

## What this does NOT do
- Replace your RL optimizer: `Policy.update` is yours; this package decides
  *what to train on and with what advantage weights*, not how to descend.
- A general safety claim for GRPO-style std-normalized advantages. Historical
  audited maze logs raise an objective-by-curriculum interaction hypothesis,
  but a corrected factorial is still required (REPORT.md F2).
