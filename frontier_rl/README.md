# frontier_rl — the curriculum-MaxRL schedule as a reusable framework

`frontier_rl` packages the default training loop so it can be applied to gym
environments, robotics simulators, or LLM prompt sets without touching the
core. The core is NumPy-only, with no torch/gym dependency.

> **Validation scope (2026-08-04).** The corrected adapters and unit tests
> enforce the behavioral contracts below, but the proposed exact sampler plus
> common-destination relabeler has not been validated end to end in a neural
> or LLM run. Historical maze/Countdown experiments used different relabeling
> semantics. Numerical claims require a committed artifact listed in
> `VALIDATION_2026-08-04.md`; older non-vendored demo tables are omitted here.

## The algorithm (one screen)

```
teacher:   Beta(α,β) posterior per task (decay 0.7) → Thompson sample p̃
           → practical-MaxRL score ν_N(p̃) = (1−p̃) − (1−p̃)^N
             [= half the expected absolute coefficient mass; exact peak
              p* = 1−N^(−1/(N−1)) ≈ ln N/N]
           → sample tasks ∝ ν_N^γ  (γ≈4 on tight shared-skill chains,
             1 by default elsewhere)
           → mixed with a 10% uniform floor

estimator: practical centered/drop MaxRL by default:
           w_i = r_i/K − 1/N for K>0; constant groups are zeroed
           (the full-control-variate option instead retains −1/N at K=0)

hindsight: an all-fail requested group may be rewritten to one verified
           destination and rescored there; this can create destination
           contrast, but the source-induced update is generally off-policy
           even when every relabeled reward is semantically exact
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

**The three hindsight contracts** make a relabeled group a well-formed
destination-conditioned practical-MaxRL coefficient update. They do not make
it an on-policy destination gradient:

1. **Semantic verification** — a relabeled success must be a *true* success
   of the relabeled task under the env's own verifier.
2. **One destination and rewritten conditioning** — choose one common
   destination for the group; if trajectories embed the goal (goal-relative
   features, `desired_goal` observations, or prompt tokens), rewrite every
   trajectory to that destination. The grid and Countdown regression tests
   verify this rewrite directly.
3. **Destination contrast** — reverify every rewritten trajectory and train
   only when `0 < K' < N`; the trainer rejects constant destination groups.

Even with all three contracts, trajectories are selected under the source
task and relabel rule rather than freshly sampled from the destination. The
result is therefore a verified but generally off-policy destination update.

## Adapters included

| adapter | role | current validation scope |
|---|---|---|
| `skill_chain` | exact-score shared-skill regression environment | used by the 20-seed estimator controls |
| `grid_reach` | goal-conditioned grid reach with coarse distance-ring teacher IDs | shared anchor, all-row rewrite, reach-at-any-time re-verification, and mixed outcomes are unit-tested |
| `countdown_llm` | dependency-free Countdown verifier/LLM hook with coarse tier IDs | shared integer anchor, all-prompt rewrite, exact re-verification, and mixed outcomes are unit-tested |
| `gym_classic` | MountainCar and CartPole binary-success controls | three-seed paper evidence; task spread and shared-parameter transfer, not a universal ordering |
| `gym_goal` | GoalEnv integration skeleton | interface example only |
| `cosmos_libero` | flow-policy/VLA integration pattern | adapter and mock tests only; no real-robot or neural-policy validation |

The adapter tests establish software and verifier semantics. They do not turn
source-task trajectories into an on-policy destination sample and do not
substitute for the missing balanced neural/LLM factorial.

### Gym control: what is supported

The committed three-seed binary-success study supports two mechanisms:
target-only practical MaxRL remains empirically frozen at the tested budget,
while spreading training over easier bins creates usable updates; those
updates transfer only when the policy parameters are shared across bins.
The gated stack reaches the official-task endpoint in all three seeds and is
faster and less variable than uniform in the reported controls. This is a
small classic-control study, not a claim that curricula always help or that
full-CV MaxRL is mathematically silent at `K=0`.

Run them:

```bash
python3 frontier_rl/test_framework.py                 # unit tests
python3 frontier_rl/examples/run_skill_chain.py       # regression anchor (~2 min)
python3 frontier_rl/examples/run_grid_reach.py        # robotics-style demo (~3 min)
python3 frontier_rl/examples/run_gym_benchmark.py     # gymnasium benchmark (~10 min, pip install gymnasium)
```

## Mapping to robotics / gym in practice

- **Task bins**: pick the axis your curriculum should walk (goal distance,
  obstacle count, object mass...). ~8–30 bins is plenty; the posterior needs
  a few groups per bin to localize the frontier.
- **Binary success**: use the env's own success predicate. Shaped rewards
  can coexist in your policy update; the *teacher* should only see binary
  outcomes (that is what the practical-estimator coefficient-mass derivation
  assumes).
- **relabel**: gymnasium GoalEnvs give you `achieved_goal` for free — map it
  to its bin and rewrite `desired_goal` in the stored observations
  (contract 2). For non-goal envs with no meaningful relabel, return `None`;
  you keep the teacher benefits and lose only the hindsight term.
- **Group size N**: the practical score peaks near p ≈ ln N/N. N=16 puts
  the peak near 17% success; raise N to move it toward harder bins.
- **On/off-policy**: ordinary requested-task updates are drawn from the task
  the teacher selected. Hindsight destinations are different: exact
  verification removes label error, but the source task and relabel rule
  induce a proposal that is generally off-policy for the destination. If
  you swap in PPO, keep the requested-task path near-on-policy and treat
  relabeled updates as off-policy data unless you add a correction.

## Design: one schedule, five execution shapes

The algorithm is deliberately factored so each piece can be swapped to match
the training regime without touching the others — the flexibility is the
design, not an accident:

| training regime | teacher variant | evidence stream | hindsight | software/evidence status |
|---|---|---|---|---|
| episodic groups, fixed task pool (RLVR/LLM prompts) | `FrontierTeacher` (Beta rows, Thompson) | group (task, K of N) | `TaskSpace.relabel` | exact-score chain + unit tests; historical maze/LLM runs are not method validation |
| episodic groups, procedural tasks | `StreamingFrontierTeacher` (kernel posterior) | (difficulty, K of N) | same | CPU demo only |
| goal-conditioned control (gym/robotics) | `FrontierTeacher` over goal bins | group | shared destination + conditioning rewrite | GridReach regression + three-seed Gym mechanism control |
| massively parallel sim (IsaacLab) | `FrontierBinTeacher` | per-reset Bernoulli stream | statistics-only occupancy credit | adapter tests + one-seed external pilot |
| dense-reward PPO | `utility="learnability"` variant | termination flag | usually skip | design analysis only |
| flow/diffusion heads (weighted SFT) | `MasteryFrontierTeacher` | group | positive-only template rewrite | adapter + mock pilot only |

The swap points and what fixes each choice:

- **utility** — `advmass` is ν_N for the practical centered/drop estimator
  when a real group size N exists (the band is then derived, with peak
  `1−N^(−1/(N−1))`); raw and full-CV MaxRL have different activity
  profiles. Use `learnability` when evidence is a reset/hazard stream with
  no N (SONIC Q2).
- **posterior** — Beta rows for fixed pools; kernel over a difficulty axis
  for procedural sources; vectorized arrays with half-life-in-episode-
  equivalents decay when throughput varies by orders of magnitude (Q4).
- **optimism** — Thompson when stochasticity is fine; `mean + k·std` under
  determinism guardrails (Q3). Both are implemented; equivalence is not a
  paper-level validated claim.
- **γ** — 4 helped on the tight shared-skill chain; 1 is the default
  elsewhere after the concentrated setting failed to transfer to a broad
  pool. This remains an empirical tuning choice.
- **hindsight** — full trajectory relabel where the env can verify one
  destination, rewrite all conditioning, and produce contrast; the update
  remains generally off-policy for that destination. Use statistics-only
  credit where trajectories cannot be rewritten (on-policy PPO), and turn
  it off where dense reward already carries partial credit.
- **weights** — practical centered/drop MaxRL (`r/K − 1/N` for `K>0`,
  with constant groups zeroed) when per-sample log-probs exist;
  `positive_part=True` (successes only, `TrainerConfig.positive_weights`) for
  weighted-SFT on flow/diffusion heads. This is a distinct success-only
  update, not the practical centered estimator; its expected positive
  coefficient sum is still ν_N, and all-pass groups self-retire
  (COSMOS3_RESPONSE.md Q1).

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

## Cosmos3 / LIBERO flow-policy adapter

`adapters/cosmos_libero.py` implements the COSMOS3_RESPONSE.md Part-II design
for RLVR on flow-matching VLA policies (no tractable per-sample log-prob):

- **positive-part weights** (`TrainerConfig(positive_weights=True)`) — the
  weighted-RFT estimator. Its expected positive coefficient sum is exactly
  ν_N, so the teacher retains a coefficient-sum interpretation, but dropping
  the failure coefficients changes the update estimator. Measured
  cost of dropping the failure term (skill-chain anchor, matched budgets,
  3 seeds): AUC 0.887→0.828, final 0.986→0.941; the full stack still
  clears plain-teacher (0.73) and uniform (0.65) by a wide margin
  (`curriculum_maxrl/positive_part_training_cost.json`). Use it only when
  per-sample log-probs are genuinely unavailable.
- **`CosmosLiberoSpace`** — arms are predicate-conjunction goals; `rollout_fn`
  is a hook for the policy-server + vector-env wave (no cosmos import here);
  live groups are verified ONLY by the sim's binary success; all-fail groups
  are relabeled to the deepest achieved sub-conjunction with the language
  conditioning rebuilt from a **fixed template per goal** (contract 2 at VLA
  scale — never free-generated) and can never be upgraded to the original
  task's success.
- **relabel-only arms** — sub-goal tasks the teacher cannot roll out directly
  (`samplable_mask()`); they exist purely as credit targets for hindsight.
  This distinction is load-bearing: letting the teacher sample the invented
  curriculum directly turns a frontier-heavy pool into a balanced one and
  erases the categorical result (measured while building the mock pilot).
- **`MasteryFrontierTeacher`** — mastery splits create init-state-bin child
  arms with hierarchical pseudo-count shrinkage toward the parent (the
  starved-450-arm fix), plus the samplable mask.
- **`PoisonRateMeter`** — Pilot 0b's instrument: per-predicate-class
  precision/recall of a self-verifier vs oracle; prunes the relabel
  vocabulary at a precision gate (the action is removal of a class, never
  lowering the gate).

`examples/run_cosmos_pilot.py` runs the preregistered Phase-1 arms **plus
baselines** (DAPO dynamic resampling, GRPO estimator arms) on a CPU mock
(Bernoulli predicate skills, exact pass rates): frontier-heavy pool where
uniform, DAPO, and teacher-alone score **0.000 in every seed** while
oracle-relabel reaches **0.862**, self-verified 0.756, and per-class gating
recovers most of the poison gap (0.842) — the V5 categorical result and the
poison→gate story reproduced end-to-end on the exact code path the real
integration will use. It also surfaced a base-rate warning for the real
Pilot 0: with rare true achievements, precision measured on failure-heavy
rollouts is dominated by false-positive opportunity (~65:1 at q=0.015), so
the probe set must be enriched with successes or the gate will mis-prune
clean classes.

The rest of the pipeline to real training is in place and unit-tested with
fakes that mirror the verified cosmos-framework APIs:

- `adapters/cosmos_live.py` — `LiveRolloutBackend` (one group = one
  `/predict_batch` wave against SubprocVectorEnv, per-episode init states,
  end-of-episode predicate snapshots for all-fail groups only),
  `goal_predicates_of` (BDDL `goal_state` → canonical predicates),
  `WeightedCFMBuffer` (Policy → JSONL manifest for the weighted
  flow-matching SFT: `(w·per_instance_loss).sum()/w.sum()` at the existing
  `compute_flow_matching_loss` call site), `Phase1Round` (collect → train →
  redeploy loop with teacher-state persistence).
- `evaluation.py` — unbiased success@k, easy-decile retention (fixed probe),
  teacher-calibration (the posterior-inflation detector), `RunLedger` +
  `matched_budget_report` (both currencies: matched rollouts AND matched
  wall-clock, with live/relabel update counts separated).
- `pilot0.py` — the three gate instruments (within-group variance, poison
  rate with success-enriched probes, surrogate-fidelity cosine) and the
  go/no-go verdict.
- `trainer.py` — baseline arms: `estimator="grpo"/"rloo"`,
  `dapo_max_redraws` (paid redraws, V5 protocol). Note: the H6 GRPO-collapse
  ablation requires function approximation — it does not reproduce on
  tabular-exact testbeds (a measured non-result, consistent with DESIGN.md
  §8) — so it is a real-model Phase-2 claim, in per-seed success@k currency.

`../READINESS.md` is the launch runbook: what is done, the ordered R1–R6
checklist to real training-vs-baseline (env plumbing → checkpoint/data
freeze → Pilot 0 → weighted-SFT hook → four-arm launch → baselines), each
with its gate.

## Streaming / procedural task sources

`streaming.py` provides `StreamingFrontierTeacher` for sources with **no
fixed task pool** (every task fresh: generated mazes, sampled goals,
synthetic problems with a difficulty parameter d ∈ [0,1]). It replaces the
per-task Beta rows with a kernel (Nadaraya-Watson) pass-rate posterior over
the difficulty axis + Thompson sampling on a difficulty grid, with optional
isotonic projection when d orders pass rates. A five-seed CPU demo reports
similar streaming and discrete-bin endpoints, but it is not part of the
paper's validated evidence and does not establish equivalence. Use this
adapter when the task generator has a difficulty dial; use bins for a fixed
prompt set.

## What this does NOT do
- Replace your RL optimizer: `Policy.update` is yours; this package decides
  *what to train on and with what advantage weights*, not how to descend.
- Establish a universal estimator ordering. The historical maze archive has
  an estimator-associated coverage pattern, but its adaptive schedules,
  warm-start reuse, and unbalanced sampler mix do not identify a causal
  curriculum-by-estimator interaction.
