# Fixed-completion MaxRL N-sweep on SkillChain

## Question and status

This is a post-guidance, CPU-only follow-up asking whether the paper's
rollout-aware sampling score

\[
u_N(p)=1-(1-p)^N-p
\]

is more useful than the N-agnostic learnability score `p(1-p)` as MaxRL's
group size grows. It is a synthetic mechanism test, not an additional neural
domain and not a preregistered confirmatory experiment.

The `p(1-p)` arm is a score-level ProCuRL/SFL-style comparator. It is not a
faithful reproduction of either complete algorithm.

## Controlled protocol

- Environment: default `SkillChainEnv` (3 nested chains x 12 levels, 10
  actions per skill).
- Group sizes: `N = {2, 4, 8, 16, 32}`.
- Samplers: uniform, `p(1-p)`, and matched `u_N`.
- Estimator: the same practical dropped-group MaxRL weights in every arm.
- Hindsight: disabled, isolating task sampling.
- Budget: exactly 51,200 sampled completions in every cell. This is the
  existing fixed-N protocol's `400 x 8 x 16` budget.
- Seeds: 0--7, paired within each N.
- Checkpoints: every 2,560 completions, including the common initial policy.
- Teacher: discounted Beta pseudo-counts, Thompson draws, utility power 1,
  and a 10% uniform floor. Group decay is converted to a common
  per-completion rate so changing N does not change the teacher's effective
  memory horizon.
- Primary metric: normalized trapezoidal AUC of exact mean pass rate against
  sampled completions. Comparisons across samplers within an N are causal in
  the testbed; raw comparisons across N also change the number of groups and
  optimizer updates.

## Results

| N | Uniform AUC | `p(1-p)` AUC | `u_N` AUC | `u_N - p(1-p)` | paired signs |
|---:|---:|---:|---:|---:|---:|
| 2  | 0.7507 | 0.7401 | 0.7401 | +0.0000 | identical |
| 4  | 0.7842 | 0.7503 | 0.7810 | +0.0307 | 8/8 |
| 8  | 0.7420 | 0.6809 | 0.7729 | +0.0920 | 8/8 |
| 16 | 0.6566 | 0.5493 | 0.7019 | +0.1526 | 8/8 |
| 32 | 0.5071 | 0.3997 | 0.5906 | +0.1909 | 8/8 |

The main mechanism result is unusually clean: `u_2` is exactly `p(1-p)`, and
every nontrivial N has the same contrast sign in all eight paired seeds
(descriptive two-sided exact sign test p=0.0078125 at each N; no multiplicity
correction). The *mean* AUC gap increases from +0.0307 at N=4 to +0.1909 at
N=32, but only 5/8 individual seed trajectories are monotone across
`N={4,8,16,32}`. We therefore do not claim a scaling law.

Against uniform sampling, `u_N` is not universally superior: its mean AUC is
slightly lower at N=2 and N=4, then higher by +0.0309, +0.0453, and +0.0836
at N=8, 16, and 32 respectively. The paired sign is 8/8 at N=8 and 7/8 at
N=16 and N=32. This is useful negative calibration: the experiment supports
the N-aware distinction from `p(1-p)`, not a blanket claim that a curriculum
always beats uniform sampling.

Final scores mostly saturate at small N, so the completion-indexed AUC is more
diagnostic than final accuracy. The full JSON also records pass@8 AUC, final
metrics, group outcomes, coefficient mass, task counts, every checkpoint,
paired contrasts, source hashes, and budget invariants.

## Reproduce

```bash
python3 -m unittest curriculum_maxrl.test_fixed_budget_n_sweep -v
python3 curriculum_maxrl/run_fixed_budget_n_sweep.py --workers 4
```

Expected checks:

- every cell consumes exactly 51,200 completions;
- every cell uses the same completion checkpoints;
- paired `u_2` and `p(1-p)` trajectories are identical; and
- the five-seed uniform/N=16 final mean exactly matches
  `results_fixed_n.json` (0.965880204501467).

Artifacts:

- `run_fixed_budget_n_sweep.py`: protocol and deterministic runner;
- `test_fixed_budget_n_sweep.py`: unit, determinism, and budget checks;
- `results_fixed_budget_n_sweep.json`: complete structured result.
