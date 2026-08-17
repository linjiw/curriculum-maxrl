# Why coefficient activity loses on AMaze, and what to change

Cross-method development sweep, 35 runs, 7 arms x 5 seeds x 5,000 updates,
upstream configs verbatim except where noted. Held-out return averaged over
`Maze-SixteenRooms`, `Maze-Labyrinth`, `Maze-StandardMaze`.
**Development only — not evidence.**

## Result: we do not beat the benchmark

| arm | mean | sd |
|---|---|---|
| **plrMM32 — upstream robust PLR, MaxMC, 32x1** | **0.6288** | 0.091 |
| drControl — no curriculum | 0.5390 | 0.168 |
| accelCA8 — ours, in ACCEL, 4x8 | 0.5005 | 0.207 |
| plrCA8 — ours, in PLR, 4x8 | 0.4817 | 0.134 |
| accelMM32 — upstream ACCEL | 0.4464 | 0.223 |
| accelMM4x8 | 0.4248 | 0.239 |
| plrMM4x8 | 0.4222 | 0.205 |

Upstream PLR wins, and our best arm does not even beat the no-curriculum
control. Stated plainly: **at 5,000 updates, coefficient activity as a PLR
level score is not competitive on AMaze.**

## The diagnosis: the configuration loses, not the score

Paired by seed, exact two-sided sign-flip:

| contrast | mean | pairs | p |
|---|---|---|---|
| **plrMM4x8 − plrMM32** | **−0.2065** | **0/5** | **0.0625** |
| plrCA8 − plrMM32 | −0.1470 | 1/5 | 0.125 |
| accelCA8 − accelMM4x8 | +0.0757 | 4/5 | 0.5625 |
| plrCA8 − plrMM4x8 | +0.0595 | 3/5 | 0.6875 |
| accelCA8 − accelMM32 | +0.0541 | 3/5 | 0.75 |
| accelMM4x8 − accelMM32 | −0.0216 | 1/5 | 0.8125 |

The strongest signal in the table has nothing to do with our score:
**restructuring 32 levels x 1 eval into 4 levels x 8 evals costs −0.207 return,
0/5 seeds, at the minimum attainable p for five seeds.** MaxMC pays that cost
too.

Within the 4x8 configuration our score *helps* — +0.060 in PLR and +0.076 in
ACCEL, positive in 3/5 and 4/5 seeds — but those gains are a third of what the
restructuring destroys, and neither is close to significant.

**So the loss is self-inflicted.** We adopted 4x8 to give a success-rate score
enough Bernoulli observations to rank levels, and paid more in lost level
diversity than the score could recover.

## Why we thought 4x8 was necessary, and why it may not be

Upstream runs every maze config at `n_eval=1`, so a success-rate score sees one
Bernoulli per level per visit: `Beta(1+s, 2−s)` takes two values, which is
nearly all ties under rank-based prioritisation. MaxMC extracts a continuous
regret signal from the same rollout. Multi-episode counting cannot help either,
since `max_episode_steps=250` against a 256-step rollout permits at most one
completed episode per stream.

But the PLR buffer **already persists per-level evidence**: `success_counts` and
`trial_counts` are buffer fields, new levels start at zero, replayed levels
inherit their stored counts, and the runner scores from the accumulated
posterior (`plr.py:373-378, 407-415`; `plr_runner.py:297-303`). A level's
posterior therefore sharpens across revisits **without** sacrificing diversity.
With a 4,000-slot buffer and 32 levels per update, surviving levels are revisited
many times over 5,000 updates.

That was never tested. Every coefficient-activity arm so far ran at 4x8.

## Two changes

**A. Run at full diversity (32x1), no code change.** Set
`plr_frontier_require_n_eval_match=False` so the declared N is decoupled from
`n_eval`, keep `n_parallel=32, n_eval=1`, buffer 4,000, and let accumulation
supply fidelity over time instead of buying it with diversity. This makes our
arm directly comparable to `plrMM32`, the best arm. **Running now** at N=8 and
N=16 for PLR and N=8 for ACCEL, 5 seeds.

**B. Discount stale evidence (overlay v5).** Accumulation is currently
*undecayed*: `candidate_successes = base_successes + observed_successes`
(`plr.py:377`). A level's pass rate drifts as the student learns, so undecayed
counts describe **historical** difficulty — a level solved 40 times early stays
"mastered" forever, and an early-hard level stays "hard" after it is mastered.
For a curriculum score whose entire premise is targeting the current frontier,
that is a defect.

Our own deployed tracker already solves this: `frontier_rl/teacher.py` and
`curriculum_maxrl/verl_curriculum.py:88` use a **discounted** Beta tracker,
`alpha = 1 + (alpha−1)·decay + k` with decay 0.7, precisely because pass rates
drift. Overlay v5 ports that into the PLR buffer as `--plr_frontier_decay`
(default 1.0, reproducing current behaviour); counts become float32 so evidence
can be discounted. Smoke-tested at decay 0.7, rc=0.

## What would change the verdict

If A closes the gap to `plrMM32`, the story is "activity scoring matches or
beats regret scoring at equal diversity". If A does not, the honest conclusion
is that a purely success-rate-based level score is at a structural disadvantage
against a regret proxy in a setting that grants one binary observation per
level visit — which is itself a publishable boundary and directly parallels the
Acrobot finding that the *shape* helps while the *peak location* does not.

Either way this is reported. Nothing here is preregistered evidence, and no
number in this document may enter the paper as a result.
