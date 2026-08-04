# IsaacLab 5-arm pilot — our read of the fixed-grid eval (P-B verdict)

*2026-07-29. The IsaacLab team's pilot (their run, our co-designed
protocol: ISAACLAB_DESIGN.md P-A/P-B/P-C + REVIEW_ADVICE.md). Fixed-grid
eval artifacts (mid=300 / final=599 iters, per-terrain-level pass rates,
same seed/grid for every arm), Anymal-C rough, 1 seed, from-scratch.*

## Final (iter 599) fixed-grid results

| arm | mean pass | easy levels (0–1) | hard levels (7–9) |
|---|---|---|---|
| **greedy (stock ±1 walker)** | **.372** | .76/.71 | .16/.14/.16 |
| uniform | .297 | .51/.47 | .20/.18/.13 |
| control (no curriculum motion) | .291 | .47/.43 | .17/.16/.13 |
| teacher (frontier, learnability) | .278 | .50/.40 | **.23**/.15/.14 |
| scripted ramp | .266 | .49/.42 | .16/.16/.11 |

## Verdict against the pre-registered predictions

**P-B: the honest null landed exactly as pre-registered — and stronger.**
ISAACLAB_DESIGN.md predicted parity-with-greedy on the stock grid
("greedy is near-optimal when every level is learnable"); the data says
greedy WINS outright (+.075 mean over the teacher), dominating the easy
rows. The teacher's only edge is a small one at level 7 (.23 vs greedy's
.16) — frontier concentration buying a sliver of deep coverage at a large
easy-row cost. One seed; the sliver is within noise.

## Why this matters for the paper (third-scale replication)

This is the SAME pattern as the maze step-matched analysis (allocation
gain ≈ 0 when everything is learnable) and Countdown v1 (uniform breadth
beat the teacher on a learnable-everywhere pool): **on pools without
unlearnable-at-budget regions, allocation does not pay — at 1.26M-param
maze, 360M LLM, and now legged-robot locomotion.** Three scales, three
task families, one regime rule. The paper's Q1 answer ("what can
allocation contribute, at most") gains its robotics data point, and the
regime boundary — allocation pays only where dead/unlearnable regions
exist to avoid — is now the empirically supported statement.

Attribution: the runs are the IsaacLab team's; the protocol (fixed-grid
probe, P-A/P-B/P-C, honest-null pre-registration) was co-designed; this
read uses only their on-disk eval artifacts. The phase-2 discriminating
condition (grids WITH unlearnable rows) remains theirs to run.
