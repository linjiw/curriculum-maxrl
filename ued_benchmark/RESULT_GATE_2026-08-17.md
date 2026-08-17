# Gate result: activity-gated MaxMC is our best AMaze arm and still does not beat upstream

Development sweep, 15 runs, 3 arms x 5 seeds x 5,000 updates, overlay v6,
upstream configs verbatim except the score. **Development only — not evidence.**

## What was tested

`score = max(MaxMC, 0) · u_N(posterior)` — keep MaxMC's per-timestep,
continuous value-error signal and use activity only to suppress levels the
posterior says are mastered (p̂→1) or hopeless (p̂→0). This addresses the
information deficit that sank every *replacement* variant, instead of paying
for it with diversity. Three arms: gate at N=8, gate at N=8 with posterior
decay 0.7 (overlay v5), and gate inside ACCEL.

## Result

Held-out return, mean over the three shipped test mazes, alongside every arm
run in this lane (65 runs total, all 5-seed):

| arm | mean | sd | vs plrMM32 | pairs | p |
|---|---|---|---|---|---|
| **plrMM32 — upstream robust PLR** | **0.6288** | 0.091 | — | — | — |
| **plrGateN8 — ours** | **0.5895** | 0.141 | **−0.039** | 1/5 | 0.56 |
| plrCA32N16 (pure activity) | 0.5510 | 0.087 | −0.078 | 1/5 | 0.19 |
| drControl (no curriculum) | 0.5390 | 0.168 | −0.090 | 2/5 | 0.50 |
| plrGateN8d7 (gate + decay .7) | 0.5282 | 0.044 | −0.101 | 1/5 | 0.125 |
| accelGateN8 | 0.5097 | 0.166 | −0.119 | 1/5 | 0.125 |
| plrCA32N8 (pure activity) | 0.5004 | 0.090 | −0.128 | 0/5 | 0.0625 |

## Reading it calmly

**Gating is our best arm by a clear margin.** plrGateN8 at 0.590 is the first
activity variant to beat the no-curriculum control (+0.051), and it recovers
most of the gap to upstream: from −0.128 (pure activity, 0/5) to −0.039 (1/5).
Against the pure replacement at the same 32x1 structure it is +0.089, 4/5
seeds. That confirms the diagnosis in `MATH_REVIEW_2026-08-16.md` §3 — the
loss was information, and restoring the per-timestep signal restores most of
the performance.

**It does not beat upstream.** −0.039, one seed in five ahead, paired SD 0.20.
Statistically that is a tie at n=5 — the interval comfortably includes zero in
both directions — and the honest reading of a tie against a stronger baseline
is "not better." The bar was `plrMM32`, and it was not cleared.

**The per-maze split is the most interesting thing here.** On two of the three
test mazes the gate arm is ahead of upstream: SixteenRooms .855 vs .792 and
StandardMaze .486 vs .365. It loses badly on Labyrinth, .427 vs .729, and that
single maze accounts for the entire deficit. Labyrinth is the long-corridor
maze; a plausible mechanism is that suppressing "hopeless" levels early starves
the student of exactly the long-horizon layouts Labyrinth tests. That is a
hypothesis from a five-seed development sweep, not a finding.

**Decay hurt.** Gate + decay 0.7 is −0.061 below plain gate (2/5) with a much
tighter spread (sd .044). Forgetting evidence at 0.7 per revisit appears to
throw away the posterior sharpening that makes gating work at all; if decay
belongs anywhere it is much closer to 1.

**ACCEL did not benefit.** accelGateN8 at .510 is +0.063 over upstream ACCEL
(.446, itself well below upstream PLR at this budget) and +0.109 over pure
activity in ACCEL, but still below the control. ACCEL's mutation step already
does its own difficulty targeting; adding ours on top does not compound.

## Where the lane stands

Three configurations have now been tested at 5 seeds each:

| configuration | best mean | vs upstream | status |
|---|---|---|---|
| (a) activity replaces MaxMC | 0.551 | −0.078 | **closed** — never beats upstream, 4x8/32x1, PLR/ACCEL, N=8/16 |
| (b) activity gates MaxMC | 0.590 | −0.039 | tested — best arm, recovers most of the gap, does not clear the bar |
| (c) MaxRL-estimator student | — | — | not run — the faithful mapping, a separate labelled study |

The result the paper can state, and now does: on AMaze the activity score's
value is as a **shape** applied to a richer signal, not as a standalone signal.
Gating recovers most of what replacement loses; neither beats upstream robust
PLR at 5,000 updates.

## What would be worth doing, and what would not

**Worth doing.** A single confirmatory campaign of plrGateN8 vs plrMM32 at the
full 30,000 shipped updates, 10 fresh seeds, preregistered, with the
per-maze breakdown as a frozen secondary. At ~1.1 GPU-h per run that is
~22 GPU-hours. It resolves whether the −0.039 is a tie or a loss, and it tests
the Labyrinth hypothesis. If it comes back "tie or better on two mazes, loss on
Labyrinth," that is a defensible and interesting result. If it comes back a
clean loss, the lane closes with a full-budget negative.

**Not worth doing.** Tuning the gate — exponent, clip, decay near 1, ACCEL
variants — against `plrMM32` on five development seeds until something wins.
Paired SDs here are 0.09–0.20; five seeds cannot separate +0.04 from noise, and
every additional development arm is another draw from a seed lottery.

Nothing in this document is evidence. No number here enters the paper as a
result; the paper's `sec:amaze` gains one sentence describing the gate outcome
qualitatively.
