# Measured minimax throughput — 2026-08-15, and a correction

**The lane closure of earlier today was wrong on its central factual premise.**
`LANE_CLOSURE_2026-08-15.md` recorded "throughput is unmeasured and the
248.97 tr/s figure is compile-contaminated" as the one open item, and named a
bounded throughput probe as the single permitted future action. That probe has
now been run. It took under two minutes and it overturns three of the four
reasons the lane was closed.

## The measurement

Bounded 200-update run of the **unmodified upstream** `maze/dr` configuration,
on the local RTX 5090 under the JAX 0.6.2 / CUDA 12.9 Blackwell environment.
Engineering only: no evidence, no benchmark endpoint, no multi-seed result.

```
steady-state window: updates 50 -> 200 in 18.99 s
  0.1266 s/update        64,693 transitions/s
```

Steady state excludes the first logged tick, which folds in JIT compilation.
Total environment steps confirm the configuration ran as shipped: 200 updates
x 32 parallel x 256 steps = 1,638,400 transitions.

**The recorded 248.97 tr/s figure is wrong by a factor of 260.** It was
measured over a run short enough that JIT compile dominated. The lane has been
planned for weeks against a number that was off by more than two orders of
magnitude, in the pessimistic direction.

## What a full-scale run actually costs

All three headline methods were measured the same way, 200 updates each from
the unmodified upstream config. They come out nearly identical, because
minimax counts **gradient updates** and the extra replay/mutation-evaluation
rollouts are already folded into the per-update wall clock:

| upstream config | s/update | 30,000 updates |
|---|---|---|
| `maze/dr` | 0.1266 | **1.06 GPU-h** |
| `maze/plr` (robust PLR) | 0.1339 | **1.12 GPU-h** |
| `maze/accel` | 0.1243 | **1.04 GPU-h** |

So the earlier "robust PLR costs ~2x DR" reasoning, based on counting rollouts
rather than measuring updates, was also wrong: it is 1.06x.

The 492,036,096-transition budget in `UED_MATCHED_DEV_PREREG.md:75` is not an
inflated custom protocol — it is exactly the upstream standard of 30,000
updates at 32x256. It is also **2.1 hours**, not a day, so the "no resume path
against a 1-day `gpuq` cap" objection does not bind.

## Which closure reasons survive

| closure reason | status |
|---|---|
| "~492M transitions/run against a 1-day cap with no resume path" | **VOID** — 2.11 GPU-hours |
| "five dev seeds cap the exact two-sided p at .0625" | **VOID as stated** — five seeds was a response to believed compute scarcity. At ~2 GPU-hours/run, 10 seeds cost ~21 GPU-hours and give a floor of 2/1024 = .00195 |
| "no ACCEL/PAIRED/robust-PLR arm exists in `ued_benchmark/configs/`" | **VOID** — true of *our* configs, but upstream **ships tuned configs** for dr, plr, pplr, accel, paccel, paired (and S5 variants) at `config/configs/maze/`. We do not need to implement the baselines; we need to stop hand-rolling them |
| "its own prereg says never paper evidence" | **STANDS**, but applies to the *matched-dev selection protocol*, not to a new campaign run against upstream configs and the upstream test set |
| four remote-hardening blockers unfalsifiable locally | **STANDS but is now irrelevant** — those govern the Hopper v4 remote ladder. The measurement above is local, and a local campaign does not need that ladder at all |

## Why this was missed

The lane spent its effort on a bespoke matched-development protocol, a v4
remote-hardening ladder, and calibration telemetry — none of which required
knowing the throughput — while the one number that determines feasibility sat
unmeasured behind a figure the repository itself had flagged as contaminated.
The correct first action for any compute-bound lane is to measure the compute.

## What is now possible before Sept 25

A genuine competitive comparison on the upstream benchmark, using upstream
configs unmodified for every baseline and changing only the PLR scoring
function for our arm.

| arm | source | GPU-h/seed |
|---|---|---|
| DR | upstream `maze/dr`, verbatim | 1.06 (measured) |
| robust PLR | upstream `maze/plr`, verbatim | 1.12 (measured) |
| ACCEL | upstream `maze/accel`, verbatim | 1.04 (measured) |
| PAIRED | upstream `maze/paired`, verbatim | ~1.1 (not yet measured) |
| **coefficient activity** | upstream `maze/plr` with only the score swapped | ~1.12 |

At **10 seeds** that is roughly **55 GPU-hours**: about 2.3 days on the single
5090 alone, and less with Hopper A100 slices in parallel. Ten seeds put the
exact two-sided sign-flip floor at 2/1024 = .00195, which removes the
inferential dead end of the five-seed design.

Evaluation uses the upstream held-out set already named in every config
(`Maze-SixteenRooms,Maze-Labyrinth,Maze-StandardMaze`) via `minimax.evaluate`,
so the comparison is against the authors' own tuned settings and their own
test environments rather than against our reimplementation of either.

This is feasible. It should be preregistered separately, and it should use
upstream configs verbatim so that any comparison is against the authors' own
tuned settings rather than against our reimplementation of them.

## Caveats on this measurement

- One configuration (`dr`), one GPU, one 200-update window. PLR and ACCEL do
  more work per update (replay/mutation evaluation) and are measured separately.
- The Blackwell lane runs JAX 0.6.2 with a two-line `jax.tree_map` compatibility
  patch, not the source-faithful JAX 0.4.31 pin. Throughput is an engineering
  quantity and is not sensitive to that difference, but **any evidence campaign
  must state which lane produced it**, and the 0.6.2 lane has an open numerical
  parity gate against the 0.4.31 CPU reference.
- 64,693 tr/s is with the default LSTM student on a 13x13, 60-wall maze. Other
  student models or maze sizes will differ.
