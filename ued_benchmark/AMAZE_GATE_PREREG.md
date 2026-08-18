# Preregistration — activity-gated MaxMC vs upstream robust PLR on AMaze

**Status:** FROZEN 2026-08-17 before any confirmatory run.
**Schema:** `curriculum-maxrl/amaze-gate-confirmatory/v1`
**Overlay:** v6 (`ued_benchmark/overlay_v6_gated/gate.patch`) on upstream
`facebookresearch/minimax` @ `d053054c5290a04c1c4cd8b55704d999cad73e30`.

## 1. What is being tested, and why now

Three development sweeps (65 runs, 5 seeds each, 5,000 updates) established:

- replacing MaxMC with `u_N(p̂)` never beats upstream robust PLR (closed);
- **gating** MaxMC by `u_N(p̂)` — `score = max(MaxMC,0) · u_N(p̂)` — recovers
  most of the gap: −0.039 to upstream, 1/5 seeds, paired SD 0.20;
- the whole residual deficit is one test maze (Labyrinth .427 vs .729) while
  the gate arm is *ahead* on SixteenRooms (.855 vs .792) and StandardMaze
  (.486 vs .365).

At five seeds and one-sixth of the shipped budget, −0.039 with SD 0.20 is
indistinguishable from zero. This campaign resolves it at the shipped budget
with fresh seeds, and tests the per-maze pattern as a frozen secondary. It is
the one experiment that turns a development tie into an answer.

## 2. Design

Two arms, ten paired seeds, upstream `maze/plr` configuration **verbatim**
except the score, at the **full shipped budget**.

| arm | source | score |
|---|---|---|
| `plrMM` | upstream `config/configs/maze/plr.json`, unmodified | MaxMC (upstream) |
| `plrGate` | same config + `--ued_score=coefficient_activity --plr_frontier_mode=gate --plr_frontier_n_rollouts=8 --plr_frontier_require_n_eval_match=False` | `max(MaxMC,0) · E[u_8(p) \| D]` |

Shared, from the shipped config: `n_total_updates=30000`, `n_parallel=32`,
`n_eval=1`, `plr_replay_prob=0.5`, `plr_buffer_size=4000`,
`plr_use_robust_plr=True`, `plr_use_score_ranks=True`, `lr=3e-4`,
13x13 maze, 60 walls, `max_episode_steps=250`, `n_rollout_steps=256`,
`default_student_cnn` (LSTM student). No decay (`plr_frontier_decay=1.0`);
development showed decay 0.7 hurts.

**Seeds: 2001–2010.** Disjoint from every development seed (1001–1005) and from
every other campaign in this repository. Both arms of a seed share the seed, so
initialisation and level-generation streams are paired.

**Exponent N=8** is fixed by prior decision, not selected here: it is the
setting used in every development gate arm and equals the evaluation group
size convention of the v3 overlay. It is not re-tuned.

## 3. Evaluation

Each run's **final checkpoint** (`checkpoint.pkl` at update 30,000) is
evaluated with the shipped `minimax.evaluate` on the shipped held-out set
`Maze-SixteenRooms, Maze-Labyrinth, Maze-StandardMaze`, `n_episodes=100`
per maze, evaluation `seed=1`. Both `test_solved_rate` and `test_return` are
recorded. This replaces the development sweeps' last-logged in-training eval
with the protocol the benchmark's authors ship.

## 4. Primary estimand and decision rule

**Primary:** paired difference in mean held-out `test_solved_rate` over the
three mazes, `plrGate − plrMM`, over the ten seeds.

**Test:** exact two-sided paired sign-flip over 2^10 = 1,024 assignments,
α = .05, plus a 20,000-resample paired bootstrap 95% CI.

**SESOI:** +0.02 solved rate (2 percentage points), set before any
confirmatory data.

Outcomes, fixed in advance:

| condition | verdict |
|---|---|
| mean ≥ +0.02 **and** p ≤ .05 | **gate beats upstream** on the shipped protocol |
| CI upper < +0.02 | gate **does not** beat upstream; report as a full-budget negative |
| otherwise | inconclusive at n=10; report the interval, claim nothing |

Development results are not combined with these; they were selection, not
evidence.

## 5. Frozen secondaries (descriptive, no decision)

1. Per-maze paired differences on `test_solved_rate` and `test_return`, with
   sign-flip p per maze — this tests the Labyrinth hypothesis. It is
   **descriptive**: three mazes at n=10 cannot support a corrected claim, and
   none is made.
2. Mean `test_return` over mazes, paired, as a continuity check against the
   development metric.
3. Training-curve `plr/weighted_frontier_probability` for the gate arm, to
   report where its replay mass sits.

## 6. Cost and execution

Measured on this host: robust PLR 0.1339 s/update, gate 0.126 s/update at
32x1. Twenty runs × 30,000 updates ≈ **22 GPU-hours** at 3 concurrent on the
RTX 5090, ≈ 1 day wall. Evaluation adds minutes.

Runs are launched by `ued_benchmark/scripts/run_jobs.sh` from a frozen
`jobs.tsv`; each writes `checkpoint.pkl`, `logs.csv`, `meta.json` under its
own xpid. Failed runs are re-run from the identical seed; seeds are never
substituted and the block is never extended.

The analyzer refuses an incomplete 2×10 matrix, refuses to run twice, and
requires the shipped evaluate CSV for every cell.

## 7. What this cannot establish

- One benchmark, one student architecture, one budget, N=8 only.
- A "beats" verdict is a comparison against upstream **robust PLR with MaxMC**
  under its shipped configuration; it is not a claim against tuned ACCEL,
  PAIRED, or any published number.
- The AMaze student is PPO+GAE, so `u_N` here is a difficulty-shape gate on a
  regret signal, not the paper's estimator-derived mechanism. A positive
  result supports "activity-shaped gating improves regret-based replay," not
  "coefficient activity is the right curriculum objective."
- Ten seeds put the sign-flip floor at 2/1024 = .00195 and give roughly 80%
  power to detect +0.05 at the development SD; a true +0.02 will often read
  inconclusive, and that is what the rule says to report.
