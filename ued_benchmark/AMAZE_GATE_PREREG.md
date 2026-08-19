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

---

## Amendment 2026-08-18 — analyzer read path (pre-data)

**Timing.** Written while training was 3/20 complete and **before any
`minimax.evaluate` output existed**. No held-out number for any confirmatory
cell had been produced or read. The primary estimand, test, SESOI, verdict
table, seeds, and secondaries are unchanged.

**What was wrong.** The frozen analyzer read cell configuration from a flat
`meta["args"]` key. minimax writes `meta.json` as
`{config: {..., train_runner_args: {...}}}`, so the analyzer would have
refused every cell with `n_total_updates != 30000` and never analysed the
campaign. It also relied on `meta.successful`, which upstream initialises to
`False` and never updates.

**What changed.** The config checks now read `config.n_total_updates`,
`config.seed`, and `config.train_runner_args.{n_parallel,n_eval,ued_score,
frontier_mode,frontier_n_rollouts}` — and additionally assert the shipped
32x1 batch structure. Completion is established from `logs.csv`
(`n_updates >= 29990`) instead of `meta.successful`. Verified against the
three completed cells: budget 30,000, correct seeds, 32x1, correct score/mode/N,
final `n_updates` 29,990–29,999.

**Also recorded.** minimax's `_tick` counter is roughly 2x `n_updates` under
robust PLR at replay probability 0.5, because new-level evaluation cycles do
not update the student; 59,950 ticks is exactly the shipped 30,000-update
budget. And per-run wall time is 13,135–13,197 s (3.65 h) rather than the
~65 min projected from an idle-GPU measurement, because the RTX 5090 is shared
with four other lab jobs (~19 GB); the campaign will take ~24 h wall, not 7–8.
Neither affects the protocol.

---

## Amendment 2026-08-19 — evaluated checkpoints were not the final model (outcome-blind)

**Timing and blindness.** Written after the 2026-08-17 execution finished all 20
runs and 20 evaluations, and after the frozen analyzer **refused to run**. No
`minimax.evaluate` output has been opened, no held-out number for any cell has
been read, and `AMAZE_GATE_ANALYSIS.json` was never created. The primary
estimand, test, SESOI, seeds, verdict table and secondaries are **unchanged**.

**What went wrong.** The analyzer's completion check (`logs.csv` final
`n_updates >= 29990`, added in the 2026-08-18 amendment) rejected
`arm-plrMM-s2004-u30000` at 29,981. Investigating that rejection uncovered a
larger defect.

`minimax`'s `xp_runner` writes checkpoints inside the training loop on
`tick % checkpoint_interval == 0`, and the file ends with that block — **there
is no post-loop save**. Under robust PLR at replay probability 0.5 a tick is
roughly half an update, so a 30,000-update run reaches ~59,650–60,200 ticks.
Our launcher `run_arm.sh` overrode the shipped `checkpoint_interval` of 1000
with `$UPDATES` (30,000), conflating updates with ticks. Checkpoints were
therefore written only at ticks 30,000 and 60,000, and runs whose tick stream
stopped short of 60,000 kept the tick-30,000 file.

Read directly from each checkpoint's own stored `n_updates`:

| seed | plrMM | plrGate | paired | fraction of budget |
|---|---|---|---|---|
| 2001 | 14,945 | 14,945 | yes | 49.8% |
| 2002 | 29,940 | 29,940 | yes | 99.8% |
| 2003 | 14,899 | 14,899 | yes | 49.7% |
| 2004 | 15,003 | 15,003 | yes | 50.0% |
| 2005 | 29,932 | 29,932 | yes | 99.8% |
| 2006 | 14,938 | 14,938 | yes | 49.8% |
| 2007 | 15,098 | 15,098 | yes | 50.3% |
| 2008 | 15,124 | 15,124 | yes | 50.4% |
| 2009 | 29,929 | 29,929 | yes | 99.8% |
| 2010 | 29,900 | 29,900 | yes | 99.7% |

Six of ten seeds were evaluated at about **half** the shipped budget. §3 of this
preregistration evaluates "each run's **final checkpoint** (`checkpoint.pkl` at
update 30,000)"; that description is false of this artifact, so the campaign
did not execute the registered study. Pairing survives — both arms of a seed
sit at an identical `n_updates`, because the tick stream is seed-determined —
but a mixture of six half-budget and four full-budget paired comparisons is not
the design that was frozen, and no verdict may be issued from it.

**What is done about it.**

1. The 2026-08-17 campaign is **quarantined, not analysed**. Its directory is
   retained under `gate-confirmatory-20260817-DEFECTIVE-ckpt-budget/` for audit.
   Its 20 evaluation CSVs remain unread. The single authorized analysis run of
   this preregistration is therefore still unspent.
2. `run_arm.sh` now sets `--checkpoint_interval=100` (ticks, about 50 updates).
   `safe_checkpoint` writes to a temp file and `os.replace`s onto
   `checkpoint.pkl`, and `archive_interval=0`, so this overwrites in place and
   costs I/O but no storage. It changes no scientific parameter, consumes no
   randomness, and leaves training bit-identical.
3. A new fail-closed guard: `verify_checkpoint_budget.py` reads each
   checkpoint's stored `n_updates` into `ckpt_budget.json`, and the analyzer
   now **requires** that file and refuses any cell whose evaluated checkpoint
   holds fewer than 29,900 updates. The old `logs.csv` bound is relaxed to a
   loose 29,000 sanity check, because `logs.csv` is flushed every
   `log_interval` ticks and its final row legitimately sits up to one interval
   short of the budget — the 29,990 threshold was itself calibrated on three
   early cells and was wrong.
4. The campaign is **re-run** on the same seeds 2001–2010, same two arms, same
   frozen configuration, changing only `checkpoint_interval`. Under §6 seeds
   are never substituted and the block is never extended; both hold.

**Why re-running is legitimate here.** This is not a re-run for a scientific
reason and not a response to any outcome: nothing about any endpoint is known
to anyone. Training executed correctly and to budget in all 20 runs (`rc=0`,
zero failures); what failed was the artifact-writing step, which saved the
wrong model state. The final weights were never written to disk and cannot be
recovered, so re-execution is the only way to obtain the registered study.

**What this does not change.** No claim in the manuscript rests on this
campaign; the AMaze development negative already reported there is a separate,
clearly-labelled five-seed development sweep at 5,000 updates that used
last-logged in-training evaluation rather than checkpoints, and is unaffected.

### Operational verification 2026-08-19, 8/20 runs into the re-run (outcome-blind)

The re-run's fix was checked mid-campaign on the completed cells, reading only
each checkpoint's stored `n_updates` — a provenance field, not an endpoint. No
evaluation output exists yet and none was opened.

| cell | stored `n_updates` |
|---|---|
| `arm-plrMM-s2001` / `arm-plrGate-s2001` | 29,975 / 29,975 |
| `arm-plrMM-s2002` / `arm-plrGate-s2002` | 29,990 / 29,990 |
| `arm-plrMM-s2003` / `arm-plrGate-s2003` | 29,971 / 29,971 |
| `arm-plrMM-s2004` / `arm-plrGate-s2004` | 29,960 / 29,960 |

All within 40 updates of the 30,000 budget and above the analyzer's 29,900
guard, against 14,899–15,124 for six of ten seeds in the quarantined campaign.
Paired arms agree exactly per seed, confirming the tick stream is
seed-determined and the pairing is intact. Wall time 8,189–10,141 s per run at
concurrency 2.
