# Experiment schedule & tracking

*Living document — updated as runs complete. GPU times are A10G wall-clock;
local Gymnasium studies report serial Mac CPU wall-clock separately.*

> **July 2026 audit.** The GPU E/F entries below are historical and
> provisional: those runs used the legacy `u_{N+1}` score, mixed `K=0` and
> `K=N` in the old zero-weight counter, measured path length rather than BFS
> depth for hindsight, trained all levels with the deepest response budget,
> and scaled dense-hindsight loss with relabel count. They are retained as an
> execution record, not as corrected validation. Historical GPU AUC values
> were step-indexed without the post-SFT anchor despite
> wall-clock-matched endpoints. The tile-coded MountainCar P2 result below uses the repaired
> estimator and evaluation stack. The old CartPole three-seed smoke run has
> not been rerun under that protocol and is excluded from current evidence.

## Current local studies

| study | status | completed decision | next allowed action |
|---|---|---|---|
| Acrobot V5A | ✅ complete and independently verified | 27/27 runs valid; all learning-outcome-field-blind gates passed; fresh `U*=250`; V5B authorized | preserve immutable V5A evidence |
| Acrobot V5B | 🛑 180/180 complete; procedural NO-GO | zero run failures and raw integrity passed, but the frozen analyzer failed exact diagnostic reconstruction; no primary result | retain V5B without rescue analysis; review a tolerance-aware verifier and seal fresh V5C seeds |
| Neural MountainCar V1R2 | 🛑 complete development NO-GO | all 15 runs/reconstruction checks passed, but feasibility failed: 1,932 dead, 474 mixed, 0 all-pass; hardest-goal AUC zero in every run | do not touch seeds `18000..18019`; design fresh V2 adequacy development |
| UniLab Stewart native V2 | ✅ 33 arm runs across nine seed-level multi-arm replicates complete | raw-mass mechanism and target-correction cost verified; gradient-moment sampler improved ESS/second moment but not AUC (`-.0004` vs uniform) | calibrate/shrink moments on a common charged stream; compare frozen sampler vs online tracker before any confirmation |

V5A's projected 180-run serial runtime was `7.0557400375` hours. Passing V5A
is authorization evidence only. V5B completed all 180 runs with zero run
failures and intact raw records (53,510 groups, 45,000 updates, and 1,080
checkpoints), but its frozen exact-reconstruction rule failed on 377/720
step-norm diagnostic floats. The maximum difference was
`1.9984014443252818e-15` (11 ULP). The official primary family is therefore a
procedural NO-GO and has no reported outcome. The post-hoc compatibility audit
passed the remaining checks but is non-authorizing. The source locks are local
pre-execution locks, not externally timestamped preregistrations. See the
[V5B verification erratum](frontier_rl/examples/ACROBOT_HINDSIGHT_V5B_VERIFICATION_ERRATUM.md)
and [forensic report](frontier_rl/examples/acrobot_hindsight_v5b_forensic_verification.json).

## Currently executing (2026-08-05: balanced factorial wave)

**Title decision, committed in advance (fresh-eyes review finding 5):**
the title "The Estimator Decides" is conditional on wave-2 P-F2. If
P-F2 confirms (covAUC ordering >=5/6 both samplers on fresh blocks),
the title stands on that registered result. If P-F2 fails, the title
becomes the subtitle ("What Curricula and Failure Recycling Can and
Cannot Do in RL with Verifiable Rewards") — the slogan may not outlive
the claim it names.


| # | run | status | purpose |
|---|---|---|---|
| FACT-W2 | confirmation factorial wave 2: {maxrl,grpo}×{uniform,frontier_un}× fresh blocks 6–11 | **RUNNING** (`maze_gpu/run_factorial_wave2.sh`, prereg d6aea90 committed pre-launch) | P-F2: the 12/12 exploratory covAUC ordering becomes the registered primary on new blocks; falsification branch: ≤4/6 either sampler → no cross-estimator coverage claim of any kind at neural scale. P-F3: easy-band majority sign test |


| # | run | status | purpose |
|---|---|---|---|
| FACT | balanced maze factorial {maxrl,grpo}×{uniform,frontier_un}×6 blocks + grpo_mass + grpo_nostd (250 fixed steps) | **DONE — P-F1 FAILED, claim retracted in paper (e27b5d9); P-G0a confirmed; P-G0c failed. Verdict: `maze_gpu/FACTORIAL_VERDICT.md`. Repair pass folding in contention casualties.** | draft-review P0-3: P-F1 prereg, falsification branch committed — and executed |
| ARM-A/B | designed-gate B3 ×3 + replay control ×3 (Countdown) | **RUNNING** — s1 done; s2 OOMed near end (marker withheld, no step-60 ckpt); s3 in flight; repair pass armed behind the driver lock (`smollm/run_reviewer_arms_repair.sh`) | fixed-code gate validation + dose-matched replay |
| OTG | E-LLM-2c one_target_per_group ×3 (Countdown; prereg `bdca4aa` P-OTG1/2) | queued behind ARM-B + chained jobs | P0-2 LLM-side test: does the shared-K coupling penalty transfer through verl normalization? |
| E-LLM-1b | steering-controlled GSM8K (m3s died at step-25 ckpt, node OOM) | chained behind ARM-B completion | decisive LLM-scale cell |

Seed-0 block interim read (1/6 blocks — NO conclusions): uniform sampler
maxrl +.024 vs grpo −.130 Δcov8 (easy band −.396, as predicted); teacher
sampler REVERSES it this block (maxrl −.038 vs grpo −.019) — under the
u_N teacher, GRPO's coverage loss shrank and MaxRL's went negative.
If that pattern holds across blocks, P-F1's "both samplers" clause fails
and the falsification branch executes (claim rescopes from "the
estimator decides" toward sampler-conditioned). Exactly what the
factorial exists to decide; wait for 6 blocks.

## New CPU results (2026-08-05, all prereg'd in-script, all in paper)

| experiment | verdict | artifact |
|---|---|---|
| gate-variants (Q1/Q8): freq heuristic vs true-p gate, 10 seeds | true-p gate keeps ~all recycling value (.879 vs ungated .881); freq gate pays (.798, 0/10); f_hat anti-correlates with true p(g') (−.27) | `results_gate_variants.json` |
| row-vs-group relabel (Q2/P0-2), 10 seeds | per-row-uncoupled .952 > one-target .881 > shared-K coupled .749 (≈ no-recycling .705); coupling is the cost, both orderings 10/10 | `results_row_vs_group.json` |
| schedule-matched + grpo_nostd (Q7), 5 seeds × 2 frozen schedules | no-SD GRPO collapses onto RLOO's coverage profile (.148 vs RLOO .161 vs GRPO .762) — variance normalization is the tail mechanism | `results_schedule_matched.json` |
| grpo-own-mass teacher (Q6), 5 seeds | GRPO scheduled by its own mass functional does NOT close the gap (5/5) and serves GRPO worse than the u_N teacher (0/5) | `results_grpo_own_mass.json` |

## Previous wave (2026-07-23, standing loop)

| # | run | status | purpose |
|---|---|---|---|
| G1 | GSM8K 2×2 cell 2: maxrl, cur=false (50 steps) | **RUNNING** (~25 min/step) | MaxRL baseline at LLM scale |
| G2 | GSM8K 2×2 cell 3: grpo, cur=true | queued | **P-G2: the H6-at-LLM-scale test** |
| G3 | GSM8K 2×2 cell 4: grpo, cur=false | queued | GRPO baseline |
| G4 | GSM8K cell 1 re-run: maxrl, cur=true | queued (requeue_cell1.sh) | teacher channel at LLM scale (collision casualty re-run) |
| A1 | docs-evidence audit workflow v2 | running | AUDIT.md: every number vs logs |
| — | PR #1 union merge (`pr1-union-proposal` fce7fc5) | **awaiting human merge** | see PR1_REVIEW_VERDICT.md |

Pre-registered predictions for G1–G4: `curriculum_maxrl/GSM8K_A10G_PLAN.md`
(P-G1..P-G5, committed 39520fa before any cell finished). Analysis contract:
AUC of val reward@1 on the common step grid + final pass@8 + dead-fraction
trajectories; partial cell-1 excluded from headlines.

Watcher: `smollm/watch_2x2_event.sh` — exits (=> notification) on cell
transition, new OOM, checkpoint, or queue drain.

## Recently completed (maze wave, 2026-07-22)


| # | run | duration | purpose | decision it feeds |
|---|---|---|---|---|
| E1 ✅ | ck_uniform_maxrl (2400 s + ckpt) | done | efficiency baseline | — |
| E2 ✅ | ck_uniform_grpo (2400 s + ckpt) | done | efficiency baseline | — |
| E3 ✅ historical | ck_frontier_alp_maxrl_hsd (2400 s + ckpt) | done | archived checkpoint run | audited protocol; no champion claim |
| E4 ⚠ archived | eval_efficiency over E1–E3 | not reproducible as shipped | samples-to-coverage | Historical post-hoc targets gave 0.5× to 11×; checkpoints were not retained, so no monotone difficulty claim. Rerun with saved checkpoints, fixed RNG, and one preregistered target. |
| F1 ✅ | long_falp_hsdense (9600 s) | done | is level 6+ a duration question? | **NO** — mean 0.258→0.269, L5 doubles, L6 stays ≈0.01. Mechanism revision needed at depth; CPU-validate depth-scaled move budgets / param-sharing check first |
| F2 ✅ | matched_falp_p4_hsdense (2400 s) | done | γ=4 on GPU | **did not improve the historical run** (legacy unanchored step-AUC 0.231 vs 0.236); γ stays 1 for the corrected maze rerun pending clean evidence |
| F3 ✅ historical | dense-hindsight seed 1 (2400 s) | done | exploratory three-seed point estimates | see F4 |
| F4 ✅ | dense-hindsight seed 2 (2400 s) | done | exploratory multi-seed | 0.252±0.005 final / 0.229±0.009 legacy unanchored step-AUC; positive deltas for two configurations across three seeds, but no inferential test and audit confounds prevent a reliability claim |

**GPU QUEUE DRAINED (all E and F runs complete).** Next wave now unblocked.

**Parallel CPU (done): MountainCar categorical result** — flag-only 0.000 →
uniform 0.058 → teacher 0.664 → **full stack 0.848±0.058** (corrected 10-seed transition-matched study; the earlier 0.889/0.944/1.000 figures had no artifact and are retracted); plus
the transfer lesson (per-bin params never reach the flag: curricula operate
through shared parameters).

Paired AUC effects with 95% bootstrap CIs that survive Holm correction of exact
sign-flip tests are γ=4 over uniform (+0.141 [0.076, 0.202]), γ=4 over γ=1
(+0.116 [0.060, 0.172]), centered hindsight over none (+0.191 [0.155, 0.231]),
success-only hindsight over none (+0.197 [0.160, 0.238]), and shared centered
over per-bin centered (+0.492 [0.464, 0.522]). The same family does not support
γ=1 over uniform, exact adv-mass over legacy, exact adv-mass over learnability,
or centered over success-only. The shared/per-bin control supports transfer through shared
parameters, but also changes model capacity and data sharing.

Historical watcher: `watch_gpu.sh`; the recorded queue is drained and no watcher is claimed active.

**Parallel CPU P6 (done): neural MountainCar V1R2 development.** This is not
the positive tile-coded P2 study above. V1R2 used five neural conditions:
frontier/shared H64, uniform/shared H64, hardest-only/shared H64,
uniform/disjoint-total H8×8, and uniform/disjoint-active H64×8. All 15 runs
and independent reconstruction checks passed, but the predeclared adequacy gate
returned NO-GO. The native hardest-goal AUC was zero in all 15 runs; pooled
regimes were 1,932 all-fail, 474 mixed, and zero all-pass. Supporting mean-pass
AUC deltas were `+0.0065104`, `+0.0119792`, `+0.00546875`, and `+0.00429688`
in the registered contrast order. They are development-only descriptions and
do not authorize a performance claim. Confirmatory seeds remain untouched.

## Decision tree after the queue drains

```
Corrected efficiency table
├─ retained checkpoints and one preregistered target show separation
│    → report seeded curves with uncertainty
└─ otherwise → keep the archived table out of headline evidence

F1 long-horizon
├─ level 6 leaves 0 by 9600 s
│    → run F1b: same budget, uniform baseline (is it the schedule or just time?)
└─ still 0 → implement depth-scaled move budgets (hindsight-min-depth
     curriculum); CPU-validate first, then one 2400 s GPU run

F2 γ=4
├─ AUC ≥ dense-hindsight baseline +0.003 → set teacher-power=4 default for
│    level-structured tasks in maze + frontier_rl docs
└─ tie/worse → document as CPU-only effect (compounding weaker at 13 levels
     than 36 tasks); keep γ=1 GPU default

F3–F4 historical seeds
├─ point-estimate ordering holds → report descriptively with audit caveats
└─ margins overlap seed variation → make no ranking claim
```

## Next-wave tracking

| priority | experiment | est. | prerequisite |
|---|---|---|---|
| P1 | **Efficiency eval of F1's long-horizon checkpoint** — does 4× training turn into inference-time speedup at deep levels? | 30 min | F1 |
| P2 ✅ | **Corrected tile-coded MountainCar benchmark** — 10 paired seeds, ≥500k transitions/condition, γ and hindsight ablations, shared/per-bin control | done; results and family-corrected tests above | none |
| P3 | **Corrected maze factorial** — uniform vs exact `u_N` vs legacy `u_{N+1}` vs learnability at γ=1, followed by a hindsight ablation | GPU | audited training stack |
| P4 | **Streaming-pool teacher prototype** (parametric density over a continuous difficulty axis, ALP-GMM-style) — unblocks procedural/generative task sources; CPU-validate on a continuous-difficulty variant of grid_reach | CPU | none |
| P5 | SmolLM2-360M + GSM8K 2×2 via `verl_integration/` | 8-GPU node | **blocked on hardware** |
| P6 🛑 | **Neural MountainCar capacity-matched development** — 3 seeds × 5 cells, hardest-goal primary | complete NO-GO; no confirmation | fresh V2 adequacy design |
| P7 🛑 | **Acrobot optimizer-matched hindsight V5B** — 20 seeds × 9 cells, update-matched `U*=250` | 180/180 complete; frozen verifier exact-equality failure makes the primary family a procedural NO-GO | fresh V5C with a prereviewed tolerance-aware verifier |
| P8 | **UniLab calibrated gradient-moment allocation** — uniform sham calibration vs refreshed calibrated `rho/q`; 32 groups/task at 0 and 60, disjoint 16-group/task audits | 37,209,600 Mac-CPU transitions on development seeds 3–5 | implement the frozen post-V2 protocol and fail-closed mechanism analyzer |

## Standing cadence

- After every completed run: analyze → update EXPERIMENTS.md/REPORT.md →
  sync logs to this repo → push.
- Website results tables refreshed when a headline number changes.
- Every negative result gets written up with its diagnosis (the H6 reversal
  and conditioning-rewrite lesson were the two most valuable findings so far).

## Risks being tracked

- **Shared GPU:** another user's job took the card once already; the final
  sweep waits politely (`pgrep` loop) and the watcher distinguishes
  "waiting" from "stalled".
- **Seed noise:** finals vary ±0.01–0.015 across seeds; no single-seed claim
  goes in REPORT.md without either multi-seed confirmation or an explicit
  single-seed caveat.
- **Toy→real gap:** every CPU win must re-prove itself outside the toy. γ=4
  helped corrected MountainCar but did not help the historical GPU maze;
  concentration is a task-graph knob, not a universal default.
- **Neural MountainCar headroom:** V1R2 produced zero hardest-goal AUC and no
  all-pass groups. Treat this as an adequacy failure, not a zero effect or a
  contradiction of the older tile-coded mean-pass study.
- **Post-hoc rescue:** V5B's frozen analyzer did not accept the completed
  artifact, so no cell performance or primary contrast may be reported. The
  compatibility audit diagnoses a numerical verifier defect but cannot
  authorize the family or tune V5C; use fresh seeds after preregistering the
  tolerance rule.
- **Historical source availability:** V3 and later manifests match current
  bytes. V2's locked runner hash refers to bytes not present at HEAD; retain
  that mismatch in every external review.
