# Experiment schedule & tracking

*Living document — current through 2026-08-09. GPU times are A10G wall-clock;
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
| Acrobot fixed-pool curriculum tournament | ✅ V2 complete; registered primary confirmed | all 9 development and 60 confirmation runs valid; target-uniform transition-AUC means: uniform `.6452319465`, `p(1-p)` `.6390719632`, `u_16` `.6871056515`; primary `u_16-p(1-p)=+.0480336884`, CI `[.0209366676,.0738485654]`, exact `p=.0033607483`, 15/20 positive, clearing both frozen filters; Holm secondary `u_16-uniform=+.0418737050` supported and `p(1-p)-uniform=-.0061599834` not supported | preserve the [V2 result](frontier_rl/examples/ACROBOT_CURRICULUM_TOURNAMENT_RESULTS.md); do not expand its score-shape claim into a named-method claim |
| Acrobot paid-probe ProCuRL selection attachment | ✅ 12/12 development and 320/320 confirmation runs complete | registered `u_16-ProCuRL` fixed-paid-AUC mean `+.004894`, `t(79)=1.9773`, `p=.05149`, below the `.02` SESOI: unsupported. Probed-arm paid-transition fractions were about `.932`; ordinary-uniform AUC was `.65149` versus `.33771` ProCuRL-env, `.33942` probe-sham, and `.34261` `u_16` | preserve the [result](frontier_rl/examples/ACROBOT_PROCURL_SELECTION_RESULTS.md); report only probe-cost domination at this frozen refresh cadence; do not claim that full PPO ProCuRL is inferior or extrapolate to a cheaper cadence |
| Exact-probability Digits factorial | ✅ complete; internally frozen primary not supported | 24/24 fresh paired blocks; interaction `+.01589`, CI `[-.01686,+.04712]`, exact `p=.350`; RLOO reverses its predicted sampler preference and both matched samplers lose to uniform | keep as main-text negative evidence; compact release ships contrasts/receipts plus a 2,904-file manifest, while the 5.08 GB replay payload remains external |
| Capped-HORA robustness matrix | ✅ 800/800 exploratory runs complete | all 32 deployable adaptive-minus-fixed AUC means are positive; cap 32 cuts mean maximum group size 58.07% for `-.00271` sampler-averaged AUC versus uncapped; coefficient mass does not mediate the gain | appendix/supporting evidence only; no named-method validation or multiplicity-controlled claim |
| Acrobot V5A | ✅ complete and independently verified | 27/27 runs valid; all learning-outcome-field-blind gates passed; fresh `U*=250`; V5B authorized | preserve immutable V5A evidence |
| Acrobot V5B | 🛑 180/180 complete; procedural NO-GO | zero run failures and raw integrity passed, but the frozen analyzer failed exact diagnostic reconstruction; no primary result | retain V5B without rescue analysis; review a tolerance-aware verifier and seal fresh V5C seeds |
| Neural MountainCar V1R2 | 🛑 complete development NO-GO | all 15 runs/reconstruction checks passed, but feasibility failed: 1,932 dead, 474 mixed, 0 all-pass; hardest-goal AUC zero in every run | do not touch seeds `18000..18019`; design fresh V2 adequacy development |
| UniLab Stewart native V2 | ✅ 33 arm runs across nine seed-level multi-arm replicates complete | raw-mass mechanism and target-correction cost verified; gradient-moment sampler improved ESS/second moment but not AUC (`-.0004` vs uniform) | calibrate/shrink moments on a common charged stream; compare frozen sampler vs online tracker before any confirmation |
| Maze factorial wave 2 | ✅ fresh-wave result complete | fresh blocks are positive 6/6 under uniform and 6/6 under the frontier sampler (exact sign p=.03125 each); sampler-averaged independent blocks are positive in 12/12 descriptively across both waves; pre-specification is recorded externally but its lock object is absent | keep the cross-wave 12/12 statement descriptive; easy-band localization is suggestive only |
| Countdown reviewer arms | ✅ all six ARM-A/B runs complete | fixed-code strong gate failed; the faulty-decay under-gated point remains suggestive; higher-dose live-group replay exceeded recycling on both logged meters but cannot separate update dose from update direction; the logged best@16 field is a bootstrap proxy, not standard pass@16 | do not call the gate a validated operating point, a causal bound, or a relabel-specific test; recover task outcomes |
| GSM8K steering-controlled `g3p` | ⚠ complete but treatment not delivered | minimum dead-sampling criterion passed, but run mean 0.601480 missed the registered `<0.60` gate by 0.00148 | report the interaction as inconclusive by design; do not interpret endpoints causally |
| Paper/artifact repair | ✅ result integration, final rebuild, public-PDF synchronization, and anonymous release verification complete | the registry generator emits and checks exactly 562 records, including 441 Acrobot records. The ICLR wrapper has 13 total pages; its conclusion ends and references begin on page 9, and the appendix begins on page 10. The content-addressed release is frozen and verified; its containing Git commit is the repository publication record. Tier-0 SFT overlap is 27/128 and the 101-task numeric reanalysis remains blocked | preserve the checked anonymous receipts and release hash; publish the current research branch; do not launch another Mac experiment for this submission |
| Next GPU program | 🧭 handoff complete; no GPU job launched | [GPU handoff](GPU_EXPERIMENT_HANDOFF.md) ranks estimator-specific maze LR calibration first and makes the dose-matched Countdown control conditional on recovering/rebuilding its missing v2 execution assets; both lanes have disjoint seeds, power targets, closed schemas, outcome-blind gates, and release criteria | implement and independently seal Lane A before any engineering run; launch Lane B only after its asset and 10-GB feasibility gates pass |

Acrobot V2 earned the frozen **P+/U+** interpretation only: score-shape
evidence in one fixed eight-threshold Acrobot family under one shared H64,
640-parameter practical-MaxRL learner at `N=16`. It is not a full ProCuRL,
SFL, PLR, PAIRED, ACCEL, or ALP-GMM comparison; it has no held-out-task
generalization test or prospective power calculation, and its exact sign-flip
interpretation assumes paired-sign exchangeability. Source-lock, development
raw, development-gate, confirmation-raw, and locked-analysis SHA-256 hashes are
`0e6438d42ddc53b89d774233805c465dc562bb6be5f8ac93ecf8a4d09b5d9af3`,
`c616912569f4d19e36ea4a8685616a35bef037934e5c8d366ee7bd51bb2c3311`,
`6dc908e22e874550e0536f1fcd52f2b3a1768d1a89c510275bef7efc2e2baac6`,
`f533d0b84cdb3f7d3ede4bc4c94aa11e3b0ffc58c8bc7ea1a26491476873b2c6`,
and `463fa1a01d95922976f09f75b21f6d8f2c6a8d256081ebedfa4ba968a06f356b`.
These digests bind source, runtime, gate, and artifacts internally, but no
immutable public pre-execution commit in this checkout establishes their
timing.

The paid-probe selection attachment completed all 12 development runs and all
320 confirmation runs. Its registered `u_16-ProCuRL` fixed-paid-AUC contrast
was `+.004894` (`t(79)=1.9773`, `p=.05149`), below the `.02` SESOI and
therefore unsupported. About 93.2% of paid transitions in each probed arm were
probes. Ordinary uniform reached `.65149` fixed-paid AUC, whereas ProCuRL-env,
probe-sham uniform, and `u_16` reached `.33771`, `.33942`, and `.34261`.
This establishes probe-cost domination only for this frozen actor-only,
fixed-pool attachment and refresh cadence; it does not establish that full PPO
ProCuRL is inferior. The compact artifact includes the manifest and receipts,
while the 1,374,886,097-byte raw ledger remains external with SHA-256
`b1f8756c249effab8c77101c8bca73ddf708a5e143c18fe8742fd5712fdd7c12`.
The source/runtime/gate/artifact hashes bind the execution internally, but no
immutable public pre-execution commit establishes timing.

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

## Completed paper-critical queue (outcomes current 2026-08-08)

**Historical title decision, now resolved:** the title "The Estimator
Decides" was conditioned on wave-2 P-F2. P-F2 met its registered bar under
both samplers (6/6 each), so that branch permits the title. This does not
upgrade the easy-band localization, whose block-level interval crosses zero.


| # | run | status | purpose |
|---|---|---|---|
| FACT-W2 | confirmation factorial wave 2: {maxrl,grpo}×{uniform,frontier_un}× fresh blocks 6–11 | **DONE — P-F2 confirmed 6/6 under each sampler** (`maze_gpu/run_factorial_wave2.sh`, prereg d6aea90 committed pre-launch) | Exact sign p=.03125 per sampler. Across waves, all 12 sampler-averaged independent blocks are positive descriptively. P-F3 is only suggestive: 4 positive, 1 tie, 1 negative block average; interval crosses zero. |


| # | run | status | purpose |
|---|---|---|---|
| FACT | balanced maze factorial {maxrl,grpo}×{uniform,frontier_un}×6 blocks + grpo_mass + grpo_nostd (250 fixed steps) | **DONE — P-F1 FAILED, claim retracted in paper (e27b5d9); P-G0a confirmed; P-G0c failed. Verdict: `maze_gpu/FACTORIAL_VERDICT.md`. Repair pass folding in contention casualties.** | draft-review P0-3: P-F1 prereg, falsification branch committed — and executed |
| ARM-A/B | designed-gate B3 ×3 + live-group replay ×3 (Countdown) | **DONE** | ARM A refuted the strong-gate dial claim; the earlier faulty-decay under-gated point remains descriptive. ARM B used `ppo_epochs=2`, a higher update dose on every live group: it provides a higher-dose alternative improving both logged metrics, but is not a dose-matched direction test or causal bound. |
| OTG | E-LLM-2c one_target_per_group ×3 (Countdown; prereg `bdca4aa` P-OTG1/2) | **NO LOCALLY EVIDENCED COMPLETION** | Outside the current evidence registry and paper claims. |
| E-LLM-1b | steering-controlled GSM8K `g3p` | **DONE — treatment-delivery gate missed by 0.00148** | Run mean 0.601480 did not satisfy `<0.60`; interaction inconclusive by the committed branch. |

The earlier one-block interim read is superseded by the completed six-block
analysis and is not evidence. The registered coverage-AUC contrast is positive
in all six fresh blocks under each sampler; the easy-band localization did not
earn the same claim.

## New CPU results (2026-08-05, all prereg'd in-script, all in paper)

| experiment | verdict | artifact |
|---|---|---|
| gate-variants (Q1/Q8): freq heuristic vs true-p gate, 10 seeds | true-p gate keeps ~all recycling value (.879 vs ungated .881); freq gate pays (.798, 0/10); f_hat anti-correlates with true p(g') (−.27) | `results_gate_variants.json` |
| row-vs-group relabel (Q2/P0-2), 10 seeds | per-row-uncoupled .952 > one-target .881 > shared-K coupled .749 (≈ no-recycling .705); coupling is the cost, both orderings 10/10 | `results_row_vs_group.json` |
| schedule-matched + grpo_nostd (Q7), 5 seeds × 2 frozen schedules | no-SD GRPO collapses onto RLOO's coverage profile (.148 vs RLOO .161 vs GRPO .762) — variance normalization is the tail mechanism | `results_schedule_matched.json` |
| grpo-own-mass teacher (Q6), 5 seeds | GRPO scheduled by its own mass functional does NOT close the gap (5/5) and serves GRPO worse than the u_N teacher (0/5) | `results_grpo_own_mass.json` |

## Historical queue snapshot (2026-07-23)

*The status words in this table are retained as a dated execution record, not
as current run state. Current GSM8K and paper-critical outcomes are in the
tables above.*

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

## Next-step tracking

**Submission stop rule:** every planned Mac experiment is complete. Work now
stops at evidence recovery, registry generation, and the final paper/artifact
rebuild. PLR, PAIRED, ACCEL, and full SFL comparisons are post-submission
projects, not prerequisites for this draft.

| priority | experiment | est. | prerequisite |
|---|---|---|---|
| P0 ✅ | **Compact ICLR main-paper rebuild** — coefficient activity as hypothesis generator, mixed positive/negative tests, qualified neural diagnostics | 13-page ICLR PDF; main text fits within 9 pages; exact 562-row registry; public PDF synchronized; anonymous clean-release check passed | preserve the final anonymous receipts and release hash |
| P0a | **Recover central raw evidence** — 24 wave-2 checkpoint trajectories plus complete Countdown B1/B2 task-level outcomes and seed records | artifact retrieval, not new training | external execution storage; needed for AUC multiverse and standard pass@16 |
| P0b | **Recover tier-0 clean-subset inputs if they still exist** — frozen SFT/eval manifests plus all 16 per-task outcomes per retained arm/seed, or compatible checkpoints | artifact retrieval, not new training | external execution storage |
| P0c ✅ | **Fresh Acrobot fixed-pool tournament V2** — practical MaxRL with uniform vs `p(1-p)` vs `u_16`, 20 fresh paired seeds, common nominal transition budget with bounded complete-group overshoot | complete: 9/9 development and 60/60 confirmation runs valid; primary `u_16-p(1-p)=+.0480336884`, CI `[.0209366676,.0738485654]`, exact `p=.0033607483`; frozen +.01 point-estimate and `p≤.05` filters both passed | preserve [result and caveats](frontier_rl/examples/ACROBOT_CURRICULUM_TOURNAMENT_RESULTS.md) |
| P0d ✅ | **Paid-probe ProCuRL selection attachment** — ProCuRL-env, probe-sham uniform, ordinary uniform, and range-matched `u_16`; 80 paired seeds | 320/320 complete; registered primary `+.004894`, `t(79)=1.9773`, `p=.05149`, below `.02` SESOI and unsupported | preserve the narrow cadence-specific probe-cost interpretation; no full-PPO inferiority claim |
| P1 | **Efficiency eval of F1's long-horizon checkpoint** — does 4× training turn into inference-time speedup at deep levels? | deferred historical branch | retained checkpoint required |
| P2 ✅ | **Corrected tile-coded MountainCar benchmark** — 10 paired seeds, ≥500k transitions/condition, γ and hindsight ablations, shared/per-bin control | done; results and family-corrected tests above | none |
| P3 ✅ | **Balanced maze factorial and fresh-wave confirmation** | complete: wave-1 endpoint claim failed; wave-2 coverage-AUC contrast landed 6/6 under each sampler; registration timing is externally recorded, not locally auditable | independent-block analysis checked in |
| P4 ↪ | **Streaming-pool teacher prototype** (parametric density over a continuous difficulty axis, ALP-GMM-style) | deferred post-submission | requires a separately frozen study; not a submission blocker |
| P5 | SmolLM2-360M + GSM8K 2×2 plus steering-controlled follow-up via `verl_integration/` | complete; follow-up missed delivery gate | no causal interaction claim; further GPU reruns are deferred |
| P6 🛑 | **Neural MountainCar capacity-matched development** — 3 seeds × 5 cells, hardest-goal primary | complete NO-GO; no confirmation | fresh V2 adequacy design |
| P7 🛑 | **Acrobot optimizer-matched hindsight V5B** — 20 seeds × 9 cells, update-matched `U*=250` | 180/180 complete; frozen verifier exact-equality failure makes the primary family a procedural NO-GO | fresh V5C with a prereviewed tolerance-aware verifier |
| P8 ↪ | **UniLab calibrated gradient-moment allocation** — uniform sham calibration vs refreshed calibrated `rho/q` | deferred post-submission | no further Mac execution for this submission |

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
