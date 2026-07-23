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

## Currently executing (GPU queue, in order)

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

**Parallel CPU P2 (done): corrected tile-coded MountainCar paired study.** Ten paired
seeds, at least 500,000 transitions per condition, 64 fixed common-random-number
evaluation episodes per target, and a task-agnostic shared tile policy. The
environment uses official MountainCar-v0 dynamics with custom nested binary
thresholds, so pass-rate AUC is not standard Gymnasium return. AUC mean ± sample
SD is 0.389±0.071 uniform, 0.414±0.081 exact adv-mass teacher γ=1,
0.530±0.059 exact adv-mass teacher γ=4, 0.720±0.029 γ=4 + centered hindsight,
and 0.727±0.023 γ=4 + success-only hindsight. The per-bin centered control is
0.229±0.031.

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
