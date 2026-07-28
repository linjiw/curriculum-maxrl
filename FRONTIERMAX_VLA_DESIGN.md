# FrontierMax-VLA — design doc & pre-registration draft

*v1.0, 2026-07-28. For the VLA team. Target: LfH's exact benchmark
(arXiv:2607.09042 — LIBERO-PRO OOD, RLinf-Pi05-LIBERO-SFT checkpoint,
GRPO group rollouts, matched rollout budgets, 4 seeds), with a design that
is stronger at every point where LfH is weak. Companion reading:
LFH_NOTES.md (their method + our takes), SHARPENING_SYNTHESIS.md (the
coverage-cost theory), COUNTDOWN_ANALYSIS.md + DYNAMICS_ANALYSIS.md (what
we learned running recycling at scale), EVIDENCE.md (validated defaults).*

---

## 0. One-paragraph thesis

LfH showed hindsight relabeling recovers GRPO's discarded all-fail groups
for VLA post-training (5× sample efficiency on LIBERO-PRO) — using a 235B
VLM as both relabeler and judge, with no admission control, no curriculum,
and no coverage accounting. FrontierMax-VLA replaces the judge with the
simulator's own predicates (exact, free, theorem-backed), adds the derived
admission rule that prevents the coverage collapse we measured on
Countdown, adds the frontier teacher in exactly the task-count regime
where its posterior is well-fed, and reports the metric their evaluation
cannot see (pass@k). Every design choice below is either validated in this
repo or pre-registered as a prediction.

## 1. Setting (identical to LfH — non-negotiable for comparability)

- **Policy**: `RLinf-Pi05-LIBERO-SFT` (public; π0.5-class VLA,
  flow-matching action head). Backbone generality pass afterward on GR00T
  / OpenVLA-OFT as LfH did.
- **Tasks**: LIBERO-PRO OOD perturbations (object / target-state / action
  swaps on LIBERO-90 scenes) — the low-success regime where the initial
  policy fails almost everything.
- **RL loop**: RLinf (their infra, open-source) — group size K per
  instruction, binary env success reward, KL-to-reference per their
  config. Budget protocol: fixed collected-rollout count, gain metric
  Gain(t) = SR_t/SR_0 − 1, eval with 480 rollouts, 4 seeds, mean±SE.
- **Their numbers to beat**: 5× sample efficiency over GRPO; final gain
  ~2.0 vs GRPO ~0.9 (their Fig. 3a); groups-kept 70–80% vs 20–40%.

## 2. The five components (each mapped to existing code)

### 2.1 Predicate relabeler — the verifier IS the relabeler
LIBERO goals are BDDL predicate conjunctions readable from simulator
state. A failed group's achieved predicate sets are extracted from final
states; the relabel target is a *verified achieved sub-conjunction*, and
the instruction is rebuilt from a **fixed template per predicate set**
(P6 contract 2 — never free-generated).
- Code: `frontier_rl/adapters/cosmos_libero.py::CosmosLiberoSpace.relabel`
  (implemented, adversarially reviewed, mock-pilot-validated:
  oracle-relabel 0.862 vs 0.000 baselines) +
  `cosmos_live.py::LiveRolloutBackend` (wave loop matching the real
  closed-loop eval; goal_predicates_of for BDDL→canonical).
- Delta vs LfH: zero VLM calls, zero judge noise, exactness guarantee.
- **Built-in experiment**: run LfH's VLM relabeler in shadow mode on the
  same failed groups and score its instructions against the simulator
  predicates → a measured VLM-relabel error rate (precision/recall per
  predicate class). This is `PoisonRateMeter` (implemented + tested)
  pointed at their judge. Publishable table regardless of who wins.

### 2.2 Shared-anchor contrastive relabeling (take from LfH, improve)
Relabel the WHOLE group to one anchor target g′ (their argument: group
advantages need comparable rewards — matches our design review's
mixed-target concern). Improvement: score every trace against g′ with the
EXACT verifier, so the relabeled group has true K-of-N contrast
(successes AND failures). All-positive relabeled groups are pure
sharpening pressure under mean normalization (measured:
DYNAMICS_ANALYSIS.md — HS cells' train-reward advantage was 100% injected
relabels); contrastive groups restore the estimator's negative space.
- Anchor choice: the trace with the DEEPEST achieved sub-conjunction
  (maximizes the chance other traces partially satisfy g′).
- Code change: `relabel(mode="shared_anchor")` in cosmos_libero — small;
  the per-trace mode stays for the ablation.

### 2.3 Utility-gated admission (the derived mitigation — our math)
Admit a relabel only if the destination's estimated pass rate sits in the
band: maintain a decayed Beta posterior over relabel-destination hit
rates; reject g′ with p̂ > 0.5. Theory: u(p) = pass@N − pass@1 → 0 as
p → 1 (P1); training there buys sharpening, not signal (GCSL
marginal-shift bound + curation theorem; SHARPENING_SYNTHESIS.md).
- Code: implemented for Countdown in `verl/utils/hindsight.py`
  (utility_gate); port = swap the value-key for the predicate-set key.
- LfH has no equivalent (no posterior over tasks at all).

### 2.4 Frontier teacher over the task pool (the regime is finally right)
Thompson sampling ∝ u(p̃)^γ over LIBERO-PRO's task list (γ=1, decay 0.7,
floor 0.1 — validated defaults). LIBERO-PRO has ~dozens of tasks →
hundreds of visits per task at LfH's budget: the posterior-starvation
failure of GSM8K (0.4 visits/prompt) is structurally impossible, and the
pool HAS dead and mastered tasks (OOD perturbations at ~0%; some swaps
trivial), which is the waste-avoidance regime the maze validated
(+22–35% steps/hour).
- Code: `FrontierTeacher` unchanged; sampler = the per-chunk-redraw
  CurriculumSampler (post-fix) over the task list.
- HONESTY GATE (from E-LLM-1/2 lessons): log the axis-validation table
  (per-task p̂ vs realized success; visit distribution vs uniform) BEFORE
  claiming allocation effects; report step-matched numbers beside
  budget-matched (Opus review M3).

### 2.5 Estimator: MaxRL weights via the positive-part form + λ loss weight
π0.5's flow-matching head has no tractable per-sample log-prob → the
**positive-part weighted-RFT estimator** (w⁺ = 1/K − 1/N on successes;
E[Σw⁺·S] = Σ_{k=2..N}(1/k)∇pass@k and E[Σw⁺] = u(p), both MC-verified;
measured variance cost on the anchor: AUC 0.887→0.828, price is variance
not bias). Where the backbone exposes log-probs (OpenVLA-OFT),
full MaxRL weights + the H6 safety comparison vs GRPO.
- Hindsight term enters at LOSS level with weight λ (LfH's working dose
  knob; our reward-level scale is a no-op under normalization). λ=1
  default, λ ∈ {0.5, 1} if budget allows.
- Their hindsight importance ratio π_θ(a|g′)/π_θold(a|g): adopt for the
  GRPO arms (unbiased hindsight-PG); for weighted-SFT arms it reduces to
  our rewrite-before-scoring. Evaluate both on one cell before committing.

## 3. Experiment grid (pre-register verbatim before any GPU)

Same budget protocol as LfH, 4 seeds:

| arm | relabeler | estimator | gate | teacher | tests |
|---|---|---|---|---|---|
| A1 GRPO | — | GRPO | — | — | their baseline, reproduce |
| A2 GRPO+LfH | VLM (their recipe) | GRPO | — | — | reproduce their result + shadow-score the VLM |
| A3 GRPO+predicates | **predicates** | GRPO | — | — | exact vs judged relabels, same estimator |
| A4 FrontierMax-VLA | predicates | **positive-part MaxRL** | **on** | **on** | the headline |
| A5 = A4 − gate | predicates | positive-part | off | on | sharpening mitigation at VLA scale |
| (A6 budget-permitting) = A4 − teacher | predicates | positive-part | on | — | allocation isolation |

**Metrics (all arms)**: their Gain(t) + groups-kept; **pass@k (k=1,4,8,16)
per task tier** — the coverage currency; relabel precision vs simulator
predicates (A2's VLM vs A3's oracle — the PoisonRateMeter table);
easy-task retention (SONIC Q8 meter); step-matched AND budget-matched.

**Pre-registered predictions (freeze before launch):**
- P-V1: A3 ≥ A2 on gain at equal budget (exact relabels ≥ judged) with
  measured VLM error rate as the mechanism witness.
- P-V2 (the risky one): A2 (LfH's recipe) LOSES pass@16 relative to A1's
  trajectory on tasks where its relabel dose is high — our sharpening
  finding as a prediction about their method, on their benchmark.
- P-V3: A4 ≥ A3 (gate + teacher add on top of exact relabels) on gain,
  AND A4 retains coverage where A5 loses it (gate = mitigation).
- P-V4: teacher telemetry shows real allocation (visit distribution
  diverges from uniform; dead-task sampling falls) — the axis-validation
  gate, checked BEFORE any P-V3 allocation claim.
- Kill criteria: if A2 fails to reproduce LfH within their error bars,
  stop and debug infra before any comparison claims; if the pool turns
  out learnable-everywhere (A1 near-saturates), swap in the harder
  LIBERO-PRO split before running A4–A6.

## 4. Compute plan

- **Does NOT fit the A10G**: π0.5 (~3B) + LIBERO sim + RL training needs
  ≥ A100-80GB (LfH used RLinf multi-GPU). Ask: one A100/H100 node for
  ~7–10 days (grid ≈ 6 arms × 4 seeds × 40 steps at their per-step cost),
  or their exact 4×A100 RLinf config for ~3 days.
- **CPU-buildable NOW (this repo, before any GPU)**: shared-anchor mode in
  cosmos_libero + tests; the gate port (predicate-set keys); the
  PoisonRateMeter shadow-scorer harness; RLinf integration of
  FrontierTeacher + the loss-level λ; pilot0 gates (0a variance / 0b
  poison / 0c surrogate-cosine) re-pointed at LIBERO — ALL have tested
  scaffolds already.
- **Pilot sequence (from READINESS.md, adapted)**: pilot0 gates → 1-seed
  A1/A2/A3 smoke (kill criteria live) → full grid.

## 5. Risks, honestly

1. **A2 reproduction risk** — their recipe has unpublished VLM prompt
   details; mitigation: their appendix + RLinf defaults, and the kill
   criterion above.
2. **Predicate relabels may be COARSER than language relabels** (BDDL
   sub-conjunctions vs free-form "pick up the mug") → fewer distinct
   relabel targets. Counter: exactness + the gate may matter more than
   diversity; and A2-vs-A3 measures exactly this trade.
3. **Positive-part variance at VLA scale** is unmeasured (CPU cost was
   −0.06 AUC); the OpenVLA-OFT arm (real log-probs) hedges this.
4. **Sharpening may not appear at LfH's dose** (their λ and relabel rate
   are lower than our Countdown cap=48 regime) → P-V2 could null. That
   is itself informative: it would locate the dose threshold between
   their operating point and ours.
5. Team unfamiliarity with RLinf — budget 2–3 days of infra ramp; the
   verl experience transfers (same GRPO loop shape).

## 6. What this buys the research program

If P-V1–P-V3 land: FrontierMax-VLA beats the nearest-neighbor method on
its own benchmark with less machinery (no 235B judge), a theory-backed
admission rule, and the coverage accounting the field is missing — the
three-channel story validated in robotics, which was the project's
original goal. If P-V2 alone lands: the sharpening finding generalizes
beyond Countdown, and the safety-paper framing gets its third scale. If
the nulls dominate: we will have measured VLM-relabel error rates and the
exact-vs-judged trade on a public benchmark — standalone value either way.
