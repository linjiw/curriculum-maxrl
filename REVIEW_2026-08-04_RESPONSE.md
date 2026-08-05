# Response to the 2026-08-04 independent draft review

Date: 2026-08-05. Review: `PAPER_DRAFT_REVIEW_2026-08-04.md` (4/10 weak
reject). Every finding audited against the code and the manuscript;
verdicts below with what changed and what is running.

## Verdict summary

| finding | verdict | resolution |
|---|---|---|
| P0-1 gate is not a destination pass-rate gate | **VALID** | reframed in text (renamed); genuine `p(g')` gate open |
| P0-2 mixed-target relabel groups ≠ MaxRL objective | **VALID** | characterized as weighted-SFT in Remark 3 |
| P0-3 title claim not causally identified | **VALID** | balanced factorial RUNNING (prereg committed) |
| P0-4 page/abstract budget | **VALID (deferred)** | needs a dedicated cut pass — see plan |
| P1 Prop 3 false at `N_i = 0` | **VALID** | fixed: `n_i >= 1` mandatory + defect stated |
| P1 intended gate not validated | **VALID** | already disclosed in 6.9; ARM A (designed gate, 3 seeds) queued |
| P1 mass magnitudes normalization-dependent | **VALID** | caveat added to Prop 2 interp; grpo_nostd arm running |
| P1 LLM figure outruns evidence | **VALID** | fig3 + fig2d now show both seeds; noise as scale bar |
| P1 artifact not self-contained | **VALID** | hindsight.py vendored + manifest gaps closed (partial) |
| P1 novelty wording | **VALID** | narrowed, "to our knowledge" + search caveat |
| Venue: ICLR not ICRA | **AGREE** | ICLR wrapper is primary |

## P0-1: the gate statistic (verified against code — the review is right)

Re-derived from `verl_integration/vendored/hindsight.py` (now vendored):
the posterior update is hit `(a,b) -> (a*decay+1, b*decay)`, miss
`(a,b) -> (a*decay, b*decay)`. **`b` starts at 0 and is never
incremented** — so `p_hat = (a+1)/(a+b+2)` is a pure decayed-recency
statistic on the relabel stream: one hit puts a key at 2/3 (> 0.5 →
rejected); ~29 miss-batches to prune back to admissible. It never rolls
out `g'` and never observes a destination failure. It is exactly what
the review says: an achieved-goal frequency/novelty filter.

Resolution taken (review's option 2, rename): abstract, Q3, Algorithm 1
(`f_hat` with definition), the §6.8 takeaway (explicit "what this is
not"), related work, knobs table, contribution 4, conclusion.

**Measured 2026-08-05** (`run_gate_variants.py`, 10 paired seeds, skill
chains where true p(g') is exact — closes reviewer Q1/Q8): the derived
true-p gate preserves ~all of ungated recycling's value (AUC .879 vs
.881) while the frequency heuristic pays a real toll (.798; true-p
wins 10/10). At decision time the heuristic's statistic
**anti-correlates** with true p(g') (−.27) — it tracks the recycler's
own recent output, not policy competence; the two rules agree only
once everything saturates (.66 early → .99 late). §6.8 now says the
LLM gate results are evidence about a recency-novelty filter and names
the genuine pass-rate gate as the indicated upgrade. Artifact:
`results_gate_variants.json`.

The second mismatch (u_N's high-p zero is at p=1, threshold 0.5 is
tuned) is now stated wherever the threshold appears.

## P0-2: mixed-target groups (verified — the review is right)

`hindsight.py` relabels each parseable rollout to its own achieved
value inside one uid group; the module docstring already admitted the
consequences. Remark 3 now carries the object-level characterization:
per-row relabeling with a shared success count K is a **verified
weighted-likelihood (weighted-SFT) update**, not a `J_{N-1}` gradient
for any single achieved task; the -1/N failure weight is a conditional
push-down on malformed outputs, not a zero-mean baseline; success
weights couple across unrelated destinations. The CPU reference
(each relabeled trajectory as its own K=1 group at weight 1-1/N) IS the
single-destination object of the remark, and the appendix wiring
contract cross-references this. All Countdown/Jugs hindsight results
are now explicitly evidence about the weighted-SFT objective.

**Implemented 2026-08-05** (maxrl fork commit 2700198): the
one-destination-per-group variant ships as a config flag
(`+data.hindsight.one_target_per_group=true` — modal achieved value
per dead group, only certifying rows relabeled, group-level gating),
CPU-tested. **Measured at the exact rung** (`run_row_vs_group_relabel.py`,
10 paired seeds): per-row relabels as their own K=1 groups lead (AUC
.952) > one-destination-per-group (.881) > shared-K coupled per-row
(.749, barely above no-recycling .705); both orderings 10/10 seeds.
The coupling is what costs — Remark 3 now carries the measurement,
scoped to what it can say about the deployed loop (different
normalization/failure conditioning; the LLM-side test is the flag).
Artifact: `results_row_vs_group.json`. The LLM ablation run itself is
queued behind the factorial and reviewer arms.

## P0-3: causal identification — the factorial is running

`run_factorial.sh` + `fact_analyze.py` committed (maxrl repo,
`curriculum_maxrl/maze_gpu/`), pre-registered 2026-08-05 before any
run:

- {maxrl, grpo} x {uniform, frontier_un} x **6 independent seed blocks**,
  identical per-block SFT warmstarts shared across all 4 cells.
- **Matched step budget** (250 steps): matches generation AND
  optimization budgets, removing the extra-steps confound the old
  wall-clock protocol had (and robust to the GPU now being shared).
- One prespecified endpoint: delta mean pass@8 (13 levels, fixed
  held-out set, Chen-unbiased).
- P-F1 with a committed falsification branch: MaxRL-GRPO paired
  delta-cov positive in >=5/6 blocks under BOTH samplers, else the
  estimator-conditioned coverage claim is dropped (6/6 gives exact
  sign-test p=.031).
- The review's two control arms fold into the same protocol under the
  standing 2026-08-04 prereg: **grpo_mass x grpo** (GRPO scheduled by
  its own mass functional, P-G0a/b) and **uniform x grpo_nostd**
  (Dr.GRPO-style no-std, P-G0c) — 6 seeds each.

The fixed realized-prompt-schedule control the review asks for already
exists at the exact-gradient rung (`results_schedule_matched.json`,
10/10 paired) and is quoted in §6.3; the neural-scale analogue remains
future work if the factorial lands.

GSM8K steering-controlled cells (E-LLM-1b): g3s OOMed after delivering
weak treatment (inconclusive-by-design per prereg gate), m3s reached
step 25 before the node OOM. The queue (`run_reviewer_arms.sh`,
holding its lock) waits for GPU headroom behind the factorial.

## P0-4: length — deferred to a dedicated pass

Not addressed in this round (the other P0s change what several sections
must say before a cut pass is stable). Plan on file: cut to ICLR
9+refs+appendix by (i) moving IsaacLab/Gym/Jugs/run-history to the
appendix, (ii) removing Prop 3 from the main text if the factorial
lands (it is not used by the deployed teacher — review's "best fix"),
(iii) 180–220-word abstract with one exact result, one confirmatory
experiment, one consequence, one limitation. Do this after the
factorial resolves which claims survive.

## P1 resolutions (this commit)

- **Prop 3**: restated over `N_i >= 1` with mandatory initial rollout +
  budget feasibility; the `u_0(p) = -p` vs true-mass-0 defect and the
  activation-cost structure at `N_i >= 0` stated inside the
  proposition. (Removal remains the likely end state — see P0-4.)
- **Normalization**: Prop 2 interpretation now says absolute
  cross-estimator ratios are implementation facts under a common
  learning rate; shape and zeros survive recalibration; empirical
  claims are tails and signs. The grpo_nostd arm is the measurement.
- **Fig 3 / Fig 6(=fig2d)**: both GRPO seeds plotted as paired
  trajectories (registered thick, replication thin); eval noise drawn
  once as a scale bar / corner note, never per-point method bars;
  captions state "shape 1-of-2, sign 2-of-2". Data tables and manifest
  checksums refrozen.
- **Artifact**: `verl/utils/hindsight.py` vendored verbatim at
  execution commit 0ad11b1 (`verl_integration/vendored/` + provenance
  README); body.tex reference fixed; fig6_gym manifest inputs +
  checksums added; GSM8K 3x harness discrepancy moved into Known gaps
  with its status. Still open: fig2/fig3 endpoint tables derived from
  seed-level logs (currently transcribed; disclosed), IsaacLab raw
  logs (other team; disclosed).
- **Novelty**: "first exact-verifier hindsight experiment in RLVR" →
  "our first ... in an RLVR loop" + precedent pointer; retained claims
  prefixed "to our knowledge" with an explicit no-exhaustive-search
  caveat; "one method, no per-domain switches" replaced by the honest
  statement (per-domain relabel map + destination key, Jugs keying bug
  cited); "derived gate/mitigation" → "motivated/saturation gate"
  globally.

## Running now (check back)

| what | where | analyzer |
|---|---|---|
| balanced factorial 4 cells x 6 seeds + grpo_mass + grpo_nostd | `maze_gpu/fact250_*.jsonl` (driver `factorial_driver.log`) | `fact_analyze.py` |
| ARM A designed-gate B3 x 3 seeds; ARM B replay control x 3 seeds | `~/ckpt/countdown_a10g/` (queued behind GPU) | prereg in `run_reviewer_arms.sh` |
| E-LLM-1b m3s completion + harness reconciliation | watchers in place | `analyze_e_llm1b.py` |

Paper claims are NOT updated with factorial outcomes yet — the text
still grades the maze evidence exploratory. When `fact_analyze.py`
reports, either the claim upgrades (with the prereg citation) or the
falsification branch executes.
