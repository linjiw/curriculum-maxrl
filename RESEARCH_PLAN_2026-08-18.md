# Research plan — 2026-08-18: two papers, one boundary

Supersedes the execution sections of `RESEARCH_PLAN_2026-08-15.md`. Derived
from `PI_JUDGMENT_2026-08-18.md` and `SURPRISE_GUIDANCE_2026-08-18.md`,
checked against `utility_audit/UTILITY_AUDIT_RESULT.md`.

## The frame, in one line each

- **Current paper (ICLR 2027):** each group estimator induces a task-activity
  geometry, `A_N(p)=2(pass@N−pass@1)`; activity is a diagnostic of *available*
  update, not a measure of learning utility. Freeze the boundary; close with
  MAZE-SCORE.
- **Next paper:** among active tasks, what makes one *useful* — continuation
  utility `U_H^Q(x;θ)`, estimated as a residual over the activity sampler
  (`β=0` recovers it exactly), tested where activity and utility disagree.

## What today's audit contributes to each

To the current paper (goes in now):
- `u_16` ranks ground-truth continuation utility +.106 better than `p(1−p)`
  (10 seeds, p=.002) — and *only* under MaxRL; under RLOO they coincide.
  Estimator conditioning is the load-bearing part, measured against ground
  truth.
- True utility peaks in the tail (median p .02–.002 at H≥8), below both
  `p*_16` and `p*_64`. That is the mechanism behind the exponent sweep.
- Structural compounding `C` is inert on linear chains — and cannot be tested
  there. That is a statement about the substrate, not the hypothesis.

To the next paper: the branch-and-continue harness exists, is exact, and runs
in seconds. It is the oracle every G2 experiment needs.

## Ordered work

### A. Current paper — freeze the boundary (this week, no compute)
1. Apply the five wording fixes in the judgment §九: no "sampling by it is
   the curriculum"; N determines *which tasks are active*, not which maximize
   utility; "no manually specified ZPD center or width" instead of "zero
   difficulty hyperparameters" (γ and floor remain); hindsight as complementary
   channel with the allocation-vs-creation sentence; MAZE-SCORE as closing.
2. Add the audit's two contributions above to `sec:peaktest` (mechanism) and
   the contract table (a ground-truth-utility row). Two paragraphs, no figure.
3. Rebuild, hold page 9.

### B. MAZE-SCORE — the closing experiment (Hopper, this week)
Per the power memo: 48 blocks, cap 40→48, exact test retained. Prereg is
still DRAFT; the four N encodings must move together (`test_sample_size_
contract.py` guards it). Do NOT add a continuation predictor to it.

### C. AMaze confirmatory — running (tonight)
Report the verdict once; one sentence into `sec:amaze`. It refines
Takeaway 4; it cannot flip the frame.

### D. Next-paper foundation — exact task graphs (CPU, this week)
The judgment's three experiments all need a substrate the shipped testbed
lacks. Build, in `utility_audit/`:
1. **Branching skill graph** — tasks share skills in a DAG, not a chain, so
   two tasks can have equal `p` and equal `A_N` but different downstream
   fan-out. Parameterised: independent (fan-out 0) → chain (fan-out 1) →
   branching (fan-out k). This is what makes Experiment ① constructible.
2. **Shared-continuation oracle** — `U_H^Q` with the first update on `x`, then
   `H−1` updates from a common pre-generated schedule `π_c`; oracle labels
   averaged over ≥3 independent schedules so a predictor cannot leak schedule
   identity. Quality-qualified `J_Q`: `Δmean@1` subject to
   `Δpass@8 ≥ −ε`, `Δcoverage ≥ −ε_c`.
3. **Experiment ①** on the branching pool: activity-matched,
   transfer-mismatched pairs, `H ∈ {5,20,100}`. Prereg first.
4. **Experiment ②**: `A_N, U_0, U_5, U_20, U_100` ranking reversal, same
   substrate. Prereg first. (The H-probe above is exploratory and does not
   count.)
5. **G2 residual predictor**, deliberately linear/low-rank on charged-
   trajectory features only, with the six-arm gate (uniform / activity /
   +residual / +shuffled / β=0 / oracle ceiling). Prereg first. **G2 gates
   Acrobot.**

### E. Deletion ladder (SURPRISE §R, cheap, as CPU frees)
floor deletion (half day; pass@k safety pressure test); Beta+Thompson →
empirical pass rate + softmax(γ); teacher deletion under hindsight as a formal
regime map; RLOO + frontier teacher as the third point on the inversion axis.

### F. Post-deadline
E2c′ raw-outcome recycling re-measurement (measurement infra, not before
Sept 25); E-LLM-3 continuous dial + kernel posterior; Hopper/RA-
