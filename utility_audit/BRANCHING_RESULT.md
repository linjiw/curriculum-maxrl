# Branching-pool result — PI judgment Experiment ① — 2026-08-18

Two runs are reported. The **preregistered** one, and a **corrected** one whose
amendment was written after seeing the first. They disagree, and the
disagreement is the finding.

## 1. Preregistered run (u_N-matched pairs, warm 400)

| estimator | pairs/seed | Δ = U(high-C) − U(low-C) | CI | p | pos | verdict |
|---|---|---|---|---|---|---|
| MaxRL | 94 | **−0.00153** | [−.00343, −.00006] | .092 | 2/10 | **activity suffices** |
| RLOO | 1020 | +0.00007 | [+.00006, +.00008] | .002 | 10/10 | **activity suffices** |

By the frozen rule (CI upper < SESOI +0.002) both say the compounding term is
not needed. Taken at face value this would have closed the question.

## 2. The construction was wrong

`u_N` is **unimodal**. Matching two tasks on `u_N` alone does not match their
difficulty — it admits pairs on *opposite sides of the peak*. In the `C` ratio
≥ 5× band the pairs had mean pass rate **0.852 on the high-`C` side and 0.011
on the low-`C` side**, mean depth 1.06 versus 3.94. That band was not
measuring "same availability, different transfer"; it was measuring
"mastered versus nearly dead", and unsurprisingly the nearly-dead task gained
more at `H=8`. The negative MaxRL Δ was an artifact of the matching rule.

The PI specification was "相同的当前 pass rate；相同的 A_N(p)" — same pass
rate **and** same activity. I enforced only the second. Recorded as
outcome-adjacent discovery: the flaw is visible from `p`, `C` and depth alone,
but I went looking only after the >5× band came back negative.

## 3. Corrected run (p-matched ≤ .02 **and** u_N-matched, warm 200)

| estimator | pairs/seed | Δ | CI | p | pos | verdict |
|---|---|---|---|---|---|---|
| **MaxRL** | 390 | **+0.00313** | **[+.00185, +.00456]** | **.0039** | **9/10** | **TRANSFER MATTERS** |
| RLOO | 1020 | +0.00006 | [+.00003, +.00008] | .0039 | 9/10 | activity suffices |

Under the deployed MaxRL estimator, two tasks with the **same pass rate** and
therefore the **same coefficient activity** differ in true continuation
utility by **+0.0031 in J units** according to how many downstream tasks their
skills unlock — above the +0.002 SESOI, 9/10 seeds, exact p = .0039. The
effect grows with the mismatch: +0.0027 in the 3–5× band, +0.0147 in the >5×
band (n=5, descriptive).

Under RLOO the same contrast is +0.00006 — fifty times smaller and far below
SESOI — even though `ρ(C, U) = +0.87` there. Structure predicts utility under
both estimators; it adds utility *beyond matched activity* only under MaxRL.

## 4. What this establishes, and what it does not

**Establishes.** `activity ≠ utility` is now demonstrated causally rather than
inferred from a downstream loss. Hold the estimator's own availability signal
exactly constant and true continuation utility still varies with the task's
position in the skill graph. The PI judgment's third layer — continuation
value as the unsolved object — has direct evidence on a substrate where
utility is computable.

**Does not establish.** That `A_N·C` is the right functional form: the
secondary rank correlations barely move (`ρ(u_N,U) = .641` vs
`ρ(u_N·C,U) = .638`). The compounding information is real but the product
form does not capture it — consistent with the linear-chain audit, and it
means the next object should be *learned residual* on top of activity, exactly
as the judgment's §五 proposes, not a hand-multiplied `C`.

**Also does not establish** anything about linear chains: the original audit's
"not needed" verdict remains correct *there*, and correct for the reason
diagnosed — `corr(p,C) = 0.889` makes the test unconstructable, not the effect
absent.

## 5. Status of these numbers

The preregistered run is evidence under its frozen rule. The corrected run is
**exploratory** — its amendment is post-hoc and declared as such. It should be
re-frozen and re-run with fresh seeds before it carries any weight in a paper.
Neither belongs in the current ICLR submission, whose boundary is closed; both
belong to the next paper's motivation.
