# Utility audit result — SURPRISE experiment ① — 2026-08-18

Frozen rule applied once. 10 seeds × 3 depths × 2 pools × 2 estimators, 36
tasks each, true continuation utility measured exactly by branch-and-continue.

## Verdict on the thesis: the factorization is not needed here

Structured pool, deployed practical MaxRL, `Δ_struct = ρ(A·C, U_H) − ρ(u_N, U_H)`:

    mean −0.021   CI [−0.038, −0.004]   exact p = .051   positive 3/10

**CI upper < +0.05 → "factorization not needed."** Adding the structural
compounding count to availability does not improve — and slightly degrades —
how well the score ranks true continuation utility. Under RLOO the contrast
is +0.001 (CI [+0.0005, +0.002], 8/10): technically positive, but three
orders of magnitude below the SESOI. Under both estimators `u_N` already
ranks utility as well as `u_N·C` does.

The flat pool behaved as the prereg predicted (`A·C ≡ u_N` up to scale, ρ
identical), and utility is much harder to rank there under every predictor
(≈.47 vs ≈.83): compounding through shared skills makes utility *predictable*
from availability, it does not make availability *insufficient*.

## What the secondaries say, and this is the actual finding

| contrast (structured, MaxRL) | mean Δρ | p |
|---|---|---|
| **`u_N` − `p(1−p)`** | **+0.106** | **.002** |
| `u_64` − `u_N` | −0.016 | .029 |

The deployed-N availability score `u_16` ranks true continuation utility
**substantially better than learnability `p(1−p)`** — +.106 in rank
correlation, 10 seeds, p=.002 — and better than the over-shooting `u_64`.
Under RLOO, whose coefficient mass *is* `2p(1−p)`, the two are identical
(+0.0001, p=.97): **the advantage of `u_N` over `p(1−p)` appears exactly when
the deployed estimator's mass is not `p(1−p)`.** That is the paper's
estimator-conditioning claim, now measured against ground-truth utility
rather than against a downstream outcome.

And where does true utility peak? Under MaxRL, the argmax-`p` of `U_H` has
median **.026** (IQR .017–.056) — well below `p*_16 = .169` and even below
`p*_64 = .064`. Under RLOO it is **.194** (IQR .151–.253) — near `p*_16` and
nowhere near .026. So the *location* where one-step continuation utility
peaks is estimator-dependent, sits far in the tail under MaxRL, and matches
neither the deployed peak nor the over-shoot: it is not a fixed target of the
score family at all.

## How this bears on the guidance

- **Object U ≈ A_N·C, with C the structural count: not supported** at this
  operating point. The compounding term is empirically inert as a *ranking*
  correction on the pool where it should have mattered most. This is the
  guidance's own "if opposite" branch: retreat to "harder-peaked helps" as an
  empirical rule, and drop the structural-count `C`.
- **But the reframe's deeper premise survives and sharpens.** Availability
  ranks utility well (ρ≈.83) *because* utility on a shared-skill pool is
  largely a function of where the estimator can emit mass. The estimator
  conditioning is the load-bearing part — under RLOO, `u_N` collapses to
  learnability and so does its ranking power. "Availability ≠ utility" is
  true, but on this substrate the gap is not where `C` looks for it.
- **The utility peak is far in the tail under MaxRL.** This is consistent
  with the exponent-sweep finding (harder peaks keep winning) and offers a
  mechanism the sweep could not: MaxRL's `(N−1)×` mass at `p→0` means the
  highest-utility tasks are the nearly-dead ones with a live frontier — and
  neither `u_16` nor `u_64` peaks there. That is a candidate for *why*
  utility rises past the deployed N, and it is testable.

## Limits (as preregistered)

One-task-at-a-time utility at H=8; the structural `C` is the simplest member
of its family; exact gradients, 360-parameter policy, synthetic pool. This
audit says the structural count does not help ranking on this substrate; it
does not say no `C` could. A learned or curvature-based `C` is a different
hypothesis and would need its own preregistration.

---

## Addendum 2026-08-18 — reading the audit against the PI judgment

The PI judgment (`PI_JUDGMENT_2026-08-18.md`) predicts
`activity ≠ immediate gain ≠ continuation utility`, and its Experiment ②
predicts the gap opens with horizon. An exploratory horizon probe on the same
harness (5 seeds × 3 depths, structured pool, MaxRL, **not preregistered**):

| H | ρ(u_16, U_H) | where U_H peaks (median p) |
|---|---|---|
| 1 | +.651 | .051 |
| 4 | +.797 | .075 |
| 8 | +.818 | .023 |
| 20 | +.810 | .004 |
| 50 | +.784 | .002 |

Two things at once, and they must be stated together:

1. **Ranking does not collapse with horizon.** Availability keeps ranking
   continuation utility at ρ≈.80 out to H=50. On this substrate, "activity
   ≠ utility" is *not* a ranking failure of `u_N`.
2. **The utility peak migrates into the dead zone.** From H=8 onward the
   highest-utility tasks sit at p≈.002–.02 — tasks `u_16` scores near zero,
   below both `p*_16` and `p*_64`. The gap is real, and it is a *peak-location*
   gap: sampling proportional to `u_N` systematically under-weights the tasks
   with the highest long-horizon value. This is a sharper statement of the
   exponent-sweep finding, and it explains it: performance rose past the
   deployed N because true utility peaks far past it.

**Why the structural `C` was inert, precisely.** The shipped pool is three
*linear* chains: task level `l` requires skills `1..l`, so every task's skills
are a prefix of every harder task's in its chain. Compounding is monotone in
level and level is monotone in `p`, so `C` is a function of `p` and cannot
carry information `u_N(p)` lacks. **The judgment's Experiment ① — activity-
matched, transfer-mismatched pairs — cannot be constructed on this pool.** It
requires a *branching* task graph where two tasks share `p` but differ in how
many downstream tasks their skills unlock. Building that pool is the
prerequisite for testing the compounding hypothesis at all; today's "not
needed" verdict is a statement about linear chains only and does not close
the question.

**What this means for the paper's boundary.** The estimator-conditioning
claim got *stronger* today (`u_16` beats `p(1−p)` at ranking ground-truth
utility, +.106, p=.002, and only under MaxRL). The peak-location boundary got
a mechanism (true utility peaks in the tail). Neither requires the continuation
machinery to state, and both belong in the current paper's closing.
