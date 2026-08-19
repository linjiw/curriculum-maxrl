# Reconciling the editorial charter with the count-law pivot

Two guidance documents are now in force. They agree on process and conflict on
scope. This file records which governs where, and why.

## Verdict: adopt the count-law pivot's science, keep the charter's process

**Adopted from `PI_GUIDANCE_COUNTLAW_PIVOT_2026-08-19.md`:**

1. **The count-law generalization.** Activity is a functional of the
   success-count law, `A_E(z) = Σ_k P(K=k|z)·M_E(k)`, with closed-form
   `M_E` for MaxRL, RLOO and GRPO. Verified against the deployed estimators
   over N ∈ {4,8,16,32} and every k before adoption: MaxRL and RLOO exact to
   <1e-9, GRPO exact to the zero-ε idealization with the deployed value equal
   to `ideal × std/(std+EPS)`, EPS = 1e-6, matched to 1e-12. This is a strict
   generalization of what the paper had and it costs nothing.
2. **The ℓ₁ justification.** Three properties and one limit, now stated where
   the surrogate is introduced. Reviewers were going to ask why `Σ|w_i|` and
   not gradient norm; "because it is closed-form" is not an answer.
3. **Both overclaim corrections.** Real errors, and they were in a published
   PDF:
   - the aggregation gap is **not** monotone in N — with all `p_X > 0`, both
     `E[(1−p_X)^N]` and `(1−p̄_z)^N` vanish, so only the local statement near
     the active region is safe;
   - the corollary predicts the **calibration** bias, not the endpoint sign.
     "The sign Cor. 2 predicts" is now "a sign consistent with Cor. 2, which
     predicts the calibration bias and not the endpoint."
4. **The scope call on recycling.** With the count-law framework in the main
   text, the creation channel is a second paper's spine competing with this
   one's. It returns to the appendix with a one-sentence pointer and its
   practitioner-box bullet reduced to a cross-reference.

**Retained from `EDITORIAL_CHARTER_2026-08-19.md`:** the tier constitution,
the language law, registration discipline with both verdict branches drafted
before data, "an untraceable number does not ship", cut-by-tier-from-the-bottom,
and the 9-page bound with the conclusion ending on page 9. The pivot does not
contradict any of these; it changes what the paper is about, not how claims are
policed.

## The conflicts, named

| item | charter | pivot | resolution |
|---|---|---|---|
| P3: promote probe gate + relabel ordering to main text | do it | move recycling out of the main line | **pivot.** Executed the charter's version in `10413fe`, then reversed here. The material is verified and correctly tiered; it is not wrong, it is off-thesis. |
| Fig 1 | three-regime map (allocate/create/safety) | same-mean/different-count-law counterexample | **unresolved, flagged.** Kept the three-regime map for now; the counterexample figure is the stronger hook for the new thesis and is the next figure to build. |
| Title | keep "Learnability, Reweighted" | retitle to "When Pass Rate Is Not Enough" | **unresolved, flagged.** The pivot's title is more accurate to the current thesis. Not changed unilaterally: the charter was explicit, and a title change is the PI's call. |
| P0 design | per-level vs per-task posterior | naive plug-in vs group-law score | **pivot is better and I will re-register.** The group-law score is estimable from the K's already observed, so it works in the fresh-sample regime where a per-task posterior is undefined — the exact obstacle recorded in the current registration. |

## What changed in the paper this round

Abstract rewritten to lead with the sufficiency question and the counterexample,
still at the charter's four numbers. Definition 1 is now the count law with the
three deployed masses. The ℓ₁ remark added. Contribution 1 states the
count-law generality. Both overclaims fixed. Creation channel to the appendix.
Conclusion ends on page 9.

## Not yet done, in priority order

1. Re-register P0 as naive-plug-in vs group-law-score (supersedes the current
   registration, which has not started running).
2. Build the same-mean/different-count-law counterexample figure.
3. Implement the Group-Law Activity Teacher (decayed Dirichlet over K).
4. Controlled coarsening experiment with matched-mean, different-heterogeneity
   bins; calibration error as primary, held-out AUC as co-primary.
5. Title decision.
