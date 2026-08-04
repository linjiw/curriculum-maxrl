# E-LLM-3 postmortem (2026-08-03): three nulls, two bugs, one real finding

Verdicts (analyze_e_llm3.py against the binding prereg, commit 728d7aa):
P-J1 NOT CONFIRMED, P-J2 VACUOUS, P-J3 NOT CONFIRMED. Every arm
(B1/B2/B3 × seeds 1-3) converged to the same fixed point:
t0 mean@16 = pass@16 = **0.450 exactly**, t1 pass@16 **0.15 → 0.00**,
t2–t4 flat 0. Artifacts: `cells/`, `e_llm3_verdicts.json`,
`jugs_noise_floor.json`, `entropy_trajectories.json`.

## What happened, mechanically

1. **Entropy collapse dominated everything.** Actor entropy fell
   1.36 → 0.02–0.21 in every arm within 60 steps. mean@16 == pass@16
   exactly means the eval policy (temp 0.6) is per-task deterministic:
   16 samples, all identical, right or wrong. The solved set is the
   min-moves-2 stratum (47% of t0 ≈ the 0.45 ceiling) plus nothing.
2. **The learnable band was one stratum wide.** t0 = min_moves 2–4 was
   designed as the bootstrap rung, but 2-move tasks are solvable by a
   1–2 token template ("fill B" / "fill A, pour A->B"). At N=16 and
   temp 1.0, 16 rollouts of a 2-move plan are near-identical, so the
   estimator sees K≈0 or K≈N — the i.i.d.-rollout diversity that the
   whole group-contrast machinery assumes is absent for very short
   plans. Mastering the template kills the exploration needed for t1's
   4–7-move plans; t1 coverage dies (0.15→0).
3. **The gate over-rejected by three orders of magnitude** (B3: reject
   fraction .997–.998, ~0.3 admitted relabels/step vs B2's 169/step).
   Root cause is a **granularity bug we shipped**: the gate posterior
   keys on the bare destination *value*. In Countdown values span
   hundreds of integers, so value ≈ task. In Jugs achievable amounts
   are ≤ 20 small integers shared across ALL tasks — the posterior
   saturates on "3" globally after a few groups, even though task
   ([7,12], target 3) and task ([3,5,19], target 3) are different
   tasks (577 distinct capacity-sets in the pool). The gate key must
   be the *relabeled task*, not the value.
4. **B2's dose accelerated the collapse** (entropy endpoint 0.02–0.03
   vs B1's 0.07–0.21; 169 relabels/step at yield .88): the sharpening
   direction at extreme dose — but with all arms pinned to the same
   0.450 ceiling by (1)+(2), the B2-vs-B1 coverage contrast the prereg
   wanted never got room to run.

## The real finding (paper-relevant, adverse to our own headline)

**B1 — plain MaxRL, no recycling, uniform sampling — exhibits the
classic pass@k collapse on this pool** (t1 coverage 0.15 → 0.00 while
t0 masters; entropy 1.36 → 0.07–0.21). The maze result ("every MaxRL
run grows coverage, every GRPO run loses it") is therefore **pool-
conditional, not absolute**: when the learnable band is one narrow
stratum and solutions are 1–2 tokens of template, MaxRL's
success-conditioned weights also liquidate frontier coverage through
shared parameters. This is exactly the mechanism named in the paper's
Remark (mass counts weighted rollouts, not cross-task interference
through the policy) — now with a measured instance against our own
estimator. The limitations section must carry it.

## Redesign (E-LLM-3b), gated on the fixes

1. **Gate key = relabeled task** (tuple(caps) + target), not value —
   one-line change in JugsHindsight._relabel_candidates/gate bookkeeping
   + regression test. Countdown semantics unchanged (its value keys
   were effectively unique per task).
2. **Pool v2: widen the band.** Replace the 2-move stratum with a
   graded ladder (min_moves ∈ {3,4}, {5,6}, {7,9}, {10,13}, {14+}),
   dropping the 1-template tier; feasibility first (expect t0' pass@1
   in [.01,.2] rather than .06-with-a-.45-template).
3. **Diversity at the frontier is the binding constraint**, not signal
   allocation: report per-task rollout-set entropy at step 0 as a
   design gate — if the 16 rollouts of a band task are >90% identical,
   the group contrast cannot form and no sampler/recycler fixes that.
4. Re-register P-J1'/P-J2'/P-J3' only after (1)-(3); the 2026-08-02
   prereg is spent (correctly — it did its job by making these nulls
   unambiguous).

## Cost

~19 A10G-hours (noise floor + 9 cells). The three CPU testbeds and
Countdown remain the paper's recycling evidence; E-LLM-3 v1 enters the
negatives ledger with the two bug diagnoses attached.
