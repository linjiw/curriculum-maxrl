# Reviewer 3: methodology hawk — Rating 5/10, Confidence 4/5

## Sharpest catches (ranked by damage)
1. FLOOR P-VALUES: p=0.0079 = 1/C(9,4) is the MINIMUM ACHIEVABLE for the
   5v4 design (perfect separation gives it regardless of magnitude); the
   band-asymmetry p=0.0001 = 1/C(22,4) is also the floor. One-sided.
   No multiplicity control across ~10 tests. Must disclose floor nature.
2. PERMUTATION NULL: pooling across curriculum arms assumes exchangeability;
   shared per-seed warmstarts across estimators create pairing the
   unrestricted permutation ignores (paired sign permutation floor: 0.0625).
   Run composition 9 vs 22 vs ~30 never reconciled — wants run-accounting table.
3. PREREG UNVERIFIABLE from the submission: no third-party timestamps/hashes/
   OSF; mutable self-controlled repo. "Good lab notebook, not preregistration."
4. Single-seed anchors under multi-seed language: IsaacLab null (n=1, no MDE),
   dial's full-strength endpoint (n=1 post-bug), GSM8K cells (1-2).
5. OUTCOME SWITCHING in abstract (same catch as R5) — ALREADY FIXED.
6. NOISE-FLOOR INCONSISTENCY: GSM8K's eval-repeat floor excludes training-seed
   variance (which seed 2 shows dominates); Countdown "+.046 outside seed
   noise" vs baseline SD .054 is INSIDE 1 SD unpaired — show paired deltas.
7. Scale (as R2).
8. "6/6 paired wins" = 2 correlated metrics x 3 seeds counted as 6.

## Questions: run accounting (Q1), stratified/paired permutation (Q2),
   external verification (Q3), abstract (Q4, fixed), Countdown paired
   deltas (Q5), gate dose sweep timing (Q6), IsaacLab MDE (Q7),
   harness reconciliation (Q8).

## Path to 7: steering-controlled multi-seed LLM cell + stratified/paired
   permutation analysis + externally verifiable registration + dose sweep.
