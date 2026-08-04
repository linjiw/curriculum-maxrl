# Reviewer 2: RLVR practitioner — Rating 4/10, Confidence 4/5

## Headline weaknesses
- W1: evidence at claimed scale is anecdotal (360M, 50 steps, 1-2 seeds vs 7B+ practice)
- W2: P-G2 shape non-replication; abstract's "2/2 seeds" is a sign match on within-noise deltas
- W3: teacher starved in the only LLM test; per-prompt Beta posterior CANNOT work at ~1 epoch pools — needs bucket posteriors or difficulty prediction, or honest small-pool scoping
- W4: maze throughput mechanism (skip backward on dead groups) is structurally absent at LLM scale (85% gen time; dead groups pre-pay generation) — must compare vs DAPO resample-to-fill at matched GENERATION budget
- W5: recycling under-specified for adoption (no GSM8K/code relabel map; loss placement ambiguous; no pseudocode of verl wiring in paper)
- W6: gate 3-seed result is at the buggy operating point; designed point has 1 seed
- W7: amplification INTERACTION underpowered everywhere; missing GRPO-native teacher arm (schedule by sqrt(K(N-K))/N) — mismatch confound
- W8: Jugs null makes safety claim pool-conditional; diagnosing band structure ex ante needs the census the curriculum was meant to spare
- W9: only 2 trained estimators for a paper titled "The Estimator Decides"
- W10: density; unreconciled 3x pass@k harness discrepancy is not a footnote for a coverage paper

## Questions (10) — most actionable:
Q3 (wall-clock value at LLM scale vs resample-to-fill), Q4 (exact loss placement
for relabeled groups), Q5 (GSM8K/code relabel map), Q7 (rerun gate at designed
point, 3 seeds), Q8 (GRPO-native teacher arm), Q10 (reconcile harness).

## Strengths acknowledged
- Identity useful + unification real; coverage-as-currency right instrument;
  sharpening novel; epistemic honesty best-of-cycle; step-matched control right.

(Full review in the agent transcript; this file is the actionable digest.)
