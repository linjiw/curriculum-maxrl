# Wave-2 verdict branches — text prepared BEFORE the verdict (2026-08-06)

Prereg: run_factorial_wave2.sh (commit d6aea90). Analyzer:
`python3 fact_analyze.py --seed-start 6 --seeds 6`. P-F2 = paired
MaxRL−GRPO cov_auc_delta ≥5/6 blocks under BOTH samplers. P-F3 =
easy-band endpoint asymmetry ≥7/12 pairs.

Both text branches are written now so the post-verdict edit is
mechanical, not creative. Numbers marked ⟨⟩ get filled from
results_factorial.json.

---

## BRANCH A — P-F2 CONFIRMS (≥5/6 both samplers)

**Paper §6.3b appendix sentence** (replace "a confirmation wave ... is
running with exactly that endpoint"):

> A confirmation wave on six fresh seed blocks (seeds 6–11, identical
> protocol, analyzer code path frozen) then **registered and
> confirmed** the time-integrated ordering: MaxRL over GRPO in
> ⟨X/6⟩ blocks under uniform and ⟨Y/6⟩ under the teacher (P-F2;
> pre-registered ≥5/6 both samplers; exact two-sided sign p = 0.031
> at 6/6), means ⟨+.0xx⟩/⟨+.0xx⟩. The easy-band asymmetry held in
> ⟨Z/12⟩ pairs (P-F3). Across the two waves the ordering now stands
> at ⟨12+n⟩/⟨12+12⟩ paired blocks — first found exploratorily,
> then confirmed at a registered endpoint on fresh randomness.

**Contribution 3** gains: "and a confirmation factorial on fresh seed
blocks then registered and confirmed the time-integrated form (P-F2)."

**Abstract**: "an exploratory time-integrated form of the ordering
survive" → "and a time-integrated form of the ordering — confirmed on
fresh seed blocks at a registered endpoint — survive".

**Title**: stands (per the committed decision rule in SCHEDULE.md).

**Limitations power paragraph**: append "The confirmation wave landed
⟨counts⟩, closing the loop the power analysis opened."

## BRANCH B — P-F2 FAILS (≤4/6 either sampler)

**Paper §6.3b**: the exploratory-ordering sentence is REPLACED by:

> A confirmation wave on six fresh seed blocks then **failed to
> confirm** the time-integrated ordering (⟨X/6⟩ and ⟨Y/6⟩; P-F2
> required ≥5/6 under both samplers), and per the committed branch we
> drop it: **no cross-estimator coverage claim of any kind survives at
> neural scale.** What remains is the exact-rung ordering (10/10
> frozen schedules) and the easy-band decomposition as descriptive
> statistics of these runs.

**Contribution 3** becomes: "the cohort's divergence failed its
factorial and the surviving exploratory ordering failed its
confirmation wave; what stands is exact-rung robustness and two
controls" — evidence-level honesty, full stop.

**Abstract**: drop the exploratory-ordering clause entirely.

**Title**: becomes "What Curricula and Failure Recycling Can and
Cannot Do in RL with Verifiable Rewards" (per the committed rule).

**Conclusion**: "12/12 paired factorial blocks — the next registered
endpoint" → "which failed its confirmation wave and is dropped".

## P-F3 sub-branches (independent of P-F2)

- ≥7/12: easy-band asymmetry keeps its place in 6.4 with "confirmed
  in the wave-2 majority sign test ⟨Z/12⟩".
- <7/12: 6.4's asymmetry text gains "did not replicate in wave 2
  (⟨Z/12⟩) and is demoted to a description of the wave-1 runs".

## Guard

Do NOT run the analyzer on partial blocks for a verdict. Interim reads
are health checks only. The verdict requires all 24 cells final.
