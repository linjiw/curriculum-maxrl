# Balanced maze factorial — verdict against prereg (2026-08-05)

Prereg: `run_factorial.sh` (committed 2026-08-05 before any run).
Results: `results_factorial.json` (31 complete cells; repair pass
completing the last frontier_un cells is running and will be folded in,
but every prereg decision below is already decided by the completed
cells).

## P-F1 (registered primary): FAILED → falsification branch executes

Endpoint: paired (same seed block, same sampler) MaxRL − GRPO on
Δ mean pass@8 (final step-250 eval minus post-SFT), ≥5/6 blocks
positive under BOTH samplers required.

Final counts (all 12 blocks, repair pass folded in):

- uniform: **3/6 positive** (+.154, +.014, −.014, −.019, .000, +.014)
- frontier_un: **4/6 positive**

Per the committed branch: the abstract's estimator-conditioned
coverage claim is **dropped, not softened**. Additional damage to the
old cohort claim: only **4/12 MaxRL cells grew coverage at all** —
"every MaxRL-labeled run grows pass@8" does not replicate in a
balanced, no-hindsight design. The cohort pattern conflated
recycling's coverage contribution with the estimator effect (the
cohort's MaxRL runs included hindsight arms; the factorial isolates
the estimator).

## What survives (exploratory, NOT registered — next prereg's hypothesis)

- **Time-integrated coverage ordering**: MaxRL > GRPO on coverage-AUC
  (mean in-training cov8 minus init) in **12/12 pairs across both
  samplers** (would be p=.0005 had it been registered; it was not).
  Arm means: uniform −.009 vs −.027; teacher +.005 vs −.025.
- **Easy-band asymmetry**: MaxRL loses less easy-band (L1–3) coverage
  in 9/12 pairs (p=.15); GRPO's mean easy-band loss is ~2× MaxRL's
  (−.208 vs −.101 under uniform; −.139 vs −.024 under the teacher).
- The endpoint metric is a single 16-maze/level eval (noise ~±.03 per
  eval) against effects of ~.02–.05 — the registered endpoint was
  likely underpowered relative to eval noise. Design lesson recorded;
  not an excuse.

## P-G0a (grpo_mass; standing prereg 2026-08-04): CONFIRMED

GRPO scheduled by its own mass functional loses coverage like every
other GRPO arm (final: mean Δcov8 −.042 = uniform-GRPO's −.042; 5/6
cells negative, one zero). No scheduler choice rescued the estimator —
the "teacher-estimator mismatch" alternative is closed at neural scale
too (matches the exact-rung result).

## P-G0c (grpo_nostd holds easy band): FAILED → committed revision executes

Prediction: no-std GRPO holds easy-band pass@8 in ≥2/3 seeds.
Result: easy band negative in **6/6 seeds** (mean −.201, same as
sample-SD GRPO's −.208). Per the committed branch: at neural scale
the easy-band liquidation is NOT explained by variance normalization
alone — mean-centered advantages without normalization lose the easy
band just as much. This CONTRADICTS the exact-gradient rung, where
no-std collapsed onto RLOO's (coverage-preserving) profile. The
mass account's Fig-1B story holds where gradients are exact and
per-task; through shared parameters at neural scale it does not
predict the easy-band outcome. Both results now go in the paper side
by side.

## Interaction (exploratory): teacher does NOT amplify GRPO's decay

Final: teacher−uniform Δcov8 under GRPO: +.111, +.019, −.014, +.038,
−.024, −.043 (mean +.015 — mildly protective, not amplifying). Under
MaxRL: −.062, +.019, +.067, +.058, +.029, +.014 (mean +.021). The
GSM8K P-G2 story (teacher-induced GRPO regression) receives NO
corroboration from the maze factorial at this budget.

## Protocol notes

250 fixed steps (matched generation + optimization) vs the cohort's
2400 s wall-clock (~580+ steps): budgets differ, so cohort-vs-factorial
level comparisons are invalid; only within-factorial contrasts are
quoted. GPU was shared with unrelated jobs; per-cell throughput varied
but budgets are step-matched by construction. Cells that died to
contention are being rerun by `run_factorial_repair.sh` (same command,
resumable); the P-F1 verdict cannot flip (already ≤4/6 under uniform).
