# Audit round 2 — docs/index.html, frontier_rl/README.md, SONIC_RESPONSE.md, COSMOS3_RESPONSE.md

Status: FLAGS RESOLVED (2026-07-24). The agent's read-through flags were all
verified and fixed on main (see commit); its full verdict tables were lost to
repeated API timeouts, but every preliminary flag below was independently
checked before fixing:
- index.html echoes b/d/f/g: FIXED (single-seed qualifier ×1, N=2 slice,
  ordering claim weakened to what curves.json supports, T=N−1 ×3 — the
  adaptive-T table row now reads 'fixed T=N−1').
- COSMOS3 §0.3 'every seed': FIXED (single-seed qualifier).
- COSMOS3 Q1 tail-sum upper limit: CONFIRMED CORRECT as written — MC at
  N=8,p=0.3: Σ_{k=2}^{N}(1/k)∇pass@k = 0.44965 vs MC 0.44969 (N−1 gives
  0.43235). The T=N−1 correction is about the CENTERED estimator's
  objective order; the positive-part identity's limit is N.
Remaining known-unverified: the four docs' long-tail numbers (they quote
EVIDENCE/REPORT tables already audited in rounds 2a-2c).

- docs/index.html — PENDING
- frontier_rl/README.md — PENDING
- SONIC_RESPONSE.md — PENDING
- COSMOS3_RESPONSE.md — PENDING

Evidence already loaded (for resume): results_fixed_n.json, results_speed.json,
positive_part_training_cost.json, v7_oracle_result.json, mountaincar_scaled.json,
docs/curves.json summary, maze_gpu/EXPERIMENTS.md, efficiency.json, efficiency_p1.json.

Preliminary flags found on read-through (to be confirmed):
- index.html line ~260 and ~778: "amplified GRPO's collapse in every seed" — frontier+grpo was single-seed (defect echo b).
- index.html line ~783: learnability as "N=1 slice" (defect echo d; correct is N=2, u1≡0).
- index.html lines ~303/351/639/802: T=N claims (defect echo g; correct T=N−1 for drop-K=0).
- index.html line ~680: "ordering holds on every environment tested" vs curves.json MountainCar teacher AUC < uniform (defect echo f).
- COSMOS3_RESPONSE.md §0.3: "pass@8 0.332→0.269, every seed" (defect echo b).
- COSMOS3_RESPONSE.md Q1: pass@k tail sum upper limit N vs N−1 — MC check pending.
