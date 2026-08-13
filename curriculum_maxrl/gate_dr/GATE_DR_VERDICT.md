# GATE-DR verdict — corrected-code utility-gate dose–response

**Date:** 2026-08-13. **Prereg:** `hopper/GATE_DR_PREREG.md` (frozen commit `16b95b7`;
amendments 2026-08-13 symlink retry and 2026-08-13b sealed-eval instrument, both
outcome-blind). **Runs:** Slurm arrays 9357906 (tasks 0–2) + 9357948 (tasks 3–11);
sealed evaluation job 9358123. **Analysis:** `analyze_gate_dr.py` → `gate_dr_analysis.json`.

## Preregistered verdict

**Rule 4: GRADED DOSE WITHOUT USEFUL OPERATING POINT.**

- **Transfer gate (rule 1): PASSED.** Ungated recycling (g0) beat no-recycling (b1h)
  on tier-1 mean@16 in 3/3 paired seeds (+.0073, +.0596, +.0337).
- **Manipulation check: PASSED.** Gate rejection under corrected code is graded
  exactly as designed: 0 (no gate) → .72–.80 (`gate_max_p=.85`) → .88–.90 (`.70`);
  ARM A's designed `.5` sits at ~.93 (historical). The *dose* is a dial.
- **Usefulness (rule 2): FAILED at both settings.** g085 mean-kept fraction .136
  (per-seed −.35/+1.14/−.38 — sign-inconsistent); g070 −.330. Neither retains the
  frozen ≥.40 of the ungated mean gain with 3/3 positive pairs. The bug-era
  "useful operating point" did not reproduce under corrected code at any tested
  strength: the *effect* of the dial is not.

## Descriptive observations (not preregistered claims)

- With raw 16-bit outcomes retained, seed 1 shows the paper's central ambiguity in
  **standard observed-set pass@16** (not the proxy): g0_s1 mean@16 rose vs b1h_s1
  (+.0073) while standard pass@16 fell .656→.414. Seeds 2–3 do not repeat it.
  First non-proxy instance in the project record; strictly descriptive at n=3.
- Per-seed heterogeneity dominates arm effects at this scale, consistent with the
  ARM-A "high-variance recycling-off point" reading.

## Paper consequence (applied)

`paper/body_iclr.tex` gate sentence updated: corrected-code sweeps (strong gate +
this dose–response) found graded dosage but no operating point retaining the frozen
fraction of the ungated mean gain. Appendix "Negative branches" carries the detail
including the raw-outcome descriptive observation. The former "corrected-code
validation of the useful setting remains open" framing is closed — negatively.

## Environment note

Independent A100 environment (GMU Hopper), SDPA attention, vLLM sealed evaluator
(engine seed 20260813), uniform across all 12 runs; all references on-Hopper, so no
cross-environment contrast is claimed. Raw artifacts: `eval/*_eval.json` (per-task
16-bit outcomes), `accounting/*.jsonl` (dose), `logs/` (training stdout).
