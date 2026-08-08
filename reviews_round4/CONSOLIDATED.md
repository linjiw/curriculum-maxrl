# Review round 4: five simulated ICLR reviewers (2026-08-04)

| Reviewer | Background | Rating | Confidence |
|---|---|---|---|
| R1 | RL theory (bandits, UED, curricula) | **6** | 4 |
| R2 | RLVR practitioner (verl, 7B+ pipelines) | **4** | 4 |
| R3 | Methodology / statistics | **5** | 4 |
| R4 | HER / goal-conditioned RL | **5** | 4 |
| R5 | Senior AC | **6** | 4 |

**Final average 5.2 (4,5,5,6,6)** — borderline at ICLR; would likely
land reject-with-encouragement or borderline-AC-discussion. The
distribution matches R5's prediction. Every reviewer independently
scored the honesty apparatus as the best they'd seen and the LLM-scale
evidence as the binding constraint.

## Convergent demands (multiple reviewers, highest priority)

1. **GRPO-own-teacher control** (R2-Q8, R5-Q1): schedule GRPO by its own
   mass functional. STATUS: built (GRPOMassTeacher), pre-registered with
   falsification branch (P-G0a/b), queued on GPU.
2. **Multi-seed steering-controlled LLM cell** (R2-Q1, R3, R5-Q2):
   STATUS: E-LLM-1b running (m3s mid-run; g3s/g3p/g3u follow).
3. **Corrected-decay gate at >=3 seeds / dose sweep** (R2-Q7, R3-Q6,
   R4-Q4): the dial's full-strength point is n=1 post-bug.
   STATUS: to queue after current GPU work (~6h/run x 3).
4. **Dead-group-skip / resample-to-fill dose baseline** (R2-Q3, R4-Q5):
   if skipping captures the 83% dose share, recycling's deployment case
   reduces to the +0.037 direction term. STATUS: design needed — on
   Countdown, arm = B1 + skip dead groups + refill batch with fresh
   prompts at matched GENERATION budget.
5. **Abstract emphasis** (R3-Q4, R5-Q2): registered outcome first.
   STATUS: FIXED and pushed.

## Single-reviewer items already fixed

- Floor p-value disclosure + one-sidedness (R3): FIXED.
- 6/6 -> 3/3-on-two-meters tally (R3): FIXED.
- Countdown mean-gain noise qualification (R3-Q5): FIXED.
- HER goal-selection prior art + gate novelty narrowed to derivation
  (R4-Q6): FIXED (5 citations added).
- Prop 4 -> attributed Remark; sharpening scoped to "first measurement
  in RLVR" (R4-Q1): FIXED.
- Relabel granularity in Alg 1 (R4-Q2): FIXED.

## Open items requiring text work

- Run-accounting table (R3-Q1): 9 vs 22 vs ~30 runs reconciliation.
- Stratified/paired permutation reanalysis (R3-Q2) — CPU, artifact
  exists (maze JSONLs); compute both variants, report alongside.
- §6.7 rewrite once E-LLM-1b lands (all reviewers flag its density).
- Claims-vs-evidence-strength table (R5-W4) — natural home: §6 preamble
  or App A.
- Gate cold-start behavior in-paper (R2-Q6, R4-Q3).
- Loss-placement pseudocode for relabeled groups (R2-Q4) — the module
  docstring content, promoted to appendix.
- FrontierMax at ~1-epoch pools honesty paragraph (R2-Q2/W3): per-prompt
  posteriors cannot work at 10^5-prompt pools; state bucket-posterior /
  difficulty-model extension or scope claim to small pools.
- Prop 3 justification or removal (R5-Q6).

## Open items requiring runs (GPU queue order)

1. E-LLM-1b (running) — decides §6.7.
2. sweep_grpo_own (queued) — decides title-claim scope.
3. Corrected-gate 3 seeds (~18h).
4. Dead-group-skip dose baseline on Countdown (~18h).
5. (stretch) 7B single run if hardware materializes — R2's ask; not
   feasible on one A10G.

## Meta-read

R5's AC paragraph is the strategy: don't let honesty be priced as
weakness or as substitute for evidence. The path from ~5 to ~6.5-7:
items 1-4 above (all queued/running), the §6 restructure, and scoping
every LLM-scale sentence to what the runs support. The lasting-value
framing to protect in all edits: "measure pass@k under your data
intervention, conditioned on your estimator" + sharpening named before
it ships at 70B.

## R1 (theory) additions to the plan

- NEW DECISIVE ABLATION (R1-Q3, unanimousable): Dr.-GRPO / no-std arm.
  Theory predicts mass ~ RLOO => easy-band coverage held. Sharpest
  falsifiable consequence; also disambiguates success-conditioning vs
  variance normalization. Maze arm cost ~2.4h x 3 seeds. QUEUE IT.
- State the mass->coverage bridge as an explicit hypothesis (text fix).
- Reframe partition as exposition; put Prop 2's tail ratios forward as
  the estimator-specific content (text fix, aligns with R5-W6).
- Quantify Lemma 1 magnitude at N=16 (one MC cell, CPU).
- Step-matched uniform-only GRPO arm (R1-Q7) — partially answerable
  from existing step-matched analysis; check opus5 data first.

## Final convergence map (what >=3 reviewers demand)

1. Multi-seed steering-controlled LLM cell (R1,R2,R3,R5) — RUNNING.
2. GRPO-family estimator ablations: own-teacher (R2,R5) + no-std
   (R1) — own-teacher QUEUED; no-std arm to add to same sweep.
3. Gate at designed operating point >=3 seeds (R2,R3,R4) — TO QUEUE.
4. Section 6 restructure + claims table (R1,R2,R5) — after E-LLM-1b.
5. Scale honesty: abstract/title claims vs 360M evidence (R1,R2,R3,R5)
   — partially fixed; final pass after verdicts land.
