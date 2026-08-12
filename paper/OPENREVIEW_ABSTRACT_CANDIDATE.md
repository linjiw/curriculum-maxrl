# OpenReview submission candidate — ICLR 2027

**Prepared:** 2026-08-12. **Freeze deadline:** Sept 16 (abstract due Sept 18).
**Source of truth:** `paper/main_iclr2027.tex` (title) and `paper/body_iclr.tex` (abstract). If those change, regenerate this file; do not edit the abstract here independently.

## Title

Coefficient Activity and Data Selection in Verifiable-Reward RL

## Abstract (plain text for the OpenReview form; synced to body_iclr.tex 2026-08-12 post-referee revision)

RL with verifiable rewards often treats task curricula and failure recycling as estimator-agnostic. We ask what task-level activity the deployed group estimator makes available to either intervention. For the practical MaxRL convention that drops all-fail groups, the expected absolute coefficient mass over N i.i.d. binary rollouts is exactly 2(pass@N − pass@1) — the estimator's own pass@N gap. The score's peak moves toward harder tasks as N grows, it recovers the common learnability score p(1−p) at N=2, and dropping all-fail groups shifts the truncation order to N−1. It is an estimator-side diagnostic, not a theorem of learning progress. Rollout-aware selection carries empirical content: in a fresh 20-seed Acrobot tournament at N=16, sampling by the deployed-N score improves target-uniform AUC over p(1−p) by +.0480 (95% paired-bootstrap CI [+.0209, +.0738]) and over uniform sampling as well. Its scope is bounded by design: in an exact-probability Digits counter-test the mismatched RLOO estimator prefers the same sampler, rejecting a universal estimator-to-sampler mapping, and an externally recorded six-block maze factorial reports higher time-integrated coverage under MaxRL than under GRPO in 6/6 blocks under each sampler at common optimizer settings — an estimator-conditioned ordering, not universal superiority. A reported three-seed Countdown aggregate couples higher mean@16 with a lower logged VERL bootstrap best@16 coverage proxy under a deployed recycling package, motivating pass@k recomputed from retained raw outcomes. Data-selection interventions should be evaluated with the estimator beneath them, and coefficient activity treated as a source of curriculum hypotheses, not a universal curriculum objective.

## Pending conditional update

If the E2c nine-arm endpoint completes before the freeze, the Countdown sentence must be replaced with the prereg-worded result (standard observed-set pass@16 from retained binary outcomes, three seeds, paired contrasts) — direction as observed, wording per `autoresearch/iterate-260810-2240/E2C_PREREG.md`.

## Form metadata suggestions

- Primary area: reinforcement learning
- Keywords: reinforcement learning with verifiable rewards; curriculum learning; group-relative estimators; pass@k; task selection; MaxRL
- TL;DR: The deployed MaxRL estimator's exact per-task coefficient mass, 2(pass@N − pass@1), is an N-aware task-selection score; controlled positive and negative experiments map where it helps and where it provably cannot.
