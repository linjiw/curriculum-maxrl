# OpenReview submission candidate — ICLR 2027

**Prepared:** 2026-08-12. **Regenerated:** 2026-08-17. **Freeze deadline:** Sept 16 (abstract due Sept 18).
**Source of truth:** `paper/main_iclr2027.tex` (title) and `paper/body_iclr.tex` (abstract). If those change, regenerate this file; do not edit the abstract here independently.

## Title

Coefficient Activity and Data Selection in Verifiable-Reward RL

## Abstract (plain text for the OpenReview form; regenerated from body_iclr.tex 2026-08-17 after the peak-location and AMaze results)

RL with verifiable rewards often treats task curricula as independent of the group estimator that converts sampled outcomes into updates. We ask which tasks that estimator makes active. For the practical MaxRL convention that drops all-fail groups, the expected absolute coefficient mass over N i.i.d. binary rollouts is exactly 2(pass@N-pass@1), the estimator's own pass@N gap. Its peak moves toward harder tasks as N grows, it recovers p(1-p) at N=2, and the all-fail rule shifts the corresponding truncation order to N-1. This is an estimator-side diagnostic, not a theorem of learning progress. In a fresh 20-seed Acrobot tournament at N=16, the deployed-N score improves target-uniform AUC over p(1-p) by +.0480 (95\ in this fixed eight-threshold family; the effect replicates on two further platforms (+.0322, +.0307). Two preregistered follow-ups then bound the claim: holding the deployed estimator at N=16 and sweeping the score exponent, performance rises past} N=16 (argmax at u_64), so the deployed-N peak location} is not what helps — harder-peaked scores are; and dropped into robust PLR on AMaze in place of MaxMC, the pure activity priority is starved of signal at one Bernoulli per level visit and does not beat the upstream baseline. An exact-probability Digits counter-test rejects a universal estimator-to-sampler mapping. On neural gradients, an externally recorded six-block maze factorial finds higher time-integrated coverage under MaxRL than GRPO in 6/6 blocks under each sampler, an estimator-conditioned ordering rather than a direct score test. Data-selection methods should therefore be evaluated with the estimator they ship, while coefficient activity is treated as a rollout-aware curriculum hypothesis rather than a universal objective.

## Pending conditional update

If the E2c nine-arm endpoint completes before the freeze, the Countdown sentence must be replaced with the prereg-worded result (standard observed-set pass@16 from retained binary outcomes, three seeds, paired contrasts) — direction as observed, wording per `autoresearch/iterate-260810-2240/E2C_PREREG.md`.

## Form metadata suggestions

- Primary area: reinforcement learning
- Keywords: reinforcement learning with verifiable rewards; curriculum learning; group-relative estimators; pass@k; task selection; MaxRL
- TL;DR: The deployed MaxRL estimator's exact per-task coefficient mass, 2(pass@N − pass@1), is an N-aware task-selection score; controlled positive and negative experiments map where it helps and where it provably cannot.
