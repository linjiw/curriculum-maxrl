# Reviewer 4: HER/goal-conditioned RL — Rating 5/10, Confidence 4/5

## Sharpest catches
1. PROP 4 IS DEFINITIONAL and its content is known: Eysenbach et al. 2020
   (relabeling-as-inverse-RL, induced distribution), Ghosh et al. 2021
   (conditional-law gap in GCSL's bound), bias-corrected HER line. Should be
   a Remark with attribution OR strengthened into a real bound linking
   conditional-law divergence to the observed coverage loss. NOTE: the
   paper's own sharpening mechanism runs through the MARGINAL, so Prop 4
   does no load-bearing work for the central recycling finding.
2. SHARPENING = Skew-Fit/GCSL pathology renamed: richer-get-richer
   concentration on reachable set is Skew-Fit's founding motivation
   (Pong et al. 2020 — NOT CITED). What's new: the currency (pass@k in
   RLVR) + setting. Soften "new failure mode" to "first measurement in RLVR".
3. GATE PRIOR ART uncited: GoalGAN intermediate-difficulty (Florensa 2018),
   HGG reachability (Ren 2019), Curriculum-guided HER (Fang 2019 — literally
   gates relabel-goal admission), Skew-Fit alpha as the same dial. Residual
   novelty = threshold DERIVED from u_N (unifying teacher and gate) — narrow
   the claim to that.
4. PLACEBO ACCOUNTING: 83% of +0.22 = compute utilization; direction worth
   ~17%; one-shot streams +0.01; the uniquely-alive regime has pool pass
   rate 1e-5 (constructed corner). Abstract foregrounds the corner case.
   MISSING BASELINE (Q5): DAPO-style dead-group skip with reclaimed compute
   on more prompts — if that captures the 83%, deployment case = +0.037.
5. Gate mitigation = 3 under-gated seeds + 1 corrected seed + deferred sweep.
   Phenomenon better supported than its mitigation.
6. Exactness verified only where true by construction; no measurement
   instrument for practitioner environments (probe was uninformative).
7. RELABEL GRANULARITY AMBIGUITY (Q2): per-group or per-trajectory? If
   per-trajectory all-success, K=N makes stated weights vanish.
   [ANSWER EXISTS: per-trajectory, but groups keep unparseable failures so
   K<N — the hindsight.py mixed-target caveat; MUST go into the paper.]
8. Gate cold-start on never-seen destinations unspecified in paper.

## Path to 6: (a) related-work rewrite + Prop 4 repositioned w/ attribution,
   (b) corrected-decay gate sweep >=3 seeds, (c) dead-group-skip dose baseline.
