# Wave-2 confirmation factorial — VERDICT (2026-08-06)

Prereg: run_factorial_wave2.sh (commit d6aea90, before any run).
Analyzer: fact_analyze.py --seed-start 6 --seeds 6 (code path frozen
from wave 1). All 24 cells final; raw results in
results_factorial.json + fact250_*_s{6..11}.jsonl.

## P-F2 (registered primary): CONFIRMED at 6/6 + 6/6

Paired (same block, same sampler) MaxRL − GRPO on cov_auc_delta
(time-integrated coverage):

- uniform: **6/6 positive**, mean +.0150
- frontier_un: **6/6 positive**, mean +.0240

Bar was ≥5/6 under BOTH samplers; 6/6 gives exact two-sided sign-test
p = 0.031 per sampler. The ordering first found exploratorily in
wave 1 is now **registered and confirmed on fresh randomness**. The
two sampler contrasts share each seed/warm-start block and are not
independent replicates. Averaging samplers within block gives 6/6
positive wave-2 blocks (mean +.01950, post-hoc 95% t interval
[+.01148,+.02752]); all 12 independent block averages across both
waves are positive descriptively (mean +.02175, interval
[+.01663,+.02688]), with no pooled confirmatory p-value.

## P-F3 (registered secondary): PAIR-LEVEL BAR MET; localization suggestive

Easy-band (L1–3) endpoint delta, MaxRL − GRPO paired: 10/12 positive
(registered bar ≥7/12). These are two correlated sampler observations
inside each of six seed blocks. After sampler averaging, 4 blocks are
positive, 1 is tied, and 1 is negative; the mean is +.08333 and the
post-hoc 95% t interval [−.00330,+.16996] includes zero. We therefore
record that the registered pair-level bar was met but treat easy-band
localization as suggestive rather than established; no p-value is
attached to the correlated pair count.

## Unregistered observations (stated for completeness, no claims)

- The wave-1-style ENDPOINT contrast, which failed in wave 1 (3/6,
  4/6), lands 5/6 + 5/6 on the fresh blocks — consistent with the
  power analysis (endpoint power 0.4–0.8 at this effect size: sometimes
  it clears, sometimes not). The endpoint claim stays retired; the
  registered claim is the covAUC ordering.
- Interaction read again mildly protective/neutral under GRPO
  (teacher−uniform Δcov8 under GRPO: 5/6 non-negative) — consistent
  with wave 1; still no corroboration for the GSM8K regression pattern.

## Per the committed branches (WAVE2_BRANCHES.md): BRANCH A executes

- Paper §6.3b: exploratory sentence upgraded to registered+confirmed.
- Contribution 3, abstract, conclusion: updated per branch text.
- TITLE: **stands** ("The Estimator Decides") per the committed
  decision rule — it now names a registered, twice-replicated result.
- Limitations power paragraph: closes its loop.

The two-wave arc, in one line: the endpoint form of the claim died at
its registered test; the time-integrated form was then registered on
fresh blocks and confirmed at 6/6 + 6/6 — the claim that survives is
the one the power analysis said the design could actually detect.
