# Validation record — 2026-08-04

This record accompanies the major revision of *The Estimator Decides*. It
separates corrections that are complete, CPU experiments reproduced in this
checkout, historical evidence that was reclassified, and experiments that
remain outside the evidence available on this Mac.

## Resolved mathematical and implementation issues

- The manuscript and code now distinguish three finite-group estimators:
  raw MaxRL, full-control-variate MaxRL, and the practical centered/drop
  estimator used in the experiments. GRPO and RLOO are reported separately.
- The expected practical centered/drop MaxRL coefficient mass is stated as
  `2(pass@N - pass@1)`, and the expected update is factorized explicitly.
- The paper no longer infers finite-group activity from a population weight
  alone. Practical MaxRL is zero on constant-reward groups; full-control-
  variate MaxRL assigns nonzero negative coefficients and invokes the policy
  update callback on an all-fail group. The testbed records callback
  invocation, not a guaranteed nonzero parameter gradient.
- GRPO uses the sample standard deviation over the requested group, including
  its finite-`N` scale factor and the implementation stabilizer. Historical
  three-seed summaries were converted from population SD to sample SD.
- Hindsight relabeling is described as a data-proposal change with an
  off-policy caveat, not as an equivalent estimator coefficient.
- The corrected grid and Countdown adapters choose one shared destination,
  rewrite every row, re-verify every trajectory, and require mixed outcomes.
  Their public teacher IDs remain coarse ring/tier buckets rather than exact
  destination identities. Regression tests include a grid trajectory that
  reaches the shared anchor and then ends elsewhere.
- Historical neural-maze and Countdown artifacts used different semantics:
  isolated per-trajectory positive imitation and per-trace achieved targets,
  respectively. They do not validate the proposed common-destination loop.

## New reproducible CPU controls

All three experiments use 3,200 groups of 16 rollouts. Paired arms receive the
same task IDs and rollout uniforms. Each replicate creates independently
spawned rollout and task-schedule streams with
`SeedSequence([replicate_id, 20260804]).spawn(2)`, avoiding deterministic
cross-replicate stream reuse from the earlier additive-offset scheme while
retaining common random numbers within a pair.

### Common-learning-rate finite-group comparison

`curriculum_maxrl/run_estimator_variants.py` evaluates 20 paired streams. On
the balanced pool, evaluation-grid AUC (mean ± sample SD) is 0.7209 ± 0.0210
for raw MaxRL, 0.6934 ± 0.0189 for full CV, 0.7183 ± 0.0231 for practical
MaxRL, 0.3968 ± 0.0182 for GRPO, and 0.1416 ± 0.0098 for RLOO. These arms use
one common learning rate and therefore are a mechanism control, not a tuned
estimator ranking.

On the frontier-heavy pool, whose maximum initial exact pass rate is
`1e-5`, full CV assigns nonzero negative coefficients and invokes the
policy-update callback on every all-fail group; its mean AUC is
`1.740e-6`. Practical MaxRL plus exact hindsight reaches
0.9269 ± 0.0032 AUC and 0.9788 ± 0.0006 final mean, winning all 20 paired
comparisons against full CV (two-sided exact sign-test `p = 1.907e-6`). This
supports the narrow conclusion that this all-fail coefficient branch did not reliably
bootstrap this tabular frontier, while verified hindsight did by changing
conditioning, destination, and proposal.

### Full-CV learning-rate sensitivity

`curriculum_maxrl/run_fullcv_lr_sensitivity.py` is a post-hoc robustness
sweep over ten learning rates. Median frontier-heavy AUC stays near zero at
every rate. One of 20 streams crosses the prespecified final-mean threshold at
learning rates 1.25 and 2.0; no other rate has a crossing. This is recorded as
non-confirmatory sensitivity evidence, not a tuned success.

### Fixed-schedule, held-out tuning comparison

`curriculum_maxrl/run_schedule_matched_estimators.py` uses ten tuning streams
(IDs 1000–1009), a one-standard-error selection rule, and twenty separate
held-out streams (IDs 2000–2019). Selected raw learning rates are 12 for
practical MaxRL, 16 for GRPO, and 64 for RLOO. Held-out AUC is
0.9889 ± 0.0089, 0.9841 ± 0.0121, and 0.9869 ± 0.0099, respectively; the
largest mean separation is 0.0048. The pool is near ceiling, and both the
selected raw rates and ordering depend on parameterization and the tuning
rule. This control does not identify an intrinsic estimator ranking or the
adaptive neural-maze curriculum-by-estimator interaction.

## Claim and evidence corrections

- Earlier broad claims about estimator superiority, frontier resurrection,
  gate recovery, and generalization were withdrawn or narrowed to the exact
  supported setting.
- Countdown reports a three-seed aggregate mean-versus-pass@16 sharpening
  pattern for a legacy per-trace recycler. The individual endpoint rows are
  not vendored, and only two pairs retain checkpoint
  trajectories, and their coverage paths have different timing; no general
  early-stopping claim is made. The unintended moderate gate is a point
  estimate; the corrected stronger gate remains a single-seed endpoint.
- The maze analysis is descriptive. Its AUC is normalized trapezoidal
  optimizer-step AUC on each run's own logged grid; wall-clock is the stopping
  budget, not the AUC abscissa. Figure 8 reads a frozen 22-row registry
  containing four GRPO-labelled and eighteen practical-MaxRL-labelled runs;
  recurring warm starts and unbalanced sampler composition are not treated as
  independent factorial replication.
- The GSM8K study is labelled as a two-seed treatment-delivery pilot with a
  replacement confound, not as confirmatory interaction evidence. Only 2,561
  of 7,473 prompts were visited across 3,200 groups (0.428 visits per pool
  prompt; 1.25 per visited prompt).
- The historical exact-score placebo battery is treated as behavioral
  evidence, not a causal dose--direction decomposition: update counts/norms
  were not logged, and the random-target arm used a stale reward vector and
  an unseeded draw rather than exact re-verification.
- The manuscript distinguishes independent training-seed SD from repeated
  fixed-checkpoint evaluation noise and reports pass@k alongside means.
- Missing generated-token counts are stated rather than reconstructed.
- Unsupported historical one-shot, gridworld rewrite, maze utility-form, and
  adaptive-truncation numbers were removed or explicitly marked non-vendored.
- The adaptive-T subset estimator now retains the all-fail control variate,
  recovers full-CV MaxRL at `T=N`, and has exact expectation tests. Its prior
  empirical comparison is withdrawn pending a schedule-matched full-CV study.

## Reproducibility and document checks

- `python -m pytest -q frontier_rl/test_framework.py curriculum_maxrl/test_verl_curriculum.py curriculum_maxrl/test_estimators.py`
  passes: 42 tests, with one
  upstream PyTorch sampler deprecation warning.
- Each committed result artifact was independently reproduced into a fresh
  temporary directory against the current experiment sources. JSON content matched
  exactly after excluding only the recorded command string, whose explicit
  output path necessarily differed.
- SHA-256 values recorded in all three result artifacts match their current
  experiment and estimator sources.
- `make -C paper package` is the authoritative repository-root build/check
  command. The readable draft builds to 19 pages and the ICLR-styled version
  to 18 pages.
  Their core, appendix, and bibliography are synchronization-checked;
  bibliography placement differs only to follow each layout convention.
- The final LaTeX logs contain no overfull boxes, undefined references or
  citations, duplicate destinations, package errors, emergency stops, or
  fatal errors.
- Every page of both final PDFs was rendered to an image and visually
  inspected. Figures, tables, equations, references, and page boundaries are
  readable, with no observed clipping or overlap.
- The final PDFs have blank author metadata and retain the anonymous-author
  byline.

## Evidence still required for stronger conclusions

The revision does not claim completion of experiments that cannot be
reconstructed or run credibly from the available artifacts and local
hardware. The main remaining studies are:

1. a balanced schedule-replay neural-maze factorial with independent
   seed/warm-start blocks;
2. a corrected fixed-decay multi-seed destination-gate sweep with a
   prespecified coverage noninferiority test;
3. a replacement-matched, steering-controlled, multi-seed LLM factorial with
   frozen evaluation and generated-token accounting;
4. independent IsaacLab locomotion replication; and
5. recovery of raw GSM8K, corrected-gate, IsaacLab, and complete Jugs
   run-level archives.

No neural or LLM experiment in this checkout jointly instantiates the exact
coefficient-mass sampler and the shared-destination contrastive relabeler.
End-to-end validation of the proposed combination remains open.

These are stated as limitations and future validation requirements, not
silently promoted to supported claims.

## Guidance provenance and anonymity

The exact PDF, TeX, and Markdown guidance inputs supplied on 2026-08-04 are
preserved under `research_guidance/2026-08-04/` with SHA-256 checksums. They
are review inputs rather than experimental evidence. Because the supplied
documents retain reviewer-facing repository and author references, that
directory must be excluded from any anonymous conference-submission bundle.
The repository's public `docs/index.html` also contains project-owner links;
an anonymous bundle should contain the manuscript PDF and required paper
assets only, not the public project site.
