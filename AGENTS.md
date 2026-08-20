# AGENTS.md — Curriculum-MaxRL research operating contract

This file governs work in the whole repository. It is a research-integrity
contract, not a project summary. Read it before changing code, experiments,
artifacts, the website, or either manuscript.

## 1. Mission and current thesis

Target: ICLR 2027. The official abstract deadline is 2026-09-18 AOE and the
paper/supplement deadline is 2026-09-25 AOE. The submission main text is at
most nine pages; references, the required AI-use statement, and the optional
reproducibility statement do not count toward that limit.

The paper is a boundary-mapped theory paper, not a universal curriculum-system
paper. Its current spine is:

> A finite-group estimator acts on the success-count law of the unit it
> consumes. Mean pass rate is sufficient for coefficient activity only under
> the appropriate atomic, conditionally-i.i.d. model. At a coarser curriculum
> unit, score the count law rather than pretending that the aggregate is one
> Bernoulli task. Even exact coefficient activity is not learning utility.

For practical centered MaxRL with all-fail groups dropped,

```text
M_MaxRL(k) = 2 (1 - k/N) 1{k>0}
A_N(Q)     = 2 (Pr_Q[K>0] - E_Q[K]/N)
```

under any binary group law. The familiar
`2(pass@N - pass@1) = 2(1 - p - (1-p)^N)` is the conditionally-i.i.d. atomic
slice. The count-law framework is

```text
A_E(z) = sum_k P(K=k | z) M_E(k).
```

The nonnegative aggregation-gap corollary requires the stated mixture regime:
one atomic instance is shared by a group and rollouts are conditionally i.i.d.
given that instance. Do not extend its sign to arbitrary count laws; an
under-dispersed or anti-correlated group law can reverse it.

Use one term in prose: **coefficient activity**. Gloss “advantage mass” once if
needed. Activity is an estimator-controlled contrast envelope and an exact
zero diagnostic. It is not gradient norm, signal-to-noise ratio, policy
improvement, learning progress, or long-horizon curriculum value.

## 2. Source-of-truth precedence

When files disagree, use this order:

1. A frozen preregistration plus its immutable analyzer and terminal result
   artifact govern that experiment. A later result supersedes an earlier
   status memo.
2. `GUIDANCE_RECONCILIATION_2026-08-19.md` governs scientific scope. It adopts
   the count-law pivot while retaining the process rules in
   `EDITORIAL_CHARTER_2026-08-19.md`.
3. `paper/body_iclr.tex` governs the submission claim perimeter;
   `paper/body.tex` is the extended record.
4. `paper/CLAIM_TRACE_ICLR.md`, structured analysis JSON, and source manifests
   govern numbers. Untraceable numbers do not ship.
5. `LITERATURE_POSITIONING.md` governs literature wording only after its
   source checks are committed and reflected in the paper.
6. README, website, research notes, progress reports, and plans are navigation
   or working memory. They never promote a claim.

Explicitly superseded designs remain useful audit records but are not plans of
record. In particular,
`granularity_flip/GRANULARITY_FLIP_PREREG_v1_SUPERSEDED.md` is superseded by
`granularity_flip/GROUP_LAW_FLIP_PREREG.md`.

## 3. Evidence language

Every load-bearing statement belongs to exactly one tier, and its verb must
match.

- **Tier 1 — proved and machine-verified.** State as an identity/theorem with
  assumptions. This includes the count-law masses, the arbitrary-law MaxRL
  identity, the conditional-i.i.d. reduction, the factorization, the practical
  `T=N-1` truncation result, and the properly scoped granularity corollary.
- **Tier 2 — preregistered and confirmed.** “Confirmed” is reserved for a
  frozen primary that executed and passed its rule.
- **Tier 2' — controlled but descriptive.** Say “descriptive” or
  “exploratory” at the point of use. Do not attach a p-value after deciding to
  treat a result as descriptive.
- **Tier 3 — preregistered and bounded/refuted.** Report the negative and its
  diagnosis without weakening the frozen rule.
- **Tier 4 — open or inconclusive.** Keep out of the abstract and contribution
  perimeter. A plan, smoke test, delivery failure, or one-of-two seed pattern
  is not evidence.
- **Retired.** Delete on sight rather than resurrecting it with softer words.

Independent seed/warmstart blocks, not correlated evaluations or rollout
groups, are the replicate for training-method claims. Every `+/-` must name
whether it is SD, SE, or an interval. “Registered” requires a frozen immutable
record before the first scientific run; an internal draft is not registered.

## 4. Scientific status as of 2026-08-20

### Established and usable

- The count-law theory is implemented in
  `curriculum_maxrl/group_law_teacher.py` and is checked against deployed
  MaxRL, RLOO, and GRPO coefficients. The jittable sufficient-statistic port
  for MaxRL/RLOO lives in `curriculum_maxrl/count_law_stats.py`; preserve it
  and its tests.
- Acrobot is the central registered positive: at deployed `N=16`, `u_16`
  beats `p(1-p)` by `+.0480`, paired 95% CI `[+.0209,+.0738]`. The two later
  platform results are portability replications on the same seeds, not two
  new independent seed cohorts.
- The exponent sweep bounds peak specificity: performance rises beyond the
  deployed `N` and peaks at `u_64` in the tested sweep. Safe wording: a
  harder-peaked shape helped there; deployed `N` is a floor on the score
  exponent, not an identified optimum.
- Pure activity priority fails as a replacement for MaxMC in AMaze because it
  receives one terminal Bernoulli per visit rather than a per-timestep critic
  signal. This is a bandwidth boundary, not evidence that the algebra is
  false.
- MAZE-SCORE is a 48-block registered negative for `u_32` versus `p(1-p)`:
  `-.00324`, CI `[-.00543,-.00111]`. Its telemetry verifies the coefficient
  calibration gap to floating point. The telemetry does not prove mediation
  of the downstream endpoint.

### Open, and never to be described as running evidence

- **P0 count-law flip is frozen and running blind.** The protocol was frozen
  in commit `f27ba8a` before the first evidence submission. The clean evidence
  manifest is `b0cf3d2d...f3a2c95a`; Hopper array `9419991` covers seeds
  3001–3024, while 3025–3048 waits only for queue capacity. Do not open result
  JSONL or endpoint summaries. Scheduler state, completion receipts, hashes,
  and telemetry integrity are the permitted monitoring surface. Standalone
  tests or the successful smoke `9419940` are not intervention evidence.
- P0 uses 48 paired blocks and requires the observed mean to be at least its
  `+.005` SESOI. It is powered at `.901` for `+.0075` under the pessimistic
  historical paired SD `.0135`; a true effect exactly at the SESOI still has
  about `.503` support probability because of the point-estimate clause.
- The prospective P0 verdict text must not say that Corollary 2 predicted a
  downstream endpoint sign. A positive intervention can show that count-law
  correction matters on this substrate; the corollary itself predicts only
  coefficient-activity calibration.
- The full-budget AMaze gate rerun in
  `/data/robotixx/ued_bench/gate-confirmatory-20260819` is terminal at 20/20
  training cells and 20/20 evaluations, but it is not yet a result. There are
  no `ckpt_budget.json` receipts or `DONE` markers and no local
  `AMAZE_GATE_ANALYSIS.json`. Do not inspect evaluation values or state a
  verdict until the outcome-blind checkpoint-budget closure passes and the
  frozen analyzer is run once. Before reusing its driver, replace checkpoint
  presence as the restart predicate with an explicit completion marker.
- The LLM teacher-by-estimator claim is `1-of-2 seeds,
  treatment-intensity-dependent`, hence open. The precommitted go/no-go is due
  2026-08-26. A second delivery-gate failure permanently de-scopes it. Do not
  spend 40–80 A10G-hours unless a frozen-checkpoint smoke first shows that the
  coarse-state treatment moves the delivery diagnostics.

### Literature correction that must land before submission

At fixed group size, SFL's realized score `(k/N)(1-k/N)` equals the RLOO
realized coefficient mass times `(N-1)/(2N)` at every `k`. Therefore SFL is an
existing realized count-law curriculum for RLOO, not an unprincipled heuristic
that this work replaces. The surviving wedge is estimator-specific mass shape
(MaxRL peaks at `k=1`, RLOO/SFL at the middle), coarse-unit pooling bias,
variable-`N` semantics, and scoring cost. Preserve the user's uncommitted
correction in `LITERATURE_POSITIONING.md` and propagate it to the paper,
README, and website before any external release.

## 5. Priority queue and acceptance tests

Work in this order. Do not start optional scale experiments while a higher
item is red.

### P0 — close the count-law intervention

1. Finish the P0 design review before freeze: full-rule power, powered-for
   effect, treatment-delivery estimand, missing-cell behavior, both calibrated
   verdict branches, and exact artifact paths.
2. Integrate the group-law score into the existing MAZE-SCORE trainer while
   holding estimator, `N`, warmstart, task generator, budget, floor, and seeds
   fixed. The arm difference must be only the statistic used for sampling.
3. Maintain fail-closed synthetic and smoke tests: identical priors at
   initialization, deliberate separation on same-mean/different-count-law
   streams, terminal model checkpoints and zero-exit receipts, complete 2x48
   matrix, exactly ten evaluation points, arm-paired source/warmstart hashes,
   treatment-delivery accounting, and analyzer single-use protection.
4. Freeze the preregistration, analyzer hash, source manifest, environment,
   campaign ID, and both verdict branches in a commit before the first run.
5. Run blind. Do not inspect endpoint-bearing files until every cell is
   terminal and hashes/completeness pass. Analyze once. Report the registered
   verdict even if negative.
6. Integrate the result into the abstract, main evidence section, limitations,
   claim trace, registry, manifest, README, and website at the correct tier.

P0 is the minimum missing scientific closure for this ICLR story. A 7B+ LLM
training result would improve impact, but it is not allowed to displace this
causal intervention. If extra compute remains, prefer a frozen-checkpoint
LLM count-law calibration study before committing to full RL training.

### P1 — repair the manuscript's live state

- Replace both claims that P0 is “registered and running.” Until freeze use
  “drafted, not launched”; after freeze use “preregistered, not yet run”; only
  a live terminal-tracked campaign may be “running.”
- Replace the obsolete per-level-versus-per-task description with the current
  plug-in-versus-count-law design.
- Rewrite the SFL paragraph using the exact RLOO identity. Never use
  “principled instead of heuristic” as the novelty claim.
- Decide the title with the PI. Do not silently override the current title.
- Regenerate `paper/OPENREVIEW_ABSTRACT_CANDIDATE.md` from the current title
  and abstract; it presently describes the pre-pivot paper.
- Keep the abstract to at most four numbers. Keep the conclusion within page
  9. Cut whole low-tier rungs before thinning every section.

### P2 — make the artifact claims true

- Rebuild `paper/results/manifest.json` around every figure actually used by
  the compact paper. The current manifest predates the counterexample,
  claim-map, and MAZE-SCORE figures.
- Repair `reproduce.sh`. The pinned verify path currently fails byte comparison
  at `fig1_utility.pdf`; one derivation still reaches outside the repository at
  `../maxrl`; the pinned build hashes predate the count-law manuscript. Provide
  a documented portable verification path and retain a stricter byte-exact
  path when the pinned toolchain is available.
- Refresh `paper/CLAIM_TRACE_ICLR.md` after every quantitative edit. The latest
  trace reaches rows 76–82, but the introductory count and historical header
  are easy to misread; report one unambiguous total.
- Build and verify from a clean anonymous clone. No source claim may depend on
  an unpublished branch, a personal absolute path, or an unlisted external
  cache. If raw artifacts must remain external, disclose that boundary rather
  than calling the release one-command complete.
- Scan the PDF, supplement, repository snapshot, metadata, and links for
  identity leaks. ICLR 2027 double-blind violations can be desk-rejected.

### P3 — submission operations

- By 2026-08-26: bind the LLM go/no-go and default to de-scope if delivery is
  not demonstrated cheaply.
- By 2026-09-12: freeze the claim table and evidence perimeter.
- By 2026-09-16: lock a genuine OpenReview abstract and title; confirm all
  authors, ordering, profiles, and reciprocal-reviewing eligibility.
- By 2026-09-22: freeze the full draft; after that, correctness, anonymity,
  references, figures, and artifact checks only.
- By 2026-09-24: clean-clone reproduction green or every remaining portability
  limitation stated explicitly.

## 6. Experiment discipline

- A preregistration is immutable after execution starts. Amendments are new,
  timestamped records and may not change an observed scientific endpoint.
- Draft both outcome branches before data. If one branch is hard to write, the
  endpoint is underspecified.
- Never add seeds, change an endpoint, substitute a metric, or rerun an arm for
  a scientific reason after looking. Engineering reruns require outcome
  blindness, a written defect, preserved failed artifacts, and unchanged
  scientific settings.
- Completion means a validated final training state plus an explicit marker,
  not merely a checkpoint file or zero process count.
- Treatment-delivery gates are conjunctive and final. A miss by `0.00148` is a
  miss. A failed gate makes efficacy inconclusive, not negative.
- Match the cost the claim names: rollouts/tokens for generation claims,
  environment steps for control, optimizer updates for dose, and wall clock
  only for throughput claims. Free probe sweeps are not budget matched.
- Report anytime AUC and final performance when long-horizon utility is at
  issue. Preserve raw per-task binary outcomes for standard pass@k.
- Never feed relabeled outcomes into the requested-task posterior. Relabeling
  is appendix/next-paper material unless a new registered result promotes it.

## 7. Writing law

- Lead with the same-mean/different-count-law counterexample. Every theorem or
  proposition gets a plain-English interpretation immediately after it.
- Say: “The estimator defines the coefficient map; the curriculum defines the
  unit over which that map is averaged. These operations do not commute.”
- Say: “activity ranks utility well and locates it poorly” only with the
  supporting utility-audit context.
- Do not say: universal curriculum law, universal estimator superiority,
  deployed-`N` optimum, neural-scale causal effect, the corollary predicted the
  endpoint, SFL is merely heuristic, or a delivery-gated LLM result is
  established.
- Historical Countdown metrics are “VERL bootstrap best@k proxy,” never
  standard pass@k. Standard pass@k requires retained raw outcomes.
- `p(1-p)` is RLOO/SFL's count-law shape and the `N=2` MaxRL slice. Those are
  complementary facts; do not erase the SFL precedence by stating only the
  second.
- Cross-estimator magnitude depends on normalization and learning rate.
  Emphasize shapes, exact zeros, tails, and matched or swept optimizer settings.
- Binary verifiable reward is the scope. Name graded rewards once as future
  work rather than implying the theorem already covers them.
- Negatives are findings with diagnoses. Never soften a frozen failure or
  compensate for it with an unregistered secondary.

## 8. Session startup and verification

At the start of any research or manuscript session:

1. Run `git status --short --branch` and preserve unrelated/user changes.
2. Read this file, `GUIDANCE_RECONCILIATION_2026-08-19.md`, the active
   preregistration/result for the task, and the relevant claim-trace rows.
3. Recompute days to the two official deadlines from the current date.
4. Check live jobs and receipts without opening endpoint-bearing data.
5. State which evidence tier the intended change affects. Do not edit first
   and decide the tier later.

Minimum theory checks:

```bash
python -m curriculum_maxrl.test_mass_formulas
python control_port/verify_note_claims.py
python -m unittest curriculum_maxrl.test_audit_countdown_sft_overlap
python curriculum_maxrl/test_verl_curriculum.py
python frontier_rl/test_framework.py
```

The group-law tests contain both legacy top-level imports and package-relative
imports, so run them in the matching contexts with an environment that has
pytest. If ROS/global plugins contaminate discovery, set
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

```bash
(cd curriculum_maxrl && python -m pytest -q test_group_law.py test_group_law_teacher.py)
python -m pytest -q \
  curriculum_maxrl/test_count_law_stats.py \
  curriculum_maxrl/test_relabel_degeneracy.py
```

For manuscript/artifact work, also run `git diff --check`, refresh every
affected figure from declared inputs, compile with the official ICLR 2027
style, confirm the conclusion ends by page 9, scan logs for undefined
references/citations and overfull boxes, and run the repaired reproduction
entrypoint. A green unit-test suite does not make a red manifest green.

## 9. Definition of submission-ready

The work is ready to submit only when all of the following hold:

- P0 has a frozen prospective verdict and is integrated at its earned tier, or
  the paper is explicitly rewritten so that no contribution depends on the
  missing intervention.
- No Tier-4 result appears in the abstract or contribution perimeter.
- The paper, abstract candidate, README, website, claim trace, registry, and
  experiment status all agree.
- SFL precedence and the arbitrary-law/mixture-law distinction are correct.
- Every quantitative claim resolves to a committed artifact and analyzer.
- The compact PDF is anonymous, uses the official style, and has at most nine
  main-text pages.
- Reproduction passes from the release artifact, or its unavoidable external
  dependencies and raw-data gaps are stated exactly.
- Two readers outside the project can answer, without coaching: What is the
  one claim? Which prospective experiment tests it? What result would refute
  its learning consequence?

Calibration is the product. A narrower paper whose complete claim perimeter is
true is preferable to a broader paper whose most visible evidence is still a
delivery failure or a draft protocol.
