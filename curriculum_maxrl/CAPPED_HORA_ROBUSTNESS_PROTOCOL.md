# Frozen post-guidance protocol: capped HORA robustness

Frozen 2026-08-08 before any outcome from this new matrix was computed.  This
is an exploratory diagnostic motivated by the completed HORA-style factorial;
it is not a preregistration, an independent confirmation, or a reproduction of
HORA's neural experiments.  The specification below was tightened after an
independent pre-run code/math audit; no matrix cell had run before the
amendment.

## Question

The first factorial found that mass-aware rollout allocation improved learning
AUC while *reducing* realized practical-MaxRL coefficient mass per completion,
and sometimes assigned roughly 76 rollouts to one prompt.  This study tests
how sensitive that pattern is to allocation concentration and probability
information quality; it cannot identify a unique causal explanation.

## Frozen matrix

Every cell uses the existing `SkillChainEnv`, practical group-size-specific
MaxRL weights, no hindsight, eight sampled tasks per synchronous update, four
probe completions per task, average final group size 16, exactly 51,200 paid
completions, checkpoints every 2,560 completions, and 16 paired logical seeds
`0..15`.

- Cross-prompt sampler: `{uniform, u_16}`.
- Fixed anchor: final group size 16 for every sampled task.
- Adaptive allocator: `{HORA hit marginal, fresh-group mass proxy}`.
- Maximum final group size: `{24,32,48,uncapped}`.  The minimum remains the
  four probes.  Allocation continues greedily among positions below the cap;
  every batch must still contain exactly 128 completions.
- Probability information for adaptive allocation:
  - `same_step`: `Beta(1+c,1+4-c)` from the four current probes only, matching
    the first factorial's HORA-style convention;
  - `history_plus_probe`: the task sampler's discounted pseudo-count state
    immediately before the batch, plus the current position's four probes;
  - `oracle_preupdate`: the simulator's exact task success probability under
    the frozen pre-update policy.  This is a non-deployable diagnostic, not a
    baseline claim.

The full design therefore has 50 cells per seed: two fixed anchors and
`2 samplers x 2 adaptive allocators x 4 caps x 3 information sources`.
No cell may be dropped after outcomes are seen.

The HORA-hit score for the next Phase-B rollout after `ell` additional
rollouts is the posterior expectation of `p(1-p)^ell`.  The fresh-group mass
proxy uses `p(1-p)^(4+ell)`, the one-rollout marginal of unconditional
`u_N(p)` after counting the four probes in final group size.  It is **not** the
exact conditional marginal change in realized coefficient mass after the
observed probe successes are pooled: that quantity also depends on the
realized success count and the changing coefficient normalization.  The old
short label “mass-aware” is retained only when linking the first artifact;
this matrix uses the precise `fresh_group_mass_proxy` name.
For `Beta(a,b)` information and exponent `e`, the registered posterior score
is the exact moment `B(a+1,b+e)/B(a,b)`. The oracle score substitutes the
current exact `p`: `p(1-p)^ell` for HORA-hit and
`p(1-p)^(4+ell)` for the fresh-group proxy.

## Exact configuration and deterministic semantics

- Environment: `SkillChainEnv` with three nested chains, 12 levels per chain,
  and 10 actions.
- Learning rate `.5`; task-sampler uniform floor `.1`; completion-normalized
  discounted pseudo-count decay `.9` per 16 observations; reference `N=16`;
  evaluation `k=8`; initial Beta pseudo-counts `(1,1)`.
- Locked runtime: CPython `3.9.6` and NumPy `1.26.4`.
- Environment seed is the logical seed; task-teacher seed is `seed+1000`.
  Task positions are sampled with replacement. Adaptive allocation is
  deterministic and consumes no RNG.
- If allocation marginals tie, choose the lowest batch position, exactly the
  behavior of NumPy's first-index `argmax`. A capped position is removed from
  later choices. `uncapped` means the mathematical maximum of 100 final
  rollouts for one position (`4+96`).
- Every registered cap is feasible because it is at least the average group
  size 16, so eight capped positions can always hold all 128 completions.
- For `history_plus_probe`, snapshot every task's discounted pseudo-counts
  once before the batch. Repeated occurrences of one task share that history
  snapshot but add only their own four probe outcomes; no within-batch
  evidence update occurs before allocation.
- Allocation point predictions are posterior means:
  `(1+c)/6` for `same_step`,
  `(alpha_before+c)/(alpha_before+beta_before+4)` for
  `history_plus_probe`, and exact pre-update `p` for `oracle_preupdate`.
- The oracle is read-only: it changes neither task sampling nor evidence
  state. Every cell starts from a fresh environment/teacher instance.
- Snapshot the behavior policy, exact pass probabilities, and teacher
  pseudo-count state before drawing any probes. All probes and Phase-B
  completions in the batch use that behavior snapshot; evidence is updated
  only after allocation and collection, followed by one synchronous update.
- Logical seeds pair cells, but differing rollout counts change later random
  consumption. They are paired initializations, not an assertion of exact
  common random numbers after control flow diverges.
- Engineering audit seeds are `90` and `91`; they are not part of the 16-seed
  result. The retained scientific cells use only logical seeds `0..15`.

## Required accounting checks

1. Every cell and checkpoint has the exact paid-completion budget.
2. Every batch has eight groups, at least four completions per group, exactly
   128 completions total, and respects its registered cap.
3. The fixed anchor always uses eight groups of size 16.
4. All probes and Phase-B completions are used in the final MaxRL group and
   are charged once.
5. All task-sampler evidence updates are completion-normalized and see exactly
   the sampled final groups; oracle probabilities never update the sampler.
6. A serial and four-worker run over a two-seed audit subset must be byte
   identical before the full matrix is retained.
7. A canonical JSON lock must bind this protocol, runner, analyzer, tests,
   estimator, teacher, and testbed source hashes; exact Python and NumPy
   versions; the complete matrix; seeds; and RNG/tie rules. Both runner and
   analyzer must independently verify it before retaining the full result.
8. The six cells overlapping the completed factorial---the two fixed anchors
   and, for each sampler, uncapped `same_step` HORA-hit and fresh-group-proxy
   cells---must reproduce its core checkpoint, allocation, outcome, and mass
   fields exactly before new contrasts are interpreted.

## Frozen readouts

Required for every cell:

- normalized pass@8 AUC versus paid completions;
- normalized mean-pass AUC and both terminal metrics;
- realized absolute coefficient mass per completion;
- dead, mixed, and all-pass group counts;
- maximum, nearest-rank 95th percentile, and Gini summaries of group sizes;
- mean absolute and squared probability error at allocation time against the
  simulator's exact pre-update task probability.
- marginal-score MAE against the oracle score over every eligible position at
  every Phase-B choice, plus chosen-oracle regret: the largest eligible oracle
  marginal minus the oracle marginal of the selected position.

Required paired contrasts, reported for every sampler and information source:

1. each cap minus uncapped within allocator;
2. fresh-group mass proxy minus HORA-hit within cap;
3. history-plus-probe minus same-step within cap and allocator;
4. oracle minus each deployable information source;
5. every adaptive cell minus the fixed anchor.

For group sizes `x_1,...,x_m`, nearest-rank P95 is sorted element
`ceil(.95*m)-1`, and Gini is
`sum_i sum_j |x_i-x_j| / (2*m^2*mean(x))`. Probability error gives every batch
position equal weight and compares the registered point prediction with the
exact pre-update task probability. Marginal-score errors give every eligible
position-decision pair equal weight; oracle regret gives every Phase-B
allocation decision equal weight. The exact Beta moment, not the posterior
mean plugged into a nonlinear score, is the registered deployable marginal.

Learning AUC is the trapezoidal integral over common checkpoints including
completion zero and 51,200, divided by 51,200. Terminal metrics are the exact
51,200-completion values.

For every paired contrast, report the equal-seed mean, sample SD with
`ddof=1`, Student-`t` 95% interval on the paired differences, and positive / zero
/ negative signs. The study is multiverse-style and descriptive: there are no
nominal contrast p-values and no cell can be promoted to confirmatory status.

## Pre-outcome engineering rule

For a later neural pilot only, a cap is *eligible* if, in the
`fresh_group_mass_proxy`
same-step cells averaged equally over the two samplers and 16 seeds, it:

1. reduces the paired mean per-run maximum group size by at least 25% relative
   to uncapped; and
2. has pass@8-AUC change no worse than `-0.005` relative to uncapped.

If several caps qualify, choose the smallest cap.  This is a prospective
engineering filter for a future experiment, not evidence of equivalence and
not a paper-level success criterion.  If none qualifies, retain uncapped and
report that the proposed cap calibration failed.

For this rule, first average the two sampler cells within each logical seed.
The maximum-size reduction is
`1 - mean_seed(cap_max)/mean_seed(uncapped_max)`; the AUC change is the
equal-seed mean of the within-seed cap-minus-uncapped difference. The filter
uses point estimates only and is not an equivalence test.

## Claim boundary

At most, this study can test cap/information sensitivity inside one synthetic
shared-skill testbed and choose a safer configuration for later work. It does
not include an exact conditional pooled-probe mass optimizer. It cannot
validate HORA, establish neural-RLVR performance, prove that coefficient mass
mediates learning, or add a fourth confirmed paper contribution.
