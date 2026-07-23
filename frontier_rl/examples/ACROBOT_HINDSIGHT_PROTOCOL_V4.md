# Neural Acrobot hindsight protocol — V4 optimizer-matched scale study

Status: frozen on 2026-07-22 while the V3 shared-policy efficacy confirmation
was still in progress and before inspecting any V3 arm outcome or executing
the fresh V4 seed blocks. The corrections below were made from a read-only
design audit, still before creating the V4A source lock or observing V4 data.

This protocol is independent of V3's efficacy decision. It studies the
optional hindsight component under the frontier-`u_N` teacher; it does not
turn verifier-valid relabeling into a statistically exact gradient claim.

## 1. Question and motivation

The retained skill-chain sensitivity sweep improved through hindsight scale
eight, but multiplying the relabeled weights also multiplies the auxiliary SGD
step. That curve cannot distinguish useful recycled data from an effective
learning-rate increase.

V4 asks:

> At a matched number of nonzero optimizer updates, how do hindsight scale and
> base learning rate interact for the shared neural Acrobot policy?

The environment, eight thresholds, strict post-state verifier, shared H64
actor, frozen-group SUM score estimator, `N=16`, frontier teacher
(`u_16`, `gamma=1`, decay `0.7`, floor `0.1`), first-hit positive-prefix
rewrite, and plain SGD are inherited from the locked Acrobot implementation.

The relabeler is described as a **verifier-valid centered auxiliary update**.
It conditions on source-task failure, selects the hardest lower threshold with
`0<K'<N`, and therefore cannot have the full law of a nondegenerate fresh
target group. V4 makes no unbiased-hindsight claim.

## 2. Stage A: hindsight-effect-blind feasibility

Stage A is effect-blind with respect to the Stage B hindsight estimands: it
contains no positive-scale arm, and neither evaluation performance, a
hindsight contrast, nor the V3 outcome enters its deterministic launch rule.
Its feasibility diagnostics are nevertheless post-learning-rate quantities
and are not claimed to be treatment-invariant.

Use fresh development seeds `13000..13002`. Run only scale-zero cells at
learning-rate multipliers `[0.5,1,2]` around base rate `3e-4`:

```text
actor:             shared H64
sampling:          frontier-u_16 teacher, gamma=1
hindsight scale:   0
learning rates:    1.5e-4, 3e-4, 6e-4
target:            400 nonzero SGD updates
group-start cap:   4,000,000 actual environment transitions
evaluation:        initialization and every 50 nonzero updates
episodes:          32 per threshold
```

Scale zero must still construct every eligible hindsight relabel and compute a
nonmutating unscaled auxiliary-gradient preview. It must never apply or count a
hindsight update. Requested mixed-group updates proceed normally.

No new group begins once cumulative transitions are at least `4,000,000`.
Because every started group is completed, the final coordinate may exceed
this group-start cap by at most `16 x 500 = 8,000` transitions.

Define the selected update budget before evaluating any launch gate:

```text
U* = 400,  if all nine runs reach 400 updates;
     250,  otherwise if all nine runs reach at least 250 updates;
     STOP, otherwise.
```

For run `j`, let `tau_j(U*)` be the first completed group after which its
nonzero-update count equals `U*`. Every gate below is computed only from the
prefix through `tau_j(U*)`; saved records after that prefix are retained but
excluded. The runner must record cumulative wall seconds at updates 250 and
400 so the fallback does not use a terminal-time approximation.

The factorial is authorized only if all hindsight-effect-blind gates pass:

1. The artifact exactly matches the V4A source/runtime lock and every run
   passes finite-value, binary-verifier, parameter-count, transition/group/task/
   update accounting, and evaluation-state invariance checks.
2. `U*` is selected by the rule above; every selected prefix ends at exactly
   `U*`, with no missing or invalid run.
3. Every run has at least ten eligible relabel candidates, and every candidate
   produces exactly one finite, strictly positive, nonmutating unscaled
   auxiliary-gradient preview. In every scale-zero prefix,
   `preview_count = relabel_candidate_count`, `relabeled_updates = 0`, and
   `optimizer_updates = requested_live_updates`. A deterministic test must
   additionally show that preview-only scale zero and a no-hindsight shadow
   produce identical training trajectories and final state.
4. Every cell observes all-fail, mixed, and all-pass requested groups within
   the selected prefix.
5. For each run `j`, let `G_j` contain selected-prefix groups whose
   `transition_start >= 200000`. The set must be nonempty and

   ```text
   mean_TV_j = mean_{g in G_j} TV(q_g, Uniform) > 0.05.
   ```

6. Freeze the serial runtime projection as

   ```text
   projected_hours_90 = 90 * max_j(wall_seconds_through_tau_j(U*)) / 3600.
   ```

   It must be no more than 24 hours on the development Mac.

No learning score, hindsight contrast, or V3 outcome enters these gates. The
complete feasibility artifact is retained even if a gate fails.

## 3. Stage B: frozen `3 x 3` factorial

Only after Stage A passes, write a V4B amendment and source lock that record the
selected update budget (`400` or the single fallback `250`), hashes, exact
commands, and the gate result before touching fresh confirmatory seeds
`14000..14009`. The amendment may not change the cells, metrics, contrasts,
thresholds, or claim rules below after inspecting Stage A.

Before the amendment is written, the independent V4 verifier must reproduce
all Stage A gates from the immutable feasibility artifact. The amendment and
V4B lock must record the verification-report hash; the Stage B runner must
refuse to start without that exact passing report.

Run all nine cells:

```text
learning-rate multiplier:  0.5, 1, 2
hindsight scale:           0, 1, 2
sampling/actor:             frontier-u_16 teacher / shared H64
matching axis:              nonzero SGD updates
transition group-start cap: 4,000,000
```

One requested mixed-group update and one accepted hindsight update each count
as one nonzero optimizer update. Scale-zero relabel previews do not count.
Every group is completed under the same at-most-8,000-transition overshoot
rule. A capped run that misses the frozen update target is retained as censored
and invalidates the entire paired primary analysis; it is never replaced. The
primary analysis additionally requires every final update coordinate to equal
`U*` exactly.

## 4. Primary metric and contrast family

The primary metric is normalized target-uniform mean-pass AUC over optimizer
update count, including update zero. It measures learning per nonzero SGD
update, not per environment transition, wall-clock second, or generated token.
Transition-indexed AUC, transitions required to reach the update target,
native return/success, relabel rate, update source, and gradient diagnostics
are secondary descriptive outcomes. Transition-indexed results are not in an
inferential family and therefore cannot rescue or independently establish a
failed update-indexed claim.

On all ten complete paired seeds, test exactly this four-contrast family:

```text
C1 = AUC(lr=1,   scale=1) - AUC(lr=1,   scale=0)
C2 = AUC(lr=1,   scale=2) - AUC(lr=1,   scale=1)
C3 = [AUC(lr=.5, scale=2) - AUC(lr=.5, scale=0)]
     - [AUC(lr=1, scale=1) - AUC(lr=1, scale=0)]
C4 = [AUC(lr=1, scale=2) - AUC(lr=1, scale=0)]
     - [AUC(lr=2, scale=1) - AUC(lr=2, scale=0)]
```

Use exact two-sided paired sign-flip tests and Holm step-down control at
familywise `alpha=0.05`; report 20,000-resample paired-seed bootstrap intervals.
C1/C2 support a directional local improvement only when their mean is at
least `+0.03` normalized update-AUC units and their Holm-adjusted test rejects.
C3/C4 support a material departure from the restricted model below only when
their absolute mean is at least `0.03` and their Holm-adjusted test rejects.
The randomization interpretation assumes independent seed-level contrasts
whose null distributions are sign-exchangeable; exact enumeration of all
`2^10` sign patterns does not make that assumption automatic.

C1 tests whether one unit of auxiliary scale helps at the base rate after
matching total applied-update count. C2 tests the incremental second unit; it
does not test scale two directly against scale zero. C3 and C4 are symmetric
iso-auxiliary-coefficient diagnostics. If `Y(a,s)` denotes update-indexed AUC
at learning-rate multiplier `a` and hindsight scale `s`, both equal zero under
the restrictive separable model

```text
Y(a,s) = F(a) + G(a*s).
```

A material significant C3 or C4 rejects the hypothesis that the auxiliary
effect depends only on `a*s` and adds independently to the requested-update
learning-rate effect over the tested cells. It does not by itself identify
semantic data value: update-source composition, gradient norms, relabel
frequency, and policy trajectories co-evolve. Failure to reject is not
equivalence.

For diagnosis, record cumulative applied-step norm and squared norm separately
for hindsight and requested updates:

```text
M_source = sum_u ||learning_rate_u * scale_u * gradient_u||
Q_source = sum_u ||learning_rate_u * scale_u * gradient_u||^2,
```

where `scale_u=1` for requested updates. These are descriptive possible
mediators, not additional causal tests.

## 5. Claim boundary

V4 can establish optimizer-update-matched local algorithm effects of a
verifier-valid auxiliary update on this fixed Acrobot threshold family. Equal
total nonzero-update count does not hold requested-update count, auxiliary-
update count, cumulative update norm, parameter displacement, or generated
transitions fixed. V4 therefore cannot establish pure semantic data value,
sample efficiency, wall-clock efficiency, statistical exactness of hindsight,
transfer causality, a universal optimal scale, or generalization to language
models. No interim analysis, seed replacement, outcome-dependent grid change,
or pooling with development/V3 data is allowed.
