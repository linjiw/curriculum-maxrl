# Curriculum-MaxRL: Independent Research Review Handoff

**Review target:** ICLR-style research paper  
**Snapshot date:** 2026-08-11  
**Compact paper source:** `origin/codex/curriculum-maxrl-research@9277141:paper/body_iclr.tex`  
**Latest causal-accounting work:** `autoresearch/iterate-260810-2240/`

## What the paper now claims

The paper asks a deliberately narrow question: **what task-level activity does
a deployed group estimator make available to curricula and failure
recycling?** For the zero-stabilizer practical MaxRL convention with $N$
i.i.d. binary rollouts, the expected absolute coefficient mass is

\[
A_N(p)=2\{1-(1-p)^N-p\}=2(\mathrm{pass@}N-\mathrm{pass@}1).
\]

This identity yields a rollout-aware activity score with exact zeros at
$p=0$ and $p=1$, a unique interior peak, and the common $p(1-p)$ score as
its $N=2$ case. It is an estimator-side activity diagnostic, not a theorem
that maximizing coefficient mass maximizes learning progress.

The paper's practical thesis is correspondingly restrained:

> A curriculum or recycler must be evaluated together with the estimator it
> modifies, and mean accuracy should be reported beside a coverage metric.

## Proposed compact abstract

RL with verifiable rewards often treats task curricula and failure recycling
as estimator-agnostic data interventions. We ask what task-level activity a
deployed group estimator makes available to either intervention. For the
zero-stabilizer practical MaxRL convention with $N$ i.i.d. binary rollouts,
the expected absolute coefficient mass is exactly
$2(\mathrm{pass@}N-\mathrm{pass@}1)$. The resulting activity score has a
unique compute-dependent peak, recovers $p(1-p)$ at $N=2$, and shows that
dropping all-fail groups targets truncation order $N-1$, not $N$. This is an
estimator-side diagnostic, not a theorem of learning progress. In a fresh
20-seed Acrobot tournament at $N=16$, the rollout-aware score improves
target-uniform AUC over $p(1-p)$ by $+.0480$ (95% paired-bootstrap CI
$[+.0209,+.0738]$, exact sign-flip $p=.0034$). A fresh 24-block
exact-probability Digits counter-test rejects the stronger universal mapping:
the registered estimator-by-sampler interaction is unsupported ($p=.350$),
RLOO reverses its predicted ordering, and both matched samplers lose to
uniform. A qualified, externally recorded six-block maze wave further reports
higher time-integrated MaxRL than GRPO coverage under both samplers at common
optimizer settings. Together, the positive and negative results support a
calibrated conclusion: coefficient activity is a useful source of curriculum
hypotheses, not a universal curriculum objective. Data-selection interventions
should be evaluated with the estimator beneath them and with raw mean@$k$ and
pass@$k$ outcomes retained.

## Claim hierarchy

### 1. Formal claims — ready

- Exact coefficient-mass identity under the stated estimator and rollout
  assumptions.
- Exact factorization of the expected update into $u_N(p)$ times the
  success--failure conditional score contrast.
- The practical drop-all-fail convention targets truncation order $N-1$,
  not $N$.
- The $N$-dependent peak and frontier asymptotics.

These claims have analytic derivations plus exact-enumeration/Monte-Carlo
checks. The paper now distinguishes the zero-stabilizer idealization from the
released finite-denominator stabilizer.

### 2. Controlled empirical claims — supported but scoped

- **Acrobot:** in one fixed task pool and common MaxRL scaffold, $u_{16}$
  beats $u_2=p(1-p)$ by $+.0480$, paired bootstrap 95% CI
  $[+.0209,+.0738]$, exact sign-flip $p=.0034$, across 20 paired seeds.
  This is a score-shape result, not superiority over a complete named method.
- **Maze:** an external execution record reports that the time-integrated
  MaxRL--GRPO coverage contrast was fixed before the six-block fresh wave. It
  is positive in 6/6 blocks under each sampler; sampler-averaged wave-2 mean
  $+.0195$, post-hoc 95% interval $[+.0115,+.0275]$. The locking object and
  checkpoint trajectories are not included in the compact release.
- **Fixed-completion CPU sweep:** $u_N$ beats $u_2$ in all eight paired
  seeds for every tested $N>2$, but the sweep is descriptive and does not
  establish a general scaling law.

### 3. Counterevidence that defines the scope — important, not incidental

- **Digits exact-probability factorial:** the frozen estimator-by-sampler
  interaction is unsupported ($p=.350$); RLOO reverses its predicted
  sampler preference, and both matched samplers lose to uniform. Coefficient
  mass is therefore not a universal curriculum objective.
- **Paid-probe Acrobot:** $u_{16}$ does not clear the registered comparison
  with source-faithful ProCuRL selection; probes consume about 93.2% of the
  paid budget, making this a probe-cost diagnosis rather than a general method
  comparison.
- The original maze endpoint claim failed and was retracted. Easy-band
  localization is suggestive after using the seed block as the independent
  unit.
- Removing GRPO's SD normalization did not eliminate the neural easy-band
  loss, so variance normalization is not established as the maze mechanism.
- The corrected saturation gate failed; GSM8K failed treatment delivery; Jugs
  produced the preregistered all-null boundary.

### 4. Applied Countdown observation — reportable, not yet causal

The surviving three-seed aggregate is:

- mean@16: $0.278 \rightarrow 0.324$;
- logged VERL bootstrap best@16 proxy: $0.541 \rightarrow 0.492$.

The second quantity is **not standard unbiased pass@16**. Complete per-task
outcomes and paired seed records are missing, so the current record cannot
support standard pass@16, paired seed signs, timing claims, or a
relabel-specific causal mechanism. A higher-dose replay arm improves both
logged metrics but does not match dose or isolate relabel direction. Two
prospective matched controls failed treatment delivery.

E2c is the frozen prospective repair: three seeds, immutable train-only source
reservoir, fixed-slot and response-token dose matching, delivery gates before
endpoint exposure, nine paired endpoint evaluations, and standard pass@16
recomputed from retained binary task outcomes. Its code and tests are complete;
execution is waiting for the shared GPU to fall below the frozen memory
ceiling.

Even a valid E2c result does not isolate relabel direction by itself. B2 uses
current-policy failures relabeled to achieved goals, while E2c draws informative
groups from a frozen-SFT reservoir. Source age, behavior distribution, task
selection, and on/off-policy character differ. The clean interpretation is
therefore “achieved-target relabel package versus frozen informative-replay
package at matched delivered dose,” with only three training-seed replicates.

## Non-negotiable terminology

Please flag any manuscript sentence that violates these boundaries:

- Say **coefficient activity/mass**, not “expected learning signal” without a
  scope qualification.
- Say **logged VERL bootstrap best@16 coverage proxy**, not pass@16, for the
  historical Countdown result.
- Say **reported three-seed aggregate**, not replicated per-seed direction.
- Say **higher-dose replay control**, not dose-matched control, for ARM B.
- Say the gate **did not validate under corrected code**; do not call the
  faulty-decay setting a validated operating point.
- Say the GSM8K interaction is **inconclusive by treatment-delivery rule**.
- Treat sampler observations within one seed/warm-start block as repeated
  measurements, not independent replicates.
- Treat internal hashes without an immutable public pre-execution record as
  mechanical provenance, not independently auditable preregistration.

## Highest-value questions for an independent reviewer

1. Is the coefficient-mass identity sufficiently novel and useful once it is
   explicitly scoped as activity rather than optimization progress?
2. Does the exact expected-update factorization provide an adequate bridge
   from the scalar identity to the empirical hypotheses, or is a stronger
   theorem/bound required?
3. Is Acrobot plus the Digits counter-test a convincing positive--negative
   pair, or does the shared small-policy setting make both too synthetic?
4. Can the externally recorded maze wave remain a central result when the lock
   object and checkpoint trajectories are absent from the release?
5. Is the historical Countdown aggregate worth main-text space before E2c,
   given that its coverage quantity is a bootstrap proxy and raw outcomes are
   missing?
6. Does the paper clearly separate failure recycling as a verified selected
   auxiliary update from an unbiased on-policy gradient?
7. Which single claim would a skeptical reviewer identify as overstated?
8. Is the compact paper's narrative coherent enough to survive removal of any
   one empirical family?

## Suggested review order

1. Compact submission draft:
   `git show origin/codex/curriculum-maxrl-research:paper/body_iclr.tex`
2. Latest project status: `autoresearch/iterate-260810-2240/STATUS.md`
3. Frozen research objective: `autoresearch/iterate-260810-2240/GOAL.md`
4. E2c preregistration: `autoresearch/iterate-260810-2240/E2C_PREREG.md`
5. Independent-unit repair:
   `curriculum_maxrl/maze_gpu_factorial/block_reanalysis.json`
6. Current claim ledger: `EVIDENCE.md`
7. Compact release accounting:
   `git show origin/codex/curriculum-maxrl-research:ANONYMOUS_RELEASE.md`

## Requested review output

Please return:

1. a one-paragraph accept/reject-style summary;
2. the three strongest contributions;
3. the three most serious validity or novelty concerns;
4. every sentence-level overclaim you find;
5. a recommended main-text claim hierarchy;
6. whether E2c is submission-critical or only an upgrade;
7. concrete replacement language for the abstract and conclusion where needed.
