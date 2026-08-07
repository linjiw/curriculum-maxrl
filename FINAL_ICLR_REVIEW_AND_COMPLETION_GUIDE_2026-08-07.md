# Final ICLR Review and Research-Completion Guide

**Audit date:** 2026-08-07
**Target:** ICLR 2027
**Repository state reviewed:** commit `09d0d5d`
**Primary manuscript:** `paper/body.tex` and `paper/main_iclr.pdf`

## 1. Executive Verdict

The project has a defensible ICLR paper inside it, but the current manuscript is
not submission-ready.

The research goal is mostly achieved at the conceptual level:

1. There is an exact, useful estimator result:
   \(A_N(p)=2(\mathrm{pass@}N-\mathrm{pass@}1)\) for the deployed
   success-conditioned estimator.
2. There is an independently replicated neural experiment showing that the
   estimator changes time-integrated coverage under the same sampling
   interventions.
3. There is a replicated applied phenomenon on Countdown: exact-verifier
   recycling can increase mean accuracy while reducing pass@\(k\) coverage.
4. Negative results and failed preregistered predictions are unusually well
   documented.
5. The checked-in artifact currently passes its proposition, integration,
   manifest, figure, and preregistration checks.

The submission is blocked by presentation and claim calibration, not by lack of
work. The highest-risk issues are:

- the main text is about 17 pages before references, versus ICLR 2027's strict
  9-page initial-submission limit;
- the wave-2 easy-band \(10/12,\ p=.039\) claim counts correlated sampler
  contrasts as independent;
- the Countdown replay arm is higher-dose, not dose-matched;
- the useful gate operating point used buggy decay code, while the corrected
  strong-gate sweep failed;
- the sharpened GSM8K treatment cell completed but failed its preregistered
  treatment-delivery gate;
- the paper's artifact description says no figure values are hard-coded, but
  `fig9_passk.py` hard-codes a single seed's curves;
- the Countdown SFT warmstart overlaps 27/128 tier-0 evaluation tasks;
- the paper still uses the ICLR 2026 style and lacks the required ICLR 2027 AI
  use statement.

**Recommended submission position:** make the exact estimator result, the
balanced maze confirmation, and Countdown sharpening the three-part paper.
Treat the saturation gate as a provisional mitigation unless it is replicated
with corrected code. Move GSM8K to an appendix treatment-delivery study rather
than using it as evidence for an estimator-by-curriculum interaction.

## 2. The Paper's Defensible Research Claim

Use one central thesis:

> In RL with verifiable rewards, curricula and failure recycling cannot be
> evaluated independently of the advantage estimator. The estimator induces a
> compute-indexed coefficient-mass functional that predicts where sampled tasks
> can emit learning signal; controlled neural experiments confirm an
> estimator-conditioned coverage ordering, while LLM experiments show that
> recycling can trade pass@\(k\) coverage for mean accuracy.

This thesis is narrower than the current manuscript and better supported.

It does **not** require proving that coefficient mass determines neural learning
dynamics. It requires:

- an exact per-task estimator result;
- a controlled empirical test of an estimator-conditioned ordering;
- an applied demonstration that mean accuracy alone misses a material cost;
- honest boundaries where the proposed teacher or gate fails.

## 3. Claim Ledger

### 3.1 Claims ready for the main paper

| Claim | Verdict | Main-paper wording |
|---|---|---|
| MaxRL coefficient mass is \(2(\mathrm{pass@}N-\mathrm{pass@}1)\) under the stated binary i.i.d. assumptions | **Supported** | State as an exact proposition with assumptions |
| The deployed drop-\(K=0\) estimator targets truncation order \(T=N-1\), not \(T=N\) | **Supported** | Keep as a concise lemma/correction |
| The utility has dead, learnable, and mastered regimes whose location changes with \(N\) | **Supported algebraically** | Describe as coefficient-mass regimes, not guaranteed optimization regimes |
| Wave 2 confirms positive MaxRL-minus-GRPO time-integrated coverage under both samplers | **Supported** | Report 6/6 seed blocks separately under each sampler; do not pool sampler contrasts as independent |
| Countdown recycling raises tier-1 mean@16 and lowers pass@16 across three seeds | **Supported** | Keep as the applied headline result |
| The fixed-code strong-gate sweep failed | **Supported negative** | Keep briefly; it prevents a false monotone-dial claim |
| Jugs produced a preregistered null | **Supported negative** | Appendix/limitations; use to define boundary conditions |

### 3.2 Claims that must be narrowed

| Current claim | Problem | Required replacement |
|---|---|---|
| "24/24 paired blocks across both waves" | Two sampler observations share each seed/warmstart block | "Across 12 independent seed blocks, the sampler-averaged contrast was positive in all 12; wave 2 independently met its preregistered 6/6 criterion under each sampler." |
| "Easy-band concentration confirmed at 10/12, \(p=.039\)" | The 12 sampler contrasts are correlated within six seed blocks | "The easy-band pattern was directionally consistent in 10/12 sampler contrasts, but block-level aggregation gave four positive blocks, one exact tie, and one negative block; treat this as descriptive." |
| "Dose-matched replay control" | `ppo_epochs=2` doubles updates on all live groups, versus about 19% added relabel groups | "Higher-dose live-group replay control" or run a genuinely matched control |
| "One validated gate operating point" | The three-seed moderate point used faulty decay; corrected strong gating failed | "An under-gated operating point recovered frontier coverage; corrected-code validation of the useful setting remains open" |
| "LLM-scale interaction" | Two original seeds differ in trajectory; steering is weak; `g3p` fails the treatment gate | "A treatment-delivery study showing prompt-level posterior steering was not reliably delivered at this budget" |
| "No hard-coded figure values" | `fig9_passk.py` contains hard-coded arrays and has no manifest inputs | "Most figures regenerate from structured inputs; Fig. X uses transcribed single-seed telemetry" until fixed |
| "Both LLM pools carry zero train/eval overlap" | Countdown RL split may be clean, but SFT overlaps 21% of tier 0 | Explicitly distinguish RL-pool overlap from SFT/eval overlap |

### 3.3 Claims to remove from the main paper

- A general neural mechanism claim that GRPO's variance normalization causes the
  easy-band loss. The no-SD neural control failed that mechanism prediction.
- Any claim that prompt-level FrontierMax improves GSM8K at the present visit
  budget.
- Any implication that the strong gate defines a monotone mean-versus-coverage
  dial.
- Any inferential claim based on the single-seed pass@\(k\)-versus-\(k\) curve.
- The claim that recycling's mean gain has been causally decomposed at LLM
  scale. The present replay arm is an upper-bound control.

## 4. Statistical Corrections

### 4.1 Maze factorial: use the actual independent unit

The paper correctly states that the independent unit is an independently
trained seed block, but the easy-band inference violates that rule.

Each seed has two sampler contrasts:

- uniform: MaxRL minus GRPO;
- frontier sampler: MaxRL minus GRPO.

They share the seed/warmstart block and must not be treated as two independent
replicates.

For wave 2, sampler-averaging within each independent block gives:

| Seed | covAUC contrast | easy-band contrast |
|---|---:|---:|
| 6 | +0.02264 | +0.18750 |
| 7 | +0.01883 | 0.00000 |
| 8 | +0.01963 | +0.08333 |
| 9 | +0.01903 | +0.09375 |
| 10 | +0.00661 | -0.02083 |
| 11 | +0.03025 | +0.15625 |

The primary covAUC result remains strong:

- 6/6 positive block-level contrasts;
- mean \(+0.01950\);
- post-hoc 95% t interval \([+0.01148,+0.02752]\).

The secondary easy-band inference does not:

- four positive blocks, one exact tie, one negative;
- mean \(+0.08333\);
- post-hoc 95% t interval \([-0.00330,+0.16996]\).

Across both waves, the block-averaged covAUC contrast is positive in 12/12
independent blocks, with descriptive mean \(+0.02175\) and 95% t interval
\([+0.01663,+0.02688]\). Do not attach a pooled confirmatory p-value because
wave 1 was exploratory and wave 2 was the registered confirmation.

**Required reporting:**

1. Preserve the preregistered wave-2 conjunctive result: 6/6 under uniform and
   6/6 under frontier sampling.
2. Show one dot per seed block, with sampler contrasts connected or shown as
   within-block repeated observations.
3. Report a block-level interval.
4. Label the easy-band localization as descriptive.
5. Remove \(p=.039\), "10/12 confirmed," and any wording that calls 24
   observations independent blocks.

### 4.2 Countdown: distinguish training seeds, tasks, and samples

For each main endpoint:

- training seed is the independent unit;
- task-level samples estimate evaluation uncertainty within a trained seed;
- \(k\) values from the same 16 generations are repeated measurements;
- checkpoints from the same run are longitudinal measurements.

Use seed-level paired contrasts as the primary display. Task bootstrap intervals
may be added as evaluation precision, but must not replace across-seed
uncertainty.

### 4.3 Multiple claims

The current draft contains many preregistered and exploratory outcomes. In the
main paper, designate:

- **Primary theory claim:** exact coefficient mass.
- **Primary neural claim:** wave-2 time-integrated coverage contrast.
- **Primary applied claim:** Countdown tier-1 mean/pass@16 sharpening contrast.
- **Secondary:** timing, entropy, band localization, gate behavior.
- **Exploratory:** per-\(k\) crossing, GSM8K magnitude-dose pattern, heterogeneous
  maze cohort, single-checkpoint inference-currency examples.

This hierarchy should appear once in the experiment setup, not be reconstructed
from the appendix.

## 5. Latest GSM8K Result

The newly completed `g3p` arm must be incorporated before the next paper draft.

Its preregistered treatment-delivery gate required:

- minimum dead-sampled fraction \(<0.50\); and
- run-mean dead-sampled fraction \(<0.60\).

Observed from the completed 50-step log:

- minimum: 0.413;
- run mean: 0.601480;
- maximum: 0.719;
- final mean@4: 0.10547;
- final pass@4: 0.19834.

The arm passes the minimum criterion but fails the mean criterion by 0.00148.
Under the committed decision rule, this is **inconclusive by design**: the
intended steering treatment was not reliably delivered. The endpoint must not
be interpreted as evidence for or against the estimator-by-teacher interaction.

**Paper action:**

- remove GSM8K from the abstract, contributions, and main result figures;
- summarize it in limitations or an appendix as a treatment-delivery failure;
- update the stale analyzer/verdict artifact;
- finish or archive already-running cells for artifact completeness, but do not
  let them delay the main paper;
- do not launch another full cell until a short pilot passes the steering gate.

## 6. Experiment Plan

### Priority P0: required before submission

#### E1. Rebuild the factorial analysis at seed-block level

**Cost:** analysis only.
**Purpose:** correct the independent-unit error without changing the core claim.

Deliver:

- structured JSON with both sampler contrasts and their within-block average;
- seed-block plot;
- exact preregistered per-sampler signs;
- block-level mean and interval;
- revised verdict document.

**Stop rule:** complete when every number in the manuscript is generated from
the structured result file and no inferential count treats sampler contrasts as
independent.

#### E2. Run a genuinely dose-matched live-group replay control

**Cost:** three Countdown training runs, plus evaluation.
**Purpose:** determine whether the recycling mean gain is caused by extra
optimization dose or by the relabeled direction.

Match B2 on:

- generated rollout groups;
- accepted auxiliary groups per step;
- optimizer token/example presentations;
- optimizer steps or accumulated loss weight;
- learning-rate schedule and wall-clock checkpoint budget.

Do not use `ppo_epochs=2` as the match. Instead, sample exactly as many live
groups as B2 accepted relabel groups, with the same auxiliary loss coefficient
and token accounting.

Preregister:

- primary endpoint: tier-1 paired mean@16 contrast against B1;
- safety endpoint: tier-1 paired pass@16 contrast against B1;
- direction test: matched replay versus B2;
- three fixed seeds and one frozen final checkpoint.

Decision:

- If matched replay reproduces B2's mean gain without the coverage loss,
  recycling's gain is not direction-specific on this pool.
- If B2 beats matched replay at equal dose, the relabeled direction contributes
  measurable value.
- If both lose coverage, the damage is a generic auxiliary-dose effect rather
  than a hindsight-specific effect.

#### E3. Replace the single-seed pass@\(k\) curve

**Cost:** evaluation only if per-seed checkpoints remain available.
**Purpose:** make the "crossing in \(k\)" claim match its stated three-seed
evidence.

For B1, B2, and any gate arm retained:

- evaluate all three seeds on the same frozen task set;
- generate at least 16 samples per task;
- derive pass@1, 2, 4, 8, and 16 from structured raw outcomes;
- show seed-level curves or mean curves with across-seed uncertainty;
- check in the JSON and derivation script.

Keep the crossing as a main claim only if its sign is consistent across seeds.
Otherwise retain the robust endpoint statement: mean@16 up, pass@16 down.

#### E4. Resolve Countdown tier-0 contamination

**Cost:** analysis/evaluation only for the minimum repair.
**Purpose:** prevent a data-leakage objection.

Minimum acceptable repair:

- check SFT examples against every evaluation tier;
- evaluate tier 0 on the 101 non-overlapping tasks;
- mark the original 128-task tier-0 absolute values as contaminated;
- update `data_integrity_check.json`.

Stronger repair:

- regenerate SFT while excluding all evaluation keys;
- retrain the shared warmstart;
- rerun only experiments whose central claims depend on tier 0.

The current central tier-1 and tier-2 arm contrasts do not require a full
retraining because their measured overlap is zero.

### Priority P1: required only if the mitigation remains a contribution

#### E5. Corrected-code moderate-gate replication

**Cost:** three Countdown training runs.
**Purpose:** validate the useful operating point with the code the paper claims
to implement.

Do not rerun the already-failed strong setting. Calibrate the corrected decay
and threshold to reproduce the original moderate arm's cumulative admitted
relabel dose or rejection trajectory, then freeze the configuration before
training.

Preregister:

- exact implementation commit;
- expected admission/rejection range;
- tier-1 and frontier-tier mean and pass@16;
- comparison against B1 and B2;
- minimum useful criterion, such as recovering at least half of B2's coverage
  loss while retaining a positive mean gain over B1.

Decision:

- If it passes, the gate can remain the third empirical contribution.
- If it fails, remove the gate from the abstract and contributions; retain it as
  an appendix heuristic and negative result.

### Priority P2: artifact and mechanism quality

#### E6. Make relabel response rewriting semantically safe

The current word-boundary rewrite can alter decimals and abandoned-path
arithmetic, for example `12.5 -> 99.5` or `4 * 3 = 12 -> 4 * 3 = 99`.

Before release:

- rewrite only explicit goal-statement contexts;
- protect decimal and signed-number contexts;
- add fuzz tests over integer substrings, decimals, negatives, and intermediate
  equations;
- record skip/failure counts in the run artifact.

#### E7. One-target-per-group ablation

Run this only if recycling remains central after E2. It tests whether the
current per-row mixed-destination auxiliary batch is responsible for sharpening.
Match total accepted rows and loss weight. Compare mixed-destination versus one
destination per original dead group.

### Priority P3: scale-up, only after P0 is closed

A clean 1B-1.5B replication on one established mathematical reasoning benchmark
would improve reviewer confidence, but it is lower value than fixing the
independence, dose, and contamination issues.

Only run it if:

- the treatment can be shown to move sampling in a cheap pilot;
- at least three independent seeds or two models can be afforded;
- the experiment has one primary endpoint and a frozen stop rule;
- it will finish by 2026-08-28.

Do not spend the remaining schedule on a single 7B seed. It would add scale but
not statistical credibility.

## 7. GPU Queue Order

Use the available compute in this order:

1. Finish or safely archive the already-started GSM8K retry; update its status,
   but do not use it as a paper blocker.
2. E2: true dose-matched replay, three seeds.
3. E5: corrected moderate gate, three seeds, only if the gate stays in the
   contribution list.
4. E3 evaluations if they require GPU generation.
5. A scale-up replication only after all preceding verdicts are frozen.

Every new run needs:

- a committed preregistration;
- a unique run ID;
- exact code/config commit;
- one row in the run registry;
- structured final and trajectory outputs;
- automatic treatment-delivery checks;
- a predetermined manuscript branch for pass, fail, and inconclusive outcomes.

## 8. Nine-Page Paper Blueprint

The current chronological "escalating ladder" is too long and asks reviewers to
reconstruct the research history. Organize by argument instead.

### Title

Recommended:

> **Estimator-Conditioned Curricula and Failure Recycling in RL with Verifiable
> Rewards**

More assertive alternative:

> **The Estimator Decides: Coverage Trade-offs in Curricula and Failure
> Recycling for RLVR**

Use the first unless the final corrected controls strengthen the estimator
story. Avoid implying that the teacher itself consistently improves LLM
training.

### Abstract: 180-220 words

Use five moves:

1. Problem: curricula and failure recycling are usually evaluated as
   estimator-agnostic.
2. Exact result: coefficient mass equals
   \(2(\mathrm{pass@}N-\mathrm{pass@}1)\).
3. Controlled result: fresh balanced seed blocks confirm the
   estimator-conditioned time-integrated coverage ordering under both samplers.
4. Applied result: Countdown recycling raises mean accuracy but lowers
   pass@\(k\) across three seeds; state the replay/gate conclusion at its final
   calibrated strength.
5. Practice: report pass@\(k\) beside mean accuracy and condition intervention
   claims on the estimator.

Do not include:

- the full chronology of failed hypotheses;
- GSM8K interaction language;
- `24/24`;
- the invalid easy-band p-value;
- "dose-matched" unless E2 is complete;
- "validated gate" unless E5 passes.

### Main-text budget

| Section | Budget | Content |
|---|---:|---|
| Title + abstract | 0.5 page | One thesis, three results |
| 1. Introduction | 0.8 page | Problem, insight, three contributions |
| 2. Estimator-induced learnability | 1.4 pages | Assumptions, proposition, shape, truncation lemma, scope |
| 3. Curricula and recycling | 0.7 page | FrontierMax definition and verified auxiliary-update contract |
| 4. Experimental design | 0.5 page | Independent unit, metrics, primary endpoints, domains |
| 5. Estimator-conditioned coverage | 1.3 pages | Exact rung plus balanced maze confirmation |
| 6. Recycling-induced sharpening | 1.4 pages | Three-seed Countdown result, matched replay, optional validated gate |
| 7. Related work | 0.5 page | RLVR coverage, curricula, hindsight |
| 8. Limitations and conclusion | 0.7 page | Shared parameters, scale, treatment delivery, practical recommendation |
| Float/layout reserve | 1.2 pages | Three figures and one compact table |

Target 8.5 pages before the final formatting pass. A draft that is exactly 9.0
pages before copyediting will overflow.

### Three main figures

1. **Theory figure:** coefficient-mass curves and estimator comparison.
2. **Balanced factorial figure:** one dot per independent seed block, showing
   time-integrated coverage contrasts under both samplers.
3. **Countdown figure:** three-seed mean-versus-pass@16 sharpening; add the
   multi-seed per-\(k\) inset only if E3 confirms it.

Move to appendix:

- algorithm schematic;
- channel map;
- full ladder figure/table;
- heterogeneous maze band inventory;
- GSM8K trajectories;
- single-checkpoint sample-efficiency example;
- all gate trajectories and failed strong-gate points;
- Gym, IsaacLab, Jugs, and detailed negative-result chronology.

### Three contributions

1. **Theory:** an exact compute-indexed coefficient-mass functional and the
   \(T=N-1\) implementation correction.
2. **Controlled evidence:** an estimator-conditioned coverage ordering
   confirmed on fresh balanced seed blocks under two samplers.
3. **Applied evidence:** recycling-induced sharpening in RLVR, showing why
   pass@\(k\) must accompany mean accuracy; include the mitigation only if E5
   passes.

FrontierMax itself should not be a separate headline contribution unless a
clean experiment shows that the teacher improves a meaningful neural endpoint.

## 9. Section-Level Writing Instructions

### Introduction

- Open with the deployment decision: a fixed rollout budget must be allocated
  across tasks, but the estimator determines which sampled groups emit update
  mass.
- Introduce coefficient mass before naming all domains.
- State exactly three contributions.
- Put failed predictions in a compact "scope and falsification" paragraph, not
  throughout the introduction.
- Remove the long Q1/Q2/Q3 sequence and the detailed experiment itinerary.

### Theory

- Put all assumptions immediately before the proposition.
- Separate exact algebra from interpretation:
  - exact: expected absolute coefficient mass;
  - interpretation: a sampler-visible surrogate for potential signal;
  - not proved: monotone relation to neural improvement.
- Keep one proposition in the main text and move enumeration details and full
  proofs to the appendix.
- Use one normalization convention throughout.
- State the shared-parameter limitation before the experiments.

### Method

- Define the sampler in five to eight lines.
- Define recycling as a verified auxiliary update, not as an exact MaxRL
  objective.
- State the exactness and conditioning-rewrite contracts.
- Describe the frequency gate as a heuristic, not a pass-rate estimator.

### Experiments

- Start with a compact table containing model, task, rollout count, seeds,
  independent unit, and primary metric.
- Present confirmatory results before exploratory decompositions.
- Use "registered," "exploratory," and "descriptive" consistently.
- Report negative results once, then point to the appendix.
- Avoid comparing standard deviations from different sources as if they were
  confidence intervals.

### Related work

Organize into three short paragraphs:

1. curricula and difficulty sampling in RLVR;
2. estimator-induced weighting and pass@\(k\) coverage loss;
3. hindsight relabeling, self-imitation, and achieved-goal distribution shift.

The novelty claim should be:

> prior work studies these components largely in isolation; this paper derives
> and tests how the estimator conditions the effects of curricula and recycling,
> and measures recycling's pass@\(k\) cost in an RLVR loop.

### Limitations

State plainly:

- coefficient mass is not a convergence theorem;
- shared parameters can overturn per-task mechanism predictions;
- the largest model is 360M unless a scale-up lands;
- prompt-level posterior steering failed its treatment-delivery gate at the
  tested GSM8K budget;
- the useful gate point needs corrected-code replication unless E5 passes;
- Countdown is a structured exact-verifier environment, not representative of
  every reasoning task;
- the current relabel implementation is an auxiliary weighted-SFT update.

## 10. Exact Language Replacements

Replace:

> "24/24 paired blocks across both waves"

with:

> "The registered confirmation was positive in all six fresh seed blocks under
> each sampler. When the two sampler observations are averaged within each
> independent seed block, the contrast is positive in all 12 blocks across the
> exploratory and confirmation waves."

Replace:

> "easy-band concentration replicating at 10/12 pairs (\(p=.039\))"

with:

> "The easy-band localization was directionally consistent in 10/12
> within-block sampler contrasts, but its block-level interval includes zero;
> we therefore treat localization as descriptive."

Replace:

> "dose-matched replay control"

with:

> "higher-dose live-group replay control"

until E2 is complete.

Replace:

> "one validated operating point"

with:

> "one promising under-gated operating point whose corrected-code replication
> remains open"

until E5 passes.

Replace the GSM8K takeaway with:

> "At this prompt count and visit budget, posterior-based prompt steering did
> not reliably deliver a separated sampling treatment, so this experiment
> cannot decide the estimator-by-curriculum interaction."

## 11. Artifact Completion

### Required fixes

1. Move the `fig9_passk.py` arrays into a versioned JSON file.
2. Add a derivation script from raw per-task/per-seed outcomes.
3. Update the manifest inputs and checksums.
4. Remove the false "none are hard-coded" sentence.
5. Update the run registry with all final GSM8K and reviewer-control cells.
6. Regenerate `e_llm1b_verdicts.json` after the completed `g3p` arm.
7. Add SFT/evaluation overlap to `data_integrity_check.json`.
8. Freeze one final commit hash only after manuscript values and artifact values
   agree.
9. Sanitize absolute paths such as `/home/...`, `/tmp/...`, Ray session paths,
   and tool-specific working directories from the anonymous submission bundle.
10. Add a clean-machine artifact test, ideally in a fresh environment.

### Release test

The release candidate passes only if:

```bash
bash reproduce.sh --build
git diff --check
```

and:

- all figures reproduce from declared structured inputs;
- every quoted number has one source-of-truth artifact;
- no result depends on an uncommitted log in `/tmp`;
- the PDF has no undefined references, citations, or overfull boxes;
- the anonymous archive contains no author names, usernames, private URLs,
  machine paths, or repository history that reveals identity.

## 12. ICLR 2027 Compliance

Official sources checked on 2026-08-07:

- `https://iclr.cc/Conferences/2027/AuthorGuidelines`
- `https://iclr.cc/Conferences/2027/AIPolicyForAuthors`
- `https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip`

### Dates

- **Abstract deadline:** 2026-09-18 AOE.
- **Full paper and supplementary deadline:** 2026-09-25 AOE.
- No authors may be added or removed after the abstract deadline.

### Format

- Initial main text: at most 9 pages, strictly enforced.
- Rebuttal/camera-ready main text: at most 10 pages.
- References do not count.
- Appendices after references are unlimited, but reviewers need not read them.
- Use `iclr2027_conference.sty`; the current wrapper uses the 2026 style.
- References should be alphabetized; migrate the hand-maintained bibliography
  to the official BibTeX style or reorder it deterministically.
- Do not alter style dimensions or fonts to meet the limit.

### Mandatory and recommended statements

- Add the required **AI use statement** before references. It does not count
  toward the page limit and must accurately disclose assistance with any
  applicable research design, implementation, interpretation, writing,
  literature work, figures, and editing.
- Add the recommended paragraph-long **Reproducibility statement** before
  references. It does not count toward the page limit.
- Add an ethics statement only if the authors identify relevant concerns.

### Anonymity and administration

- Author identity in the main paper or supplement causes desk rejection.
- Audit paper text, metadata, PDF properties, code comments, artifact paths, and
  repository links.
- Ensure every author has a current OpenReview profile before submission.
- Confirm reciprocal-reviewer requirements and register at least one qualified
  author where required.
- Submit a genuine abstract; placeholders are deleted.

## 13. Schedule to Submission

### 2026-08-07 to 2026-08-09: freeze the claim set

- Accept the three-contribution structure.
- Commit corrected statistical analysis plan.
- Preregister E2 and, if retained, E5.
- Stop adding new domains.
- Update GSM8K status to treatment-delivery inconclusive.

### 2026-08-10 to 2026-08-16: analysis and artifact repairs

- Complete E1 block-level factorial analysis.
- Complete E3 multi-seed pass@\(k\) evaluation.
- Complete E4 clean tier-0 evaluation.
- Repair the figure input pipeline and manifest.
- Draft the new three-figure paper skeleton.

### 2026-08-17 to 2026-08-28: deciding experiments

- Run E2 true dose-matched replay.
- Run E5 corrected moderate gate only if it remains a contribution.
- Analyze each cell automatically as it completes.
- Execute preregistered pass/fail/inconclusive manuscript branches.
- Hard stop new training on 2026-08-28.

### 2026-08-29 to 2026-09-04: result lock

- Freeze every table and figure.
- Update claims appendix, run registry, checksums, and provenance.
- Remove unsupported branches from the paper.
- Produce the first 9-page ICLR 2027 build.

### 2026-09-05 to 2026-09-11: full rewrite

- Rewrite the abstract and introduction last, after the result sections are
  stable.
- Cut the chronological ladder from the main text.
- Complete related work, limitations, AI use, and reproducibility statements.
- Run an anonymity audit.

### 2026-09-12 to 2026-09-17: red-team review

Ask reviewers to answer only:

1. What is the single central claim?
2. Which result is confirmatory?
3. Are the independent units clear?
4. Does any wording outrun the evidence?
5. Can the paper be understood without the appendix?
6. Is every main figure necessary?

Resolve all P0 findings before submitting the abstract.

### 2026-09-18: abstract submission

- Submit the genuine final-direction abstract.
- Confirm final author list and OpenReview profiles.
- Do not plan any experiment that could change the paper's identity after this
  date.

### 2026-09-19 to 2026-09-24: release candidate

- Build from a clean checkout.
- Run the complete artifact test.
- Verify 9-page limit with the official style.
- Inspect every PDF page visually.
- Upload a draft at least 24 hours before the deadline.

### 2026-09-25: submit

Submit at least 12 hours before the AOE deadline. Download the uploaded PDF and
supplement from OpenReview and verify them rather than trusting the local copy.

## 14. Definition of Done

The research phase is complete when:

- the main claim no longer depends on GSM8K;
- the maze analysis uses six independent blocks in wave 2, not 12 correlated
  sampler contrasts;
- the replay result is either genuinely dose-matched or accurately labeled;
- the gate is either corrected-code replicated or demoted;
- the pass@\(k\) curve is multi-seed and artifact-derived, or removed;
- tier-0 contamination is disclosed and excluded from absolute claims;
- every central number is generated from a structured committed artifact.

The paper is complete when:

- one sentence states the thesis;
- there are exactly three contributions;
- the main text is at most 9 pages in the ICLR 2027 style;
- the abstract contains only claims supported by independent replication;
- the main paper has no more than three figures and one compact setup table;
- theory assumptions and empirical independent units are explicit;
- negative results bound the claims without dominating the narrative;
- AI use, reproducibility, anonymity, and author-administration requirements are
  satisfied;
- `bash reproduce.sh --build` passes from the frozen release commit.

## 15. Bottom Line

Do not add breadth. Finish the causal accounting and simplify the story.

The strongest paper is not "a universal curriculum plus recycler plus gate that
works everywhere." The evidence supports a more useful conclusion:

> the estimator determines which difficulty interventions can emit and preserve
> learning signal, and mean accuracy alone can hide a coverage cost.

That is a coherent research result, it survives the failed predictions, and it
can fit a complete ICLR paper once the statistical unit, replay dose, gate
status, and manuscript length are corrected.
