# Math, Results, and Writing Review

Review target: `paper/main_iclr.tex` at commit `bc67108` on 2026-08-04.

Canonical PDF: `paper/main_iclr.pdf`. The workspace-level
`../paper-draft.pdf` is older and has a different hash; do not circulate it
as the current paper.

This review checks the manuscript against the committed code, JSON/JSONL
artifacts, figure scripts, and the sibling `../maxrl` run artifacts. It does
not independently verify literature-priority claims or bibliography metadata.

## Executive verdict

The central coefficient-mass identity is correct and useful:

\[
\mathbb E\!\left[\sum_i |a_i|\right]
=2\left(1-(1-p)^N-p\right).
\]

The practical-estimator truncation result is also correct under the stated
i.i.d. binary-reward model. Those are the strongest mathematical parts of the
paper.

The current draft is not yet mathematically or statistically safe to submit.
The main blockers are:

1. `u_N` is defined both with and without the factor of two.
2. The displayed GRPO mass is not the exact mass of the GRPO implementation
   used in the experiments.
3. The hindsight proposition overstates an off-policy, adaptively selected
   auxiliary update as an ML gradient.
4. "Dead zone," "band width," and "only channel" are asserted without
   operational definitions and beyond what the identity proves.
5. The maze `p=0.0079` analysis treats heterogeneous, correlated runs as
   exchangeable independent observations.
6. Figure 8's `p=0.0001` is pseudoreplicated: the script pools 18 heterogeneous
   MaxRL runs, mostly sharing seeds/warmstarts, against four GRPO runs.
7. The Countdown results do not establish a monotone gate-strength frontier.
8. GSM8K is a two-seed pilot with weak treatment delivery, a replacement
   confound, and evaluation-noise bars that are not training uncertainty.
9. The 85% / 272-step timing statement does not reproduce from the committed
   timing artifact.
10. Several figures hard-code paper numbers, and the current repo omits the
    committed Jugs result artifacts that exist in `../maxrl`.

The right paper is still available after these corrections: an exact
coefficient-mass analysis, an exploratory but consistent estimator-associated
maze pattern, and a three-seed demonstration of recycling-induced sharpening
with one useful gated operating point.

## Claim grades

| Claim | Grade now | Required action |
|---|---|---|
| Practical MaxRL estimator targets truncation `T=N-1` | **Proved, with assumptions** | Define the objective and show the score-function expectation |
| MaxRL coefficient mass is `2(pass@N-pass@1)` | **Proved** | Use one factor convention everywhere |
| Peak is `1-N^{-1/(N-1)}` | **Proved** | Call it a peak, not an undefined band center/width |
| RLOO half-mass is `p(1-p)` | **Proved for the stated normalization** | State the normalization |
| Current GRPO curve and tail ratios | **Wrong for deployed code** | Re-derive using sample SD or change code and rerun |
| Greedy variable-`N_i` water-filling | **Proved only for a narrow static problem** | Add constraints; disconnect it from the sampling oracle |
| Hindsight gives an ML gradient | **Not proved; generally false as written** | Replace with an explicit shifted/off-policy update characterization |
| Maze estimator main effect, `p=0.0079` | **Descriptive/suggestive** | Reanalyze by independent seed blocks; complete a balanced design |
| Maze band mechanism, `p=0.0001` | **Invalid inference** | Remove p-value; use a controlled arm and seed-level analysis |
| Recycling-induced sharpening | **Supported at tier 1, three seeds** | Report raw paired seed deltas and uncertainty |
| Gate restores frontier coverage while retaining mean | **Supported as one operating point** | Avoid equivalence/monotonicity language without a sweep |
| Gate creates a monotone dial | **Not established** | Run a fixed-decay, multi-seed dose sweep |
| GSM8K estimator-by-teacher interaction | **Pilot only** | Remove from abstract headline; finish treatment-controlled replication |
| Jugs boundary result | **Result exists, artifact missing here** | Sync preregistration, per-cell results, and postmortem |

## 1. Correct mathematical core

### 1.1 Fix the notation first

Let \(R_i\in\{0,1\}\), \(K=\sum_{i=1}^N R_i\), and let the effective
per-rollout coefficients of the practical estimator be

\[
a_i =
\mathbf 1\{K>0\}\left(\frac{R_i}{K}-\frac1N\right).
\]

Use two separate symbols:

\[
A_N(p):=\mathbb E\!\left[\sum_i |a_i|\right],\qquad
u_N(p):=\frac{A_N(p)}2=1-(1-p)^N-p.
\]

Then write:

\[
\boxed{A_N(p)=2u_N(p)=2(\operatorname{pass@}N-\operatorname{pass@}1).}
\]

This resolves the contradiction between:

- `paper/main_iclr.tex:238-242`, where `u_N` includes the factor two;
- `paper/main_iclr.tex:190`, Algorithm 1, the code, and Figure 1, where
  `u_N` omits it.

The factor does not affect normalized sampling proportional to
\(u_N^\gamma\), but it does affect every statement claiming an exact mass.

### 1.2 Recommended proof

For \(K\ge1\),

\[
\sum_i |a_i|
=K\left(\frac1K-\frac1N\right)+(N-K)\frac1N
=2\left(1-\frac KN\right).
\]

Therefore,

\[
\begin{aligned}
A_N(p)
&=2\mathbb E\left[\left(1-\frac KN\right)\mathbf1\{K\ge1\}\right]\\
&=2\left(\Pr[K\ge1]-\frac{\mathbb E[K]}N\right)\\
&=2\left(1-(1-p)^N-p\right).
\end{aligned}
\]

Call this **expected absolute coefficient mass**, not "the exact learning
signal." It omits score-vector geometry, sequence length, token masking,
optimizer state, PPO clipping, and covariance between coefficients and score
vectors. Remark 1 acknowledges this, but the abstract and figure captions
currently revert to the stronger wording.

### 1.3 Make the truncation lemma self-contained

Define the truncated objective, for example:

\[
J_T(p)=-\sum_{t=1}^{T}\frac{(1-p)^t}{t},\qquad
\nabla J_T(p)=
\left(\sum_{j=0}^{T-1}(1-p)^j\right)\nabla p.
\]

For the practical drop-\(K=0\) estimator,

\[
\mathbb E[\widehat G_N]
=\left(
\frac{1-(1-p)^N}{p}-(1-p)^{N-1}
\right)\nabla p
=\left(\sum_{j=0}^{N-2}(1-p)^j\right)\nabla p.
\]

Hence it targets \(J_{N-1}\), for \(N\ge2\). The current lemma has the right
answer but does not define \(J_T\), the score estimator, or the assumptions
needed to obtain the expectation.

An exact relation worth adding is

\[
u_N(p)=p(1-p)\,w_{N-1}(p),\qquad
w_T(p)=\frac{1-(1-p)^T}{p}.
\]

This cleanly connects coefficient mass to the practical estimator's
weight-function view without claiming they are the same object.

### 1.4 Peak is correct; "band width" is not defined

For \(N\ge2\),

\[
u_N'(p)=N(1-p)^{N-1}-1,\qquad
u_N''(p)=-N(N-1)(1-p)^{N-2}<0,
\]

so

\[
p^\star=1-N^{-1/(N-1)}
=\frac{\log N}{N-1}
+O\!\left(\frac{(\log N)^2}{N^2}\right).
\]

This proves a unique peak. It does not by itself define:

- a dead-zone boundary;
- a mastered-tail boundary;
- a band width;
- three discrete regions.

The only exact zeros are \(p=0\) and \(p=1\). If the paper wants regions,
define them operationally. One option is:

\[
\Pr(0<K<N)=1-(1-p)^N-p^N,
\]

and call a prompt operationally dead/mastered at budget \(N\) only after
choosing a threshold on all-fail/all-pass probability. Another option is an
\(\eta\)-utility band
\(\{p:u_N(p)\ge \eta u_N(p^\star)\}\). State \(\eta\) and report its
boundaries. Otherwise remove "width" and replace "partition" with
"continuous low-, intermediate-, and high-pass-rate regimes."

Also scope "no sampler can reach" to \(p=0\) under the current policy and
support. For \(0<p\ll1\), more groups or a larger \(N\) can obtain a success.

### 1.5 RLOO is correct under the paper's normalization

With

\[
a_i^{\mathrm{RLOO}}
=\frac{R_i-\bar R_{-i}}{N},
\]

\[
\mathbb E\sum_i|a_i^{\mathrm{RLOO}}|
=2p(1-p).
\]

Thus its half-mass is \(p(1-p)=u_2(p)\). Keep "the \(N=2\) slice," but
always say half-mass utility when using \(u_N\).

### 1.6 GRPO must match the implementation

The paper and `fig1_utility.py` use the population-SD result

\[
\frac12 A_N^{\mathrm{GRPO}}(p)
=\frac1N\mathbb E\sqrt{K(N-K)}.
\]

However, both:

- `curriculum_maxrl/estimators.py:28-31`; and
- `../maxrl/verl/trainer/ppo/core_algos.py:270`

use sample SD (`ddof=1` / PyTorch's default unbiased SD).

For sample SD,

\[
s_K=\sqrt{\frac{K(N-K)}{N(N-1)}},
\]

and the exact half-mass is

\[
\frac12 A_{N,\mathrm{sample}}^{\mathrm{GRPO}}(p)
=\sqrt{\frac{N-1}{N}}\,
\frac1N\mathbb E\sqrt{K(N-K)}.
\]

Consequently, for the deployed implementation,

\[
\frac{A_N^{\mathrm{MaxRL}}}{A_N^{\mathrm{GRPO}}}
\to\sqrt N\quad(p\to0),
\]

while

\[
\frac{A_N^{\mathrm{GRPO}}}{A_N^{\mathrm{MaxRL}}}
\to\frac{N-1}{\sqrt N}\quad(p\to1).
\]

The symmetric \(\sqrt{N-1}\) tail ratios in the paper are correct only for
population SD. Choose one convention, state it, make code and figure agree,
and include the epsilon convention for degenerate groups.

### 1.7 Narrow the allocation proposition

The greedy result is valid for the following static problem:

\[
\max_{\{N_i\}}\sum_i u_{N_i}(p_i)
\quad\text{s.t.}\quad
\sum_iN_i=B,\quad N_{\min}\le N_i\le N_{\max},
\]

with fixed \(p_i\), one group per prompt, and integer \(N_i\). The marginal is

\[
u_{N+1}(p)-u_N(p)=p(1-p)^N.
\]

It does **not** prove that sampling prompts proportional to
\(u_N(p)^\gamma\) is optimal. At fixed group size, expected immediate mass is
linear in the prompt-sampling distribution, so the unconstrained myopic
optimum puts all probability on an argmax. The deployed sampler deliberately
does something else for exploration and batching.

It also does not describe the "oracle" in Section 6.1. That oracle samples
prompts proportional to true-\(p\) utility; it does not vary \(N_i\) and does
not perform water-filling. Rename it **true-pass-rate utility oracle**.

The paper also calls the `.8885` oracle "floor- and gamma-matched." In
`frontier_rl/examples/run_hindsight_controls.py`, `oracle_g4` has floor zero
while the Thompson full stack has floor 0.1. Report it as the **no-floor,
gamma-matched oracle**.

### 1.8 Replace the hindsight proposition

The current proposition is not valid in general. Exact verification and
conditioning rewrite establish semantic correctness of a relabeled example;
they do not establish unbiasedness.

The implemented update has the form

\[
G_{\mathrm{rel}}(\theta)=
\mathbb E\!\left[
\sum_i a_i(\widetilde R)\,
\nabla_\theta\log\pi_\theta(\widetilde Z_i\mid G')
\;\middle|\;
K_x=0,\ G'=h(Z_{1:N})
\right].
\]

In the actual algorithm:

1. \(G'\) is selected from the same group being scored.
2. Conditioning on the original task failing correlates the rollouts.
3. Rewriting the prompt changes the distribution under which log-probability
   is evaluated.
4. The practical finite-\(N\) weights target \(J_{N-1}\), not exact ML.
5. No importance ratio corrects old-condition sampling to new-condition
   sampling.

Therefore this is generally an adaptively selected, off-policy auxiliary
gradient. It equals an on-policy truncated-objective gradient only under
strong conditions: the relabeled group law must equal an i.i.d. fresh group
law for a fixed destination, and the score must be evaluated under that same
law. The skill-chain construction can give directional alignment by symmetry;
that does not prove equality of distributions or magnitudes.

Suggested replacement:

> **Proposition (Relabeled-update characterization).** Conditional on the
> original group and relabel map, the implemented update is a verified
> weighted-likelihood update for the achieved task. It is an unbiased
> estimator of that task's \(T=N-1\) on-policy objective only if the selected,
> rewritten trajectories have the same joint law as an i.i.d. fresh rollout
> group for the achieved task. Otherwise it optimizes a shifted auxiliary
> distribution.

This agrees with the older, more accurate caveat in
`curriculum_maxrl/run_hindsight.py`: the relabeled group is not generally an
unbiased estimator and should be treated as an auxiliary imitation term.

### 1.9 The gate is motivated by the utility; it is not the same utility

The implementation admits a destination when
\(\hat p_{g'}\le\texttt{gate_max_p}\). It does not evaluate
\(u_N(\hat p_{g'})\), and it does not reject low-\(p\) destinations.

Use:

> "The high-\(p\) zero of \(u_N\) motivates a destination-saturation gate."

Do not use:

> "The same utility is applied a third time."

Also avoid "a self-achieved destination has \(p\approx1\) by construction."
One selected success does not imply a high fresh-rollout pass rate. The
defensible mechanism is that the achieved-goal marginal overweights
currently reachable destinations; independent destination estimates are
needed to determine which are actually saturated.

## 2. Results and statistical audit

### 2.1 Establish one statistical contract

The paper currently moves among training seeds, runs, sampler variants,
levels, prompts, checkpoints, and repeated evaluations as if each could be an
independent replicate. They cannot. Use the following contract throughout:

1. The independent unit for a training-method claim is an independently
   trained seed block.
2. Runs sharing a seed, warmstart, data order, or checkpoint are paired or
   correlated observations, not additional independent samples.
3. Levels, prompts, values of \(k\), checkpoints, and metrics from the same
   trained model are repeated measurements, not new training replicates.
4. Evaluation resampling measures evaluation noise. It does not measure
   training-seed uncertainty.
5. A factorial main effect requires the same sampler composition in each
   estimator arm. An interaction requires all cells of the factorial.
6. "Restores," "ties," and "equivalent" require a prespecified
   non-inferiority/equivalence margin and an interval estimate. Failure to
   reject a difference is not equivalence.

Add a methods paragraph that defines every reported quantity:

\[
\operatorname{mean@}k
=\frac1M\sum_{j=1}^M\frac{c_j}{k},
\qquad
\operatorname{pass@}k
=\frac1M\sum_{j=1}^M\mathbf1\{c_j>0\},
\]

when exactly \(k\) samples are drawn for each of \(M\) prompts. If \(n>k\)
samples are drawn and the Chen et al. estimator is used, say so and give

\[
\widehat{\operatorname{pass@}k}
=\frac1M\sum_{j=1}^M
\left(1-\frac{\binom{n-c_j}{k}}{\binom nk}\right).
\]

Also define:

- the prompt/level averaging measure;
- sampling temperature and decoding settings;
- whether AUC is a normalized trapezoidal integral or a checkpoint mean;
- whether its horizontal axis is steps, groups, tokens, or wall-clock;
- how the final checkpoint is selected;
- whether every `+/-` value is seed SD, seed SE, a confidence interval, or
  repeated-evaluation SD.

Do not use `+/-` without naming its source in the same caption or table note.

### 2.2 Maze: the arithmetic reproduces, the inference does not

The nine changes in mean pass@8 used for the estimator claim are:

| paper label | final minus warmstart pass@8 |
|---|---:|
| MaxRL | +.028846, +.033654, +.048077, +.024038, +.004808 |
| GRPO | -.019231, -.038462, -.033654, -.062500 |

The difference of group means is \(0.06635\). Exhaustively assigning five of
the nine values to the MaxRL label gives

\[
\frac{1}{\binom95}=0.0079365
\]

for a statistic at least as extreme as the observed separation. Thus the
reported arithmetic is reproducible.

The permutation reference distribution is not defensible, however. The five
MaxRL observations combine three `frontier_alp` seeds and two uniform seeds;
the four GRPO observations combine three uniform seeds and one frontier seed.
Seed IDs and warmstarts recur across configurations, and the sampler mix is
unequal across estimator labels. The nine labels are therefore neither
independent nor exchangeable. The value `p=0.0079` is the answer to a
counterfactual randomization that the experiment did not perform.

The phrase "6/6 paired-seed wins" has a related problem. It is two method
contrasts evaluated over the same three seed blocks, not six independent
seeds. With three paired blocks, the smallest possible two-sided exact
sign-flip p-value is \(2/2^3=0.25\). The reported step-matched
`p approximately 0.04` is a paired t-test with \(n=3\), so it is highly
sensitive to the normality assumption and any one block.

**Safe current conclusion**

> Across the selected maze runs, all five MaxRL-labeled pass@8 changes were
> positive and all four GRPO-labeled changes were negative. Because the run
> pool mixes sampler configurations and reuses seed blocks, this is
> descriptive evidence for an estimator-associated coverage difference, not
> a valid nine-replicate permutation test.

**Required confirmatory design**

Run a balanced

\[
\{\text{MaxRL},\text{GRPO}\}
\times
\{\text{uniform},\text{same fixed teacher}\}
\times S\text{ independent seed blocks}
\]

with identical warmstart rules, evaluation tasks, wall-clock budget, and
checkpoint rule. Use one primary endpoint, for example final-minus-warmstart
mean pass@8. For each seed \(s\), estimate

\[
d_s^{\rm est}=\frac12\left[
(Y_{M,U,s}-Y_{G,U,s})+(Y_{M,T,s}-Y_{G,T,s})
\right]
\]

and, separately, the estimator-by-teacher interaction

\[
d_s^{\rm int}=
(Y_{M,T,s}-Y_{M,U,s})-(Y_{G,T,s}-Y_{G,U,s}).
\]

Report all seed-level values, their paired mean/median, and an interval across
seed blocks. If a two-sided exact paired randomization test is desired, use at
least six independent blocks; fewer than six cannot attain \(p<.05\) even
with unanimous signs. Until this design exists, remove `p=0.0079` from the
abstract and treat the maze result as exploratory.

### 2.3 Figure 8 must not use directory globs as a cohort definition

`paper/figures/fig8_bands.py` currently classifies every
`matched_*grpo*.jsonl` file as GRPO and every other `matched_*.jsonl` file as
MaxRL. In this checkout that produces four GRPO runs and 18 MaxRL runs. The
MaxRL pool includes:

- uniform and multiple teacher forms;
- no, sparse, and dense hindsight;
- teacher-feedback and power-sweep variants;
- a wide-model run;
- repeated seed IDs and shared warmstarts.

These are not 22 independent replicates of one estimator contrast. The exact
`p=0.0001` is pseudoreplicated and must be removed. Adding another matching
log file also silently changes the paper figure.

Replace the globs with a checked-in manifest of exact files. For a figure that
isolates the estimator, use only a balanced cohort such as uniform MaxRL
versus uniform GRPO at the same seeds. If the teacher is part of the claim,
complete the missing factorial cells first. Show one point per independent
seed block and label ranges as ranges, not confidence intervals. Keep the
current 18-versus-4 plot only as a clearly labeled heterogeneous run inventory
with no p-value and no estimator-causal language.

### 2.4 Countdown supports sharpening, not a monotone gate frontier

The committed aggregate endpoint artifact
`paper/figures/data/b_scoreboard_3seed.json` contains:

| tier | arm | pass@16 (seed SD) | mean@16 (seed SD) |
|---|---|---:|---:|
| 1 | B1 no recycling | .541 +/- .020 | .278 +/- .054 |
| 1 | B2 ungated | .492 +/- .011 | .324 +/- .012 |
| 1 | B3 moderate gate | .484 +/- .011 | .282 +/- .031 |
| 1 | corrected strong gate | .564 (one seed) | .220 (one seed) |
| 2 | B1 no recycling | .274 +/- .015 | .117 +/- .030 |
| 2 | B2 ungated | .237 +/- .065 | .143 +/- .014 |
| 2 | B3 moderate gate | .279 +/- .019 | .133 +/- .006 |
| 2 | corrected strong gate | .238 (one seed) | .083 (one seed) |

The sharpening result is visible in the three-seed tier-1 point estimates:
ungated recycling raises mean@16 by .046 and lowers pass@16 by .049.

The gate points do not trace a monotone frontier:

- At tier 1, B2 to B3 worsens both point estimates: mean and coverage fall.
- At tier 2, B3 is a useful trade point: coverage is near B1 while
  \((.133-.117)/(.143-.117)\approx62\%\) of the B2 mean gain remains.
- The one-seed strong gate loses tier-2 coverage again and lowers both
  tier-2 metrics relative to B3.
- B3 was run with the old decay bug, while the strong point uses corrected
  code. Gate setting and implementation version are confounded.

`b_scoreboard_3seed.json` contains only means and SDs, so paired seed deltas
cannot be reconstructed. Add a raw table with one row per
arm, seed, tier, checkpoint, metric, effective threshold, and code version.

**Safe current conclusion**

> Ungated recycling shows a three-seed mean-versus-coverage trade on tier 1.
> On frontier tier 2, the moderate gate returned the coverage point estimate
> near the no-recycling baseline while retaining about 62% of the ungated
> mean-gain point estimate. The available settings do not establish a
> monotone dose-response.

Do not say "restores" as a statistical conclusion until a non-inferiority
margin for pass@16 is specified and the paired interval clears it.

For a confirmatory gate experiment, freeze the corrected decay code and run
the same seeds over a prespecified `gate_max_p` grid, including no gate.
Analyze paired changes on both metrics and report the Pareto-dominated points
rather than connecting every point as a frontier.

### 2.5 GSM8K is a pilot, not an abstract-level replication

The defensible facts are:

- the headline GRPO comparison has two training-seed pairs;
- the preregistered second-half regression occurs in one of two seeds;
- endpoint teacher deficits have the same sign in both seeds, but the second
  seed's mean@4 difference is inside the measured evaluation variability;
- mean@4 and pass@4 are computed from the same generations and are strongly
  correlated, so "4/4 signed contrasts" is not four independent
  confirmations;
- the delivered teacher treatment is weak: run-average dead-sampled
  fractions are approximately .66 in all cells;
- teacher sampling is with replacement while uniform sampling is without
  replacement, producing about 34% versus 43% unique prompts seen;
- the pass@k harness is approximately three times below the trainer's
  absolute validation metric, and that discrepancy is unresolved.

The `.0094` mean@4 and `.0172` pass@4 bars come from stochastic evaluations
of an unmodified base model. They estimate evaluation variability, not
between-training-seed uncertainty. They should not be plotted as if they were
the uncertainty of each trained cell, and `z` ratios based on them are
descriptive signal-to-noise ratios rather than formal hypothesis tests.

**Safe current conclusion**

> In a two-seed GSM8K pilot, the GRPO teacher endpoint was below its uniform
> counterpart on mean@4 and pass@4 in both seed pairs, but the preregistered
> second-half regression replicated in only one seed. Treatment delivery was
> weak and replacement policy was confounded with curriculum assignment.

Remove the LLM replication claim from the abstract. A confirmatory experiment
must:

1. use the same replacement policy in teacher and uniform arms;
2. log treatment separation and prompt uniqueness by seed;
3. replicate every cell, not only the GRPO pair;
4. use one frozen evaluation harness and reconcile its absolute scale;
5. report training-seed intervals separately from evaluation-resampling
   intervals;
6. prespecify one primary metric and one primary checkpoint contrast.

### 2.6 The timing headline does not match the committed artifact

`paper/figures/data/generation_timing.json` reports:

- 1,188 total mixed GSM8K and Countdown steps;
- overall step-weighted mean generation fraction .3117;
- eight sessions whose session mean exceeds .8, containing 282 steps;
- step-weighted mean .8337 within that post-selected subset.

It does not contain a labeled 272-step GSM8K cohort with an 85% mean.
Therefore "85% of step time, averaged over 272 logged steps across our
GSM8K sessions" is not reproducible.

Either remove the number or regenerate the artifact with suite, cell, seed,
session, numerator seconds, denominator seconds, and an aggregation rule.
With the current file, the only transparent statement is:

> Across all 1,188 mixed GSM8K and Countdown telemetry steps, the
> step-weighted mean generation fraction was 31.2%. Eight
> generation-dominated sessions selected by a greater-than-80% session mean
> contained 282 steps and averaged 83.4%.

The second sentence is a selected-subset description, not an estimate of
typical training cost.

### 2.7 Jugs is an important negative, but its evidence is outside this repo

The paper reports the nine-cell Jugs result and says its artifacts are in the
repository. They are absent from this checkout. The local
`jugs_llm/DESIGN_E_LLM3.md` still marks preregistration and B-arm launches as
unchecked.

The result artifacts exist in sibling repo `../maxrl`, beginning at commit
`eba7929`, under `curriculum_maxrl/jugs/`, including:

- `PREREG_E_LLM3.md`;
- `E_LLM3_POSTMORTEM.md`;
- `e_llm3_verdicts.json`;
- `entropy_trajectories.json`;
- `cells/*.json`;
- the analysis and noise-floor files.

Sync the exact result snapshot into this paper repo and record both the
preregistration commit and execution-code commit. Later bug-fix commits must
not silently replace the code provenance of the original runs.

The Jugs result is a negative boundary case, not support for the gate:
P-J1 was not confirmed, P-J2 was vacuous, P-J3 was not confirmed, and the
gate had a task-key granularity bug. Its valuable conclusion is that plain
MaxRL can also lose frontier coverage when the pool's learnable region
collapses to a short-template stratum. This directly rules out an
unconditional "MaxRL is coverage-safe" claim.

### 2.8 Other evidence-level corrections

- The IsaacLab result is one seed and the raw logs are held by another team.
  Call it a preregistered pilot/null case, not proof that allocation pays
  *only* in one regime.
- Five unanimous paired CPU directions are useful, but with five blocks the
  minimum two-sided sign-flip p-value is .0625. Report the raw paired
  differences.
- The `.8885` true-pass-rate oracle and `.8895` full stack have nearby point
  estimates. Call them "similar within the observed five-seed variation,"
  not statistically equivalent, unless an equivalence margin is added.
- A result seen at multiple \(k\) values, tiers, or metrics from one model is
  a response profile, not multiple replications.

## 3. Recommended reanalysis protocol

Use this protocol for every main-paper contrast.

### 3.1 Freeze the estimand before calculating a p-value

For each claim, write one row containing:

| field | required content |
|---|---|
| scientific claim | what causal/descriptive statement is being tested |
| primary endpoint | exact metric, task aggregation, and checkpoint |
| treatment contrast | exact cells and signs |
| independent unit | normally training seed |
| pairing/blocking | warmstart, seed, dataset order, and eval set |
| budget | steps, groups, tokens, or wall-clock |
| exclusion rule | fixed before reading outcomes |
| uncertainty | seed SD/CI, eval noise, or both |
| multiplicity | primary versus secondary analyses |

Do not define a cohort by outcome availability or a filename wildcard after
the results are known.

### 3.2 Emit one tidy seed-level artifact

Create a machine-readable table, for example
`paper/results/seed_endpoints.csv`, with at least:

```text
suite,experiment_id,code_commit,seed,warmstart_id,estimator,sampler,
recycler,gate_setting,budget_axis,budget,checkpoint_rule,tier,metric,value,
eval_sample_count,artifact_path
```

Every plotted aggregate and manuscript number should be derived from this
table or from a trajectory table linked by `experiment_id`. Keep failed,
stopped, and excluded runs in the manifest with a status and reason.

### 3.3 Separate three uncertainty sources

Report these separately:

1. **Training uncertainty:** variation across independent training seeds.
2. **Evaluation uncertainty:** finite prompts and stochastic generations for
   a fixed checkpoint.
3. **Temporal/checkpoint variation:** repeated observations along one run.

A hierarchical bootstrap may resample seeds first and prompts within seeds
second, but it cannot manufacture more training replicates. With very small
\(S\), show all seed points and avoid precision implied by asymptotic tests.

### 3.4 Match the analysis to the design

- Balanced paired cells: paired differences and a seed-block randomization
  test.
- Full \(2\times2\): estimator main effect, sampler main effect, and
  difference-in-differences interaction, each computed within seed.
- Unbalanced historical run collection: descriptive summaries or a
  sensitivity analysis, not a simple label permutation.
- Time curves: a prespecified scalar summary is primary; curve bands are
  descriptive unless generated from independent seeds.
- Equivalence/restoration: two one-sided tests or an interval against a
  prespecified practical margin.

Effect sizes and raw seed values should lead; p-values are secondary.

## 4. Artifact and reproducibility requirements

### 4.1 Remove hard-coded result paths

The current figure pipeline has four high-risk points:

| script | issue | required fix |
|---|---|---|
| `paper/figures/fig2_ladder.py` | result values are literals copied from documents | read a versioned result table |
| `paper/figures/fig3_gsm8k.py` | trajectories and noise values are literals | read per-seed checkpoints and eval-noise artifact |
| `paper/figures/fig7_sharpening.py` | strong-gate point is hard-coded | vendor its raw seed endpoint and derive it |
| `paper/figures/fig8_bands.py` | unstable directory globs define cohorts | use an explicit run manifest |

Figures should fail with a clear error when a declared input is missing.
They should not discover new experimental runs automatically.

### 4.2 Add a frozen paper manifest

Add `paper/results/manifest.json` or YAML with:

- manuscript and analysis commit;
- exact input path and checksum for every figure/table;
- suite, cell, seed, warmstart, code commit, and run status;
- metric implementation and checkpoint-selection function;
- whether a value is raw, derived, or manually transcribed;
- the command that regenerates each output.

The manifest should distinguish the reviewed run code from later bug fixes.

### 4.3 Make one clean-checkout command authoritative

Provide one command that:

1. validates all manifest files and checksums;
2. rebuilds derived result tables;
3. rebuilds every figure;
4. rebuilds the PDF;
5. fails on missing inputs, stale outputs, or undefined references.

The current reproducibility sentence "every figure regenerates from JSON
artifacts committed to the repository" is false until the hard-coded figures
and missing Jugs artifacts are fixed.

### 4.4 Keep one manuscript body

`paper/main.tex` and `paper/main_iclr.tex` duplicate the manuscript body and
already require edits in parallel. Move the shared content to one included
file and keep format-specific preambles/wrappers. This prevents math and
claim fixes from diverging.

### 4.5 Build/PDF cleanup

The canonical `paper/main_iclr.pdf` is 16 pages, with the appendix starting on
PDF page 14. If the target is nine main-text pages, the paper is still over
budget before a proper conclusion is added.

The current log also contains:

- an overfull box of about 19.8 pt around the main results table;
- an overfull box of about 79.6 pt around the hyperparameter table;
- duplicate figure and table destination identifiers.

Use shorter tables or `tabularx`, move details to the appendix, and fix the
counter/anchor issue. A release build should have no overfull boxes and no
duplicate-destination warnings. Do not circulate the stale workspace-level
`../paper-draft.pdf`; its hash differs from `paper/main_iclr.pdf`.

## 5. Writing revision plan

### 5.1 Replace categorical claims with scoped claims

| current idea | safe replacement |
|---|---|
| "expected learning signal" | "expected absolute coefficient mass" |
| "the zeros partition difficulty" | "the utility vanishes at \(p=0,1\) and has one interior peak" |
| "band location and width are derived" | "the peak is derived; any band width requires a stated threshold" |
| "no sampler can reach the dead zone" | "at \(p=0\) under the current policy/support, sampling alone cannot produce a success" |
| "recycling is the only channel" | "verified relabeling is one way to create a positive auxiliary update for an all-fail requested task" |
| "hindsight yields the ML gradient" | "hindsight yields a verified, adaptively selected auxiliary update; it is on-policy only under a distribution-matching condition" |
| "the same utility is applied to the gate" | "the high-\(p\) zero motivates a pass-rate saturation threshold" |
| "oracle water-filling" | "true-pass-rate utility oracle" |
| "every MaxRL run grows coverage" | scope to the audited maze cohort and disclose its heterogeneous design |
| "LLM replication" | "two-seed GSM8K pilot with a same-sign endpoint trend" |
| "monotone gate dial/frontier" | "one useful moderate-gate operating point on frontier tier 2" |
| "across four task suites" | name the exact suite and evidence status for each claim |

### 5.2 Suggested abstract at the current evidence level

> Group-relative RL with verifiable binary rewards is often combined with
> difficulty sampling and hindsight relabeling, although these interventions
> are usually designed independently of the estimator. For the practical
> success-conditioned estimator with \(N\) i.i.d. rollouts, we derive its
> expected absolute coefficient mass,
> \(2[1-(1-p)^N-p]=2(\operatorname{pass@}N-\operatorname{pass@}1)\),
> and show that the drop-all-fail implementation targets truncation order
> \(T=N-1\). The identity supplies a compute-dependent sampling heuristic,
> while also clarifying its limits: coefficient mass is not gradient norm,
> and verified hindsight is generally an off-policy auxiliary update. In
> maze experiments, a heterogeneous historical run set shows opposite
> pass@8 directions for MaxRL- and GRPO-labeled runs, motivating a balanced
> confirmatory factorial. In three-seed Countdown experiments, ungated
> recycling raises mean accuracy while reducing pass@16 coverage; a moderate
> destination-pass-rate gate gives one useful frontier-tier trade point, but
> the available settings do not establish monotonicity. A two-seed GSM8K
> pilot and a negative Jugs study expose treatment-delivery, diversity, and
> pool-design limits. The results motivate auditing both mean accuracy and
> coverage when adding curricula or recyclers.

After the balanced maze and fixed-code gate experiments are complete, replace
the exploratory sentences with their prespecified estimates and intervals.
Do not put the current `p=0.0079`, GSM8K replication, or monotone-dial claims
back into the abstract without that evidence.

### 5.3 Reorganize around the strongest contribution

The strongest paper is narrower than the current nine-rung narrative:

1. Exact coefficient-mass and truncation analysis.
2. Estimator-matched sampling heuristic, with explicit surrogate limits.
3. Balanced estimator experiment as the primary empirical test.
4. Recycling sharpening and the gated operating point.
5. GSM8K, IsaacLab, and Jugs as pilots/negative boundary cases.

Move most ladder controls, run histories, and diagnostic prose to the
appendix. Keep one primary table with effect, number of independent seeds,
uncertainty source, and evidence grade. Add a short conclusion; the current
paper goes directly from limitations to references.

### 5.4 Tighten result prose

For each paragraph, use this order:

1. experimental question and design;
2. primary estimate with raw seed count and interval;
3. one sentence on the prespecified decision;
4. secondary mechanism evidence;
5. limitation or alternative explanation.

Avoid treating a post hoc mechanism story as if it were the tested outcome.
Avoid "exact," "proved," "confirmed," and "replicated" for empirical claims
unless the design and inference justify those words.

## 6. Engineering priorities and acceptance criteria

### P0: required before submission

- [ ] Replace the overloaded `u_N` notation with \(A_N=2u_N\) everywhere,
  including Figure 1, Algorithm 1, captions, and code comments.
- [ ] Make the GRPO derivation match sample-SD deployment, or change the
  implementation and rerun every affected experiment.
- [ ] Replace the hindsight ML-gradient proposition with the conditional,
  shifted-update characterization.
- [ ] Remove undefined band-width/dead-zone and "only channel" claims.
- [ ] Remove the maze `p=0.0079` and Figure 8 `p=0.0001` unless a valid
  seed-block analysis replaces them.
- [ ] Remove GSM8K replication language from the abstract and contributions.
- [ ] Replace the monotone gate-frontier claim with the tier-2 operating-point
  statement.
- [ ] Correct or remove the 85%/272-step timing statement.
- [ ] Sync the Jugs preregistration, raw cells, verdict, and postmortem into
  this repo.
- [ ] Replace hard-coded figure values and unstable run globs with manifest
  inputs.

### P1: strongest use of the next experiment budget

- [ ] Run the balanced maze estimator-by-sampler factorial with at least six
  independent seed blocks for a two-sided exact test.
- [ ] Run a fixed-code, fixed-decay, paired multi-seed gate-strength sweep.
- [ ] Add raw per-seed Countdown endpoints and paired differences.
- [ ] Run a replacement-matched, treatment-separated GSM8K factorial only if
  an LLM-scale headline is still required.
- [ ] Add the metric/statistical contract and one primary endpoint per suite.
- [ ] Consolidate the two TeX bodies, add a conclusion, and meet the page
  target.

### P2: robustness and presentation

- [ ] Add formula enumeration tests for MaxRL, RLOO, and both GRPO SD
  conventions over several \(N,p\) values.
- [ ] Add a clean-checkout paper build in CI.
- [ ] Report evaluation and training uncertainty separately in every figure.
- [ ] Add sensitivity analyses for checkpoint choice and wall-clock versus
  step matching.
- [ ] Publish a compact claim-to-artifact table in the appendix.

### Final acceptance gate

The revision is ready to circulate only when all of the following are true:

1. Every theorem/proposition states its probability law, normalization, and
   deployment assumptions.
2. Figure 1 numerically matches the estimator code used in the experiments.
3. Every main empirical point can be traced to a raw seed row and exact run
   artifact.
4. No p-value treats levels, metrics, checkpoints, reused seeds, or
   heterogeneous arms as independent replicates.
5. Abstract and contribution claims are no stronger than the evidence grades
   in this review.
6. A clean checkout regenerates all figures and the canonical PDF.
7. The PDF has no overfull boxes, duplicate destinations, or stale alternate
   copy.

## 7. Recommended claim after revision

The defensible center of the paper is:

> For the practical success-conditioned estimator under i.i.d. binary
> rewards, expected absolute coefficient mass is exactly
> \(2(\operatorname{pass@}N-\operatorname{pass@}1)\), and the dropped-all-fail
> implementation targets truncation order \(N-1\). This identity motivates a
> compute-aware sampling heuristic but does not by itself prove a learning
> band, optimal prompt allocation, or on-policy hindsight. Empirically,
> estimator choice and verified-relabel admission can change the trade
> between mean accuracy and pass@k coverage; those effects must be estimated
> with balanced seed-level designs.

That claim is mathematically correct, useful to practitioners, and compatible
with the negative results already present in the project.
