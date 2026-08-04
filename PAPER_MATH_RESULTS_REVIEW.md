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
