# Research notes · curriculum learning × MaxRL · updated 2026-08-04

## Scope of this review

I reviewed the 17-page working draft, **“The Estimator Decides: What Curricula and Failure Recycling Can and Cannot Do in RL with Verifiable Rewards”**, against the 44-page MaxRL baseline, **“Maximum Likelihood Reinforcement Learning”** by Tajwar, Zeng, et al. The review focuses on mathematical correctness, the exact relationship to the baseline paper, causal and statistical support for the headline claims, and how to make the writing cleaner and more persuasive.

This is an internal paper review, not an independent reproduction of the code or an exhaustive novelty search across every cited 2025–2026 paper.

---

# 1. Overall assessment

The draft has a genuinely strong core idea. The most valuable observation is not merely that a curriculum should sample medium-difficulty tasks. It is that, for the **practical centered MaxRL estimator actually used in training**, the finite-group coefficient profile can be derived exactly from the estimator. This gives a principled bridge between:

- the objective-level weight function studied in MaxRL;
- the finite-group event structure that determines whether a sampled prompt produces contrast;
- the data-level decision of which prompts to sample before spending rollouts;
- and the behavior of relabeling when it redirects training toward already-saturated destinations.

That is a coherent paper.

The draft also shows unusually good scientific discipline: it reports negative results, discloses a gate implementation bug, retracts earlier interpretations, distinguishes wall-clock from step-matched effects, and includes a boundary suite that contradicts an unconditional safety story. Those are not weaknesses. They are signs that the project has matured beyond a demo.

The present version, however, asks the evidence to support claims that are broader than it currently can. The main risks are:

1. **The central “learning signal” theorem is named too strongly.** The exact quantity is coefficient/advantage mass, not gradient norm or expected improvement. A stronger and cleaner factorization theorem can fix this.
2. **The “dead zone” is partly an implementation choice.** The full control-variate estimator in the MaxRL paper can emit a nonzero update on an all-fail group; the practical algorithm chooses to zero that group. The paper must scope its impossibility claim to the deployed zero-constant-group estimator and should test the full-control-variate alternative.
3. **The paper mostly establishes an estimator main effect, not yet a curriculum × estimator interaction.** The maze result shows MaxRL coverage grows across variants while GRPO coverage falls across variants. It does not by itself prove that adding the teacher harms GRPO. The GSM8K interaction is still underpowered and confounded.
4. **The hindsight theorem remains too generous.** Exact verification and prompt rewriting make a relabel semantically valid, but they do not make source-generated trajectories on-policy for the destination prompt. The update is generally an off-policy semi-gradient unless a conditional-law matching condition holds.
5. **Some statistical units appear correlated or multiply counted.** The run inventory and the exact unit of permutation need to be made explicit, especially where the paper alternates between 5 MaxRL runs and 18 MaxRL runs.
6. **The gate headline should wait for corrected multi-seed reruns.** The current three-seed operating point came from a decay bug; the corrected setting has one seed.
7. **The prose is excellent at generating momentum, but too often outruns the result.** The draft needs fewer slogans, narrower qualifiers, and a cleaner separation between theorem, empirical mechanism, and interpretation.
8. **The anonymous PDF currently deanonymizes the submission through the public GitHub URL.** This is a submission-blocking issue.

My recommendation is **major revision, not a conceptual restart**. The paper should be narrowed around one precise thesis:

> For a fixed group estimator, its finite-sample coefficient profile determines where sampled prompts can produce a contrastive update. This profile can guide allocation, but it does not determine cross-task transfer; relabeling can create contrast where allocation cannot, while destination selection can induce concentration and coverage loss.

That version is both stronger mathematically and more defensible empirically.

---

# 2. Submission-blocking priorities

## P0.1 — Anonymize the paper and artifacts

Page 1 says “Anonymous authors” while linking directly to `github.com/linjiw/curriculum-maxrl`. This reveals the author identity immediately. Before anonymous submission:

- replace the repository with an anonymous artifact link;
- scrub usernames from paths, commit metadata, issue links, and generated JSON;
- verify the PDF metadata, supplementary zip, code comments, and README history;
- avoid references such as “our earlier public note” that can be searched back to the author.

## P0.2 — Replace “expected learning signal” with an exact factorization

The current proposition proves

\[
\mathbb E\!\left[\sum_i |w_i|\right]
=2\left(\operatorname{pass@}N(p)-p\right),
\]

not an equality with gradient norm, SNR, or improvement. The paper already admits this in Remark 1, but the abstract, figure captions, and interpretation repeatedly call the quantity “the expected learning signal.” A reviewer will correctly say that score-vector geometry can make two prompts with identical coefficient mass produce very different gradients.

The right fix is not merely to weaken the prose. Add the exact factorization in Section 3; it makes the theorem more meaningful while remaining correct.

## P0.3 — Scope the dead-zone claim to the practical estimator and run the missing baseline

The draft says that no loss-level choice can rescue an all-fail group and that recycling is the only channel into the dead zone. That is too broad.

The MaxRL baseline distinguishes three related estimators:

1. the raw success-conditioned estimator in its Theorem 2;
2. the variance-reduced estimator in Eq. (10), whose control variate is retained even when \(K=0\);
3. the practical Algorithm 1 implementation, which zeroes the entire group when \(K=0\).

Only the third creates the exact two-zero activity profile used by this paper. The full Eq. (10) estimator emits \(-\frac1N\sum_i S_i\) on an all-fail group and remains unbiased for the \(T=N\) objective. It may be noisy or empirically harmful, but it is an obvious alternative that directly challenges “recycling is the only channel.”

Add a baseline that retains the all-fail control variate. If it fails, that is useful evidence: negative-only redistribution is not an adequate substitute for a verified positive anchor. If it works, the paper’s boundary must change.

## P0.4 — Separate the estimator main effect from the curriculum interaction

The strongest maze result is:

- every selected MaxRL run has positive \(\Delta\operatorname{pass@8}\);
- every selected GRPO run has negative \(\Delta\operatorname{pass@8}\).

This supports an **estimator-conditioned coverage main effect that is robust across curriculum variants**.

It does **not** by itself establish:

> “The same curriculum that trains stably under MaxRL degrades GRPO.”

That is an interaction claim. It requires a powered comparison of GRPO+teacher against GRPO+uniform, relative to the corresponding MaxRL teacher effect, ideally with:

- the same warmstarts;
- matched prompt schedules or a frozen external schedule;
- paired seeds;
- and a factorial interaction statistic.

The GSM8K result is suggestive but currently has two seeds, weak delivered steering, replacement/repetition confounding, and only one seed showing the pre-registered regression shape. Until the decisive experiment finishes, use the more accurate headline:

> “Across curriculum variants, coverage behavior separates by estimator: MaxRL variants grow pass@k while GRPO variants lose it.”

## P0.5 — Rewrite the hindsight theorem as an off-policy-at-destination result

A trajectory generated under source prompt \(x\) is not generally sampled from the policy conditioned on relabeled destination prompt \(g'\). Rewriting the prompt and verifying the response makes the label correct, but does not make the data on-policy for \(g'\).

The correct statement is:

> Hindsight produces a verified, destination-conditioned semi-gradient under the source-induced relabel distribution. It equals the fresh destination ML gradient only when the success-conditioned source-induced law and destination-policy law have matching score expectations; equality of the conditional laws is a sufficient condition.

This distinction should appear in the abstract, not only in Proposition 4’s caveat.

## P0.6 — Rerun the corrected gate at multiple seeds

The current paper openly states that a decay bug made all three-seed gated runs weaker than designed, while the corrected full-strength gate has one seed. The honesty is good, but the main headline “the gate restores coverage” should rest on corrected multi-seed runs.

At minimum, complete a fixed-decay sweep with:

- no recycling;
- ungated recycling;
- 2–3 gate strengths;
- 3–5 paired seeds;
- a pre-declared primary tier and metric;
- and the actual destination-pass-rate distributions.

## P0.7 — Rebuild the statistical unit-of-analysis table

The paper reports:

- five MaxRL and four GRPO runs for the 9-run main effect;
- later, eighteen MaxRL and four GRPO runs for the level-resolved analysis;
- “6/6 paired seed wins” although there are three random seeds and two teacher variants;
- exact permutation tests over run-level observations that may share warmstarts and seeds.

Create a run registry with one row per independent training run:

| run ID | suite | estimator | teacher | recycler | gate | seed | warmstart ID | schedule ID | confirmatory role | included in which test |
|---|---|---|---|---|---|---|---|---|---|---|

Then use seed/warmstart clusters as the inferential unit. If multiple configurations within a seed are included, use a blocked or cluster permutation, not an unrestricted run-label permutation.

## P0.8 — Remove or subordinate preliminary LLM claims in the abstract

The LLM-scale result is currently useful as a stress test and treatment-delivery diagnosis, not yet as confirmatory evidence. The abstract should not say “confirmed at LLM scale” until the steering-controlled, uniform-with-replacement, multi-seed cell is complete.

---

# 3. Mathematical review

## 3.1 The most important distinction: three MaxRL estimators

The baseline paper’s Theorem 2 is correct for its raw estimator. Your correction applies to the **practical centered algorithm**, not to MaxRL as a whole. This distinction should be explicit and visual.

Let

\[
r_i\in\{0,1\},\qquad K=\sum_{i=1}^N r_i,\qquad
S_i=\nabla_\theta\log m_\theta(z_i\mid x),\qquad
p=\Pr(r=1\mid x).
\]

### A. Raw success-conditioned estimator

\[
\widehat g_N^{\rm raw}
=\mathbf 1\{K\ge1\}\frac1K\sum_{i=1}^N r_iS_i.
\]

This is the estimator in the MaxRL baseline’s Theorem 2 and is unbiased for the \(T=N\) truncated objective.

Its coefficient mass is

\[
\mathbb E\!\left[\sum_i|w_i|\right]
=\Pr(K\ge1)=1-(1-p)^N.
\]

This profile is monotone and does **not** have a mastered-tail zero at \(p=1\). Therefore, the ZPD-shaped curve is not a generic property of “success-conditioned MaxRL”; it comes from centering plus the practical constant-group convention.

### B. Full variance-reduced estimator from Eq. (10)

\[
\widehat g_N^{\rm fullCV}
=\mathbf1\{K\ge1\}\frac1K\sum_i r_iS_i-rac1N\sum_iS_i.
\]

The second term is retained even at \(K=0\). It is a zero-mean control variate, so this estimator remains unbiased for \(T=N\).

On an all-fail group it emits

\[
-\frac1N\sum_iS_i,
\]

which is generally nonzero at the sample level. Its expected coefficient mass is

\[
M_N^{\rm fullCV}(p)=2(1-p)-(1-p)^N.
\]

At \(p\to0\), this tends to 1 rather than 0. This estimator is the clearest counterexample to an unconditional “no loss-level update is possible” claim.

### C. Practical centered estimator used in Algorithm 1

\[
\widehat g_N^{\rm prac}
=\mathbf1\{K\ge1\}\sum_{i=1}^N
\left(\frac{r_i}{K}-\frac1N\right)S_i.
\]

When \(K=0\), the entire group is zeroed. This is the estimator your theory analyzes. It is unbiased for \(T=N-1\), not \(T=N\).

Its coefficient mass is

\[
M_N^{\rm prac}(p)
=2\left(1-(1-p)^N-p\right)
=2\left((1-p)-(1-p)^N\right).
\]

This is the ZPD-shaped profile.

### Recommended presentation

Add a compact table near the beginning of Section 3:

| estimator | \(K=0\) behavior | objective order | expected coefficient mass |
|---|---:|---:|---:|
| raw success average | 0 | \(T=N\) | \(1-(1-p)^N\) |
| full control variate | \(-N^{-1}\sum S_i\) | \(T=N\) | \(2(1-p)-(1-p)^N\) |
| practical centered/drop | 0 | \(T=N-1\) | \(2[(1-p)-(1-p)^N]\) |

This table would strengthen the title: the estimator implementation literally decides the activity profile.

---

## 3.2 Fix the factor-of-two inconsistency

The current draft uses \(u_N\) in two incompatible ways on page 3:

- Figure 1 defines \(u_N(p)=\operatorname{pass@}N-\operatorname{pass@}1\) and labels the y-axis “advantage mass / 2.”
- Proposition 1 defines the full mass \(2(\operatorname{pass@}N-\operatorname{pass@}1)\equiv u_N(p)\).

Use two symbols:

\[
A_N(p):=\mathbb E\!\left[\sum_i|w_i|\right],
\qquad
u_N(p):=\frac12 A_N(p)
=\operatorname{pass@}N(p)-p.
\]

Then:

- all exact “mass” statements use \(A_N\);
- the teacher samples by \(\nu_N^\gamma\);
- RLOO’s normalized half-mass is \(p(1-p)\);
- GRPO’s normalized half-mass is \(N^{-1}\mathbb E\sqrt{K(N-K)}\).

The scale factor does not affect sampling, but inconsistent notation makes the theorem appear unstable.

---

## 3.3 Add the stronger exact factorization theorem

This theorem is the cleanest way to answer the “mass is not learning signal” objection.

For the practical centered estimator and \(0<p<1\), define

\[
\mu_+(x)=\mathbb E[S\mid r=1,x],\qquad
\mu_-(x)=\mathbb E[S\mid r=0,x].
\]

Since

\[
\mu_+(x)=\frac{\nabla p}{p},\qquad
\mu_-(x)=-\frac{\nabla p}{1-p},
\]

we have

\[
\mu_+(x)-\mu_-(x)=\frac{\nabla p}{p(1-p)}.
\]

For a group with \(K=k\ge1\), the total positive coefficient is

\[
k\left(\frac1k-\frac1N\right)=1-\frac{k}{N},
\]

and the total negative coefficient has the same magnitude. Therefore

\[
\mathbb E[\widehat g_N^{\rm prac}\mid K=k,x]
=\left(1-\frac{k}{N}\right)(\mu_+-\mu_-).
\]

Averaging over \(K\) gives

\[
\boxed{
\mathbb E[\widehat g_N^{\rm prac}\mid x]
=\nu_N(p)\,[\mu_+(x)-\mu_-(x)]
=\frac{\nu_N(p)}{p(1-p)}\nabla p.
}
\]

Because

\[
\frac{\nu_N(p)}{p(1-p)}
=\frac{1-(1-p)^{N-1}}{p}
=\sum_{j=0}^{N-2}(1-p)^j,
\]

this simultaneously proves the \(T=N-1\) objective result.

### Why this theorem is better

It lets you say something exact but properly scoped:

> The normalized coefficient mass \(\nu_N(p)\) is the estimator-controlled scalar multiplying the success-versus-failure score separation. It is not the entire gradient geometry, but it is not merely a heuristic either.

This is stronger than the current “surrogate” wording and safer than “exact expected learning signal.”

A more general version can cover MaxRL, RLOO, and GRPO: for any symmetric zero-sum binary group estimator with equal aggregate positive and negative coefficient mass, the expected gradient is half the expected coefficient mass times \(\mu_+-\mu_-\).

---

## 3.4 Clarify what the two endpoint zeros do and do not prove

The current text says the “two zeros partition difficulty into a dead zone, a learnable band, and a mastered tail.” Mathematically, \(\nu_N(p)\) has two **point zeros**, at \(p=0\) and \(p=1\), and one interior maximum. The zeros alone do not define the width of three intervals.

Use:

> “The endpoint zeros and unique interior maximum induce three operational regimes.”

Then define the regimes by budget or a level set.

### Exact peak

\[
\nu_N(p)=(1-p)-(1-p)^N.
\]

The unique maximizer is

\[
p^*=1-N^{-1/(N-1)}.
\]

The approximation \(p^*\approx \log N/N\) is good for the rollout counts used, but the exact expression should remain in the theorem. An optional expansion is

\[
p^*=\frac{\log N}{N-1}
-\frac{(\log N)^2}{2(N-1)^2}
+O\!\left(\frac{(\log N)^3}{N^3}\right).
\]

### “Dead” should be budget-indexed

For a finite training allocation of \(B\) groups to a task, the expected number of nonconstant groups is

\[
B\left[1-p^N-(1-p)^N\right].
\]

A task is operationally dead at budget \((B,N)\) when this is much less than one. That is more precise than treating all small \(p\) as a mathematical zero.

At \(p=0\) exactly, no reallocation can create success under the same policy and verifier. At \(p>0\), allocating more groups or increasing \(N\) can eventually obtain a live group. The paper should distinguish “strictly unreachable” from “negligible at the deployed budget.”

### “Width is derived” needs a definition

The curve’s shape is derived. A band width requires a threshold, for example

\[
\mathcal B_{N,\eta}
=\{p:\nu_N(p)\ge\eta\nu_N(p^*)\}.
\]

Without an \(\eta\), there is no unique mathematical width. The safer abstract wording is:

> “The utility profile’s shape and peak are set by \(N\), without hand-setting a target difficulty band.”

---

## 3.5 Correct the mastered-tail comparison with RLOO

The current Figure 1 caption says MaxRL “releases mastered prompts fastest.” For fixed \(N\), as \(p\to1\), letting \(q=1-p\),

\[
\nu_N^{\rm MaxRL}=q-q^N\sim q,
\qquad
\nu^{\rm RLOO}=p(1-p)\sim q.
\]

MaxRL and RLOO have the same first-order decay in the mastered tail; in fact, for \(N>2\), MaxRL’s mass is slightly larger in the interior because

\[
(q-q^N)-q(1-q)=q^2-q^N\ge0.
\]

The important comparison is with GRPO:

\[
\frac{\nu_N^{\rm GRPO}}{\nu_N^{\rm MaxRL}}
\to\sqrt{N-1}
\qquad\text{as }p\to1
\]

under the finite-\(N\), population-standard-deviation convention used in the draft.

Use:

> “MaxRL and RLOO release mastered prompts linearly, whereas finite-group GRPO retains a \(\sqrt{N-1}\)-larger coefficient mass in the mastered tail.”

---

## 3.6 Keep finite-group GRPO and population GRPO separate

The MaxRL baseline’s weight function

\[
w_{\rm GRPO}(p)=\frac1{\sqrt{p(1-p)}}
\]

is a population/infinite-group result. Your finite-group expected half-mass is

\[
\nu_{N}^{\rm GRPO}(p)
=\frac1N\mathbb E_{K\sim\mathrm{Bin}(N,p)}\sqrt{K(N-K)},
\]

with constant-reward groups contributing zero. The corresponding finite-\(N\) expected gradient weight is

\[
w_{N}^{\rm GRPO}(p)
=\frac{\nu_N^{\rm GRPO}(p)}{p(1-p)}.
\]

As \(N\to\infty\), this approaches the population curve. State this explicitly. It makes your finite-sample critique of the baseline weight view precise rather than rhetorical.

Also specify whether the implementation uses population standard deviation or the \(N-1\) sample correction. The exact constant depends on that convention.

---

## 3.7 Water-filling is a one-step mass oracle, not a universal curriculum ceiling

Proposition 3 is correct under the intended separable, fixed-\(p_i\), integer allocation model. If task \(i\) currently has \(N_i\) rollouts, adding one rollout increases normalized mass by

\[
\nu_{N_i+1}(p_i)-\nu_{N_i}(p_i)
=p_i(1-p_i)^{N_i},
\]

the probability that the new rollout is the first success after \(N_i\) failures. Because this marginal decreases with \(N_i\), greedy allocation is optimal for the one-step separable mass objective.

But it is not a ceiling on final AUC or final held-out performance. Sequential curricula involve:

- policy change;
- cross-task transfer;
- interference;
- uncertainty;
- exploration floors;
- and delayed unlocking of harder tasks.

Use “one-step mass-optimal allocator” or “mass oracle,” not “the oracle ceiling for pure samplers.”

### Important logical correction in Section 6.1

The draft says that the true-pass-rate oracle ties the full stack and therefore “perfect difficulty information is worth almost nothing over the cheap posterior.” The reported numbers do not support that inference:

- posterior teacher: about 0.728 AUC;
- sharper posterior teacher: about 0.782 in the project notes;
- true-pass-rate oracle: about 0.8885;
- posterior + recycling full stack: about 0.8895.

The oracle tying **posterior + recycling** does not show that pass-rate information is valueless. It shows that recycling can compensate for imperfect allocation. The value of perfect information must be assessed by comparing oracle-only against posterior-only under matched floor and concentration.

A correct conclusion is:

> “On this chain, the mass oracle substantially improves over the posterior-only teacher; the posterior-plus-recycling stack nevertheless reaches the same empirical level, and recycling still adds a smaller gain on top of the oracle.”

---

## 3.8 The estimator determines activity, not task transfer

The draft already hints at this through the per-bin-policy failure, the one-shot-stream result, and the Jugs boundary. It should become an explicit theoretical limitation:

> The estimator determines whether a task can emit a contrastive update. It does not determine whether that update improves other tasks.

For evaluation objective \(J_\rho\), the first-order value of training on task \(i\) is closer to

\[
\Delta J_\rho(i)
\approx \eta\,\nabla J_\rho(\theta)^\top
\mathbb E[\widehat g_i],
\]

not simply \(\|\mathbb E[\widehat g_i]\|\) or \(\nu_N(p_i)\). A curriculum needs both:

1. **local estimator activity**, measured by \(\nu_N(p_i)\);
2. **transfer/alignment**, determined by shared parameters and task structure.

This explains several results more cleanly than “the band collapsed” alone:

- shared-policy MountainCar succeeds; per-bin parameters do not;
- \(\gamma=4\) helps skill chains with compounding transfer but not flat pools;
- one-shot streams obtain little value from relabeling;
- learnable-everywhere IsaacLab favors breadth.

A future extension could learn a transfer multiplier, but the present paper should state the boundary directly.

---

## 3.9 Tighten the hindsight mathematics

### The current issue

Suppose a trajectory is generated from

\[
z\sim \pi_\theta(\cdot\mid x)
\]

and relabeled to destination \(g'=H(x,z)\). After rewriting the conditioning, the update uses

\[
\nabla_\theta\log\pi_\theta(z'\mid g').
\]

The sample was not generally drawn from \(\pi_\theta(\cdot\mid g')\). Therefore exact verification does not imply an unbiased policy-gradient or ML-gradient estimator for destination \(g'\).

### Recommended formalization

Let \(q_\theta^{x\to g'}\) be the distribution of rewritten trajectories induced by sampling from source \(x\), selecting \(g'\), and applying the rewrite. The expected hindsight score is

\[
g_{\rm H}(g')
=\mathbb E_{z'\sim q_\theta^{x\to g'}(\cdot\mid R_{g'}=1)}
\left[\nabla_\theta\log\pi_\theta(z'\mid g')\right].
\]

The fresh destination ML score is

\[
g_{\rm ML}(g')
=\mathbb E_{z'\sim \pi_\theta(\cdot\mid g',R_{g'}=1)}
\left[\nabla_\theta\log\pi_\theta(z'\mid g')\right].
\]

Then:

- equality of the two success-conditioned laws is sufficient for equality of gradients;
- equality of score expectations is necessary and sufficient;
- otherwise, hindsight is a distribution-shifted semi-gradient.

If the score norm is bounded by \(B\), a simple diagnostic bound is

\[
\|g_{\rm H}-g_{\rm ML}\|
\le 2B\,\mathrm{TV}\!\left(q_+^{x\to g'},\pi_+^{g'}\right).
\]

This gives the paper a precise reason to measure destination shift.

### Same-group destination selection adds another bias

If \(g'\) is chosen as the deepest outcome achieved by the same group used for training, the destination is adaptively selected from that group. The resulting trajectories are exchangeable but not generally i.i.d. from a fixed destination law.

A clean control is cross-fitting:

- choose the destination from one subset of trajectories;
- evaluate/reweight a disjoint subset under that destination;
- or choose the destination from a separate proposal group.

This is not necessarily required for the deployed method, but it should be discussed as the cleanest route to a theorem.

### Better wording

Replace:

> “Hindsight gradients are exact, not approximate.”

with:

> “Hindsight gradients are exact only under a measurable conditional-law matching condition; exact verification alone guarantees semantic correctness, not on-policy sampling at the destination.”

---

## 3.10 Formalize the sharpening pressure through size-biased destination selection

The current mechanism sentence says that a self-achieved destination has \(p\approx1\). One sampled achievement does not by itself imply destination pass rate near one, especially after the prompt is rewritten.

A more general mechanism is **occupancy bias**:

- the policy produces outcomes according to its current output distribution;
- relabel destinations are selected from those outcomes;
- therefore high-probability destinations are sampled more often and trained more often;
- repeated self-training can create a rich-get-richer concentration effect.

For categorical destinations with source occupancy probabilities \(q(g)\), an ungated relabel samples \(G\sim q\), and

\[
\mathbb E[q(G)]=\sum_g q(g)^2.
\]

This is the collision probability: sampled destinations are automatically biased toward modes. A gate with acceptance \(a(g)\) changes the destination distribution to

\[
\widetilde q(g)
=\frac{q(g)a(g)}{\sum_h q(h)a(h)}.
\]

Choosing \(a(g)\) to decrease as destination pass rate saturates directly suppresses the rich-get-richer tail. This does not prove pass@k loss by itself, but it gives recycling-induced sharpening a clearer mathematical mechanism than “the value was produced once, so \(p\approx1\).”

Empirically, show:

- histogram of destination \(\hat p_{g'}\) over time;
- destination occupancy versus destination pass rate;
- admitted and rejected destination distributions;
- coverage change as a function of cumulative relabel dose.

---

## 3.11 The gate is motivated by the algebra, but its threshold is tuned

The current method says the same utility “supplies the fix,” but Algorithm 1 uses a hard condition

\[
\widehat p_{g'}\le \texttt{gate\_max\_p},
\]

with a tuned threshold. That is a reasonable method, but not a fully derived gate.

Use one of two honest framings:

### Framing A — saturation-motivated threshold

> “The endpoint zero predicts that relabels to saturated destinations should be rejected. We instantiate that prediction with a tunable pass-rate threshold.”

### Framing B — utility-based admission

Use an acceptance score proportional to the normalized mass itself:

\[
a(g')\propto \nu_N(\widehat p_{g'}),
\]

or a Lagrangian threshold

\[
\text{admit if }\nu_N(\widehat p_{g'})\ge\lambda.
\]

Then \(\lambda\) is explicitly the mean-versus-coverage tradeoff parameter. This is more algebraically faithful than a raw upper pass-rate threshold.

Either way, do not claim “no new hyperparameters.” The defensible claim is:

> “No hand-set difficulty target or band width is needed; the method still has concentration, replay-floor, posterior-decay, dose, and gate-strength parameters.”

---

## 3.12 “Coverage” is defensible; “diversity premium” is not the cleanest term

Average pass@k measures the fraction/probability of tasks solved at least once within \(k\) samples. It is an excellent operational coverage metric.

However,

\[
\operatorname{pass@8}-\operatorname{pass@1}
\]

is not a pure measure of output diversity. It also changes when success probability is redistributed across tasks. Rename “diversity premium” to:

- **multi-sample coverage gain**;
- **sampling premium**;
- or **pass@k gain over pass@1**.

This supports the paper’s own argument that token entropy and task coverage are distinct.

---

## 3.13 Theory-to-code contract: make exact assumptions auditable

A paper centered on estimator algebra needs a short implementation-contract subsection. The exact theory assumes:

1. binary rewards;
2. i.i.d. rollouts within a group;
3. exact \(\epsilon=0\) normalization for \(K>0\);
4. one trajectory score \(S_i=\nabla\log\pi(z_i\mid x)\);
5. on-policy data for the requested task;
6. no importance-ratio clipping or stale-policy correction;
7. the stated standard-deviation convention for GRPO;
8. group statistics computed globally rather than per-device fragments.

The baseline’s LLM implementation includes a small denominator epsilon and token-level averaging over the minibatch. If your code follows the same structure, exact equivalence can be perturbed by:

- response-length weighting;
- total-token normalization;
- nonzero epsilon;
- gradient clipping;
- multiple optimizer minibatches;
- and prompt rewrite/off-policy destination data.

Add a table:

| theoretical assumption | code behavior | discrepancy | measured sensitivity |
|---|---|---|---|

For binary \(K\ge1\), you can set epsilon exactly to zero safely, because \(K/N\ge1/N\). Doing so would simplify the exactness story.

---

# 4. Empirical and statistical review

## 4.1 The strongest empirical claims

The most convincing results in the current draft are:

1. **Exact/control chains:** the estimator-derived profile is useful, the replay placebo explains much of recycling’s gain, and the true relabel direction carries an additional behavioral effect.
2. **Frontier-heavy pool:** pure allocation variants remain at zero while relabeling ignites learning; uniform+recycling matching teacher+recycling is a strong attribution control.
3. **Maze coverage sign separation:** selected MaxRL variants increase pass@8 while selected GRPO variants decrease it.
4. **Countdown sharpening:** recycling increases mean success while reducing pass@16 across three seeds.
5. **Boundary conditions:** task spread, shared parameters, a graded pool, and relabel compounding all matter.

These are enough for a good paper if the claims are scoped correctly.

## 4.2 What is not yet established

The present evidence does not yet firmly establish:

- a statistically powered curriculum × estimator interaction at LLM scale;
- a universal “curriculum is unsafe under GRPO” rule;
- that the gate works at its intended corrected setting across seeds;
- that recycling is the only possible way to obtain updates from all-fail groups;
- or that relabeled updates are generally exact destination ML gradients.

The writing should not treat these as settled.

---

## 4.3 Run a schedule-matched estimator experiment

Using the same teacher algorithm under two estimators is not the same as exposing them to the same realized curriculum. The teacher adapts to each policy’s evolving pass rates, so the sampled prompt sequences can diverge.

A decisive causal test is:

1. construct a fixed prompt schedule from an external pass-rate oracle or a frozen warmstart posterior;
2. replay the exact same prompt IDs, group sizes, and order under MaxRL and GRPO;
3. pair seeds and warmstarts;
4. compare pass@1 and pass@k trajectories.

A second test can restore adaptation and measure the full feedback loop. The fixed-schedule test isolates the estimator; the adaptive test measures the deployed system.

---

## 4.4 Treat seeds and warmstarts as clusters

The exact permutation \(p=0.0079\) appears to be the minimum one-sided value from assigning five of nine run labels. That value is only meaningful if the nine observations are exchangeable and independent under the null.

If several runs share:

- the same seed;
- a common warmstart checkpoint;
- the same generated evaluation samples;
- or closely related configurations;

then unrestricted run-label permutation is anti-conservative.

Recommended analysis:

- report every paired seed contrast;
- use blocked permutation within matched configurations;
- use a cluster bootstrap over warmstart/seed;
- call configuration-seed contrasts “contrasts,” not independent seeds;
- report the one-sided direction and its pre-registration explicitly.

With only three random seeds, raw paired plots are often more honest than very small p-values.

---

## 4.5 Explain the 5-versus-18 MaxRL run counts

Section 6.3 says five MaxRL runs enter the main coverage sign result. Section 6.4 says 18 MaxRL runs enter the level-resolved result. This may be legitimate, but the paper does not make the inclusion rule transparent.

A skeptical reviewer will ask:

- Are the 18 observations independent runs, configuration variants, checkpoints, or repeated measurements?
- Why do only five enter the main permutation?
- Were the eligible arms declared before looking at coverage?
- Are multiple variants from one seed being counted separately?

The run registry should answer this without requiring the repository.

---

## 4.6 Fix the “6/6 seeds” phrasing

If there are three seeds and two teacher variants, say:

> “All six teacher-variant × seed paired contrasts were positive, grouped within three random seeds.”

Do not call them six seeds.

---

## 4.7 Add the full-control-variate all-fail baseline

This is the most important missing estimator control. Compare:

- practical MaxRL, drop \(K=0\);
- full-CV MaxRL, retain \(-N^{-1}\sum S_i\) at \(K=0\);
- practical MaxRL + recycling;
- perhaps an entropy/unlikelihood all-fail update.

Run it first on:

- the exact chain;
- the frontier-heavy \(p\le10^{-5}\) suite;
- and one maze setting.

This experiment will determine whether the “creation” channel is a property of relabeling specifically or of any mechanism that injects a nonzero all-fail update.

---

## 4.8 Rerun the corrected gate and make the primary tier unambiguous

Figure 7 is titled “Countdown, tier 1,” while the caption’s strongest gate claim concerns the frontier tier. At tier 1, the moderate gate appears largely to neutralize relabeling; at the frontier tier, it reportedly restores coverage while preserving about 60% of the mean gain.

Do not mix these stories in one panel/caption. Show two panels:

- saturated tier: gate blocks most relabels and erases the gain;
- frontier tier: gate preserves a useful mean/coverage compromise.

Then rerun the corrected setting across seeds.

---

## 4.9 Match generated tokens as well as wall-clock and optimizer steps

The maze teacher receives 22–35% more optimization steps per GPU-hour, partly because selected trajectories terminate earlier and constant groups skip backward computation. That is a valid systems benefit, but it confounds “better curriculum” with “cheaper sampled trajectories.”

Report three currencies:

1. wall-clock;
2. optimizer steps;
3. generated tokens/environment transitions.

If the teacher wins only at wall-clock, the contribution is throughput. If it also wins per generated token or per step, there is a learning-allocation effect. The draft already moves in this direction; make the decomposition a main figure rather than prose.

---

## 4.10 The LLM experiment should be framed as treatment-delivery diagnosis

The GSM8K posterior result is interesting:

- the posterior learns a real difficulty ordering;
- the prompt pool is too large for the posterior to become decisive within 50 steps;
- Thompson sampling remains near-uniform;
- replacement increases prompt repetition.

This is a valuable negative result about **posterior starvation**. Present it as such unless the decisive interaction replication is complete.

Before using it as a headline safety confirmation, finish:

- uniform-with-replacement control;
- stronger steering, likely through buckets or a kernel/stratified posterior;
- at least 3–5 paired seeds;
- matched prompt schedules for the estimator-only comparison;
- and a reconciled pass@k evaluation harness.

The current 360M/50-step cell is much smaller than the baseline MaxRL paper’s large-model evidence, so overplaying “LLM scale” invites an unfavorable comparison.

---

## 4.11 Do not put the 11× inference number in the abstract

The 11× result uses:

- a single checkpoint pair;
- an absolute coverage threshold of 0.25;
- log interpolation;
- and heterogeneous level-wise outcomes, including ties and one level where MaxRL is worse.

It is useful secondary evidence. It is not yet a stable headline. Keep the full curve and the “crossing near \(k=4\)” result, which is more informative and less threshold-sensitive.

---

## 4.12 Keep the boundary results, but turn them into explicit conditions

The negative suites are scientifically valuable. Consolidate them into a simple regime table:

| necessary condition | failure when absent | evidence |
|---|---|---|
| nonconstant group probability at budget | target-only zero update | sparse Gym |
| positive cross-task transfer | curriculum cannot unlock hard tasks | per-bin policy ablation |
| graded support over difficulty | no frontier to walk | Jugs |
| meaningful unlearnable region | focus loses to breadth | IsaacLab pilot |
| reusable relabeled skills | recycling cannot compound | one-shot streams |
| destination not saturated | recycling sharpens | Countdown |

This is a stronger synthesis than presenting each as a separate rung.

---

# 5. Writing and positioning review

## 5.1 The paper’s real one-sentence contribution

The current draft has several competing hooks. Use one:

> “MaxRL derives how a sampled prompt is weighted after it is drawn; we derive the finite-group coefficient profile that tells a curriculum whether drawing that prompt can produce contrast at all.”

Then add the complement:

> “Reallocation cannot manufacture contrast under the practical zero-constant-group estimator; relabeling can, but destination selection can concentrate the policy and reduce multi-sample coverage.”

Everything else should support those two sentences.

---

## 5.2 Be precise about the relationship to the baseline MaxRL paper

The baseline paper’s clean structure is:

1. RL is the \(T=1\) approximation to ML;
2. MaxRL forms a compute-indexed objective family;
3. a one-line normalization change gives an estimator;
4. population weight functions explain optimization behavior;
5. experiments escalate from exact ML to maze, fixed-data GSM8K, and larger models.

Your clean extension should be:

1. distinguish the baseline’s raw theorem from its practical centered algorithm;
2. derive the practical algorithm’s finite-group coefficient profile;
3. connect that profile exactly to its finite-\(N\) gradient weight;
4. use it before rollout generation as a sampling rule;
5. characterize what happens when all-fail groups are relabeled;
6. show estimator-conditioned coverage and destination-induced sharpening.

Do not write that the MaxRL theorem is wrong. Write:

> “Tajwar et al.’s Theorem 2 is exact for the raw estimator in Eq. (9). Their practical Algorithm 1 additionally zeroes the all-fail control-variate term; that implementation corresponds to \(T=N-1\).”

That wording is technically precise and collegial.

---

## 5.3 Replace repeated slogans with one stable vocabulary

Use the following terms consistently:

| current phrase | recommended phrase |
|---|---|
| “learns only from successes” | “requires a success anchor; live groups push successes up and failures down” |
| “expected learning signal” | “expected coefficient mass” or “estimator activity” |
| “dead zone no sampler can reach” | “operationally dead regime under the fixed zero-constant-group estimator” |
| “recycling is the only channel” | “recycling is the signal-creation mechanism studied here” |
| “identical curricula” | “the same curriculum rule,” unless schedules are literally matched |
| “safe” | “coverage-stable” or “does not reduce held-out pass@k” |
| “diversity premium” | “multi-sample coverage gain” |
| “oracle ceiling” | “one-step mass oracle” or “empirical allocation benchmark” |
| “exact hindsight gradient” | “verified destination semi-gradient; exact under conditional-law matching” |
| “environment-agnostic” | “environment-independent core with domain-specific adapters” |

The paper should use “The estimator decides” as a title and perhaps once in the introduction. Repeating it in every section makes the prose feel promotional rather than cumulative.

---

## 5.4 The current abstract is too dense and overclaims the interaction

The abstract currently contains:

- the mass theorem;
- three regimes;
- curriculum limits;
- recycler creation;
- GRPO interaction;
- maze statistics;
- LLM replication;
- sharpening;
- the gate;
- a boundary suite;
- and a practitioner diagnosis.

It reads like a compressed conclusion. Reduce it to four moves: problem, theorem, method, two decisive results plus boundary.

### Suggested conservative abstract

> Difficulty curricula and hindsight relabeling are often treated as objective-agnostic modules for reinforcement learning with verifiable binary rewards. We show that their effect depends on the group estimator underneath. For the practical centered MaxRL estimator that zeroes constant-reward groups, a prompt with pass rate \(p\) and \(N\) rollouts has expected coefficient mass
> \[
> M_N(p)=2\bigl[1-(1-p)^N-p\bigr].
> \]
> Moreover, its expected update factorizes as \(\mathbb E[\widehat g\mid x]=M_N(p)[\mu_+(x)-\mu_-(x)]/2\), so this curve is exactly the estimator-controlled scalar multiplying the task’s success-versus-failure score separation. The profile has a unique interior maximum and vanishes at \(p=0\) and \(p=1\), motivating a prompt sampler that tracks the moving frontier without hand-setting a difficulty target. The same analysis exposes a limit: under this practical estimator, prompt reallocation cannot create an update from an all-fail group. When an exact relabel map exists, hindsight can turn such groups into verified training examples, but the resulting destination data are generally off-policy and can concentrate training on already-reachable outcomes. Across maze curriculum variants, MaxRL runs increase pass@8 while GRPO runs decrease it. On Countdown arithmetic, relabeling raises mean success but lowers pass@16 across three seeds; a saturation-aware destination gate restores frontier coverage while retaining part of the mean gain. A boundary suite shows that these benefits require a graded task pool and transferable shared parameters.

After the corrected gate and powered interaction runs, the abstract can become more assertive.

---

## 5.5 Suggested introduction opening

The strongest opening is the event-level fact, not the list of curriculum papers:

> Post-training with binary verifiers is constrained by a simple event: a group-relative update needs reward contrast. If all \(N\) rollouts fail or all \(N\) succeed, the practical centered estimator is silent. This is not visible in population weight functions, which describe how a prompt is weighted after averaging over sampling. A curriculum acts one step earlier, deciding which prompts receive the rollout budget that makes those weights observable at all.

Then introduce MaxRL:

> MaxRL shows that likelihood-style objectives emphasize hard prompts after they are sampled. We ask the complementary question: given a finite group budget, which prompts can emit a usable contrast before the rollout is spent?

This gives the reader the paper’s difference from MaxRL immediately.

---

## 5.6 Shorter contribution list

Use four one-sentence contributions:

1. **Finite-group theory.** For the practical centered MaxRL estimator, we derive the exact coefficient-mass profile and its factorization into the finite-\(N\) task weight, clarifying that the shipped drop-all-fail variant corresponds to \(T=N-1\).
2. **Allocation and recycling.** We use the normalized profile as a prompt-sampling utility and characterize hindsight relabeling as a verified but generally off-policy destination semi-gradient.
3. **Coverage effects.** Across controlled and neural testbeds, coverage behavior separates by estimator, and relabeling can trade pass@k coverage for mean accuracy.
4. **Admission and boundaries.** A destination-saturation gate traces a mean-versus-coverage trade, while negative suites identify the required conditions: a graded pool, shared transfer, and reusable relabeled skills.

Avoid putting every numerical result in the contribution list.

---

## 5.7 Recommended main-paper structure

The current nine-rung ladder is too broad for the main narrative. A cleaner structure is:

### 1. Introduction

One problem, one theorem preview, three empirical claims.

### 2. From MaxRL’s population weight to finite-group activity

- setup and notation;
- raw, full-CV, and practical estimators;
- practical \(T=N-1\) result;
- general coefficient-mass factorization.

### 3. Estimator-derived allocation

- \(\nu_N\), peak, tails;
- RLOO/GRPO finite-group comparison;
- teacher distribution;
- scope: activity is not transfer.

### 4. Failure recycling and destination admission

- exact-verifier and rewrite contracts;
- source/destination conditional-law theorem;
- occupancy-bias explanation;
- gate.

### 5. Experiments

#### 5.1 Exact chains and frontier-heavy attribution

Utility, full-CV baseline, placebo decomposition, allocation versus creation.

#### 5.2 Matched maze estimator study

Same/frozen schedules, coverage sign, level/time mechanism, wall-clock/token/step decomposition.

#### 5.3 Countdown sharpening and corrected gate

Mean-versus-coverage trade, destination telemetry, corrected multi-seed sweep.

#### 5.4 Boundary conditions

One compact table for Gym/per-bin, Jugs, IsaacLab, and one-shot streams.

### 6. Related work and limitations

Move preliminary GSM8K, the 11× threshold calculation, detailed classic-control curves, and most audit history to the appendix unless the decisive replications finish.

---

## 5.8 Figure recommendations

### Figure 1 — keep, but correct and simplify

Current strengths:

- excellent central intuition;
- direct comparison among estimators;
- clear peak movement with \(N\).

Changes:

- fix the \(u_N\)/factor-of-two notation;
- replace “learning signal” with “normalized coefficient mass”;
- remove “MaxRL releases mastered prompts fastest”;
- use a logit x-axis or hard-tail inset so both endpoints are visible;
- state “practical MaxRL, \(N=16\), objective order \(T=15\).”

### Figure 2 — keep, but make relabel mechanics explicit

The diagram should show:

- source prompt \(x\);
- destination \(g'\);
- rewritten prompt/trajectory;
- re-evaluated destination success count \(K'\), not the original \(K=0\);
- that the destination score is evaluated under \(\pi(\cdot\mid g')\) although data came from \(x\);
- the gate using destination pass-rate evidence.

Right now “relabel: verified success of \(g'\)” can be read as making every trajectory successful, which would give zero centered weights if \(K'=N\).

### Figure 3 — remove from the main paper

It repeats Figure 1 and Figure 2 while making the “no sampler can” claim too absolute.

### Figure 4 — do not give the preliminary GSM8K panel equal visual status

The chain, frontier, and maze panels are useful. Move the single-/two-seed GSM8K panel to the appendix until the treatment is strong and replicated.

### Figure 5 — strongest empirical figure

Keep it, but:

- rename “diversity premium”;
- show paired seed/warmstart units;
- use cluster-aware uncertainty;
- clearly define why \(n=18\) MaxRL observations are valid.

### Figure 7 — split saturated and frontier tiers

Do not title a tier-1 figure and place the main frontier-tier claim only in the caption. Show both regimes explicitly.

### Figures 8 and 9 — appendix candidates

The classic-control result and single-checkpoint inference threshold are useful supporting evidence but not central enough for the current main-paper budget.

---

## 5.9 Sentence-level edits

### Current

> “The estimator learns only from successes.”

### Better

> “The practical estimator is success-gated: a group contributes only after at least one success appears; within a live group, successes receive positive coefficients and failures negative coefficients.”

---

### Current

> “The two zeros partition difficulty into a dead zone, a learnable band, and a mastered tail.”

### Better

> “The endpoint zeros and the unique interior maximum define three operational regimes: negligible contrast at the deployed budget, a frontier of high estimator activity, and a saturated tail with little remaining contrast.”

---

### Current

> “Recycling is the only channel into the dead zone.”

### Better

> “With the practical zero-constant-group estimator held fixed, reallocation cannot manufacture contrast in an all-fail group; relabeling is the signal-creation mechanism we study.”

---

### Current

> “The same curriculum degrades GRPO.”

### Better before the interaction experiment

> “Coverage behavior separates by estimator across curriculum variants; whether the teacher amplifies GRPO’s decline is an interaction question tested only suggestively at present.”

---

### Current

> “A failed trajectory is a verified success for the goal it actually reached.”

### Better

> “When the environment provides an exact relabel map, a failed source trajectory can be converted into a semantically valid example for an achieved destination, although the resulting destination data are generally off-policy.”

---

### Current

> “Perfect difficulty information is worth almost nothing over the cheap posterior.”

### Better

> “The true-pass-rate oracle substantially improves allocation over the posterior-only teacher, while posterior-plus-recycling reaches a similar empirical level; recycling partially compensates for allocation error.”

---

### Current

> “The bug, favorably for the method, sampled its middle.”

### Better

> “The bug produced an intermediate effective gate strength, which is reported as an unintended operating point; corrected multi-seed sweeps are required for the final claim.”

---

# 6. What to learn from the MaxRL baseline’s writing

## 6.1 What the baseline does especially well

### It separates objective, estimator, and implementation

The baseline first derives the ML/pass@k objective family, then introduces the raw estimator, then adds a control variate, then presents the practical algorithm. Your paper should preserve that separation rather than referring to all of them as “the MaxRL estimator.”

### It gives one memorable algebraic move

The baseline’s central implementation story is one normalization change: divide by \(K\), not \(N\). Your equally memorable move can be:

> “After practical centering and zeroing constant groups, the expected normalized coefficient mass is \(\operatorname{pass@}N-\operatorname{pass@}1\).”

Then immediately connect it to the gradient factorization.

### It uses the weight-function view to explain behavior

Your paper’s finite-group view is a natural complement. Make the distinction explicit:

- population weight: what happens after averaging over all groups;
- finite-group activity: how likely and how strongly a realized group can express that weight;
- curriculum: which prompts receive groups in the first place.

### It uses a compact experimental escalation

The baseline has a controlled exact-ML setting, data-rich maze, data-scarce GSM8K, and large-model reasoning. Your current nine-rung ladder is scientifically rich but narratively expensive. Three decisive experiments plus one boundary table will be stronger.

## 6.2 Where your paper can improve on the baseline

The baseline leaves several openings that your paper can address directly:

1. It does not distinguish the \(T=N\) raw theorem from the \(T=N-1\) practical drop-all-fail algorithm.
2. Its population GRPO curve does not describe finite-group constant-reward events.
3. It studies objective weighting after prompt sampling, not adaptive prompt allocation.
4. It conjectures that GRPO’s easy-tail inversion may drive sharpening but does not run identical data schedules across estimators.
5. It does not study exact-verifier hindsight relabeling or destination-induced pass@k loss.

These are enough. The paper does not need to claim that every curriculum and every recycler is governed completely by one scalar.

---

# 7. Recommended decisive experiments before v1.0

## Must run

1. **Practical MaxRL versus full-control-variate MaxRL** on chain, frontier-heavy pool, and one maze configuration.
2. **Schedule-matched MaxRL versus GRPO** with paired warmstarts and the exact same prompt sequence.
3. **Corrected gate sweep across 3–5 seeds** with destination-pass-rate telemetry.
4. **Run-registry and cluster-aware reanalysis** of all maze p-values.
5. **Uniform-with-replacement GSM8K control** before assigning the deficit to curriculum steering.

## High value

6. Generated-token-matched maze analysis.
7. Destination-law shift diagnostic for hindsight, such as a two-sample classifier or MMD between fresh destination rollouts and relabeled destination trajectories.
8. Cross-fit relabel destination selection as a bias-reduction ablation.
9. A calibrated “frontier support” diagnostic for the Jugs boundary, such as effective support of normalized \(\nu_N(p_x)\) across strata.
10. A shared-parameter transfer diagnostic showing that easier-bin updates actually improve harder-bin pass rates.

---

# 8. A compact reviewer-style verdict

## Strengths

- A coherent, estimator-first explanation of curriculum allocation.
- A correct and useful implementation-level refinement of practical MaxRL, if scoped carefully.
- An exact closed-form coefficient profile with a strong connection to learnability curricula.
- Unusually thorough negative results and attribution controls.
- A compelling new empirical phenomenon: relabeling can improve mean accuracy while reducing multi-sample coverage.
- Strong mechanistic instincts: requested-only posterior updates, goal-conditioning rewrite, dose controls, step-matched analysis, and boundary suites.

## Main weaknesses in the current version

- The central quantity is repeatedly mislabeled as the full learning signal.
- The dead-zone impossibility is not estimator-universal and omits the baseline paper’s full control-variate alternative.
- The headline curriculum × estimator interaction is not yet conclusively identified.
- Hindsight exactness is overstated relative to source/destination distribution shift.
- The primary gate experiment contains a disclosed implementation bug.
- Statistical independence and run inclusion are not sufficiently transparent.
- The paper is overstuffed, and preliminary experiments receive too much headline weight.
- The anonymous manuscript currently exposes the author identity.

## Bottom line

The paper is closest to a strong submission when it stops trying to prove that one scalar decides every aspect of curriculum learning and instead makes a narrower, deeper claim:

> **The practical group estimator defines an exact finite-sample activity profile. That profile constrains allocation, but transfer remains task-dependent; relabeling can create contrast beyond allocation, while its destination distribution can sharpen the policy.**

That claim is mathematically defensible, experimentally supported, and meaningfully extends MaxRL rather than merely adding a sampler to it.

---

# 9. Final pre-submission checklist

- [ ] Anonymous artifact and PDF are genuinely anonymous.
- [ ] \(A_N\) versus \(\nu_N=A_N/2\) notation is consistent everywhere.
- [ ] Theorem 2 of MaxRL is distinguished from practical Algorithm 1.
- [ ] Full-control-variate all-fail baseline is included or the omission is explicitly justified.
- [ ] “Learning signal” is replaced by coefficient mass/activity, with the exact factorization theorem added.
- [ ] “Dead zone” is operationally budget-indexed and scoped to the fixed estimator.
- [ ] “Width is derived” is replaced or mathematically defined.
- [ ] RLOO mastered-tail comparison is corrected.
- [ ] Water-filling is called a one-step mass oracle, not a universal ceiling.
- [ ] Oracle-versus-posterior inference is corrected.
- [ ] Hindsight is described as off-policy at the destination unless conditional laws match.
- [ ] Gate claims use corrected multi-seed runs.
- [ ] Maze statistics use seed/warmstart clusters and a public run registry.
- [ ] Main effect and interaction claims are separated.
- [ ] GSM8K replacement confound is controlled or the result is moved to preliminary evidence.
- [ ] Wall-clock, optimizer-step, and generated-token currencies are all reported.
- [ ] “Diversity premium” is renamed.
- [ ] Figure 7 separates saturated and frontier tiers.
- [ ] Main paper contains no more than three decisive experiment stories plus one boundary synthesis.
- [ ] Every load-bearing number is available in the paper or appendix, not only external Markdown files.
