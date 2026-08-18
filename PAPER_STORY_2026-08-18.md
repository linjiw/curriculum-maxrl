# The story we have not told — reframing the ICLR draft

Written 2026-08-18 after rereading the draft against
[zanette-labs.github.io/MaxRL](https://zanette-labs.github.io/MaxRL/). This is
a strategy note, not a result; nothing here changes any number.

## 1. Why MaxRL's page works and ours does not

MaxRL's page is one identity, one line of code, one figure, five takeaways:

> `∇ log p = Σ_k (1/k) ∇ pass@k`. RL optimizes the first term. Normalize by K
> instead of N and you get the rest.

Everything on the page hangs off that. The identity is stated in the first
screen; the weight-function plot makes it *visible*; the results are a Pareto
plot with a multiplier ("7.9×–19.2× more efficient"); every section ends in a
one-line takeaway a skimmer can carry away.

Our draft, reread cold, is an audit. The intro's first two paragraphs are
about how separating curricula from estimators is "fragile." The contribution
list is "derivation / controlled scope test / supporting measurement." The
abstract spends more words on what is *not* claimed than on what is. A
reviewer finishes it knowing we are careful and not knowing what we found.

Care is a virtue and every disclaimer in the draft is earned. But the
disclaimers are currently the *frame*, and the finding is inside it. It should
be the other way round.

## 2. The identity we never stated

We have exactly the kind of one-line identity MaxRL built its page on, and it
is not in the paper. It follows in two lines from things already proved:

MaxRL writes every objective's population gradient as
`∇J = E_x[ w(p) ∇p ]`, and their weight function at truncation `T` is
`w_T(p) = Σ_{k=1}^{T} (1-p)^{k-1}`.
Our Lemma 1 shows the deployed drop-all-fail estimator targets `T = N−1`.
Our Proposition 1 gives activity `u_N(p) = 1 − p − (1−p)^N`.

**Then, exactly:**

    u_N(p)  =  p(1−p) · w_{N−1}(p)

(verified numerically to 1e-15 for N = 2..64.) In words:

> **Coefficient activity is the learnability score `p(1−p)` — the variance of
> a single Bernoulli rollout — reweighted by MaxRL's own weight function at
> the truncation order the deployed estimator actually targets.**

This is the sentence the paper is missing. It does three things at once:

1. **It subsumes the prior art in one line.** ProCuRL / SFL / LILO's `p(1−p)`
   is the `N=2` slice *and* the `w=1` (pure REINFORCE) case. Every
   "learnability" curriculum in the literature is what you get by scoring
   tasks as if the learner were REINFORCE with one rollout. Our score is what
   you get when you score tasks for the estimator you actually deploy.
2. **It explains the peak.** `p(1−p)` peaks at ½; `w_{N−1}` is monotone
   decreasing in `p` and grows without bound as `p→0` for large `N`. Their
   product's peak, `1 − N^{−1/(N−1)} ≈ log N / N`, is where "the rollout is
   informative" meets "MaxRL still cares." That is a derivation, not a tuned
   number.
3. **It positions us as the curriculum half of MaxRL.** MaxRL asked "which
   *objective* should more compute buy?" and answered "closer to maximum
   likelihood." We ask "which *tasks* should that objective be fed?" and the
   answer is written in the same weight function. That is a genuine
   companion-paper position, not a derivative one.

## 3. What our results actually say, told forward

Reordered from strongest to bounding, in the voice a reader can carry:

**Takeaway 1 — the score is derived, not tuned.** One identity gives a
difficulty target for every group size, with a closed-form peak. `N=2`
recovers the literature. (Theory, exact.)

**Takeaway 2 — the shape beats the learnability slice, replicated.** In a
fixed Acrobot pool at deployed `N=16`, `u_16` beats `p(1−p)` by +.048 (frozen
primary, 15/20), replicated on two further platforms (+.032, +.031, both with
smaller p than the original). Digits shows the same shape effect at +.208
(23/24) under MaxRL and +.177 (24/24) under RLOO — the largest effect in the
paper. (One frozen primary, three concordant reads; the contract table.)

**Takeaway 3 — the shape helps; the peak location does not.** Holding the
estimator at `N=16` and sweeping the score exponent 2→128, performance rises
*past* 16 (argmax `u_64`, Spearman +.93). Harder-peaked scores keep winning at
this operating point. (Preregistered boundary. This is the honest headline of
the sweep and it is *interesting*: it says the value of the shape is in
its tail, which is exactly where `w_{N−1}` differs from `p(1−p)`.)

**Takeaway 4 — as a signal of its own, it is starved.** Dropped into robust
PLR on AMaze in place of MaxMC, activity does not beat upstream: one Bernoulli
per level visit cannot compete with a critic read at every timestep. Gating
MaxMC by activity recovers most of the gap (−.039, 1/5) — the shape helps a
richer signal but does not replace one. (Development negative, reported.)

**Takeaway 5 — the estimator belongs in the curriculum's evaluation.** Digits
rejects a universal estimator-to-sampler mapping; the maze factorial shows the
estimator ordering coverage under either sampler; Countdown shows a proxy
moving opposite to the raw outcome. Evaluate curricula with the estimator
they ship.

That is a story. Every sentence in it is already true in the draft; none of
them is currently the *first* sentence of anything.

## 4. Where to position it

**Not** as "a competitive curriculum method." We do not beat PLR on AMaze and
we say so; a method paper that loses its only named-baseline comparison is
below bar. That framing was already dropped in RESEARCH_PLAN §2.1 and the
results since have confirmed the drop.

**As** the first analysis of *what a curriculum score should be when the
learner is a modern group estimator* — MaxRL, GRPO, RLOO — with an exact,
derived answer, a controlled positive, two preregistered boundaries that
locate precisely where the derived answer stops helping, and a methodological
conclusion (evaluate with the estimator) that the field is currently
violating. Nearest neighbours: ProCuRL/SFL (learnability curricula), PLR/ACCEL
(regret curricula), MaxRL (the objective). We are the bridge between the
first and third; we are honest about losing to the second on its home turf
and we explain why in one sentence (signal, not shape).

ICLR rewards this shape of paper when the theory is exact and the negatives
are preregistered — both true here — *provided the story is legible*. Right
now it is not, and that is a writing problem, not an evidence problem.

## 5. Concrete changes, in priority order

1. **State the factorization** `u_N = p(1−p)·w_{N−1}` in the abstract, in
   the intro as the framing sentence, and as a corollary after Prop. 1 with a
   two-line proof. Cite MaxRL's weight-function view explicitly and say we are
   its curriculum counterpart. Cost: half a page. Value: the whole frame.
2. **Retitle.** "Rollout-Aware Coefficient Activity for Task Selection in
   Verifiable-Reward RL" describes a quantity. Candidates that describe a
   finding, in rough order of preference:
   - *Which Tasks Does the Estimator Make Active? Curriculum Scores for
     Group-Based Verifiable-Reward RL*
   - *Learnability, Reweighted: What a Curriculum Score Should Be Under MaxRL*
   - *The Estimator Decides: Rollout-Aware Task Selection in Verifiable-Reward RL*
   (the site already uses "The estimator decides" as its h1.)
3. **Rewrite the intro as the five takeaways above**, one paragraph each,
   *then* the scope paragraph. Move "fragile separation" from paragraph 1 to
   paragraph 3.
4. **One headline figure on page 1**, MaxRL-style, three panels: (a) `u_N`
   curves with `p(1−p)` as the `N=2` slice and the peak trajectory
   `log N / N` drawn; (b) the exponent-sweep dose–response with the deployed
   `N` marked — the visual for Takeaway 3; (c) the contract-table effects as
   a forest plot with the frozen primary bold. This replaces the current
   fig1, which is a weight-function plot with no result on it.
5. **End every results subsection in a boxed one-line takeaway** (we already
   have the sentences; box them).
6. **Site**: rebuild the top of `docs/index.html` around the same five
   takeaways with the same headline figure. The h1 is already right.

Items 1–3 fit in the current page budget by reclaiming the two "fragile
separation" paragraphs and compressing the contribution list. Item 4 costs
~0.4 page and pays for itself if fig1's current 0.84 width comes down; if not,
the maze factorial's remaining prose is the next thing to give.

## 6. What this does *not* fix, said plainly

The paper's positive evidence above 640 parameters is still MAZE-SCORE, and
MAZE-SCORE has still not run. The reframe makes the paper *legible*; it does
not make it *bigger*. If MAZE-SCORE lands as a second frozen leg at 1.26M
parameters, Takeaway 2 gains a neural row and the story is complete. If it
does not, the paper is a sharp small-scale theory-plus-boundaries paper —
which is a real paper, and a better one for being told forward.

The AMaze full-budget confirmatory (running, ~20:00 today) can only make
Takeaway 4 more precise; it cannot flip the frame.
