# Formal results for curriculum-MaxRL

Self-contained statements and proofs of the identities the algorithm rests
on. Every finite-N coefficient and population-gradient identity is also
verified by exhaustive enumeration in `test_math_claims.py`. Notation: a
prompt has pass rate `p ∈ (0,1)`; a group is `N` i.i.d. rollouts with
`K ~ Bin(N, p)` successes.

The repository uses the paper's **practical Algorithm 1** coefficients:

```
w_i = r_i/K − 1/N   for K ≥ 1,      w_i = 0 for all i when K = 0.
```

This is not identical to the unbiased Eq. (10): Eq. (10) keeps the
unconditional `−1/N` score control when K=0, while Algorithm 1 explicitly
drops both terms. The distinction changes the expected objective by one
order and is formalized first.

Here `Σᵢ|wᵢ|` is coefficient L1 mass. It is an estimator-side surrogate for
available update magnitude, not the norm of the policy gradient: score-vector
norms and cancellation are deliberately outside these propositions.

---

## Proposition 0 (Exact objective order of the two variance-reduced variants)

Let `g = ∇p`, `w_T(p) = (1−(1−p)^T)/p`, and let `V_N = N⁻¹Σ_i S_i` be the
unconditional average score.

**Claim.**

1. The success-only estimator in paper Eq. (9), and Eq. (10) with `V_N`
   retained for every group including K=0, have expectation `w_N(p)g`.
2. Practical Algorithm 1, which subtracts `V_N` only when K>0 and returns zero
   at K=0, has expectation `w_{N−1}(p)g`. At N=1 it is identically zero.

**Proof.** The paper's Theorem 2 gives `E[ĝ_N] = w_N(p)g` for Eq. (9).
Subtracting `V_N` unconditionally changes nothing because `E[V_N]=0`.
For the practical variant,

```
E[1{K>0}V_N]
  = −P(K=0) E[V_N | K=0]
  = −(1−p)^N (−g/(1−p))
  = (1−p)^(N−1) g.
```

Therefore its expectation is

```
[w_N(p) − (1−p)^(N−1)]g
  = [(1−(1−p)^(N−1))/p]g
  = w_(N−1)(p)g.                                             ∎
```

**Interpretation.** The base paper's order-N unbiasedness theorem applies to
Eq. (9) and full Eq. (10), not to the dropped-all-fail implementation used by
Algorithm 1 and this repository. The practical variant remains a
likelihood-shaped objective, but its exact truncation order is N−1.

---

## Proposition 1 (Exact coefficient mass of practical Algorithm 1)

**Claim.** `E[Σᵢ|wᵢ|] = 2·(pass@N(p) − pass@1(p)) = 2·((1−(1−p)ᴺ) − p).`

**Proof.** Condition on K. For K ≥ 1 there are K successes with weight
`1/K − 1/N > 0` (since K ≤ N) and N−K failures with weight `−1/N`, so

```
Σ|w| = K(1/K − 1/N) + (N−K)/N = 1 − K/N + 1 − K/N = 2(1 − K/N).
```

For K = 0 the group is dropped: Σ|w| = 0. Therefore

```
E[Σ|w|] = 2·E[(1 − K/N)·1{K≥1}]
        = 2·(P(K≥1) − E[K·1{K≥1}]/N)
        = 2·(P(K≥1) − E[K]/N)          (K·1{K≥1} = K a.s.)
        = 2·((1−(1−p)ᴺ) − p).          ∎
```

**Interpretation.** The expected coefficient mass assigned to a prompt equals
twice the probability it is *solvable within N attempts but not within one*.
This is the estimator's own zone-of-proximal-development functional. It does
not by itself determine the resulting gradient norm or learning progress.

The identity is specifically for practical Algorithm 1. Full Eq. (10) assigns
coefficient mass 1 to an all-fail group, so its expected mass is
`(1−p)^N + 2(pass@N−p)` and is not the same ZPD functional.

**Sampling-policy caveat.** Proposition 1 supplies a utility and ordering, not
a unique sampling distribution. For fixed known utilities uᵢ and task
probabilities qᵢ, expected normalized mass per sampled group is
`Σᵢ qᵢuᵢ`; its unconstrained maximizer puts all mass on `argmax u`. With a
uniform floor ρ, the one-step maximizer is
`q = ρ/m + (1−ρ)·δ_argmax`. Sampling proportional to `u^γ` is instead a
soft exploration, coverage, and anti-forgetting design choice whose γ must be
validated empirically.

---

## Proposition 2 (Peak of the utility)

**Claim.** `u(p) = (1−(1−p)ᴺ) − p` on [0,1] is strictly concave with unique
maximizer `p* = 1 − N^(−1/(N−1))`, and `p* = ln N / N + O((ln N / N)²)`.

**Proof.** `u′(p) = N(1−p)^{N−1} − 1`, `u″(p) = −N(N−1)(1−p)^{N−2} < 0` on
(0,1), so u is strictly concave; setting u′ = 0 gives
`(1−p*)^{N−1} = 1/N`, i.e. `p* = 1 − N^{−1/(N−1)}`. For the asymptotic,
`N^{−1/(N−1)} = exp(−ln N/(N−1)) = 1 − ln N/(N−1) + O((ln N/N)²)`, hence
`p* = ln N/(N−1) + O((ln N/N)²) ≈ ln N / N`. ∎

**Interpretation.** Doubling the group size moves the optimal difficulty
band harder by a factor ≈ `ln 2N / ln N · N/(2N) ≈ 1/2` in pass-rate terms:
the curriculum and the objective share one compute knob, with an exact rate.

---

## Proposition 3 (Greedy rollout allocation is optimal)

**Claim.** For fixed prompt set with pass rates p₁..p_m and budget
B = Σᵢ Nᵢ, total mass `M = Σᵢ u_{Nᵢ}(pᵢ)` is maximized by greedy
water-filling on the marginal `Δᵢ(N) = pᵢ(1−pᵢ)ᴺ`.

**Proof.** The increment of one more rollout on prompt i is

```
u_{N+1}(pᵢ) − u_N(pᵢ) = (1−pᵢ)ᴺ − (1−pᵢ)^{N+1} = pᵢ(1−pᵢ)ᴺ > 0,
```

which is strictly decreasing in N. So M is a sum of separable concave
functions of the integer allocation; for such problems the greedy algorithm
(repeatedly assign the next unit to the largest current marginal) is exact —
the standard exchange argument: swapping any unit from a prompt with a
smaller marginal to one with a larger marginal cannot decrease M, and
marginals only shrink as units are added. ∎

**Interpretation.** `pᵢ(1−pᵢ)ᴺ = P(rollout N+1 is the group's first
success)`. Optimal compute allocation = "give the next rollout to the prompt
where it is most likely to flip a dead group live."

---

## Proposition 4 (RLOO advantage mass = learnability)

**Claim.** For RLOO weights `wᵢ = (rᵢ − r̄₋ᵢ)/N` (leave-one-out baseline),
`E[Σ|w|] = 2p(1−p)` exactly for every `N ≥ 2`.

**Proof.** With K successes: a success has weight `(1 − (K−1)/(N−1))/N =
(N−K)/(N(N−1))` and a failure `−K/(N(N−1))` in magnitude. Summing:

```
Σ|w| = K(N−K)/(N(N−1)) + (N−K)K/(N(N−1)) = 2K(N−K)/(N(N−1)).
E[K(N−K)] = N·E[K] − E[K²] = Np − (Np(1−p) + N²p²) = N(N−1)p(1−p).
⇒ E[Σ|w|] = 2p(1−p).                                        ∎
```

**Interpretation.** SFL's "learnability" curriculum objective p(1−p)
(Rutherford et al. 2024) *is* the advantage mass of the RLOO estimator. The
curriculum literature and the estimator algebra converge on the same
functional from opposite directions — and our Prop. 1 shows MaxRL
generalizes it to a compute-indexed family whose N=2 member is learnability
(the N=1 member is identically zero).

---

## Proposition 5 (Coefficient-mass ordering on the frontier)

**Claim.** For all N ≥ 2 and p ∈ [0,1],
`u_MaxRL(p) ≥ u_RLOO(p)` with ratio `→ N−1` as p → 0.

**Proof.** `u_MaxRL(p)/2 = (1−(1−p)ᴺ) − p = Σ_{k=1}^{N} C(N,k)pᵏ(−1)^{k+1}... `
simpler: for small p, `1−(1−p)ᴺ = Np − C(N,2)p² + O(p³)`, so
`u_MaxRL/2 = (N−1)p + O(p²)` while `u_RLOO/2 = p(1−p) = p + O(p²)`.
Ratio → N−1 ≈ N. Both vanish at p ∈ {0,1}; the MaxRL mass is at least as
large since `(1−(1−p)ᴺ) − p ≥ p(1−p)` ⇔ `1−(1−p)ᴺ ≥ 2p−p²
= 1−(1−p)²`, true for N ≥ 2. Equality holds throughout for N=2; for N>2
the inequality is strict on p ∈ (0,1). ∎

**Interpretation.** On frontier prompts (p small but nonzero) MaxRL's
estimator concentrates `(N−1)` times more expected coefficient mass than
RLOO's. This is a finite-sample mechanism consistent with the paper's
"MaxRL extracts more learning signal" observation (their Fig. 7), but the
mass ordering alone is not a proof of larger gradient norm or policy gain.

---

## Proposition 6 (Hindsight relabeling: bias characterization)

Setting: dead group (K = 0) on task τ; each rollout i has an achieved
prefix/goal g(zᵢ); relabel to goal g* achieved by at least one rollout, with
r̃ᵢ = 1{g(zᵢ) reaches g*}, and apply the MaxRL weights w̃ to the truncated
trajectories.

Let `Q_g` be the actual **joint group law** after conditioning on an original
dead group and selecting the achieved goal g from that same group. Let
`P_g^N` be the joint law of N fresh on-policy rollouts requested directly on
g, and let `G_g(z_1..z_N)` be the practical Algorithm 1 update after the
required conditioning rewrite.

**Claim.**

```
bias(g) = E_Qg[G_g] − E_Pg^N[G_g].
```

It is zero if the two joint group laws match. If `||G_g|| ≤ M`, then
`||bias(g)|| ≤ 2M·TV(Q_g, P_g^N)` under the convention
`TV(P,Q)=sup_A|P(A)−Q(A)|`.

**Proof.** The first identity is the definition of bias relative to fresh
on-policy groups. The total-variation bound is the standard bounded-function
expectation inequality. Under `P_g^N`, Proposition 0 says the repository's
practical weights estimate the order-(N−1) MaxRL gradient, not the exact ML
gradient. ∎

**Interpretation.** Verifier-correct relabels and conditioning rewrites are
necessary, but they do not establish unbiasedness: choosing g from the same
dead group couples the samples and guarantees at least one relabeled success,
so equality of per-trajectory conditionals is insufficient. V1 measures a
stronger empirical fact on the skill chain: per-group cosine matches fresh
groups and the mean update has cosine 1.000 to the ML direction. It does not
test update magnitude or prove equality of the joint laws. Hindsight should
therefore be described as an empirically aligned HER-style auxiliary update,
with coverage drift and conditional-law mismatch as the two risks.

---

## Proposition 7 (Explore/exploit: what the Thompson teacher pays)

The teacher faces a nonstationary bandit: arm = prompt, payoff = advantage
mass `u_N(p_t(a))` where `p_t` moves as the student learns. Two structural
facts shape the design:

1. **Bounded regret-per-step against a static oracle.** With Thompson
   sampling on a Beta posterior with decay (effective sample size
   `ESS = 1/(1−γ)` at decay γ), the posterior tracks a drifting p with lag
   `O(1/ESS)`; the sampling distribution differs from the oracle's by
   `O(|û−u|/Σu)`. V2 measures this gap end-to-end (oracle vs Thompson AUC).
2. **The floor is not a tuning nicety but a lower bound on information.**
   With floor f, every prompt is sampled at rate ≥ f/m, so the posterior's
   staleness is bounded and *mastered-then-forgotten* prompts are re-detected
   within `O(m/f)` groups. Setting f = 0 makes forgetting undetectable
   (a prompt with p̂ ≈ 1, u ≈ 0 is never revisited) — V3's f = 0 arm tests
   exactly this failure.

These are design constraints rather than theorems; V2/V3 quantify them.
