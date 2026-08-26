# Porting count-law activity to control: what the measurement says

Written 2026-08-19 (ICLR abstract T-30d, paper T-37d) in response to the
proposal to carry coefficient activity into robotics/continuous control and
compare against PLR/minimax. I agree with the direction and disagree with the
substrate. The reason is measured, not argued.

## 1 · The proposed architecture is already under test, and must not be pre-empted

The proposal's core is a gated priority,

    Priority(x) = mean_t |delta_t^GAE| · [u_N(p_x)]^gamma,

with activity used as a gate on a dense signal rather than as a replacement.
That is exactly the arm we already built (`overlay_v6_gated`, `frontier_mode=
gate`, score `max(MaxMC,0)·E[u_N(p)|D]`) and exactly what the **currently
running** full-budget confirmatory campaign measures: 20 runs, 10 paired seeds,
frozen analyzer, SESOI +0.02, verdict table fixed in advance. Development gave
−.039 to upstream at 1/5 seeds; the campaign is at 10/20.

Launching a robotics version of the same architecture before that result lands
would spend compute on a question we are days from answering under a frozen
rule. **Wait for it.**

## 2 · The count-law upgrade is not estimable in PLR — measured

The theory's newest and strongest piece is that activity is a functional of the
success-count law, `A_E(z) = Σ_k P(K=k|z) M_E(k)`, and that scoring by a mean
pass rate is a lossy plug-in. Estimating that requires *groups*: repeated
outcomes on the same unit that can be assembled into counts.

PLR cannot supply them at this operating point. From the running campaign's
training telemetry (a registered secondary; no endpoint was read):

| quantity | value |
|---|---|
| `plr/weighted_frontier_trials` (observations accumulated per level) | median **4.20**, range 4.12–4.37 across 6 runs |
| `plr/frontier_n_rollouts` (target group size) | 8 |
| `plr/frontier_group_size_match` | **0 in every run** |

A level accumulates about **four** Bernoulli observations against the eight
needed to complete a single group — so not one group ever closes, let alone the
many needed to estimate `P(K|z)`. This is the paper's "bandwidth, not shape"
diagnosis, now quantified: the constraint is 4.2 trials per level.

Worse for the theory, the observations that *are* accumulated arrive at
**different times under a drifting policy**, so even pooled they are not a
conditionally-i.i.d. group. PLR's replay buffer gives revisitability but not
simultaneity.

## 3 · The substrate insight, which inverts the proposal

The proposal suggests GPU-parallel simulators (Isaac Lab, Brax) as the venue and
PLR as the framework. The measurement says these pull in opposite directions:

> **A GPU-parallel simulator is precisely what makes a group formable — N
> rollouts of the *same* environment instance under the *same* policy, which is
> a genuine conditionally-i.i.d. group. PLR is precisely what makes it
> unformable, because it accumulates across visits under a policy that has
> moved in between.**

So the control study that tests the theory's newest claim should use the
parallel simulator to *form groups*, and should not route the score through a
replay buffer to do it.

## 4 · Where the aggregation actually lives in robotics

The proposal says never aggregate across instances. In robotics that advice is
close to unusable: domain randomization *is* aggregation, and it is usually the
point. The sharper claim is the corollary's:

> Aggregate if you want, but then score the count law, because the penalty for
> scoring the mean is exactly `2[Pr(K=0|z) − (1−p̄_z)^N]`.

And the counterexample has an exact robotics instantiation. Take a friction
range as the scored unit, where half the sampled coefficients admit a stable
gait and half do not. Its mean success rate is 1/2; every group is unanimous;
its realized activity is **zero**, while any `f(p̄)` scores it at the frontier
maximum. That is Level B of Fig. 1, in a control setting, and it is the regime
domain randomization produces by default.

## 5 · The experiment that would earn the claim

Substrate: a GPU-parallel sim with a binary terminal predicate (Brax
Ant/Humanoid over rough terrain with a height/distance threshold, or an
Isaac Lab peg-in-hole with a pose tolerance). N parallel rollouts per
environment instance at fixed policy — a real group.

Scored unit is a **randomization cell** (a box in friction × mass × terrain
amplitude), deliberately coarse so the aggregation is real. Arms, all sharing
substrate, estimator, budget, warmstart and seeds:

1. domain randomization (uniform over cells);
2. plug-in activity `u_N(p̂_z)`;
3. count-law activity `q̂_z − p̂_z`;
4. PLR/ACCEL with its native regret score, as the external baseline.

Primary: paired time-integrated success on a held-out cell distribution.
Secondary: measured per-cell gap `2[P̂(K=0|z) − (1−p̄_z)^N]`, and a
treatment-delivery gate on the sampler TV distance between arms 2 and 3 — the
same gate the P0 registration uses, for the same reason.

The decisive contrast is 2 vs 3, which isolates the count law. Arm 4 says
whether any of it competes with a tuned regret method, and should be read with
the AMaze result in hand: a one-bit terminal statistic did not beat a
per-timestep critic there, and there is no reason to expect it to here.

## 6 · Scope call

This is next-paper work. The abstract deadline is T-23 days and the paper
deadline is T-30 days; the perimeter ends at the neural maze. P0 closed on
2026-08-26 with its preregistered supported verdict, and the charter's
discipline is still to stop expanding. Opening a robotics lane now would trade
the paper's main asset — that everything inside the perimeter is demonstrated
— for a gesture at a second domain.

Sequenced instead: let the AMaze confirmatory close the gated-priority question
under its frozen rule; preserve P0's bounded causal claim; and only then port to
control, where the parallel simulator makes the group structure the theory
needs actually available.
