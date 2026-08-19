# Carrying count-law activity into PLR/minimax and into control: how

Written 2026-08-19, superseding sections 2–3 of `RESEARCH_NOTE_CONTROL_TRANSFER_2026-08-19.md`.
That note's substrate argument rested on two telemetry keys I have now read line by line, and
both mean something other than what the note said. The direction is reaffirmed; this note is
the *how*, with the honest constraints folded in as design constraints rather than as reasons
to stop.

## 1 · The one-line answer

**Set `n_eval = N` and score a coarse unit.** minimax already forms genuine
conditionally-i.i.d. groups in the training path — `BatchEnv._set_state`
(`batch_env.py:70-74`) replicates each PLR-sampled level `n_eval` times and vmaps a pure,
RNG-free `set_state` (`environment.py:129-136`) with parameters `stop_gradient`-frozen across
the whole scan — so group formation is a flag, not a patch. What is missing is the *estimator*
(`PLRBuffer` carries two scalars per level and destroys K) and, more importantly, the *unit*:
at an atomic maze the count law is provably identical to the plug-in, so an atomic minimax
experiment cannot test the theory at all.

## 2 · What the measurement actually said — the 4.2 figure was misread

Section 2 of the prior note reported "a level accumulates about four Bernoulli observations
against the eight needed" and "not one group ever closes". Both statements are wrong, and the
correction matters because the whole substrate inversion was built on them.

| key | prior reading | what it actually is |
|---|---|---|
| `plr/frontier_group_size_match` | runtime measurement of group closure | `jnp.asarray(self.frontier_n_rollouts == self.frontier_n_eval)` — a **static comparison of two constructor arguments** (`plr.py:560-561`). The campaign launched `--n_eval=1 --plr_frontier_n_rollouts=8`, so `8==1 → 0.0` by construction. |
| `plr/weighted_frontier_trials` | observations accumulated per level | `(plr_buffer.trial_counts*replay_dist).sum()` (`plr.py:563`) — a **replay-probability-weighted mean**, top-heavy at `--plr_temp=0.3`. Unweighted mean at the same checkpoint is `40954/4000 = 10.24`, 2.6× the weighted 3.94. |
| `plr/frontier_incomplete_group_count` | — | only incremented inside the strict branch (`plr.py:344-352`); with `require_n_eval_match=False` it is identically 0 and proves nothing. |

The decisive evidence is a run already on disk. `pilot-20260815-2031/arm-frontN8-s1001-u2500`
ran `--n_parallel=4 --n_eval=8 --plr_frontier_n_rollouts=8` under strict matching. I read all
100 logged rows through update 2482: `frontier_group_size_match = 1.0` on every row,
`frontier_incomplete_group_count = 0.0` on every row, `weighted_frontier_trials = 8.000001`
exactly on the first row. **Every group closed at exactly 8.** PLR's replay buffer was never
the mechanism supplying simultaneity — the `n_eval` axis was, and it is upstream, not a fork
feature (stock `minimax/envs/batch_env.py:72` does the same repeat).

## 3 · The minimax patch: what to set, what to add, what it costs

**Group formation: zero lines of code.**
`--n_parallel=4 --n_eval=8 --plr_frontier_n_rollouts=8 --plr_frontier_require_n_eval_match=True`.
The strict predicate `frontier_group_is_valid(trials, n_eval, ...)` returns `trials == n_eval`
(`plr_runner.py:40-44`) and already sets `ued_scores = -inf` on incomplete groups. Porting to
*stock* minimax additionally needs one character-level fix, `dr_runner.py:138`,
`x.at[:,:self.n_parallel]` → `x.at[:,:self.n_parallel*self.n_eval]`, or `n_eval>1` shape-errors
with any recurrent student.

**Estimator: two new float32 arrays, not a Dirichlet.** Under strict groups
`observed_trials ≡ N`, so `trial_counts = N·W` where `W = Σ wᵢ` is the decayed group count and
`success_counts = Σ wᵢKᵢ`. Only the zero-group mass is new:

- `zero_counts` `Z = Σ wᵢ·1{Kᵢ=0}` → `Â_MaxRL = 2(1 − N·Z/trial_counts − success_counts/trial_counts)`
- `sqsuccess_counts` `S₂ = Σ wᵢKᵢ²` (RLOO only) → `Â_RLOO = 2(N·S − S₂)/(W·N(N−1))`
- **`gap = 2[Z/W − (1−p̄)ᴺ]`** — the paper's corollary, in closed form, as a logged number.

Insert in the decay block at `plr.py:394-398`, branch at `plr.py:399-415`, mirror in the
runner's provisional score at `plr_runner.py:309-315` or buffer and runner disagree. `frontier_n`
is already `struct.field(pytree_node=False)` (`plr.py:43`), so nothing becomes traced. GRPO is
dropped from JAX: `√(k(N−k))` is not polynomial, needs the full `(buffer_size, N+1)` histogram,
and at `decay=0.7` the effective-sample ceiling is `prior_mass + 1/(1−decay) = 4.33` groups,
which cannot fill 33 bins at N=32.

**One correctness trap.** If `frontier_decay < 1` and a buffer slot receives *m>1* groups in one
update, `S ← d·S + Σφ(kⱼ)` is wrong; the exact form is `S ← dᵐ·S + Σⱼ d^(m−j)φ(kⱼ)` (measured
divergence 4.7e-2 at m=2, d=0.7). At the shipped `frontier_decay=1.0` this vanishes. Under a
coarse unit with few families, m>1 is the *modal* case, not an edge case — so either run
decay=1.0 or implement the rank-weighted form and unit-test it.

**Cost.** Group formation is free in compute and expensive in diversity. From the pilot's own
logs, matched within one campaign (so contention is controlled):

| arm | layout | s/update | GPU-h per 30k updates |
|---|---|---|---|
| `arm-maxmc32x1-s1001` | 32×1 | 0.4290 | 3.58 |
| `arm-maxmc4x8-s1001` | 4×8 | **0.3676** | 3.06 |
| `arm-frontN8-s1001` | 4×8, strict N=8 | 0.3235 | 2.70 |

Grouping is ~14% *faster* at matched slot count. But `DIAGNOSIS_2026-08-16.md` measured the
restructuring cost at **−0.207 held-out return** (0/5 seeds, p=.0625), and group-matched MaxMC
pays it too, so it is a layout cost, not a score defect. The 2.11 GPU-h figure carried in
earlier planning came from `THROUGHPUT_2026-08-15.md`'s measurement of the upstream DR config
(0.1266 s/update, no LSTM student, no frontier overlay) and should be replaced by 2.7–3.6.

## 4 · Where the count law buys, and where it provably does not

**At an atomic level the count law is the plug-in.** Under one frozen policy the N `n_eval`
streams are conditionally i.i.d., `P(K|z)` is exactly Binomial, and the granularity gap is
identically zero. An atomic-unit minimax arm is therefore not a test of the corollary. Design A
proposed an alternative buy — a "two-sided squeeze" claiming the plug-in under-estimates by
−0.297 at small evidence and over-estimates by +0.249 under pooled drift. **Both halves are
refuted.** The −0.297 was computed for `u_N(K/N)`, which minimax does not deploy (the default
is `posterior_mode='expected_activity'`, `plr.py:57`), and against the shipped estimator the
bias is sign-varying, not a Jensen under-statement. The +0.249 was simulated on an adversarial
0→0.9 drift schedule; the temporal component measured on real deployed group streams is +0.011.

More decisively, the count-law estimator trades bias for variance and loses that trade on
near-homogeneous units. I simulated both estimators at N=8 against the same truth (200k reps):

| unit | W=4 groups | RMSE count-law | RMSE plug-in |
|---|---|---|---|
| homogeneous, p=0.15 | | 0.364 | **0.247** |
| homogeneous, p=0.30 | | 0.214 | **0.090** |
| 10% dead mass, p_live=0.5 | | **0.224** | 0.237 |
| 20% dead mass, p_live=0.5 | | **0.254** | 0.358 |
| 20% dead mass, p_live=0.9 | | **0.102** | 0.520 |

The count law is unbiased and high-variance because it estimates `P(K>0)` from W
group-Bernoullis while the plug-in uses all N·W individual Bernoullis. **It wins only above a
dead-mass threshold: ≈0.40 at p_live=0.2, ≈0.09 at 0.5, <0.05 at 0.9.** That crossover curve is
the estimator's admissibility condition and must be pre-registered as such.

**So the minimax experiment must score a coarse unit — and the natural coarse unit in AMaze
does not deliver treatment.** Design A's premise replicates: over 20,000 BFS draws of the
shipped DR config (13×13, `n_walls=60`, `replace_wall_pos=True`, `maze.py:151-200`) I measure
**15.4% of levels have an unreachable goal**, so Fig. 1's Level B is already in the substrate.
But dead mass rises *monotonically* with Manhattan distance, the same axis the plug-in already
ranks on, so the penalty is monotone in the existing order:

| Manhattan band | prior mass | dead fraction | sampler TV at p_pass=0.2 | at 0.6 | at 0.95 |
|---|---|---|---|---|---|
| 1–4 | 0.192 | 0.076 | | | |
| 5–8 | 0.325 | 0.127 | | | |
| 9–12 | 0.292 | 0.181 | 0.022 | 0.065 | 0.154 |
| 13–16 | 0.145 | 0.221 | | | |
| 17–24 | 0.046 | 0.287 | | | |

The *score* gap is 0.14–0.31 across the same sweep — so Design A's proposed delivery gate
(`weighted_granularity_gap > 0.05`) **passes while the sampler barely moves**. Gate on the
realised visit-distribution TV between the plug-in and count-law arms, not on the gap.

**And Design A's family arm as specified measures nothing.** Its `_expand_group_levels` draws
`n_eval` *distinct* instances per family. If each member draws its own instance, the members
are marginally i.i.d. Bernoulli(p̄) and `K|z ~ Binomial(N, p̄)` **exactly**, so the gap is
identically zero. Design B's rule is the correct one and is adjudicated in its favour against
Critic 3's contrary reading: **one instance per group, shared by all N members, resampled only
across groups.** That is also what Kinetix's SFL does (`experiments/sfl.py:391-399`) and what
MaxRL's own group structure is (N samples on one prompt). Accepting it forfeits Design A's
claim that family units repay the −0.207 diversity cost: with a shared representative the batch
still contains only `n_parallel` distinct mazes. Two further blockers on that arm: with 5
families in a 500-slot buffer, `is_warm = filled.sum()/buffer_size ≥ min_fill_ratio`
(`plr.py:265,270`) is `0.01 ≥ 0.5 → False`, so PLR **never replays** and the arm is domain
randomization with a write-only buffer; and the `tie_aware_score_ranks` config lineage those
arms inherit is not implemented in any tree on disk.

**Verdict for minimax.** Ship the estimator and the telemetry, run the atomic strict-group arm
as an engineering and instrumentation result, and treat the coarse-unit AMaze arm as
**contingent on a calibration sweep** that certifies (i) sampler TV ≥ 0.05 and (ii) dead mass
above the crossover, using a family key chosen to be *orthogonal* to difficulty rather than
collinear with it. The Manhattan-band key fails both.

## 5 · The score

Deploy exactly this, and normalise:

```
Priority(z) = max(dense(z), 0) · [ Â_E(z) / A*_N ]^γ ,   A*_N = 2(1 − 1/N)
```

Four decisions, each forced by something measured.

1. **Gating, not replacement — already shipped.** `frontier_mode='gate'` computes
   `score = jnp.maximum(score, 0.0)*activity` (`plr.py:406-415`). Pure activity is closed on
   AMaze: best pure arm 0.5510 against upstream 0.6288, below the no-curriculum control.
2. **`dense = mean_t|δ_t^GAE|` is a one-token swap.** `compute_l1_value_loss`
   (`ued_scores.py:254-258`) is `compute_episodic_stats(jnp.abs(batch.advantages), ..., time_average=True)`
   — literally the user's proposed dense term. Replace the hardcoded `UEDScore.MAX_MC` at
   `plr_runner.py:319-321` with a `--plr_frontier_gate_base` selector.
3. **γ must be exposed.** It is currently pinned at 1 by the shipped gate.
4. **The normaliser must be `2(1−1/N)`, not `2·u_N(p*)`.** The count-law activity's true maximum
   is `max_k M_MaxRL(k) = 2(1−1/N)`, attained by a law concentrated at K=1 — 1.75 at N=8 against
   `A*_8 = 1.3002` under the Binomial-only bound. Without normalisation the 2× scale difference
   between `group_law_teacher` (`A_E`) and `estimators.coefficient_activity` (`u_N`) silently
   doubles every gated score and corrupts the insertion threshold
   `should_insert = score >= scores[insert_idx]` against a buffer holding old-scale values. Any
   checkpoint resumed across the estimator switch is invalid.

## 6 · level-replay: do not

`facebookresearch/level-replay` is archived at one 2020 commit under CC-BY-NC-4.0, needs Python
3.8 / gym<0.16 / tensorflow==2.2.1 / three custom forks, and breaks on numpy≥1.24. The cohort
patch is correct in outline — `sample_for_actor` pins G actors to one seed, PPO's 256-step
segment (`train.py:185`) is a free frozen-policy barrier, Procgen self-pins after
`venv.seed(z,e)`, and `_average_value_l1` (`level_sampler.py:120-126`) is already
`mean_t|δ^GAE|` — but the specified group ("first completed episode of each actor in the
segment") consists almost entirely of episodes that *began* under the previous policy, and those
straddlers are length-biased toward timeouts, biasing K down by 0.12–0.25 in pass-rate units.
The repair (start *and* end inside the segment) works only on MultiRoom. Four days of 2020
archaeology, on a discrete-action substrate that by its own blockers cannot carry the theory to
continuous control, for no throughput advantage over minimax. **Cut it.** Keep only the
estimator contract: a `GroupLawSpec` plus a three-op numpy/JAX/torch backend, verified against
the deployed `GroupLawPosterior` to 1.6e-15 over 400 non-Binomial dependent streams, with tiered
tolerances (float64 1e-12, cross-backend 1e-9, float32 1e-5 — float32 measures 9.4e-7 worst
case, so a single 1e-9 assertion would be disabled on first contact with JAX).

## 7 · The robotics instantiation

**Primary: Kinetix** (`/data/robotixx/ued_bench/src/kinetix-probe`, MIT, jax 0.6.2, installed
and measured on the 5090). It satisfies every requirement without new simulator work: groups
already form at frozen policy over one shared instance (`sfl.py:391-399`); the cell axes are
continuous physics scalars (`SimParams.base_friction`, `base_motor_power`, per-body
`friction`/`inverse_mass`); the binary predicate is native (`info["GoalR"] = reward > 0`,
`env.py:238`); and it ships MuJoCo-analogue control levels plus a procedural locomotion
distribution. Two corrections to Design B before anything runs:

- **`override_reset_state` must be passed to every `env.step`.** Kinetix auto-resets to a fresh
  *random* level on done (`env.py:78-88`) unless overridden, and `done = reward != 0`
  (`env.py:120`); a group runner that ORs `GoalR` over 128 steps without the override folds
  unrelated levels into the same Bernoulli and silently inflates K.
- **The Level-B construction must be written for the `EnvParams` axes.** The shipped
  `friction_absolute_range_exclude` kwarg drives the *per-body* level friction (six independent
  draws per instance, `locomotion_distribution.py:612,625,...`), which central-limits the
  bimodality away — the exact path the design discards. The shell sampler
  (`sample_uniform_and_exclude`, `:283`) is reusable in ~5 lines against `base_friction` /
  `base_motor_power`, but it is unwritten.

**Cell partition:** a product box per axis with exclusion radius E; Level A is `E=0`, Level B is
`E=0.75R` at the same centre — identical support, identical parameter mean by symmetry, opposite
within-cell heterogeneity. **Arms** (all sharing warmstart, seeds, sampler, budget and group
structure, all reading the *same* accumulators, only the functional differing): uniform ·
plug-in `u_N(p̂)` · count-law `q̂−p̂` · SFL plug-in `p̂(1−p̂)` · unbiased-SFL `K(N−K)/(N(N−1))` ·
gated `mean_t|δ|·u_N^γ`. The per-member-resample control must be **score-only**: as a training
arm it changes the MaxRL weights themselves and confounds "gap=0" with "estimator invalid".

**Gates, in order.** (1) A 31-second calibration sweep at the warmstart checkpoint, outcome-blind,
that certifies per-instance bimodality, matched means `|Δp̄| ≤ 0.05` on ≥32 of 64 pairs, and
predicted sampler TV ≥ 0.05 — the Acrobot retrospective in §9 says this is the gate the whole
programme lives or dies on. (2) Warmstart coverage ≥ 0.15 on ≥18/20 seeds, or the MaxRL-weighted
policy gradient does not learn and the campaign is dead with no salvage; pre-specify
PPO-with-MaxRL-weighted-advantage as the fallback learner. (3) The dead-mass crossover from §4.

**Endpoint and power.** Paired per-seed ΔcovAUC over 10 fixed checkpoints, SESOI +0.02. The
three-part support rule (`p ≤ .05` AND CI lower > 0 AND point ≥ SESOI) **caps power at 50% when
the true effect equals the SESOI**, by symmetry, independent of n — replace it with a proper
superiority-margin test (CI lower ≥ SESOI) powered at 2× SESOI. A 6-seed SD pilot has 32%
relative error and passes a "SD ≤ 0.030" gate 18% of the time when the true SD is 0.045, so it
cannot enforce itself: **budget n=40 up front** and drop to 20 only on a ≥25-seed certification.

**Confirmatory 3D: mjlab 1.6.0** (installed, MJWarp, heightfield collisions supported —
the "no rough terrain" constraint is an MJX/Brax limitation, not this stack). Two arms × 5
seeds, descriptive, no decision rule, gated on a throughput probe. **Isaac Lab is not the
primary**: its `EventManager` resamples the DR draw per env at reset, which is exactly the
per-member-resample regime where the gap is identically zero — making a group there requires
pinning the DR vector into the level written by `reset_to`, a patch to the event path rather
than a config change, and its own pilot already lost to the stock ±1 walker (.278 vs .372).

**Free, and larger than any of this: BARN campaign 003.** Genuine N=8 groups (all 8 episodes on
one sampled Gazebo course inside one runtime launch, `barn_gazebo.py:700-720`), a deliberately
coarse scored unit (10 difficulty strata × 24 courses, `barn_protocol.json`), ablation cells at
N∈{2,4,16} already frozen, launched 2026-08-14 and never sealed. `hopper/finalize_barn_ledger.sh`
then `finalize_barn_campaign.sh` are written and ready; no BARN artifact exists locally. That is
a real robot-navigation count-law dataset at the exact coarse unit the theory indicts, two
commands from retrieval, and it should be the first action taken.

## 8 · Relabeling / HER: excluded, and it is a lemma, not a judgement

The proposed form — relabel each group member to its own achieved goal — is **provably
degenerate**. Every member then succeeds, so K=N, and `M_MaxRL(N) = M_RLOO(N) = M_GRPO(N) = 0`
identically (verified at N∈{4,8,16,32}). The relabeled group carries *exactly as much coefficient
mass as the all-fail group it replaced*: zero. It has additionally destroyed exchangeability, so
K is a shared statistic over unrelated tasks and the count law no longer describes anything.
This generalises to every permutation-equivariant binary group estimator, since all have zero
mass at k=N — which is a standalone paper lemma answering the proposed architecture directly.

The only admissible variant is **group-consistent** relabeling: one common goal g′ for the whole
group, drawn from a uniformly chosen member, giving K′∈[1,N] and mass `2(1−K′/N) > 0` whenever
K′<N. Even that is anti-correlated with need: simulated at N=8, when all members end in the same
place (the stuck regime that motivates HER) the mass is 0.000; it rises to 0.876 at 2 distinct
achieved goals and 1.532 at 8. So HER injects gradient in proportion to behavioural diversity,
which is lowest exactly when the dead zone binds. **Keep it out of the confirmatory arms.**
The K=0 dead zone is covered by `plr_replay_prob=0.5` fresh-DR mixing / Kinetix's
`sampled_envs_ratio: 0.5`. If run later, it is a separate update-count-matched 2×2
(score × HER), descriptive, with the hard rule enforced *in code*: relabeled groups update the
policy and never any cell's posterior.

## 9 · Novelty ledger

State this defensively, because one intended novelty is already refuted.

| claim | status |
|---|---|
| "The count law replaces heuristic p(1−p)" | **False for RLOO.** SFL's learnability `p̂(1−p̂)` equals the realized RLOO mass `M_RLOO(k)=2k(N−k)/(N(N−1))` times exactly `(N−1)/(2N)` — verified constant at every k for N∈{8,10,16,32}. Ranking is scale-invariant, so **SFL (NeurIPS 2024) already *is* the count-law curriculum for RLOO.** `LITERATURE_POSITIONING.md`'s framing of SFL as "a heuristic we replace" must be corrected before a reviewer does it. |
| The aggregation correction | **Survives and is stronger.** SFL pools `ΣK/Σn` then applies the curve (`sfl.py:92-94`), over-stating count-law activity by the empirical `V̂ar(K)/N²` of the pooled groups. Closed-form, one-line diagnosis of a NeurIPS-2024 method. |
| Estimator-specific mass shape | **Survives.** `M_MaxRL` peaks at k=1, `M_RLOO`/SFL at k=N/2 — genuinely different curricula, not a rescaling. |
| Variable-N defect | **Survives.** Kinetix's default `learnability_mode: "timesteps"` forces `rollout_episodes=1` (`sfl.py:126-127`), so N varies per level and the shipped `correction = n/(n+1)` shrinks in the *wrong* direction (`E[p̂(1−p̂)] = p(1−p)(n−1)/n`), giving a combined `(n−1)/(n+1)` low bias that varies across levels. A citable, fixable defect in an ICLR-2025-Oral codebase. |
| Cost accounting | **New.** Kinetix's SFL spends 16×16384×512 = 134,217,728 env-steps per refresh against 50×2048×256 = 26,214,400 training steps between refreshes — **5.12× scoring overhead**, so the default 1.07e10 budget really consumes 6.6e10 steps. A count-law score read off the training groups costs 0. Any SFL baseline must be charged this, or matched on total env-steps and reported as crippled by it. |
| HER lemma | **New** (§8). |

Relative to PLR/ACCEL: they estimate no success probability at all — the score is a value-error
statistic over one episode, replaced rather than accumulated on revisit (`plr.py:318`). Relative
to ADR: it scores a boundary hyperplane by a mean over m episodes with everything else
randomized, which is Level B by construction, but its metric is consecutive-successes, so
instantiating the count law there breaks comparability with the published numbers.

## 10 · Staged plan, kill-gates, and the T-30d call

The GPU is not free: the gate-confirmatory campaign is running seeds 2007 of 2001–2010 right
now (PIDs 170529 / 271894, 89% util, 28.3/32.6 GB), ~3 pairs remaining. Nothing launches before
it lands, and its verdict conditions Stage 2.

**Stage 0 — this week, 0 GPU-hours.** (a) Retrieve BARN 003 (two commands). (b) Commit the
retrospective count-law audit as a script + JSON: it already ran, on 240 deployed
continuous-control runs / 76,208 N=16 groups, and produced the single most decision-relevant
number in this note. (c) Correct `LITERATURE_POSITIONING.md` on SFL. (d) Ship the
`GroupLawSpec` + numpy/JAX/torch backends + conformance suite (~2 days, no compute, de-risks
every downstream option).

> **KILL-GATE 0.** From the retrospective: sampler TV between the plug-in and count-law teachers
> is **0.0255 mean / 0.0388 at p90**, exceeding 0.05 in **1.25% of runs**; Spearman(count-law,
> plug-in) = **0.976**; granularity gap **median 0.0000**, mean 0.0395, p90 0.1379, max 0.4756
> against `A*_16 = 1.559`. **At naturally-arising coarse units the two teachers are nearly the
> same teacher.** If BARN's 10 strata do not clear TV ≥ 0.05, every campaign must be
> pre-registered as a *constructed* counterexample with a natural-cell control arm predicted
> null. That reframing — "the plug-in is breakable by construction, and here is the exact
> construction" — is honest, cheap, and publishable; discovering it after 78 GPU-hours is not.

**Stage 1 — after the confirmatory lands, ~3 GPU-hours.** The minimax estimator patch (§3) at
the atomic strict-group config that already ran, plus the telemetry rewrite (kill the
`frontier_group_size_match` config echo, add `weighted_granularity_gap`,
`frontier_closed_group_count`, a *runtime* closure fraction).
> **KILL-GATE 1.** Bitwise parity against v6 at `estimator='plugin'`; runtime closure fraction
> reads 1.0 under strict `n_eval=8`. Both must pass or the patch is wrong, not the theory.

**Stage 2 — contingent, ~40 h wall at 2-way concurrency.** Coarse-unit AMaze arms *only if* a
calibration sweep certifies a difficulty-orthogonal family key with TV ≥ 0.05 and dead mass
above the §4 crossover, at ≥20 families with `plr_buffer_size` ≤ n_families/min_fill_ratio.
> **KILL-GATE 2.** TV < 0.05 over the second half of training ⇒ "treatment not delivered", stop,
> report the null, do not escalate. Pre-register that both arms may sit below upstream 0.6288
> while A2−A1 is positive, and that this is the honest publishable outcome.

**Stage 3 — next paper.** Kinetix, with §7's corrections, n=40, the SFL baseline charged 5.12×,
and mjlab as the 3D confirmatory.

**T-30d call.** The 2–3 week minimum viable version is Stages 0 and 1 plus the two paper-side
results that need no compute at all: the **HER degeneracy lemma** (§8) and the **SFL–RLOO
identity plus its pooling-bias correction and 5.12× cost accounting** (§9). Those are theory
contributions that strengthen the existing manuscript rather than opening a lane. The robotics
campaign is a real design with real gates and it should be built — but it is Stage 3, and
`LANE_CLOSURE_2026-08-15.md`'s prohibition on citing the UED lane as manuscript evidence remains
standing until explicitly lifted.

---

## Appendix · Independent re-derivation of the three load-bearing claims

The claims this note turns on were re-derived outside the research fan-out that produced it,
so that nothing structural rests on a single agent's reading. `control_port/verify_note_claims.py`
reproduces all three from numpy alone in ~15 s:

| claim | §  | result |
|---|---|---|
| The granularity gap requires a **shared** atomic instance per group. Resample per member and `K\|z ~ Binomial(N, p̄)` exactly, so the count law collapses onto the plug-in. | 4 | Level B activity `0.000000` shared vs `0.496094` resampled — the latter equal to `u_8(p̄)` to 1e-12, confirmed by 200k-rep Monte Carlo. |
| SFL's learnability `p̂(1−p̂)` equals `M_RLOO(k)·(N−1)/(2N)` exactly at every `k`. | 9 | Ratio constant to 1e-15 at N ∈ {4,8,10,16,32}. `M_MaxRL` peaks at k=1 against RLOO/SFL at k=N/2 — a rival ranking, not a rescaling. |
| The gate normaliser is `2(1−1/N)`, not the Binomial-only `2·max_p u_N(p)`. | 5 | 1.750 vs 1.300 at N=8 (ratio 1.346); 1.938 vs 1.733 at N=32. |

The empirical half of §2 was likewise re-read directly from
`/data/robotixx/ued_bench/pilot-20260815-2031/arm-frontN8-s1001-u2500/logs.csv`: all 100 logged
rows carry `frontier_group_size_match = 1.0` and `frontier_incomplete_group_count = 0.0`, with
`weighted_frontier_trials = 8.000001` on the first row. **Groups form and close in minimax
today**, under a config that has already run. The prior note's substrate inversion is withdrawn.

One consequence deserves restating because it contradicts the proposal as written: the proposed
robotics group — *"对同一个环境实例并行采样 N 条不同动作轨迹"*, N rollouts of one **atomic**
instance — is conditionally i.i.d. by construction, so its count law **is** the plug-in and the
`u_N(p̂)` and `q̂−p̂` arms are the same algorithm. The group must share one instance drawn from a
**coarse cell**, resampled only across groups. That single structural choice is what makes the
experiment a test of the theory rather than a null by construction.
