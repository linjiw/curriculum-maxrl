# Curriculum-RL comparison plan and result: fixed-pool evidence first

Updated 2026-08-09. The common-scaffold Acrobot score tournament and the
separately frozen paid-probe selection attachment are complete. The remaining
named-method extensions are a post-submission roadmap, not a claim that the
listed methods have been run in one common benchmark. Online citations below
are primary paper or proceedings pages. Repository results are linked to their
retained local protocols and artifacts.

## Decision in one paragraph

The Acrobot V3 design served as a historical engineering anchor for a **new,
concurrent fixed-pool tournament with a common nominal transition budget and
bounded complete-group overshoot**. The completed V2 study held the practical
MaxRL learner, shared H64 actor, `N=16`, eight thresholds, evaluation, and
teacher-state machinery fixed; it varied only the task-selection rule among
uniform, the estimator-derived `u_16(p)=1-(1-p)^16-p`, and the nearest
learnability score `p(1-p)`. The last arm is the score-level overlap with
ProCuRL under its experimental `PoS*=1` assumption and with SFL;
algebraically it is also the in-family ablation `u_2`. It is not a claim to
reproduce either paper's full sampling system. The separately locked
paid-probe attachment also completed: all 320/320 confirmation runs were
valid, but the registered `u_16-ProCuRL` fixed-paid-AUC contrast was only
`+.004894` (`t(79)=1.9773`, `p=.05149`), below the `.02` SESOI and therefore
unsupported. The probed arms spent about 93.2% of paid transitions on probes;
ordinary uniform reached `.65149` fixed-paid AUC versus `.33771`, `.33942`,
and `.34261` for ProCuRL-env, probe-sham uniform, and `u_16`. This diagnoses
probe-cost domination only at the frozen actor-only refresh cadence, not
inferiority of full PPO ProCuRL. PLR and ALP-GMM remain secondary studies:
original PLR is compatible with a fixed pool, but its published value-error
priority requires a critic in the current actor-only scaffold; ALP-GMM
requires a continuous task generator.
PAIRED and ACCEL are out of scope for Acrobot because they require a
learned or editable environment-design space. V3 outcomes were not reused:
all three arms ran on 20 fresh paired confirmation seeds under one V2 lock,
after 9/9 valid outcome-blind development runs. The registered primary was
confirmed: `u_16-p(1-p)=+0.0480336884` [95% bootstrap CI
`0.0209366676, 0.0738485654`], exact sign-flip `p=0.0033607483`, 15/20
positive pairs, clearing both the frozen +.01 point-estimate and `p≤.05`
filters. The result is the protocol's P+/U+ score-shape outcome, not a full
named-method comparison. All Mac experiments required for this submission are
now complete; PLR, PAIRED, ACCEL, and full SFL are deferred.

## Repository-history answer: the Gym work was not deleted

The full Git history contains no deletion or rename of `frontier_rl/` or its
Gym examples. Classic MountainCar and CartPole entered in commit `6ba05c4` and
remain on `origin/main`. The later neural/audited Acrobot and MountainCar
protocols entered the research-only commit `14dbc9f`, which is not an ancestor
of `origin/main`; those later files were never merged to main, not deleted.
A subsequent merge (`81daaff`) combined the July 30
`TilePolicy(shared=...)` API with older callers still passing
`share_across_tasks=...`, which broke the optional classic-Gym constructor
path without removing its files. The main artifact command did not run the
optional Gymnasium tests, so that regression could coexist with a green paper
build.

This pass repaired the two caller names, restored historical
`hindsight_estimator` routing, charged every DAPO redraw transition, and added a
pinned Gymnasium/Acrobot release-test lane. The evidence decision remains
selective: old three-seed CartPole is excluded; corrected tile-coded
MountainCar stays appendix-only; the fresh Acrobot standard-control tournament
was rerun under the V2 audit and completed cleanly.

## What each cited method is, and its exact role here

| Method and online primary source | What the source contributes | Exact experimental role | Claim boundary |
|---|---|---|---|
| [MaxRL](https://arxiv.org/abs/2602.02710) | A compute-indexed family of sample-based objectives approaching maximum likelihood for binary-outcome RL, with policy-gradient estimators. | **Held-fixed student estimator.** The treatment is this repository's cross-task `u_N` sampler derived for the deployed practical-MaxRL convention; it is not a curriculum claimed by the MaxRL paper. | A sampler result under practical MaxRL does not establish a result for PPO, GRPO, another MaxRL convention, or maximum likelihood in general. |
| [ProCuRL](https://openreview.net/forum?id=8WUyeeMxMH) | In a pool-based contextual multi-task setting, scores a task by `PoS_t(s)[PoS*(s)-PoS_t(s)]`, where `PoS*` is success under an optimal/target policy; the experiments assume the unknown `PoS*=1`, giving `p(1-p)`. Its practical variants sample with `exp(beta p(1-p))` using rollout- or critic-based estimates. | **Completed score comparator plus completed paid-probe selection attachment:** V2 contains the common-scaffold `p(1-p)=u_2` score arm; Stage 2b tests ProCuRL-env softmax selection while charging every rollout probe. Its registered `u_16-ProCuRL` primary was unsupported. | The common-scaffold arm tests only the score. Stage 2b tests one actor-only fixed-pool selection attachment at one expensive refresh cadence, not full PPO ProCuRL or its published domains. |
| [SFL / No Regrets](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1d0ed12c3fda52f2c241a0cebcf739a6-Abstract-Conference.html) | Sampling for Learnability estimates `p(1-p)`, keeps high-learnability generated levels in a buffer, and trains with a parameterized buffer/random-level mixture; some reported settings use only the buffer. | **Same completed score comparator, not a duplicate primary arm.** A full SFL arm belongs in a procedural-level study where probe cost, buffer refresh, and random-level generation can be reproduced and charged. | Equality of the local score does not make the Acrobot arm a reproduction of SFL's buffer/generator algorithm or robustness evaluation. |
| [Original PLR](https://proceedings.mlr.press/v139/jiang21b.html) | Replays identifiable, resampleable levels using value/GAE-derived priority, rank transformation, and staleness. It permits a generic policy optimizer and does not require procedural generation. | **Deferred selection-semantics attachment.** Fixed-pool Acrobot is admissible, but `value_l1` requires a task-conditioned shadow critic in every comparison arm. | The attachment can test value-error/rank/staleness selection on the same pool. It cannot test PLR's unseen-level generalization result, and replacing value error with another score is only “PLR-style.” |
| [Robust PLR / PLR-perp](https://proceedings.neurips.cc/paper/2021/hash/0e915db6326b6fb6a3c56546980a8c93-Abstract.html) | Scores newly generated levels while stopping student-gradient updates on uncurated levels, then trains from a prioritized replay buffer. | **Out of scope for fixed-pool Acrobot.** Reserve for a procedural UED study with generation, replay, and out-of-distribution evaluation. | Robust PLR is not interchangeable with original fixed-pool PLR; removing its generator and stop-gradient exploration changes the method. |
| [PAIRED](https://arxiv.org/abs/2012.02096) | Trains an environment adversary using antagonist-minus-protagonist return as induced regret, producing structured, solvable environments. | **Out of scope for the Acrobot tournament.** Reserve for a separate UED benchmark with an environment generator and both student agents. | A threshold sampler is not PAIRED; there is no learned generator or antagonist in V3. |
| [ACCEL](https://proceedings.mlr.press/v162/parker-holder22a.html) | Generates curricula by editing previously useful levels within a regret-based environment-design framework. | **Out of scope for Acrobot; later procedural benchmark only.** It needs a level genotype, valid mutation operator, replay buffer, and regret score. | Mutating a scalar Acrobot success threshold would not test ACCEL's environment-complexity claim. |
| [ALP-GMM](https://proceedings.mlr.press/v100/portelas20a.html) | Samples continuously parameterized environments to maximize absolute learning progress, modeled with Gaussian mixtures. | **Secondary continuous-threshold extension.** Train on a continuous tip-height parameter while evaluating only on the frozen eight-task target distribution; charge random exploration and teacher fitting. | Discretizing ALP over the eight V3 thresholds is an ALP-style ablation, not full ALP-GMM. Absolute progress is not the same quantity as estimator coefficient mass. |

### Required versus optional

The minimum control experiment needed for the narrow paper claim—the fresh
three-arm Acrobot tournament with `uniform`, `p(1-p)`, and `u_16`—is complete.
It supports that the estimator-derived arbitrary-`N` score beats both no
curriculum and its `N=2`/closest intermediate-difficulty score under the same
learner and common nominal budget at deployed `N=16`. It does not alone establish behavior
across `N`;
that role belongs to the separate fixed-completion `N={2,4,8,16,32}` mechanism
sweep already retained in this repository.
PLR and ALP-GMM become required only before making a broader claim of
superiority to historical-dynamics or continuous-task teachers. Full SFL,
PAIRED, and ACCEL become required only for a separate claim about procedural
level discovery or UED. They are not missing controls for the narrow
fixed-pool score claim. Likewise, the current `p(1-p)` arm does not justify a
claim of beating full ProCuRL. The separately locked Stage 2b study tested
ProCuRL-env selection semantics with its probes included in the budget. Its
registered primary was unsupported, and the dominant empirical fact was the
cost of probing at the frozen cadence. No additional Mac experiment is needed
for the current submission; PLR, PAIRED, ACCEL, and full SFL remain deferred
unless a future paper broadens the claim.

## What the comparison papers actually test

| Paper | Empirical package in the primary source | Design lesson for this project |
|---|---|---|
| MaxRL | Exact-likelihood calibration on sampled ImageNet classification; approximately 999,744 training and 256 held-out 17-by-17 mazes produced by the released generator; repeated-data GSM8K; and Qwen3-1.7B/4B math training, with Pass@1, Pass@K, rollout-compute, and scaling comparisons. The release does not mechanically establish that every generated maze is unique or that the splits cannot overlap. | It validates an estimator/objective across data and model regimes, not a cross-task curriculum. Our RL control should therefore hold practical MaxRL fixed and vary only task selection. |
| ProCuRL | PPO on fixed contextual task pools: binary/non-binary PointMass, BasicKarel, BallCatching, and AntGoal, evaluated by uniform pool performance, total environment steps, and clock time. | The closest precedent for Acrobot is a fixed pool, uniform target evaluation, and charged learner/teacher compute. The common-scaffold `p(1-p)` arm tests the score induced by its experimental `PoS*=1` assumption, not every ProCuRL selection detail. |
| Original PLR | PPO on all 16 Procgen games plus challenging MiniGrid tasks, prioritizing identifiable levels by value-error/GAE-style scores and staleness and evaluating generalization to unseen levels. Its sampler is compatible with generic policy optimization even though the released experiments use actor-critic PPO. | Fixed-pool sampling is admissible, but this actor-only Acrobot scaffold needs a separately frozen, task-conditioned critic. Held-out levels are required for reproducing the paper's generalization claim, not for the sampler definition itself. |
| PAIRED | An adversary generates environments while protagonist and antagonist returns define regret; experiments track emergent gridworld complexity and zero-shot performance on novel human-designed mazes, with an additional continuous-control Hopper study. | A fair reproduction requires two learners plus a trainable environment generator and out-of-distribution test levels. Scalar threshold choice is not PAIRED. |
| ACCEL | PPO on MiniHack lava grids, MiniGrid mazes, and BipedalWalker terrains; high-regret levels are replayed and edited. Robust PLR is the main concurrent reference, ALP-GMM is included for BipedalWalker, and some PAIRED/minimax maze results are imported from prior work. | A fair comparison needs valid level mutations, replay state, regret scoring, and edit/probe compute. It is a later UED benchmark, not an Acrobot arm. |
| SFL / No Regrets | Single- and multi-agent JaxNav, MiniGrid, and XLand-MiniGrid; generated levels are probed, high-`p(1-p)` levels buffered, and robustness is evaluated with adversarial/CVaR-style metrics. Reported runs use 10 seeds in Minigrid/single-agent JaxNav and 5 in the multi-agent/XLand settings. Its wall-clock controls reduce PPO updates to match ACCEL in single-agent JaxNav and PLR in XLand; MiniGrid and multi-agent JaxNav use equal PPO-update counts, not universal transition matching. | Our shared `p(1-p)` arm is the necessary score control. A full SFL reproduction additionally needs generation, probe accounting, buffer refresh, and worst-tail evaluation. |
| ALP-GMM | Continuously parameterized BipedalWalker variants across learner embodiments, unlearnable-region ratios, and parameter-space dimension, using absolute learning progress to fit a mixture teacher. | Use it only in a continuous-threshold extension with a frozen target distribution; do not call an eight-bin heuristic full ALP-GMM. |

Together these papers support a two-layer evidence strategy: first isolate the
score on a fixed binary task family with a common nominal transition budget
and bounded complete-group overshoot, as completed in V2, then—only if the
paper expands to general curriculum discovery—add procedural generation,
unseen-level robustness, and native PLR/SFL/PAIRED/ACCEL machinery.

## Why Acrobot V3 is useful design history, not a reusable control

The [V3 protocol](frontier_rl/examples/ACROBOT_NEURAL_PROTOCOL_V3.md) already
isolates the desired variable better than the other Gym studies:

- official Gymnasium [`Acrobot-v1`](https://gymnasium.farama.org/environments/classic_control/acrobot/)
  dynamics and its 500-step limit;
- eight fixed nested tip-height predicates, with the hardest threshold equal
  to native Acrobot success;
- one task-agnostic shared H64 categorical actor (640 parameters), so task
  selection can change shared learning without exposing the task ID;
- practical MaxRL, `N=16`, no hindsight, and one learning rate in both arms;
- uniform versus exact `u_16`, paired over 20 sealed seeds with 2,000,000
  nominal environment transitions per run;
- target-uniform mean-pass AUC over actual transitions, initialization
  included, with a single registered contrast, source hashes, and independent
  reconstruction.

V3 reported uniform `0.648669` versus `u_16` `0.685021`, a descriptive paired
difference of `+0.0363524`. A later audit found that neighboring paired seeds
reuse one numeric root across actor-parameter and actor-action RNG domains.
Alternating-seed sensitivity estimates remain positive (even seeds `+0.0251`,
odd seeds `+0.0476`) but are imprecise and cannot restore the original paired
inference. V3 therefore supplies environment, budget, and instrumentation
design history—not a reusable control or clean efficacy claim. The completed
V2 tournament reran all three arms under globally separated domain roots, with
paired common random numbers shared only within each logical seed.

## Current post-hoc V3 mechanism audit

The optional file
[`acrobot_v3_mechanism_audit.json`](frontier_rl/examples/acrobot_v3_mechanism_audit.json)
exists and says no new training was performed. Its headline paired means are:

| Frozen-run endpoint | Uniform | `u_16` | Paired difference | Pair signs (`+/-/=`) |
|---|---:|---:|---:|---:|
| coefficient mass / sampled group | 0.6179 | 0.7510 | +0.1331 | 20/0/0 |
| coefficient mass / million transitions | 105.40 | 116.19 | +10.79 | 17/3/0 |
| nonzero-mass group fraction | 0.7507 | 0.8789 | +0.1282 | 20/0/0 |
| native success AUC | 0.3122 | 0.3763 | +0.0641 | 15/5/0 |
| native return AUC | -465.67 | -457.40 | +8.26 | 16/4/0 |
| final native success | 0.6203 | 0.7469 | +0.1266 | 18/1/1 |

The audit's separately seeded, descriptive bootstrap recomputation of the V3
primary difference is `[+0.016844,+0.055270]`; it does not replace the retained
confirmatory interval above. Every endpoint in this audit was selected after
the V3 outcomes existed, its tests are unadjusted, and none establishes that
coefficient mass causally mediates learning. In particular, the audit contains
no comparison with `p(1-p)`, PLR, ALP-GMM, PAIRED, or ACCEL.

## Why MountainCar is appendix-only and CartPole is excluded

MountainCar supplies useful boundary evidence, not the clean primary control.
The corrected ten-seed tile-coded study changes several ingredients at once
(tile policy, `gamma=4`, and in its strongest arm hindsight), so it cannot
isolate the V3 score comparison. More importantly, the newer
[neural MountainCar V1R2 result](frontier_rl/examples/MOUNTAINCAR_NEURAL_TRANSFER_V1_RESULTS.md)
is a registered development **NO-GO**: its 15 runs contained 1,932 all-fail,
474 mixed, and zero all-pass groups, while hardest-goal AUC was exactly zero in
every arm. It lacked both native-goal headroom and the mastered regime the
teacher should retire. Keep the corrected tile-coded result and the neural
failure as appendix sensitivity/adequacy evidence; do not open its reserved
confirmatory seeds under the failed protocol.

CartPole is excluded. The repository's current schedule identifies it as an
old three-seed smoke study that was not rerun under the repaired estimator and
evaluation stack, and the retained-curve exporter deliberately omits it. It
therefore supplies neither a registered control nor current evidence. A future
CartPole study would need a new protocol, fresh seeds, transition accounting,
and a nondegenerate native endpoint before it could re-enter.

## Completed fixed-pool stages and optional extensions

### Stage 0 — lock and accounting complete (Mac)

V2 froze one common sampler API, per-group accounting, globally separated
domain roots, actual-transition charging, a portable verifier, and the
decision rules before confirmation outcomes. Its source/runtime-lock SHA-256 is
`0e6438d42ddc53b89d774233805c465dc562bb6be5f8ac93ecf8a4d09b5d9af3`.
The lock and digests bind the source, runtime, gate, and artifacts internally,
but no immutable public pre-execution commit in this checkout establishes
their timing.

### Stage 1 — common-scaffold adequacy complete (Mac)

All nine development runs—three paired logical seeds across uniform,
`p(1-p)`, and `u_16`—were valid and passed the frozen outcome-blind launch
gate. The development raw and gate SHA-256 hashes are
`c616912569f4d19e36ea4a8685616a35bef037934e5c8d366ee7bd51bb2c3311`
and `6dc908e22e874550e0536f1fcd52f2b3a1768d1a89c510275bef7efc2e2baac6`.

### Stage 2 — fresh fixed-pool tournament complete (Mac, primary)

All 60 confirmation runs completed and passed the locked checks: 20 fresh
paired logical seeds per arm, a nominal 2,000,000 actual-transition budget
with the final group completed, and V3's fixed target-uniform evaluation grid.
The inferential unit was the paired seed.

| Registered comparison | Target-uniform transition-AUC arm means | Paired mean difference | Paired bootstrap 95% CI | Exact sign-flip p | Multiplicity and decision |
|---|---:|---:|---:|---:|---|
| `u_16-p(1-p)` (primary) | .6871056515 vs .6390719632 | **+.0480336884** | **[.0209366676, .0738485654]** | **.0033607483**; 15/20 positive | single primary; **confirmed**, clearing the frozen +.01 point-estimate and `p≤.05` filters |
| `p(1-p)-uniform` (secondary) | .6390719632 vs .6452319465 | -.0061599834 | [-.0226437219, .0121954971] | .507843 | Holm .507843; not supported |
| `u_16-uniform` (secondary) | .6871056515 vs .6452319465 | **+.0418737050** | **[.0218239396, .0605859853]** | **.000808716**; 17/20 positive | **Holm .001617432; supported** |

This is the registered **P+/U+** outcome: in one fixed eight-threshold Acrobot
pool, the deployed-`N` score beats its `N=2` score ablation and uniform under
one shared H64, 640-parameter practical-MaxRL learner at `N=16`. It is not a
full ProCuRL, SFL, PLR, PAIRED, ACCEL, or ALP-GMM implementation, and it does
not test held-out-task generalization. The 20-seed choice was inherited from
V3 rather than selected by a prospective power calculation. The exact test
requires paired-sign exchangeability under the sharp null, and clearing the
SESOI point-estimate filter does not prove the population effect is at least
+.01. See the retained
[V2 result report](frontier_rl/examples/ACROBOT_CURRICULUM_TOURNAMENT_RESULTS.md).
The confirmation-raw and locked-analysis SHA-256 hashes are
`f533d0b84cdb3f7d3ede4bc4c94aa11e3b0ffc58c8bc7ea1a26491476873b2c6`
and `463fa1a01d95922976f09f75b21f6d8f2c6a8d256081ebedfa4ba968a06f356b`.

### Stage 2b — native ProCuRL selection semantics (Mac, secondary)

This separately locked paid-probe study is **complete**. All 12 development
runs and all 320/320 confirmation runs passed the frozen accounting checks.
It tested ProCuRL-env softmax task selection with `beta=20`, 20 rollout probes
per task, and an estimate refresh every 5,120 student transitions; every probe
transition was charged. The confirmation used four arms on 80 paired seeds
and remains separate from the completed Stage-2 primary family.

The registered `u_16-ProCuRL` fixed-paid-AUC mean difference was `+.004894`,
with `t(79)=1.9773` and `p=.05149`. It did not reach the `.02` SESOI, so the
registered primary is unsupported. The three probed arms each spent about
93.2% of paid transitions on probes. Their fixed-paid AUCs were `.33771`
(ProCuRL-env), `.33942` (probe-sham uniform), and `.34261` (`u_16`), compared
with `.65149` for ordinary uniform.

This extension tests selection semantics beyond the common-scaffold
`p(1-p)` score, not the arbitrary-`N` identity itself. The safe conclusion is
that probe cost dominated this actor-only fixed-pool attachment at the frozen
refresh cadence. The result must not be appended post hoc to the completed
three-arm family, used to relabel that score arm as full ProCuRL, or cited as
evidence that full PPO ProCuRL is inferior or that cheaper probing behaves the
same way.

The compact repository boundary retains receipts and a content manifest. The
1,374,886,097-byte raw ledger remains external with SHA-256
`b1f8756c249effab8c77101c8bca73ddf708a5e143c18fe8742fd5712fdd7c12`.
The source/runtime/gate/artifact chain is internally bound, but there is no
immutable public pre-execution commit establishing timing. The registry
generator emits and checks exactly 562 total records, including 441 Acrobot
records; the generated registry owns the exact totals. Full registered and
descriptive results are in
[`ACROBOT_PROCURL_SELECTION_RESULTS.md`](frontier_rl/examples/ACROBOT_PROCURL_SELECTION_RESULTS.md).

### Stage 3a — PLR selection-semantics attachment (Mac, deferred)

Original PLR does not require procedural levels or PPO, but its released
`value_l1` priority requires a learned value function and GAE. The current
actor-only learner therefore cannot host that selector unchanged. Moreover,
the nested Acrobot success predicates make a task-blind value function
ill-defined: the same physical state can have different returns under the
eight hidden thresholds.

The smallest defensible attachment would freeze three fresh arms:

1. `uniform + shadow critic`;
2. `u_16 + shadow critic`; and
3. `PLR(value_l1, rank, staleness) + shadow critic`.

Use the same task-conditioned critic—eight value heads or an explicit
threshold input—in every arm, with no shared actor parameters. Keep the actor,
practical MaxRL update, `N=16`, learning rate, thresholds, evaluation grid,
and nominal two-million-transition budget fixed. Compute complete-trajectory
GAE with the pre-update critic, update the task score and staleness state, and
then update critic and actor. Freeze `gamma=.999`, `lambda=.95`, rank
prioritization, temperature `.1`, staleness coefficient `.1`, and the released
full-pool fill behavior as the named Procgen-style source setting.

The group of 16 same-task trajectories is an unavoidable MaxRL adaptation.
For the source-nearest primary, process them in a frozen ledger order,
increment staleness time by completed episode, and retain the final
trajectory's score as the latest task score. Label any group-average score as
a robustness variant. Charge all transitions and report group count, actor
and critic updates, elapsed time, scores, ranks, staleness, and task
probabilities. This can support only a fixed-pool selection-semantics claim,
not native PLR, UED, robustness, or held-out generalization.

This attachment remains **deferred/no-go for the current paper**. Launch it
only under a new prospective protocol if a post-submission claim expands to
temporal value-error curricula.

### Stage 3b — full PLR generalization study (post-submission)

An unqualified PLR/generalization comparison should instead use a shared
actor-critic learner, identifiable fixed training levels, and a disjoint
held-out pool from the same lightweight procedural generator. Compare uniform
with the named PLR source setting, retain all value/GAE/score/rank/staleness
state, and report both training-pool sample efficiency and held-out
generalization. A small MiniGrid or procedural-maze family is the appropriate
CPU target. See [`PLR_GO_NO_GO.md`](PLR_GO_NO_GO.md).

### Stage 3c — continuous-task historical dynamics (post-submission)

Add ALP-GMM only as a separate continuous-threshold study with the same fixed
eight-threshold evaluation distribution. Before calling it native ALP-GMM,
freeze the nearest-neighbor absolute-return-progress definition, fit window,
GMM component selection and fit cadence, random-exploration rate, task bounds,
seeds, and budget. Give it its own uniform control because its probes and task
space change the learner's experience. It must not be pooled into the primary
score-function family after results are known.

### Stage 4 — estimator interaction (post-submission GPU study, <=10 GB)

The Acrobot tournament itself does not need a GPU. Use the constrained GPU for
the existing 1.26M-parameter maze model: on development blocks, select a frozen
learning rate separately for practical MaxRL and GRPO, then confirm a
`{MaxRL,GRPO} x {uniform,p(1-p),u_N}` factorial on fresh shared warm-start/seed
blocks. Match groups, `N`, steps, evaluation, and prompt pool; report both
common-rate and tuned-rate results. This is the experiment that can show an
estimator-by-sampler interaction. It still does not reproduce PLR, SFL,
PAIRED, ACCEL, or ALP-GMM.

### Stage 5 — procedural UED only after post-submission scope expands

A joint UED benchmark comparing SFL, PAIRED, ACCEL, and Robust PLR needs a
common procedural benchmark and method-specific native machinery: repeated
candidate generation/probing and a retained learnability buffer for SFL;
stop-gradient evaluation of uncurated generated levels and prioritized replay
for Robust PLR; a valid mutation/editor operator for ACCEL; and an environment
adversary plus protagonist and antagonist for PAIRED. That is a separate
paper-scale systems experiment. Report environment interactions, student
gradient updates, and wall-clock cost because the source papers do not use one
universal compute-matching convention. Start with a memory/timing pilot, but
do not promise it under a 10-GB limit until all agents, buffers, and
environment batches are measured.
If it does not fit, preserve the Acrobot fixed-pool claim and defer UED rather
than substituting approximations under the original method names.
Include Domain Randomization/random-level sampling as the common generator
baseline, and compare ACCEL to its primary Robust-PLR reference rather than
silently substituting plain PLR. Uniform over the frozen Acrobot thresholds is
only the fixed-pool analogue of random level sampling.

## Stop/go claims

- **Supported by completed Stage 2:** “Under the fixed practical-MaxRL learner
  on one eight-threshold Acrobot family, `u_16` improved target-uniform
  transition-AUC versus both uniform and the common-scaffold `p(1-p)` score.”
- **Not supported by completed Stage 2b:** a registered fixed-paid-AUC
  advantage of `u_16` over ProCuRL-env at the frozen `.02` SESOI. The allowed
  descriptive conclusion is that paid probe cost dominated all three probed
  arms at the frozen refresh cadence.
- **Allowed after Stage 4:** an estimator-by-sampler interaction on the frozen
  maze design, if the preregistered interaction test supports it.
- **Allowed only after a separately frozen Stage 3a:** “On the same fixed
  Acrobot pool, `u_16` differed from original PLR's
  value-error/rank/staleness selection semantics.” This would still not be a
  UED, robustness, or held-out-generalization claim.
- **Not allowed from this roadmap alone:** superiority to full ProCuRL or SFL;
  superiority to PLR/ALP-GMM; any comparison to PAIRED/ACCEL; general UED,
  robotics, LLM, or maximum-likelihood claims; causal mediation by coefficient
  mass; or reuse of post-hoc audit tests as confirmation.

The submission stop rule is therefore: **all planned Mac experiments are
complete.** The remaining work is evidence recovery and the final
paper/artifact rebuild. PLR, PAIRED, ACCEL, full SFL, estimator-interaction,
continuous-task, and UED systems studies are deferred until a post-submission
protocol can honor their native interfaces and compute accounting.
