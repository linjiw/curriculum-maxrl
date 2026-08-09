# Curriculum-MaxRL

**Teacher-guided curriculum learning driven by the MaxRL objective's own algebra.**

Research codebase exploring the integration of curriculum learning (teacher–student,
ZPD/learnability targeting) with Maximum Likelihood Reinforcement Learning
([MaxRL, arXiv:2602.02710](https://arxiv.org/abs/2602.02710)). Built against the
official MaxRL implementation (a [verl](https://github.com/verl-project/verl) fork).

## The idea in one paragraph

MaxRL reweights per-prompt gradients by the bounded truncated-likelihood weight
`(1-(1-p)^T)/p` (approaching `1/p` only as `T→∞`), which acts as an
*implicit, gradient-level* curriculum — but it cannot
rescue prompts whose rollout groups come back all-fail (K=0 → group dropped, zero
gradient), and it wastes compute re-rolling mastered prompts. We add an *explicit,
data-level* teacher whose utility function is **derived from the estimator itself**:
the expected total scalar coefficient magnitude from N rollouts is exactly

```
E[Σ|w|] = 2 · (pass@N(p) − pass@1(p)) = 2 · ((1−(1−p)^N) − p)
```

— twice the probability the prompt is *solvable within N attempts but not within one*.
This identity is for practical Algorithm-1 weights that drop the entire `K=0`
group; their population objective is order `T=N-1`, while the raw and
always-retained-control-variate estimators in the paper are order `T=N`.
This is a compute-indexed formalization of the zone of proximal development, peaking at
p* ≈ ln(N)/N. The teacher uses discounted Beta pseudo-counts and
prioritizes prompts by this utility; the myopic known-pass-rate rollout
allocation is greedy water-filling on the marginal `p(1−p)^N` (the probability
the next rollout is a group's first success).

## The experiment ladder — what each experiment asks, and what to expect

Every experiment isolates one or two *channels* of the method. Reader's map:
the **teacher** reallocates compute (channel 1), **hindsight** creates signal
from failures (channel 2), the **objective** changes how a curriculum behaves
(channel 3). External execution records label several predictions as specified
before their deciding cells, but the maze/Countdown locking objects are not all
vendored, so their timing is not independently auditable from this checkout.

| experiment | the question | externally recorded expectation | outcome — including our own retractions |
|---|---|---|---|
| **CPU skill-chain** (36 tasks, exact gradients) | do the channels work at all? | teacher > uniform; recycling adds | ✓ 0.65→0.73→0.89 — **corrected**: a floor-and-γ-matched true-p oracle *ties* the full stack (0.8885 vs 0.8895; "beats the oracle" is retracted); recycling adds on top of even the oracle (0.8935) |
| **V5 frontier-heavy regime** (max pool p=1e-5) | what happens when NO task is samplable? | pure samplers get exactly 0; recycling invents the curriculum below the pool | ✓ categorical: 0.93 AUC vs 0.00 for uniform/DAPO/teacher-alone |
| **Acrobot fixed-pool tournament V2** (shared H64 practical MaxRL, `N=16`, 8 thresholds, 20 paired seeds) | does the deployed-`N` score beat both its `N=2` score ablation and uniform? | primary `u_16-p(1-p)` point estimate ≥ +.01 and exact sign-flip `p≤.05` | **✓ P+/U+ confirmed.** Target-uniform transition-AUC means were .6452319465 uniform, .6390719632 `p(1-p)`, and .6871056515 `u_16`. The primary was +.0480336884 [95% bootstrap CI .0209366676, .0738485654], exact `p=.0033607483`, with 15/20 positive pairs. `u_16-uniform` was also Holm-supported; `p(1-p)-uniform` was not. This is score-shape evidence in one fixed Acrobot pool, not a full named-method comparison. |
| **Acrobot paid-probe selection attachment** (4 arms, 80 paired seeds) | does `u_16` outperform ProCuRL-env selection when every probe transition is charged? | registered primary `u_16-ProCuRL` fixed-paid AUC, with `.02` SESOI | **Primary unsupported.** All 320/320 confirmation runs were valid. The mean difference was `+.004894` (`t(79)=1.9773`, `p=.05149`), below the `.02` SESOI. The three probed arms spent about 93.2% of paid transitions on probes and had AUCs `.33771`, `.33942`, and `.34261`, versus `.65149` for ordinary uniform. This diagnoses probe-cost domination at the frozen refresh cadence; it is not evidence that full PPO ProCuRL is inferior. |
| **Digits exact-probability factorial** (MaxRL/RLOO × uniform/`p(1-p)`/`u_8`, 24 paired blocks) | does the estimator determine which intermediate-difficulty sampler learns best when `p` is exact? | MaxRL favors `u_8`; RLOO favors `p(1-p)`; positive registered interaction | **Primary not supported.** Interaction +.01589 [95% CI -.01686, +.04712], exact `p=.350`. MaxRL strongly favored `u_8` over `p(1-p)` (+.20842), but RLOO also favored `u_8`, reversing its prediction; both matched samplers were below uniform. This source-locked contextual-bandit negative result shows that coefficient mass is activity, not universal curriculum optimality. |
| **Balanced maze factorial** ({maxrl,grpo}×{uniform,teacher}×6 blocks, 250 fixed steps) | does the estimator coverage divergence survive a clean design? | ≥5/6 paired blocks MaxRL>GRPO under both samplers | The first wave's endpoint claim **failed (3/6, 4/6) and is retracted**. The external record says its time-integrated ordering was specified before six fresh blocks; it was positive 6/6 under each sampler (exact sign p=.03125 each). Averaging correlated samplers within block gives 12/12 positive independent block averages descriptively across both waves. Easy-band localization is suggestive only (4 positive, 1 tie, 1 negative in wave 2; interval crosses zero). |
| **E-LLM-1: GSM8K** (SmolLM2-360M, one A10G) | do channels 1+3 transfer to LLM RLVR? | steering-controlled treatment-delivery gate before interpreting the interaction | The completed controlled `g3p` cell reached the minimum criterion but its run mean was 0.601480, missing the registered `<0.60` delivery gate by 0.00148. **The interaction is inconclusive by design.** |
| **E-LLM-2/2b: Countdown** (exact-verifier recycling, v2 pool) | channel 2 at LLM scale | recycling ignites unreachable tiers | The ignition predictions were null (guesser-saturated pool). In three-seed aggregates, the recycling package raises tier-1 mean@16 0.278→0.324 while VERL bootstrap best@16 falls 0.541→0.492. This logged metric is a with-replacement coverage proxy, not standard unbiased pass@16; missing task outcomes block recomputation. The original under-gated point used faulty decay and the fixed-code strong gate failed. A higher-dose live-group replay arm improves both logged metrics but cannot isolate dose from direction. |
| **Jugs (water-measuring)** | does the whole family fail together where no band exists? | externally recorded all-null | ✓ all-null landed — the negative control |

Full LLM-experiment roadmap with novelty checks and differentiation map:
[`NEXT_EXPERIMENTS.md`](NEXT_EXPERIMENTS.md). Latest LLM results:
[`GSM8K_ANALYSIS.md`](GSM8K_ANALYSIS.md).
Current Mac/10-GB research priorities and literature positioning:
[`SMALL_COMPUTE_RESEARCH_PLAN.md`](SMALL_COMPUTE_RESEARCH_PLAN.md). The
executable next-stage GPU design, with separate maze and Countdown go/no-go
lanes, frozen seed ranges, schemas, gates, power targets, and recovery rules,
is in [`GPU_EXPERIMENT_HANDOFF.md`](GPU_EXPERIMENT_HANDOFF.md). The
fixed-pool control/RL comparison and completed Acrobot result are summarized in
[`RL_CURRICULUM_EXPERIMENT_PLAN.md`](RL_CURRICULUM_EXPERIMENT_PLAN.md), and a
pinned read-only audit of the authors' released implementation is in
[`MAXRL_SOURCE_AUDIT.md`](MAXRL_SOURCE_AUDIT.md). The reason a native PLR arm
is not being attached to the actor-only Acrobot learner is recorded in
[`PLR_GO_NO_GO.md`](PLR_GO_NO_GO.md). The completed 800-run synthetic cap and
information-sensitivity study, including its strict exploratory boundary and
selected cap-32 engineering configuration, is reported in
[`curriculum_maxrl/CAPPED_HORA_ROBUSTNESS_RESULTS.md`](curriculum_maxrl/CAPPED_HORA_ROBUSTNESS_RESULTS.md).
The completed source-locked exact-probability factorial and its registered
negative result are reported in
[`curriculum_maxrl/digits_factorial/RESULTS.md`](curriculum_maxrl/digits_factorial/RESULTS.md).

## Current paper and artifact status (2026-08-09)

- The registry generator emits and checks **exactly 562 records, including 441
  Acrobot records**, after integrating the paid-probe study. The generated
  registry is the owner of the exact totals; do not hand-maintain a competing
  count in prose. Missing raw logs remain explicitly marked, and aggregate
  summaries are not presented as vendored raw runs. The 800 HORA and 408
  Digits executions retain their separate raw/receipt accounting.
- A stored Countdown identity summary reports **27/128 tier-0 evaluation tasks
  overlap the SFT warm start**, leaving a nominal 101-task subset, and zero
  measured overlap for tiers 1--2. The missing source manifests prevent
  independent recomputation. A numerical 101-task reanalysis is blocked until
  per-task 16-sample verifier outcomes or compatible checkpoints and frozen
  manifests are recovered.
- The final paper rebuild and public-PDF synchronization are complete. The ICLR
  wrapper has 13 total pages: the conclusion ends on page 9, references begin
  later on page 9, and the appendix begins on page 10, so the main text fits
  the nine-page submission limit. The alternate working wrapper has 14 pages.
  The main text is organized around the exact identity, the fresh Acrobot score
  test, the externally recorded maze comparison, and the explicitly labeled
  Countdown bootstrap-coverage-proxy tradeoff, with the paid-probe null used
  as a narrow selection-cost diagnostic.
- This content-addressed release is frozen and verified. The Git commit that
  contains these files is the authoritative repository publication record;
  the manifest does not attempt to embed its own self-referential commit hash.
- The highest-value remaining evidence work is still to recover the 24 wave-2
  checkpoint trajectories and complete B1/B2 per-task outcomes (needed to
  recompute standard pass@16), but an exhaustive local
  [`recovery audit`](curriculum_maxrl/analysis/ARTIFACT_RECOVERY_AUDIT.md)
  found no surviving copy on this Mac. Recovery now requires the original
  EC2/EBS/S3/W&B side; otherwise those cells must be rerun. The frozen next
  GPU priority is estimator-specific LR calibration on the 1.26M maze model;
  a truly dose-matched small-model recycling control is conditional on
  recovering or rebuilding its missing execution assets. Both are specified
  in [`GPU_EXPERIMENT_HANDOFF.md`](GPU_EXPERIMENT_HANDOFF.md).
- The Mac-only capped-HORA robustness matrix is complete: all 800/800 runs
  passed independent reconstruction and accounting checks.  The frozen
  engineering filter chose cap 32, which reduced mean per-run maximum group
  size by 58.07% with a -0.00271 pass@8-AUC change versus uncapped after
  averaging the two samplers.  This is exploratory synthetic evidence, not HORA
  validation, neural-RLVR evidence, or proof of coefficient-mass mediation.
- The exact-probability Digits factorial is complete with 24/24 fresh paired
  confirmation blocks and no failures. Its primary estimator-by-sampler
  interaction is negative evidence: the interval crosses zero and exact
  `p=.350`; RLOO reverses its registered sampler preference, and both matched
  samplers lose to uniform. Because all selected rates equal `.1`, tuned and
  common ledgers/checkpoints are byte-identical and their scientific summary
  content matches after removing phase/authorization labels. They are one
  scientific result, not independent replications. The compact repository
  includes all 24 block contrasts and a SHA-256/size manifest for 2,904
  scientific files, but the 5.08 GB full replay payload remains local and has
  no download URI yet; a clean clone cannot replay the historical checkpoints.
- The paid-probe attachment is complete: 12/12 development and 320/320
  confirmation runs passed the frozen accounting checks. The registered
  `u_16-ProCuRL` fixed-paid-AUC contrast is `+.004894`
  (`t(79)=1.9773`, `p=.05149`), below the `.02` SESOI and therefore
  unsupported. At the frozen cadence, the probed arms used about 93.2% of paid
  transitions for probes; ordinary uniform reached `.65149` fixed-paid AUC,
  versus `.33771` ProCuRL-env, `.33942` probe-sham uniform, and `.34261`
  `u_16`. This supports only the conclusion that probe cost dominated this
  actor-only, fixed-pool attachment. It does not establish inferiority of full
  PPO ProCuRL or performance under a cheaper refresh cadence. The compact
  release carries receipts and hashes; the 1,374,886,097-byte raw ledger
  remains external with SHA-256
  `b1f8756c249effab8c77101c8bca73ddf708a5e143c18fe8742fd5712fdd7c12`.
  The lock binds the source, runtime, gate, and artifacts internally, but no
  immutable public pre-execution commit establishes timing. See the
  [paid-probe result](frontier_rl/examples/ACROBOT_PROCURL_SELECTION_RESULTS.md)
  for the complete registered family and release boundary.
- **Submission stop rule:** all Mac experiments planned for this submission
  are complete. PLR, PAIRED, ACCEL, and full SFL studies are deferred to
  post-submission work; current effort is evidence recovery and the final
  paper/artifact rebuild, not launching another local experiment.

## Repo map

| path | contents |
|---|---|
| `paper/` | **The paper** — compact submission source in `body_iclr.tex` (wrapped by `main_iclr.tex` and `main.tex`), with the full research record retained in `body.tex`; the rendered website copy is `docs/paper-draft.pdf`. |
| `GUIDE.md` | Design guide: approaches tried, verification status of each, and what's next |
| `REPORT.md` | Full experiment report: math→algorithm→evidence chain, findings, goal assessment |
| `SCHEDULE.md` | Live experiment tracking: executing queue, decision trees, next wave |
| `REVIEW_NOTES.md` | Reviewer entry point: claim boundaries, lock provenance, current run status, and audit order |
| `GPU_EXPERIMENT_HANDOFF.md` | Ranked GPU continuation: LR-calibrated maze confirmation first, dose-matched Countdown conditional on asset recovery, with frozen design and release gates |
| `curriculum_maxrl/THEORY.md` | Exact coefficient-mass formulas, derived utility, myopic fixed-p allocation theorem, adaptive-T audit |
| `curriculum_maxrl/PROOFS.md` | Proof-level estimator conventions, practical `N-1` result, positive-part corollary, coefficient-mass limits, and hindsight moment/law conditions |
| `curriculum_maxrl/DESIGN.md` | Original integration design, hypotheses H1–H5, CPU validation tables |
| `curriculum_maxrl/RESEARCH.md` | Deep-research synthesis of modern curriculum RL (PAIRED/PLR/ACCEL, ALP-GMM, SFL learnability, RLVR curricula) — 3-vote adversarially verified against primary sources |
| `curriculum_maxrl/*.py` | CPU prototype: skill-chain testbed, 5 estimators, 5 teachers, experiment runners |
| `curriculum_maxrl/maze_gpu/` | GPU testbed: 1.26M-param transformer on 17×17 mazes, goal-distance curriculum (13 levels), pass@k eval, matched wall-clock sweep protocol + logs |
| `curriculum_maxrl/digits_factorial/` | Internally locked CPU exact-probability estimator × sampler factorial: compact analyses, receipt chain, external-payload content manifest, and frozen negative result; full ledgers/checkpoints are retained off-repository |
| `frontier_rl/` | Reusable grouped trainer, estimator/teacher implementations, CPU/Gym adapters, corrected artifacts, and external-environment protocols |
| `frontier_rl/examples/UNILAB_ROBOTICS_PROTOCOL_V1.md` | Mac-CPU robotics ladder separating reset-stream PPO curriculum tests from the exact grouped estimator experiment |
| `frontier_rl/examples/UNILAB_ROBOTICS_RESEARCH_ROADMAP_V2.md` | Root audit of all 40 native UniLab environments, fixed-target `D_8 -> J_7` math, curriculum integrity defects, and the falsifiable Stewart-to-Go2 robotics program |
| `frontier_rl/examples/UNILAB_STEWART_RESULTS_V1.md` | Audited three-seed development result for the exact grouped Motrix manipulation pilot, including its failed first task axis and next discriminating experiment |
| `frontier_rl/examples/UNILAB_STEWART_NATIVE_RESULTS_V2.md` | Current native UniLab result: 33 arm runs across target-preserving coefficient-mass and gradient-second-moment studies, mechanism gains, performance null, and next estimation gate |
| `docs/` | Static project website and curves exported only from retained corrected artifacts |
| `verl_integration/` | Production integration for the MaxRL verl fork: `curriculum.py` (drop-in module), patches for `main_ppo.py` / `ray_trainer.py`, SmolLM+GSM8K launch script |

## Quick start (CPU, numpy only; commands run from the repository root)

```bash
python3 curriculum_maxrl/run_experiment.py --steps 400 --seeds 5
python3 curriculum_maxrl/run_speed.py
python3 curriculum_maxrl/test_verl_curriculum.py
python3 -m frontier_rl.examples.run_skill_chain_ablation
```

### Artifact verification

From a clone with Git LFS and `uv` installed:

```bash
git lfs pull
bash reproduce.sh          # tests, endpoint derivations, all declared compact inputs
bash reproduce.sh --build  # additionally rebuild both LaTeX wrappers
```

The build needs a TeX installation and a Python environment with NumPy and
Matplotlib. The script provisions the pinned Gym/Digits environments through
`uv`; capped-HORA uses CPython 3.9.6 with NumPy 1.26.4 (set `HORA_PYTHON` to an
existing matching interpreter, or let `uv` provision it). This verifies the
compact release. It does not download the unshipped 5.08 GB Digits replay
payload, rerun evidentiary training, recover missing maze/Countdown data, or
turn summary-backed records into raw artifacts.

Gymnasium smoke check (Python ≥3.10; install `requirements-gym.txt` first):

```bash
python3 -m pip install -r requirements-gym.txt
python3 frontier_rl/examples/run_mountaincar_shared.py --quick
```

Omit `--quick` for the ten-seed, 500k-transition validation. Quick mode writes
`mountaincar_shared_quick.json` so it cannot overwrite the canonical result.

UniLab Stewart-platform probes run from the sibling UniLab worktree. The first
command reproduces the fixed-policy geometry diagnostic; the second exercises
the repaired 16-observation PPO task owner on CPU:

```bash
uv run --extra motrix python \
  ../curriculum-maxrl/frontier_rl/examples/unilab_stewart_base_rate.py \
  --output ../curriculum-maxrl/frontier_rl/examples/unilab_stewart_base_rate_seed0_4.json
uv run train --algo ppo --task stewart_balance_grouped --sim motrix \
  training.device=cpu training.no_play=true
```

The current exact grouped runner is native to the sibling UniLab worktree at
`scripts/train_grouped_maxrl.py`; it uses Hydra configs under
`conf/grouped_maxrl/`, strict owner-checkpoint provenance, and the 16-input
task owner.  See `frontier_rl/examples/UNILAB_STEWART_NATIVE_RESULTS_V2.md`.
The older independent 15-input runner and its development artifacts remain in
this repository as V1 history, not the current robotics claim.

Historical GPU maze testbed (needs torch; use the handoff's engineering memory
gate before any new scientific run):

```bash
python3 curriculum_maxrl/maze_gpu/train.py --teacher advmass --estimator maxrl --steps 300
python3 curriculum_maxrl/maze_gpu/analyze.py curriculum_maxrl/maze_gpu/matched_*.jsonl
```

## verl integration (into the MaxRL repo)

1. Copy `verl_integration/curriculum.py` to `verl/utils/curriculum.py`.
2. Apply `verl_integration/main_ppo.patch` and `ray_trainer.patch`
   (`git apply verl_integration/*.patch` from the MaxRL repo root).
3. Launch with:

```
+data.curriculum.enable=true
+data.curriculum.floor=0.1            # uniform replay floor (anti-forgetting)
+data.curriculum.decay=0.7            # pseudo-count decay (tracks the moving policy)
+data.curriculum.utility=advmass      # derived utility; "frontier" = older heuristic
```

Teacher state is checkpointed/restored automatically; wandb gets
`curriculum/visited_frac`, `curriculum/frac_dead_p_lt_0.05`,
`curriculum/frac_mastered_p_gt_0.9`. See `verl_integration/smollm_curriculum.sh`
for a full GSM8K recipe.

## Headline results and audit status

The CPU identities and skill-chain results below reproduce. A July 2026 audit
found that historical GPU logs classified both `K=0` and `K=N` zero-weight
groups as "dead," used the legacy `u_{N+1}` frontier score, trained every
level with the deepest response budget while evaluating level-specific
budgets, and let dense-hindsight loss scale with relabel count. The code now
separates all-fail from all-pass groups and exposes an exact `advmass` GPU
condition. Historical GPU AUC was also step-indexed despite wall-clock-matched
endpoints, and its legacy integration omitted the post-SFT AUC anchor.
Historical GPU numbers are exploratory evidence pending a corrected rerun.

On the retained 12-seed CPU skill-chain ablation (36 nested shared-skill tasks,
400 matched trainer steps), the checkpoint mean including step zero is 0.660
for uniform/no hindsight, 0.732 for the exact gamma=1 teacher, 0.781 for the
gamma=4 teacher, 0.866 for uniform+centered hindsight, and 0.886 for the
reference gamma=4 full stack at hindsight scale 1. The direct hindsight effect
under gamma=4 is +0.1050 [0.1012, 0.1087], and the teacher/concentration stack
remains +0.0205 [0.0190, 0.0223] above uniform+centered hindsight. All declared
effects survive Holm correction. The teacher x hindsight and gamma x hindsight
interactions are negative: the components each help, but with diminishing
returns rather than synergy.

The same retained sweep rises from 0.832 to 0.936 as centered-hindsight scale
moves from 0.25 to 8. This is sensitivity evidence, not an optimum: the best
point is at the tested boundary, and scaling also changes effective auxiliary
learning rate and optimizer work. Full protocol and raw curves are in
`frontier_rl/examples/skill_chain_component_ablation.json`.

On the corrected **tile-coded** Gymnasium MountainCar mechanism study (official
dynamics, custom nested binary thresholds, at least 500k transitions, ten
paired seeds), exact-mass
sampling at `γ=4` improves mean-pass AUC over uniform by +0.141 [95% paired
bootstrap CI 0.076, 0.202], and success-only hindsight adds +0.197 [0.160,
0.238]. Both survive Holm correction across nine AUC contrasts. Exact mass at
`γ=1` is not separated from uniform, the legacy `u_{N+1}` score, or
learnability; concentration is an empirical ingredient in this shared-policy
task, not a theorem. See `curriculum_maxrl/VALIDATION.md` V8.

A separate **neural** MountainCar V1R2 development study tested shared H64
against hardest-only and exact total-/active-capacity disjoint controls. All 15
runs and all reconstruction checks completed, but the predeclared feasibility
rule returned **NO-GO**: pooled groups contained 1,932 all-fail, 474 mixed, and
zero all-pass groups, while hardest-goal AUC was zero in every run. Supporting
mean-pass AUC deltas were small and descriptive only (`+0.0065104`,
`+0.0119792`, `+0.00546875`, and `+0.00429688`). Reserved seeds
`18000..18019` remain untouched. This null-headroom development result is not a
contradiction of the older positive tile-coded study: the policies, controls,
primary metrics, and evidentiary roles differ.

On the historical GPU maze testbed, the logged zero-weight-group rate was
~65% under uniform and ~49% under the frontier teacher. Because that counter
included all-pass groups, those percentages must not be read as corrected
`K=0` rates. New runs log `dead_groups` and `all_pass_groups` separately.

The production verl integration now assigns teacher slots from post-filter
dataset positions (rather than trusting potentially colliding source IDs),
validates feedback, and checkpoints a stateful sampler for mid-epoch resume.
Its patch files passed a local application check against an official MaxRL
checkout; that check does not yet retain the upstream commit hash.

## Neural Acrobot evidence and feasibility ledger

### What happened to the earlier Gym experiments

They were not deleted. The classic MountainCar and CartPole scripts and
artifacts introduced in commit \`6ba05c4\` remain on \`origin/main\`. The later
neural/audited Acrobot and MountainCar work entered research commit
\`14dbc9f\`, which was never merged into \`origin/main\`. Merge \`81daaff\`
then combined the newer \`TilePolicy(shared=...)\` API with older optional-Gym
callers using \`share_across_tasks=...\`, so that path failed at construction
even though the files remained. The default artifact command did not exercise
the optional Gymnasium lane.

This checkout repairs that merge regression, restores the historical
\`hindsight_estimator\` route, charges discarded DAPO redraw transitions, and
adds pinned Gymnasium tests to the release check. The evidence policy remains
selective: the old three-seed CartPole smoke study stays excluded, the
corrected tile-coded MountainCar study stays appendix-only, and the fresh
Acrobot threshold-family tournament is the classical-control experiment used
to isolate the curriculum score.

The Acrobot evidence is chronological and deliberately claim-narrow:

| protocol | status | what it permits |
|---|---|---|
| early neural V1 | **failed launch gate** | Pilot saturation and missing post-warmup all-fail exposure stopped confirmation. |
| early neural V2 | **failed development gate** | Disjoint controls missed the every-cell learning/headroom gate; no transfer or capacity-control confirmation launched. |
| V3 | **historical/descriptive after RNG audit** | The retained mean contrast is positive, but neighboring seeds reuse one numeric root across parameter and action RNG domains; its original paired inference is not treated as clean confirmation. |
| V4A | **stopped: feasibility gate failed** | Integrity checks passed and the fallback selected `U*=250`, but gate 3 failed in exactly 3/9 runs; V4B was not authorized or run. |
| V5A | **all launch gates passed** | Fresh 3×3 feasibility completed across 27 runs, selected `U*=250`, and independently authorized V5B without reading learning-outcome fields. |
| V5B | **completed; procedural NO-GO** | All 180 runs and raw-integrity checks passed, but the frozen analyzer failed exact diagnostic reconstruction; the official primary family is not authorized and no performance result or contrast is claimed. |
| fixed-pool tournament V1 | **aborted before primary-arm outcomes** | An independent audit found cross-seed RNG-root reuse and verifier gaps after one uniform run; the partial artifact was quarantined and the entire seed block burned. |
| fixed-pool tournament V2 | **complete; registered primary confirmed** | All 9 development and 60 confirmation runs were valid. The primary `u_16-p(1-p)` contrast cleared both its frozen +.01 point-estimate filter and exact `p≤.05`; the Holm-controlled `u_16-uniform` secondary was also supported. |

For V3's normalized target-uniform mean-pass AUC over actual transitions,
including initialization, uniform scored `0.648669` and the frontier-`u_16`
coefficient-mass teacher scored `0.685021`. Their paired difference was
`+0.0363524` over 20 nominally paired seeds. A later audit found that the
action RNG for seed `s` reused the numeric root used to initialize parameters
for seed `s+1`. Alternating-seed sensitivity estimates remain positive (even
seeds `+0.0251`; odd seeds `+0.0476`) but are imprecise and do not restore the
original sign-exchangeability argument. We therefore retain the observed
contrast and raw records as descriptive history, not clean confirmatory
evidence. Secondary final mean pass was `0.864258` for uniform and `0.916992`
for the teacher.

The completed [fixed-pool tournament V2](frontier_rl/examples/ACROBOT_CURRICULUM_TOURNAMENT_RESULTS.md)
uses the official Gymnasium [`Acrobot-v1`](https://gymnasium.farama.org/environments/classic_control/acrobot/),
one fixed eight-threshold family, and one task-agnostic shared H64 actor (640
parameters). It holds practical MaxRL, `N=16`, the two-million-transition
nominal budget, evaluation, and teacher-state machinery fixed while changing
only the selection rule. Runs complete their final group, so realized totals
have arm-specific but frozen-bounded overshoot rather than identical transition
counts. All 9 outcome-blind development runs and all 60 confirmatory runs were
valid.

The target-uniform normalized transition-AUC arm means were `0.6452319465`
(uniform), `0.6390719632` (`p(1-p)`), and `0.6871056515` (`u_16`). The registered
primary `u_16-p(1-p)` difference was `+0.0480336884`, with paired bootstrap 95%
CI `[0.0209366676, 0.0738485654]`, exact two-sided sign-flip `p=0.0033607483`,
and 15/20 positive pairs. It met both the frozen `+0.01` point-estimate filter
and `p≤0.05`, so the primary is confirmed. In the Holm-controlled secondary
family, `p(1-p)-uniform` was `-0.0061599834` (CI
`[-0.0226437219, 0.0121954971]`, raw/adjusted `p=0.507843`; not supported),
whereas `u_16-uniform` was `+0.0418737050` (CI
`[0.0218239396, 0.0605859853]`, raw `p=0.000808716`, Holm
`p=0.001617432`, 17/20 positive; supported).

This is the protocol's **P+/U+** outcome: score-shape evidence for one fixed
Acrobot pool under one shared H64, 640-parameter practical-MaxRL learner at
`N=16`. It is not a full ProCuRL, SFL, PLR, PAIRED, ACCEL, or ALP-GMM result;
it has no held-out-task generalization test and no prospectively powered sample
size. The sign-flip interpretation assumes paired-sign exchangeability, and
clearing the `+0.01` point-estimate filter does not establish a population
effect of at least `+0.01`.

Tournament provenance SHA-256 hashes are: source/runtime lock
`0e6438d42ddc53b89d774233805c465dc562bb6be5f8ac93ecf8a4d09b5d9af3`;
development raw
`c616912569f4d19e36ea4a8685616a35bef037934e5c8d366ee7bd51bb2c3311`;
development gate
`6dc908e22e874550e0536f1fcd52f2b3a1768d1a89c510275bef7efc2e2baac6`;
confirmation raw
`f533d0b84cdb3f7d3ede4bc4c94aa11e3b0ffc58c8bc7ea1a26491476873b2c6`;
and locked analysis
`463fa1a01d95922976f09f75b21f6d8f2c6a8d256081ebedfa4ba968a06f356b`.
These digests bind the source, runtime, gate, and artifacts internally, but no
immutable public pre-execution commit in this checkout establishes their
timing.

V4A then tested only whether the planned optimizer-matched hindsight factorial
was feasible; all nine Stage-A cells used hindsight scale zero. The independent
verifier reproduced the artifact and selected the registered fallback
`U*=250`. All gates except gate 3 passed. Gate 3 required at least ten
positive, one-to-one, nonmutating previews in every run, but exactly three runs
had only `8`, `5`, and `6` previews. The projected serial runtime for the
90-run factorial was `3.452702` hours, within its gate, but the preview failure
stopped the protocol: Stage B was not authorized and was not run. This is a
feasibility stop, not evidence for or against hindsight efficacy.

V4A provenance hashes are
`b19488783e1adba8cbac44ce8256c725a4470d8108c1192f9491ecc4882f1d8c`
(lock),
`69b827dc425014f3b568186981e9c24d95158c72653125e0ade181272def2891`
(artifact), and
`c633e09df8e056f1589e631ff4d311913e1ac5594c3647790acc4b05990fca88`
(independent report). In that report, top-level `all_checks_passed=true`
means verifier integrity and recomputation passed; the launch decision is
instead recorded by `gates.all_pass=false` and
`stage_b_factorial_authorized=false`. The frozen lock's direct-path analyzer
command has a module-import defect; the exact working `python -m` invocation is
recorded in
[`ACROBOT_HINDSIGHT_V4_ERRATA.md`](frontier_rl/examples/ACROBOT_HINDSIGHT_V4_ERRATA.md).

V5A replaced neither V4A nor its stopped decision. It used fresh seeds
`15000..15002`, ran all nine learning-rate×hindsight-scale cells, completed all
27 runs, passed every outcome-field-blind technical gate, and selected the
registered fallback `U*=250`. The independently verified launch decision
authorized V5B. V5B used fresh seeds `16000..16019` for a 20-seed 3×3
factorial with four predeclared contrasts. All 180 runs completed with zero run
failures; a post-hoc forensic audit covered 53,510 group records, 45,000
updates, and 1,080 checkpoints.

The frozen analyzer then failed deterministically: the runner's NumPy
step-norm reductions and the analyzer's Python scalar reductions differed in
377 of 720 diagnostic floats. The largest absolute difference was
`1.9984014443252818e-15`, or 11 ULP. Step norms are diagnostics, but the frozen
acceptance rule requires exact runner/analyzer dictionary equality. The
official V5B primary family is therefore a **procedural NO-GO**. No outcome,
cell ranking, contrast, sign, or hindsight-effect result is claimed. A
post-hoc tolerance-aware compatibility audit passed the remaining checks but
is non-authorizing; a reviewed tolerance-aware verifier and fresh V5C seeds
are required. See the
[`V5B verification erratum`](frontier_rl/examples/ACROBOT_HINDSIGHT_V5B_VERIFICATION_ERRATUM.md)
and
[`forensic verification report`](frontier_rl/examples/acrobot_hindsight_v5b_forensic_verification.json).

The V5A lock, artifact, and verification hashes are
`5c277413c5238f5839d281e09810537221a16737f831a498a3e0217ca5b1502e`,
`9cf741c91dcb82218cada9b451b76e0811c67aa4cbf1786ac0ba926806479b0a`,
and `a46b5e9f732b7f9e1796e2d4a2ff344c9ff738574c464b28631e884faaa6ba19`.
The V5B amendment and lock hashes are
`11975381874842bc3019074ea9d8168006c0517982ac11e00ad0b488e7671f36`
and `dfc930bbaf8e51c96fd1dab5851179457fce4f151def8c138ddf0cf17402bcf2`;
the completed artifact hash is
`c633886a121906ee2bceb03f3117e4bea5dc20ab314e43f9b702ef8d88f495ac`.

**Artifact storage.** Raw JSON paths explicitly listed in `.gitattributes`
use Git LFS; locks, compact manifests, hashes, and verification reports remain
ordinary Git files. The HORA JSONs are ordinary Git artifacts. The Digits
ledgers/checkpoints are different: they remain off-repository and are covered
by a relative-path SHA/size manifest with a null download URI. Run
`git lfs install && git lfs pull` to materialize the declared LFS paths, but do
not mistake that for retrieval of the external Digits replay bundle.

**Provenance boundary.** “Registered,” “sealed,” and “predeclared” in this
repository refer to local source/runtime locks created before the corresponding
seed block was executed. They are not externally timestamped preregistrations.
At reviewed snapshot `2dbda4e`, the V3-and-later manifests matched the listed
files; post-lock Cosmos support subsequently evolved three shared core files
(`estimators.py`, `trainer.py`, and `adapters/__init__.py`). The locks remain
unchanged and their exact bytes are recoverable from that commit. The older V2
runner remains the historical exception whose exact locked bytes are not in
the retained tree. See `REVIEW_NOTES.md` for the exact scope.

## Citation / provenance

Builds on the MaxRL paper and codebase (Tajwar, Zeng et al., 2026). The
curriculum design draws on PLR (Jiang et al.), PAIRED (Dennis et al.), ALP-GMM
(Portelas et al.), and SFL learnability (Rutherford et al., NeurIPS 2024) — see
`curriculum_maxrl/RESEARCH.md` for the verified literature synthesis.
