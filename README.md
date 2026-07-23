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

## Repo map

| path | contents |
|---|---|
| `PAPER.md` | **The story** — 30-second pitch, why this direction, the three insights, what problem it resolves, real + hidden benefits |
| `FRAMEWORK.md` | Research contract: assumptions, reference algorithm, target-mixture choice, hindsight gates, ablation matrix, and claim ladder |
| `GUIDE.md` | Design guide: approaches tried, verification status of each, and what's next |
| `REPORT.md` | Full experiment report: math→algorithm→evidence chain, findings, goal assessment |
| `SCHEDULE.md` | Live experiment tracking: executing queue, decision trees, next wave |
| `REVIEW_NOTES.md` | Reviewer entry point: claim boundaries, lock provenance, current run status, and audit order |
| `curriculum_maxrl/THEORY.md` | Exact coefficient-mass formulas, derived utility, myopic fixed-p allocation theorem, adaptive-T audit |
| `curriculum_maxrl/DESIGN.md` | Original integration design, hypotheses H1–H5, CPU validation tables |
| `curriculum_maxrl/RESEARCH.md` | Deep-research synthesis of modern curriculum RL (PAIRED/PLR/ACCEL, ALP-GMM, SFL learnability, RLVR curricula) — 3-vote adversarially verified against primary sources |
| `curriculum_maxrl/*.py` | CPU prototype: skill-chain testbed, 5 estimators, 5 teachers, experiment runners |
| `curriculum_maxrl/maze_gpu/` | GPU testbed: 1.26M-param transformer on 17×17 mazes, goal-distance curriculum (13 levels), pass@k eval, matched wall-clock sweep protocol + logs |
| `verl_integration/` | Production integration for the MaxRL verl fork: `curriculum.py` (drop-in module), patches for `main_ppo.py` / `ray_trainer.py`, SmolLM+GSM8K launch script |

## Quick start (CPU, numpy only; commands run from the repository root)

```bash
python3 curriculum_maxrl/run_experiment.py --steps 400 --seeds 5
python3 curriculum_maxrl/run_speed.py
python3 curriculum_maxrl/test_verl_curriculum.py
python3 -m frontier_rl.examples.run_skill_chain_ablation
```

Gymnasium smoke check (Python ≥3.10; install `requirements-gym.txt` first):

```bash
python3 -m pip install -r requirements-gym.txt
python3 frontier_rl/examples/run_mountaincar_shared.py --quick
```

Omit `--quick` for the ten-seed, 500k-transition validation. Quick mode writes
`mountaincar_shared_quick.json` so it cannot overwrite the canonical result.

GPU maze testbed (needs torch + one ~24GB GPU):

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

## Neural Acrobot confirmation and feasibility ledger

The Acrobot evidence is chronological and deliberately claim-narrow:

| protocol | status | what it permits |
|---|---|---|
| V1 | **failed launch gate** | Pilot saturation and missing post-warmup all-fail exposure stopped confirmation. |
| V2 | **failed development gate** | Disjoint controls missed the every-cell learning/headroom gate; no transfer or capacity-control confirmation launched. |
| V3 | **registered decision supported** | Twenty paired seeds confirm only positive shared-H64 curriculum efficacy with hindsight off. |
| V4A | **stopped: feasibility gate failed** | Integrity checks passed and the fallback selected `U*=250`, but gate 3 failed in exactly 3/9 runs; V4B was not authorized or run. |
| V5A | **all launch gates passed** | Fresh 3×3 feasibility completed across 27 runs, selected `U*=250`, and independently authorized V5B without reading learning-outcome fields. |
| V5B | **completed; procedural NO-GO** | All 180 runs and raw-integrity checks passed, but the frozen analyzer failed exact diagnostic reconstruction; the official primary family is not authorized and no performance result or contrast is claimed. |

For V3's normalized target-uniform mean-pass AUC over actual transitions,
including initialization, uniform scored `0.648669` and the frontier-`u_16`
coefficient-mass teacher scored `0.685021`. Their paired difference was
`+0.0363524` (95%
paired-seed bootstrap interval `[0.0164536, 0.0553949]`; exact two-sided
sign-flip `p=0.00263977`; `n=20`). The observed mean met the registered
`>=+0.03` decision rule and the test met `p<=0.05`, supporting the
preregistered decision and a positive local shared-policy effect. Because the
interval begins at `0.0164536`, it does not show that the population effect is
above `+0.03`. Secondary final mean pass was `0.864258` for uniform and
`0.916992` for the teacher.

The [official Gymnasium `Acrobot-v1`](https://gymnasium.farama.org/environments/classic_control/acrobot/)
experiment uses a fixed eight-threshold family and one task-agnostic shared
H64 policy. V3 does not establish transfer causality, a parameter-sharing or
capacity advantage, hindsight efficacy or scale, wall-clock efficiency, or
generalization beyond Acrobot; V1/V2 remain failed gates. Artifact SHA-256:
`30da4b9759828acb9357f5518a48196a6be98d314dffb7830b0ba4f89a31e423`.
Independent verification-report SHA-256:
`1bb604925b36050c6b1520fce847919d7962ec8f0e300b50de49242b70b7b394`.

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

**Artifact storage.** The seven raw experiment JSON files larger than 1 MB are
stored with Git LFS; locks, manifests, hashes, and verification reports remain
ordinary Git files for lightweight review. Run `git lfs install` and
`git lfs pull` to materialize the raw artifacts. Without them, the test suite
still runs the source/lock checks and skips only tests that require raw bytes.

**Provenance boundary.** “Registered,” “sealed,” and “predeclared” in this
repository refer to local source/runtime locks created before the corresponding
seed block was executed. They are not externally timestamped preregistrations.
An audit of current bytes finds one historical exception: the V2 lock records
an earlier `run_acrobot_neural.py` hash, while the current file is the V3-era
runner. The V2 lock still discloses the expected historical hash, but those
runner bytes are not present at HEAD. The V3 and later listed source manifests
match the current checked-in bytes. See `REVIEW_NOTES.md` for the exact scope.

## Citation / provenance

Builds on the MaxRL paper and codebase (Tajwar, Zeng et al., 2026). The
curriculum design draws on PLR (Jiang et al.), PAIRED (Dennis et al.), ALP-GMM
(Portelas et al.), and SFL learnability (Rutherford et al., NeurIPS 2024) — see
`curriculum_maxrl/RESEARCH.md` for the verified literature synthesis.
