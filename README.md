# Curriculum-MaxRL

**Teacher-guided curriculum learning driven by the MaxRL objective's own algebra.**

Research codebase exploring the integration of curriculum learning (teacher–student,
ZPD/learnability targeting) with Maximum Likelihood Reinforcement Learning
([MaxRL, arXiv:2602.02710](https://arxiv.org/abs/2602.02710)). Built against the
official MaxRL implementation (a [verl](https://github.com/verl-project/verl) fork).

## Current evidence status (2026-08-04)

The paper, `VALIDATION_2026-08-04.md`, and committed result JSON files are the
authoritative evidence for this revision. Several older research notes in this
repository record hypotheses and historical runs; they are not current claims.
In particular, the historical maze and Countdown recyclers did not implement
the proposed common-destination contrast, and no neural or LLM run yet combines
that relabeler with the exact coefficient-mass sampler.

## The idea in one paragraph

For the practical centered MaxRL estimator used here, constant-reward groups
are dropped. Its expected absolute coefficient mass for a prompt with pass
rate `p` and `N` rollouts is exactly

```
E[Σ|w|] = 2 · (pass@N(p) − pass@1(p)) = 2 · ((1−(1−p)^N) − p)
```

— twice the probability the prompt is solvable within `N` attempts but not
within one. This coefficient-mass profile peaks at
`p* = 1 − N^(−1/(N−1))`. It motivates a posterior task sampler, but it is
implementation-specific: raw MaxRL and full-control-variate MaxRL have order
`N`, while the practical drop-all-fail estimator has order `N−1`. Full
CV can assign negative coefficients on all-fail groups; in the new 20-seed
tabular control, that branch does not reliably bootstrap the extreme frontier,
whereas exact shared-prefix hindsight does.

## Evidence ladder

| setting | independent units | supported conclusion |
|---|---:|---|
| Exact-score skill chains | 20 paired seeds for new controls | full CV's all-fail coefficient branch is unreliable in the extreme frontier; shared-prefix hindsight unlocks it; three tuned centered estimators finish within 0.005 AUC |
| Historical neural maze | 3 shared warm-start blocks; heterogeneous archive | descriptive estimator-associated coverage pattern under a legacy frontier-ALP teacher and positive-only recycler |
| GSM8K 360M | 1–2 seeds per cell | treatment-delivery diagnosis only; posterior starvation and replacement confounding remain |
| Historical Countdown v2 | three-seed aggregate endpoints; two trajectory pairs retained | legacy per-trace recycling has higher mean success and lower pass@16 coverage; the gate result is preliminary |
| Gym binary-success controls | 3 paired seeds | task spread creates usable updates and shared parameters enable transfer |
| IsaacLab Anymal-C | 1 seed | hypothesis-consistent pilot only |

The exact protocol boundaries, withdrawals, and missing evidence are recorded
in `VALIDATION_2026-08-04.md`.

Historical planning and analysis notes remain in
[`NEXT_EXPERIMENTS.md`](NEXT_EXPERIMENTS.md) and
[`GSM8K_ANALYSIS.md`](GSM8K_ANALYSIS.md); consult the paper and validation
record before treating any result in those notes as current evidence.

## Repo map

| path | contents |
|---|---|
| `PAPER.md` | **The story** — 30-second pitch, why this direction, the three insights, what problem it resolves, real + hidden benefits |
| `GUIDE.md` | Design guide: approaches tried, verification status of each, and what's next |
| `REPORT.md` | Full experiment report: math→algorithm→evidence chain, findings, goal assessment |
| `SCHEDULE.md` | Live experiment tracking: executing queue, decision trees, next wave |
| `curriculum_maxrl/THEORY.md` | Exact advantage-mass formulas per estimator (MC-verified), derived teacher utility, optimal allocation, and the corrected/open adaptive-T analysis |
| `curriculum_maxrl/DESIGN.md` | Original integration design, hypotheses H1–H5, CPU validation tables |
| `curriculum_maxrl/RESEARCH.md` | Deep-research synthesis of modern curriculum RL (PAIRED/PLR/ACCEL, ALP-GMM, SFL learnability, RLVR curricula) — 3-vote adversarially verified against primary sources |
| `curriculum_maxrl/*.py` | CPU prototype: skill-chain testbed, 5 estimators, 5 teachers, experiment runners |
| `curriculum_maxrl/maze_gpu/` | GPU testbed: 1.26M-param transformer on 17×17 mazes, goal-distance curriculum (13 levels), pass@k eval, matched wall-clock sweep protocol + logs |
| `verl_integration/` | Production integration for the MaxRL verl fork: `curriculum.py` (drop-in module), patches for `main_ppo.py` / `ray_trainer.py`, SmolLM+GSM8K launch script |

## Quick start (CPU, numpy only)

```bash
cd curriculum_maxrl
python3 run_experiment.py --steps 400 --seeds 5   # teacher × estimator sweep, ~1 min
python3 run_speed.py                              # learning-speed + adaptive-N comparison
python3 test_verl_curriculum.py                   # unit tests for the verl module
```

GPU maze testbed (needs torch + one ~24GB GPU):

```bash
cd curriculum_maxrl/maze_gpu
python3 train.py --teacher frontier --estimator maxrl --steps 300  # or --max-seconds 2400
python3 analyze.py matched_*.jsonl
```

## verl integration (into the MaxRL repo)

1. Copy `verl_integration/curriculum.py` to `verl/utils/curriculum.py`.
2. Apply `verl_integration/main_ppo.patch` and `ray_trainer.patch`
   (`git apply verl_integration/*.patch` from the MaxRL repo root).
3. Launch with:

```
+data.curriculum.enable=true
+data.curriculum.floor=0.1            # uniform replay floor (anti-forgetting)
+data.curriculum.decay=0.7            # historical tuned default; revalidate per suite
+data.curriculum.utility=advmass      # derived utility; "frontier" = older heuristic
```

Teacher state is checkpointed/restored automatically; wandb gets
`curriculum/visited_frac`, `curriculum/frac_dead_p_lt_0.05`,
`curriculum/frac_mastered_p_gt_0.9`. See `verl_integration/smollm_curriculum.sh`
for a full GSM8K recipe.

## Reproduced headline controls

- In the balanced 20-seed common-rate test, raw MaxRL reaches
  0.7209 ± 0.0210 AUC, full CV 0.6934 ± 0.0189, and practical MaxRL
  0.7183 ± 0.0231. This is a mechanism control, not a tuned ranking.
- In the frontier-heavy pool (maximum initial pass rate 1e-5), full CV
  invokes the update callback on every all-fail group but reaches only
  1.740e-6 mean AUC. Practical MaxRL plus exact shared-prefix hindsight
  reaches 0.9269 ± 0.0032 and wins all 20 paired seeds.
- With identical held-out schedules and separate tuning seeds, practical
  MaxRL, GRPO, and RLOO finish within 0.005 AUC of one another on a
  near-ceiling tabular pool.
- Historical neural and LLM results are explicitly descriptive or pilot
  evidence. They do not constitute end-to-end validation of the proposed
  exact sampler plus shared-destination recycler.

## Citation / provenance

Builds on the MaxRL paper and codebase (Tajwar, Zeng et al., ICML 2026). The
curriculum design draws on PLR (Jiang et al.), PAIRED (Dennis et al.), ALP-GMM
(Portelas et al.), and SFL learnability (Rutherford et al., NeurIPS 2024) — see
`curriculum_maxrl/RESEARCH.md` for the verified literature synthesis.
