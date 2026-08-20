# Curriculum-MaxRL

**Teacher-guided curriculum learning driven by the MaxRL objective's own algebra.**

Research codebase exploring the integration of curriculum learning (teacher–student,
ZPD/learnability targeting) with Maximum Likelihood Reinforcement Learning
([MaxRL, arXiv:2602.02710](https://arxiv.org/abs/2602.02710)). Built against the
official MaxRL implementation (a [verl](https://github.com/verl-project/verl) fork).

## The idea in one paragraph

MaxRL reweights per-prompt gradients by ~1/pass-rate (a truncated Maclaurin expansion
of `log p`), which acts as an *implicit, gradient-level* curriculum — but it cannot
rescue prompts whose rollout groups come back all-fail (K=0 → group dropped, zero
gradient), and it wastes compute re-rolling mastered prompts. We add an *explicit,
data-level* teacher whose utility function is **derived from the estimator itself**:
the expected total |advantage| a prompt receives from a group of N rollouts is exactly

```
E[Σ|w|] = 2 · (pass@N(p) − pass@1(p)) = 2 · ((1−(1−p)^N) − p)
```

— twice the probability the prompt is *solvable within N attempts but not within one*.
More generally, every finite-group estimator induces an **activity geometry** over pass
rates (the expected L1 mass of the coefficient vector it can emit, a Bernstein polynomial
in p); practical MaxRL is notable because that sum collapses to a closed form, and RLOO's
collapses to 2p(1−p) — so the canonical learnability score is the N=2 slice. The teacher
Thompson-samples a decayed Beta posterior over each prompt's pass rate and samples
proportional to this utility; the optimal per-prompt rollout allocation is greedy
water-filling on the marginal `p(1−p)^N`.

## Where the claim stands (read this before the ladder)

Activity says where an update is *available*. It does not say where training has the most
value. Four preregistered results draw that line:

- **The shape helps.** In a fixed Acrobot pool at deployed N=16, `u_16` beats `p(1−p)` by
  +.0480 (95% CI [+.0209, +.0738]), replicated on two further platforms (+.0322, +.0307).
- **The peak location does not.** Holding the estimator at N=16 and sweeping only the score
  exponent, performance keeps rising *past* N=16 (argmax at `u_64`). The harder-peaked
  shape is what helps — not the derived peak.
- **It is not a standalone signal.** Dropped into robust PLR on AMaze in place of MaxMC,
  activity does not beat upstream: one Bernoulli per level visit cannot replace a critic
  read at every timestep.
- **The `p`-only score does not survive a coarser task unit.** MAZE-SCORE, 48
  preregistered blocks at 1.26M parameters and deployed N=32, where the
  curriculum scores a *level* that aggregates many concrete mazes: `u_32`
  *loses* to `p(1−p)` (−.0032, CI [−.0054, −.0011], 15/48 positive) —
  practically ruled out at the registered +.005 SESOI. Both adaptive samplers
  still beat uniform (`u_32` − uniform = +.0089), so curriculum sampling
  helps; what fails is the claim that the rollout-aware shape is the better one.

**Why, and it is a theorem rather than a diagnosis.** Realized group mass is
the deterministic `M(K) = 2(1 − K/N)·1{K>0}`, so for **any** joint binary
group law — no independence, no identical distribution —

```
A_N(Q) = 2 ( Pr(K>0) − E[K]/N )
```

The familiar `2(1 − p − (1−p)^N)` is only its conditional-i.i.d. slice. A group
here repeats **one** concrete maze N times, so rollouts inside it *are* i.i.d.;
what is coarse is the *scored* unit. Scoring an aggregate `z` by its mean pass
rate therefore over-predicts activity by exactly twice its excess all-fail
probability:

```
A_N(p̄_z) − 2·E_X[u_N(p_X)] = 2 [ Pr(K=0 | z) − (1−p̄_z)^N ]  ≥ 0
```

Verified to floating point on the campaign (max deviation 2.8e-16 over 41,101
cells, invariant to window width). At p̂≈.11 the plug-in predicts 2.2% silent
groups and **51.2%** are silent. Because `|u_N''|` at `u_N`'s own peak is
≈34.7 for N=32 against `u_2(p̄) − E[u_2(P)] = Var(P)` exactly, the
harder-peaked score pays far more for the same heterogeneity: clustered on the
48 seed blocks, `u_32` realizes .580 [.570,.590] of its predicted mass against
.703 [.691,.715] for `p(1−p)`, negative in 48/48 blocks.

*The estimator defines the coefficient map; the curriculum defines the unit
over which that map is averaged. These operations do not commute.*

**This organizes a post-hoc cross-study pattern; it does not predict learning
signs.** The corollary predicts coefficient-activity calibration only. The observed
score contrasts align with whether the curriculum scored the unit the estimator
consumed:

| study | scored unit | = the i.i.d. unit? | `u_N` vs `p(1−p)` |
|---|---|---|---|
| CPU skill chain | task, exact gradients | yes | +.031 → +.191 |
| Acrobot ×3 | predicate; fresh reset seed *per rollout* | yes | +.048, +.032, +.031 |
| Digits | one example, exact class probability | yes | +.208, +.177 |
| **MAZE-SCORE** | **level** — one maze per group, posterior pools over the level | **no** | **−.0032** |

Read *post hoc*, across studies not designed to test it. The prospective P0 test
holds the substrate, estimator, budget, seeds, and four-moment count-law posterior
fixed and varies *only* the score functional: the i.i.d.-at-the-mean plug-in versus
realized MaxRL count-law activity. Its 48 paired blocks (seeds 3001--3048) were
frozen before launch and are running blind; no endpoint is claimed here.

So: coefficient activity is an estimator-conditioned *source of curriculum
hypotheses*, not a universal measure of learning utility — and a mean pass
rate determines it only when the scored unit is the one the estimator
consumes. The ICLR submission body
(`paper/body_iclr.tex`, rendered at [`docs/paper-iclr.pdf`](docs/paper-iclr.pdf)) is
written to exactly that scope.

## The experiment ladder — what each experiment asks, and what to expect

Every experiment isolates one or two *channels* of the method. Reader's map:
the **teacher** reallocates compute (channel 1), **hindsight** creates signal
from failures (channel 2), the **objective** decides whether a curriculum is
safe at all (channel 3). Predictions are pre-registered (committed before any
cell finishes) so results are readable as confirmations or refutations, not
post-hoc stories.

| experiment | the question | what we expected (pre-registered) | outcome — including our own retractions |
|---|---|---|---|
| **CPU skill-chain** (36 tasks, exact gradients) | do the channels work at all? | teacher > uniform; recycling adds | ✓ 0.65→0.73→0.89 — **corrected**: a floor-and-γ-matched true-p oracle *ties* the full stack (0.8885 vs 0.8895; "beats the oracle" is retracted); recycling adds on top of even the oracle (0.8935) |
| **V5 frontier-heavy regime** (max pool p=1e-5) | what happens when NO task is samplable? | pure samplers get exactly 0; recycling invents the curriculum below the pool | ✓ categorical: 0.93 AUC vs 0.00 for uniform/DAPO/teacher-alone |
| **Balanced maze factorial** ({maxrl,grpo}×{uniform,teacher}×6 blocks, 250 fixed steps, pre-registered) | does the estimator coverage divergence survive a clean design? | ≥5/6 paired blocks MaxRL>GRPO under both samplers | **✗ FAILED (3/6, 4/6) — the cohort's zero-exception claim is retracted.** Wave 2 re-registered time-integrated coverage as primary and landed 6/6 fresh blocks under *each* sampler (block-mean +.0195); reported as an estimator-conditioned ordering at common optimizer settings, not universal estimator superiority |
| **E-LLM-1: GSM8K 2×2** (SmolLM2-360M, one A10G) | do channels 1+3 transfer to LLM RLVR? | P-G2: grpo+teacher does NOT beat grpo | Registered run landed P-G2 (only regressing cell); **replication seed with weaker measured steering climbed instead — 1-of-2 seeds, dose-dependent, not established**; teacher-deficit *direction* is 2/2 seeds |
| **E-LLM-2/2b: Countdown** (exact-verifier recycling, v2 pool) | channel 2 at LLM scale | recycling ignites unreachable tiers | v1 pool: both nulls (guesser-saturated). The reported 3-seed aggregate is mean@16 up while VERL bootstrap best@16 falls; this is a coverage proxy, not standard pass@16, and complete seed records are missing. The buggy-decay gate did not validate; corrected strong gating failed. Higher-dose replay does not isolate dose from direction; matched E2/E2b controls failed delivery and E2c prospectively retains raw outcomes. |
| **Jugs (water-measuring)** | does the whole family fail together where no band exists? | pre-registered all-null | ✓ all-null landed — the negative control |

Full LLM-experiment roadmap with novelty checks and differentiation map:
[`NEXT_EXPERIMENTS.md`](NEXT_EXPERIMENTS.md). Latest LLM results:
[`GSM8K_ANALYSIS.md`](GSM8K_ANALYSIS.md).

## Repo map

| path | contents |
|---|---|
| `paper/` | **The paper** — ICLR submission body `body_iclr.tex` (9-page bound, rendered at `docs/paper-iclr.pdf`); `main.tex`/`body.tex` is the extended research record (`docs/paper-draft.pdf`); figures with vendored data. (`PAPER.md` is the superseded markdown outline) |
| `GUIDE.md` | Design guide: approaches tried, verification status of each, and what's next |
| `REPORT.md` | Full experiment report: math→algorithm→evidence chain, findings, goal assessment |
| `SCHEDULE.md` | Live experiment tracking: executing queue, decision trees, next wave |
| `curriculum_maxrl/THEORY.md` | Exact advantage-mass formulas per estimator (MC-verified), derived teacher utility, optimal allocation, adaptive-T negative result |
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
+data.curriculum.decay=0.9            # posterior decay (tracks the moving policy)
+data.curriculum.utility=advmass      # derived utility; "frontier" = older heuristic
```

Teacher state is checkpointed/restored automatically; wandb gets
`curriculum/visited_frac`, `curriculum/frac_dead_p_lt_0.05`,
`curriculum/frac_mastered_p_gt_0.9`. See `verl_integration/smollm_curriculum.sh`
for a full GSM8K recipe.

## Headline validated results

On the CPU skill-chain testbed (36 tasks, initial pass rates 10^-level, 5 seeds):

- **Curriculum and MaxRL are complementary.** Teacher fixes the K=0 dead zone MaxRL
  can't reach; MaxRL extracts more per in-band group. `frontier+maxrl` is fastest to
  the deepest level (206 steps vs 248 uniform+maxrl vs 262 zpd+grpo) and best in the
  beyond-frontier-heavy regime (0.961 vs 0.871 / 0.847 for each alone).
- **MaxRL already does most of what a curriculum does on moderate distributions**
  (+0.01 from teacher) while GRPO needs the teacher badly (+0.23) — empirical support
  for the paper's "implicit curriculum" reading.
- **The derived advantage-mass utility matches the hand-tuned ZPD band with zero
  band hyperparameters.**

On the GPU maze testbed: uniform sampling wastes ~65% of rollout groups (dead K=0);
the frontier teacher cuts that to ~49% and runs ~2× more steps in the same wall-clock.
Matched-wall-clock sweep in progress; see `curriculum_maxrl/maze_gpu/EXPERIMENTS.md`.

## Citation / provenance

Builds on the MaxRL paper and codebase (Tajwar, Zeng et al., ICML 2026). The
curriculum design draws on PLR (Jiang et al.), PAIRED (Dennis et al.), ALP-GMM
(Portelas et al.), and SFL learnability (Rutherford et al., NeurIPS 2024) — see
`curriculum_maxrl/RESEARCH.md` for the verified literature synthesis.
