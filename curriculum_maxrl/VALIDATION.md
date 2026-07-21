# Validation results (CPU, exact gradients — `run_validation.py`)

Companion to PROOFS.md: each proposition's empirical check plus the
explore/exploit measurements that inform the teacher design. Skill-chain
testbed, 5 seeds unless noted.

## V1 — Hindsight gradient fidelity (validates Prop. 6)

Probe: a level-11 task at true p = 0.0005 (99% of groups dead). Relabel each
dead group to its deepest correct prefix j; compare the relabeled gradient to
the *exact* success-conditioned ML gradient of the level-j task, and to
fresh unbiased groups sampled directly on that task:

| prefix j | n groups | hindsight cosine (per-group) | fresh-group cosine | cosine of MEAN hs gradient |
|---|---|---|---|---|
| 8 | 1811 | 0.861 ± 0.071 | 0.854 ± 0.078 | **1.000** |
| 9 | 1876 | 0.948 ± 0.016 | 0.946 ± 0.017 | **1.000** |
| 10 | 276 | 0.956 ± 0.012 | 0.958 ± 0.007 | **1.000** |

**Reading:** per-group, the hindsight gradient is exactly as well-aligned as
an unbiased fresh group (same cosine, same spread); in the mean it converges
to the true gradient with cosine 1.000. On structures where the relabeled
conditional law matches (Prop. 6a), hindsight is not a biased approximation
— it *is* the ML gradient of the relabeled task, harvested from compute that
would otherwise produce zero. The caveat that survives: coverage drift
(relabeled goals are policy-reachable ones), not gradient direction.

## V2 — The oracle gap (what exploration costs)

Teacher variants, identical training loop, utility = advantage mass:

| teacher | AUC | final | advantage mass collected |
|---|---|---|---|
| uniform | 0.650 ± 0.006 | 0.966 | 720 |
| Thompson posterior (decay 0.9) | 0.700 ± 0.005 | 0.979 | 838 |
| **oracle (true p)** | **0.851 ± 0.002** | 0.985 | 841 |

Two lessons. (1) The Thompson teacher already collects **as much advantage
mass as the oracle** (838 vs 841) — mass-at-collection is nearly saturated.
(2) Yet the oracle's AUC is far higher: *when* you collect matters, not just
how much. The oracle moves to each level the moment it becomes learnable;
the posterior arrives ~1 posterior-lag later. **The remaining gap is a
tracking problem, not a signal-quantity problem.**

## V2b — Closing the tracking gap

- **Faster decay** (posterior forgets faster → tracks the moving student):
  AUC 0.681 / 0.700 / **0.728** / 0.719 / 0.727 at decay 0.95 / 0.9 / 0.7 /
  0.5 / 0.3. Decay 0.7 closes ~19% of the oracle gap for free; too-fast
  decay adds Thompson noise back.
- **Monotone (isotonic) projection**: difficulty is ordered within a chain,
  so true pass rates are monotone in level. Projecting Thompson draws onto
  nonincreasing sequences (pool-adjacent-violators) shares statistical
  strength across levels: AUC 0.711 at decay 0.9, **0.733 combined with
  decay 0.7** — together ~22% of the oracle gap. Structure-aware posteriors
  are the right direction; per-prompt independence wastes the difficulty
  ordering that curricula have by construction.

## V3 — Explore/exploit floor curve (validates Prop. 7's design reading)

AUC is *flat* (0.699–0.712) for floor ∈ [0, 0.4] and only collapses at
floor = 1 (pure uniform, 0.638). The advantage-mass utility with Thompson
sampling is self-exploring — the posterior's optimism already probes
uncertain arms, so the floor adds little *on this testbed* (no distribution
shift, no forgetting pressure: mastered skills stay mastered because
gradients never reverse them). Keep the floor for real settings — Prop. 7's
staleness bound is about *re-detecting* change, which this toy cannot
exhibit — but expect low sensitivity to its exact value.

## V4 — Hindsight→teacher feedback (CPU analog of `--hindsight-to-teacher`)

| variant | AUC | final | dead-group rate |
|---|---|---|---|
| hindsight only | 0.874 ± 0.004 | 0.984 | 0.494 |
| + feedback to posterior | 0.864 ± 0.003 | 0.983 | 0.491 |

**Slightly negative on the toy** — and V2 explains why: the posterior lag it
was designed to fix is invisible here because hindsight already trains the
prefix tasks, whose posteriors then update from *natural* successes almost
immediately. Dead-group rate does not rise (no runaway optimism), so the
mechanism is safe but redundant on this testbed. Prediction for the GPU
A/B/C: feedback matters only if natural successes lag relabeled competence
(long-horizon regimes); treat C ≤ B on the maze as consistent with this.

## V5 — Head-to-head vs DAPO dynamic sampling + regime map (`run_baselines.py`)

Matched *generation* budget (3200 groups, discarded draws count), 5 seeds.
DAPO-style dynamic sampling = redraw prompts until the group is live
(0 < K < N), paying for every draw. Three task-pool regimes:

| regime | uniform+maxrl | dapo+maxrl | teacher+maxrl | **teacher+maxrl+hindsight** |
|---|---|---|---|---|
| easy-heavy (levels 1–6), AUC | 0.946 | 0.929 | 0.953 | **0.975** |
| balanced (1–12), AUC | 0.734 | 0.825 | 0.784 | **0.931** |
| frontier-heavy (5–12), AUC | 0.000 | 0.000 | 0.000 | **0.928** |
| frontier-heavy, final | 0.000 | 0.000 | 0.000 | **0.981** |

**Findings:**

1. **The full method dominates every regime**, and its margin *grows with
   difficulty*: +0.03 AUC easy-heavy, +0.11 vs best baseline balanced,
   and **0.93 vs 0.00 frontier-heavy**.
2. **The frontier-heavy result is categorical, not incremental.** With max
   initial pool pass rate 10⁻⁵, the expected number of live groups in the
   whole budget is ≈0.5 — uniform, DAPO, and the plain teacher all flatline
   at exactly 0 because there is *nothing to sample toward*: DAPO's redraws
   and the teacher's predictions both only reallocate compute among tasks
   none of which can produce a success. Only hindsight *creates* signal.
3. **The bootstrapping mechanism, traced:** in the first ~140 groups
   hindsight relabels dead groups to prefix tasks (61 below the pool — i.e.
   it *invents the missing curriculum* below the given distribution — and
   81 in-pool); those prefix gradients raise in-pool pass rates into
   learnability; by draw 400 there are already 257 live groups and normal
   MaxRL takes over; hindsight then goes almost silent (81→81 relabels from
   draw 800 to 3200). It is a cold-start igniter, not a永-running crutch.
4. **DAPO comparison nuance:** dynamic sampling helps in the balanced regime
   (0.825 vs 0.734 uniform — its redraws effectively concentrate on live
   tasks) but *hurts* in easy-heavy (0.929, discarding all-pass groups
   wastes budget that uniform spends on still-useful gradients) and does
   nothing in frontier-heavy. The teacher-with-hindsight subsumes its
   benefit in every regime at the same compute.

## Consolidated design updates

1. **Posterior decay 0.9 → 0.7** in the teachers and verl
   `FrontierTeacher` default (V2b: +0.028 AUC, ~19% of oracle gap, zero cost).
2. **Isotonic projection** available when a difficulty ordering exists
   (maze levels: yes; GSM8K prompts: no ordering — skip).
3. **Floor default stays 0.1** (V3: harmless here, load-bearing under
   drift/forgetting per Prop. 7).
4. **Dense hindsight is the confirmed direction** (V1: relabeled gradients
   are exact on-structure); the teacher-feedback loop is optional and should
   be judged by the GPU A/B/C, not assumed.
