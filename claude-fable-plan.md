# Curriculum-MaxRL → ICRA 2027: Review and Plan

**Author:** Claude (Fable 5) review pass, 2026-08-11
**Deadline anchor:** ICRA 2027 papers due **Sept 15, 2026, 23:59 PT** (conference May 24–28, 2027, Seoul). That is **5 weeks from today**.
**ICLR 2027 for comparison:** abstract Sept 18, full paper Sept 25 AOE.

---

## Part 0 — Honest review of where the project stands

### What is genuinely strong

1. **The theory core is clean and finished.** The exact identity
   `A_N(p) = 2(pass@N − pass@1)` with a unique compute-dependent peak at
   `p* ≈ ln(N)/N`, recovery of `p(1−p)` at N=2, the drop-all-fail
   truncation-order result, and the expected-update factorization — all
   analytically derived and MC/exact-enumeration verified. This is the
   asset every downstream paper is built from, and it is done.
2. **One crisp, well-powered positive result:** Acrobot, 20 paired seeds,
   `u_16` beats `p(1−p)` on target-uniform AUC by **+.0480**
   (95% CI [+.0209, +.0738], sign-flip p=.0034). This is the single most
   tellable result in the project — a *compute-aware* learnability score
   beating the standard learnability score on a control task.
3. **One registered-and-confirmed multi-seed result:** maze factorial wave 2 —
   time-integrated MaxRL−GRPO coverage contrast positive **6/6 per sampler**
   (p=.031 each), and after the independent-unit repair, **6/6 blocks,
   +.0195 [+.0115, +.0275]**; 12/12 blocks cross-wave. Navigation-shaped
   domain, real gradients, fresh randomness.
4. **A rare and real epistemic culture:** preregistration, committed
   falsification branches, executed retractions, delivery gates. This is a
   long-run career asset regardless of venue.

### What is weak or blocked

1. **The LLM evidence (Countdown/GSM8K) is the most caveated part of the
   project** — bootstrap-proxy metrics, failed delivery gates, missing raw
   outcomes, E2c blocked on a shared RTX 5090. It is also the part with
   *zero* value to an ICRA audience.
2. **The current manuscript is ICLR-shaped and over-length** (~17 main-text
   pages vs a 9-page limit; a 9-page candidate exists on
   `origin/codex/curriculum-maxrl-research@9277141` and must be reconciled
   deliberately per `BRANCH_RECONCILIATION.md`).
3. **There is no robot, no robot benchmark, and no sim-robotics centerpiece.**
   The IsaacLab and MountainCar rungs are compressed boundary checks, not
   headline experiments.

### The strategic fact that drives everything below

**ICRA is not a venue you can reach by reframing this draft.** An ICRA paper
is 6 pages (+ references), two-column IEEE, reviewed by people who ask "does
this make robots learn better?" The current paper's center of mass — LLM RLVR,
estimator-side coefficient algebra, a chronicle of preregistered retractions —
would read as an out-of-scope ML paper. What *does* fit ICRA perfectly is the
part of the project you have treated as side rungs: **automatic curricula for
robot RL** (navigation, locomotion, control), where the derived utility is a
drop-in, hyperparameter-free replacement for ALP-GMM-style learnability and
hand-designed difficulty schedules, and where hindsight relabeling (HER) is
native vocabulary.

---

## Part 1 — High-level strategy: run two tracks

### Track A (keep, don't grow): the ICLR paper

The existing codex plan is correct and already scheduled; do not let ICRA
cannibalize it. Its critical path is unchanged:

- E2c executes when the shared 5090 frees (sole blocker; no substitutes).
- Hard stop on new ICLR-bound training **Aug 28**.
- 9-page rebuild on `iclr2027` style, argument-ordered (not the ladder),
  AI-use statement, anonymity sweep — Sept lock, submit by Sept 25.

The only interaction with Track B: **GPU scheduling** (see risks) and the
rule that the two submissions must be **content-distinct** (IEEE and ICLR
both prohibit dual submission of the same work — this is fine if Track B has
its own experiments and claims, but do not reuse paragraphs or figures).

### Track B (new): the ICRA paper

**One-sentence thesis:** *group-based RL's own advantage estimator tells you
exactly which tasks are worth rolling out — using it as the curriculum score
trains robot policies faster per GPU-hour and covers more of the task
distribution than uniform sampling, learnability heuristics, or hand-designed
difficulty schedules.*

**Working title direction:** "The Estimator Knows Where to Train:
Compute-Aware Automatic Curricula for Robot Reinforcement Learning" (keep
"The Estimator Decides" reserved for the ICLR paper — don't collide).

**The three contributions (write these first, before any experiment):**

1. **A closed-form, hyperparameter-free curriculum utility** derived from the
   group estimator: `u_N(p) = 2(pass@N − pass@1)`. Peak moves with rollout
   budget N — a *compute-indexed* zone of proximal development. `p(1−p)`
   (ALP-GMM/SFL-style learnability) is the N=2 special case. No band
   hyperparameters, no bins.
2. **A practical teacher** that needs only per-task success/failure counts:
   decayed Beta posterior + Thompson sampling + uniform replay floor, plus
   greedy water-filling rollout allocation on `p(1−p)^N`. Drop-in for any
   trainer that evaluates N rollouts per task (group-based PPO/GRPO-family,
   or episodic RL with repeated resets).
3. **Robotics validation**: on [chosen domain — Part 2], the u_N teacher
   beats uniform, `p(1−p)`, and a hand-ordered difficulty curriculum on
   success over held-out environments at matched wall-clock, with an
   N-ablation showing the compute-awareness is load-bearing. Optional but
   high-value: real-robot validation + goal-conditioned hindsight arm.

**What ports over from the existing corpus (cheap, already done):**

- Theory §: identity, peak, N=2 recovery — compressed to ~0.75 page.
  (Re-derive presentation for ICRA: call it "expected update magnitude" or
  "learnability score"; the phrase *coefficient activity/mass* is ICLR-paper
  vocabulary and will lose an ICRA reader.)
- Acrobot 20-seed result — a control task ICRA reviewers accept; this is the
  "score shape matters" evidence and it is already publication-grade.
- Maze factorial — navigation-shaped; usable as supporting evidence for
  coverage-vs-mean metrics IF you can present it without the MaxRL/GRPO
  framing overwhelming a 6-page paper. Candidate for a compressed subsection
  or supplementary material.
- `verl_integration/curriculum.py`'s teacher logic is numpy and
  testbed-agnostic — reuse it as the reference implementation.

**What stays OUT of the ICRA paper (be ruthless):**

- Everything Countdown/GSM8K/LLM. All of it.
- The retraction chronicle. Keep the *hygiene* (prereg, seeds, CIs), drop
  the *narrative*. ICRA gives you 6 pages; one honest limitations paragraph
  is the right dose. The Digits counter-test survives as a single scoping
  sentence ("the score is a curriculum hypothesis generator, not a universal
  objective; on exact-probability counter-tests it can lose to uniform").
- MaxRL-vs-GRPO estimator theology, hindsight exactness contracts, the
  gate/dial saga.

### Fallback ordering (decide once, now)

If by the **Aug 24 checkpoint** (below) the robotics experiments are not
producing a defensible positive, do NOT force a weak ICRA submission:

1. **Fallback 1 — RA-L**: 8 pages, journal review, submit *any time* (with
   optional presentation at a later ICRA/IROS). Same paper, no deadline
   pressure, and RA-L reviewers reward exactly this kind of rigor. This is
   the natural home if the robot experiments need 8 weeks instead of 4.
2. **Fallback 2 — ICLR only** this cycle; ICRA 2028/IROS 2027 with a mature
   robotics campaign next cycle.

---

## Part 2 — The ICRA experiment campaign (low level)

### 2.1 Domain choice (decide by Aug 13 — this is the one decision that gates everything)

**Option 1 (recommended): BARN-style navigation curriculum.**
Procedurally generated obstacle environments with a published difficulty
metric; the lab has the environments, the Jackal platform, and deep
familiarity. The task pool is *exactly* the shape the teacher wants: hundreds
of environments spanning trivial→near-impossible, parameter-shared policy
(lidar → velocity commands), binary success per episode, cheap groups of N
episodes per environment. Real-robot validation is logistically realistic
here and nowhere else in 5 weeks.

**Option 2: legged-locomotion terrain curriculum (IsaacLab).**
The standard "game-inspired"/terrain curriculum in legged RL is hand-designed
promotion/demotion — replacing it with u_N is a clean, recognizable story,
and you already have an IsaacLab rung. Costs more GPU and more integration
risk; no real robot unless the lab has one ready.

**Option 3: goal-conditioned navigation with HER.**
Goal relabeling in navigation is *exact by construction* (the robot verifiably
reached where it reached) — the hindsight channel's exactness contract holds
natively, so "u_N decides where HER pays" becomes a second contribution.
Best done as an *arm inside Option 1*, not a separate domain.

Recommendation: **Option 1 as the centerpiece, Option 3 as one arm inside
it, Option 2 only if BARN infrastructure surprises you.**

### 2.2 The registered matrix

Keep the prereg culture, scaled to the deadline ("prereg-lite": commit
predictions + analysis code before unblinding, one file, no gate bureaucracy).

- **Samplers (5 arms):** u_N teacher (ours) · uniform · `p(1−p)` learnability
  (ALP-GMM-style, source-faithful) · hand-ordered difficulty curriculum
  (BARN difficulty metric, staged promotion) · PLR (if time; else cite and
  drop — 4 strong baselines beat 5 rushed ones).
- **Seeds:** ≥5 per arm (these environments are cheap relative to LLM runs;
  don't repeat the 1-of-2-seeds wound).
- **Budget matching:** matched **wall-clock AND matched steps, report both**
  — this is the project's own hard-learned lesson (the teacher's throughput
  advantage is real but must be reported as throughput, not smuggled in).
- **Primary endpoint (pre-commit):** success rate on a held-out environment
  set spanning the difficulty range, at fixed wall-clock.
- **Secondary endpoints:** coverage across difficulty deciles (the coverage
  meter is a genuine methodological contribution — one figure), easy-decile
  retention (forgetting), steps/GPU-hour, dead-group (all-fail) rate.
- **Ablations (pick 2):**
  1. **N-sweep** (e.g., N ∈ {2, 4, 8, 16}): does the u_N peak location
     matter, i.e., does u_N at the deployed N beat u_2 = p(1−p)? This is
     the Acrobot result reproduced where it counts and is the paper's
     distinguishing claim — *do not skip this one*.
  2. Floor/decay sensitivity (one small grid, appendix-bound).
- **Optional HER arm:** u_N-gated relabeling vs relabel-everything HER vs no
  relabeling, inside the goal-conditioned variant.

### 2.3 Real-robot validation (decides "solid" vs "strong")

Sim-only is *acceptable* at ICRA but the acceptance odds move materially with
hardware. Minimum viable: take the best sim policy per arm (ours vs uniform
vs hand-ordered), run each on ~10 physical courses spanning difficulty,
report success counts + one trajectory figure + one photo. This is ~2–3 lab
days on a Jackal if the sim-to-real stack already exists. Schedule it for
week 4 and treat it as a bonus, not a gate.

### 2.4 Engineering checklist

- [ ] Environment pool loader + per-env success verifier + difficulty
      metadata (held-out split frozen and committed before training).
- [ ] Group-rollout evaluation loop (N episodes per sampled env per round).
- [ ] Port `curriculum.py` teacher (Beta posterior, decay, floor, Thompson,
      water-filling) — it is already unit-tested; add an env-pool adapter.
- [ ] Baseline implementations: source-faithful ALP-GMM sampling; staged
      difficulty schedule; (PLR if kept).
- [ ] Logging from day one: coverage@k by decile, easy retention, dead-group
      rate, teacher p̂-vs-eval calibration, wall-clock per step. Every number
      that could appear in the paper writes a committed artifact
      (standing hygiene rule 1 — keep it).
- [ ] `prereg_icra.md`: predictions + endpoint definitions + analysis script
      hash, committed before the first full run finishes.

---

## Part 3 — Writing plan (6 pages + references, IEEE two-column)

*(Verify the exact page rule against the [ICRA 2027 CFP](https://2027.ieee-icra.org/announcements/call-for-technical-papers/); recent ICRAs allow 6 content pages + unlimited references, with a paid extra page at camera-ready.)*

### Page budget

| § | content | pages |
|---|---|---|
| 1 | Intro: rollouts are the cost center of robot RL; curricula are usually heuristic; the estimator already computes the answer. Fig. 1 = concept figure | 1.0 |
| 2 | Related work: ALP-GMM, teacher–student (Matiisen), PLR/ACCEL/PAIRED, SFL learnability, HER, terrain curricula in legged RL, BARN-line navigation learning | 0.5 |
| 3 | Method: identity + peak + N=2 recovery (0.5), teacher + allocation algorithm box (0.5) | 1.0 |
| 4 | Experimental setup: domain, arms, matching protocol, endpoints | 0.75 |
| 5 | Results: main table (5 arms × endpoints), coverage-by-decile figure, N-ablation figure, Acrobot paragraph, robot validation | 2.0 |
| 6 | Limitations (incl. one-sentence Digits scope) + Conclusion | 0.5 |
| — | slack / figures overflow | 0.25 |

### Figures (make these 4, no more)

1. **Concept:** u_N(p) curves for several N with the moving peak; inset:
   "uniform wastes rollouts here (p≈0) and here (p≈1)".
2. **Main result:** success on held-out envs vs wall-clock, all arms, mean ±
   CI over seeds.
3. **Coverage:** heatmap or small-multiples of success by difficulty decile
   over training — the "mean hides the tail" story in one image.
4. **Robot:** courses + trajectories + success table (or the N-ablation if
   hardware slips).

### Terminology mapping (ICLR dialect → ICRA dialect)

- "coefficient activity/mass" → "expected update magnitude the estimator
  assigns a task" (define once, then "utility").
- "coverage proxy / covAUC" → "success across the difficulty distribution,
  time-integrated" — define plainly.
- Keep the honesty norms from `NEXT_RESEARCH.md`'s standing rules (artifact
  per number, convention stated inline, seed counts in the sentence), drop
  the ledger vocabulary ("registered primary", "delivery gate") from prose.

### Review loop

- Aug 31: skeleton + figs 1–2 drafted → run `/ars-reviewer` panel on the
  draft framed *as an ICRA submission* (robotics personas).
- Sept 8: full draft → second panel pass + red-team (overclaim grep: every
  claim sentence names its seed count and matching convention).
- Sept 12: freeze; Sept 13–14 buffer; submit **Sept 14**, not Sept 15.

---

## Part 4 — Timeline (today → Sept 15)

| week | Track B (ICRA) | Track A (ICLR) — unchanged |
|---|---|---|
| **Aug 11–17** | Domain decision (Aug 13). Env pool + teacher adapter + baselines running end-to-end on 1 seed. Commit `prereg_icra.md`. | E2c fires whenever 5090 frees; 9-page reconciliation per `BRANCH_RECONCILIATION.md` |
| **Aug 18–24** | Full matrix launched (5 arms × 5 seeds). **Aug 24 checkpoint:** is the u_N arm ≥ uniform and ≥ p(1−p) directionally? If not → activate RA-L fallback, stop ICRA spend. | training hard-stop Aug 28 |
| **Aug 25–31** | N-ablation + HER arm. Draft skeleton + figs 1–2. First reviewer-panel pass. | result lock begins |
| **Sep 1–7** | Real-robot runs. Results §, all figures final. Full draft Sept 7. | 9-page rebuild |
| **Sep 8–15** | Panel pass 2, red-team, freeze Sept 12, **submit Sept 14**. | polish; abstract Sept 18, paper Sept 25 |

---

## Part 5 — Risks and pre-committed responses

1. **GPU contention with E2c.** E2c owns the shared RTX 5090 by frozen
   protocol (no relaxing its occupancy ceiling). BARN-style training is
   deliberately the low-compute option — run it on any other GPU (or CPU
   farm for the 2D variants). Never queue ICRA jobs on the 5090 while E2c
   is pending.
2. **u_N ties p(1−p) in the robot domain.** Plausible — at small effective N
   the scores are similar. That's why the N-ablation is mandatory: run with
   a deployed N where the peaks genuinely separate (N ≥ 8). If it still
   ties, the paper survives on {beats uniform + beats hand-designed +
   throughput + coverage metric + zero hyperparameters}, with the tie
   reported honestly; if it doesn't even beat uniform, that's the Aug 24
   kill criterion → RA-L timeline.
3. **Timeline slip.** The single most likely failure. Mitigation: the Aug 24
   checkpoint is a real go/no-go, and the RA-L fallback means no work is
   wasted — the same paper just ships without a deadline.
4. **Dual-submission overlap.** The ICRA paper must not reuse ICLR text,
   figures, or headline claims. Theory identity appears in both — that is
   fine (it's a citation-able result of the project) but write it fresh and
   keep each paper's *empirical* content disjoint. If the ICLR paper is on
   arXiv before ICRA review, cite it.
5. **Reviewer "this is just ALP-GMM" objection.** Pre-empt in Related Work:
   ALP-GMM tracks *learning progress* over a continuous space with a GMM;
   ours is a closed-form function of pass rate *and rollout budget* derived
   from the deployed estimator, with no fitting. The N-ablation is the
   empirical teeth.

---

## Part 6 — If you actually meant ICLR

If "icra" was a typo for ICLR: ignore Parts 1B–5 above; the correct plan is
already written and scheduled — `FINAL_REVIEW_RESPONSE_AND_GUIDANCE_2026-08-07.md`
Part IV (block-reanalysis propagation ✅ done, E2c on GPU-free, training stop
Aug 28, 9-page argument-ordered rebuild on the 2027 style, AI-use statement,
red-team, abstract Sept 18 / paper Sept 25). The highest-leverage remaining
items on that track, in order: (1) reconcile local `main` with the remote
nine-page candidate at `9277141` per the frozen merge rules, (2) run E2c the
moment the 5090 clears, (3) rebuild the paper argument-first (theory → one
positive/negative pair (Acrobot/Digits) → maze wave as the real-gradient
confirmation → Countdown as scoped applied observation), (4) the anonymity/
path sweep including JSON artifacts.
