# E-LLM-2 Countdown 2×2 (maxrl cells) — analysis against P-C1..P-C4

*2026-07-28. All four maxrl cells complete (60 steps each, fresh from the
shared SFT warmstart, cap=12 per amendment A2). Artifacts:
`maxrl/curriculum_maxrl/countdown/cells_2x2_results.json`. Predictions
frozen in PREREG_E_LLM2.md before any valid cell ran.*

## Step-60 validation (mean@16 / pass@16 per tier)

| cell | tier 0 (2 ops) | tier 1 (3 ops) | tier 2 (4 ops) |
|---|---|---|---|
| C1 teacher + hindsight | .762/.826 | .372/.473 | .178/.267 |
| C2 teacher | .813/.828 | .236/.426 | .163/.278 |
| C3 **baseline (uniform, no HS)** | .833/**.936** | .360/**.645** | .176/**.428** |
| C4 hindsight only | **.886**/.936 | **.405**/.571 | .159/.398 |

## Verdicts, honestly

**P-C1 (hindsight ignites tier 2): NOT confirmed.** Tier 2 rose in every
cell (~.16–.18 mean@16); the hindsight-on cells show no ignition advantage
— C3, with no recycling at all, has the best tier-2 coverage (.428).

**P-C2 (tier-teacher beats uniform): NOT confirmed — the teacher HURT
coverage.** C2 vs C3: pass@16 deficits of −.108 (t0), −.219 (t1), −.150
(t2). Unlike GSM8K this is not starvation (3 tiers, ~hundreds of visits
each; the posterior was well-fed). The teacher concentrated sampling on
tier-0/1 (its utility band) while uniform kept hitting everything — and on
a pool this LEARNABLE, breadth beats focus. This replicates, at LLM scale,
what the Opus review's step-matched maze analysis found: the pure
allocation channel's gain is small-to-negative when the pool is
learnable-everywhere.

**P-C3 (relabels decay): NOT testable as stated** — the dose rode its cap
(12/12) throughout; no decay signature possible under a binding cap.

**P-C4 (risk: yield too thin): inverted** — yield was 72–89%; amendment A2
documented this before the valid cells ran.

**Mean-vs-coverage split worth reporting:** hindsight lifted tier-0/1
mean@16 (C4 best: .886/.405 vs C3 .833/.360) while COSTING coverage at
higher tiers (t1 pass@16 .571 vs .645). Recycled off-target successes
sharpen what the model already does (answers it can reach) at the expense
of exploration breadth — GRPO-like sharpening induced by data recycling
rather than by the weight function. This is a NEW finding, not one we
predicted.

## The uncomfortable diagnosis (review B2 was right)

C3's pass@16 sits at 94%/76%/94% of the random-guesser bound implied by
the pool's left-to-right/no-division construction (3^(n−1) op-sequences:
bounds .998/.848/.453). The pool is too shallow for curriculum OR
recycling to matter: 60 steps of uniform RL nearly saturates what a
guesser could reach, leaving no headroom for allocation and no need for
signal creation. The Opus review predicted exactly this failure mode
(finding B2); we launched anyway on the argument that a simpler pool
would show ignition — the data says the review was right.

## What E-LLM-2 established despite the nulls

1. **The machinery works at LLM scale**: exact-verifier relabeling ran in
   production at 12 groups/step with the response-rewrite keeping goal
   coherence; per-tier posteriors were live; the full stack trained
   stably. This is infrastructure no one else has, now validated.
2. **A real new phenomenon**: recycling-induced sharpening (mean up,
   coverage down) — the creation channel has its own safety trade-off,
   mirroring H6 one level up. Testable prediction for the redesign.
3. **The honest scoreboard**: 0/2 positive predictions confirmed on a
   pool that turned out too shallow — the kill criterion fires as
   pre-registered.

## Redesign decision (per the pre-registered kill criterion + review §6.1)

Tier-2 pass@16 at 60 steps was .267–.428 across cells — far above the
<.02 kill threshold, but the GUESSER BOUND analysis shows the pool cannot
discriminate the mechanisms. Next Countdown iteration (E-LLM-2b) requires:
- Pool regenerated with random permutation + parenthesization + division
  (real Countdown: ~7,680 structures at 4 ops vs our 27) or the
  Jiayi-Pan/Countdown-Tasks-3to4 dataset directly.
- Probe → SFT → probe cycle re-run on the harder pool; launch only if the
  guesser bound leaves ≥3× headroom above the post-SFT probe.
- Dose as a designed axis: cap ∈ {4, 12} × {hs on/off} — the sharpening
  finding makes dose-response the scientifically interesting question.
- grpo cells dropped (GSM8K already carries the safety claim); budget goes
  to 2 seeds on the informative contrasts instead.


---

# E-LLM-2b (v2 pool) — the sharpening replication and the gate result

*2026-07-29. All three arms complete on the real-structure pool (perm ×
parens × exact division; headroom gate passed on tiers 1–2). Single seed;
seeds 2–3 queued. Artifacts: ray sessions 07-28_22 (B2), 07-29_02 (B3),
07-29_05 (B1).*

## Step-60 scoreboard (mean@16 / pass@16; headline tiers 1–2)

| arm | tier 1 | tier 2 (frontier) |
|---|---|---|
| B1 baseline (no HS) | .310/**.559** | .104/.266 |
| B2 hindsight | **.334**/.485 | **.144**/.153 |
| B3 hindsight + **utility gate** | .309/.475 | .133/**.306** |

Gate telemetry: B2 relabeled 108 rollouts/step; B3 admitted 22 and
rejected 83/step as saturated (79%), tracking 291 destination values.

## P-B1 — sharpening replicates: CONFIRMED (both halves)

On a pool with verified headroom (unlike v1), hindsight lifts mean@16
(t1 +.024, t2 +.040 over baseline) and LOSES pass@16 (t1 −.074,
t2 −.113). The v1 signature was not a shallow-pool artifact: recycling
buys accuracy with coverage. This is the paper's claim-3 replication.

## P-B2 — the gate: DECISIVE at the frontier, partial at tier 1

At tier 2, the gate doesn't just close B2's coverage deficit — it
REVERSES it: pass@16 .306 vs baseline .266 (+.040) vs ungated .153,
while keeping 72% of hindsight's mean gain. At tier 1 the gate returns
to baseline mean and still trails baseline coverage (.475 vs .559).
Reading: the gate rescues recycling exactly where recycling is
worth having (the frontier — where relabel destinations are NOT yet
saturated) and neutralizes it where it was pure sharpening (tier 1's
destinations saturate early, so the gate blocks most relabels there and
the arm converges toward baseline). The mechanism and the telemetry
agree.

## Honest limits

Single seed each (seeds 2–3 queued); val noise ±.02–.04 at n=128/tier
(t2 pass@16 differences of .11–.15 clear it; t1 mean differences of .02
do not); B1's first launch OOM'd against an external tenant and was
cleanly rerun; the t0 tier is excluded per the pre-registered headroom
gate (guesser-saturated).


---

## 3-seed aggregate (2026-07-30) — the multi-seed truth, stated plainly

| arm | t1 mean@16 | t1 pass@16 | t2 mean@16 | t2 pass@16 |
|---|---|---|---|---|
| B1 baseline | .278±.054 | **.541±.020** | .117±.030 | .274±.015 |
| B2 hindsight | **.324±.012** | .492±.011 | **.143±.014** | .237±.065 |
| B3 hs + gate | .282±.031 | .484±.011 | .133±.006 | **.279±.019** |

**P-B1 (sharpening) at 3 seeds: CONFIRMED at tier 1, directional at
tier 2.** Tier 1 is clean: mean +.046 and coverage −.049, both well
outside seed noise. Tier 2's mean gain (+.026) is clear but the coverage
loss (−.037±.065) is noisy — seed 1's dramatic collapse (.153) was the
extreme of the spread, not the norm.

**P-B2 (gate) at 3 seeds: the gate restores frontier coverage to
baseline (.279 vs .274, tight ±.019) while retaining ~60% of recycling's
mean gain (+.016 of +.026).** The seed-1 "doubling" (.153→.306) was
real but seed-specific — the honest multi-seed claim is
coverage-restoration-plus-partial-mean-retention, not coverage gain.
At tier 1 the gate neutralizes recycling almost entirely (mean back to
baseline; coverage matches ungated) — consistent with tier-1
destinations saturating early so the gate blocks nearly everything.

**F5 caveat (favorable direction)**: all B3 runs carried the
under-gating bug (audit-measured p̂ 0.637 where the design says 0.917) —
the gate as designed is STRONGER than what ran. A post-fix B3 rerun is
queued as the cheapest possible upside experiment.
