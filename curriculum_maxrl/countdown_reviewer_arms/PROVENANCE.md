# Reviewer arms A + B (Countdown v2) — provenance

Pre-registration: `run_reviewer_arms.sh` in this directory, committed to
../maxrl as 79473b2 (2026-08-04, before any cell ran); marker hardening
2c95170; OOM repair driver ecdc461 (same commands, reruns cells whose
completion marker is missing).

## ARM A — designed-gate B3, corrected-decay code, seeds 1–3 (COMPLETE)

Prereg P-R1: the designed operating point lands on the dose-response
line between the under-gated 3-seed point (~60% mean kept, coverage
restored) and the single-seed full-strength point — mean-kept fraction
in [0, 0.60] of B2's tier-1 gain, tier-1 pass@16 in [.541, .571].
Committed falsification branch: "If it lands OFF the line the dial
claim is refuted and Fig 7a must be redrawn as a scatter, not a
frontier."

Results (step-60 val, tier 1; artifacts `armA_b3fix_s{1,2,3}.json`):

| seed | mean@16 | pass@16 | gate reject frac |
|---|---|---|---|
| 1 | .212 | .513 | .934 |
| 2 | .322 | .573 | .934 |
| 3 | .264 | .488 | .944 |
| agg | .266±.045 | .525±.036 | — |

References (same pool/protocol, `b_scoreboard_3seed.json`):
B1 no-recycling .278±.054 / .541±.020; B2 ungated .324±.012 / .492±.011.

**VERDICT: P-R1 REFUTED.** Mean-kept fraction (.266−.278)/(.324−.278)
= −0.26, below the window — the designed gate is statistically
indistinguishable from recycling-off. Coverage .525±.036 is NOT
restored above baseline (window floor .541); the earlier single-seed
"coverage above baseline" reading (.564, `b_strong_gate_1seed.json`)
did not replicate — it sits inside ARM A's seed spread (.488–.573).
Seed spread on both axes spans most of the B1↔B2 range: the designed
setting is a high-variance recycling-off point, not a third rung on a
dial. Falsification branch executed: Fig 7a redrawn as operating-point
scatter; §6.9 dial language replaced.

Notes: seed 2's first attempt died to node OOM near the end (marker
correctly withheld — completion-gated on `global_step:60`); the repair
driver reran it to completion (ray session 2026-08-06_05-04-49).
Extraction pulls per-tier val metrics from ray worker logs by POOL_TAG.

## ARM B — dose-matched replay (ppo_epochs=2, hindsight OFF), seeds 1–3

Prereg P-R2: replay captures ≥ half of B2's tier-1 mean@16 gain with
no pass@16 loss; committed branch — "If it captures ~all, recycling's
LLM-scale case reduces to the direction term and 6.8 must say so."

Results so far (step-60 val, tier 1; `armB_replay_s{1,2}.json`;
seed 3 training at vendor time):

| seed | mean@16 | pass@16 |
|---|---|---|
| 1 | .459 | .585 |
| 2 | .475 | .674 |
| interim agg | .467 | .629 |

**INTERIM (2/3 seeds): P-R2's strongest branch is firing.** Replay
does not merely capture B2's mean gain (.278→.324) — it exceeds it
(.467), while GAINING coverage over baseline (.629 vs .541) where
recycling paid coverage (.492). Both seeds individually exceed B2 on
both axes. If seed 3 agrees, §6.8 must state that at this scale the
dose control dominates hindsight recycling on both axes: the mean gain
recycling buys is available from extra optimizer epochs on live groups
alone, and the relabel direction's marginal contribution is negative
on both axes relative to the dose control. Dose caveat fixed at design
time: ppo_epochs=2 doubles updates on all live groups, which is
"roughly B2's extra update dose", not an exact match.

Final verdict to be recorded here when `armB_replay_s3.json` lands.

Machine-readable verdicts: `reviewer_arms_verdicts.json`.
