# Reviewer arms A + B (Countdown v2) — provenance

The external execution record labels `run_reviewer_arms.sh` as locked in
../maxrl at 79473b2 before these cells ran; that commit object is not vendored
here, so this checkout cannot independently audit the timing. Marker hardening
is recorded as 2c95170 and the OOM repair driver as ecdc461 (same commands,
rerunning cells whose completion marker was missing).

Metric warning: every legacy `pass16` field below is VERL bootstrap best@16,
a with-replacement coverage proxy—not standard unbiased pass@16. See the
machine-readable `METRIC_PROVENANCE.json` sidecar.

## ARM A — designed-gate B3, corrected-decay code, seeds 1–3 (COMPLETE)

Prereg P-R1: the designed operating point lands on the dose-response
line between the under-gated 3-seed point (~60% mean kept, coverage
restored) and the single-seed full-strength point — mean-kept fraction
in [0, 0.60] of B2's tier-1 gain, tier-1 bootstrap best@16 proxy in
[.541, .571].
Committed falsification branch: "If it lands OFF the line the dial
claim is refuted and Fig 7a must be redrawn as a scatter, not a
frontier."

Results (step-60 val, tier 1; artifacts `armA_b3fix_s{1,2,3}.json`):

| seed | mean@16 | bootstrap best@16 proxy | gate reject frac |
|---|---|---|---|
| 1 | .212 | .513 | .934 |
| 2 | .322 | .573 | .934 |
| 3 | .264 | .488 | .944 |
| agg | .266±.045 | .525±.036 | — |

References (same pool/protocol, `b_scoreboard_3seed.json`; stored population SD):
B1 no-recycling .278±.054 / .541±.020; B2 ungated .324±.012 / .492±.011.

**STORED-RULE VERDICT: P-R1 REFUTED.** Using the exact stored aggregates,
the mean-kept fraction is −0.25, below the window. This does not establish
equivalence to recycling-off. The bootstrap proxy .525±.036 is below the
window floor .541; the earlier single-seed proxy-above-baseline reading
(.564, `b_strong_gate_1seed.json`) did not replicate—it sits inside ARM A's
seed spread (.488–.573).
Seed spread on both axes spans most of the B1↔B2 range: the designed
setting is a high-variance recycling-off point, not a third rung on a
dial. Falsification branch executed: Fig 7a redrawn as operating-point
scatter; §6.9 dial language replaced.

Notes: seed 2's first attempt died to node OOM near the end (marker
correctly withheld — completion-gated on `global_step:60`); the repair
driver reran it to completion (ray session 2026-08-06_05-04-49).
Extraction pulls per-tier val metrics from ray worker logs by POOL_TAG.

## ARM B — higher-dose replay (ppo_epochs=2, hindsight OFF), seeds 1–3

The source-recorded P-R2 rule asks whether replay captures at least half of
B2's tier-1 mean@16 gain without reducing the logged bootstrap proxy. Because
replay doubles optimizer epochs on every live group, this is not dose matched
and cannot isolate relabel direction.

Results, final (step-60 val, tier 1; `armB_replay_s{1,2,3}.json`):

| seed | mean@16 | bootstrap best@16 proxy |
|---|---|---|
| 1 | .459 | .585 |
| 2 | .475 | .674 |
| 3 | .500 | .646 |
| agg | .478±.021 | .635±.046 |

**STORED RULE MET (3/3 seeds).** Replay reaches mean@16 .478 and bootstrap
proxy .635, improving both logged metrics relative to the retained aggregate.
This shows the reported tradeoff is not present in this higher-dose arm, but it
does not identify whether dose, update direction, or another package difference
causes the contrast.

Machine-readable verdicts: `reviewer_arms_verdicts.json`.
