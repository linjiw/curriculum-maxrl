# AMaze gate confirmatory result — inconclusive at ten seeds

**Campaign:** `/data/robotixx/ued_bench/gate-confirmatory-20260819`, 20 cells
(two arms × seeds 2001–2010), full shipped 30,000-update protocol.
**Preregistration:** `ued_benchmark/AMAZE_GATE_PREREG.md`, frozen 2026-08-17;
the outcome-blind checkpoint amendment was recorded 2026-08-19.
**Analyzer:** `ued_benchmark/scripts/analyze_gate_confirmatory.py`, SHA-256
`aaf54f22…51d755`, invoked **once** after the complete gate passed.
**Analysis artifact:** `ued_benchmark/AMAZE_GATE_ANALYSIS.json`, SHA-256
`c0162e99…c86774e`.

## Closure integrity

The terminal rerun had 20/20 distinct training receipts, 20/20 distinct
shipped-evaluation receipts, zero failure receipts, and all expected
checkpoint, metadata, log, and evaluation files. Before any evaluation value
was read, the checkpoint-budget verifier recorded all 20 stored training
states in `[29960,29997]`, above the frozen 29,900 floor; paired arms matched
exactly within seed. The canonical budget receipt has SHA-256
`8d3557ba…9d4db`, and all 20 cells now have explicit `DONE` markers. No result
JSON existed before the single analyzer invocation.

## Frozen primary: inconclusive at n=10

Primary endpoint: paired difference in mean held-out `test_solved_rate` over
the three shipped mazes, `plrGate - plrMM`.

| quantity | result |
|---|---:|
| arm means | gate .9733; upstream robust PLR .9100 |
| paired difference | **+.0633** |
| paired SD | .1214 |
| paired-bootstrap 95% CI | **[+.0003,+.1410]** |
| exact two-sided sign-flip p | **.1562** |
| positive / negative / tied pairs | **5 / 3 / 2** |
| SESOI | +.0200 |
| frozen verdict | **`inconclusive_at_n10`** |

The positive branch required both a mean of at least `+.02` and `p <= .05`.
The point estimate clears the SESOI, but the exact test does not. The negative
branch required the CI upper bound to be below `+.02`; it is `+.1410`.
Therefore neither directional branch fires. Per the prospectively written
fallback: **report the interval and claim nothing.**

This is Tier 4 (inconclusive), not a registered positive and not a negative.
It does not promote the five-seed development gate result, establish that
activity-shaped gating improves robust PLR, or alter the separate registered
finding that pure activity is starved when it replaces MaxMC. No seeds are
added, no endpoint is substituted, and neither arm is rerun.

## Frozen secondaries — descriptive only

The paired mean `test_return` difference is `+.0647`, 95% CI
`[+.0040,+.1355]`. Per-maze solved-rate differences are `+.0060` for
SixteenRooms, `+.0350` for Labyrinth, and `+.1490` for StandardMaze. These
were frozen as descriptive secondaries; no inferential claim or per-maze
mechanism claim is made from them.

## Consequence for the paper

The abstract and contribution perimeter remain unchanged. The AMaze main-text
result remains the clearly labelled five-seed development negative for using
coefficient activity as a standalone replay priority. The full-budget gate
follow-up is disclosed as inconclusive in the appendix/status record and does
not upgrade takeaway (iii) to Tier 2.
