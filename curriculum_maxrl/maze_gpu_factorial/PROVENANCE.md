# Balanced maze factorial — vendored artifacts

Vendored 2026-08-05 from the execution fork
(`maxrl/curriculum_maxrl/maze_gpu/`, fork commit `9f7dd2e`) so the
paper repository is self-contained for the §6.3 factorial claims.

| file | role |
|---|---|
| `run_factorial.sh` | the pre-registration (committed in the fork BEFORE any run; prereg text in the header) |
| `fact_analyze.py` | the analyzer (registered endpoint + tagged exploratory secondary) |
| `results_factorial_wave1.json` | wave-1 per-cell endpoints (seeds 0–5 + grpo_mass/grpo_nostd arms, 36 cells, repair pass folded; the P-F1 verdict file). Regenerate: `fact_analyze.py --seed-start 0 --seeds 6` |
| `results_factorial_wave2.json` | wave-2 per-cell endpoints (fresh seeds 6–11, 24 cells; the P-F2 verdict file). Regenerate: `fact_analyze.py --seed-start 6 --seeds 6` |
| `FACTORIAL_VERDICT.md` | verdict against every prereg criterion |

Raw per-step seed logs (`fact250_*.jsonl`, ~36 files) live in the fork
at the commit above; they are reproducible from `run_factorial.sh`
(fixed seeds, fixed 250-step budgets, warmstarts
`seed{0..5}_sft_warmstart.pt` committed alongside).

Headline, wave 1: **P-F1 failed at the registered endpoint** (final:
3/6 uniform, 4/6 teacher paired blocks) → the falsification branch
executed and the paper's zero-exception cohort claim is retracted
(paper commit `e27b5d9`). P-G0a confirmed; P-G0c failed (both rungs
reported in §6.3–6.4).

Wave 2 (2026-08-06, vendored alongside): the wave-1 exploratory
covAUC ordering was pre-registered as the primary
(`run_factorial_wave2.sh`, P-F2, with both post-verdict texts written
in advance in `WAVE2_BRANCHES.md`) on six fresh seed blocks (6–11)
and **confirmed 6/6 under each sampler** (p=.031 per sampler). The two
sampler contrasts share each seed/warm-start block and are repeated
observations, not independent replicates. After averaging samplers
within block, wave 2 is positive in 6/6 blocks (mean +.01950, 95% t
interval [+.01148,+.02752]); all 12 block averages across waves are
positive descriptively. Easy-band P-F3 met its registered pair-level
bar at 10/12, but block averaging gives 4 positive, 1 tie, 1 negative,
with an interval including zero, so localization is suggestive. Power
analysis motivating the endpoint change: `power_note.json`. Verdict:
`WAVE2_VERDICT.md`. Raw wave-2 cells (`fact250_*_s{6..11}.jsonl`)
live in the fork alongside wave 1's.
