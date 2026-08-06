# Balanced maze factorial — vendored artifacts

Vendored 2026-08-05 from the execution fork
(`maxrl/curriculum_maxrl/maze_gpu/`, fork commit `9f7dd2e`) so the
paper repository is self-contained for the §6.3 factorial claims.

| file | role |
|---|---|
| `run_factorial.sh` | the pre-registration (committed in the fork BEFORE any run; prereg text in the header) |
| `fact_analyze.py` | the analyzer (registered endpoint + tagged exploratory secondary) |
| `results_factorial.json` | per-cell endpoints as analyzed for the verdict (31 complete cells at verdict time; the repair pass folds in GPU-contention casualties — reruns of identical commands — and cannot flip P-F1, already ≤4/6 under uniform) |
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
and **confirmed 6/6 under each sampler** (p=.031 per sampler; 24/24
paired blocks across both waves; easy-band P-F3 10/12). Power
analysis motivating the endpoint change: `power_note.json`. Verdict:
`WAVE2_VERDICT.md`. Raw wave-2 cells (`fact250_*_s{6..11}.jsonl`)
live in the fork alongside wave 1's.
