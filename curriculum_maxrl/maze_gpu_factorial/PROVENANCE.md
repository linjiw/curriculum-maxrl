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

Headline: **P-F1 failed at the registered endpoint** (3/6 uniform, 1/3
teacher paired blocks) → the falsification branch executed and the
paper's zero-exception cohort claim is retracted (paper commit
`e27b5d9`). P-G0a confirmed; P-G0c failed (both rungs reported in
§6.3–6.4).
