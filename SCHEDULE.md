# Experiment schedule & tracking

*Living document — updated as runs complete. Times are A10G wall-clock.*

## Completed GPU queue

| # | run | duration | purpose | decision it feeds |
|---|---|---|---|---|
| E1 ✅ | ck_uniform_maxrl (2400 s + ckpt) | done | efficiency baseline | — |
| E2 ✅ | ck_uniform_grpo (2400 s + ckpt) | done | efficiency baseline | — |
| E3 ✅ | ck_frontier_alp_maxrl_hsd (2400 s + ckpt) | done | champion checkpoint | — |
| E4 ✅ | eval_efficiency over E1–E3 | done | samples-to-coverage | **RESULT: up to 11× vs GRPO at level 5, speedup grows with difficulty (1.2×/2.7×/11×); GRPO curves flatten at large k. Decision: efficiency leads the benefits table on site + PAPER.** |
| F1 ✅ | long_falp_hsdense (9600 s) | done | is level 6+ a duration question? | **NO** — mean 0.258→0.269, L5 doubles, L6 stays ≈0.01. Mechanism revision needed at depth; CPU-validate depth-scaled move budgets / param-sharing check first |
| F2 ✅ | matched_falp_p4_hsdense (2400 s) | done | γ=4 on GPU | **does not transfer** (AUC 0.231 vs 0.236) — ODE model predicted it (weak compounding at 13 broad levels); γ stays 1 on GPU/verl, CPU/chain-only effect |
| F3 ✅ | dense-hindsight seed 1 (2400 s) | done | champion multi-seed | see F4 |
| F4 ✅ | dense-hindsight seed 2 (2400 s) | done | champion multi-seed | champion 0.252±0.005 final / 0.229±0.009 AUC; paired deltas vs plain teacher 6/6 positive but final margin mostly seed-0; **honest headline: reliable AUC gain + never worse; final edge small in infinite-data regime** |

**GPU QUEUE DRAINED (all E and F runs complete).** Next wave now unblocked.

**Parallel CPU (done): MountainCar categorical result** — flag-only 0.000 →
uniform-mix 0.889 → teacher 0.944 → **full stack 1.000 every seed**; plus
the transfer lesson (per-bin params never reach the flag: curricula operate
through shared parameters).

## Resolved decisions

- Efficiency improves strongly at selected frontier targets (up to 11× versus
  GRPO), but reverses at one mid-level; report the full curve.
- Duration alone does not solve level-6 pass@1. Per-step legality is the binding
  mechanism; the wide-model probe improves tail coverage, not execution enough
  for mastery.
- γ=4 is a CPU chain result only; γ=1 remains the GPU/verl default.
- Dense hindsight has a reliable AUC gain over the plain teacher across three
  seeds, while its final-score edge is small.

## Next wave (planned, not yet queued)

| priority | experiment | est. | prerequisite |
|---|---|---|---|
| P1 | **Isaac Lab Anymal-C rough pilot + fixed-grid evaluation** | GPU | live pilot retry |
| P2 | **Deep-supervision maze ablation** (all verified prefixes or deeper SFT exposure) | GPU | none |
| P3 | **Streaming-teacher validation on a real procedural source** | CPU/GPU | synthetic prototype complete |
| P4 | **Per-prompt rollout counts in verl** for greedy allocation | multi-GPU integration | allocator CPU tests |
| P5 | SmolLM2-360M + GSM8K 2×2 via `verl_integration/` | 8-GPU node | **blocked on hardware** |

## Standing cadence

- After every completed run: analyze → update EXPERIMENTS.md/REPORT.md →
  sync logs to this repo → push.
- Website results tables refreshed when a headline number changes.
- Every negative result gets written up with its diagnosis (the H6 reversal
  and conditioning-rewrite lesson were the two most valuable findings so far).

## Risks being tracked

- **Shared GPU:** another user's job took the card once already; the final
  sweep waits politely (`pgrep` loop) and the watcher distinguishes
  "waiting" from "stalled".
- **Seed noise:** finals vary ±0.01–0.015 across seeds; no single-seed claim
  goes in REPORT.md without either multi-seed confirmation or an explicit
  single-seed caveat.
- **Toy→real gap:** γ=4 failed to transfer from the chain to the maze; every
  CPU-only win remains provisional until tested in its target domain.
