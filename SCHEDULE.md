# Experiment schedule & tracking

*Living document — updated as runs complete. Times are A10G wall-clock.*

## Currently executing (GPU queue, in order)

| # | run | duration | purpose | decision it feeds |
|---|---|---|---|---|
| E1 ✅ | ck_uniform_maxrl (2400 s + ckpt) | done | efficiency baseline | — |
| E2 ✅ | ck_uniform_grpo (2400 s + ckpt) | done | efficiency baseline | — |
| E3 ▶ | ck_frontier_alp_maxrl_hsd (2400 s + ckpt) | running | champion checkpoint | — |
| E4 | eval_efficiency.py over E1–E3 | ~30 min | **samples-to-90%-coverage per level** — the paper's currency | headline table for site/report |
| F1 | long_falp_hsdense (9600 s) | 2.7 h | **is level 6+ a duration question?** | if yes: frontier-march story complete; if no: revise mechanism (depth-scaled budgets / min-depth curriculum) |
| F2 | matched_falp_p4_hsdense (2400 s) | 40 min | γ=4 concentration on GPU | promote γ to maze + verl defaults, or record CPU-only |
| F3–F4 | dense-hindsight seeds 1–2 (2×2400 s) | 1.3 h | champion multi-seed due diligence | error bars on the 0.258 headline |

Watcher: `watch_gpu.sh` (running) — notifies on stall (>25 min no log growth) or queue completion.

## Decision tree after the queue drains

```
E4 efficiency table
├─ champion shows ≥2× samples-to-coverage on frontier levels
│    → add efficiency chart to website; lead REPORT with it
└─ no separation → coverage parity note; keep AUC as headline

F1 long-horizon
├─ level 6 leaves 0 by 9600 s
│    → run F1b: same budget, uniform baseline (is it the schedule or just time?)
└─ still 0 → implement depth-scaled move budgets (hindsight-min-depth
     curriculum); CPU-validate first, then one 2400 s GPU run

F2 γ=4
├─ AUC ≥ dense-hindsight baseline +0.003 → set teacher-power=4 default for
│    level-structured tasks in maze + frontier_rl docs
└─ tie/worse → document as CPU-only effect (compounding weaker at 13 levels
     than 36 tasks); keep γ=1 GPU default

F3–F4 seeds
├─ ordering holds → REPORT tables get ±std; done
└─ champion within noise of frontier_alp → soften "champion" claim to tie
```

## Next wave (planned, not yet queued)

| priority | experiment | est. | prerequisite |
|---|---|---|---|
| P1 | **Efficiency eval of F1's long-horizon checkpoint** — does 4× training turn into inference-time speedup at deep levels? | 30 min | F1 |
| P2 | **MountainCar scaled benchmark** (steps 120→600, 5 seeds, γ ablation) — does hindsight reach the flag (hardest bin > 0)? First *external* env where the full stack could show a categorical win | ~2 h CPU (parallel, no GPU) | none |
| P3 | **Best-config maze run with all validated knobs** (frontier_alp + dense hs + γ=4 + greedy rollout allocation if F2 passes) — the "everything on" run | 40 min | F2 |
| P4 | **Streaming-pool teacher prototype** (parametric density over a continuous difficulty axis, ALP-GMM-style) — unblocks procedural/generative task sources; CPU-validate on a continuous-difficulty variant of grid_reach | CPU | none |
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
- **Toy→real gap:** every CPU win must re-prove itself on GPU (γ=4 is the
  current test case); two CPU pre-registrations have transferred correctly
  so far.
