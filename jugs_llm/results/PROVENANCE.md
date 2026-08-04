# Jugs (E-LLM-3) result snapshot — provenance

Synced 2026-08-04 from the sibling execution repo `../maxrl`
(`curriculum_maxrl/jugs/`), which holds the runs' original history.

| event | commit (in ../maxrl) | date |
|---|---|---|
| Preregistration committed (`PREREG_E_LLM3.md`) | `63e01d4` | 2026-08-02 |
| B-arm results committed (all three predictions NOT CONFIRMED) | `eba7929` | 2026-08-03 |
| Gate granularity fix (posterior keys on relabeled TASK) | `df8b2cf` | 2026-08-03 |

The nine `cells/*.json` files were produced by the pre-fix code
(execution provenance = the code as of `eba7929`); the granularity fix
in `df8b2cf` post-dates the runs and must not be read as their code
version. The value-keyed gate that rejected 99.8% of relabels during
the runs is the bug the postmortem describes.

Contents:

- `PREREG_E_LLM3.md` — predictions P-J1/P-J2/P-J3, committed before the
  B-arm launches.
- `cells/*.json` — per-cell raw endpoints, 3 arms x 3 seeds.
- `e_llm3_verdicts.json` — prediction verdicts (none confirmed).
- `entropy_trajectories.json` — actor entropy per step per cell.
- `jugs_noise_floor.json` — repeated-eval noise floor.
- `E_LLM3_POSTMORTEM.md` — analysis: pool's learnable region collapses
  to a single 1–2-move template stratum; every arm (including plain
  MaxRL, no recycling) converges to the same deterministic policy.
- `analyze_e_llm3.py` — the analysis script that produced the verdicts.

Paper linkage: §Limitations ("The coverage main effect is
pool-conditional") and the Jugs appendix paragraph quote these
artifacts.
