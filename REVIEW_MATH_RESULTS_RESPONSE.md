# Response to PAPER_MATH_RESULTS_REVIEW.md (2026-08-04)

Review target was commit `bc67108`; this response applies to the
manuscript after the R3/R4/R5 fixes (`cb420f3`) plus the changes in
this commit. Where an item was already partially fixed by the R3/R4
round, that is noted; this pass finishes the P0 list.

## Critical assessment of the review

We accepted the review's findings essentially in full. Independent
checks performed before editing:

- **GRPO SD convention (§1.6): CONFIRMED by Monte Carlo.** The deployed
  code (`estimators.py`, verl `core_algos.py`) uses sample SD (ddof=1);
  MC of the deployed weights matches the review's
  `sqrt((N-1)/N) * (1/N) E[sqrt(K(N-K))]` half-mass to 4 decimals and
  does NOT match the population-SD curve the paper plotted
  (at N=16, p=.3: deployed 0.851 vs paper's 0.879). Tail ratios
  `sqrt(N)` (p→0) and `(N-1)/sqrt(N)` (p→1) confirmed numerically.
- **Timing artifact (§2.6): CONFIRMED.** `generation_timing.json`
  reproduces 1,188 steps / 31.17% overall / 8 sessions >0.8 with 282
  steps at 83.37%. There is no labeled 272-step / 85% GSM8K cohort.
- **Fig 8 globs (§2.3): CONFIRMED.** The script classified cohorts by
  filename wildcard (4 GRPO vs 18 usable MaxRL files spanning teacher
  forms, hindsight doses, a wide model, repeated seeds).
- **Oracle floor (§1.7): CONFIRMED.** `run_hindsight_controls.py` gives
  `oracle_g4` floor 0.0 vs Thompson floor 0.1 — "no-floor,
  gamma-matched oracle" is the correct label.
- **Maze permutation (§2.2) and hindsight proposition (§1.8):** the
  R3/R4 round had already weakened these (floor-p disclosure;
  Prop→Remark), but the review is right that disclosure does not cure
  exchangeability, and the remark still claimed an "ML gradient". Both
  now fully adopted.

One place we went slightly beyond the review: the fig8 manifest also
records the excluded `matched_uniform_maxrl_s0.jsonl` (no passk rows)
with its reason, which is the same run that explains the 5v4
composition in §6.3.

## P0 items — all resolved in this commit

| review item | resolution |
|---|---|
| `u_N` factor-of-2 collision | Prop. 1 now defines `A_N := E[sum abs(w_i)] = 2 u_N`, with `u_N = pass@N − pass@1` the sampling utility used by figures/algorithm/code; every exact-mass statement carries the 2. Added the exact link `u_N = p(1−p) w_{N−1}(p)`. |
| GRPO mass ≠ deployed code | Derivation, Fig. 1 curve, caption, and interpretation all switched to the sample-SD (ddof=1) convention with tail ratios `sqrt(N)` / `(N−1)/sqrt(N)`; `fig1_utility.py` regenerated; enumeration test covers both conventions. |
| Truncation lemma not self-contained | Lemma 1 now defines `J_T`, states the i.i.d. binary-reward model, and gives the expectation identity showing unbiasedness for `J_{N−1}` (N≥2). |
| Hindsight "ML gradient" overclaim | Remark now characterizes the update as a verified, adaptively selected, off-policy auxiliary update, lists the five reasons it is not unbiased (selection, conditioning correlation, law change, no IS correction, `J_{N−1}` weights), and states the distribution-matching condition. Contribution #2 reworded to match. |
| Undefined band width / dead zone / "only channel" / "no sampler can reach" | "Partition" language replaced by three continuous regimes anchored at the exact zeros; any "band" is now the explicit threshold set `{p : u_N ≥ η u_N(p*)}` (η=0.5 stated); dead zone scoped to p=0 under current policy/support; "only channel/mechanism" replaced by "one way to create a positive auxiliary update" with the hand-added-subgoals equivalence stated. |
| Maze `p=0.0079` | Removed (also from ladder table). Reported as perfect sign separation over a heterogeneous, non-exchangeable 9-run pool, descriptive only; sensitivity reads (within-arm permutation, warmstart pairs) reported as direction-consistent without curing the design; the ≥6-seed balanced factorial specified in-text with the two-sided p<.05 floor argument. |
| Fig. 8 `p=0.0001` | Removed from text, caption, and script. Cohort now frozen in `paper/figures/data/fig8_run_manifest.json` (exact files + exclusions with reasons); script hard-errors on missing/empty inputs and no longer discovers runs by glob. |
| GSM8K in abstract | Replication language removed; abstract states "two-seed pilot", direction 2/2, registered shape 1/2. Body adds the governing caveat that eval-noise bars are not training-seed uncertainty and that mean@4/pass@4 are coupled meters (not 4 independent confirmations); "z≈2" language replaced by descriptive signal-to-noise. |
| Monotone gate dial | Replaced everywhere (abstract, intro Q3, §6.9, takeaway, fig7 caption/annotations, knob table) by: one useful frontier-tier operating point; settings consistent with a trade; strong point is 1 seed confounded with code version; dial claim deferred to the pre-registered fixed-decay sweep. |
| 85%/272-step timing | Replaced in intro and appendix by the artifact's actual numbers: 31% step-weighted mean over all 1,188 steps; 83% over the 8 generation-dominated sessions (282 steps), labeled a selected-subset description. |
| Jugs artifacts missing | Synced from `../maxrl` into `jugs_llm/results/` (prereg, 9 cells, verdicts, entropy, noise floor, postmortem, analysis script) with `PROVENANCE.md` recording prereg commit `63e01d4`, execution commit `eba7929`, and post-run gate fix `df8b2cf`. |
| Hard-coded figures | fig2/fig3 literals moved to versioned data JSONs with per-panel provenance; fig7 strong-gate point vendored as `b_strong_gate_1seed.json`; `paper/results/manifest.json` freezes every figure's inputs with sha256 checksums and regen commands. |
| Two TeX bodies | Consolidated into shared `paper/body.tex`; `main.tex` / `main_iclr.tex` are format wrappers only. |

## Also done from P1/P2 while in the file

- Statistical contract paragraph added to §6 preamble (independent
  unit, pairing rules, eval vs training noise, metric definitions,
  ± source naming).
- Conclusion section added (the paper previously ended at
  limitations).
- Oracle relabeled "no-floor, γ-matched true-pass-rate utility oracle";
  §6.1 tie language weakened to "similar within observed five-seed
  variation, not tested for equivalence"; Prop. 3 narrowed to the
  static known-p problem and explicitly disconnected from the sampler
  and the §6.1 oracle.
- 6/6→3/3-on-both-meters and n=3 paired-t caveat (min sign-flip p=.25)
  now stated in §6.3.
- Countdown "restores" language: frontier-tier claim kept as point
  estimates with both spreads; tier-1 B3-worse-on-both disclosed in the
  monotonicity discussion.
- IsaacLab regime rule reworded to a pilot-level observation.
- Formula enumeration tests: `curriculum_maxrl/test_mass_formulas.py`
  (MaxRL, RLOO, both GRPO conventions, tail ratios) — passing.
- Both PDFs rebuild with zero overfull boxes and zero duplicate
  destinations (`hypertexnames=false`); `docs/` copies refreshed.

## Remaining P1 (require GPU runs, already queued per CONSOLIDATED.md)

1. Balanced maze estimator×sampler factorial, ≥6 independent seed
   blocks (decides whether a confirmatory p-value ever returns).
2. Fixed-decay, multi-seed `gate_max_p` dose sweep (decides the dial).
3. Raw per-seed Countdown endpoint table (scoreboard currently has
   means/SDs only — blocks paired-delta reporting).
4. Steering-controlled, replacement-matched GSM8K multi-seed cell
   (E-LLM-1b running).
5. Seed-level `paper/results/seed_endpoints.csv` so fig2/fig3 derive
   rather than transcribe.

## Page budget

The ICLR build is 19 pages total (appendix included). The review's §5.3
reorganization (move ladder controls and run histories to the appendix,
tighten §6.7) is the remaining writing task if a 9-page main text is
the target; deferred until the E-LLM-1b verdict lands to avoid
rewriting §6.7 twice.
