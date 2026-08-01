# E-LLM-3: Jugs — design + pre-registration scaffold (2026-08-01)

Question (from BENCHMARK_SURVEY.md): do recycling-induced sharpening and
the derived gate generalize beyond Countdown — from one-shot expression
search to stateful sequential planning — and does the frontier teacher pay
when the pool has unlearnable-at-budget tiers?

## Why Jugs

Exact simulator verifier (vendored reasoning-gym); the relabel map is
*richer* than Countdown's: a failed move sequence certifies every amount
any jug ever held (multiple candidate relabels per failure, each with a
truncated trajectory that is itself the verified solution — trajectory
coherence is exact, not approximate). Tiers = (num_jugs, min_moves) grid.

## Infrastructure (this directory)

- `pool.py` — tier grid t0..t4 (2j/2m → 5j/16m), prompt contract
  (`<answer>` + one move per line), `verify()`, `relabel_candidates()`,
  `relabeled_task()` (conditioning rewrite). All exact; smoke-tested:
  true solution verifies, relabeled prefix verifies under rewritten goal,
  garbage/invalid moves parse to None.
- `pool_v1.jsonl` — 200 tasks/tier, seed 0. Generation cost dominated by
  t4 (~8s/puzzle BFS); one-time.
- Feasibility gate (task #4, before any prereg numbers): base pass@1 and
  pass@16 per tier for candidate models. Decision rule:
  - need ≥1 tier with pass@1 in [1%, 40%] (the band) AND ≥1 tier at
    pass@16 ≈ 0 (unlearnable-at-budget, for the regime rule);
  - SmolLM2-360M-Instruct first (continuity with GSM8K rung); if t0 < 1%
    even with 2-shot prompting → Qwen2.5-0.5B-Instruct, then 1.5B
    (TinyZero precedent says 0.5B fails Countdown; Jugs t0 is easier).
  - format-priming SFT (as Countdown used) is allowed and will be
    disclosed; it primes the move syntax, never solutions.

## Planned cells (to be pre-registered AFTER feasibility, BEFORE any RL)

2×2 on the chosen model: {no-recycling, recycling-ungated, recycling-gated}
× {uniform, frontier teacher}, N=16, matched step budget, 3 seeds for the
headline arms (budget permitting; single-seed arms labeled).

Predictions to commit (drafts — numbers TBD after feasibility):
- **P-J1 (sharpening replicates):** ungated recycling lifts mean@16 and
  loses pass@16 on tiers whose relabel destinations saturate (the t0/t1
  amounts are few and quickly mastered — the Countdown tier-1 analogue).
- **P-J2 (gate transparency):** the same gate_max_p=0.5, no per-domain
  tuning, restores coverage to the no-recycling baseline at ≥50% mean
  retention where P-J1 bites, and is transparent (≈no-op) elsewhere.
- **P-J3 (regime rule):** the frontier teacher beats uniform iff the pool
  includes the unlearnable-at-budget tier(s); on the learnable-everywhere
  subpool it is a null. This is the three-scale regime rule's fourth test.
- **P-J4 (posterior-tilt, exploratory):** if the u_N-form maze sweep
  validates the (1-p)·u_N tilt (P-U2), run it as a fifth arm here.

Meters: per-tier mean@16 / pass@16 (the coverage currency), dead-group
fraction, relabel yield + gate rejection rate, policy entropy. Noise
floor measured FIRST (5 repeated evals of the base checkpoint) — no
claim inside it. All artifacts JSONL, committed.

## Status

- [x] pool + verifier + relabel map, smoke-tested
- [ ] pool_v1.jsonl generation (running)
- [ ] feasibility: base pass@k per tier per model (GPU queued behind
      H6 job and sweep_un_form)
- [ ] prereg commit with numeric predictions
- [ ] verl integration (reuse Countdown's hindsight.py machinery — the
      gate + relabel plumbing is domain-generic; only the verifier and
      relabel-candidate fn are swapped)
