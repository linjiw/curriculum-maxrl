# Consistency Audit — curriculum-maxrl (2026-07-23)

## Summary counts

| Doc | Claims checked | CONFIRMED | WRONG | STALE | UNVERIFIABLE |
|---|---|---|---|---|---|
| curriculum_maxrl/PROOFS.md | 16 | 13 | 2 | 1 | 0 |
| ISAACLAB_DESIGN.md | 21 | 17 | 0 | 2 | 2 |
| **Total** | **37** | **30** | **2** | **3** | **2** |

**Coverage warning — only 2 of 20 repo documents were audited.** NOT audited:
COSMOS3_RESPONSE.md, EVIDENCE.md, GUIDE.md, PAPER.md, PR1_REVIEW_VERDICT.md, README.md,
REPORT.md, SCHEDULE.md, SONIC_RESPONSE.md, curriculum_maxrl/{DESIGN.md, GSM8K_A10G_PLAN.md,
README.md, RESEARCH.md, THEORY.md, VALIDATION.md}, curriculum_maxrl/maze_gpu/EXPERIMENTS.md,
frontier_rl/README.md, verl_integration/README.md. (EVIDENCE.md, VALIDATION.md, THEORY.md and
SONIC_RESPONSE.md were touched only as cross-reference targets, not audited themselves.)

## Must-fix (ranked by damage, headline claims first)

1. **PROOFS.md, Prop 4 claim line (lines 97-98) — WRONG headline formula.**
   RLOO advantage mass is `E[Σ|w|] = 2p(1−p)` exactly for all N ≥ 2 (MC: 0.42001 at N=8, p=0.3),
   not `2p(1−p)·N/(N−1)` (0.48000). The doc's own proof (line 106) already concludes the correct
   value — the claim line's prefactor is spurious.
   *Edit:* claim line → "E[Σ|w|] = 2p(1−p) exactly for all N ≥ 2" (delete `·N/(N−1) →`).
   *Also:* fix proof typo at line 105: `N·E[K] = N²p`, not `Np` (final `N(N−1)p(1−p)` is correct).

2. **PROOFS.md, Prop 5 claim line (lines 119-120) — WRONG headline advantage ratio.**
   The MaxRL/RLOO utility ratio limit as p → 0 is `N−1`, not `N` (exact: 6.99979 at N=8,
   30.99535 at N=32). The proof body (line 125) correctly says "Ratio → N−1 ≈ N"; only the
   claim line overstates.
   *Edit:* claim line → "with ratio → N−1 (≈ N) as p → 0".

3. **ISAACLAB_DESIGN.md header (lines 8-9), §0 item 3, footer (lines 164-167) — STALE test artifact claim.**
   Claims the adapter is "unit-tested on CPU" via `test_framework.py::test_isaaclab_adapter`,
   but no such test exists in the file (11 tests, none isaaclab) or in git history. The
   behaviors do reproduce when run manually, but the claimed artifact is absent.
   `isaaclab_curriculum.py:24` also references the nonexistent test.
   *Edit:* add `test_isaaclab_adapter` to `frontier_rl/test_framework.py` (the stub-env script
   already reproduces all claimed behaviors), or reword to "manually CPU-verified against a
   stub env" — in both the doc and the `isaaclab_curriculum.py:24` docstring.

4. **ISAACLAB_DESIGN.md §1 D3, line 59 — STALE family index.**
   Says learnability `p(1−p)` is "the N→1 member" of the advmass family. Wrong: `u_1(p) ≡ 0`
   identically; `u_2(p) = p(1−p)` exactly. SONIC_RESPONSE.md:55 already has the correct
   "N=2 member".
   *Edit:* "the N→1 member" → "the N=2 member (u₁ is identically zero)".

5. **PROOFS.md header, lines 4-5 — STALE cross-reference.**
   Points to "THEORY.md §5 snippet"; the verification snippet is THEORY.md §6 (line 197);
   §5 is hindsight relabeling.
   *Edit:* "THEORY.md §5 snippet" → "THEORY.md §6 snippet".

6. **(Code, flagged incidentally) curriculum_maxrl/estimators.py:38 docstring — STALE.**
   Says "Unbiased for the truncated ML objective with T = N", contradicting the landed
   PROOFS.md correction (practical estimator is unbiased for T = N−1, MC-verified).
   *Edit:* docstring → "T = N−1".

## Contradictions (cross-doc / doc-vs-code)

1. **ISAACLAB_DESIGN.md:59 vs SONIC_RESPONSE.md:55** — "N→1 member" vs "N=2 member" for
   learnability p(1−p) in the advmass family. SONIC_RESPONSE is correct (u₁ ≡ 0, u₂ = p(1−p)
   to machine precision). Fix ISAACLAB_DESIGN (Must-fix #4).

2. **PROOFS.md header vs THEORY.md section numbering** — PROOFS.md cites "THEORY.md §5
   snippet"; the snippet is §6 (THEORY.md:197). Fix PROOFS.md (Must-fix #5).

3. **PROOFS.md Prop 1 correction (T = N−1) vs estimators.py:38 docstring (T = N)** — the doc
   carries the verified correction; the code comment was never updated. Fix code docstring
   (Must-fix #6).

4. **ISAACLAB_DESIGN.md "unit-tested" claims vs frontier_rl/test_framework.py contents** —
   doc (and isaaclab_curriculum.py:24) name a test that does not exist anywhere in the repo
   or its history. Fix per Must-fix #3.

5. **Internal contradictions inside PROOFS.md** (claim line vs own proof body):
   - Prop 4: claim line has spurious `·N/(N−1)` factor; proof line 106 concludes `2p(1−p)`.
   - Prop 5: claim line says ratio → N; proof line 125 says "Ratio → N−1 ≈ N".
   In both cases the proof body is right and the headline claim line is wrong.

## Load-bearing UNVERIFIABLE claims

1. **All "their guide" references in ISAACLAB_DESIGN.md** (D1, D2, D4, D6, D7, D8, D9, §3, §4:
   guide §3.2, §8, §9.1-9.5, §10 cadence, determinism pitfall, `terrain_levels_vel` greedy ±1
   walker, `update_env_origins` max-level randomization). The "Isaac Lab RL Infrastructure
   Guide" (2026-07-22) is not in the repo and no isaaclab source tree exists locally, so the
   design's core stock-behavior premises cannot be audited. **Load-bearing:** the entire
   adapter design rationale rests on these.
   *Artifact needed:* check the guide document into the repo (or pin exact upstream isaaclab
   commit/URLs for `terrain_levels_vel` and `update_env_origins`), ideally with an isaaclab
   checkout or vendored source excerpts.

2. **ISAACLAB_DESIGN.md §1 D9, line 104 — "Our own SIM-M3-style negative on flat data says the
   same thing."** "SIM-M3" appears nowhere else in either repo; the external guide defining it
   is absent. **Load-bearing:** it is the cited evidence for a design decision (D9).
   *Artifact needed:* cite the internal artifact directly (GPU maze non-transfer,
   EVIDENCE.md §2, "γ=1 on broad/flat pools — predicted", EVIDENCE.md:83) or name the external
   guide's experiment precisely and include the guide.

## Clean bill

None. Both audited documents carry at least one WRONG or STALE finding:

- curriculum_maxrl/PROOFS.md — 13/16 confirmed (all core props 1-3, 6, 7, corrections, and MC
  numbers reproduce), but blocked by Prop 4/Prop 5 claim lines and the §5/§6 cross-ref.
- ISAACLAB_DESIGN.md — 17/21 confirmed (all defaults, telemetry keys, aging, tensor handling,
  and stub-env behaviors reproduce), but blocked by the phantom unit test, the N→1 slip, and
  two unverifiable external-guide dependencies.
