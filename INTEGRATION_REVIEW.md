# Integration review: frontier_rl (curriculum × MaxRL) × Isaac Lab

*Expert review of the implementation design and integration status, 2026-07-23
(evening). Scope: the vendored `frontier_rl/` package, the Isaac Lab bridge in
`isaaclab_integration/`, the experiment now running, and the road ahead
(ladder steps 2–3, task families beyond locomotion). Complements — does not
repeat — `isaaclab_integration/REVIEW_ADVICE.md` (the curriculum-MaxRL side's
code review) and `ISAACLAB_DESIGN.md` (the pre-registered design). Everything
quantitative below was measured or re-derived on this machine today.*

---

## 1. Status snapshot (live)

| piece | state |
|---|---|
| CPU test suite (`isaaclab_integration/test_frontier_terms.py`) | **14/14 pass** (incl. construction-reset guard, optimism-projection, tripwire, probe-macro tests) |
| `frontier_rl/test_framework.py` (vendored core) | pass |
| **Pilot arm 1/5 (`control`)** | **DONE — first successful real-simulator training run of this integration.** 600 iters × 1024 envs in 1421 s (~2.4 s/iter), artifacts complete: 13 checkpoints, TB events, `params/{env,agent,arm}.yaml` |
| Pilot arms 2–5 (`greedy`, `scripted`, `uniform`, `teacher`) | running sequentially now under the claim-file protocol (~24 min/arm ⇒ all five done ≈ 2 h) |
| GPU co-tenancy | resolved: mutual queue-chaining agreed with the verl 2×2 owner (their `run_2x2_resume.sh` waits on our process names; our gate waits on `verl.trainer.main_ppo` + `/tmp/gpu_claim_verl`; both sides hold claim files while training). FIFO by remaining cost put our 2 h pilot first — their cell 2 resumes from its step-25 checkpoint afterwards. |
| Fixed-grid eval + analyzer | implemented (`eval_arms.py`, `analyze_arms.py`), not yet run — pending arms |

The verl GSM8K 2×2 (estimator × curriculum, SmolLM2-360M) on this same box is
the *other half* of the joint program: it tests the **group-estimator regime**
(advmass utility, real N=16) that Isaac Lab's reset-stream regime cannot
express. Together the two runs cover both rows of the regime table in
`frontier_rl/README.md` — worth stating in any writeup that these are one
program, not two projects.

## 2. Design verdict

The integration is architecturally faithful to the validated schedule, and the
recent hardening pass made two changes that are *better than the upstream
design*, one of which corrects a genuine math bug:

**2.1 The optimism fix is load-bearing — upstream should adopt it.**
The original `FrontierBinTeacher` applied deterministic optimism as
`p̃ = mean + k·std` and then evaluated `u(p̃)`. For a *concave, non-monotone*
utility this is wrong in exactly the regime optimism exists for: an unseen bin
(Beta(1,1): mean 0.5, std 0.29) got `p̃ = 0.79` → `u = 0.167`, i.e. naive
optimism **penalized unvisited bins by 33%** relative to their mean-utility
0.25 — anti-optimism. The current implementation maximizes `u` over the
confidence interval `[mean−k·std, mean+k·std]` (projecting the utility peak
onto the interval), which restores true optimism-under-uncertainty: unseen
bins score the maximal `u = 0.25` until evidence says otherwise. At the
pilot's cold start (all p̂ ≈ 0.01–0.06, everything beyond-frontier) this
materially changes early sampling: under the old rule, *visited dead* bins
(p̃ small → u small) and *unvisited* bins (p̃ ≈ 0.79 → u = 0.167) competed
incorrectly. This fix plus the tripwire-hardening (`max_prob` now raises
instead of silently renormalizing — a Q9-faithful assertion, not shaping) and
the exact RNG-state checkpointing should all flow upstream to
`github.com/linjiw/curriculum-maxrl`.

**2.2 Teacher state is now atomic with the policy checkpoint.** The
`FrontierOnPolicyRunner` embeds teacher state in every `model_*.pt` and
refuses to resume a teacher arm from a stateless checkpoint. This closes the
resume-consistency hole the JSON-sidecar design had (sidecar written every N
calls ⇒ up to N reset-batches stale at crash). The JSON stays as the
human-auditable artifact. This is the right split — checkpoint for machines,
JSON for humans — and stricter than what the SONIC design asked for.

**2.3 The five-arm design is the correct experiment.** Same term name, same
graded terrain grid, signal-identical `tile` predicate for teacher and greedy,
artifact-based success detection, per-arm `arm.yaml`. Nothing to change before
the full run.

## 3. Quantitative facts to carry into the analysis

These are measured properties of the environment/stack that will shape the
readout; none require code changes, all belong in the results doc:

**3.1 Evidence-rate asymmetry across bins.** Episode length depends on
outcome: a failing episode ends at base-contact (measured ≈ 5 s of sim time in
the smoke logs) while a surviving one runs the full 20 s. At p = 0.9 a bin
emits ~195 episodes/env-hour; at p = 0.05, ~626 — **hard bins generate ~3×
more Bernoulli evidence per env-hour than easy bins.** Consequences: (a) the
posterior localizes dead bins *faster* than mastered ones (helpful); (b) the
"half-life in episode-equivalents" is not uniform in wall-clock across bins —
decay runs ~3× faster where the teacher samples hard; (c) sampling-mass
telemetry (`effective_bins`) and *evidence*-mass distribution are different
objects — don't read one as the other.

**3.2 Init-level support truncation.** `max_init_terrain_level=5` means
`control` (levels frozen at init draw) lives on rows 0–5 forever, and every
arm's *first* posterior evidence comes only from rows 0–5 (rows 6–9 are
unvisited until an arm's sampler sends envs there — the interval-projected
optimism now handles this correctly, but the *first* `frontier_bin` telemetry
values will reflect prior, not data). Analyzer note already exists; keep it.

**3.3 Terrain difficulty is linear-by-row by construction.** The generator
interpolates each sub-terrain's params over difficulty ∈ [0,1] across 10 rows
(e.g. stair heights 5→23 cm). So the latent difficulty axis is smooth and
monotone — the *good* case for a 10-bin Beta-row teacher, and exactly the
"goal distance in a fixed maze" transfer regime from the frontier_rl report
(not the maze-size cliff regime). If P-A fails here, suspect the signal or the
code, not the axis.

**3.4 The pilot starts beyond-frontier everywhere.** Smoke-run posterior after
800 reset batches: p̂ ∈ [0.01, 0.06] on all 10 rows. Per the regime map, the
teacher ≈ floor+optimism ≈ uniform-ish until frontier emergence. The
discriminating quantity is therefore **AUC from frontier-emergence time**
(first batch with any p̂ ∈ [0.2, 0.8]), which the analyzer should report next
to full-run AUC — and if 600 iters × 1024 envs never exits the prefix, the
pilot verdict is "P-A gate only; P-B deferred", which is a scheduling fact,
not a method result. (REVIEW_ADVICE §2 proposes shared warmstart for the full
run; endorsed — it also removes the biggest seed-variance source.)

## 4. Integration-status matrix (beyond this experiment)

`isaaclab_integration/TASK_STATUS.md` is accurate; the one-line summary: **one
task family (Anymal-C rough locomotion / terrain axis) is fully integrated
end-to-end; everything else is design-ready but unbuilt.** The reusable
surface for the next family is exactly five contract items (difficulty
actuator, binary verifier, fixed-grid evaluator, atomic teacher+policy
checkpoint, baseline arms) — the locomotion implementation is the template.

Priority order for expansion, with the specific Isaac Lab seam each needs:

1. **Ladder step 2 — commanded-speed bins on flat terrain** (their predicted
   clearest-win case). Needs one custom `CommandTerm` subclass whose
   `_resample_command(env_ids)` reads a per-env bin tensor the teacher writes
   (stock `UniformVelocityCommand.ranges` is global). ~50 lines + a
   `FrontierBinTeacher` keyed by speed band; the curriculum term then only
   feeds evidence (`make_curriculum_term` pattern, already vendored). The
   existing `FixedLevelProbe` generalizes: pin env i to speed-bin i%n.
2. **Sparse-success manipulation (Lift/Reach)** — the only family where
   ladder step 3 (MaxRL estimator) is meaningful, because the verifier is a
   real task predicate (`object_reached_goal`) rather than a locomotion proxy.
   Estimator work goes in rsl_rl's advantage computation (`rsl_rl.algorithms.ppo`
   — swap std-normalization for success-conditioned weights on a per-env-group
   basis); keep it out of the env side entirely.
3. Everything else (dexterous, factory, navigation) after one of 1–2 produces
   a positive or a clean null.

## 5. Advice (ranked)

1. **Don't touch the arms mid-pilot.** All five arms must run the code that is
   on disk right now; the CPU suite is green and arm 1's artifacts are
   complete. Any further improvement lands after the pilot readout.
2. **Read P-B in three layers**: fixed-grid macro-pass (primary), AUC from
   frontier-emergence (secondary), full-run AUC (context only). Report the
   P-A gate verdict first and separately — the pre-registered honest-null on
   this grid means a P-B parity with a P-A pass is a *successful* step 1.
3. **Phase-2 grid should manufacture unlearnable-at-budget rows** — that's the
   regime where the design predicts separation. Cheapest construction: extend
   `ROUGH_TERRAINS_CFG` to `num_rows=16` and widen `step_height_range` to
   (0.05, 0.45) — rows ~12+ exceed Anymal-C's physical step ceiling, giving
   genuinely-dead bins to avoid. That is a 3-line cfg subclass, and the
   teacher needs zero changes (dead-bin avoidance is its designed edge).
4. **Send the hardening upstream** (optimism projection, tripwire semantics,
   RNG-in-state, input validation) — flagged conflicts: upstream's new
   `test_isaaclab_adapter` in `test_framework.py` (merge, don't overwrite).
5. **Keep the queue-chaining protocol as infrastructure**, not a one-off: any
   future GPU work on this box should adopt the claim-file + pgrep-chain
   pattern now proven between the two queues. It cost one lost pilot launch to
   learn; it's in `run_experiment.sh` and `run_2x2_resume.sh` now.
6. **Frame the two experiments as one program in the writeup.** The GSM8K 2×2
   tests estimator×curriculum where groups exist (advmass, N=16); the Isaac
   pilot tests the reset-stream port (learnability, no N). A positive in both
   regimes — or a consistent regime-dependent split — is a much stronger
   result than either alone, and the shared theory (§0–§3 of the research
   notes) predicts *which* differences to expect: coverage effects on the LLM
   side, waste-avoidance effects on the sim side.

## 6. Watchlist for the analysis (failure modes to check before believing any number)

- `zpd_bins = 0` batches early in every teacher run → targeting ratio must be
  computed only where defined (analyzer already does this; verify in output).
- `control` vs others on training telemetry: not comparable (init-support
  truncation, §3.2) — fixed-grid eval only.
- Greedy's `Curriculum/terrain_levels` scalar is a mean level, the other arms
  log `mean_bin` under a sub-key — the analyzer's tag-fallback handles it;
  confirm both appear in the table.
- If the teacher arm's `dead_frac` rises *late* in training (posterior decay
  forgetting mastered easy rows it no longer visits — floor should prevent
  this), that's Q8's escalation trigger, not a bug: raise floor / add ALP
  term per SONIC Q8 before rerunning.
- Teacher determinism: same-seed rerun of the teacher arm should reproduce
  level assignments exactly (CPU-verified; if a GPU rerun diverges, suspect
  the co-tenant scheduling changing reset *timing* — batch boundaries are not
  part of the seed).

---

*Bottom line: the integration is sound, one real math bug (anti-optimism on
unseen bins) was caught and fixed before it could contaminate the teacher arm,
the first real-simulator run of the stack completed cleanly, and the remaining
four arms are running under a co-tenancy protocol that can't repeat yesterday's
collision. The next decision point is the P-A gate readout in ~2 hours.*
