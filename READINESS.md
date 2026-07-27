# Phase-1 Readiness: Self-Verified Frontier RL on Cosmos3/LIBERO

*Runbook for taking the implemented stack to real training-vs-baseline runs.
Status: 2026-07-23. Everything in §1 is implemented and unit-tested in this
repo (17/17 tests, `python3 frontier_rl/test_framework.py`); §2 is the
ordered checklist of what remains, each item with its owner-side and gate.*

---

## 1 · What is DONE (implemented + tested here)

| piece | file | verified by |
|---|---|---|
| Positive-part MaxRL estimator (weighted RFT, Q1) | `frontier_rl/estimators.py` (`positive_part=True`), `trainer.py` (`positive_weights`) | `test_positive_part_estimator` — MC identity `E[Σw⁺] = pass@N − pass@1`, all-pass self-retirement |
| Estimator-swappable trainer + baselines | `trainer.py` (`estimator="grpo"/"rloo"`, `dapo_max_redraws`) | `test_baseline_estimator_arms` |
| Cosmos/LIBERO TaskSpace (predicate goals, template rewrites, never-upgrade rule, relabel-only sub-goal arms, mastery splits, hierarchical shrinkage) | `adapters/cosmos_libero.py` | `test_cosmos_libero_relabel_contracts`, `test_cosmos_mastery_split_and_shrinkage`, `test_cosmos_posterior_hygiene_end_to_end` |
| Poison-rate meter (per-class precision gate) | `adapters/cosmos_libero.py::PoisonRateMeter` | `test_cosmos_poison_meter_gates_vocabulary` |
| Live glue for cosmos-framework (wave loop, oracle verifier from BDDL `goal_state`, weighted-CFM manifest, round loop) | `adapters/cosmos_live.py` | `test_cosmos_live_glue` (fake client/venv mirroring the real APIs) |
| Evaluation harness (unbiased success@k, easy-decile retention, teacher calibration, dual-currency budget ledger) | `evaluation.py` | `test_evaluation_harness` |
| Pilot-0 instruments (0a variance, 0b poison w/ success-enriched probe, 0c surrogate cosine, go/no-go verdict) | `pilot0.py` | `test_pilot0_instruments` |
| End-to-end mock pilot (6 arms + wiring check) | `examples/run_cosmos_pilot.py` | frontier-heavy: uniform/dapo/teacher **0.000** → oracle 0.862, self 0.756, gated 0.842; all 5 preregistered checks PASS |

Real-surface verification done against the local checkout
`~/work/cosmos3edge/cosmos-framework` (not from the proposal's claims):

- `ActionEnvironmentClient(server_url, domain_name, prompt, image_size,
  timeout)` — `.prompt` is a mutable attribute sent with every
  `/predict` / `/predict_batch` request ⇒ per-group conditioning is one
  assignment (`closed_loop_eval.py:130-296`).
- Wave loop: `_run_task_vectorized` (`closed_loop_eval.py:909`) —
  SubprocVectorEnv + spawn-safe `_LiberoEnvFactory` (:863), per-episode
  init states via `venv.set_init_state(states, id=slots)` (:985), success
  from `info["success"]` (:1039), `TASK_MAX_STEPS` caps (:66).
- Init-state control: `task_suite.get_task_init_states(task_id)` +
  custom-JSON path (`_load_initial_states`, :749).
- Weighted-CFM hook: `compute_flow_matching_loss` returns
  `(weighted_mean, per_instance[B])`
  (`model/generator/algorithm/loss/flow_matching.py:86-91`) — the Phase-1
  change is `(w·per_instance).sum()/w.sum()` at the call site in
  `omni_mot_model.py`; the loss function itself needs no edit.

## 2 · What REMAINS before real training (ordered; each with gate)

### R1. Environment plumbing (cosmos-framework venv; ~1 day)
The curriculumrl venv has no libero/cosmos deps by design. Write the launch
script **inside cosmos-framework** (`uv run --group libero`) that constructs
`LiveRolloutBackend`'s six injected callables from the real modules:
- `venv_of_task`: cache one SubprocVectorEnv per task (waves reuse it);
  BDDL path exactly as `closed_loop_eval.py:950`.
- `init_states_of`: slice `get_task_init_states`; **freeze a held-out eval
  subset per task before any training** (the pairing requirement).
- `get_images` / `action_to_env`: bind `_get_libero_images`,
  `_format_action` + `_framewise_action_to_delta` + `_remap_gripper` with
  the run's camera/action config (copy the arg values from the released
  eval recipe).
- `prompt_of_template`: `_augment_task_prompt_with_viewpoint` (+ JSON
  prompt format if the checkpoint trained with it — the server auto-detects,
  `action_policy_server_libero.py:662-678`).
- `predicate_snapshot`: LIBERO's goal-predicate evaluation on the worker env
  (`parsed_problem["goal_state"]` + the suite's predicate registry; route
  through `venv.env_method` if worker handles aren't exposed — the
  `_worker_env` helper centralizes this).
**Gate:** one `rollout_group` of N=8 on one libero_90 task returns rewards +
per-failure achieved predicates; ledger counts match the wave logs.

### R2. Checkpoint + data (blocked on which SFT baseline)
Decide and pin, before any RL run (Q4.3 — preregister baseline strength):
- **Few-demo SFT checkpoint**: train (or obtain) the LIBERO-Long SFT from
  1–10 demos/task with the 24 GB stack; record demos/task, steps, and the
  full-SFT reference number next to it.
- **Eval init states**: freeze `eval_states.json` (held-out per task) and
  `train_states.json`; commit hashes.
- **Sub-goal template table**: enumerate libero_90's BDDL goal conjunctions
  (`goal_predicates_of`), write ONE canonical instruction per admissible
  sub-conjunction, human-review the table, commit it. This is the closed
  rewrite vocabulary (Q3.2) — it is data, not code.
**Gate:** checkpoint + both state files + template table committed; SFT
baseline mean success measured on the frozen eval states (3 seeds of eval).

### R3. Pilot 0 (A10G, ~2–3 days, uses R1+R2)
Run in order; each instrument is already implemented (`pilot0.py`):
- **0a** on ~50 groups from the SFT checkpoint: `GroupVarianceProbe` on
  rewards + first-chunk actions. *Gate:* contrast ≥ 5%, action std above
  floor. Fail ⇒ tune server sampler noise/temperature first (the knob
  exists server-side; SimpleVLA-RL precedent).
- **0b** on ~500 mixed rollouts **success-enriched** (demo replays + easy
  init states for rare predicates): `run_poison_probe` with the ID-prefilter
  on/off. *Gate:* ≥1 predicate class allowed at 90% precision ⇒ self arm
  unblocked; else oracle arm only (still a publishable measurement — the
  first published poison rate for world-model self-verification).
- **0c** on ~50 groups: weighted-CFM direction vs a reference direction.
  Cheapest reference: finite-difference PG on the server's own sampler
  (perturb, re-score), or defer to the toy-exact check (already PASSing in
  the mock) + monitor training-curve sanity in round 1. *Gate:* mean-dir
  cosine ≥ 0.8; below ⇒ Phase 1 proceeds, ReinFlow promoted in Phase 2.

### R4. Weighted-SFT trainer hook (cosmos-framework; ~0.5 day)
One call-site edit where `compute_flow_matching_loss`'s per-instance vector
is reduced: accept a `weight` column from the round manifest
(`WeightedCFMBuffer` rows) and reduce `(w·per_instance).sum()/w.sum()`.
Dataloader side: manifest rows → (video, rewritten `language_goal`, actions)
samples — the rewritten goal replaces the caption/prompt field, nothing
else changes.
**Gate:** unit parity — all-weights-1.0 manifest reproduces the unweighted
SFT loss bit-for-bit on one batch (same standard as the 24 GB stack's
EMA/VAE checks).

### R5. Phase-1 four-arm launch (A10G, 2–3 weeks)
Arms (all `TrainerConfig(positive_weights=True)`, N=8, γ=1, decay 0.7,
floor 0.1, task-level arms): **uniform / teacher / teacher+oracle-relabel /
teacher+self-relabel** (self arm contingent on 0b). Round loop =
`Phase1Round`: collect ~40 groups → manifest → weighted SFT → redeploy →
repeat; teacher state persists across rounds.
Protocol (locked by `evaluation.py`):
- eval sweep (`EvalProtocol`) every round on frozen eval states; success@k
  k∈{1,4,8}, easy-decile retention (probe fixed from the SFT baseline),
  teacher calibration.
- budgets in BOTH currencies from `RunLedger`; report via
  `matched_budget_report` at matched episodes AND matched wall-clock.
- ≥3 seeds, paired via shared per-seed SFT warmstart.
**Preregistered predictions** (from the mock pilot + regime map):
uniform ≈ teacher ≈ SFT baseline on the dead tail; relabel arms ignite
(burst→silence in `relabels/step`); self within poison-rate of oracle;
easy-decile retention non-decreasing in every MaxRL arm.
**Success:** ordering `uniform ≤ teacher < teacher+relabel` on success@8 at
matched rollouts, paired per-seed.

### R6. Baseline comparisons (Phase 1.5)
- **DAPO arm** (`dapo_max_redraws=4`, uniform teacher) at matched
  *generation* budget — already wired; the mock predicts ~0 on few-demo.
- **GRPO arms** (`estimator="grpo"`, uniform + teacher): the H6 collapse
  ablation. NOTE (measured while building the mock): H6 collapse needs
  function approximation + sampled coverage — it does NOT reproduce on
  tabular-exact toys, so this claim is only testable on the real flow
  policy with per-seed success@k curves. Do not promise it from CPU
  evidence.
- SimpleVLA-RL protocol comparison (2,500 trajectories/suite) once the
  four-arm result is in.

## 3 · Risk deltas discovered during implementation (new since the response)

1. **Relabel-only arms are load-bearing** — sub-goal tasks must be excluded
   from the teacher's sampling support (`samplable_mask`); letting the
   teacher roll out the invented curriculum flips the regime from
   frontier-heavy to balanced and erases the categorical result. On the
   real stack this is physical (no BDDL file for arbitrary sub-goals), but
   the mask must still exist for mastery-split bookkeeping.
2. **Poison probes must be success-enriched** — precision on failure-heavy
   probes is dominated by false-positive base rates (~65:1 at p≈0.015);
   `run_poison_probe` refuses to prune classes with <20 oracle-positive
   examples (UNMEASURED ≠ disallowed).
3. **All-success relabeled groups are silent** — if every failed rollout
   achieves the same sub-goal, the relabeled group is K=N and self-retires.
   Correct behavior (an uncontrasted group carries no likelihood signal),
   but budget dashboards must count these separately or "relabel yield"
   reads as mysteriously low. `RunLedger.relabel_yield` covers it.
4. **H6 needs the real model** — the collapse ablation is a
   function-approximation phenomenon (CPU-exact GRPO looks healthy,
   DESIGN.md §8 agrees). Phase-2 claim, per-seed success@k currency.

## 4 · One-command checks

```bash
# unit tests (16, incl. live-glue fakes + pilot-0 instruments)
python3 frontier_rl/test_framework.py
# end-to-end mock pilot: 6 arms + wiring check (~2 min CPU)
python3 frontier_rl/examples/run_cosmos_pilot.py
```
