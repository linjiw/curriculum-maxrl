# frontier_rl × Isaac Lab — ladder step 1 (CPU-tested; simulator pilot active)

*Companion to `../ISAACLAB_DESIGN.md` (their design decisions D1–D10) and
`/workspace/IsaacLab/RL_INFRA_GUIDE.md` (the infra guide it responds to). This doc
records what is built, what has only CPU evidence, and what remains unverified in
Isaac Sim. It also answers their §4 open items and gives the runbook for the
pre-registered 5-arm experiment.*

Status (2026-07-23): CPU integration tests **16/16 pass**, the finite-N math
suite passes, and `frontier_rl/test_framework.py` passes. The initial five-arm
GPU pilot failed at CUDA-context creation because a concurrent verl worker
occupied the GPU. An exclusive-GPU retry is now actively training the control
arm. `_pilot/completed_pilot.txt` is still empty and no log contains a
`Training time:` artifact, so this is not yet a completed simulator smoke or an
experiment result. See `TASK_STATUS.md` for task-family scope.

## What's here

```
isaaclab_integration/
  frontier_terms.py       the five arms as curriculum terms + success predicates
  train_frontier.py       launcher (stock rsl_rl train.py flow + --arm/--success_fn)
  test_frontier_terms.py  CPU tests, stub env, no isaaclab import
  run_experiment.sh       the 5-arm × N-seed pre-registered experiment driver
  eval_arms.py            fixed-level, equal-seed checkpoint evaluation
  analyze_arms.py         P-A/P-B/P-C readout from TB event files (run in-container)
  TASK_STATUS.md          task-family implementation/readiness matrix
```

## 1. Their §4 open items — answered in code

**Q1: registration idiom for a stateful term.**
`CurrTerm(func=<Class>, params={...})` where the class subclasses
`isaaclab.managers.ManagerTermBase`. Verified against this fork's manager code
(`manager_base.py`): the manager initializes the class lazily with
`func(cfg=term_cfg, env=env)` when the sim starts playing, and — decisive —
`class_to_dict`/Hydra serialize `func` as a `module:ClassName` **string** and can
re-resolve it via `string_to_callable`. A pre-built *instance* (their
`FrontierTerrainTeacherTerm(...)` as `func`) also runs, but its `__init__(cfg,env)`
never fires (it's not a class at init time) and it serializes as an opaque object —
the class is the durable contract. Our terms therefore subclass `ManagerTermBase`
and lazy-init the teacher on first `__call__` (the terrain grid doesn't exist until
the scene is built anyway). Their `FrontierBinTeacher` is consumed **unchanged** —
only the wrapper is ours.
One constraint their doc should note: **term params cannot be injected via Hydra
CLI overrides** — `update_class_from_dict` raises `KeyError` on keys absent from
the registered default cfg. The arm swap must happen in Python after Hydra parses
(that's what `train_frontier.py` does), or in a cfg subclass.

**Q2: `success_fn` plumbing.**
Implemented as a **string key** into a module-level registry (`SUCCESS_FNS`) —
strings survive the env.yaml round-trip; closures/callables in `params` would make
the config dump either crash or lie. Three predicates ship:

| key | meaning | notes |
|---|---|---|
| `survival` | reached time_out without early termination | default; saturates on easy levels late in training |
| `distance` | walked ≥ `distance_fraction` × commanded distance (`‖cmd_vel‖·T·f`) | commanded speed read from `command_manager.get_command("base_velocity")` at reset time — confirmed available in `_reset_idx` (command resample happens *after* curriculum in `step()`) |
| `tile` | walked ≥ half a terrain tile | **greedy's own move_up signal** — use this for the signal-identical teacher-vs-greedy comparison their D2 asks for |

**Q3: teacher-state checkpointing.**
Implemented in both machine and human-readable forms. `FrontierOnPolicyRunner`
embeds the exact teacher state (posterior, RNG, cached distribution, call counter,
and configuration) in every `model_*.pt` checkpoint and restores it with the
policy/optimizer. A mismatched teacher configuration fails loudly. The term also
writes `<log_dir>/curriculum_teacher/teacher_state.json` every
`save_every_calls` reset batches and at run end; this supports inspection and
legacy resume via `--teacher_param load_state=<path>`. Fixed-grid evaluation
replaces the training curriculum with `FixedLevelProbe`, so it does not restore or
adapt the teacher.

## 2. The five arms (one term name, one task, one diff)

All arms replace `curriculum.terrain_levels.func` **under the same term name**, so
`terrain_generator.curriculum=True` stays on and every arm trains on the identical
row-graded terrain grid. (A `terrain_levels=null` control would silently switch the
generator to per-tile random difficulty — a different task. This is why the control
arm is a do-nothing term, not a removed term.)

| arm | term | what it does |
|---|---|---|
| `control` | `StaticTerrainLevels` | levels frozen at init draw — no curriculum motion |
| `greedy` | stock `terrain_levels_vel` | cfg untouched; the ±1 threshold walker |
| `scripted` | `ScriptedTerrainLevels` | open-loop linear level-cap ramp over `--scripted_total_steps` |
| `uniform` | `UniformTerrainLevels` | uniform level resample every reset (frontier_rl's uniform baseline) |
| `teacher` | `FrontierTerrainTeacher` | FrontierBinTeacher: observe → sample ∝ u(p̃)^γ + floor → actuate |

Teacher defaults = their D3–D6 (learnability, utility maximized over a
mean±1·std confidence interval, half-life 2048 episode-equivalents, floor 0.1,
γ=1); every knob overridable via
`--teacher_param k=v`. Teacher and stub RNGs seed from `env.cfg.seed` → same-seed
runs reproduce exactly (CPU-verified).

## 3. Intended container invocation

```bash
docker exec isaac-lab-base bash -c "cd /workspace/isaaclab && /isaac-sim/python.sh \
  scripts/curriculum-maxrl/isaaclab_integration/train_frontier.py \
  --task Isaac-Velocity-Rough-Anymal-C-v0 --headless --num_envs 1024 \
  --max_iterations 600 --seed 42 --arm teacher --success_fn tile \
  agent.run_name=s42"
```

The active retry uses this command shape and has reached policy updates in the
control arm. It has not yet emitted the required completion artifact, and the
teacher arm has not started, so simulator compatibility remains unverified
until artifact completion.

## 4. Runbook — the pre-registered experiment

```bash
# pilot: 1 seed × 5 arms × 600 iters × 1024 envs   (~5-6 h on one A10G)
scripts/curriculum-maxrl/isaaclab_integration/run_experiment.sh pilot
# full:  5 seeds × 5 arms × 1500 iters × 4096 envs
scripts/curriculum-maxrl/isaaclab_integration/run_experiment.sh full
# readout (in-container; tensorboard lives in kit python):
docker exec isaac-lab-base /isaac-sim/python.sh \
  /workspace/isaaclab/scripts/curriculum-maxrl/isaaclab_integration/analyze_arms.py
```

`analyze_arms.py` prints the pre-registered readout: **P-A** requires both
internal sampler conformance and concentration on ZPD levels defined by fixed-grid
evaluation; **P-B** uses macro-averaged fixed-level pass rates, with
terrain/tracking telemetry as biased diagnostics; **P-C** compares easy-level
fixed-grid pass rates at mid/final checkpoints against a supplied τ. Arms are
identified from each run's `params/arm.yaml`, not directory names. `eval_arms.py`
reuses one fixed seed for every arm and checkpoint.

Decision rules (frozen before the pilot, per their §2):
- P-A fails → fix the teacher/evidence signal; no outcome claims.
- P-B parity with greedy on the stock 10-row grid is the *expected* honest-null
  (their own prediction: greedy is near-optimal when every level is learnable).
  The discriminating condition is a grid with unlearnable-at-budget rows — add
  steeper rows / heavier randomization as a phase-2 grid variant.
- P-C regression in the teacher arm but not uniform → raise floor / escalate to
  ALP per SONIC Q8.

## 4b. Pilot launch postmortem (2026-07-23) — two operational facts

The pilot launch failed instantly for all five arms, and the original driver
reported **"OK in 0 min"** for each. Two lessons are now baked into
`run_experiment.sh`:

1. **Isaac Sim's kit python exits 0 on CUDA-context crashes.** All five arms died
   at startup (`omni.physx.tensors` "CUDA error: out of memory → Failed to create
   primary CUDA context") yet returned exit 0. Success is therefore detected by
   the training artifact (`Training time:` in the log), never the exit code.
2. **The GPU is time-shared with a verl curriculum×MaxRL run.** Its footprint
   varies by phase and has reached roughly 22.5 GB on a 23 GB card. The pilot
   landed with insufficient memory and every arm was killed at context
   creation. The driver now gates each arm on an **exclusive, quiet GPU**
   (0 compute procs + <2 GB used, stable across 2 checks, up to `GATE_HOURS`).

## 4c. Design-alignment review (2026-07-23, done while gate-waiting)

Audited the integration line-by-line against `frontier_rl/README.md`,
`ISAACLAB_DESIGN.md` D1–D10, and `SONIC_RESPONSE.md` Q1–Q10. Confirmed aligned:
teacher consumed unchanged; learnability default (Q2); deterministic
utility-space confidence optimism (Q3); evidence-scaled decay half-life 2048
(Q4, kept as pre-registered);
γ=1 (Q5 — γ>1 needs gated unlock structure terrain rows don't have);
statistics-only hindsight = the posterior itself (Q6); floor 0.1 + retention
metric in every arm (Q8). Three real gaps found and fixed:

1. **Construction-reset poisoning (bug, would have skewed the pilot's teacher
   arm).** `RslRlVecEnvWrapper.__init__` triggers one full `env.reset()` before
   any episode runs; the teacher term observed it as `num_envs` failures at the
   init level draw — ~1024 fake Bernoulli events (half a half-life of bogus
   evidence, Q4's warning about evidence-scaled decay makes this worse at scale).
   Now guarded: reset batches where `episode_length_buf` is all-zero are actuated
   but not observed. Regression test added.
2. **P-A gate quantity wasn't logged (pre-registration gap).** Their §2 gate is a
   *targeting ratio* over training, not a final posterior snapshot. The teacher
   term now logs `zpd_mass` (sampling mass on p̂∈[0.2,0.8] bins) and `zpd_bins`
   every reset batch → `Curriculum/*` TB series; `analyze_arms.py` computes the
   run-averaged targeting ratio and prints an explicit PASS/FAIL.
3. **P-B/P-C had no unbiased readout (the "gotcha" our own infra guide §7
   warned about).** Training telemetry is conditioned on each arm's own sampling
   distribution. Added `FixedLevelProbe` (curriculum term that pins env i to
   level i%n_bins and tallies per-level success) + `eval_arms.py`, which rolls
   out each run's mid/final checkpoints on the SAME fixed grid with the same
   `tile` predicate → `eval_frontier_<iter>.json` per run, merged into the
   analyzer's table. This is the equal-compute per-level comparison the
   frontier_rl REPORT uses (their mean-pass/AUC), ported to Isaac Lab.

Also exposed the Q9 `max_prob` tripwire as a teacher param (inactive by default
at 10 bins — floor + `effective_bins` telemetry are the auditable guards there).
CPU suite now 16/16.

## 5. Known limits (deliberate scope)

- `--distributed` and `--video` paths were dropped from the launcher fork — single-GPU
  arms only for now (determinism per their D4; the stock train.py keeps those paths).
- rsl_rl only. skrl/rl_games arms would need the same 20-line fork of their train.py.
- Ladder step 2 (commanded-speed bins via a custom CommandTerm) and step 3
  (MaxRL estimator in rsl_rl on a sparse-success manipulation task) are gated on
  this experiment's P-A/P-B outcome, per their §3.
- No effectiveness, adaptivity, or simulator-compatibility claim is justified
  until at least one artifact-verified smoke run completes; comparative claims
  still require the pre-registered multi-seed protocol.
