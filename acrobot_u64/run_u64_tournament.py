"""U64 over-shooting arm: does the DEPLOYED-N peak location actually matter?

Every positive score result in the paper is confounded with peak hardness.
p*_N = 1 - N^(-1/(N-1)) falls .5 -> .169 -> .106 as N goes 2 -> 16 -> 32, so in
every study the winning arm is also the harder-peaked arm.  "Harder peaks beat
p(1-p)" and "the deployed-N peak is correct" make identical predictions in all
existing data, and only the second is an advance over the ProCuRL/SFL/LILO
p(1-p) literature.

This runner separates them.  It adds a fourth arm that scores with u_64
(p* = .0639) while the deployed estimator stays at N = 16 (p* = .169).  The
u_64 arm therefore over-shoots the deployed peak.  Three outcomes:

  u16 > p1mp and u16 > u64   -> peak-location specificity is supported
  u64 >= u16                 -> the finding is "harder beats softer", and
                                deployed-N peak specificity is NOT supported
  neither separates          -> inconclusive at this budget

DERIVATION AND PROVENANCE
This is a derivative of the sealed V2 tournament
(`ACROBOT_CURRICULUM_TOURNAMENT_LOCK.json`, sealed 2026-08-08T07:00:25Z).  The
V2 runner is NOT modified and NOT reused for evidence: its lock demands an
exact runtime match including platform and machine, and it ran on macOS arm64.
This host is Linux x86_64, so the V2 runner fail-closes here by design.

Instead every scientific constant is inherited verbatim from the V2 sources,
which are vendored byte-identically under `acrobot_u64/vendor/` and verified
against the V2 lock by `verify_vendor_lock.py`.  The simulation itself is the
vendored engine, called unmodified.  Only the sampling score changes.

WHY THIS IS A CLEAN 4-ARM DESIGN RATHER THAN "V2 PLUS ONE ARM"
The engine derives every RNG stream from (logical seed, domain) alone --
`engine_master_seed(s) = BASE + s * STRIDE` -- never from the arm index, and
the arms deliberately share roots as paired common random numbers.  Adding a
fourth arm therefore perturbs no existing arm's stream, and u_64 is CRN-paired
with u_16 automatically.  Running all four arms fresh on the same host also
makes the u16 - p1mp contrast a genuine cross-platform REPLICATION of the
frozen V2 primary (+.04803, CI [+.02094, +.07385]).

THE SCORE/ESTIMATOR DECOUPLING IS STRUCTURAL, NOT A HACK
`FrontierTeacher.n_rollouts` is used in exactly one place, `utility()`
(teacher.py:50), and the engine never reads it -- the engine uses its own
module constant `N_ROLLOUTS = 16` for `rollout_group` and for every estimator
decision.  Constructing the teacher with 64 therefore changes the sampling
score and nothing else.  `assert_score_estimator_decoupling()` checks this at
runtime and fail-closes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

import numpy as np  # noqa: E402

from frontier_rl.examples import run_acrobot_neural as engine  # noqa: E402
from frontier_rl.teacher import FrontierTeacher  # noqa: E402

SCHEMA = "curriculum-maxrl/acrobot-u64-tournament/v1"
LOCK_PATH = HERE / "ACROBOT_U64_LOCK.json"

# ---- constants inherited verbatim from the sealed V2 tournament -------------
N_ROLLOUTS = 16                     # deployed group size, identical across arms
TEACHER_DECAY = 0.7
TEACHER_FLOOR = 0.1
TEACHER_GAMMA = 1.0
LEARNING_RATE = 3e-4
EVALUATION_SEED_BASE = 1_000_000
ENGINE_MASTER_BASE = 50_000_000_000
ENGINE_MASTER_STRIDE = 10_000_000

TRANSITION_BUDGET = 2_000_000
EVAL_INTERVAL = 100_000
EVAL_N = 32
DEVELOPMENT_TRANSITION_BUDGET = 200_000
DEVELOPMENT_EVAL_INTERVAL = 50_000
DEVELOPMENT_EVAL_N = 16
QUICK_TRANSITION_BUDGET = 8_000
QUICK_EVAL_INTERVAL = 4_000
QUICK_EVAL_N = 2

# ---- schedule --------------------------------------------------------------
# Confirmatory seeds match V2 so that u16 - p1mp is a direct replication.
CONFIRMATORY_SEEDS = tuple(range(20_000, 20_020))
DEVELOPMENT_SEEDS = (20_100, 20_101, 20_102)
QUICK_SEEDS = (20_200,)

# arm name -> score exponent for FrontierTeacher.utility (None = special-cased)
ARMS: dict[str, int | None] = {
    "uniform_shared_h64": None,   # constant 1/8 over the task pool
    "p1mp_shared_h64": None,      # p(1-p) == u_2
    "u16_shared_h64": 16,         # matches the deployed N
    "u64_shared_h64": 64,         # OVER-SHOOTS the deployed N  <-- the new arm
}
ARM_ORDER = tuple(ARMS)

SOURCE_RELATIVE_PATHS = (
    "run_u64_tournament.py",
    "vendor/frontier_rl/examples/run_acrobot_neural.py",
    "vendor/frontier_rl/adapters/acrobot_neural.py",
    "vendor/frontier_rl/teacher.py",
    "vendor/frontier_rl/trainer.py",
    "vendor/frontier_rl/estimators.py",
    "vendor/frontier_rl/interfaces.py",
    "vendor/frontier_rl/__init__.py",
)

PINNED_RUNTIME_VERSIONS = {
    "python": "3.12.13",
    "numpy": "2.5.1",
    "gymnasium": "1.3.0",
}


# ---------------------------------------------------------------- provenance
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    out = {}
    for rel in SOURCE_RELATIVE_PATHS:
        p = HERE / rel
        if not p.is_file():
            raise RuntimeError(f"locked source is missing: {rel}")
        out[rel] = _sha256(p)
    return out


def live_runtime() -> dict:
    import gymnasium
    return {
        "python_implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "gymnasium": gymnasium.__version__,
    }


def runtime_versions(rt: dict) -> dict:
    return {k: rt[k] for k in PINNED_RUNTIME_VERSIONS}


def engine_master_seed(logical_seed: int) -> int:
    if type(logical_seed) is not int:
        raise TypeError("logical seed must be int")
    return ENGINE_MASTER_BASE + logical_seed * ENGINE_MASTER_STRIDE


def locked_schedule() -> dict:
    return {
        "arms": list(ARM_ORDER),
        "arm_score_exponent": {k: ARMS[k] for k in ARM_ORDER},
        "deployed_n_rollouts": N_ROLLOUTS,
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "quick_seeds": list(QUICK_SEEDS),
        "engine_master_base": ENGINE_MASTER_BASE,
        "engine_master_stride": ENGINE_MASTER_STRIDE,
        "transition_budget": TRANSITION_BUDGET,
        "eval_interval": EVAL_INTERVAL,
        "eval_n": EVAL_N,
        "learning_rate": LEARNING_RATE,
        "teacher_decay": TEACHER_DECAY,
        "teacher_floor": TEACHER_FLOOR,
        "teacher_gamma": TEACHER_GAMMA,
        "evaluation_seed_base": EVALUATION_SEED_BASE,
    }


def assert_score_estimator_decoupling() -> None:
    """The score exponent must not leak into the deployed estimator."""
    if engine.N_ROLLOUTS != N_ROLLOUTS:
        raise RuntimeError(
            f"engine deploys N={engine.N_ROLLOUTS}, expected {N_ROLLOUTS}")
    src = (VENDOR / "frontier_rl" / "examples" / "run_acrobot_neural.py").read_text()
    if ".n_rollouts" in src:
        raise RuntimeError(
            "engine reads teacher.n_rollouts; the score exponent would leak "
            "into the deployed estimator and the u64 arm would be invalid")
    # u_N identity spot check, including the N=2 learnability slice
    p = np.linspace(0.0, 1.0, 11)
    t2 = FrontierTeacher(8, 2, decay=TEACHER_DECAY, floor=TEACHER_FLOOR,
                         gamma=TEACHER_GAMMA, seed=0)
    if not np.allclose(t2.utility(p), np.maximum(p * (1 - p), 0.0), atol=1e-15):
        raise RuntimeError("u_2 does not reduce to p(1-p)")


def load_and_verify_lock(path: Path = LOCK_PATH) -> tuple[dict, str]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"source lock is missing: {path}")
    lock = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    if lock.get("schema") != SCHEMA:
        errors.append("lock schema mismatch")
    rt = live_runtime()
    if runtime_versions(rt) != PINNED_RUNTIME_VERSIONS:
        errors.append(f"library runtime is not pinned: {runtime_versions(rt)!r}")
    if lock.get("runtime") != rt:
        errors.append(f"runtime mismatch: live={rt!r}")
    if lock.get("schedule") != locked_schedule():
        errors.append("schedule mismatch")
    live_hashes = source_hashes()
    if set(lock.get("source_sha256", {})) != set(SOURCE_RELATIVE_PATHS):
        errors.append("source key set is not exact")
    if lock.get("source_sha256") != live_hashes:
        errors.append("source hash mismatch")
    if errors:
        raise RuntimeError("u64 source/runtime lock failed: " + "; ".join(errors))
    return lock, _sha256(path)


# ------------------------------------------------------------------- teacher
class _U64TournamentTeacher(FrontierTeacher):
    """V2 tournament teacher plus a u_64 over-shooting sampler."""

    def __init__(self, arm: str, seed: int):
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        exponent = ARMS[arm]
        # For uniform/p1mp the exponent is unused; keep the deployed N so the
        # object is well formed and never silently implies another score.
        super().__init__(
            len(engine.THRESHOLDS), exponent if exponent else N_ROLLOUTS,
            decay=TEACHER_DECAY, floor=TEACHER_FLOOR, gamma=TEACHER_GAMMA,
            seed=seed,
        )
        self.arm = arm
        self.score_exponent = exponent
        self.distribution_records: list[dict] = []

    def utility(self, p: np.ndarray) -> np.ndarray:
        if self.arm == "p1mp_shared_h64":
            return np.maximum(p * (1.0 - p), 0.0)
        return super().utility(p)      # exact u_N with n_rollouts = exponent

    def distribution(self) -> np.ndarray:
        posterior = self.pass_rate_estimates().copy()
        if self.arm == "uniform_shared_h64":
            probabilities = np.full(self.n_tasks, 1.0 / self.n_tasks)
        else:
            probabilities = super().distribution()
        self.distribution_records.append({
            "posterior_mean_pass_rates_before_group": posterior.tolist(),
            "task_probabilities": probabilities.tolist(),
        })
        return probabilities


_BASE_TEACHER_FACTORY = engine._teacher_for
_CAPTURE: list | None = None


def _factory(condition, seed):
    if _CAPTURE is None:
        raise RuntimeError("teacher factory used outside a managed run")
    teacher = _U64TournamentTeacher(condition.name, seed)
    _CAPTURE.append(teacher)
    return teacher


@contextmanager
def _patched_teacher_factory():
    global _CAPTURE
    if engine._teacher_for is not _BASE_TEACHER_FACTORY or _CAPTURE is not None:
        raise RuntimeError("teacher factory is already patched")
    capture: list = []
    _CAPTURE = capture
    engine._teacher_for = _factory
    try:
        yield capture
    finally:
        engine._teacher_for = _BASE_TEACHER_FACTORY
        _CAPTURE = None


def condition_for(arm: str):
    return engine.Condition(arm, "tournament", arm, "shared", 64, LEARNING_RATE)


# ----------------------------------------------------------------------- run
MODES = {
    "confirmatory": (CONFIRMATORY_SEEDS, TRANSITION_BUDGET, EVAL_INTERVAL, EVAL_N),
    "development": (DEVELOPMENT_SEEDS, DEVELOPMENT_TRANSITION_BUDGET,
                    DEVELOPMENT_EVAL_INTERVAL, DEVELOPMENT_EVAL_N),
    "quick": (QUICK_SEEDS, QUICK_TRANSITION_BUDGET, QUICK_EVAL_INTERVAL, QUICK_EVAL_N),
}


def run_one(arm: str, seed: int, mode: str, *, verify_lock: bool = True) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    seeds, budget, interval, eval_n = MODES[mode]
    if type(seed) is not int or seed not in seeds:
        raise RuntimeError(f"seed {seed!r} is not registered for {mode}")

    assert_score_estimator_decoupling()
    lock_digest = None
    if mode != "quick" and verify_lock:
        _, lock_digest = load_and_verify_lock()

    master_seed = engine_master_seed(seed)
    with _patched_teacher_factory() as capture:
        run = engine.run_condition(
            condition_for(arm),
            master_seed,
            budget=engine.RunBudget(transition_budget=budget),
            eval_interval_transitions=interval,
            eval_interval_updates=1,
            eval_n=eval_n,
            eval_seed_base=EVALUATION_SEED_BASE,
        )
    if len(capture) != 1:
        raise RuntimeError("exactly one teacher must be constructed per run")
    teacher = capture[0]
    if run.get("seed") != master_seed:
        raise RuntimeError("engine did not retain its master seed")
    if teacher.arm != arm:
        raise RuntimeError("teacher arm mismatch")

    return {
        "schema": SCHEMA,
        "arm": arm,
        "score_exponent": ARMS[arm],
        "deployed_n_rollouts": N_ROLLOUTS,
        "logical_seed": seed,
        "engine_master_seed": master_seed,
        "mode": mode,
        "transition_budget": budget,
        "eval_interval": interval,
        "eval_n": eval_n,
        "run": run,
        "teacher_distribution_records": teacher.distribution_records,
        "provenance": {
            "runtime": live_runtime(),
            "source_sha256": source_hashes(),
            "lock_sha256": lock_digest,
            "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--mode", default="confirmatory", choices=sorted(MODES))
    ap.add_argument("--output", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--build-lock", action="store_true",
                    help="write the lock file instead of running")
    args = ap.parse_args(argv)

    if args.build_lock:
        assert_score_estimator_decoupling()
        lock = {
            "schema": SCHEMA,
            "purpose": ("Source/runtime lock for the U64 over-shooting arm, a "
                        "derivative of the sealed V2 Acrobot tournament."),
            "derived_from": {
                "lock": "vendor/frontier_rl/examples/ACROBOT_CURRICULUM_TOURNAMENT_LOCK.json",
                "sealed_utc": "2026-08-08T07:00:25.740908+00:00",
                "note": ("V2 sources are vendored byte-identically and verified "
                         "by verify_vendor_lock.py; the V2 runner is unmodified "
                         "and is not used for this campaign because its lock "
                         "requires macOS arm64."),
            },
            "runtime": live_runtime(),
            "schedule": locked_schedule(),
            "source_sha256": source_hashes(),
        }
        out = Path(args.output)
        out.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        print(f"wrote lock {out}")
        return

    out = Path(args.output)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    record = run_one(args.arm, args.seed, args.mode)
    tmp = out.with_suffix(out.suffix + ".partial")
    tmp.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out)                      # atomic: no partial file is ever final
    print(f"{args.arm} seed={args.seed} mode={args.mode} -> {out}")


if __name__ == "__main__":
    main()
