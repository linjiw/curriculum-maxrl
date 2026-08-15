"""Score-exponent dose-response on Acrobot, deployed estimator fixed at N=16.

See ACROBOT_NSWEEP_PREREG.md.  Frozen before any result of this study or of the
U64 campaign was inspected.

Reuses, without modification, the V2-lock-verified vendored tree under
`acrobot_u64/vendor/` (SOURCE LOCK VERIFIED, 16/16 against the sealed V2 lock).
Only the sampling score's exponent varies; the engine's own `N_ROLLOUTS = 16`
governs every rollout group and every estimator decision, and the engine never
reads `teacher.n_rollouts`.
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
VENDOR = HERE.parent / "acrobot_u64" / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

import numpy as np  # noqa: E402

from frontier_rl.examples import run_acrobot_neural as engine  # noqa: E402
from frontier_rl.teacher import FrontierTeacher  # noqa: E402

SCHEMA = "curriculum-maxrl/acrobot-nsweep/v1"
LOCK_PATH = HERE / "ACROBOT_NSWEEP_LOCK.json"

N_ROLLOUTS = 16
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

CONFIRMATORY_SEEDS = tuple(range(20_000, 20_020))
DEVELOPMENT_SEEDS = (20_100, 20_101, 20_102)
QUICK_SEEDS = (20_200,)

SCORE_EXPONENTS = (2, 4, 8, 16, 32, 64, 128)
ARMS: dict[str, int | None] = {"uniform": None}
for _n in SCORE_EXPONENTS:
    ARMS[f"u{_n}"] = _n
ARM_ORDER = tuple(ARMS)

SOURCE_RELATIVE_PATHS = (
    "run_nsweep.py",
    "analyze_nsweep.py",
    "../acrobot_u64/vendor/frontier_rl/examples/run_acrobot_neural.py",
    "../acrobot_u64/vendor/frontier_rl/adapters/acrobot_neural.py",
    "../acrobot_u64/vendor/frontier_rl/teacher.py",
    "../acrobot_u64/vendor/frontier_rl/trainer.py",
    "../acrobot_u64/vendor/frontier_rl/estimators.py",
    "../acrobot_u64/vendor/frontier_rl/interfaces.py",
    "../acrobot_u64/vendor/frontier_rl/__init__.py",
)

PINNED_RUNTIME_VERSIONS = {
    "python_implementation": "CPython",
    "python": "3.12.13",
    "numpy": "2.5.1",
    "gymnasium": "1.3.0",
    "machine": "x86_64",
}


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    out = {}
    for rel in SOURCE_RELATIVE_PATHS:
        p = (HERE / rel).resolve()
        if not p.is_file():
            raise RuntimeError(f"locked source missing: {rel}")
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


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def engine_master_seed(s: int) -> int:
    if type(s) is not int:
        raise TypeError("logical seed must be int")
    return ENGINE_MASTER_BASE + s * ENGINE_MASTER_STRIDE


def locked_schedule() -> dict:
    return {
        "arms": list(ARM_ORDER),
        "arm_score_exponent": {k: ARMS[k] for k in ARM_ORDER},
        "score_exponents": list(SCORE_EXPONENTS),
        "deployed_n_rollouts": N_ROLLOUTS,
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "transition_budget": TRANSITION_BUDGET,
        "eval_interval": EVAL_INTERVAL,
        "eval_n": EVAL_N,
        "learning_rate": LEARNING_RATE,
        "teacher_decay": TEACHER_DECAY,
        "teacher_floor": TEACHER_FLOOR,
        "teacher_gamma": TEACHER_GAMMA,
        "evaluation_seed_base": EVALUATION_SEED_BASE,
    }


def assert_decoupling() -> None:
    if engine.N_ROLLOUTS != N_ROLLOUTS:
        raise RuntimeError(f"engine deploys N={engine.N_ROLLOUTS}")
    src = (VENDOR / "frontier_rl" / "examples" / "run_acrobot_neural.py").read_text()
    if ".n_rollouts" in src:
        raise RuntimeError("engine reads teacher.n_rollouts; exponent would leak")
    p = np.linspace(0.0, 1.0, 11)
    t2 = FrontierTeacher(8, 2, decay=TEACHER_DECAY, floor=TEACHER_FLOOR,
                         gamma=TEACHER_GAMMA, seed=0)
    if not np.allclose(t2.utility(p), np.maximum(p * (1 - p), 0.0), atol=1e-15):
        raise RuntimeError("u_2 does not reduce to p(1-p)")


def load_and_verify_lock(path: Path = LOCK_PATH) -> tuple[dict, str]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"lock missing: {path}")
    lock = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    if lock.get("schema") != SCHEMA:
        errors.append("schema mismatch")
    rt = live_runtime()
    pinned = {k: rt[k] for k in PINNED_RUNTIME_VERSIONS}
    if pinned != PINNED_RUNTIME_VERSIONS:
        errors.append(f"pinned runtime mismatch: {pinned!r}")
    if lock.get("runtime_pinned") != PINNED_RUNTIME_VERSIONS:
        errors.append("lock lacks expected pinned runtime")
    if lock.get("schedule") != locked_schedule():
        errors.append("schedule mismatch")
    live = source_hashes()
    if set(lock.get("source_sha256", {})) != set(SOURCE_RELATIVE_PATHS):
        errors.append("source key set is not exact")
    if lock.get("source_sha256") != live:
        errors.append("source hash mismatch")
    if errors:
        raise RuntimeError("nsweep lock failed: " + "; ".join(errors))
    return lock, _sha256(path)


class _NSweepTeacher(FrontierTeacher):
    def __init__(self, arm: str, seed: int):
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        exponent = ARMS[arm]
        super().__init__(
            len(engine.THRESHOLDS), exponent if exponent else N_ROLLOUTS,
            decay=TEACHER_DECAY, floor=TEACHER_FLOOR, gamma=TEACHER_GAMMA,
            seed=seed,
        )
        self.arm = arm
        self.score_exponent = exponent
        self.distribution_records: list[dict] = []

    def distribution(self) -> np.ndarray:
        posterior = self.pass_rate_estimates().copy()
        if self.arm == "uniform":
            probs = np.full(self.n_tasks, 1.0 / self.n_tasks)
        else:
            probs = super().distribution()
        self.distribution_records.append({
            "posterior_mean_pass_rates_before_group": posterior.tolist(),
            "task_probabilities": probs.tolist(),
        })
        return probs


_BASE_FACTORY = engine._teacher_for
_CAPTURE: list | None = None


def _factory(condition, seed):
    if _CAPTURE is None:
        raise RuntimeError("factory used outside a managed run")
    t = _NSweepTeacher(condition.name, seed)
    _CAPTURE.append(t)
    return t


@contextmanager
def _patched():
    global _CAPTURE
    if engine._teacher_for is not _BASE_FACTORY or _CAPTURE is not None:
        raise RuntimeError("factory already patched")
    cap: list = []
    _CAPTURE = cap
    engine._teacher_for = _factory
    try:
        yield cap
    finally:
        engine._teacher_for = _BASE_FACTORY
        _CAPTURE = None


MODES = {
    "confirmatory": (CONFIRMATORY_SEEDS, TRANSITION_BUDGET, EVAL_INTERVAL, EVAL_N),
    "development": (DEVELOPMENT_SEEDS, DEVELOPMENT_TRANSITION_BUDGET,
                    DEVELOPMENT_EVAL_INTERVAL, DEVELOPMENT_EVAL_N),
    "quick": (QUICK_SEEDS, QUICK_TRANSITION_BUDGET, QUICK_EVAL_INTERVAL, QUICK_EVAL_N),
}


def run_one(arm: str, seed: int, mode: str) -> dict:
    seeds, budget, interval, eval_n = MODES[mode]
    if type(seed) is not int or seed not in seeds:
        raise RuntimeError(f"seed {seed!r} not registered for {mode}")
    assert_decoupling()
    lock_digest = None
    if mode != "quick":
        _, lock_digest = load_and_verify_lock()
    master = engine_master_seed(seed)
    with _patched() as cap:
        run = engine.run_condition(
            engine.Condition(arm, "nsweep", arm, "shared", 64, LEARNING_RATE),
            master,
            budget=engine.RunBudget(transition_budget=budget),
            eval_interval_transitions=interval,
            eval_interval_updates=1,
            eval_n=eval_n,
            eval_seed_base=EVALUATION_SEED_BASE,
        )
    if len(cap) != 1:
        raise RuntimeError("exactly one teacher per run")
    if run.get("seed") != master:
        raise RuntimeError("engine did not retain master seed")
    return {
        "schema": SCHEMA,
        "arm": arm,
        "score_exponent": ARMS[arm],
        "deployed_n_rollouts": N_ROLLOUTS,
        "logical_seed": seed,
        "engine_master_seed": master,
        "mode": mode,
        "transition_budget": budget,
        "eval_interval": interval,
        "eval_n": eval_n,
        "run": run,
        "teacher_distribution_records": cap[0].distribution_records,
        "provenance": {
            "runtime": live_runtime(),
            "source_sha256": source_hashes(),
            "lock_sha256": lock_digest,
            "hostname": platform.node(),
            "cpu_model": _cpu_model(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--mode", default="confirmatory", choices=sorted(MODES))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--build-lock", action="store_true")
    args = ap.parse_args(argv)

    if args.build_lock:
        assert_decoupling()
        lock = {
            "schema": SCHEMA,
            "purpose": "Score-exponent dose-response, deployed N fixed at 16.",
            "derived_from": {
                "vendored_tree": "acrobot_u64/vendor (V2 SOURCE LOCK VERIFIED 16/16)",
                "companion_campaign": "acrobot_u64 (Hopper 9375605 + 9375630)",
            },
            "runtime_pinned": PINNED_RUNTIME_VERSIONS,
            "build_host_runtime": live_runtime(),
            "schedule": locked_schedule(),
            "source_sha256": source_hashes(),
        }
        Path(args.outdir).write_text(
            json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote lock {args.outdir}")
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for arm in ARM_ORDER:
        out = outdir / f"{arm}-{args.seed}.json"
        if out.exists() and not args.overwrite:
            print(f"skip existing {out.name}", flush=True)
            continue
        rec = run_one(arm, args.seed, args.mode)
        tmp = out.with_suffix(out.suffix + ".partial")
        tmp.write_text(json.dumps(rec, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(out)
        print(f"{arm} seed={args.seed} -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
