"""Analyzer for the frozen AMaze gate confirmatory campaign.

Implements exactly AMAZE_GATE_PREREG.md sections 4-5.  Refuses an incomplete
2x10 matrix, refuses to run twice, and requires the shipped minimax.evaluate
CSV for every cell.

Primary: paired difference in mean held-out test_solved_rate over the three
shipped mazes, plrGate - plrMM, ten seeds; exact two-sided sign-flip over
2**10 assignments; SESOI +0.02.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ARMS = ("plrMM", "plrGate")
SEEDS = tuple(range(2001, 2011))
MAZES = ("Maze-SixteenRooms", "Maze-Labyrinth", "Maze-StandardMaze")
SESOI = 0.02
ALPHA = 0.05
BOOT = 20_000
BOOT_SEED = 20260817


class AnalysisError(RuntimeError):
    pass


def sign_flip_p(d: np.ndarray) -> float:
    if d.size > 24:
        raise AnalysisError("exact sign-flip refuses n>24")
    sums = np.zeros(1)
    for v in d:
        sums = np.concatenate((sums - v, sums + v))
    obs = abs(float(d.sum()))
    tol = max(1e-15, 1e-14 * max(1.0, obs))
    return float(np.count_nonzero(np.abs(sums) >= obs - tol) / sums.size)


def boot_ci(d: np.ndarray) -> list[float]:
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, d.size, size=(BOOT, d.size))
    m = d[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def load_eval(path: Path) -> dict[str, float]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise AnalysisError(f"{path}: expected one evaluation row, got {len(rows)}")
    r = rows[0]
    out = {}
    for m in MAZES:
        for k in ("test_solved_rate", "test_return"):
            col = f"eval/a0:{k}:{m}"
            if col not in r:
                raise AnalysisError(f"{path}: missing column {col}")
            out[f"{k}:{m}"] = float(r[col])
    return out


MIN_CHECKPOINT_UPDATES = 29_900


def load(results: Path) -> dict:
    # Amendment 2026-08-19 (outcome-blind): minimax checkpoints on
    # `tick % checkpoint_interval == 0` and never saves after the loop, so
    # checkpoint.pkl is NOT guaranteed to be the final model.  The prereg
    # evaluates "each run's final checkpoint at update 30,000", so the stored
    # training state of every evaluated checkpoint is now verified here.
    budget_path = results / "ckpt_budget.json"
    if not budget_path.is_file():
        raise AnalysisError(
            f"missing {budget_path}; run verify_checkpoint_budget.py first")
    budget = json.loads(budget_path.read_text())
    cells = {}
    for arm in ARMS:
        for s in SEEDS:
            xpid = f"arm-{arm}-s{s}-u30000"
            d = results / xpid
            ev = results / "eval" / f"{xpid}.csv"
            if not (d / "checkpoint.pkl").is_file():
                raise AnalysisError(f"missing checkpoint for {xpid}")
            if not (d / "meta.json").is_file():
                raise AnalysisError(f"missing meta.json for {xpid}")
            if not ev.is_file():
                raise AnalysisError(f"missing shipped evaluate CSV for {xpid}: {ev}")
            # minimax writes meta.json as {config: {..., train_runner_args: {...}}}.
            # Amendment 2026-08-18 (pre-data): the first draft read a flat
            # "args" key that does not exist and would have refused every cell.
            meta = json.loads((d / "meta.json").read_text())
            cfg = meta.get("config", {})
            tra = cfg.get("train_runner_args", {})
            if int(cfg.get("n_total_updates", -1)) != 30000:
                raise AnalysisError(f"{xpid}: n_total_updates != 30000")
            if int(cfg.get("seed", -1)) != s:
                raise AnalysisError(f"{xpid}: seed mismatch")
            if int(tra.get("n_parallel", -1)) != 32 or int(tra.get("n_eval", -1)) != 1:
                raise AnalysisError(f"{xpid}: batch structure is not the shipped 32x1")
            if arm == "plrGate":
                if tra.get("ued_score") != "coefficient_activity":
                    raise AnalysisError(f"{xpid}: not the gate score")
                if tra.get("frontier_mode") != "gate":
                    raise AnalysisError(f"{xpid}: frontier_mode != gate")
                if int(tra.get("frontier_n_rollouts", -1)) != 8:
                    raise AnalysisError(f"{xpid}: N != 8")
            else:
                if tra.get("ued_score") != "max_mc":
                    raise AnalysisError(f"{xpid}: baseline is not max_mc")
            # Completion is n_updates reaching the budget in logs.csv; upstream
            # never flips meta.successful, so it is not consulted.
            log = (d / "logs.csv").read_text().splitlines()
            hdr = log[0].lstrip("# ").split(",")
            last = dict(zip(hdr, log[-1].split(",")))
            # logs.csv is flushed every `log_interval` TICKS, so its last row
            # sits up to one interval short of the budget; a loose bound only.
            if int(float(last.get("n_updates", 0))) < 29_000:
                raise AnalysisError(f"{xpid}: training did not reach 30000 updates")
            stored = budget.get(xpid)
            if stored is None:
                raise AnalysisError(f"{xpid}: no entry in ckpt_budget.json")
            if int(stored) < MIN_CHECKPOINT_UPDATES:
                raise AnalysisError(
                    f"{xpid}: evaluated checkpoint holds n_updates={stored}, "
                    f"below the required {MIN_CHECKPOINT_UPDATES}; this is not "
                    f"the final checkpoint the preregistration evaluates")
            cells[(arm, s)] = load_eval(ev)
    return cells


def vec(cells, arm, key):
    return np.array([np.mean([cells[(arm, s)][f"{key}:{m}"] for m in MAZES])
                     for s in SEEDS])


def contrast(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    return {
        "mean_paired_difference": float(d.mean()),
        "sample_std": float(d.std(ddof=1)),
        "positive_pairs": int(np.count_nonzero(d > 0)),
        "n": int(d.size),
        "paired_bootstrap_ci95": boot_ci(d),
        "exact_two_sided_sign_flip_p": sign_flip_p(d),
        "paired_differences": [float(x) for x in d],
        "arm_means": {"plrGate": float(a.mean()), "plrMM": float(b.mean())},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results")
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"{out} exists; the prereg allows one analysis run")

    cells = load(Path(args.results))

    primary = contrast(vec(cells, "plrGate", "test_solved_rate"),
                       vec(cells, "plrMM", "test_solved_rate"))
    m, lo, hi = (primary["mean_paired_difference"],
                 *primary["paired_bootstrap_ci95"])
    p = primary["exact_two_sided_sign_flip_p"]
    if m >= SESOI and p <= ALPHA:
        verdict = "gate_beats_upstream"
    elif hi < SESOI:
        verdict = "gate_does_not_beat_upstream"
    else:
        verdict = "inconclusive_at_n10"

    per_maze = {}
    for mz in MAZES:
        for key in ("test_solved_rate", "test_return"):
            a = np.array([cells[("plrGate", s)][f"{key}:{mz}"] for s in SEEDS])
            b = np.array([cells[("plrMM", s)][f"{key}:{mz}"] for s in SEEDS])
            per_maze[f"{key}:{mz}"] = contrast(a, b)

    report = {
        "schema": "curriculum-maxrl/amaze-gate-confirmatory-analysis/v1",
        "prereg": "ued_benchmark/AMAZE_GATE_PREREG.md",
        "sesoi": SESOI, "alpha": ALPHA,
        "primary_mean_solved_rate": primary,
        "verdict": verdict,
        "secondary_mean_test_return": contrast(vec(cells, "plrGate", "test_return"),
                                               vec(cells, "plrMM", "test_return")),
        "secondary_per_maze_descriptive": per_maze,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"PRIMARY  mean solved-rate, plrGate - plrMM over {len(SEEDS)} seeds")
    print(f"  arm means : gate {primary['arm_means']['plrGate']:.4f}  "
          f"upstream {primary['arm_means']['plrMM']:.4f}")
    print(f"  diff      : {m:+.4f}  CI [{lo:+.4f},{hi:+.4f}]  exact p {p:.4f}  "
          f"{primary['positive_pairs']}/{primary['n']} pairs")
    print(f"  VERDICT   : {verdict}")
    print("\nper-maze (descriptive):")
    for mz in MAZES:
        c = per_maze[f"test_solved_rate:{mz}"]
        print(f"  {mz:<20} {c['mean_paired_difference']:+.4f}  "
              f"CI [{c['paired_bootstrap_ci95'][0]:+.4f},{c['paired_bootstrap_ci95'][1]:+.4f}]  "
              f"p {c['exact_two_sided_sign_flip_p']:.4f}  {c['positive_pairs']}/{c['n']}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
