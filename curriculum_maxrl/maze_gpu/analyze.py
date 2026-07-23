"""Summarize sweep logs: final/AUC, frontier depth, and K=0 group rate.

Both optimization-step and wall-clock AUC are reported and anchored at the
post-SFT evaluation before the first update. Matched-time runs can complete
different numbers of steps, so total process wall-clock AUC (including common
periodic evaluation overhead) is the primary comparison. The old unanchored
step AUC is retained under an explicitly historical key.

Usage: python3 analyze.py sweep_*.jsonl
"""

from __future__ import annotations

import glob
import json
import sys

import numpy as np


def _trapezoid(y, x):
    """NumPy 1.x/2.x compatible trapezoidal integration."""
    integrate = getattr(np, "trapezoid", None)
    return (np.trapz if integrate is None else integrate)(y, x)


def load(path):
    recs = [json.loads(l) for l in open(path)]
    return [r for r in recs if "eval" in r]


def frontier(ev: dict, thresh: float = 0.5) -> int:
    """Deepest level with pass rate > thresh (-1 if none)."""
    best = -1
    for k in sorted(ev, key=int):
        if ev[k] > thresh:
            best = int(k)
    return best


def summarize(path):
    recs = load(path)
    if len(recs) < 2:
        return None
    rl = [r for r in recs if r["step"] >= 0]
    steps = np.array([r["step"] for r in rl])
    elapsed = np.array([r.get("elapsed", np.nan) for r in rl], dtype=float)
    mean_ev = np.array([np.mean(list(r["eval"].values())) for r in rl])
    fr = np.array([frontier(r["eval"]) for r in rl])
    dead_snapshots = np.array([r.get("dead_groups", np.nan) for r in rl])
    base = next((r for r in recs if r["step"] < 0), None)
    base_mean = (float(np.mean(list(base["eval"].values())))
                 if base is not None else float(mean_ev[0]))
    step_x = np.concatenate(([0.0], steps.astype(float) + 1.0))
    curve = np.concatenate(([base_mean], mean_ev))
    auc_steps = float(_trapezoid(curve, step_x) / max(step_x[-1], 1.0))
    historical_auc_steps = float(
        _trapezoid(mean_ev, steps) / max(steps[-1] - steps[0], 1)
    )
    if np.isfinite(elapsed).all() and elapsed[-1] > 0:
        wall_x = np.concatenate(([0.0], elapsed))
        auc_wall = float(_trapezoid(curve, wall_x) / wall_x[-1])
    else:
        auc_wall = float("nan")
    # retention: worst level-0/1 pass over the last half of training,
    # relative to the post-SFT baseline (forgetting indicator)
    retention = None
    if base is not None:
        half = rl[len(rl) // 2:]
        easy_min = min(min(r["eval"].get("0", r["eval"].get(0, 1.0)),
                           r["eval"].get("1", r["eval"].get(1, 1.0))) for r in half)
        easy_base = min(base["eval"].get("0", base["eval"].get(0, 1.0)),
                        base["eval"].get("1", base["eval"].get(1, 1.0)))
        retention = float(easy_min - easy_base)
    last = rl[-1]
    total_groups = last.get("cumulative_groups")
    if total_groups:
        dead_rate = last.get("cumulative_dead_groups", 0) / total_groups
        all_pass_rate = last.get("cumulative_all_pass_groups", 0) / total_groups
    else:
        # Historical logs only contain the single training step coinciding
        # with evaluation. This is a sparse snapshot, not a run-wide rate.
        dead_rate = float("nan")
        all_pass_rate = float("nan")
    finite_dead_snapshots = dead_snapshots[np.isfinite(dead_snapshots)]
    historical_avg_dead_snapshot = (
        float(finite_dead_snapshots.mean())
        if finite_dead_snapshots.size else float("nan")
    )
    out = {
        "final_mean": float(mean_ev[-1]),
        "best_mean": float(mean_ev.max()),
        "auc": auc_steps,  # compatibility alias for historical consumers
        "auc_steps": auc_steps,
        "historical_auc_steps_unanchored": historical_auc_steps,
        "auc_wall_seconds": auc_wall,
        "final_frontier": int(fr[-1]),
        "best_frontier": int(fr.max()),
        "dead_group_rate": float(dead_rate),
        "all_pass_group_rate": float(all_pass_rate),
        "historical_avg_dead_snapshot": historical_avg_dead_snapshot,
        "final_eval": rl[-1]["eval"],
        "steps_run": int(steps[-1]),
        "easy_retention": retention,
    }
    # pass@8 coverage (may be absent in older logs)
    last_pk = rl[-1].get("passk")
    if last_pk:
        out["final_pass8"] = float(np.mean([v.get("8", v.get(8, 0.0))
                                            for v in last_pk.values()]))
    return out


def main():
    paths = sys.argv[1:] or sorted(glob.glob("sweep_*.jsonl"))
    print(f"{'config':34s} {'final':>6s} {'best':>6s} {'auc_t':>6s} {'auc_s':>6s} "
          f"{'frontier':>8s} {'K=0 rate':>9s} {'steps':>6s} {'pass@8':>7s}")
    for p in paths:
        s = summarize(p)
        if s is None:
            print(f"{p:34s}  (incomplete)")
            continue
        name = p.replace("sweep_", "").replace("matched_", "").replace("_s0.jsonl", "")
        p8 = f"{s['final_pass8']:7.3f}" if "final_pass8" in s else "      -"
        dead = (f"{s['dead_group_rate']:9.3f}"
                if np.isfinite(s["dead_group_rate"]) else " historical")
        auc_t = (f"{s['auc_wall_seconds']:6.3f}"
                 if np.isfinite(s['auc_wall_seconds']) else "     -")
        print(f"{name:34s} {s['final_mean']:6.3f} {s['best_mean']:6.3f} "
              f"{auc_t} {s['auc_steps']:6.3f} {s['best_frontier']:8d} {dead} "
              f"{s['steps_run']:6d} {p8}")
    print("\nper-level final pass rates:")
    for p in paths:
        s = summarize(p)
        if s is None:
            continue
        name = p.replace("sweep_", "").replace("matched_", "").replace("_s0.jsonl", "")
        lv = " ".join(f"{v:.2f}" for _, v in sorted(s["final_eval"].items(), key=lambda kv: int(kv[0])))
        print(f"  {name:32s} [{lv}]")


if __name__ == "__main__":
    main()
