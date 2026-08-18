#!/usr/bin/env python
"""Exact group-law accounting for the MAZE-SCORE telemetry.

Two identities, both algebraic rather than statistical, checked to floating
point on the real campaign.

(1) Realized mass under practical centered drop-all-fail MaxRL.  For a group
    with K successes out of N the realized absolute coefficient mass is
    M(K) = 2(1 - K/N) 1{K>0}, so for ANY joint binary group law -- no
    independence, no identical distribution --

        E[M] = 2 ( Pr(K>0) - E[K]/N ).

(2) The task-granularity gap.  A curriculum that scores an aggregate z by its
    mean pass rate p_bar computes A_N(p_bar) = 2(1 - p_bar - (1-p_bar)^N).
    Subtracting (1),

        A_N(p_bar) - E[M] = 2 [ Pr(K=0) - (1-p_bar)^N ],

    i.e. the plug-in over-prediction equals exactly twice the aggregate's
    EXCESS ALL-FAIL PROBABILITY relative to the binomial at the same mean.

The MAZE-SCORE trainer draws one concrete maze per group and repeats that same
prompt N times, while the teacher's posterior pools over the mazes of a level.
So the unit that is conditionally i.i.d. is the maze; the unit the curriculum
scores is the level.  Identity (2) is the price of that mismatch.

Reports cluster-robust intervals over the 48 seed blocks, never over the
288,000 individual group draws, and repeats the calculation at three window
widths.  Descriptive; computes no preregistered quantity.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

N_ROLLOUTS = 32
BOOT = 20_000
BOOT_SEED = 20260818


def a_n(p, n=N_ROLLOUTS):
    return 2.0 * (1.0 - p - (1.0 - p) ** n)


def load(paths):
    """-> list of (seed, arm, level, update, k, mass)"""
    rows = []
    for path in paths:
        name = Path(path).name              # mazescore_{arm}_s{seed}.telemetry.jsonl
        arm = name.split("_")[1]
        seed = int(name.split("_s")[1].split(".")[0])
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("record_type") != "telemetry":
                continue
            up = int(r["completed_updates"])
            for lvl, k, m in zip(r["selected_levels"], r["group_k"],
                                 r["coefficient_mass"], strict=True):
                rows.append((seed, arm, int(lvl), up, int(k), float(m)))
    return rows


def cell_table(rows, window):
    """(seed, arm, level, window) -> arrays of K and mass."""
    cells = defaultdict(lambda: ([], []))
    for seed, arm, lvl, up, k, m in rows:
        key = (seed, arm, lvl, (up - 1) // window)
        cells[key][0].append(k)
        cells[key][1].append(m)
    return {k: (np.array(v[0], float), np.array(v[1], float))
            for k, v in cells.items()}


def identity_check(cells):
    """Max |deviation| of both identities over every cell."""
    d1 = d2 = 0.0
    n_cells = 0
    for (ks, ms) in cells.values():
        if ks.size < 2:
            continue
        n_cells += 1
        p_bar = ks.mean() / N_ROLLOUTS
        q = float((ks > 0).mean())
        m_bar = float(ms.mean())
        d1 = max(d1, abs(m_bar - 2.0 * (q - p_bar)))
        lhs = a_n(p_bar) - m_bar
        rhs = 2.0 * ((1.0 - q) - (1.0 - p_bar) ** N_ROLLOUTS)
        d2 = max(d2, abs(lhs - rhs))
    return n_cells, d1, d2


def per_seed_arm(rows):
    """Per (seed, arm): mean plug-in prediction, realized mass, silent share."""
    agg = defaultdict(lambda: [0.0, 0.0, 0, 0])   # sum pred, sum mass, dead, n
    by_cell = cell_table(rows, 25)
    for (seed, arm, lvl, win), (ks, ms) in by_cell.items():
        if ks.size < 2:
            continue
        p_bar = ks.mean() / N_ROLLOUTS
        a = agg[(seed, arm)]
        a[0] += a_n(p_bar) * ks.size
        a[1] += ms.sum()
        a[2] += int((ks == 0).sum())
        a[3] += ks.size
    out = {}
    for (seed, arm), (sp, sm, dead, n) in agg.items():
        out[(seed, arm)] = {"predicted": sp / n, "realized": sm / n,
                            "silent": dead / n, "n": n}
    return out


def boot_ci(x):
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, x.size, size=(BOOT, x.size))
    m = x[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    rows = load(args.inputs)
    report = {
        "schema": "curriculum-maxrl/maze-score-group-law-audit/v1",
        "note": ("descriptive; no preregistered quantity is computed. "
                 "Uncertainty is clustered on the 48 seed blocks."),
        "n_group_draws": len(rows),
        "identity_checks": {},
        "window_sensitivity": {},
    }

    for window in (10, 25, 50):
        cells = cell_table(rows, window)
        n_cells, d1, d2 = identity_check(cells)
        report["identity_checks"][f"window_{window}"] = {
            "n_cells": n_cells,
            "max_abs_dev_mass_equals_2_q_minus_p": d1,
            "max_abs_dev_plugin_gap_equals_2_excess_silence": d2,
        }

    seed_arm = per_seed_arm(rows)
    seeds = sorted({s for (s, _) in seed_arm})
    for arm in ("un", "learn", "unif"):
        pred = np.array([seed_arm[(s, arm)]["predicted"] for s in seeds])
        real = np.array([seed_arm[(s, arm)]["realized"] for s in seeds])
        sil = np.array([seed_arm[(s, arm)]["silent"] for s in seeds])
        ratio = real / pred
        report[f"arm_{arm}"] = {
            "n_seed_blocks": len(seeds),
            "mean_predicted_mass": float(pred.mean()),
            "mean_realized_mass": float(real.mean()),
            "realization_ratio_mean": float(ratio.mean()),
            "realization_ratio_ci95_seed_clustered": boot_ci(ratio),
            "silent_group_share_mean": float(sil.mean()),
            "silent_group_share_ci95_seed_clustered": boot_ci(sil),
        }
    # paired, seed-clustered: does un realize less of its prediction than learn?
    r_un = np.array([seed_arm[(s, "un")]["realized"] / seed_arm[(s, "un")]["predicted"]
                     for s in seeds])
    r_le = np.array([seed_arm[(s, "learn")]["realized"] / seed_arm[(s, "learn")]["predicted"]
                     for s in seeds])
    d = r_un - r_le
    report["paired_realization_ratio_un_minus_learn"] = {
        "mean": float(d.mean()),
        "ci95_seed_clustered": boot_ci(d),
        "negative_blocks": int((d < 0).sum()),
        "n": int(d.size),
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")

    print(f"group draws {len(rows):,}   seed blocks {len(seeds)}")
    for w, c in report["identity_checks"].items():
        print(f"  {w}: {c['n_cells']:6d} cells   "
              f"max|M - 2(q-p)| = {c['max_abs_dev_mass_equals_2_q_minus_p']:.3e}   "
              f"max|gap - 2*excess silence| = "
              f"{c['max_abs_dev_plugin_gap_equals_2_excess_silence']:.3e}")
    for arm in ("un", "learn", "unif"):
        a = report[f"arm_{arm}"]
        lo, hi = a["realization_ratio_ci95_seed_clustered"]
        print(f"  {arm:>5}: predicted {a['mean_predicted_mass']:.4f}  "
              f"realized {a['mean_realized_mass']:.4f}  "
              f"ratio {a['realization_ratio_mean']:.4f} [{lo:.4f},{hi:.4f}]  "
              f"silent {a['silent_group_share_mean']*100:.1f}%")
    pr = report["paired_realization_ratio_un_minus_learn"]
    print(f"  paired ratio un-learn: {pr['mean']:+.4f} "
          f"[{pr['ci95_seed_clustered'][0]:+.4f},{pr['ci95_seed_clustered'][1]:+.4f}]  "
          f"{pr['negative_blocks']}/{pr['n']} blocks negative")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
