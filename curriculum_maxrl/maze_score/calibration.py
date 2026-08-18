#!/usr/bin/env python
"""Descriptive: does Prop. 1 predict realized coefficient mass inside a real
neural rollout process?

Prop. 1 says the expected absolute coefficient mass of a practical-MaxRL group
on a task with pass rate p is A_N(p) = 2(1 - p - (1-p)^N).  The MAZE-SCORE
telemetry records, for every group actually drawn, its success count K and its
realized mass, so the prediction is checkable at the deployed N=32.

Circularity is the whole difficulty: mass is a deterministic function of K, so
binning by the same group's K would test nothing.  We therefore estimate each
group's pass rate LEAVE-ONE-OUT from the OTHER groups drawn on the same level
in the same 25-update window, and compare the observed mass of the held-out
group against A_32 evaluated at that independent estimate.

This is a mechanism read, not an endpoint.  It is computed after the frozen
primary and changes no preregistered quantity.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

N_ROLLOUTS = 32
WINDOW = 25


def a_n(p, n=N_ROLLOUTS):
    return 2.0 * (1.0 - p - (1.0 - p) ** n)


def load(paths):
    """-> list of (level, window, k, mass, arm)"""
    rows = []
    for path in paths:
        arm = Path(path).name.split("_")[1]
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("record_type") != "telemetry":
                continue
            win = (int(r["completed_updates"]) - 1) // WINDOW
            for lvl, k, m in zip(r["selected_levels"], r["group_k"],
                                 r["coefficient_mass"], strict=True):
                rows.append((int(lvl), win, int(k), float(m), arm))
    return rows


def loo_pairs(rows):
    """Leave-one-out (predicted p, observed mass) pairs."""
    cells = defaultdict(list)
    for lvl, win, k, m, arm in rows:
        cells[(arm, lvl, win)].append((k, m))
    out = []
    for (arm, lvl, win), items in cells.items():
        if len(items) < 2:
            continue
        ks = np.array([k for k, _ in items], dtype=float)
        ms = np.array([m for _, m in items], dtype=float)
        tot, g = ks.sum(), ks.size
        for i in range(g):
            p_hat = (tot - ks[i]) / ((g - 1) * N_ROLLOUTS)
            out.append((p_hat, ms[i], arm, ks[i]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--output", required=True)
    ap.add_argument("--bins", type=int, default=12)
    args = ap.parse_args(argv)

    pairs = loo_pairs(load(args.inputs))
    p = np.array([x[0] for x in pairs])
    m = np.array([x[1] for x in pairs])
    k = np.array([x[3] for x in pairs])

    edges = np.linspace(0.0, 1.0, args.bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, args.bins - 1)
    binned = []
    for b in range(args.bins):
        sel = idx == b
        if sel.sum() < 30:
            continue
        pm = float(p[sel].mean())
        binned.append({
            "bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
            "n_groups": int(sel.sum()),
            "mean_pass_rate_loo": pm,
            "predicted_mass_A32": float(a_n(pm)),
            "observed_mass_mean": float(m[sel].mean()),
            "observed_mass_sem": float(m[sel].std(ddof=1) / np.sqrt(sel.sum())),
            # Over-dispersion check: under the i.i.d. Bernoulli model assumed by
            # Prop. 1, a group is silent with probability (1-p)^N.  Real groups
            # share a level but not a maze, so heterogeneity inside the group
            # inflates both tails and starves the middle.
            "predicted_dead_fraction_binomial": float((1.0 - pm) ** N_ROLLOUTS),
            "observed_dead_fraction": float((k[sel] == 0).mean()),
            "predicted_allpass_fraction_binomial": float(pm ** N_ROLLOUTS),
            "observed_allpass_fraction": float((k[sel] == N_ROLLOUTS).mean()),
        })

    pred = np.array([b["predicted_mass_A32"] for b in binned])
    obs = np.array([b["observed_mass_mean"] for b in binned])
    w = np.array([b["n_groups"] for b in binned], dtype=float)
    report = {
        "schema": "curriculum-maxrl/maze-score-calibration/v1",
        "note": ("descriptive mechanism read computed after the frozen primary; "
                 "changes no preregistered quantity"),
        "n_rollouts": N_ROLLOUTS,
        "window_updates": WINDOW,
        "n_group_draws_used": int(p.size),
        "weighted_mean_abs_error": float((w * np.abs(obs - pred)).sum() / w.sum()),
        "weighted_mean_signed_error": float((w * (obs - pred)).sum() / w.sum()),
        "pearson_r_binned": float(np.corrcoef(pred, obs)[0, 1]),
        "bins": binned,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"group draws: {p.size:,}   bins kept: {len(binned)}")
    print(f"weighted MAE {report['weighted_mean_abs_error']:.4f}   "
          f"signed {report['weighted_mean_signed_error']:+.4f}   "
          f"r {report['pearson_r_binned']:.4f}")
    print(f"{'p_loo':>7} {'predA32':>8} {'obsMass':>8} {'deadPred':>9} "
          f"{'deadObs':>8} {'n':>8}")
    for b in binned:
        print(f"{b['mean_pass_rate_loo']:7.3f} {b['predicted_mass_A32']:8.4f} "
              f"{b['observed_mass_mean']:8.4f} "
              f"{b['predicted_dead_fraction_binomial']:9.4f} "
              f"{b['observed_dead_fraction']:8.4f} {b['n_groups']:8d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
