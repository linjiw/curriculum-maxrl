#!/usr/bin/env python3
"""Analysis for the balanced maze factorial (run_factorial.sh, prereg
2026-08-05). Pure arithmetic on the prespecified endpoint — no model
fitting, no metric shopping.

Primary endpoint: delta mean pass@8 over the 13 levels (final step-250
eval minus post-SFT step -1), fixed held-out set.
Primary contrast P-F1: paired (same seed block, same sampler)
MaxRL - GRPO delta-cov positive in >= 5/6 blocks under BOTH samplers.
Exact two-sided sign-test p at 6/6: 0.031; at 5/6: 0.219.

Secondary (exploratory, stated in prereg): teacher-amplification
interaction under GRPO; easy-band (L1-3) vs frontier-band (L4-6)
decomposition; grpo_mass (P-G0a/b) and grpo_nostd (P-G0c) arms.

Usage: python3 fact_analyze.py [--prefix fact250] [--seeds 6]
Writes results_factorial.json next to the logs.
"""
from __future__ import annotations

import argparse
import json
import os
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def read_run(path):
    """(init, final) mean pass@8 across levels + per-level dict, or None."""
    rows = [json.loads(l) for l in open(path)]
    init = next((r for r in rows if r.get("step") == -1), None)
    fin = next((r for r in rows if r.get("final")), None)
    if init is None or fin is None:
        return None

    def cov(rec):
        pk = rec["passk"]
        per = {lv: v.get("8", v.get(8, 0.0)) for lv, v in pk.items()}
        return float(np.mean(list(per.values()))), per

    c0, p0 = cov(init)
    c1, p1 = cov(fin)
    # EXPLORATORY secondary (not the prereg endpoint): coverage AUC =
    # mean cov8 over all in-training evals minus init — integrates out
    # the single-eval endpoint noise (~±.03 per eval at 16 mazes/level).
    covs = [cov(r)[0] for r in rows
            if "passk" in r and r.get("step", -1) >= 0]
    cov_auc = float(np.mean(covs)) - c0 if covs else None
    return {"init_cov8": c0, "final_cov8": c1, "delta_cov8": c1 - c0,
            "cov_auc_delta": cov_auc,
            "init_per_level": p0, "final_per_level": p1,
            "final_step": fin.get("step")}


def band_delta(run, levels):
    lv = [str(x) for x in levels]
    d = [run["final_per_level"][x] - run["init_per_level"][x]
         for x in lv if x in run["final_per_level"]]
    return float(np.mean(d)) if d else None


def sign_test_p(k, n):
    """Exact two-sided sign test: P(#pos >= k or <= n-k) under fair coin."""
    from math import comb
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="fact250")
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()

    cells = {}
    arms = [("uniform", "maxrl"), ("uniform", "grpo"),
            ("frontier_un", "maxrl"), ("frontier_un", "grpo"),
            ("grpo_mass", "grpo"), ("uniform", "grpo_nostd")]
    for teach, est in arms:
        for s in range(args.seeds):
            p = os.path.join(HERE, f"{args.prefix}_{teach}_{est}_s{s}.jsonl")
            if os.path.exists(p):
                r = read_run(p)
                if r:
                    r["easy_band"] = band_delta(r, [1, 2, 3])
                    r["frontier_band"] = band_delta(r, [4, 5, 6])
                    cells[f"{teach}/{est}/s{s}"] = r

    out = {"cells": cells, "contrasts": {}}
    print(f"{'cell':>28} {'d cov8':>8} {'covAUC':>8} {'easy':>8} {'frontier':>8}")
    for k, r in sorted(cells.items()):
        ca = r.get("cov_auc_delta")
        print(f"{k:>28} {r['delta_cov8']:+.3f} "
              f"{(ca if ca is not None else float('nan')):+.3f} "
              f"{(r['easy_band'] if r['easy_band'] is not None else float('nan')):+.3f} "
              f"{(r['frontier_band'] if r['frontier_band'] is not None else float('nan')):+.3f}")

    # P-F1: paired estimator contrast per sampler (prereg endpoint =
    # delta_cov8; cov_auc_delta reported alongside as exploratory)
    for teach in ("uniform", "frontier_un"):
        for metric in ("delta_cov8", "cov_auc_delta"):
            diffs = []
            for s in range(args.seeds):
                a = cells.get(f"{teach}/maxrl/s{s}")
                b = cells.get(f"{teach}/grpo/s{s}")
                if a and b and a.get(metric) is not None \
                        and b.get(metric) is not None:
                    diffs.append(a[metric] - b[metric])
            if not diffs:
                continue
            npos = sum(d > 0 for d in diffs)
            tag = "P-F1" if metric == "delta_cov8" else "expl-AUC"
            out["contrasts"][f"{tag} {teach}: maxrl-grpo {metric}"] = {
                "per_seed": diffs, "n_pos": npos, "n": len(diffs),
                "sign_test_p_two_sided": sign_test_p(
                    max(npos, len(diffs) - npos), len(diffs)),
                "mean": float(np.mean(diffs)),
            }
            print(f"\n{tag} [{teach}] maxrl-grpo {metric}: "
                  f"{npos}/{len(diffs)} positive, mean {np.mean(diffs):+.4f}")

    # secondary: teacher amplification under each estimator
    for est in ("maxrl", "grpo"):
        diffs = []
        for s in range(args.seeds):
            t = cells.get(f"frontier_un/{est}/s{s}")
            u = cells.get(f"uniform/{est}/s{s}")
            if t and u:
                diffs.append(t["delta_cov8"] - u["delta_cov8"])
        if diffs:
            out["contrasts"][f"interaction {est}: teacher-uniform d_cov8"] = {
                "per_seed": diffs, "mean": float(np.mean(diffs)),
                "n_neg": sum(d < 0 for d in diffs), "n": len(diffs)}
            print(f"interaction [{est}] teacher-uniform Δcov8: "
                  f"{[f'{d:+.3f}' for d in diffs]}")

    # P-G0a: grpo_mass+grpo vs uniform+grpo (does GRPO's own scheduler save it?)
    # P-G0c: grpo_nostd holds the easy band?
    for name, key, ref in [("P-G0a grpo_mass", "grpo_mass/grpo", "uniform/grpo"),
                           ("P-G0c grpo_nostd", "uniform/grpo_nostd", "uniform/grpo")]:
        rows = [(cells.get(f"{key}/s{s}"), cells.get(f"{ref}/s{s}"))
                for s in range(args.seeds)]
        rows = [(a, b) for a, b in rows if a and b]
        if rows:
            out["contrasts"][name] = {
                "arm_delta_cov8": [a["delta_cov8"] for a, _ in rows],
                "ref_delta_cov8": [b["delta_cov8"] for _, b in rows],
                "arm_easy_band": [a["easy_band"] for a, _ in rows],
            }
            arm_s = [f"{a['delta_cov8']:+.3f}" for a, _ in rows]
            ref_s = [f"{b['delta_cov8']:+.3f}" for _, b in rows]
            print(f"{name}: arm dcov8 {arm_s} vs ref {ref_s}")

    path = os.path.join(HERE, "results_factorial.json")
    json.dump(out, open(path, "w"), indent=1)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
