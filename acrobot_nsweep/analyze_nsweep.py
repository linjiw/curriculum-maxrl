"""Analyzer for the frozen Acrobot score-exponent dose-response.

Implements exactly ACROBOT_NSWEEP_PREREG.md sections 3-5:

    P1 = u16 - u128   deployed peak beats an 8x-harder peak (over-shoot)
    P2 = u16 - u2     deployed peak beats the learnability slice (under-shoot)

Holm over {P1, P2}; supported iff mean >= +0.01 and adjusted exact two-sided
sign-flip p <= .05.  H_peak requires BOTH, because a maximum at 16 needs the
curve to fall away on both sides.

Shape statistics (frozen, descriptive, no decision attached): the arm-mean
curve, its argmax, the Spearman rank correlation between exponent and arm mean,
and the neighbour contrasts u16-u8 and u16-u32.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

METRIC = "auc_mean_pass_by_transitions"
METRIC_NAME = "target_uniform_transition_auc"
SESOI = 0.01
ALPHA = 0.05
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20260815

EXPONENTS = (2, 4, 8, 16, 32, 64, 128)
SCORED_ARMS = tuple(f"u{n}" for n in EXPONENTS)
ARMS = ("uniform",) + SCORED_ARMS
SEEDS = tuple(range(20_000, 20_020))


class AnalysisError(RuntimeError):
    pass


def exact_sign_flip_p(d: np.ndarray) -> float:
    if d.size > 24:
        raise AnalysisError(f"exact sign-flip refuses n={d.size}")
    sums = np.zeros(1, dtype=float)
    for v in d:
        sums = np.concatenate((sums - v, sums + v))
    obs = abs(float(d.sum()))
    tol = max(1e-15, 1e-14 * max(1.0, obs))
    return float(np.count_nonzero(np.abs(sums) >= obs - tol) / sums.size)


def bootstrap_ci(d: np.ndarray) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, d.size, size=(BOOTSTRAP_RESAMPLES, d.size))
    m = d[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def holm(p: dict[str, float]) -> dict[str, float]:
    order = sorted(p, key=lambda k: p[k])
    m, out, run = len(order), {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def load(d: Path) -> dict:
    if not d.is_dir():
        raise AnalysisError(f"missing {d}")
    if list(d.glob("*.partial")):
        raise AnalysisError("partial files present")
    cells = {}
    for p in sorted(d.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("mode") != "confirmatory":
            raise AnalysisError(f"{p.name}: mode {r.get('mode')!r}")
        key = (r["arm"], int(r["logical_seed"]))
        if key in cells:
            raise AnalysisError(f"duplicate {key}")
        cells[key] = r
    missing = [(a, s) for a in ARMS for s in SEEDS if (a, s) not in cells]
    if missing:
        raise AnalysisError(
            f"incomplete: {len(missing)}/{len(ARMS)*len(SEEDS)} missing, "
            f"e.g. {missing[:5]}")
    return cells


def validate(cells: dict) -> dict:
    locks, cpus, hosts = set(), set(), set()
    for (arm, seed), r in cells.items():
        run = r["run"]
        if not run.get("accounting_valid") or not run.get("numeric_valid"):
            raise AnalysisError(f"{arm} {seed}: invalid run flags")
        if r["deployed_n_rollouts"] != 16:
            raise AnalysisError(f"{arm} {seed}: deployed N != 16")
        if r["transition_budget"] != 2_000_000:
            raise AnalysisError(f"{arm} {seed}: budget != 2e6")
        if r["score_exponent"] is not None and r["arm"] != f"u{r['score_exponent']}":
            raise AnalysisError(f"{arm} {seed}: arm/exponent disagree")
        locks.add(r["provenance"].get("lock_sha256"))
        cpus.add(r["provenance"].get("cpu_model"))
        hosts.add(r["provenance"].get("hostname"))
    if len(locks) != 1 or None in locks:
        raise AnalysisError(f"not one lock digest: {locks}")
    per_seed = {}
    for (arm, seed), r in cells.items():
        per_seed.setdefault(seed, set()).add(r["provenance"].get("cpu_model"))
    split = {s: sorted(c) for s, c in per_seed.items() if len(c) > 1}
    if split:
        raise AnalysisError(f"within-seed arms on different CPUs: {split}")
    return {"lock_sha256": locks.pop(),
            "distinct_cpu_models": sorted(c for c in cpus if c),
            "distinct_hosts": sorted(h for h in hosts if h)}


def vals(cells: dict, arm: str) -> np.ndarray:
    return np.array([cells[(arm, s)]["run"][METRIC] for s in SEEDS], dtype=float)


def contrast(cells: dict, a: str, b: str) -> dict:
    d = vals(cells, a) - vals(cells, b)
    return {
        "estimand": f"{a} minus {b}",
        "n_paired_seeds": int(d.size),
        "mean_paired_difference": float(d.mean()),
        "sample_std": float(d.std(ddof=1)),
        "positive_pairs": int(np.count_nonzero(d > 0)),
        "paired_bootstrap_ci95": bootstrap_ci(d),
        "exact_two_sided_sign_flip_p": exact_sign_flip_p(d),
        "paired_differences": [float(x) for x in d],
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
    prov = validate(cells)

    P1 = contrast(cells, "u16", "u128")
    P2 = contrast(cells, "u16", "u2")
    raw = {"P1_u16_minus_u128": P1["exact_two_sided_sign_flip_p"],
           "P2_u16_minus_u2": P2["exact_two_sided_sign_flip_p"]}
    adj = holm(raw)
    for k, r in (("P1_u16_minus_u128", P1), ("P2_u16_minus_u2", P2)):
        r["raw_p"] = raw[k]
        r["holm_adjusted_p"] = adj[k]
        r["meets_sesoi"] = bool(r["mean_paired_difference"] >= SESOI)
        r["significant"] = bool(adj[k] <= ALPHA)
        r["supported"] = bool(r["meets_sesoi"] and r["significant"])

    curve = {f"u{n}": float(vals(cells, f"u{n}").mean()) for n in EXPONENTS}
    means = np.array([curve[f"u{n}"] for n in EXPONENTS])
    argmax_n = int(EXPONENTS[int(np.argmax(means))])
    rho = spearman(np.array(EXPONENTS, dtype=float), means)

    if P1["supported"] and P2["supported"]:
        verdict = ("H_peak_supported" if argmax_n == 16
                   else "interior_optimum_not_at_deployed_N")
    elif not P1["supported"] and P2["supported"]:
        verdict = "H_hard_harder_is_better"
    else:
        verdict = "u16_does_not_beat_learnability_slice"

    report = {
        "schema": "curriculum-maxrl/acrobot-nsweep-analysis/v1",
        "prereg": "acrobot_nsweep/ACROBOT_NSWEEP_PREREG.md",
        "metric": METRIC_NAME,
        "sesoi": SESOI,
        "multiplicity": "Holm over {P1, P2}",
        "provenance": prov,
        "primary_P1_u16_minus_u128": P1,
        "primary_P2_u16_minus_u2": P2,
        "shape": {
            "arm_mean_curve": curve,
            "uniform_mean": float(vals(cells, "uniform").mean()),
            "argmax_exponent": argmax_n,
            "spearman_exponent_vs_mean": rho,
            "neighbour_u16_minus_u8": contrast(cells, "u16", "u8"),
            "neighbour_u16_minus_u32": contrast(cells, "u16", "u32"),
            "all_vs_u16": {f"u{n}": contrast(cells, f"u{n}", "u16")
                           for n in EXPONENTS if n != 16},
        },
        "verdict": verdict,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")

    print(f"cpus   : {prov['distinct_cpu_models']}")
    print("\nscore-exponent dose-response (target-uniform transition AUC):")
    print(f"  {'uniform':>8s}  {report['shape']['uniform_mean']:.5f}")
    for n in EXPONENTS:
        star = "  <-- deployed N" if n == 16 else ""
        bar = "#" * int(round((curve[f'u{n}'] - min(means)) /
                              max(1e-12, (max(means) - min(means))) * 40))
        print(f"  {'u'+str(n):>8s}  {curve[f'u{n}']:.5f}  {bar}{star}")
    print(f"\nargmax  : u{argmax_n}   spearman(exponent, mean) = {rho:+.3f}")
    for lbl, r in (("P1 u16-u128", P1), ("P2 u16-u2  ", P2)):
        print(f"{lbl}: {r['mean_paired_difference']:+.5f} "
              f"CI [{r['paired_bootstrap_ci95'][0]:+.5f},"
              f"{r['paired_bootstrap_ci95'][1]:+.5f}] "
              f"p {r['raw_p']:.6f} holm {r['holm_adjusted_p']:.6f} "
              f"{r['positive_pairs']}/20 "
              f"{'SUPPORTED' if r['supported'] else 'not supported'}")
    print(f"\nVERDICT: {verdict}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
