"""Analyzer for the frozen Acrobot U64 campaign.

Implements exactly the decision rule in ACROBOT_U64_PREREG.md section 3-4 and
nothing else.  It refuses to produce any inferential quantity from an
incomplete matrix, and it never reads a partial file.

Primary metric: paired target-uniform normalized transition-AUC, which the V2
analyzer defines as the engine field `auc_mean_pass_by_transitions`
(analyze_acrobot_curriculum_tournament.py:952).

    A = u16 - u64   the decisive test of peak-location specificity
    B = u16 - p1mp  cross-platform replication of the V2 primary (+.04803)

Family {A, B}, Holm at familywise .05.  Supported iff mean >= +0.01 (SESOI)
and the exact two-sided sign-flip p, after Holm adjustment, is <= .05.
With 20 pairs the sign-flip enumerates 2**20 = 1,048,576 assignments exactly.
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

ARMS = ("uniform_shared_h64", "p1mp_shared_h64", "u16_shared_h64",
        "u64_shared_h64")
CONFIRMATORY_SEEDS = tuple(range(20_000, 20_020))

V2_PRIMARY = {"mean": 0.04803, "ci": [0.02094, 0.07385], "p": 0.003361}


class AnalysisError(RuntimeError):
    pass


def exact_sign_flip_p(diffs: np.ndarray) -> float:
    """Exact two-sided randomization p for the paired mean (2**n assignments)."""
    n = diffs.size
    if n > 24:
        raise AnalysisError(f"exact sign-flip refuses n={n} (>24)")
    sums = np.zeros(1, dtype=float)
    for v in diffs:
        sums = np.concatenate((sums - v, sums + v))
    observed = abs(float(diffs.sum()))
    tol = max(1e-15, 1e-14 * max(1.0, observed))
    return float(np.count_nonzero(np.abs(sums) >= observed - tol) / sums.size)


def bootstrap_ci(diffs: np.ndarray) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, diffs.size, size=(BOOTSTRAP_RESAMPLES, diffs.size))
    means = diffs[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def holm(pvals: dict[str, float]) -> dict[str, float]:
    order = sorted(pvals, key=lambda k: pvals[k])
    m = len(order)
    out, running = {}, 0.0
    for i, key in enumerate(order):
        adj = min(1.0, (m - i) * pvals[key])
        running = max(running, adj)
        out[key] = running
    return out


def load(results_dir: Path) -> dict[tuple[str, int], dict]:
    if not results_dir.is_dir():
        raise AnalysisError(f"missing results dir {results_dir}")
    partials = sorted(results_dir.glob("*.partial"))
    if partials:
        raise AnalysisError(f"partial files present: {[p.name for p in partials]}")

    cells: dict[tuple[str, int], dict] = {}
    for path in sorted(results_dir.glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        arm, seed = rec["arm"], int(rec["logical_seed"])
        if rec.get("mode") != "confirmatory":
            raise AnalysisError(f"{path.name}: mode is {rec.get('mode')!r}")
        if (arm, seed) in cells:
            raise AnalysisError(f"duplicate cell {arm} {seed}")
        cells[(arm, seed)] = rec

    missing = [(a, s) for a in ARMS for s in CONFIRMATORY_SEEDS
               if (a, s) not in cells]
    if missing:
        raise AnalysisError(
            f"incomplete matrix: {len(missing)} of {len(ARMS)*len(CONFIRMATORY_SEEDS)} "
            f"cells missing, e.g. {missing[:5]}")
    extra = [k for k in cells if k[0] not in ARMS or k[1] not in CONFIRMATORY_SEEDS]
    if extra:
        raise AnalysisError(f"unregistered cells present: {extra[:5]}")
    return cells


def validate(cells: dict) -> dict:
    locks, runtimes, cpus, hosts = set(), set(), set(), set()
    for (arm, seed), rec in cells.items():
        run = rec["run"]
        if not run.get("accounting_valid", False):
            raise AnalysisError(f"{arm} {seed}: accounting_valid is false")
        if not run.get("numeric_valid", False):
            raise AnalysisError(f"{arm} {seed}: numeric_valid is false")
        if rec["deployed_n_rollouts"] != 16:
            raise AnalysisError(f"{arm} {seed}: deployed N is not 16")
        if rec["transition_budget"] != 2_000_000:
            raise AnalysisError(f"{arm} {seed}: budget is not 2,000,000")
        prov = rec["provenance"]
        locks.add(prov.get("lock_sha256"))
        runtimes.add(json.dumps(
            {k: prov["runtime"][k] for k in
             ("python_implementation", "python", "numpy", "gymnasium", "machine")},
            sort_keys=True))
        cpus.add(prov.get("cpu_model"))
        hosts.add(prov.get("hostname"))

    if len(locks) != 1 or None in locks:
        raise AnalysisError(f"campaign does not share one lock digest: {locks}")
    if len(runtimes) != 1:
        raise AnalysisError(f"campaign spans multiple pinned runtimes: {runtimes}")

    # Node heterogeneity is permitted (see the runtime-scope amendment) but must
    # be reported, because a contrast confounded with node family is not clean.
    per_seed_cpu = {}
    for (arm, seed), rec in cells.items():
        per_seed_cpu.setdefault(seed, set()).add(rec["provenance"].get("cpu_model"))
    split = {s: sorted(c) for s, c in per_seed_cpu.items() if len(c) > 1}
    if split:
        raise AnalysisError(
            "within-seed arms ran on different CPU models, so pairing is not "
            f"exact: {split}")

    return {
        "lock_sha256": locks.pop(),
        "pinned_runtime": json.loads(runtimes.pop()),
        "distinct_cpu_models": sorted(c for c in cpus if c),
        "distinct_hosts": sorted(h for h in hosts if h),
        "within_seed_pairing_exact": True,
    }


def contrast(cells: dict, a: str, b: str) -> dict:
    diffs = np.array([cells[(a, s)]["run"][METRIC] - cells[(b, s)]["run"][METRIC]
                      for s in CONFIRMATORY_SEEDS], dtype=float)
    if not np.all(np.isfinite(diffs)):
        raise AnalysisError(f"non-finite differences for {a}-{b}")
    return {
        "estimand": f"{a} minus {b}",
        "metric": METRIC_NAME,
        "n_paired_seeds": int(diffs.size),
        "arm_means": {a: float(np.mean([cells[(a, s)]["run"][METRIC]
                                        for s in CONFIRMATORY_SEEDS])),
                      b: float(np.mean([cells[(b, s)]["run"][METRIC]
                                        for s in CONFIRMATORY_SEEDS]))},
        "mean_paired_difference": float(diffs.mean()),
        "sample_std": float(diffs.std(ddof=1)),
        "positive_pairs": int(np.count_nonzero(diffs > 0)),
        "paired_differences": [float(x) for x in diffs],
        "paired_bootstrap_ci95": bootstrap_ci(diffs),
        "exact_two_sided_sign_flip_p": exact_sign_flip_p(diffs),
        "test_assignments": int(2 ** diffs.size),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", help="directory of confirmatory arm JSONs")
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    out = Path(args.output)
    if out.exists():
        raise SystemExit(
            f"{out} exists; the prereg allows the analyzer to run exactly once")

    cells = load(Path(args.results))
    provenance = validate(cells)

    A = contrast(cells, "u16_shared_h64", "u64_shared_h64")
    B = contrast(cells, "u16_shared_h64", "p1mp_shared_h64")
    raw = {"A_u16_minus_u64": A["exact_two_sided_sign_flip_p"],
           "B_u16_minus_p1mp": B["exact_two_sided_sign_flip_p"]}
    adj = holm(raw)

    for key, res in (("A_u16_minus_u64", A), ("B_u16_minus_p1mp", B)):
        res["raw_p"] = raw[key]
        res["holm_adjusted_p"] = adj[key]
        res["meets_sesoi"] = bool(res["mean_paired_difference"] >= SESOI)
        res["significant"] = bool(adj[key] <= ALPHA)
        res["supported"] = bool(res["meets_sesoi"] and res["significant"])

    if A["supported"] and B["supported"]:
        verdict = "H_peak_supported"
        reading = ("The deployed-N peak location matters: u_16 beats both the "
                   "softer u_2 peak and the harder u_64 peak. The peak-hardness "
                   "confound is broken and the V2 primary replicates.")
    elif not A["supported"] and B["supported"]:
        verdict = "H_peak_not_supported_harder_beats_softer"
        reading = ("The replicated finding is 'harder-peaked beats p(1-p)'. "
                   "Deployed-N peak-location specificity is NOT supported and "
                   "must not be claimed; report this with equal prominence.")
    elif A["supported"] and not B["supported"]:
        verdict = "incoherent_inconclusive"
        reading = ("u_16 beats u_64 but not p(1-p). Incoherent under the frozen "
                   "interpretation table; report as inconclusive, claim nothing.")
    else:
        verdict = "v2_did_not_replicate"
        reading = ("The V2 primary did not replicate on this platform. Report "
                   "that first; A is uninterpretable.")

    report = {
        "schema": "curriculum-maxrl/acrobot-u64-analysis/v1",
        "prereg": "acrobot_u64/ACROBOT_U64_PREREG.md",
        "metric": METRIC_NAME,
        "engine_field": METRIC,
        "sesoi": SESOI,
        "familywise_alpha": ALPHA,
        "multiplicity": "Holm over {A, B}",
        "provenance": provenance,
        "primary_A_u16_minus_u64": A,
        "primary_B_u16_minus_p1mp": B,
        "secondary_descriptive": {
            "u64_minus_p1mp": contrast(cells, "u64_shared_h64", "p1mp_shared_h64"),
            "u16_minus_uniform": contrast(cells, "u16_shared_h64", "uniform_shared_h64"),
            "u64_minus_uniform": contrast(cells, "u64_shared_h64", "uniform_shared_h64"),
            "p1mp_minus_uniform": contrast(cells, "p1mp_shared_h64", "uniform_shared_h64"),
        },
        "v2_frozen_primary_for_reference": V2_PRIMARY,
        "verdict": verdict,
        "reading": reading,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")

    print(f"metric      : {METRIC_NAME}")
    print(f"cpu models  : {provenance['distinct_cpu_models']}")
    print(f"hosts       : {len(provenance['distinct_hosts'])} distinct")
    print()
    for label, res in (("A  u16 - u64 ", A), ("B  u16 - p1mp", B)):
        print(f"{label}: mean {res['mean_paired_difference']:+.5f}  "
              f"CI [{res['paired_bootstrap_ci95'][0]:+.5f}, "
              f"{res['paired_bootstrap_ci95'][1]:+.5f}]  "
              f"exact p {res['raw_p']:.6f}  holm {res['holm_adjusted_p']:.6f}  "
              f"{res['positive_pairs']}/20 pairs  "
              f"{'SUPPORTED' if res['supported'] else 'not supported'}")
    print()
    print(f"V2 reference: B was {V2_PRIMARY['mean']:+.5f} "
          f"CI [{V2_PRIMARY['ci'][0]:+.5f}, {V2_PRIMARY['ci'][1]:+.5f}] "
          f"p {V2_PRIMARY['p']:.6f}")
    print()
    print(f"VERDICT: {verdict}")
    print(reading)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
