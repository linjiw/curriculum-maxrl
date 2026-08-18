"""Activity-matched, transfer-mismatched pair test. See BRANCHING_PREREG.md."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "curriculum_maxrl"))
from branching_pool import make_branching                       # noqa: E402
from estimators import weights_maxrl, weights_rloo, coefficient_activity  # noqa: E402
import run_utility_audit as A                                   # noqa: E402

WARM = int(__import__("os").environ.get("BRANCH_WARM", 400))
H = A.H
N = A.N
U_TOL = 0.05
C_RATIO = 3.0
SEEDS = tuple(range(4001, 4011))


P_TOL = float(__import__("os").environ.get("BRANCH_P_TOL", "0"))  # 0 = original u_N-only matching


def matched_pairs(uN, C, p=None):
    """Pairs formed from u_N, p and C only; no utility consulted.

    P_TOL > 0 additionally requires |p_i - p_j| <= P_TOL. u_N is unimodal, so
    u_N-matching alone pairs tasks on opposite sides of the peak; see the
    2026-08-18 amendment in BRANCHING_PREREG.md."""
    n = len(uN); umax = float(uN.max()); out = []
    for i in range(n):
        for j in range(i + 1, n):
            if uN[i] <= 1e-6 or uN[j] <= 1e-6:
                continue
            if abs(uN[i] - uN[j]) / umax > U_TOL:
                continue
            if P_TOL > 0 and p is not None and abs(p[i] - p[j]) > P_TOL:
                continue
            hi, lo = (i, j) if C[i] >= C[j] else (j, i)
            if C[lo] <= 0 or C[hi] / C[lo] < C_RATIO:
                continue
            out.append((hi, lo, float(C[hi] / C[lo])))
    return out


def run_seed(seed: int, estimator: str) -> dict:
    wfn = {"maxrl": weights_maxrl, "rloo": weights_rloo}[estimator]
    env = make_branching(seed)
    A.warm(env, WARM, wfn)
    p = env.true_pass_rates(); C = env.compounding()
    uN = coefficient_activity(p, N)
    pairs = matched_pairs(uN, C, p)

    need = sorted({t for pr in pairs for t in pr[:2]})
    U = {t: A.continuation_utility(env, t, wfn, seed * 1_000_003 + WARM * 1_009 + t)
         for t in need}
    deltas = [U[hi] - U[lo] for hi, lo, _ in pairs]
    ratios = [r for _, _, r in pairs]

    # frozen secondaries over the full pool
    Uall = np.array([A.continuation_utility(env, t, wfn, seed * 7_919 + WARM * 31 + t)
                     for t in range(env.n_tasks)])
    return {
        "seed": seed, "estimator": estimator, "warm": WARM,
        "n_pairs": len(pairs),
        "mean_delta": float(np.mean(deltas)) if deltas else None,
        "deltas": [float(x) for x in deltas],
        "c_ratios": ratios,
        "rho_uN": A.spearman(uN, Uall),
        "rho_uNC": A.spearman(uN * C, Uall),
        "rho_C": A.spearman(C, Uall),
        "corr_p_C": float(np.corrcoef(p, C)[0, 1]),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--estimator", default="maxrl", choices=["maxrl", "rloo"])
    a = ap.parse_args(argv)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    for s in SEEDS:
        tag = "branch2" if P_TOL > 0 else "branch"
        f = out / f"{tag}-{a.estimator}-s{s}.json"
        if f.exists():
            continue
        r = run_seed(s, a.estimator)
        f.write_text(json.dumps(r, indent=1))
        md = r["mean_delta"]
        print(f"seed {s}  pairs {r['n_pairs']:3d}  mean_delta "
              f"{'n/a' if md is None else f'{md:+.5f}'}  "
              f"rho_uN {r['rho_uN']:+.3f}  rho_uNC {r['rho_uNC']:+.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
