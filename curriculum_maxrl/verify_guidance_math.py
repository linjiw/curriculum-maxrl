"""MC verification of the 2026-08-04 research-guidance identities.

Checks every closed form the guidance doc (research_guidance/2026-08-04)
asks the paper to state, against direct simulation and, where the
skill-chain env provides exact gradients, against the factorization
theorem E[g_hat] = nu_N(p) * (mu_plus - mu_minus).

Estimator conventions verified:
  raw       : 1{K>=1} (1/K) sum r_i S_i                  (mass 1-q^N, T=N)
  full-CV   : raw - (1/N) sum S_i, kept at K=0           (mass 2q-q^N, T=N)
  practical : 1{K>=1} sum (r_i/K - 1/N) S_i              (mass 2(q-q^N), T=N-1)
  RLOO      : (r_i - loo_mean)/N                          (mass 2pq)
  GRPO      : (r_i - mean)/(sample SD)/N                  (mass 2 sqrt((N-1)/N)
                                                           (1/N) E sqrt(K(N-K)))

Usage: python3 verify_guidance_math.py [--trials 200000] [--seed 0]
Writes verify_guidance_math.json next to this file.
"""

from __future__ import annotations

import argparse
import json
import os
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-6


# ---------------------------------------------------------------- closed forms
def q_(p):
    return 1.0 - p


def mass_raw(p, N):
    return 1.0 - q_(p) ** N


def mass_fullcv(p, N):
    return 2.0 * q_(p) - q_(p) ** N


def mass_practical(p, N):
    return 2.0 * (q_(p) - q_(p) ** N)


def nu_practical(p, N):
    return q_(p) - q_(p) ** N


def mass_rloo(p, N):
    return 2.0 * p * q_(p)


def half_mass_grpo_sample_sd(p, N):
    """Deployed convention: sample SD (ddof=1), 1/N trajectory averaging."""
    s = 0.0
    for k in range(1, N):
        s += comb(N, k) * p**k * q_(p) ** (N - k) * np.sqrt(k * (N - k)) / N
    return np.sqrt((N - 1) / N) * s


def grad_weight_raw(p, N):
    """E[g_raw] = w(p) * grad p with w = (1-q^N)/p  (T=N)."""
    return (1.0 - q_(p) ** N) / p


def grad_weight_practical(p, N):
    """E[g_prac] = w(p) * grad p with w = (1-q^{N-1})/p  (T=N-1)."""
    return (1.0 - q_(p) ** (N - 1)) / p


# ---------------------------------------------------------------- MC on masses
def mc_masses(p, N, trials, rng):
    r = (rng.random((trials, N)) < p).astype(np.float64)
    K = r.sum(axis=1)
    live = K >= 1

    # raw
    w_raw = np.zeros_like(r)
    w_raw[live] = r[live] / K[live, None]
    m_raw = np.abs(w_raw).sum(axis=1).mean()

    # full CV: raw coefficients minus 1/N everywhere, kept at K=0
    w_full = w_raw - 1.0 / N
    m_full = np.abs(w_full).sum(axis=1).mean()

    # practical: same coefficients but zeroed at K=0
    w_prac = np.where(live[:, None], w_full, 0.0)
    m_prac = np.abs(w_prac).sum(axis=1).mean()

    # RLOO
    loo = (r.sum(axis=1, keepdims=True) - r) / (N - 1)
    w_rloo = (r - loo) / N
    m_rloo = np.abs(w_rloo).sum(axis=1).mean()

    # GRPO with sample SD, degenerate groups zeroed (constant-reward groups
    # produce zero numerator, so the EPS denominator just leaves zeros)
    sd = r.std(axis=1, ddof=1)
    w_grpo = (r - r.mean(axis=1, keepdims=True)) / (sd[:, None] + EPS) / N
    m_grpo = np.abs(w_grpo).sum(axis=1).mean()

    return dict(raw=m_raw, fullcv=m_full, practical=m_prac,
                rloo=m_rloo, grpo=m_grpo)


# --------------------------------------------------- exact-gradient env checks
def env_gradient_checks(N, trials, rng):
    """On a 2-action softmax 'skill', the score is exact, so we can verify
    the gradient-level identities, not just masses:
      E[g_raw]  = ((1-q^N)/p)      * grad p
      E[g_full] = E[g_raw]                       (control variate, mean zero)
      E[g_prac] = ((1-q^{N-1})/p)  * grad p      (T = N-1)
      E[g_full | K=0] = grad p / q               (nonzero all-fail update)
      factorization: E[g_prac] = nu_N(p) (mu+ - mu-)
    """
    theta = np.array([0.4, 0.0])  # logits; action 0 correct
    ez = np.exp(theta - theta.max())
    probs = ez / ez.sum()
    p = probs[0]

    # score of action a: onehot(a) - probs
    grad_p = np.array([p * (1 - p), -p * probs[1]])  # d p / d theta

    a = (rng.random((trials, N)) >= p).astype(int)  # 0 = correct
    r = (a == 0).astype(np.float64)
    K = r.sum(axis=1)
    live = K >= 1
    onehot = np.eye(2)[a]                      # (trials, N, 2)
    score = onehot - probs[None, None, :]      # exact scores

    def mc_grad(w):
        return np.einsum("tn,tna->a", w, score) / trials

    w_raw = np.zeros_like(r)
    w_raw[live] = r[live] / K[live, None]
    w_full = w_raw - 1.0 / N
    w_prac = np.where(live[:, None], w_full, 0.0)

    g_raw = mc_grad(w_raw)
    g_full = mc_grad(w_full)
    g_prac = mc_grad(w_prac)

    dead = ~live
    g_full_dead = (np.einsum("tn,tna->a", w_full[dead], score[dead])
                   / max(dead.sum(), 1))

    # factorization pieces
    mu_plus = score[r == 1].reshape(-1, 2).mean(axis=0)
    mu_minus = score[r == 0].reshape(-1, 2).mean(axis=0)
    nu = nu_practical(p, N)

    q = 1 - p
    return {
        "p": float(p),
        "N": N,
        "raw": {"mc": g_raw.tolist(),
                "exact": (grad_weight_raw(p, N) * grad_p).tolist()},
        "fullcv": {"mc": g_full.tolist(),
                   "exact": (grad_weight_raw(p, N) * grad_p).tolist()},
        "practical": {"mc": g_prac.tolist(),
                      "exact": (grad_weight_practical(p, N) * grad_p).tolist()},
        "fullcv_allfail": {"mc": g_full_dead.tolist(),
                           "exact": (grad_p / q).tolist(),
                           "n_dead_groups": int(dead.sum())},
        "factorization": {"nu_times_contrast":
                          (nu * (mu_plus - mu_minus)).tolist(),
                          "mc_practical": g_prac.tolist()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    report = {"trials": args.trials, "mass_checks": [], "tail_ratios": {},
              "gradient_checks": None}
    worst = 0.0

    for N in (4, 16):
        for p in (0.02, 0.1, 0.3, 0.7, 0.95):
            mc = mc_masses(p, N, args.trials, rng)
            exact = dict(raw=mass_raw(p, N), fullcv=mass_fullcv(p, N),
                         practical=mass_practical(p, N),
                         rloo=mass_rloo(p, N),
                         grpo=2 * half_mass_grpo_sample_sd(p, N))
            errs = {k: abs(mc[k] - exact[k]) for k in mc}
            worst = max(worst, max(errs.values()))
            report["mass_checks"].append(
                {"N": N, "p": p, "mc": mc, "exact": exact, "abs_err": errs})

    # deployed-convention tail ratios at N=16 (guidance corrects the paper's
    # population-SD sqrt(N-1) to sqrt(N) and (N-1)/sqrt(N) for sample SD)
    N = 16
    p_lo, p_hi = 1e-4, 1 - 1e-4
    ratio_hard = (nu_practical(p_lo, N)
                  / half_mass_grpo_sample_sd(p_lo, N))
    ratio_easy = (half_mass_grpo_sample_sd(p_hi, N)
                  / nu_practical(p_hi, N))
    report["tail_ratios"] = {
        "N": N,
        "maxrl_over_grpo_p_to_0": {"mc_limit": float(ratio_hard),
                                   "predicted_sqrt_N": float(np.sqrt(N))},
        "grpo_over_maxrl_p_to_1": {"mc_limit": float(ratio_easy),
                                   "predicted_Nm1_over_sqrt_N":
                                   float((N - 1) / np.sqrt(N))},
        "rloo_hard_tail_ratio": {
            "maxrl_over_rloo_p_to_0":
            float(nu_practical(p_lo, N) / (p_lo * (1 - p_lo))),
            "predicted_N_minus_1": N - 1},
        "mastered_tail_first_order": {
            "maxrl_nu_over_q": float(nu_practical(p_hi, N) / (1 - p_hi)),
            "rloo_halfmass_over_q": float(p_hi * (1 - p_hi) / (1 - p_hi)),
            "note": "MaxRL and RLOO both decay ~q to first order; GRPO "
                    "retains the larger mass under either SD convention"},
    }

    report["gradient_checks"] = env_gradient_checks(8, args.trials, rng)

    # peak identity p* = 1 - N^{-1/(N-1)}
    report["peak_check"] = []
    for N in (2, 4, 8, 16, 32, 64):
        ps = 1 - N ** (-1 / (N - 1))
        grid = np.linspace(1e-6, 1 - 1e-6, 200001)
        ps_num = grid[np.argmax(nu_practical(grid, N))]
        report["peak_check"].append(
            {"N": N, "p_star_closed": float(ps), "p_star_grid": float(ps_num),
             "nu_at_peak": float(nu_practical(ps, N)),
             "closed_peak_value": float((1 - 1 / N) * N ** (-1 / (N - 1)))})

    report["worst_mass_abs_err"] = worst
    out = os.path.join(HERE, "verify_guidance_math.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)

    print(f"worst mass abs err over all cells: {worst:.2e} "
          f"(MC noise scale ~{3/np.sqrt(args.trials):.1e})")
    gc = report["gradient_checks"]
    for k in ("raw", "fullcv", "practical", "fullcv_allfail"):
        mc = np.array(gc[k]["mc"])
        ex = np.array(gc[k]["exact"])
        print(f"{k:>16}: mc={mc.round(5).tolist()} exact={ex.round(5).tolist()} "
              f"maxerr={np.abs(mc-ex).max():.2e}")
    fa = gc["factorization"]
    print(f"   factorization: nu*(mu+-mu-)="
          f"{np.array(fa['nu_times_contrast']).round(5).tolist()} vs "
          f"mc={np.array(fa['mc_practical']).round(5).tolist()}")
    tr = report["tail_ratios"]
    print(f"tail ratios (sample-SD GRPO, N=16): hard "
          f"{tr['maxrl_over_grpo_p_to_0']['mc_limit']:.3f} vs sqrt(N)="
          f"{np.sqrt(16):.3f}; easy "
          f"{tr['grpo_over_maxrl_p_to_1']['mc_limit']:.3f} vs (N-1)/sqrt(N)="
          f"{15/4:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
