"""Independent re-derivation of the three load-bearing claims in
RESEARCH_NOTE_PLR_CONTROL_PORT_2026-08-19.md.

Run:  python3 control_port/verify_note_claims.py
Needs only numpy.  No GPU, no jax, ~15 s.
"""
import numpy as np
from math import comb

OK = lambda b: "PASS" if b else "FAIL"


def claim_1_shared_vs_resampled():
    """The granularity gap exists ONLY when the atomic instance is SHARED by the
    whole group.  Resample per member and K|z ~ Binomial(N, p_bar) exactly, so
    the count law collapses onto the plug-in and arms 2/3 are the same estimator.
    """
    print("\n[1] shared-instance is necessary for a nonzero granularity gap")
    N = 8
    ps = np.array([1.0]*8 + [0.0]*8)          # Fig. 1 Level B
    pbar = ps.mean()
    plugin = 1 - (1-pbar)**N - pbar           # u_N(p_bar)

    pr_k0_shared    = np.mean((1-ps)**N)      # one instance per group
    pr_k0_resampled = (1-pbar)**N             # fresh instance per member
    A_shared    = (1-pr_k0_shared)    - pbar
    A_resampled = (1-pr_k0_resampled) - pbar

    print(f"    plug-in u_8(p_bar)      = {plugin:.6f}")
    print(f"    A (shared instance)     = {A_shared:.6f}   gap = {plugin-A_shared:+.6f}")
    print(f"    A (resampled per member)= {A_resampled:.6f}   gap = {plugin-A_resampled:+.6f}")
    print(f"    -> shared gap is the corollary 2[Pr(K=0)-(1-p)^N]/2 = {pr_k0_shared-pr_k0_resampled:.6f}")
    print(f"    {OK(abs(A_shared) < 1e-12)}: shared-instance Level B has EXACTLY zero activity")
    print(f"    {OK(abs(A_resampled-plugin) < 1e-12)}: resampled-instance activity EQUALS the plug-in")

    # Monte-Carlo confirmation
    rng = np.random.default_rng(0)
    T = 200_000
    idx = rng.integers(len(ps), size=T)
    K_shared = rng.binomial(N, ps[idx])
    K_res = rng.binomial(1, ps[rng.integers(len(ps), size=(T, N))]).sum(1)
    mc = lambda K: (K > 0).mean() - K.mean()/N
    print(f"    MC check: shared={mc(K_shared):+.4f} (theory {A_shared:+.4f}), "
          f"resampled={mc(K_res):+.4f} (theory {A_resampled:+.4f})")


def claim_2_sfl_is_rloo():
    """SFL's learnability p(1-p) IS the realized RLOO count-law mass, times the
    constant (N-1)/(2N).  Ranking is scale-invariant, so SFL (NeurIPS 2024) is
    already the count-law curriculum for the RLOO estimator.
    """
    print("\n[2] SFL learnability == M_RLOO(k) x (N-1)/(2N), exactly, at every k")
    allok = True
    for N in (4, 8, 10, 16, 32):
        k = np.arange(N+1)
        sfl = (k/N)*(1-k/N)                       # sfl.py:93-94
        rloo = 2.0*k*(N-k)/(N*(N-1))              # M_RLOO(k)
        nz = rloo > 0
        ratio = sfl[nz]/rloo[nz]
        good = np.allclose(ratio, (N-1)/(2*N), rtol=0, atol=1e-15)
        allok &= good
        print(f"    N={N:<3} ratio={ratio[0]:.15f}  (N-1)/(2N)={(N-1)/(2*N):.15f}  {OK(good)}")
    print(f"    {OK(allok)}: identity holds at every N tested")

    N = 8
    k = np.arange(N+1)
    maxrl = 2.0*(1-k/N)*(k > 0)
    rloo = 2.0*k*(N-k)/(N*(N-1))
    print(f"    but the MaxRL mass is a DIFFERENT shape: argmax k = {maxrl.argmax()} "
          f"vs RLOO/SFL argmax k = {rloo.argmax()}  -> a genuine rival ranking, not a rescaling")


def claim_3_normaliser():
    """The count-law activity's true max is 2(1-1/N) at a law concentrated on
    K=1, NOT the Binomial-only bound 2*max_p u_N(p).  Gating without normalising
    by the right constant corrupts the PLR insertion threshold.
    """
    print("\n[3] the gate normaliser is 2(1-1/N), not 2*max_p u_N(p)")
    for N in (8, 16, 32):
        p = np.linspace(0, 1, 2_000_001)
        u = 1-(1-p)**N-p
        binom_max = 2*u.max()
        true_max = 2*(1-1/N)
        print(f"    N={N:<3} 2*max_p u_N(p)={binom_max:.6f} (p*={p[u.argmax()]:.4f}, ln N/N={np.log(N)/N:.4f})"
              f"   max_k M_MaxRL={true_max:.6f}   ratio={true_max/binom_max:.4f}")
    print("    -> A_N = 2*u_N under conditional i.i.d.; the count law reaches strictly higher")


if __name__ == "__main__":
    claim_1_shared_vs_resampled()
    claim_2_sfl_is_rloo()
    claim_3_normaliser()
    print("\ndone.")
