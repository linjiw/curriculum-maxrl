"""Sanity check: the four arms must implement four distinct samplers.

Outcome-blind.  This inspects only the SCORE FUNCTIONS and their induced
sampling distributions over the fixed task pool -- never a run outcome.

The decisive property for the U64 design is that u_64 peaks at a strictly
harder (lower) pass rate than u_16, which peaks harder than u_2 = p(1-p):

    p*_N = 1 - N^(-1/(N-1))     ->   .5 (N=2), .169 (N=16), .0639 (N=64)
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))

import numpy as np

import run_u64_tournament as u64
from frontier_rl.teacher import FrontierTeacher


def peak(n: int) -> float:
    return 1.0 - n ** (-1.0 / (n - 1))


def main() -> int:
    ok = True
    grid = np.linspace(1e-9, 1 - 1e-9, 2_000_001)

    print("closed-form vs numeric peak of u_N:")
    for n in (2, 16, 64):
        t = FrontierTeacher(8, n, decay=u64.TEACHER_DECAY,
                            floor=u64.TEACHER_FLOOR, gamma=u64.TEACHER_GAMMA,
                            seed=0)
        numeric = grid[int(np.argmax(t.utility(grid)))]
        closed = peak(n)
        good = abs(numeric - closed) < 1e-4
        ok &= good
        print(f"  N={n:3d}  closed={closed:.6f}  numeric={numeric:.6f}  "
              f"{'ok' if good else 'MISMATCH'}")

    if not (peak(64) < peak(16) < peak(2)):
        print("PEAK ORDERING WRONG")
        ok = False
    else:
        print(f"\npeak ordering ok: u64 {peak(64):.4f} < u16 {peak(16):.4f} "
              f"< u2 {peak(2):.4f}  (u64 over-shoots the deployed N=16 peak)")

    print("\ninduced sampling distribution over the 8-task pool")
    print("(uniform posterior; floor and gamma as deployed):")
    dists = {}
    for arm in u64.ARM_ORDER:
        t = u64._U64TournamentTeacher(arm, seed=0)
        d = t.distribution()
        dists[arm] = d
        print(f"  {arm:22s} " + " ".join(f"{x:.4f}" for x in d))

    # u16 and u64 must not be the same sampler
    if np.allclose(dists["u16_shared_h64"], dists["u64_shared_h64"], atol=1e-12):
        print("\nu16 and u64 produced identical distributions -- ARM IS INERT")
        ok = False

    # every arm distinct
    names = list(u64.ARM_ORDER)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if np.allclose(dists[names[i]], dists[names[j]], atol=1e-12):
                print(f"\n{names[i]} and {names[j]} are identical -- BAD")
                ok = False

    # deployed estimator must stay at 16 for every arm
    for arm in u64.ARM_ORDER:
        t = u64._U64TournamentTeacher(arm, seed=0)
        if u64.engine.N_ROLLOUTS != 16:
            print("engine N_ROLLOUTS drifted")
            ok = False
        exp = u64.ARMS[arm]
        if exp is not None and t.n_rollouts != exp:
            print(f"{arm}: teacher exponent {t.n_rollouts} != {exp}")
            ok = False
    print(f"\ndeployed estimator N stays {u64.engine.N_ROLLOUTS} for all arms")

    u64.assert_score_estimator_decoupling()
    print("score/estimator decoupling assertion passed")

    print("\nARMS OK" if ok else "\nARMS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
