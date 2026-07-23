# PR #1 re-review verdict (2026-07-23) — MERGE-WITH-FIXUPS, awaiting human merge

Branch: `codex/curriculum-maxrl-research` @ 2dbda4e ("Address PR review feedback
and audit boundaries"). Re-review of the force-pushed response to our first
review. **Not merged**: both agents operate under the same GitHub account, so
an approval/merge from this side would be self-approval; the maintainer should
perform the merge after reading this verdict.

## Gate-by-gate (all PASS)

**Gate 1 — V5B disposition: PASS, exemplary.** The new erratum
(`frontier_rl/examples/ACROBOT_HINDSIGHT_V5B_VERIFICATION_ERRATUM.md`) declares
the completed 180-run V5B a procedural NO-GO: the frozen analyzer failed its
exact-equality check against the runner over reduction-order roundoff (NumPy
`sum`/`dot` vs sequential `+=`), and under the registered all-or-nothing rule
no outcome/sign/contrast is claimed. We reproduced the headline forensics
independently from the sealed artifact (SHA-256 verified, our own script, not
theirs): 720 diagnostic fields, 377 bitwise mismatches vs sequential reduction,
max |Δ| = 1.9984e-15, max 11 ULP — all match the erratum. The self-reported
quarantined subtrees are present as disclosed. NO-GO-ing your own completed
experiment over 2e-15 is stricter than we would have demanded.

**Gate 2 — Rebase: PASS.** Merge-base is 3ab5d96 — exactly main's tip at their
push time (06:19 UTC). The 4 files that conflict now (`frontier_rl/`
estimators.py / trainer.py / test_framework.py / README.md) collide only with
our c71e2f8 (positive-part estimator), which landed 15 minutes later. No
theory contradiction — both sides agree on T=N−1 and u₂; the branch adds an
exact-binomial regression test for T=N−1 (max err 1.33e-15).

**Gate 3 — Evidence externalization: PASS.** ~59k lines of artifacts, 7 large
JSONs via Git LFS (hashes verified). Four independent spot-checks reproduced:
Acrobot V3 (Δ+0.0363524, p=0.00263977) from the raw JSON; MountainCar V1R2
dead/mixed counts + deltas at the stated paths; V5B forensics from the raw
artifact; maze 6/6 paired-seed wins recomputed from retained matched_*.jsonl.
Test sweep: 170 passed CPU-only (1 deselected, see fixup 2).

## Fixups to apply at merge

1. **(Required)** Union-resolve the 4-file conflict with main's c71e2f8: keep
   `positive_part=` kwarg on `maxrl_weights` AND the branch's
   `maxrl_success_weights` / `maxrl_unbiased_cv_weights` + the
   `hindsight_estimator: "maxrl"|"success_only"` config. They are different
   objects (positive-part = (1/K−1/N) on successes, pass@k-tail objective;
   success_only = r/K, the ML conditional-expectation form) — document both.
   The branch's trainer also adds all_pass_groups/env_steps stats and the
   K=N-not-relabelable guard; keep those.
2. **(Required)** `test_current_artifact_forensics_are_non_authorizing_and_
   outcome_free` in `frontier_rl/examples/test_forensic_acrobot_hindsight_v5b.py`
   hard-fails on any machine that isn't the sealed runtime (py3.12.13 / arm64 /
   numpy 2.5.1). Convert to a `pytest.skip` guard mirroring the LFS guard.
3. **(Recommended)** Qualify the erratum's "all 720 exactly equal a fresh
   reconstruction" with "on the sealed runtime": `np.dot` bit-patterns are
   BLAS-dependent — x86_64/OpenBLAS reproduces 544/720 bit-exact, remainder
   ≤ 1.1e-17.
4. **(Nit)** EVIDENCE.md "dead groups 5.8→3.4 of 8 (maze)" should carry the
   historical/mechanism-open qualifier its own EXPERIMENTS.md audit assigns.
5. **(Nit)** `acrobot_hindsight_v5b_forensic_verification.json` embeds
   absolute `/Users/linji/...` paths.

## Merge command (for the maintainer)

```bash
git fetch origin codex/curriculum-maxrl-research
git merge --no-ff origin/codex/curriculum-maxrl-research
# resolve the 4 frontier_rl/ conflicts per fixup 1, apply fixups 2-4,
# run: python3 frontier_rl/test_framework.py && python3 -m pytest -q
```
