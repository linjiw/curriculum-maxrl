# Curriculum-MaxRL external review notes

*Review snapshot updated 2026-08-09. This is a routing document, not a result
artifact. When it conflicts with a hash-locked protocol, lock, artifact, or
independent verification report, the locked evidence chain controls.*

## What this pull request claims

The central mathematical claim is exact under its stated estimator convention:
for practical centered MaxRL with complete groups of `N` Bernoulli rollouts and
the all-fail group dropped,

```text
E[sum_i |w_i|] = 2 [1 - (1-p)^N - p]
                 = 2 [pass@N(p) - pass@1(p)].
```

This identity motivates an adaptive task-sampling score. It does **not** prove
that the resulting teacher maximizes gradient norm, optimization progress, or
performance in every environment.

The strongest clean Acrobot result is the fresh fixed-pool V2 tournament. In
one eight-threshold `Acrobot-v1` family with a shared H64 practical-MaxRL actor
at `N=16`, `u_16` beat `p(1-p)` by `+.0480336884` target-uniform
transition-AUC [95% bootstrap CI `.0209366676,.0738485654`], exact sign-flip
`p=.0033607483`, 15/20 positive pairs; the Holm-controlled `u_16-uniform`
secondary was also supported. This is a local score-shape result, not a full
named-method, transfer, or general neural-control result. V3's older positive
contrast is now historical/descriptive because a later audit found neighboring
cross-domain RNG-root reuse.

The separately frozen paid-probe attachment completed all 320/320 confirmation
runs, but its registered `u_16-ProCuRL` fixed-paid-AUC mean was only `+.004894`
(`t(79)=1.9773`, `p=.05149`), below the `.02` SESOI and unsupported. Probe
fractions were about `.932`; ordinary-uniform AUC was `.65149`, versus
`.33771` ProCuRL-env, `.33942` probe-sham, and `.34261` `u_16`. The defensible
interpretation is probe-cost domination at this one frozen actor-only cadence,
not inferiority of full PPO ProCuRL or a claim about cheaper probing.

The older ten-seed MountainCar result is a tile-coded local-mechanism study.
Its positive concentrated-teacher and hindsight contrasts should not be
conflated with the new neural MountainCar V1R2 study described below.

## Current studies and decision boundaries

| study | status | permitted interpretation |
|---|---|---|
| Acrobot fixed-pool tournament V2 | 9/9 development and 60/60 confirmation complete; registered primary supported | one-pool score-shape evidence under practical MaxRL; not a full ProCuRL/SFL/PLR/PAIRED/ACCEL comparison |
| Acrobot paid-probe selection attachment | 12/12 development and 320/320 confirmation complete; registered primary unsupported | probe-cost domination at the frozen actor-only cadence; no full-PPO ProCuRL inferiority claim |
| Acrobot V4A | stopped after one predeclared feasibility gate failed | valid failed calibration; no V4 hindsight effect |
| Acrobot V5A | 27/27 complete; all technical/natural-relevance gates passed; fresh `U*=250` | optimizer-matched factorial is feasible and V5B was authorized; no efficacy result |
| Acrobot V5B | 180/180 complete; raw integrity passed; frozen analyzer exact-reconstruction failure | procedural NO-GO: no outcome, sign, cell ranking, contrast, or hindsight-effect result may be claimed |
| Neural MountainCar V1R2 | 15/15 development runs complete; independent reconstruction passed; feasibility NO-GO | frozen setup lacked native-goal headroom; no efficacy, transfer, or capacity claim; confirmatory seeds untouched |

V5A's authorization used no pass-rate, return, entropy, AUC, final-performance,
or between-cell performance field. It selected `U*=250`; the projected
180-run serial runtime was `7.0557400375` hours. V5B used fresh paired seeds
`16000..16019` and four frozen update-indexed AUC contrasts. All 180 runs
completed with zero run failures. A post-hoc forensic audit covered 53,510
group records, 45,000 updates, and 1,080 checkpoints.

The frozen analyzer did not accept the completed artifact. The runner used
NumPy reductions for step-norm diagnostics, while the analyzer independently
used Python scalar reductions and then required exact dictionary equality.
A post-hoc forensic reduction audit found 377 mismatches among 720 diagnostic
floats; the maximum absolute difference was `1.9984014443252818e-15`, and the
maximum distance was 11 ULP. The protocol labels these step norms as diagnostics, but
its all-or-nothing acceptance rule requires exact runner/analyzer agreement.
The official V5B primary family is therefore a procedural NO-GO. Do not report
or infer any V5B outcome, cell ranking, sign, contrast, or hindsight effect.
The sealed runner artifact already contains case, contrast, and decision
subtrees, which violates the protocol's literal "not computed" rule; those
fields are quarantined and must remain uninterpreted.

A post-hoc tolerance-aware compatibility audit passed the remaining checks.
It is explicitly non-authorizing: it diagnoses the verifier failure but does
not rescue the primary family. Review the
[`verification erratum`](frontier_rl/examples/ACROBOT_HINDSIGHT_V5B_VERIFICATION_ERRATUM.md)
and
[`forensic report`](frontier_rl/examples/acrobot_hindsight_v5b_forensic_verification.json).
A future primary test requires an independently reviewed tolerance-aware
verifier and fresh V5C seeds; neither tolerance nor design may be tuned from
V5B outcomes.

Neural MountainCar V1R2 pooled 1,932 all-fail, 474 mixed, and zero all-pass
groups. Hardest-goal AUC was zero in all 15 runs, so all four primary
development contrasts were zero. Supporting mean-pass AUC deltas were
`+0.0065104`, `+0.0119792`, `+0.00546875`, and `+0.00429688`; these are
descriptive only. Reserved seeds `18000..18019` were not run and are not
authorized.

## Provenance: what “registered” means here

The locks are local source/runtime/configuration records written before their
corresponding seed blocks. They are machine-checkable but **not externally
timestamped preregistrations**. Review language such as “registered,”
“predeclared,” and “sealed” with that boundary in mind.

For the paid-probe attachment, the compact repository retains receipts and a
content boundary; the 1,374,886,097-byte raw ledger remains external with
SHA-256
`b1f8756c249effab8c77101c8bca73ddf708a5e143c18fe8742fd5712fdd7c12`.
There is no immutable public pre-execution commit. The registry generator emits
and checks exactly 562 records, including 441 Acrobot records, and owns the exact
totals. All Mac experiments required for this submission are complete; PLR,
PAIRED, ACCEL, and full SFL are deferred. The final paper rebuild and public
PDF synchronization are complete.

At reviewed snapshot `2dbda4e`, the listed source hashes in the V3, V4A, V5A,
V5B, and neural MountainCar V1R2 locks matched their files. The subsequent
reviewed Cosmos integration evolved `frontier_rl/estimators.py`,
`frontier_rl/trainer.py`, and `frontier_rl/adapters/__init__.py`; the historical
locks remain unchanged and those exact source bytes are recoverable from
`2dbda4e`. There is one older, non-recoverable historical exception:

- Acrobot V2 expected `frontier_rl/examples/run_acrobot_neural.py` SHA-256
  `215f126eb560bf330d44f2fb7b38792c78488e14693fa57fb493ac49820017eb`.
- The retained V3-era runner SHA-256 is
  `7bd8e4c7d2e85e98d0ca769ce6a65c997307cb4c6b65a8b90c00edb862c49cbf`.
- All other V2 manifest entries match current bytes. The V2 lock preserves the
  expected runner hash, but the exact historical runner bytes are not present
  at HEAD. Treat this as a V2 reproducibility limitation; do not retrofit V2
  outcomes to current code.

No lock in this repository proves wall-clock chronology to an external party.
The independent analyzers provide internal reconstruction and consistency, not
external registration certification.

## Evidence-chain hashes

### Acrobot V3

- result artifact:
  `30da4b9759828acb9357f5518a48196a6be98d314dffb7830b0ba4f89a31e423`
- independent verification:
  `1bb604925b36050c6b1520fce847919d7962ec8f0e300b50de49242b70b7b394`

### Acrobot V4A

- lock: `b19488783e1adba8cbac44ce8256c725a4470d8108c1192f9491ecc4882f1d8c`
- artifact: `69b827dc425014f3b568186981e9c24d95158c72653125e0ade181272def2891`
- independent verification:
  `c633e09df8e056f1589e631ff4d311913e1ac5594c3647790acc4b05990fca88`

### Acrobot V5

- V5A lock:
  `5c277413c5238f5839d281e09810537221a16737f831a498a3e0217ca5b1502e`
- V5A artifact:
  `9cf741c91dcb82218cada9b451b76e0811c67aa4cbf1786ac0ba926806479b0a`
- V5A independent verification:
  `a46b5e9f732b7f9e1796e2d4a2ff344c9ff738574c464b28631e884faaa6ba19`
- V5B amendment:
  `11975381874842bc3019074ea9d8168006c0517982ac11e00ad0b488e7671f36`
- V5B lock:
  `dfc930bbaf8e51c96fd1dab5851179457fce4f151def8c138ddf0cf17402bcf2`
- V5B completed artifact (`frontier_rl/examples/acrobot_hindsight_v5b_factorial.json`):
  `c633886a121906ee2bceb03f3117e4bea5dc20ab314e43f9b702ef8d88f495ac`
- V5B frozen-verifier erratum:
  `frontier_rl/examples/ACROBOT_HINDSIGHT_V5B_VERIFICATION_ERRATUM.md`
- V5B non-authorizing forensic verification:
  `frontier_rl/examples/acrobot_hindsight_v5b_forensic_verification.json`

### Neural MountainCar V1R2

- development lock:
  `b5edbc33048a8d3a8d7dbb992a23178ddf8424dd3c5be3165c87e6dc42a50a5c`
- development artifact:
  `2e4803805009a3323307f6bdcfae17fb625008adb5361dc2310e414a19129180`
- independent verification:
  `fdefc9e4ee2887953c341d2f44c44001bc336598b089dbdc8035175e430148a0`

## Recommended review order

1. Check the estimator convention and proofs in `curriculum_maxrl/PROOFS.md`.
   In particular, distinguish the practical drop-both estimator's `T=N-1`
   population target from the paper's raw/always-control-variate `T=N` forms.
2. Review `frontier_rl/estimators.py`, `frontier_rl/teacher.py`, and their tests
   against the coefficient-mass identity and requested-only teacher evidence.
3. Reconstruct the V3 claim from its protocol, lock, artifact, and independent
   report before reading any broader narrative.
4. Treat V4A and V5A as chronological feasibility studies. Verify that V5A
   uses fresh seeds and that its authorization projection excludes outcome
   fields.
5. Review V5B's amendment and all-or-nothing validity rule, then reproduce the
   exact diagnostic mismatch from the erratum and forensic report. Treat the
   completed artifact as a procedural NO-GO; do not inspect its performance
   outcomes or attempt to rescue the four-contrast family.
6. Review neural MountainCar V1R2's raw reconstruction and confirm that the
   missing all-pass regime forces NO-GO irrespective of the supporting deltas.
7. Only then compare the paper, report, framework, and README against these
   narrower evidence chains.

Useful focused checks from the repository root are:

```bash
python -m pytest -q \
  frontier_rl/examples/test_run_acrobot_hindsight_v5.py \
  frontier_rl/examples/test_analyze_acrobot_hindsight_v5.py

python -m pytest -q \
  frontier_rl/examples/test_mountaincar_neural_transfer_v1.py \
  frontier_rl/examples/test_analyze_mountaincar_neural_transfer_v1.py
```

## Questions for the expert review team

1. Does coefficient mass remain a useful scheduling proxy once trajectory
   score norms and task-dependent episode lengths are included?
2. Is the proposed tolerance-aware V5C verifier strict enough to catch genuine
   reconstruction defects while accepting benign NumPy/Python reduction
   roundoff, and is it frozen before fresh seeds are exposed?
3. Is hardest-goal AUC the right primary transfer metric for neural
   MountainCar, and what outcome-blind development rule best creates headroom
   without tuning on the desired method contrast?
4. Do the exact total- and active-capacity controls adequately triangulate
   sharing, or is a behaviorally calibrated optimizer another necessary axis?
5. Which claims should be removed or narrowed before submission, given that
   the locks are local and the V2 historical runner bytes are unavailable?

The desired review outcome is a claim audit, not a search for a positive
headline. A failed gate, null contrast, or reconstruction defect should remain
visible and should lower the claim boundary.
