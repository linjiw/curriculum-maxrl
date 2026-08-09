# Acrobot ProCuRL-selection result

## Scope and decision

This source-locked study tests the task-selection semantics visible in ProCuRL
commit `17904f1d7b9b29e089d4f70ae7aadf1da50ba6b2` when attached to the fixed
small MaxRL Acrobot learner used in this repository. It is not a reproduction
of ProCuRL's PPO learner and does not support conclusions about the complete
ProCuRL algorithm.

The replacement source lock was sealed before its quick, development, or
confirmation executions, but it was not published in an immutable public
preexecution commit. It is an internal audit trail, not a claim of public
preregistration.

The registered primary comparison was fixed-paid-budget target-uniform AUC for
continuous-range-matched `u16` softmax minus ProCuRL's
`softmax(20 p(1-p))`, paired over 80 fresh seeds. Support required both a mean
contrast of at least `+0.02` and a two-sided paired t-test `p <= 0.05`.
The `u16` temperature matches the continuous logit range of five; on the
20-probe `p_hat` lattice its maximum logit is approximately `4.97731`.

The observed mean contrast was `+0.004894235861048817` (sample SD
`0.022138618105654644`), `t(79) = 1.9773310205711703`, two-sided
`p = 0.05149237843697304`. The registered decision is therefore **not
supported**: the contrast misses both the minimum effect and the significance
criterion. The paired-seed bootstrap 95% interval was
`[0.00011018286171968657, 0.009727194706612728]`; the registered 1,000,000-draw
paired sign-flip robustness value was `p = 0.051614948385051616`. The bootstrap
interval does not override the frozen decision rule.

## Registered secondary family

All values below are left arm minus right arm on the same fixed-paid-budget AUC.
The decision column uses the frozen Holm familywise correction.

| Contrast | Mean | Paired-bootstrap 95% interval | Holm-adjusted p | Reject at 0.05 |
|---|---:|---:|---:|:---:|
| ProCuRL - probe sham | -0.0017042413974216685 | [-0.006058444716950572, 0.002836114579643207] | 0.4567812019012931 | no |
| u16 - probe sham | +0.0031899944636271478 | [-0.0019172824111441619, 0.008279310457178281] | 0.45468238514441794 | no |
| ProCuRL - ordinary uniform | -0.31377426306862766 | [-0.32402807048504145, -0.30315407922903126] | 3.4694712873304143e-66 | yes |
| u16 - ordinary uniform | -0.30888002720757884 | [-0.3188547190812068, -0.29875032140281493] | 9.481048628026351e-67 | yes |
| probe sham - ordinary uniform | -0.312070021671206 | [-0.3216978275577437, -0.30205809886922785] | 4.063304920294923e-68 | yes |

The adaptive selection rules do not separate from the paid probe-sham control.
All three probed arms are far below ordinary uniform under paid accounting.
This pattern localizes the large difference to the source-faithful probe cadence
and its cost in this setting, rather than demonstrating that either adaptive
selection functional is intrinsically harmful.

## Descriptive accounting

Values are arm means over 80 seeds. Parentheses give sample standard deviations
for the two outcome columns.

| Arm | Fixed-paid AUC | Final target-uniform success | Student transitions | Probe fraction | Probe sweeps | Optimizer updates |
|---|---:|---:|---:|---:|---:|---:|
| ProCuRL `beta=20`, refresh 5120 | 0.337714 (0.032698) | 0.364209 (0.043277) | 140,747.5 | 0.932002 | 28.04 | 19.83 |
| paid probe sham | 0.339419 (0.033665) | 0.366113 (0.045333) | 140,536.9 | 0.932135 | 28.06 | 15.44 |
| ordinary uniform | 0.651489 (0.048039) | 0.876904 (0.044019) | 2,002,984.3 | 0 | 0 | 259.06 |
| range-matched `u16`, refresh 5120 | 0.342609 (0.029416) | 0.366602 (0.049758) | 141,082.2 | 0.931938 | 28.13 | 16.16 |

The 20 probe episodes per task at the native 5,120-student-transition refresh
cadence consume about 93.2% of the paid transition budget. Consequently the
probed arms receive only about 141,000 student transitions and 15--20 optimizer
updates, versus about 2.003 million student transitions and 259 updates for
ordinary uniform. A slower cadence such as 80,000 transitions was not tested
and would require a separately frozen protocol and fresh seeds.

## Integrity and release boundary

The outcome-blind development gate passed all 11 checks on 12 development runs
(four arms by seeds 21300--21302). Confirmation contains 320 complete runs
(four arms by seeds 21000--21079). The engineering quick run and the archived
pre-gate entropy-summation mismatch wave are excluded from scientific and run
registry counts.

The full confirmation raw is intentionally ignored by Git and retained as an
external replay artifact. Its public content-addressed download URI is currently
`null`. The compact release retains the independent analysis, full per-seed
descriptive diagnostics, portable-verification receipt, and a manifest with one
canonical content hash for every confirmation run.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `ACROBOT_PROCURL_SELECTION_LOCK.json` | 59,833 | `b7c7f76f6aaffa1fe65557717bfe545f2ec850495d370cd97fe72a8871fc8d0f` |
| `acrobot_procurl_selection_development.json` | 11,453,535 | `6d9fa639295e35cd8a8da810ace82d330c863edace702db4e3f7d25a9ad82ba8` |
| `acrobot_procurl_selection_development_gates.json` | 3,405 | `1edf50dc0b86744b8e33a87afaee8f50dc05dd3ef0d3129bfb5eef2533cf34bf` |
| external `acrobot_procurl_selection_confirmatory.json` | 1,374,886,097 | `b1f8756c249effab8c77101c8bca73ddf708a5e143c18fe8742fd5712fdd7c12` |
| `acrobot_procurl_selection_analysis.json` | 109,299 | `2010e30b5b15a212e2d6bdfaacd43d2434e5f468a96be613cf59744b9bc2fb38` |
| `acrobot_procurl_selection_portable_verification.json` | 6,149 | `c6b754655cbe6fa0dc52e065cd46d840f6a62b9bc58104ceb7443c727b0d01ae` |
| `acrobot_procurl_selection_diagnostics.json` | 431,895 | `583d950b8e85e6ea3efb477e6a390ceeac339054e46bb5ac69ff4597438d48c7` |
| `ACROBOT_PROCURL_SELECTION_EXTERNAL_RAW_MANIFEST.json` | 79,026 | `e197c1d581bcda8679ba4c0dc428fde10db6e80b653f1bb9e2dc02314e4dfdc6` |

A deterministic `gzip -9 -n` stream of the external raw (not retained as a
release file) is 141,609,471 bytes with SHA-256
`944efd2efb7b882ff8a4e2ad369ff977a2b09eb8d862c7ee7e8c1f5e88fb2679`.

The sealed protocol, primary-source provenance, lock, runner, analyzer,
portable verifier, and their lock-bound tests remain unchanged. The external
manifest can always be checked against compact artifacts; when the full raw is
present, the same command additionally reconciles all 320 canonical run hashes
and replays strict ledger validation under CPython 3.12.13, NumPy 2.5.1, and
Gymnasium 1.3.0.
