# Capped-HORA robustness: exploratory results

## Status and scope

This is the completed readout of the frozen
[`CAPPED_HORA_ROBUSTNESS_PROTOCOL.md`](CAPPED_HORA_ROBUSTNESS_PROTOCOL.md).
It is a **post-guidance, exploratory multiverse** in one synthetic
`SkillChainEnv` testbed.  The analysis reports equal-seed paired differences
and Student-`t` 95% intervals descriptively; it assigns no nominal contrast
`p`-values and promotes no result to confirmatory status.

The 50-cell design crosses two task samplers, two adaptive allocators, four
caps, and three probability-information sources, with a fixed-`N=16` anchor
under each sampler.  Every cell uses 16 logical seeds, 51,200 paid
completions, and common completion checkpoints.

## Artifact and validation record

The retained artifacts are bound to the frozen source/runtime lock.
That lock is internally hashed, but this checkout does not provide an
immutable public pre-execution commit that independently establishes timing.

| Item | SHA-256 |
|---|---|
| source/runtime lock | `14418bb71e92b5eedf2b4220995edcaacf0a8b75337976dbfe5917ae6a7d60e7` |
| frozen protocol | `f206a58ee3bfe81b4f4b34ba18635631c11e85bc4e6e4b722d01e5951204f718` |
| engineering/overlap validation receipt | `8e1ddda1bd18590d504a3d7fed8ef9cdd18cd4c718ea0e1a77059b2002573dc8` |
| raw 800-run artifact | `b2df9946c699b4d968dee28ba39657cc2f2e532e24d0beb95a4af3f09158a768` |
| independently regenerated, path-normalized analysis | `2141bf38c399eddf32002bcebeccdc5ec8a7e47089716b9539ff470c14be0c16` |

The independent analyzer accepted **800/800** unique cell-seed runs
(`50 cells x 16 seeds`).  It checked every run against the frozen cell and
seed matrix and reconstructed its accounting.  All nine artifact-wide checks
are true: complete unique matrix; exact 51,200-completion budget; exact average
group size 16; common checkpoints; fixed anchors always at 16; every adaptive
cap respected; dead/mixed/all-pass group accounting closed; teacher evidence
charged exactly once; and allocation-diagnostic applicability exact.  The
analyzer also matched the locked runtime (CPython 3.9.6, NumPy 1.26.4), source
hashes, and RNG/tie semantics.

## Registered contrast inventory

All registered cells and contrast families are retained; none was dropped
after observing outcomes.

| Frozen contrast family | Contrasts |
|---|---:|
| adaptive minus fixed anchor | 48 |
| cap minus uncapped within allocator | 36 |
| fresh-group mass proxy minus HORA hit within cap | 24 |
| history-plus-probe minus same-step information | 16 |
| oracle-preupdate minus deployable information | 32 |
| **Total** | **156** |

Every contrast contains all 16 paired logical seeds and the full frozen metric
set.  The intervals below are therefore selected summaries of the registered
multiverse, not a reduced hypothesis family.

## What happened at the selected cap

The pre-outcome engineering rule selected **cap 32**.  In the
fresh-group-mass-proxy/same-step cells, averaged equally over samplers within
seed, cap 32 reduced the mean per-run maximum group size by
`58.0671581%` and changed normalized pass@8 AUC by `-0.0027135967` relative
to uncapped.  It therefore cleared the registered requirements of at least a
25% maximum-size reduction and AUC change no worse than `-0.005`.

For completeness, cap 24 reduced maximum size by `68.5503686%` but failed the
AUC guardrail (`-0.0114059341`).  Cap 48 was eligible (`37.1007371%`
reduction; `+0.0015296619` AUC), but cap 32 was the smallest eligible cap and
thus the frozen selection.

The two sampler-specific cap-32 versus uncapped contrasts show why the pooled
engineering result should not be called equivalence.  Under uniform sampling,
the fresh-proxy/same-step AUC difference was `-0.0090233630`, descriptive 95%
interval `[-0.0153159215, -0.0027308045]`, with 3/16 positive pairs and a
mean maximum-size change of `-43.875`.  Under `u_16`, it was
`+0.0035961697`, interval `[-0.0048906644, +0.0120830037]`, with 9/16
positive pairs and a mean maximum-size change of `-44.75`.

## Key individual learning contrasts

The table reports normalized pass@8-AUC differences for the eight deployable
cap-32 adaptive cells versus their same-sampler fixed-`N=16` anchor.  Signs are
positive/zero/negative across the 16 paired seeds.

Across all caps and both deployable information sources, all 32
adaptive-minus-fixed mean AUC contrasts were positive and 22 unadjusted
descriptive intervals were above zero. Coefficient mass was lower in 28/32
cell means, with 24 intervals below zero and none above zero. The separate 16
oracle-preupdate cells are non-deployable diagnostics and are excluded from
these counts; no multiplicity-adjusted inference is assigned.

| Sampler | Allocator | Information | Mean difference | Descriptive 95% interval | Signs |
|---|---|---|---:|---:|---:|
| uniform | HORA hit | same step | `+0.0187549836` | `[-0.0024394219, +0.0399493891]` | 10/0/6 |
| uniform | HORA hit | history + probe | `+0.0291639610` | `[+0.0169188169, +0.0414091051]` | 14/0/2 |
| uniform | fresh-group mass proxy | same step | `+0.0203163383` | `[+0.0026644259, +0.0379682508]` | 12/0/4 |
| uniform | fresh-group mass proxy | history + probe | `+0.0323792231` | `[+0.0130583539, +0.0517000924]` | 13/0/3 |
| `u_16` | HORA hit | same step | `+0.0133209312` | `[-0.0033482475, +0.0299901099]` | 12/0/4 |
| `u_16` | HORA hit | history + probe | `+0.0166281743` | `[+0.0016640587, +0.0315922900]` | 11/0/5 |
| `u_16` | fresh-group mass proxy | same step | `+0.0245942718` | `[+0.0110278624, +0.0381606812]` | 13/0/3 |
| `u_16` | fresh-group mass proxy | history + probe | `+0.0249181283` | `[+0.0106071012, +0.0392291553]` | 14/0/2 |

These fixed-anchor comparisons do not establish that the fresh-group proxy is
better than HORA hit.  At cap 32 with same-step information, fresh proxy minus
HORA hit was `+0.0015613548` under uniform sampling, interval
`[-0.0056484468, +0.0087711564]`, signs 8/0/8; under `u_16` it was
`+0.0112733406`, interval `[-0.0019416046, +0.0244882859]`, signs 11/0/5.
Both descriptive intervals cross zero.

Nor do the learning contrasts support coefficient-mass mediation.  For the
cap-32 fresh-proxy/same-step cell versus fixed, pass@8 AUC was positive under
both samplers while realized coefficient L1 mass per completion was lower:
`-0.0004009924` under uniform and `-0.0007586816` under `u_16`.  This repeats
the earlier direction mismatch; it does not identify why learning improved.

## Strict claim boundary

The supported conclusion is limited to sensitivity inside this synthetic,
shared-skill environment: a cap of 32 is the frozen engineering choice for a
later pilot, several capped deployable cells learn faster than their fixed
anchors in descriptive paired readouts, and the relative effects vary with
sampler and information source.

This experiment does **not** validate HORA, reproduce its neural results,
establish neural-RLVR performance, prove coefficient-mass mediation, or add a
fourth confirmed paper contribution.  The fresh-group score is an
unconditional one-rollout proxy, not the exact conditional marginal of the
realized pooled-probe coefficient mass.  The cap rule is a prospective
engineering filter based on point estimates, not an equivalence test, and the
oracle-preupdate cells are non-deployable diagnostics.

## Reproduction

The structured inputs are
[`results_capped_hora_robustness.json`](results_capped_hora_robustness.json)
and
[`results_capped_hora_robustness_analysis.json`](results_capped_hora_robustness_analysis.json).
From the repository root, independently verify and regenerate the analysis
with:

```bash
HORA_PYTHON=/usr/bin/python3 bash reproduce.sh
```

The stored analysis records the generating checkout's absolute raw-artifact
path. The reproduction check therefore normalizes only that location field
before comparing canonical analysis content; the raw SHA-256 remains exact.
