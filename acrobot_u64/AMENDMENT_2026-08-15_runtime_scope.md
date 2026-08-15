# U64 preregistration amendment — 2026-08-15, runtime scope

**Type:** scope clarification of the lock's runtime comparison.
**Timing:** written and applied **before any confirmatory run**. No confirmatory
run had been executed, no confirmatory output existed, and no outcome of any
kind was inspected.
**Scientific protocol changed:** none. No arm, score, seed, budget, metric,
SESOI, decision rule, or interpretation row is altered.

## What changed

The frozen preregistration says the lock pins "this host's runtime". As first
implemented, `load_and_verify_lock` required the **entire** runtime dict to be
equal, including `platform`, which is `platform.platform()` and embeds the
kernel build string, e.g.

    Linux-6.8.0-124-generic-x86_64-with-glibc2.35

The lock now compares this pinned set:

| field | compared | rationale |
|---|---|---|
| `python_implementation` | **yes** | CPython vs PyPy changes numerics |
| `python` | **yes** | pinned 3.12.13 |
| `numpy` | **yes** | pinned 2.5.1 |
| `gymnasium` | **yes** | pinned 1.3.0 |
| `machine` | **yes** | an architecture change still fail-closes |
| `platform` | **recorded, not compared** | kernel build string |

`platform` is still captured twice: once in the lock as
`build_host_runtime`, and once per run in `provenance.runtime`.

## Why

Requiring kernel-string equality makes the campaign unrunnable on any real
cluster, because compute nodes do not share a login node's kernel build and a
heterogeneous cluster does not share one across node families. The kernel build
string is not a scientific variable: it does not affect the simulation, the
estimator, the score, or the RNG. Continuing to require it would have forced the
campaign onto a single workstation for a reason unrelated to the science.

The architecture check is retained, so an unnoticed move to a different
instruction set still fails closed.

## The residual risk this creates, and how it is controlled

Relaxing `platform` admits the possibility that different runs land on
different CPU models with different floating-point/BLAS paths. For a **paired**
contrast that would be a real hazard if arms within a seed could be split
across nodes.

They cannot. The campaign runs **all four arms of a given logical seed inside a
single process on a single node** (`--all-arms`). Within-seed pairing — which
is what every primary estimand uses — is therefore exact by construction.
Between-seed node heterogeneity only enters as ordinary between-seed variance,
which the paired sign-flip test already accommodates.

To make this auditable rather than merely asserted, every run record now stores
`provenance.cpu_model` and `provenance.hostname`. The analyzer can therefore
report the node/CPU composition of the campaign, and a reviewer can check
whether any contrast is confounded with node family.

## Verification

- The lock was rebuilt after the change; `runtime_pinned` is
  `{python_implementation: CPython, python: 3.12.13, numpy: 2.5.1,
  gymnasium: 1.3.0, machine: x86_64}`.
- The source-hash discipline is unchanged and demonstrably live: an edit to
  `run_u64_tournament.py` made while the first development gate was running
  caused subsequent runs to fail with `source hash mismatch`, exactly as
  intended. That development gate was discarded and re-run in full against the
  settled sources.
- No development output from before this amendment is used for anything.
