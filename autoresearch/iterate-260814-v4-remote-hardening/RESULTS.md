# V4 remote-hardening bounded freeze

Date: 2026-08-14

## Verdict

- Narrow local contract modules: **GO as a deterministic, non-launchable engineering snapshot**.
- Full local R0/R2/R3 hardened ladder: **HOLD**; no complete execution-shaped E2E was claimed.
- Hopper staging/submission: **HOLD**. The launcher implements no submit operation and this turn performed no SSH, staging, scheduler query, submission, or endpoint access.
- Production, cost100, analyzer eligibility, and paper evidence: **HOLD / false**.
- Protected v3 job `9367063` and bundle `06ffeeeb6998e8ddb1ce` were not touched, reused, cancelled, requeued, or relabeled.

## Frozen local twin

- Candidate A: `/tmp/ued-v4h-hold-candidate.Bg0Kub/a/da74eb3e0debc7781d6d`
- Candidate B: `/tmp/ued-v4h-hold-candidate.Bg0Kub/b/da74eb3e0debc7781d6d`
- Root `SHA256SUMS`: `da74eb3e0debc7781d6d785f9406acec953a02cfcc3674afeb70c0f438619cc8`
- Bundle ID: `da74eb3e0debc7781d6d`
- `REMOTE_HARDENING_STATE.json`: `56ee7c1b8c7023c89b74f7fad83009d70a4feb198df17b186461e3fb5612fb7a`
- `ued_benchmark/OVERLAY_SHA256SUMS`: `c923440e4e9edcf7f9e7e50b4209d00f6fd962703d8d76fb760576a75ac3c13c`
- A/B `diff -qr`: clean.
- All 18 new workspace files are byte-identical to both staged twins; no symlinks or generated `__pycache__` files are present.

The historical d602 core `BUNDLE_STATE.json` is intentionally preserved independently from the sibling `REMOTE_HARDENING_STATE.json`.

## What the narrow snapshot establishes

- Sibling-only v4 namespace and local deterministic bundle builder.
- Permanently false paper/production/endpoint/cost100 flags and one-update cap.
- Slurm `NIL` design with a NUL argument envelope and no implemented submit command.
- Exact A100 MIG `1g.10gb`, resource, no-array/no-requeue/no-restart contract models.
- Installed environment byte-tree closure model.
- Pair-plan, pair-index, Phase-B package, resource receipt, and common provenance models.
- Exact archived receipt keyset checks.
- Explicit Slurm-spool bootstrap using canonical submitted-bundle arguments and spool-byte comparison.
- Correct preserved terminal Phase-A manifest filename (`SHA256SUMS`).

## Independent-audit HOLD items

The independent read-only audit found no basis to authorize remote action. The important unresolved items are:

1. The widened sibling overlay cannot be passed directly to preserved d602 R1/R2 scripts: their preserved `BUNDLE_STATE` overlay digest and the sibling overlay digest create an execution-time contradiction. A separate exact d602 core view or a newly audited non-mutating adapter is required.
2. The batch guard currently starts with system Python and imports/probes JAX there; the GPU/JAX probe must run under the exact byte-closed Python 3.10.20 environment. The shell bootstrap must not assume Hopper `/usr/bin/python3` supports newer APIs.
3. Slurm/NVML does not normally expose `gres/gpumem` accounting for MIG devices. Missing MIG gpumem must be an explicit accepted accounting-unavailable state, with capacity bound by the CUDA runtime receipt; over-cap values must still fail.
4. R2 uses a `job-<job_id>` directory while pair-plan code compares it to bare `job_id` in three live/archive paths.
5. R1/R2 need authoritative post-terminal submission/sacct receipts, including `.batch` and `.extern`, and their GPU/precheck/complete bytes must be archived and cross-bound before a common pair plan is frozen.
6. The bundle/runner/envelope start boundary still needs an externally pinned trust anchor and queue-time TOCTOU protection before any shared runner executes.
7. Phase-B/package validation still needs external tool pins, exact live/archive semantic parity, local-vs-remote schema separation, and canonical input/output non-overlap guards.
8. Exact Hopper behavior, fixed Conda path, non-test Python 3.10.20 Phase B, A100 MIG runtime, and sacct schema remain unvalidated because remote access was prohibited.

## Local verification

- `PYTHONDONTWRITEBYTECODE=1 python3 -B ued_benchmark/hopper_v4_remote_hardening/test_remote_hardening_contracts.py`
  - 7/7 pass, including a simulated Slurm spool copy and tampered-spool rejection.
- `PYTHONDONTWRITEBYTECODE=1 python3 -B ued_benchmark/hopper_v4/test_local_terminal_chain_v4.py`
  - pass: two arms, 60 actual evaluation records, sealed values.
- `bash -n` over all new shell/sbatch siblings: pass.
- Python AST parse over all new Python siblings: pass.
- Two independent local stage builds plus `diff -qr`: pass.

## Exact new-file hashes

```text
f3c495f95d630321a2092aa1c5d03418efea3484f209d39f26565f2215805780  ued_benchmark/hopper_v4_remote_hardening/__init__.py
0c76030c0196c84526ed6e4f3521b5a1a88d63d38fb84628374a6f1571b3085a  ued_benchmark/hopper_v4_remote_hardening/assemble_remote_hardened.py
7750150720ef8a00b2a0dfa0d22f972b962a0d16a15ddf42d498902a3f6249f8  ued_benchmark/hopper_v4_remote_hardening/environment_tree.py
b98b5914de53d077049a0d5ea7e71b49251a596139ec1c31292d26db64d19e5b  ued_benchmark/hopper_v4_remote_hardening/gpu_runtime_probe.py
a948bf7caa19f01c1313fb08935230e03ce95fd9090ca913a8faffd957d04d11  ued_benchmark/hopper_v4_remote_hardening/job_guard.py
0fafe808e354dbb4221e52251b08020bf5cfc4e5b2d8b7025b9fca2fed3beb7b  ued_benchmark/hopper_v4_remote_hardening/pair_index.py
056629a1c8d2c3baf56b724fd8eb9e81e17d93a513fc52237c85c43312cada3c  ued_benchmark/hopper_v4_remote_hardening/pair_plan.py
5dc31da5f844b2e60966e2b1f5679ca58ffea5d17954b704fbad239f1f454df2  ued_benchmark/hopper_v4_remote_hardening/slurm_integrity.py
b95181ac7f876923c4419c8261751d7dd4fb6c640b000cdf7cb830234f2d5057  ued_benchmark/hopper_v4_remote_hardening/test_remote_hardening_contracts.py
84d59700681d1cf8cac9a9c791b09f9620a01c1a63e7c20b10050299cc30889b  hopper/hopper_v4_remote_hardened.py
15926d78c7d493330df7f08689618e9632add0825aeed3a1fd1692f7724bf22c  hopper/hopper_v4_remote_hardened.sh
2c40e95febef4e8ce56331ace862b350fb0ff7d7c17e20bace15756907e9f931  hopper/run_ued_minimax_v4_remote_hardened.sh
92fe03220410d691994d3549de3cf0ff90559c949b84cbe369c5fe3478687cb5  hopper/setup_ued_minimax_env_v4_remote_hardened.sh
b8d1675e4f8cd6457827cabbe0a853f7430d9ae42c5a133d66ffb27dbdc255e5  hopper/stage_ued_minimax_v4_remote_hardened.sh
b03ae95e075b745e416bfe07012b82ad83463754046a6f3802cc2077c4f7cdec  hopper/finalize_ued_minimax_v4_remote_hardened.py
e3ec5d31578ecadb3eac828ae8083f2fa5511972bf8747246bf59f0977698d60  hopper/sbatch/ued_minimax_v4_remote_hardened_gpu_smoke.sbatch
d6c4e2c2284b3dea26910a96aa793e1c6e542fd988774d0f464209099bee3de5  hopper/sbatch/ued_minimax_v4_remote_hardened_one_update_smoke.sbatch
972c0c884ce256c2a1139184b25f2d9089c11b3d444a86bb36640f07716b71ed  hopper/sbatch/ued_minimax_v4_remote_hardened_terminal_chain_smoke.sbatch
```

Protected-history hashes remain:

```text
73ad318fe21f6f99c92fe09ac6ec76c5dd6fe4b0d7fec8a5ca5939c59483ba55  hopper/stage_ued_minimax.sh
5ed5186e010decdcc6bf97ff7dc820e0f4cf13e580e9b20c996a8dc561b13a14  hopper/sbatch/ued_minimax_terminal_chain_smoke.sbatch
57eb4394cedf30cc1a5bfeca4734199652cbfcfd5fbcaaca08035e8001a2c5ec  hopper/finalize_ued_minimax_terminal_chain.py
```

This is a deliberately stopped HOLD snapshot, not a request or authorization to run it on Hopper.
