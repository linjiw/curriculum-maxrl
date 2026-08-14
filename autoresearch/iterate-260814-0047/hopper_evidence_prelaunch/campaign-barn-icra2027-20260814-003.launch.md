# Amended BARN evidence relaunch — 2026-08-14

- Campaign ID: `barn-icra2027-20260814-003`.
- Outcome-blind publication-fix commit:
  `96ab585faedb041e3501fe71d732289d0d5c23fc`.
- Evidence source SHA-256:
  `b9e20a561c8edc93daec8638b15f031dd532eacb39f9d7488582c516ca3dc81c`.
- Evidence source bundle:
  `/scratch/lwang44/maxrl/bundles/barn_source/b9e20a561c8edc93daec`.
- Amended preregistration SHA-256:
  `bd6910523d8494e6386b3bb1e816a8e9841becbbaf75809166addc382bd8f0d3`.
- Byte-identical machine protocol SHA-256:
  `36007d8c979b2dacccd595a43a4620dca7be24c1f50ef91a8a9ee4e869202cb2`.
- Launch ledger SHA-256:
  `0a1fc224e71ad2437fce35b40c6561c4b8aeb8750ef6af66af0c34bae731d576`.

| cell | attempt | Slurm array |
|---|---|---:|
| primary | `primary-attempt-001` | 9367009 |
| `ablation_n2` | `n2-attempt-001` | 9367011 |
| `ablation_n4` | `n4-attempt-001` | 9367020 |
| `ablation_n16` | `n16-attempt-001` | 9367022 |

The normalized ledger contains exactly seeds 1--5 once in each exact cell,
20 incomplete rows, one unique array per complete cell attempt, and one common
set of the seven frozen hashes.  The local and canonical remote ledger bytes
have the same SHA-256 above.  No pending submission marker remains.

Primary job 9367009 stayed user-held across two source-bound ledger-install
attempts while the NFS-backed scratch object propagated between Hopper login
hosts.  The exact third resume observed the immutable staged ledger, installed
the canonical bytes, verified their digest, and released the same array.  No
replacement job or altered ledger was created.  The other three cell
transactions installed and released normally.

At `2026-08-14T10:37:04Z`, all 20 tasks were `RUNNING`, with 8 CPUs, 24 GB,
no GPU/GRES, and the frozen `1-12:00:00` limit.  The exact staged seed
publisher also passed an outcome-free scratch publication probe before
submission.  Only source/ledger hashes, scheduler state, resource requests,
and completion-marker existence were inspected.  No raw BARN log, seed
artifact, endpoint, selector, merger, or analysis was opened.
