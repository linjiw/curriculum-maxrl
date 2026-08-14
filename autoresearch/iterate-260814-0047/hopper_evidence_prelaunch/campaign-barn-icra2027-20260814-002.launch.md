# Frozen BARN evidence launch — 2026-08-14

- Campaign ID: `barn-icra2027-20260814-002`
- Amended source commit:
  `55d46ccb04ceef2707c382293248ad50087cbb58`
- Evidence source SHA-256:
  `043d73a64cd63c2bc94e7f3c8fac4a97a3ff3e6b7671775a6402d0066db27760`
- Evidence source bundle:
  `/scratch/lwang44/maxrl/bundles/barn_source/043d73a64cd63c2bc94e`
- Launch ledger SHA-256:
  `54fb6e79a833758227a30cd944ae654994d66e768c83aeace63725f83fa2364d`

| cell | attempt | Slurm array |
|---|---|---:|
| primary | `primary-attempt-001` | 9366868 |
| `ablation_n2` | `n2-attempt-001` | 9366873 |
| `ablation_n4` | `n4-attempt-001` | 9366878 |
| `ablation_n16` | `n16-attempt-001` | 9366883 |

The ledger contains exactly seeds 1--5 in each cell, 20 incomplete rows at
launch, one unique array per complete cell attempt, and one common set of the
seven frozen hashes. At 2026-08-14 08:09 UTC all 20 tasks were `RUNNING` with
the frozen `1-12:00:00` limit. Only scheduler metadata was read. No raw BARN
log, seed artifact, endpoint, selector, merger, or analysis was opened.
