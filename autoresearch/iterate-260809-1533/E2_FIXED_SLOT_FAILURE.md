# E2 fixed-slot replay treatment-delivery failure

The prospective E2 fixed-slot replay arm for seed 1 stopped before the step-12
optimizer update on 2026-08-09. This is the preregistered inconclusive branch,
not an endpoint result.

- B1 and B2 completed 60/60 steps and saved their fixed checkpoints.
- B2 produced exactly 60 schedule rows.
- Replay completed 11 valid optimizer steps and wrote 11 audit rows.
- Through step 11 it matched 52 B2 accepted groups, used no slot fallbacks,
  displaced 8 informative slots (15.4%), and ended at 2.20% cumulative
  auxiliary-token mismatch and 1.52% cumulative total optimizer-token
  mismatch.
- Before step 12's optimizer update, the replay trajectory had no informative
  live group in its current generated batch while B2 requested replay. Strict
  delivery raised and stopped the run.

Per `E2_PREREG.md`, the current-batch-only three-seed direction test is
inconclusive and will not be restarted with a weakened gate. A recent-buffer
variant is a separate follow-up (`E2B_PREREG.md`).

The already-fixed seed-1 comparator endpoints were evaluated before E2b was
frozen and are disclosed as known:

| arm | tier-1 mean@16 | tier-1 pass@16 |
|---|---:|---:|
| B1 | 0.1987 | 0.6172 |
| B2 | 0.1265 | 0.2344 |

