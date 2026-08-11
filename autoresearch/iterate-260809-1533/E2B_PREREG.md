# E2b preregistration: recent-buffer dose-matched live replay

Frozen on 2026-08-09 after E2's prospective current-batch replay arm failed
its treatment-delivery gate at step 12, and before any recent-buffer replay
training or endpoint evaluation. Seed-1 B1/B2 endpoints were already known and
are disclosed in `E2_FIXED_SLOT_FAILURE.md`; E2b is prospective only for the
replay-buffer outcomes and their contrasts with those immutable comparators.

## Change from failed E2

All artifacts, arms, seeds, training hyperparameters, fixed B2 accepted slots,
token gates, checkpoint rule, evaluation protocol, endpoints, and decision
branches remain as written in `E2_PREREG.md`. The sole treatment change is the
source pool for R:

- Rb may draw from informative live groups in the current batch or from the 64
  most recent informative group snapshots.
- A buffered source is eligible for at most eight subsequent optimizer steps.
- Current and buffered sources enter the same deterministic dynamic-programming
  token matcher; sampling is with replacement and a source cannot be the same
  current group as its target slot.
- Each audit records source kind, source step, source age, current/buffer
  candidate counts, age evictions, and buffer sources used.

The buffer contains exact previously generated prompt/response/reward groups;
it creates no new generations and uses no relabel information. Old log
probabilities are recomputed under the current model after insertion, matching
B2's auxiliary weighted-SFT placement but not claiming an unbiased off-policy
policy gradient. Rb is therefore a recent self-imitation replay placebo.

## Additional delivery gates

In addition to all original E2 gates:

1. every buffered source must have age in [1, 8];
2. every current source must have age 0;
3. buffer capacity must never exceed 64 groups; and
4. the current-plus-recent source pool must be nonempty whenever B2 requests a
   replay group.

Any violation stops the affected seed and makes E2b's three-seed direction
test inconclusive. The displaced-live-slot fraction must still be at most 25%.

## Pilot and endpoint hygiene

A seed-1 12-step delivery pilot will run first because step 12 was E2's failure
point. It has no held-out evaluation, cannot be resumed into the 60-step run,
and is excluded from endpoint analysis. If it passes, Rb seed 1 starts again
from the frozen SFT checkpoint. Seeds 2 and 3 then run clean B1, B2, and Rb
cells with the same paired evaluation protocol. No checkpoint other than step
60 is evaluated.

