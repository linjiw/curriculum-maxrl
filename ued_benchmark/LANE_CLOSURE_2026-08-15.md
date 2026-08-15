# UED / AMaze lane closure — 2026-08-15

**Decision: the lane is FROZEN IN PLACE for the ICLR 2027 run-in. All v4
remote-hardening, calibration-telemetry, and audit work stops now.**

Authority: `RESEARCH_PLAN_2026-08-15.md` §4 Cut 1, and risk-register item 9,
which requires this decision be written down once so it cannot silently
resurface under deadline pressure.

This is a **scope** decision, not a judgement that the engineering is wrong.
The work is careful and it is reversible (§4). It is being stopped because it
cannot reach a paper number before Sept 25, and because continuing it consumes
the only genuinely scarce resource in the project.

## Why

Each of these was verified in the repository, not inferred.

1. **Its own preregistration forbids the use it is being built for.**
   `UED_MATCHED_DEV_PREREG.md:5` reads, verbatim:
   `**Scope:** engineering/development selection only; never paper evidence`.
   Everything downstream of that line is development tooling by its own
   definition.

2. **The protocol does not fit the queue.** `UED_MATCHED_DEV_PREREG.md:75`
   budgets **492,036,096 student transitions per run** against 30,000 student
   PPO updates, on a documented 1-day `gpuq` cap with an explicit no-resume
   rule. A single run does not fit in a single allocation, and nothing in the
   lane implements resumption across allocations.

3. **The inference is empty even if it succeeds.** The design uses five paired
   development seeds and an exact two-sided **32-assignment** sign-flip
   (`:118`). 32 = 2^5, so the smallest attainable two-sided p-value is
   2/32 = **.0625**. No outcome of this design can reach p < .05. The prereg is
   honest about this — it calls the statistic "descriptive" — but it means the
   lane cannot produce a significance claim at its frozen seed count.

4. **The competitive arms it exists to beat do not exist.** `configs/` contains
   only `frontier` and `maxmc` variants. There is no ACCEL, no PAIRED, and no
   robust-PLR configuration anywhere in the lane. The comparison that would
   justify the lane has not been implemented, let alone run.

5. **Half the open blockers cannot be closed from here.** The v4 remote
   hardening audit's four primary blockers — protected-overlay incompatibility,
   the R2 `job-<id>` identity mismatch, system-Python GPU probing, and the
   invalid mandatory MIG-`gpumem` accounting — are all statements about remote
   Hopper behaviour that no local test can falsify. Local iteration against
   them is unfalsifiable by construction, which is why every audit round so far
   has ended by discovering more items rather than by authorizing a remote
   action.

6. **Yield.** Roughly 44,400 lines, 11 telemetry build/verify rounds, 2 bundle
   freezes and 3 independent audits have produced 3.25 minutes of real GPU
   compute, exactly one PPO update, and zero numbers in any manuscript.

## What is preserved

Nothing is deleted. The lane is frozen in a resumable state:

- upstream pin `facebookresearch/minimax` @
  `d053054c5290a04c1c4cd8b55704d999cad73e30`,
  tree `b0cace1fc54984e21a842f12d15d0b899e33d270`, Apache-2.0
  (`UPSTREAM_PIN.json`);
- reference environment Python 3.10 / JAX 0.4.31 / jaxlib 0.4.31 / flax 0.8.5 /
  chex 0.1.86 / optax 0.2.3 / NumPy 1.25.2;
- overlay contracts `OVERLAY_CONTRACT.json` and `OVERLAY_CONTRACT_V4.json`;
- the v3 bundle proven on Hopper by jobs 9366896 (import/JIT) and 9366897
  (bounded one-update), and the independently audited terminal-chain bundle
  `06ffeeeb6998e8ddb1ce`;
- the tie-aware v4 sibling bundle `d602ce7854f8…` and the bounded
  remote-contract snapshot `da74eb3e0d…`;
- the calibration-telemetry static GO: protocol `4053c520…`,
  analyzer `19b07d2f…`, 48 tests.

**The unmeasured-throughput blocker is the one thing that should be recorded as
still open:** the 248.97 tr/s figure is compile-contaminated and no
steady-state measurement exists, so no honest schedule for this lane can be
built until it does.

## What is explicitly NOT authorized before Sept 25

- Any further v4 remote-hardening work.
- Any further calibration-telemetry implementation, writer, driver, or audit.
- Any Hopper staging, submission, or endpoint access in this lane.
- Any new audit round against the four remote blockers.
- Any manuscript text that cites this lane as evidence.

## The one permitted future action, post-submission only

A single bounded steady-state throughput probe — roughly 200 updates, ≤ 1
GPU-hour — on the already-Hopper-proven v3 bundle, to replace the
compile-contaminated 248.97 tr/s figure with a real number for a future paper.
**Not before 2026-09-25.**

## How to reopen

This note is the only thing standing between the lane and resumption; it costs
one commit to revert. Reopening is justified when, and only when, all three
hold:

1. the ICLR submission is in;
2. a steady-state throughput measurement exists, so the 492M-transition budget
   can be turned into a real wall-clock estimate against the queue cap;
3. at least one genuine competitor arm (ACCEL, PAIRED, or robust PLR) is
   implemented, so the comparison the lane exists for is actually runnable.

Until then the honest description of this lane, in any venue, is: careful
engineering scaffolding with no scientific endpoint.
