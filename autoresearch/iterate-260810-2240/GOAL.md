# Research goal: close the Countdown causal-accounting gap

**Frozen:** 2026-08-10 22:40 America/New_York  
**Iteration:** `iterate-260810-2240`  
**Status at freeze:** E2 and E2b are treatment-delivery inconclusive; no
three-seed endpoint contrast from either experiment is interpretable.

## Project-level goal

Produce a submission-ready, artifact-backed account of how the advantage
estimator conditions curricula and failure recycling in RL with verifiable
rewards. The final paper should rest on three claims only:

1. an exact compute-indexed coefficient-mass result for the deployed MaxRL
   estimator;
2. a seed-block-correct neural confirmation that MaxRL and GRPO induce
   different time-integrated coverage under the same sampling interventions;
3. a reported three-seed Countdown aggregate in which recycling raises mean
   accuracy while a logged VERL bootstrap coverage proxy falls, followed by a
   prospective raw-outcome experiment capable of testing standard pass@16.

New experiments are justified only when they close a causal or artifact gap
in one of these claims. No new domain or broad mechanism claim is in scope.

## Most promising research direction

Run E2c: a three-seed, genuinely dose-matched Countdown control in which the
generic auxiliary direction comes from an immutable reservoir of informative,
train-only groups generated once from the same frozen clean-SFT checkpoint.
The reservoir removes the cold-start and sparse-support failures that made E2
and E2b inconclusive; it must not weaken or retroactively reinterpret either
prior protocol.

The scientific question is:

> At matched accepted-group count, optimizer-step count, fixed replacement
> slots, and response-token dose, does exact achieved-target relabeling improve
> tier-1 mean@16 or preserve standard raw-outcome pass@16 relative to a generic informative-group
> auxiliary update?

This is the highest-value direction because it resolves the only remaining
LLM-scale causal-accounting ambiguity in the paper's applied result. A larger
model, a new task, or another teacher sweep would add breadth without repairing
that ambiguity.

## Measurable definition of success

The iteration succeeds only when all of the following are true:

1. **Immutable source artifact.** A versioned reservoir is generated with a
   frozen generation budget from `countdown_v2_rebuilt/train.parquet` and
   `countdown_sft_clean_v1`; its checksum, model revision, data checksum,
   generation seed, retained groups, group sizes, token counts, and task
   identities are recorded.
2. **No leakage or relabel signal.** Every retained source belongs to the RL
   train split, has zero task-identity overlap with the held-out split, is
   informative under its original requested target, and contains no hindsight
   relabel metadata.
3. **Prospective delivery.** For seeds 1, 2, and 3, B2 first produces a complete
   60-step immutable accepted-slot schedule. A delivery-only preflight verifies
   schedule integrity and reservoir token support before any held-out endpoint
   is evaluated.
4. **Runtime gates.** Every E2c replay step uses exactly B2's accepted dataset
   slots, matches its auxiliary group count, uses no fallback, has a valid
   non-self reservoir source for every requested slot, preserves 128 optimizer
   rows and one optimizer step, and stays within the prospectively frozen 5%
   cumulative auxiliary- and total-response-token mismatch gates.
5. **Endpoint hygiene.** Only after every seed passes delivery are the fixed
   step-60 B1, B2, and E2c checkpoints evaluated on the same 384 held-out tasks
   with 16 samples per task and paired evaluation seeds.
6. **Decision.** Recompute and report the three seed-level tier-1 contrasts for
   mean@16 and standard observed-set pass@16 from retained task outcomes. If
   any delivery gate fails, E2c is inconclusive and no endpoint
   direction claim is made. Gates are not loosened after observing a failure.

## Primary metric and decision branches

- Primary: mean of the three within-seed tier-1 `mean@16` contrasts, E2c-B2.
- Safety: mean of the three within-seed tier-1 raw-outcome `pass@16` contrasts,
  E2c-B2; this is not the historical VERL bootstrap proxy.
- Context: B2-B1 and E2c-B1 contrasts on the same endpoints.
- Statistical unit: training seed; rollout samples and tasks measure endpoint
  precision but are not independent treatment replicates.

Interpretation is frozen as follows:

- B2 > E2c at matched delivered dose: evidence that the achieved-target
  direction contributes value on this pool.
- E2c reproduces B2's mean effect without its coverage loss: B2's mean effect
  is not direction-specific here, and sharpening is relabel-specific.
- Both auxiliary arms lose coverage versus B1: coverage damage is at least
  partly a generic auxiliary-update effect.
- Any failed delivery gate: treatment-delivery inconclusive; retain only the
  engineering/support finding.

## Stop conditions

- Do not launch while the shared GPU is occupied by another project.
- Do not inspect partial E2c held-out endpoints.
- Do not reuse E2/E2b endpoint values as E2c evidence.
- Do not seed E2b's buffer, change its 5% gate, or resume its stopped seeds.
- Stop E2c on the first prospective delivery violation and record the exact
  failure before designing any successor.

## Definition of done

E2c is complete when its code, tests, preregistration, reservoir manifest,
three complete B2 schedules, three replay delivery audits, fixed checkpoints,
paired held-out evaluations, analysis artifact, verdict, and handoff are all
present and internally consistent. The broader research phase is complete
when this verdict is propagated into the claim ledger and the remaining paper
repairs in `FINAL_ICLR_REVIEW_AND_COMPLETION_GUIDE_2026-08-07.md` are closed.
