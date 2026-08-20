# P0 autoresearch loop — group-law flip

Started: 2026-08-20
Iteration cap: 25

## Goal

Produce a freeze-ready, fail-closed P0 experiment that changes only the
curriculum score functional on the MAZE-SCORE substrate, execute it blindly,
and update the ICLR manuscript only after the registered verdict exists.

## Metric

Before evidence launch, all conjunctive engineering gates must pass:

1. shared-posterior score isolation is contract-tested;
2. full decision-rule power and powered-for effect are recorded;
3. trainer/config/analyzer/campaign schemas agree;
4. synthetic complete/missing/corrupt campaigns fail or pass as specified;
5. a non-evidence full-arm smoke is terminal and digest-verified;
6. preregistration, analyzer, source, environment, and launcher hashes are
   frozen in a commit before any evidence seed is submitted.

After launch, success means one immutable 2x20 matrix, one analyzer execution,
and manuscript/artifact updates matching the earned evidence tier. The
scientific endpoint is never an iteration metric and is never used to tune.

## Verify

- Unit and contract tests listed in root `AGENTS.md`.
- P0-specific synthetic campaign tests and local launcher tests.
- Outcome-blind remote import/full-arm smoke receipts.
- Clean content-addressed evidence bundle validation.

## Keep/discard rule

Keep a change only if it strengthens isolation or fail-closed behavior without
altering the frozen substrate. Discard any change that reads an endpoint,
weakens a gate, introduces a second algorithmic arm difference, or requires a
post-outcome choice.
