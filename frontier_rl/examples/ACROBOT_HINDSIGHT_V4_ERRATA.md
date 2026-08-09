# Acrobot hindsight V4A errata and execution note

Date: 2026-07-22

## Stage-A decision

The independently verified V4A artifact passed its integrity, source/runtime,
schedule, accounting, and recomputation checks. The registered fallback rule
selected `U*=250` optimizer updates. The feasibility decision nevertheless
**failed** because gate 3 required at least ten positive, finite,
one-to-one, nonmutating hindsight previews in every run. Exactly three of the
nine runs had only `8`, `5`, and `6` previews. Every other registered gate
passed, including the `3.452702`-hour projected serial runtime for the planned
90-run factorial.

Consequently, V4B was not authorized and was not run. V4A used hindsight scale
zero throughout and was an effect-blind feasibility study, so this stop is not
evidence for or against hindsight efficacy.

The report's top-level `all_checks_passed: true` means that the independent
verifier successfully validated the immutable inputs and recomputed the saved
decision. It does **not** mean that all feasibility gates passed. The deciding
fields are `gates.all_pass: false` and
`stage_b_factorial_authorized: false`.

SHA-256 provenance:

- V4A source/runtime lock:
  `b19488783e1adba8cbac44ce8256c725a4470d8108c1192f9491ecc4882f1d8c`
- V4A feasibility artifact:
  `69b827dc425014f3b568186981e9c24d95158c72653125e0ade181272def2891`
- independent verification report:
  `c633e09df8e056f1589e631ff4d311913e1ac5594c3647790acc4b05990fca88`

## Analyzer invocation correction

The direct file-path analyzer command recorded in the frozen V4A lock fails
with `ModuleNotFoundError: No module named 'frontier_rl'`: Python places the
script directory, rather than the repository root, first on the import path,
and the analyzer imports `frontier_rl` at module load. No locked V4 source or
protocol file was changed after observing this defect.

From the repository root, the equivalent working invocation is:

```bash
/tmp/curriculum-maxrl-gym/bin/python -m frontier_rl.examples.analyze_acrobot_hindsight_v4 \
  frontier_rl/examples/acrobot_hindsight_v4a_feasibility.json \
  --lock frontier_rl/examples/ACROBOT_HINDSIGHT_V4A_LOCK.json \
  --output frontier_rl/examples/acrobot_hindsight_v4a_verification.json
```

This invocation changes only module resolution. It runs the same hash-locked
analyzer and produced the independently verified report identified above.
