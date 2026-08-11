# E2c comparator-reuse audit

**Audited:** 2026-08-10, before seed-3 training, reservoir collection, E2c
training, or any E2c held-out generation.  
**Machine-readable authority:** `E2C_LAUNCH_READINESS.json`  
**Verdict:** B1/B2 seeds 1 and 2 are valid reusable comparators for E2c.

## What was checked

For each completed comparator, the normalized logged Hydra configuration passes
59/59 frozen checks covering the source model, optimizer and batching, rollout
budget/decoder, MaxRL settings, hindsight assignment, replay exclusion, data
paths and seed, exact verifier, output/run identity, validation exclusion,
checkpoint schedule, fresh-start state, and final step. The audit also requires
both Hugging Face config and weight artifacts and fingerprints their contents.
The added checks prove the full frozen scheduler configuration and the exact
ordered optimizer-step sequence 1--60.

For B2, all 60 dose rows are ordered steps 1--60, retain 128 optimizer rows,
request no more than eight groups per step, contain unique fixed dataset slots,
and agree exactly between per-group and summarized response-token metadata.
Ten integer delivery fields in every schedule row also match the independently
printed scalar metrics in its corresponding training-step log.

The verifier/reward file has SHA-256
`99c04d4a4914170a528c67337aec364e7410074c552d9848c714f78c0f9e2312`.
Its local modification/change time predates the first comparator launch, and
all four logs name its path and `compute_score` entry point.

## Immutable fingerprints

| run | training log SHA-256 | step-60 model SHA-256 |
|---|---|---|
| B1 seed 1 | `02fcb4693e7bd11bd41f3fdf216176ba81bfcea79bb7f4313c9ceac2739f9a8a` | `3b9508fa1deadaf963db0c31008a9ed4bf69f007915bf3a4c3d2060e5428fcae` |
| B2 seed 1 | `4f8c6395658297d73627051568f4a50b456e68e3405d01ce8ee6a3af0f333ee3` | `2669af5f8398d7c1e5be549372411b0f4de981510ccd67059b421c49a7e2b74e` |
| B1 seed 2 | `e119471dc35dce15318cd1276ea952e265c3a62e71cda5456313c7e415b95346` | `9053700be12a2c9646c9a68d0a89dd70878ab3e4a8962bd12dcf9491f6ea5617` |
| B2 seed 2 | `b35dceaac74ede15cbbc9a0bfd502e7ea65777b80f83c59ce8560875b6e61e9f` | `6d44c265a0062352c86b3f2bf72b340cff57cc07c76abe535562db6e1ce95fb0` |

All four checkpoint `config.json` files are 859 bytes with SHA-256
`283834b57c6e55af57e59b007df3bfcaf2f898dbb22fb535a46d224b73acb0cd`.
Each weight artifact is 1,447,317,080 bytes.

| B2 schedule | rows | accepted groups | SHA-256 |
|---|---:|---:|---|
| seed 1 | 60 | 329 | `119f2c62d4e307c6f68ba14e737bcdbca598bd8b510492867c7037047720d310` |
| seed 2 | 60 | 308 | `a418e9430b4331400c05051b6b3a3da8bc2d82b070f4a95f2f22581c1a2fe5a0` |

## Scope of the verdict

This audit proves comparator identity and protocol parity; it does not inspect
or rehabilitate any historical E2/E2b endpoint. B1/B2 seed 3 must pass the same
gate after training. E2c held-out evaluation remains forbidden until the
reservoir preflight and all three runtime delivery audits pass.
