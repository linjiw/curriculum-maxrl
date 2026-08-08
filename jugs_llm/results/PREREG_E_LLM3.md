# Pre-registration: E-LLM-3 (Jugs) — BINDING (2026-08-02)

Status: BINDING on commit. Written after feasibility
(`jugs_llm/feasibility_*.json`), before any RL cell starts. Disclosure
of everything seen at writing time: the three feasibility landscapes
(40 tasks/tier, N=16 sampling) and sampled SFT-checkpoint generations;
no RL step has run on this pool with any estimator.

## Question

Do recycling-induced sharpening and its derived gate generalize from
one-shot expression search (Countdown) to stateful sequential planning
(Jugs), and does the frontier teacher pay iff the pool has
unlearnable-at-budget tiers?

## Fixed design (independent of feasibility)

- Pool: `~/data/jugs_v1` (jugs_llm/pool_v1.jsonl; 160 train + 40 test
  per tier, t0–t4, 200/200 unique per tier, disjoint min-moves bands;
  two-shot exemplars baked into every prompt — the SFT route was
  measured and REJECTED: it collapsed the policy, t0 pass@16 .475→0;
  documented negative in DESIGN_E_LLM3).
- Model: SmolLM2-360M-Instruct, raw instruct checkpoint, two-shot
  prompts (decided by feasibility: t0 .058/.475 in-band; Qwen-0.5B
  worse everywhere).
- N=16 rollouts, batch 64, 60 steps, lr 1e-5, temperature 1.0,
  val n=16 at temperature 0.6 (identical to E-LLM-2b).
- Arms (5): B1 no-recycling / B2 recycling-ungated / B3 recycling-gated
  (gate_max_p=0.5, no per-domain tuning — the transparency claim) ×
  {uniform}; plus T1 teacher-uniform-recycling-off and T2
  teacher+gated-recycling. 3 seeds for B1/B2/B3; T1/T2 single-seed
  labeled (budget).
- Meters: per-tier val mean@16 and pass@16, dead-group fraction,
  relabel yield, gate rejection rate, policy entropy. Noise floor
  measured FIRST: 5 repeated evals of the warmstart checkpoint; no
  claim inside 2× that floor.
- Analysis contract: primary contrast = B2 vs B1 (sharpening) and B3 vs
  B2/B1 (gate) on the tier named in P-J1 below; teacher contrasts are
  secondary. Paired per-seed deltas; sign counts; no p-values below
  n=3 pretensions — orderings and signs are the registered outcomes.

## Predictions (BOUND to the 2-shot landscape, 2026-08-02)

Measured starting landscape (SmolLM2-360M 2-shot, N=16):
t0 pass@1=.058/pass@16=.475, t1 .006/.100, t2–t4 0/0;
relabel_yield_on_fail .638/.723/.731/.684/.616 (t0..t4).

- **P-J1 (sharpening replicates): tier = t0.** t0 destinations are the
  small jug amounts (caps 3–20, so ≤ ~20 distinct values, heavily
  reused across the 160 train tasks) and the model already reaches
  them (.475 pass@16, yield .64) — the saturated-destination
  configuration. Prediction: B2 (ungated recycling) gains mean@16 and
  loses pass@16 on t0 relative to B1, both signs in ≥2/3 seeds.
  Secondary (directional, not confirmatory): same signs on t1.
- **P-J2 (gate transparency + recovery):** B3 (gate_max_p=0.5, the
  Countdown setting, zero per-domain tuning) restores ≥ half of B2's
  t0 pass@16 loss while keeping ≥ 40% of B2's t0 mean gain; on t3–t4
  (destinations far from saturated at start) B3's relabeled dose is
  within 20% of B2's (transparency), unless the gate's posterior
  saturates them during training — report the rejection-rate
  trajectory either way.
- **P-J3 (creation in the dead zone; the frontier-heavy config at LLM
  scale): unlearnable tiers = t2, t3, t4** (pass@16 = 0 at start, yield
  .62–.73). Prediction: recycling arms (B2/B3) end with t2 pass@16 > 0
  in ≥2/3 seeds while B1 stays at 0 — signal creation where sampling
  has nothing (the §6.2 pattern; Countdown's version was blocked by a
  saturated pool). t3/t4 ignition is NOT predicted (relabels land
  mostly below tier; compounding upward within 60 steps is a stretch)
  — any t3+ ignition is a bonus finding, reported as such.
- **P-J4 (relabel-richness, exploratory):** live relabel yield in-run
  stays above Countdown's 72–89% band on parse-valid failures. Report
  only; no directional claim bound.

Noise floor protocol: 5 repeated evals of the raw 2-shot checkpoint on
the 200-row test parquet BEFORE the first cell; per-tier mean@16 and
pass@16 SDs recorded in `jugs_noise_floor.json`; no claim inside 2×
that floor.

## Deviations pre-declared

- v1 relabels use FINAL-state amounts only (response text verified
  as-is); the prefix map is a v2 ablation.
- Multi-candidate relabels: gate-on arms pick the least-saturated
  destination by the gate's own posterior; gate-off arms take the
  first. (Countdown had single candidates; this is new surface,
  disclosed.)
- Think-text target swap identical to Countdown B2-repair.

## Feasibility results (verbatim)

```
SmolLM2-360M-Instruct_2shot:
  t0: pass1=0.0578 pass16=0.475 relabel_yield_on_fail=0.638
  t1: pass1=0.0063 pass16=0.100 relabel_yield_on_fail=0.723
  t2: pass1=0.0    pass16=0.0   relabel_yield_on_fail=0.731
  t3: pass1=0.0    pass16=0.0   relabel_yield_on_fail=0.684
  t4: pass1=0.0    pass16=0.0   relabel_yield_on_fail=0.616
Qwen2.5-0.5B-Instruct_2shot: t0 .0031/.050 yield .077; t1-t4 0/0
jugs_sft_v1 (0-shot):        t0 0/0 (COLLAPSED); t1 .0031/.05; yield .85-.93
```

## Cell order (serial, one A10G)

noise floor → B1 s1 → B2 s1 → B3 s1 → B1 s2 → B2 s2 → B3 s2 →
B1 s3 → B2 s3 → B3 s3 → (T1, T2 single-seed if budget allows).
Interleaved by seed so an early failure loses balanced data, not an arm.
