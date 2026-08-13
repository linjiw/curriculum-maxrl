# GATE-DR: corrected-code utility-gate dose–response — preregistration

**Status:** FROZEN 2026-08-13 by the commit containing this line, which includes the
Hopper training-smoke receipt below; changes only via dated amendment section.
**Smoke receipt (outcome-blind engineering check, not evidence):** Slurm job 9357902
completed one full MaxRL training step + 3-tier validation on one A100.80gb:
pg_loss .010, grad_norm .754, entropy .151, advantages in [−1, 7], step-1 val
mean@4 tier0/1/2 = .412/.119/.006 (512 examples per tier), timing ≈ 6.0 s/train
step + 37.5 s/validation. No endpoint of any GATE-DR arm was observed.
**Environment:** GMU Hopper, one `A100.80gb` per run, env `/scratch/lwang44/envs/maxrl-train`
(python 3.11, torch 2.7.0+cu128, transformers 4.56.2, ray 2.49.2, setuptools<81,
torchdata 0.11.0, math-verify 0.8.0), HF rollout — mirroring the lab runtime's pins.
flash-attn is an inert import shim (the real wheel needs GLIBC ≥2.32, unavailable on
Hopper nodes); attention runs via SDPA, `use_remove_padding` stays false, and the shim
raises loudly if any flash/rmpad kernel path is ever invoked, so it cannot silently
alter computation.
**Launcher:** `hopper/countdown_hopper_gate.sh`, a byte-faithful derivative of the frozen
`verl_integration/countdown_rtx5090.sh` adding exactly one additive parameter
(`HINDSIGHT_GATE_MAX_P` → `+algorithm.hindsight.gate_max_p`); the frozen original and
`vendored/hindsight.py` are untouched (both are pinned in `E2C_CODE_MANIFEST.json`).
**Relationship to E2c:** none. Different machine, different study, no shared artifacts;
E2c remains local and frozen.

## Question

The paper currently states: "An under-gated operating point recovered frontier coverage;
corrected-code validation of the useful setting remains open." ARM A (P-R1) showed the
corrected-code gate at its designed strength (`gate_max_p=0.5`, decay 0.9) rejects ~93%
of relabels and is indistinguishable from recycling-off — refuting the dial at that
setting. **Open question: does any weaker corrected-code setting reproduce the useful
trade (part of the ungated mean gain retained, coverage at/near the no-recycling
baseline), or is the corrected gate's dose–response effectively binary?**

## Arms (4) × seeds (3) = 12 runs, all corrected code

| arm | HINDSIGHT_ENABLE | UTILITY_GATE | GATE_MAX_P |
|---|---|---|---|
| `b1h` (no-recycling reference) | false | — | — |
| `g0` (ungated recycling reference) | true | false | — |
| `g085` (weak gate) | true | true | 0.85 |
| `g070` (moderate gate) | true | true | 0.70 |

Grid rationale: designed 0.5 over-gates (93% rejection, refuted); 1.0 ≡ no gate; the
open region is between. Seeds 1–3, paired across arms (same seed ⇒ same data order).

All other settings are byte-identical to the frozen E2c B-stage invocation:
STEPS=60, TRAIN_BATCH=8, N_ROLLOUTS=16, LR=1e-5, MAX_RESPONSE_LENGTH=128,
ROLLOUT_MICRO_BATCH=128, LOGPROB_MICRO_BATCH=8, PPO_MICRO_BATCH=4,
HINDSIGHT_MAX_GROUPS=8, HINDSIGHT_ONE_TARGET=false, REWARD_MANAGER=dapo,
FILTER_OVERLONG=false, GRADIENT_CHECKPOINTING=false, model =
`countdown_sft_clean_v1` (sha-verified copy), data = `countdown_v2_rebuilt`.
Deviations from the B-stage: VAL_ON_LAST_STEP=true with VAL_N=16 (step-60
validation in-run, the ARM-A endpoint path), SAVE_FREQ=60 with hf_model retained
for later raw-outcome re-evaluation, and attn_implementation=sdpa in place of
flash_attention_2 (the prebuilt flash-attn wheel requires GLIBC ≥2.32; Hopper
compute nodes are older). SDPA is uniform across all 12 runs, so within-study
paired contrasts are unaffected; absolute levels may differ slightly from the
local B-series, which is one more reason all references are on-Hopper.

## Endpoints

- **Primary:** step-60 validation tier-1 mean@16 and tier-1 VERL bootstrap best@16
  coverage proxy (extraction identical to ARM A: per-tier val metrics from ray worker
  logs by POOL_TAG; the proxy is never called standard pass@16).
- **Dose verification (manipulation check):** per-run gate reject fraction from
  `dose_accounting.jsonl`. Expected strictly decreasing rejection from g070 → g085 → g0(0).
- **Secondary:** tier-2 mean@16 and proxy; accepted-relabel counts per step.

## Decision rules (frozen)

Let Δmean(arm) and Δproxy(arm) be seed-paired differences vs `b1h`.

1. **Transfer gate:** if Δmean(g0) is not positive in at least 2/3 seed pairs, the base
   recycling phenomenon did not transfer to this environment; declare
   **INCONCLUSIVE-BY-TRANSFER** and stop — no gate conclusions of any kind.
2. A setting g **reproduces the useful point** iff BOTH:
   (a) paired mean-kept fraction Δmean(g)/Δmean(g0) ≥ 0.40 with all 3 seed pairs positive;
   (b) Δproxy(g) ≥ −0.005 with ≥ 2/3 seed pairs ≥ 0.
3. If no setting satisfies rule 2 and both settings' reject fractions exceed 0.85,
   verdict: **corrected gate is effectively binary at practical strengths** — the
   paper's "remains open" sentence is replaced by that negative finding.
4. If reject fractions are graded but rule 2 still fails, verdict: **graded dose without
   a useful operating point** — reported as such.
5. n=3: no p-values are claimed; report all per-seed values, descriptive mean ± sample
   SD, and the paired counts above. This study is labeled calibrated small-n throughout.

## Paper consequence (either direction)

One sentence in §Evidence/Limitations replaces "corrected-code validation of the useful
setting remains open," citing this file and the run artifacts. A positive rule-2 outcome
is a validated operating point at its stated strength on this environment; any other
outcome is reported with equal prominence.

## Artifacts

Per run: full stdout log, `dose_accounting.jsonl`, step-60 hf_model checkpoint,
extraction JSON. Archived under `/scratch/lwang44/curriculum-maxrl-runtime/checkpoints/gatedr_*`
and rsynced to the lab machine under `curriculum_maxrl/gate_dr/` before analysis.
Analysis script committed with results; no endpoint inspected before all 12 runs and
the manipulation check are complete.

## Amendments

- **2026-08-13 (infrastructure retry, outcome-blind):** array 9357906 tasks 0–2
  (`b1h` seeds 1–3) completed; tasks 3–11 (all hindsight-enabled arms) failed at
  import time — `verl/utils/hindsight.py` and `curriculum.py` were rsync-copied as
  dangling symlinks into the Hopper runtime. Both replaced with the identical
  vendored file contents (source: repo `verl_integration/`, same bytes the local
  symlinks resolve to); import verified; tasks 3–11 resubmitted unchanged as array
  9357948. No endpoint of any arm was observed before or during this repair; the
  `b1h` logs remain unopened.
