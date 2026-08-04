# Beyond GSM8K — the post-2×2 experiment roadmap

*Drafted 2026-07-23 from a two-agent research sweep (datasets/tasks + harness/
related-work). GSM8K 2×2 is experiment E-LLM-1, not the finale: it tests
channels 1+3 (teacher, objective safety) but not channel 2 (hindsight), which
is our most distinctive claim. This doc picks the next experiments to close
that gap on the same A10G.*

## The headline opportunity: Countdown target-relabeling (E-LLM-2)

**Task**: Countdown/24-game — given numbers {a,b,c,d} and a target, produce an
arithmetic expression using each number once. TinyZero's setting; 490k-row HF
pool (`Jiayi-Pan/Countdown-Tasks-3to4`) + procedural generators (TinyZero's
`gen_dataset`, reasoning-gym's countdown) with a real difficulty dial
(operand count 2→6, target range, value range).

**Why it's THE hindsight experiment**: a failed trace that uses the numbers
correctly still *evaluates to some value v*. Relabel `target := v` and the
same verifier returns 1.0 — an **exact** relabel (P6 contract 1) at zero
generation cost. The conditioning rewrite (contract 2) is a template edit:
swap the target number in the prompt. Both hindsight contracts satisfiable
exactly, at LLM scale, with the repo's own verifier.

**Novelty check (agent-verified against the arXiv Countdown×RL feed)**: no
2024–2026 work relabels Countdown targets. Credit-assignment variants exist
(PAPO, d-TreeRPO); hindsight-in-RLVR does not. The closest hindsight
relatives (HiR, ECHO, SAC-GLAM) are in instruction-following/agents, not
verifiable math. **The niche is open.**

**Regime fit**: TinyZero found 0.5B base models fail on 3–4 operands; 360M is
expected pass@16 ≈ 0 on the standard pool = exactly our beyond-frontier-heavy
regime, where the CPU result is categorical (V5: teacher+hindsight 0.93 AUC
vs literally 0.00 for every pure sampler — because relabeling *invents the
curriculum below the pool*). The 2-operand tier is the bootstrap rung.

**A10G fit**: better than GSM8K — short prompts, ≤512-token responses, N=16
comfortable.

**Design (pre-register before running)**:
- Pool: fixed 10k-row slice stratified 2/3/4 operands (fixedness is the
  hindsight-compounding regime per EVIDENCE.md) + held-out eval slices/tier.
- 2×2×2: {maxrl, grpo} × {frontier teacher, uniform} × {hindsight on, off}
  — hindsight-on cells relabel dead groups (target := achieved value,
  prompt rewritten) capped per step, posterior sees requested-task evidence
  only (V4 rule).
- Predictions to freeze: (i) hindsight-on ignites the 3–4-operand tiers that
  stay at 0 for every hindsight-off cell (V5 pattern); (ii) teacher adds
  waste-avoidance on top (P1); (iii) grpo cells collapse under the teacher
  (H6); (iv) relabels decay over training (ignition→silence signature).

## Second track: streaming teacher on reasoning-gym (E-LLM-3)

`open-thought/reasoning-gym` (NeurIPS 2025 spotlight): 105 seeded procedural
generators with per-attribute difficulty levels and a `CurriculumExperiment`
API (`update_difficulty(name, increment|decrement)`). This is the missing
LLM test of `frontier_rl/streaming.py` (kernel posterior over a continuous
difficulty axis — CPU-validated to match discrete bins exactly).

- Start composite: `chain_sum` + `spell_backward` (+ `countdown` for overlap
  with E-LLM-2). These arithmetic/algorithmic families are where a 360M
  frontier is non-degenerate (RG's own 3B gets 0.0 on games).
- Built-in baseline to beat: RG paper §5's 70%-threshold curriculum (their
  result: +13 to +40 points over uniform — a published number our teacher
  must clear on the same infra).
- Integration cost: live procedural sampling into verl's parquet pipeline
  (buffer-per-step shim).

## Third track: sharpest related-work contrast (E-LLM-4, optional)

Knights & Knaves, DUMP's exact setting (fixed 6.2k pool, 7 difficulty
levels). Beating DUMP's UCB teacher on its own dataset with our
posterior+advantage-mass teacher is the crispest head-to-head available.
Risks: no exact hindsight form; 360M frontier may be degenerate at both ends
(25% random floor at 2ppl, 0 at 5ppl+). Decision rule: probe pass@16 by
level first; if degenerate, step model up to Qwen2.5-0.5B-Instruct or drop.

## Differentiation map (from the related-work sweep)

All 2025–26 difficulty-adaptive RLVR work — DUMP (UCB over buckets), AdaRFT
(scalar target-difficulty tracker), SEC (bandit over categories), RG §5
threshold, fg-expo/sGPO/CGPO — is **sampling-policy-only** over discrete
buckets or one scalar. None has: (a) a *derived* utility (sample ∝
pass@N−pass@1 = the estimator's exact advantage mass, vs their heuristic
bandit rewards); (b) a streaming kernel posterior driving a generator; (c)
hindsight relabeling as a signal-creation channel; (d) the objective-safety
result (curricula amplify GRPO collapse). GSM8K 2×2 defends (a)+(d);
Countdown defends (c); reasoning-gym defends (b).

## Free upgrade available today

`lime-nlp/GSM8K_Difficulty` annotates every GSM8K problem with a solve rate
(128 samples of a 7B model) — a per-problem difficulty coordinate for the
CURRENT experiment at zero migration cost (rank-correlated, not exact, for
360M). Use for post-hoc analysis of the 2×2: did the teacher's visited-set
walk the annotated difficulty axis?

## Sequencing (A10G-hours)

1. **Now → ~7/26**: GSM8K 2×2 completes (channels 1+3 at LLM scale).
2. **Prep in parallel (CPU)**: Countdown data prep + relabel verifier port +
   pass@16 probe script; pre-registration doc.
3. **E-LLM-2 (Countdown hindsight)**: the flagship — first exact-hindsight
   RLVR result. ~3–4 days of A10G at GSM8K-like cell costs (shorter
   responses → likely faster).
4. **E-LLM-3 (streaming/RG)** after, or interleaved if E-LLM-2 stalls.
5. **E-LLM-4 (K&K/DUMP contrast)** only if the probe shows a usable frontier.

## Harness findings (second sweep) — and what's already done

- **vllm 0.8.5.post1 INSTALLED and verified** (torch 2.6.0+cu124 pin matched
  exactly; numpy settled at 2.2.6; full verl import gauntlet incl.
  vLLMRollout passes). Our verl fork already ships sleep-level-1
  (`enable_sleep_mode`/`free_cache_engine`); config for colocated 360M:
  `rollout.name=vllm, gpu_memory_utilization≈0.45, enforce_eager=True,
  free_cache_engine=True`. Expected 5–10× on the generation phase (1250s of
  our 1450s step) → **~3–4× end-to-end**; our 64×16=1024 concurrent
  sequences are continuous batching's best case. Smoke-test on our next GPU
  window, before cell 3.
- **Stay on verl**: TRL's GRPOTrainer hard-codes group-norm advantages and
  has no sampler hook (both our deltas are private-API subclasses there);
  OpenRLHF/open-r1 are multi-GPU-shaped; nano-GRPO loops would re-pay the
  infra cost we already paid.
- **Eval upgrade**: GSM8K-Platinum is only 1,209 rows — post-vllm, evaluate
  the full platinum set at checkpoints instead of our 256-row slice
  (pass@{1,4,8,16}); lighteval has vllm-backend pass@k if we want citable
  external numbers.
- **Two one-flag baselines to add** (pre-empt the obvious reviewer asks):
  `utility=learnability` ≡ LILO (arXiv:2502.12272 — per-prompt p(1−p)
  rejection sampling, our N=2 special case) and a DUMP/SEC-style
  |advantage|-bandit. Our FrontierTeacher already parameterizes utility.
- **Sharpened differentiation** (deeper sweep): LILO is the closest
  published method (per-prompt p(1−p)); DUMP/SEC/TAC are bucket/domain-level
  bandits; PKPO/Pass@k-Training modify the objective but keep uniform
  sampling; GRESO/DAPO-dynamic-sampling only CULL dead prompts (avoidance,
  no allocation, no creation). All hindsight-for-LLM work (AgentHER, HSL)
  relabels agent trajectories with an LLM judge — **exact-verifier
  relabeling inside the RL loop remains unoccupied**, now doubly confirmed.
- **Mixed-pool gap confirmed**: all adaptive mixing (TAC, DUMP, SEC) is
  domain-level; nobody runs a per-prompt teacher across a heterogeneous
  pool. GSM8K + 2–3 reasoning-gym families with a label-free per-prompt
  teacher = E-LLM-5 candidate after the flagship.

## E-LLM-2 staging status (done 2026-07-23)

Countdown is STAGED in the maxrl repo (`curriculum_maxrl/countdown/`):
tiered 10k pool generated (~/data/countdown, tiers = 2/3/4 operands, all
solvable by construction), strict-binary verl reward fn, exact-relabel
helpers (roundtrip verified: failed trace's achieved value scores 1.0
against the rewritten prompt), and a pre-registered pass@16 probe with
frozen decision rules. `frontier_rl/adapters/countdown_llm.py` carries the
same contracts framework-side (tests green). Remaining before launch:
vllm smoke on our GPU window → probe → hindsight-in-verl wiring (relabel
dead groups inside the trainer, the one new engineering piece) → 2×2×2.
