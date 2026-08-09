# Recycling-package sharpening: literature synthesis + mitigation design

*2026-07-28. Companion to COUNTDOWN_ANALYSIS.md. Full ~30-paper review in
the session log; this doc records what changes our design and our paper.*

> **Currency note (2026-08-07):** the design below was run. What held:
> sharpening replicated at 3 seeds (tier-1 mean@16 0.278→0.324 while
> VERL bootstrap best@16 0.541→0.492). This logged field is a
> with-replacement coverage proxy, not standard unbiased pass@16; missing
> per-task outcomes prevent recomputation. The original moderate-looking point restored
> frontier-tier coverage while retaining ~60% of the mean gain, but it
> used faulty decay and is **suggestive, not a validated operating
> point**. What did NOT hold: (i) the deployed gate statistic is a
> decayed achieved-destination frequency, not the posterior-lookup
> p̂-gate designed here — and where both are measurable the true-p gate
> keeps ~all the value while the frequency heuristic pays a toll
> (results_gate_variants.json); (ii) the "dial" extension was refuted
> at 3 fixed-code designed-strength seeds (P-R1 — strong gating ≈
> recycling-off; countdown_reviewer_arms/PROVENANCE.md); (iii) at LLM
> scale a **higher-dose** live-group replay arm exceeds recycling on
> both meters (ARM B, 3/3 seeds). Because `ppo_epochs=2` increases the
> update dose for every live group, this is an upper bound on what
> generic extra updates can buy, not a dose-matched causal test of
> relabel direction. Entropy
> prioritization, KL-on-relabels, and dose-anneal were never run —
> still-open design space. The SFT audit affects 27/128 tier-0 tasks but
> finds zero tier-1 overlap, so the central sharpening comparison is
> unaffected; a clean 101-task tier-0 numerical reanalysis remains
> blocked by missing per-task outcomes. Paper §6.8–6.9 is authoritative.

## 1. The finding is novel — and now precisely locatable in theory

No published work documents hindsight-relabeling-induced pass@k loss in
RLVR. The nearest neighbors (RLEP, ExGRPO, H2SD, CodeIt, HIR) all report
mean-accuracy gains and never measure coverage. Our mean@16-up /
bootstrap-best@16-down result with a clean no-relabel baseline is a potentially publishable
connection — and it sits at the junction of three formal results:

- **GCSL bound (1912.06088):** relabel-and-imitate gives exact
  conditionals under the ACHIEVED-goal marginal — optimizing a lower bound
  whose gap is exactly the achieved-vs-target marginal mismatch. Our P6
  proof is the "conditionals exact" half; GCSL names the half we ignored.
- **Curation theorem (2407.09499):** iterating self-training on
  verifier-filtered own outputs provably concentrates on the curation
  reward — here, REACHABILITY. Precision (mean@16) preserved, recall
  (bootstrap best@16 proxy) spent — qualitatively related to the MAD-paper signature (2307.01850).
- **Binary-reward degeneracy (2605.02375):** the verifier cannot prefer
  the diverse valid policy among all valid ones; only an anchor
  (KL-to-base) or an objective change (pass@k) resists collapse.

Also: HER's classic bias corrections (ARCHER/USHER) target STOCHASTICITY
bias — inapplicable here (our verifier is deterministic). The bias we hit
is the marginal-shift bias, documented in Zhao et al. (1905.08786), whose
fix — entropy-prioritized relabeling over achieved goals — ports directly.

## 2. Convergence: our own algebra predicted this (and gives the fix)

Independent of the literature, the mass identity u(p) = pass@N − pass@1
says training signal is ZERO at p→1. A relabel to the value the model just
produced is a task the model already solves — p≈1 by construction — so
recycled updates land exactly where u says training is worthless, and the
softmax pays for them out of the exploration tail. The CPU testbeds never
saw this because their relabels landed at genuinely hard prefix tasks
(low p); Countdown's land at self-achieved outputs (high p).

**The unification: the teacher and the recycler should share one admission
rule — the derived utility, applied to the relabel DESTINATION.** This is
the same "saturation gating" the strongest empirical paper in the review
(2606.15455) reaches from the opposite direction (updating only
zero-success problems lifts pass@256 above base).

## 3. E-LLM-2b mitigation design (historical pre-run record)

Ranked by theory-backing × cost; the first two are the designed contrast:

| arm | mechanism | theory | cost |
|---|---|---|---|
| **utility-gated relabels** | admit a relabel only if the destination's estimated pass rate is in the band (posterior lookup; reject p̂ > 0.5) + entropy-prioritize rare achieved targets (Zhao et al.) | our P1 + 2606.15455 + 1905.08786 | ~15 lines |
| **KL-on-relabels** | KL-to-base penalty masked to relabeled rows (β≈0.05) | Dymetman degeneracy; KL-Cov | ~1 line + mask |
| dose-anneal | cap_t ∝ (1 − live success rate of the tier) | mixing-stability theorems (Bertrand/Seddik/MAD) | ~5 lines |
| (fallback) Clip-Higher | clip_ratio_high 0.28 | DAPO empirics | config flag |

Runner-up recorded: importance-weight relabeled rows by
exp(logπ(y|new prompt) − logπ_gen(y|old prompt)) clipped — LUFFY/USHER
specialized to prompt-swap; both logprobs already exist in the loop.

**Completed-arm verdict.** The fixed-code designed-strength gate did not land
on a mean-versus-coverage dial: over three seeds it was statistically
indistinguishable from recycling-off, and its single-seed
coverage-above-baseline reading did not replicate. The earlier under-gated
point therefore remains a hypothesis for a corrected moderate setting. The
higher-dose live-group replay arm improved both mean@16 and bootstrap best@16, but its
larger update dose makes it a bound rather than a relabel-specific control.

## 4. What this does to the paper

The replicated sharpening tradeoff is a self-contained contribution
candidate. The mitigation is not yet a contribution: the fixed-code strong
gate failed, and the only favorable under-gated point used faulty decay. The
defensible paper statement is therefore: "recycling has a measurable safety
cost; the derived utility motivates an admission rule, but a corrected
moderate operating point remains unvalidated." This composes with (rather
than replaces) the safety-paper framing, while keeping the failed mitigation
in the limitations or appendix rather than presenting it as a solved result.
