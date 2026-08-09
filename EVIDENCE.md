# The evidence, reorganized: where the strength actually comes from

> **Currency note (updated 2026-08-09):** this synthesis predates the
> balanced factorial. The "H6 estimator main effect, 9/9 runs,
> p=0.0079" claim below **failed the first factorial endpoint and is
> retracted** — see
> `maze_gpu_factorial/FACTORIAL_VERDICT.md` and paper §6.3b for what
> survives (exact-rung 10/10; the external execution record says covAUC was
> specified before wave 2, which landed **6/6 per sampler, p=.031 each**;
> the locking commit is not vendored; averaging the two
> correlated sampler observations within seed gives 6/6 positive wave-2
> blocks and 12/12 positive independent block averages descriptively across
> waves). Easy-band localization did not earn the same claim: wave 2 has 4
> positive, 1 tied, and 1 negative block average, with an interval crossing
> zero, so it is suggestive only.
> Also superseded here: the "gate strength is a dial"
> reading — the designed-strength sweep refuted it (P-R1, 3 seeds;
> `countdown_reviewer_arms/PROVENANCE.md`); and the "+0.22 hindsight
> gain" framing — at LLM scale a **higher-dose** live-group replay
> control exceeds recycling on both logged meters (ARM B, 3/3 seeds), providing
> a higher-dose alternative but not separating update dose from update
> direction. Sections
> quoting these as established are historical. The completed GSM8K `g3p`
> cell missed its registered run-mean treatment-delivery gate by 0.00148
> (0.601480 versus required `<0.60`), so its interaction is inconclusive by
> design. After the paid-probe integration, the registry generator emits and
> checks exactly 562 records, including 441 Acrobot records; the generator owns the
> exact totals, so prose should not maintain a competing count. A
> later audit found neighboring cross-domain RNG-root reuse in V3; its positive
> contrast remains historical/descriptive rather than clean paired
> confirmation. The fresh fixed-pool V2 tournament completed all 9 development
> and 60 confirmation runs. Its registered `u_16-p(1-p)` primary was
> `+0.0480336884` [95% bootstrap CI `0.0209366676, 0.0738485654`], exact
> sign-flip `p=0.0033607483`, with 15/20 positive pairs, clearing both its
> frozen +.01 point-estimate and `p≤.05` filters. The result is P+/U+
> score-shape evidence in one fixed eight-threshold Acrobot pool and one H64,
> 640-parameter practical-MaxRL learner at `N=16`, not a full ProCuRL, SFL,
> PLR, PAIRED, ACCEL, or ALP-GMM comparison. It has no held-out-task
> generalization test or prospective power calculation, and the exact test
> assumes paired-sign exchangeability; full details are in
> `frontier_rl/examples/ACROBOT_CURRICULUM_TOURNAMENT_RESULTS.md`. The
> A stored Countdown identity summary marks 27/128 tier-0 tasks SFT-exposed,
> leaves a nominal 101-task subset, and reports zero measured tier-1 overlap;
> missing source manifests prevent independent recomputation, and a numerical
> clean-subset reanalysis is blocked by missing per-task outcomes.
> The final paper rebuild and public-PDF synchronization are complete. The
> 13-page ICLR wrapper fits its main text within nine pages; references begin
> later on page 9 and the appendix begins on page 10.
> This content-addressed release is frozen and verified. The Git commit that
> contains these files is the authoritative repository publication record;
> the manifest intentionally does not embed its own self-referential hash.
> Remaining evidence blockers are the external wave-2 checkpoint trajectories,
> the complete three-seed B1/B2 records, and the per-task Countdown outcomes.
> The V2 lock and digests bind source, runtime, gate, and artifacts internally,
> but no immutable public pre-execution commit in this checkout establishes
> their timing.

> **Paid-probe update (2026-08-09):** all 12 development and 320/320
> confirmation runs completed and passed the frozen accounting checks. The
> registered `u_16-ProCuRL` fixed-paid-AUC contrast was `+.004894`,
> `t(79)=1.9773`, `p=.05149`; it is below the `.02` SESOI and unsupported.
> The probed arms spent about 93.2% of paid transitions on probes. Fixed-paid
> AUC was `.65149` for ordinary uniform, versus `.33771` for ProCuRL-env,
> `.33942` for probe-sham uniform, and `.34261` for `u_16`. This is evidence
> that probe cost dominated at the frozen actor-only fixed-pool refresh
> cadence, not that full PPO ProCuRL is inferior or that the same ordering
> holds under cheaper probing. The compact artifact carries receipts and a
> content boundary, while the 1,374,886,097-byte raw ledger remains external
> with SHA-256
> `b1f8756c249effab8c77101c8bca73ddf708a5e143c18fe8742fd5712fdd7c12`.
> The source/runtime/gate/artifact chain is internally bound, but no immutable
> public pre-execution commit establishes timing.

> **Exact-probability update (2026-08-08):** a source-locked 24-block Digits
> contextual-bandit factorial did **not** support its registered
> estimator-by-sampler interaction (`+0.01589`, 95% CI
> `[-0.01686,+0.04712]`, exact `p=.350`). MaxRL favored `u_8` over `p(1-p)`,
> but RLOO favored `u_8` too, reversing its prediction; both predeclared
> estimator-matched samplers were below uniform. The common and tuned
> ledgers/checkpoints are byte-identical and their scientific summary content
> matches after removing phase/authorization labels because every selected
> learning rate is `.1`, so they are not independent replications. This
> negative result leaves the coefficient-mass identity intact while ruling
> out the stronger reading that
> mass alone supplies a universally optimal curriculum. See
> `curriculum_maxrl/digits_factorial/RESULTS.md`. The Digits lock is internally
> hashed, but no immutable public pre-execution commit in this checkout
> independently establishes its timing. The compact release retains the full
> 24-block contrast vectors and a content manifest for 2,904 scientific files;
> the 5.08 GB ledgers/checkpoints remain local with no download URI, so a clean
> clone cannot replay the historical run states.

> **Capped-HORA robustness update (2026-08-08):** the post-guidance
> 50-cell by 16-seed SkillChain matrix completed 800/800 runs. Among the 32
> deployable adaptive-minus-fixed cells, all 32 AUC point estimates are
> positive and 22 unadjusted descriptive intervals exclude zero; realized
> coefficient mass is lower in 28/32 means (24 intervals below zero, none
> above). The frozen engineering rule selects cap 32, reducing mean maximum
> group size by 58.07% for a `-.00271` AUC change versus uncapped after
> sampler averaging. This is exploratory allocation robustness, not HORA
> validation, multiplicity-controlled inference, or evidence that coefficient
> mass mediates learning. Its 800 runs and the Digits executions are accounted
> separately from the generator-owned maze/Countdown/GSM8K/Acrobot registry.

*Synthesis pass over ~35 experiments, 7 propositions, 3 testbeds, and 2 external
ports. Not a chronology — a decomposition. REPORT.md tells the story of the
project; this document tells you how the method works, when to use which part,
and what each claim rests on.*

---

## 1. The three channels (decomposition of the method's strength)

Everything the method does flows through exactly three channels. Every
experiment we ran gains its effect through one of them, and knowing *which* is
what lets you predict where the method will and won't help.

### Channel 1 — WASTE AVOIDANCE (the teacher)
*"Don't roll out where the estimator will emit nothing."*

- Mechanism: sample ∝ u(p) = pass@N − pass@1 (P1: the estimator's exact
  expected signal). Zero at mastered (p→1) and unreachable (p→0) tasks.
- What it buys, measured: dead groups 5.8→3.4 of 8 (maze — historical
  zero-weight-group counter, mechanism-open per the EXPERIMENTS.md audit:
  it pools K=0 with K=N and cannot isolate dead-group waste); 22–35% more
  optimization steps per GPU-hour (frontier rollouts also end earlier);
  6/6 paired-seed wins vs uniform.
- Ceiling: the ORACLE bound — CORRECTED (Opus5 review B3 + our control
  battery, `frontier_rl/examples/hindsight_controls.json`). The published
  "oracle" carried a 10% floor handicap; the honest no-floor γ-matched
  oracle reaches 0.8885, TYING the full stack (0.8895). Creation still
  adds on top of perfect allocation (+0.005, oracle+HS 0.8935) but the
  channels substitute more than they compose: "beats the oracle by 0.039"
  is retracted; "+0.005 on top of the best sampler including an oracle"
  is the honest number. Realized Thompson-teacher gain is +0.05–0.08 AUC
  on CPU and ~+0.01 on the maze.
- When it's the dominant channel: mixed-difficulty pools with real spread
  (the balanced regime), and any setting where rollouts are the cost center.

### Channel 2 — SIGNAL CREATION (hindsight recycling)
*"Manufacture verified successes from the failures you already paid for."*

- Mechanism: an all-fail group's rollouts can be credited to verifier-valid
  goals they actually achieved, producing an auxiliary selected-data stream.
  Equality with a fresh target-task update requires equality of the relevant
  update moments; full rewritten-group law equality is sufficient. Verifier
  validity and conditioning rewrite are necessary but not sufficient.
- Retained local evidence: under `γ=4`, centered hindsight added `+0.1050`
  AUC on the 12-seed skill chain. In tile-coded MountainCar it added `+0.191`
  (centered) or `+0.197` (success-only) AUC over the matched teacher arms;
  final custom flag-pass was about 0.84, not a standard-return result. The
  corrected grid study ordered 0.583 uniform < 0.652 teacher < 0.702 full
  stack, but was group-step rather than transition matched.
- A frontier-heavy toy shows the clean categorical mechanism: pure sampling
  cannot create signal when every pool task is unreachable, whereas valid
  auxiliary goals can ignite learning. This is not yet a broad robotics or
  language-model result. The small one-shot-maze hindsight effect is from the
  audited historical GPU protocol and remains hypothesis-generating.

### Channel 3 — LEARNER INTERACTION (the weighting underneath)
*"A curriculum changes the data stream; the learner determines the update."*

- Mechanism: P5 — MaxRL concentrates ≈(N−1)× more signal than RLOO on
  frontier tasks as p→0, and unlike GRPO its weight function doesn't invert
  at p→1.
- Decomposition (control battery, 5 seeds): of the +0.22 hindsight gap
  on fixed pools, an extra-gradient placebo (replaying LIVE gradients on
  dead-group slots, zero relabel information) captures 83%, lr×2 captures
  68%, and random-target relabeling 69%. The relabel DIRECTION carries
  +0.037 beyond the strongest placebo — real, exactness-dependent (fake
  labels are actively harmful), but the headline "+0.22 from signal
  creation" decomposes into ~0.18 gradient-dose effect + ~0.04 direction
  information. Both numbers ship together from now on.
- What it buys, measured: the H6 reversal. The identical teacher GREW
  coverage under MaxRL every seed (pass@8 0.316→0.348); GRPO decayed
  coverage every seed, and in the seed run with a teacher the collapse
  was AMPLIFIED (0.332→0.269, easy-retention lost — single-seed arm).
  GRPO's inverted weighting was silently maintaining easy tasks; the
  curriculum removes that maintenance.
- Historical interpretation (superseded by the balanced factorial): the
  interventions can interact with the estimator, but the single-seed GRPO
  arm above does not establish a general compatibility theorem.

**The one-line synthesis: the teacher reallocates groups, hindsight adds
auxiliary targets, and the learner converts those data into updates.**

## 2. The regime map (when each channel dominates)

| observed regime | ch.1 teacher | ch.2 hindsight | evidence boundary |
|---|---|---|---|
| mixed nested thresholds | local positive effect | larger local effect | corrected CPU/Gym studies |
| fixed eight-threshold Acrobot pool | `u_16` beats `p(1-p)` and uniform under one practical-MaxRL learner | not tested | registered 20-pair V2 score-shape result; no held-out generalization |
| exact-probability Digits contextual bandit | `u_8>p(1-p)` under MaxRL, but both below uniform | not tested | registered 24-block interaction not supported; RLOO prediction reversed |
| frontier-heavy pool | no bootstrap in the limit | categorical ignition possible | synthetic mechanism only |
| fixed recurring goals | compounding is plausible | largest retained local effects | skill chain + tile-coded MountainCar |
| one-shot procedural tasks | unresolved under corrected protocol | possibly smaller | historical maze hypothesis only |
| arm-starved budget | posterior cannot localize | also hard to estimate | arithmetic/design constraint |

Second-order hypotheses worth testing:
- **Shared transfer:** shared parameters are a plausible competence-transfer
  channel, but the MountainCar shared/per-bin comparison confounds capacity
  and data flow, and Acrobot's capacity controls were behaviorally inadequate.
- **Concentration:** `γ=4` helped in two tightly shared nested-task studies;
  it is an empirical knob, not a theorem or a default for broad pools.
- **Capacity:** historical maze probes suggest a capacity/frontier
  interaction, but corrected multi-seed confirmation is still missing.

## 3. The meter lesson (how to even see the method working)

Three separate times, the *metric* hid what the method was doing:

1. **Fixed-step comparisons hid the teacher's speed** (it runs 22–35% more
   steps per hour) → matched wall-clock protocol.
2. **Peakedness hid the teacher's targeting** (ZPD utilities are diffuse by
   design — SONIC_RESPONSE Q5) → targeting-ratio criterion.
3. **pass@1 hid the deep-frontier march entirely** (L6 "stalled at 0.01–0.05"
   while coverage@64 went 0.125→0.438) → coverage currency.

Generalization: **likelihood-shaped training moves the distribution's tail
first; any single-sample metric under-reports it.** Evaluate in coverage/
efficiency currency or you will kill working runs.

## 4. Practitioner's playbook (how to actually play it)

Decision procedure distilled from every ablation:

1. **Choose and test the task axis.** Use one goal-conditioned policy where
   plausible, then measure whether training bin `k` changes held-out behavior
   at `k+1`. Include capacity- and data-adequate controls before calling the
   effect transfer.
2. **Teacher config**: learnability p(1−p) if you have no natural group size
   N (dense-PPO, hazard-style p); advmass with your real N if you have
   episodic groups. Decay 0.7 (evidence-scaled half-life if throughput
   varies). Floor 0.1. Thompson if stochasticity is acceptable; mean+k·std
   if not. Start with `γ=1`; register `γ=4` as a distinct concentration arm
   when nested shared tasks make it plausible.
3. **Hindsight**: enable only with a binary verifier, goal-conditioning
   rewrite, first-hit truncation where applicable, and an adequacy pilot.
   Compare relabeled and fresh target-task update moments; never feed relabels
   to requested-task teacher state.
4. **Learner check**: register the estimator/surrogate and include an
   objective-by-teacher ablation. Do not generalize the historical GRPO
   interaction before a corrected run.
5. **Metrics from day one**: standard environment score, fixed-target pass
   rates, `K=0`/mixed/`K=N` counts, coefficient mass, gradient norm/alignment,
   actual transitions, optimizer updates, and wall-clock.
6. **When the frontier stalls**: run the q-diagnosis (per-step accuracy →
   geometric reach). Capacity problem ⇒ wider/longer; curriculum problem ⇒
   check sharing + relabel contracts.

## 5. Claim inventory (every load-bearing claim and its strongest evidence)

| claim | strongest single piece of evidence | grade |
|---|---|---|
| u(p) = estimator's exact expected signal | P1 proof + 200k-trial MC | proved |
| deployed-`N` score shape helps in the fixed Acrobot pool | V2 target-uniform transition-AUC: `u_16` .6871056515 vs `p(1-p)` .6390719632, paired +.0480336884 [95% CI .0209366676, .0738485654], exact sign-flip p=.0033607483, 15/20 positive; all 60 confirmation runs valid | registered local confirmation; sign-exchangeability assumed |
| Acrobot `u_16` secondary beats uniform | paired +.0418737050 [95% CI .0218239396, .0605859853], raw p=.000808716, Holm p=.001617432, 17/20 positive; `p(1-p)-uniform` was -.0061599834 [−.0226437219, .0121954971], raw/adjusted p=.507843 and unsupported | multiplicity-controlled local secondary |
| paid-probe `u_16` beats ProCuRL-env at the frozen cadence | 320/320 confirmation runs; fixed-paid-AUC paired mean `+.004894`, `t(79)=1.9773`, `p=.05149`, below `.02` SESOI | unsupported registered primary |
| charged probes dominate this frozen selection attachment | probe fractions ≈.932; ordinary-uniform AUC `.65149` versus ProCuRL-env `.33771`, probe-sham `.33942`, and `u_16` `.34261` | strong cadence-specific diagnostic; no full-PPO ProCuRL inferiority claim |
| teacher beats uniform (matched clock) | 6/6 paired seeds; step-matched, teacher-only is n.s. (+0.006, p≈0.36) — the clock gain is largely throughput | multi-seed, decomposed |
| hindsight direction carries information in the CPU fixed-pool control | control battery: +0.037 beyond its dose-matched replay arm, tightest arm (±.0025); random-target loses to replay 5/5 — cosine table RETIRED as evidence (no discriminating power, Opus5 B5). This does not make the higher-dose Countdown ARM B dose-matched. | controlled CPU |
| full stack ties matched oracle; creation adds +0.005 on top | `hindsight_controls.json`: no-floor γ-matched oracle 0.8885 ≈ full stack 0.8895; oracle+HS 0.8935. ("beats the oracle 0.890 vs 0.851" RETRACTED — floor handicap) | multi-seed CPU |
| creation is the only live channel in dead regimes | V5 frontier-heavy w/ uniform+HS control (0.931 ≈ teacher+HS 0.928, both vs 0.000): allocation contributes nothing there; artifact `results_baselines_regimes.json` | controlled |
| maze coverage-AUC estimator ordering survives a fresh wave | wave 2: 6/6 positive independent blocks under uniform and 6/6 under the frontier sampler (exact sign p=.03125 each); 12/12 sampler-averaged blocks positive descriptively across waves; external record says pre-specified, locking object absent | fresh-wave confirmation; timing externally recorded |
| Countdown recycling package concentrates the logged metric | three-seed aggregate mean@16 0.278→0.324 while VERL bootstrap best@16 0.541→0.492; complete seed records are absent, and this with-replacement proxy is not standard pass@16 | multi-seed aggregate, metric-limited |
| saturation gating is a validated mitigation | **not supported**: the favorable under-gated point used faulty decay; fixed-code strong gating behaved like recycling-off and failed its registered dial criteria | refuted / suggestive only |
| GSM8K estimator–curriculum interaction | `g3p` missed the treatment-delivery mean gate by 0.00148, so endpoints are non-causal and the interaction is inconclusive by design | gated inconclusive |
| compounding drives hindsight's size | CPU +0.22 (83% captured by replay placebo — ship both numbers) vs maze +0.01 | cross-regime, decomposed |
| coverage is the right meter | L6 0.125→0.438 invisible to pass@1 | single-ckpt* |
| efficiency grows with difficulty | 11× at L5 / tie L2–3 / 3× worse L4 at the stated absolute-0.25 convention (old 1.2×/0.5× cells not reproducible from artifact — corrected) | single-seed* |
| sharing is the transfer channel | MountainCar per-bin 0 vs shared 1.000 | controlled |
| γ tracks structure | V6 + ODE model + GPU non-transfer *prediction* | pre-registered |

The historical maze coverage and efficiency multipliers remain useful for
experiment design, not for external performance claims.

## 6. What we'd still like to know (ranked)

1. Multi-seed per-k Countdown evaluation from retained checkpoints or raw
   task-level outcomes; the current crossing plot is single-seed.
2. Recover the frozen Countdown manifests and per-task 16-sample outcomes (or
   compatible checkpoints) to compute the clean 101-task tier-0 subset.
3. Does a corrected-code **moderate** destination gate reproduce the favorable
   under-gated point? Strong gating has already failed and should not be rerun
   as if it were an open result.
4. Does the wide model's L6 coverage convert to pass@1 with more budget
   (capacity × duration interaction)?
5. Streaming teacher on a real procedural source (only validated on synthetic
   continuous goals so far).
6. The charged-probe ProCuRL-env attachment is complete and its registered
   primary is unsupported. No further Mac experiment is required for this
   submission. PLR, PAIRED, ACCEL, and full SFL studies are deferred to
   post-submission work under separately frozen protocols.
