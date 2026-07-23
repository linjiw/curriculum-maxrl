# The evidence, reorganized: where the strength actually comes from

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

- Mechanism: sample ∝ `u_N(p) = pass@N − pass@1` (P1: half the practical
  estimator's exact expected scalar coefficient mass). The practical
  drop-both estimator targets `T=N−1`; `u_2(p)=p(1−p)` and `u_1≡0`. The score
  is zero at mastered (`p→1`) and unreachable (`p→0`) tasks.
- What it bought in the historical maze mechanism logs: the legacy
  zero-weight counter fell from 5.8→3.4 of 8; 22–35% more optimization steps
  ran per GPU-hour (frontier rollouts also ended earlier); and all six paired
  seeds favored the teacher. Those logs conflated all-fail (`K=0`) with
  all-pass (`K=N`) groups and used the legacy score, so they motivate—but do
  not close—the corrected mechanism claim.
- Retained evidence: the teacher improved shared-H64 Acrobot transition-AUC
  by `+0.03635` over uniform across 20 paired seeds. In tile-coded
  MountainCar, exact-mass `γ=4` improved AUC by `+0.141`, while the
  predeclared `γ=1` exact-mass arm was not separated from uniform,
  learnability, or the legacy score. These results establish local efficacy,
  not a universal effect size.
- No oracle ceiling has been established. In one CPU study, a true-pass-rate
  proportional-priority comparator collected 841 units of coefficient mass
  versus 838 for the pseudo-count teacher, yet achieved AUC 0.851 versus
  0.700. Timing, estimation, and gradient direction matter; total mass alone
  is not a sufficient learning proxy.

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
- Historical observation: the same teacher interacted in opposite directions
  with MaxRL and GRPO in an old maze sweep. That run used the pre-audit score,
  counters, budgets, and AUC protocol, so it motivates a corrected
  objective-by-teacher factorial but cannot establish that GRPO curricula are
  generally unsafe.
- Exact positive-part weighting has a pass@k-tail gradient identity only with
  true trajectory scores. For weighted flow/SFT surrogates, the coefficient
  mass remains exact but update-direction fidelity is empirical.

**The one-line synthesis: the teacher reallocates groups, hindsight adds
auxiliary targets, and the learner converts those data into updates.**

## 2. The regime map (when each channel dominates)

| observed regime | ch.1 teacher | ch.2 hindsight | evidence boundary |
|---|---|---|---|
| mixed nested thresholds | local positive effect | larger local effect | corrected CPU/Gym studies |
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
   design; SONIC's forecast pmax/uniform only 1.3–2.0) → targeting-ratio
   criterion.
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
| `2u_N(p)` = practical estimator's exact expected coefficient mass | P1 proof + 200k-trial MC | proved |
| teacher can beat uniform locally | Acrobot V3 `+0.03635`, 20 paired seeds | narrow neural result |
| `γ=4` concentration can matter | MountainCar `+0.116` over `γ=1` | corrected local result |
| verifier-valid hindsight can help | skill chain +0.105; MountainCar about +0.19 | local, exactness unproved |
| pure sampling cannot revive a zero-signal pool | frontier-heavy construction | synthetic mechanism |
| shared transfer causes the gain | current controls confounded/inadequate | open |
| MaxRL makes curricula safe | historical objective interaction only | open |
| compounding predicts hindsight size | retained fixed-pool positives; maze side historical | hypothesis |
| corrected GPU/LLM/robotics generality | no qualifying result yet | open |

The historical maze coverage and efficiency multipliers remain useful for
experiment design, not for external performance claims.

## 6. What we'd still like to know (ranked)

1. Fixed-prompt-set at LLM scale (GSM8K) — the regime map says this is where
   ch.2 compounds; the single most valuable missing cell.
2. One more seed on the efficiency multipliers (cheap, de-stars two claims).
3. Does the wide model's L6 coverage convert to pass@1 with more budget
   (capacity × duration interaction)?
4. Streaming teacher on a real procedural source (only validated on synthetic
   continuous goals so far).
