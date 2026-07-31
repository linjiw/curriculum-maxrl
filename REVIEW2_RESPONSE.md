# Response to the second external review (2026-07-31)

*The review's verdict: major revision / weak reject as written — "the
strongest defensible result is that estimator choice is associated with
opposite coverage dynamics." Verdict accepted. Each finding was verified
against our files before acting; this document records agree/disagree,
the evidence, and what changed. Fixes shipped in the same commit.*

---

## Finding-by-finding

### B1. "Advantage mass is not learning signal" — AGREE (framing), fixed

Correct: E[Σ|wᵢ|] is coefficient mass, not gradient norm/SNR/Fisher.
Prop. 3 optimizes the surrogate, not learning performance.

**Action:** New `Remark 1 (Scope of the surrogate)` in §3 states exactly
what mass is (the model-free factor of the gradient), why it is the
right *sampling* utility anyway (knowable pre-rollout; exact zeros;
band transfers empirically), and what it does not claim
(proportionality to expected improvement — §7.4a is the measured
counterexample regime). "Learning signal" → "advantage mass" wherever
the claim is quantitative.

### B2. "Practical sampler ≠ proved water-filling allocator" — AGREE, fixed

Correct: proportional u^γ sampling with fixed group sizes does not
maximize Σqᵢuᵢ (whose unconstrained optimum is a point mass on the
arg-max).

**Action:** New interpretation under Prop. 3: water-filling is the
*reference* allocator; the deployed teacher trades optimality for
group-parallel batching and posterior-noise robustness (a point-mass
allocator is brittle); the measured gap is the point — the
true-pass-rate oracle *ties* the Thompson teacher (§7.1).

### B3. "Adaptive q_t changes the objective" — AGREE, fixed

Correct: unbiasedness holds per-prompt, not for E_ρ[J].

**Action:** New `Remark 2 (The curriculum changes the training
distribution)` says this in print: every curriculum strikes this
bargain; all evaluations are on held-out pools under ρ, so mismatch
cost is charged to the method. The contribution bullet no longer says
"unbiasedness carries over" bare — it now states the per-group sense
and points at the remark.

### B4. "Retracted oracle claim still in the PDF" — PARTLY AGREE, fully fixed

The reviewer's line citation was stale (§7.1 text had been corrected to
"above the true-pass-rate oracle 0.851" — still wrong, since the 0.851
oracle carried the floor handicap). **Figure 2a still plotted the
handicapped 0.851 oracle with the stack "beating" it, and the WEBSITE
carried the retracted claim in four places** (chart-3 caption + data,
summary table, abstract paragraph).

**Action:** §7.1 rewritten around the honest control battery: matched
oracle 0.8885 TIES stack 0.8895; oracle+recycling 0.8935; placebo
captures 83% of the hindsight gap; direction carries +0.037. Fig 2a
replotted with the matched oracle and oracle+recycling bars. All four
site locations corrected, with the retraction stated inline. §9 now
names the retraction.

### B5. "Safety interaction is not factorial evidence" — AGREE, fixed

Correct: the maze design supports an estimator MAIN EFFECT on coverage
(5 MaxRL runs all grow pass@8, 4 GRPO runs all lose it, permutation
p=0.0079); the frontier+GRPO "amplification" is single-seed; uniform
MaxRL also gains coverage.

**Action:** Abstract, §1 Q2, and §7.3 all restated as the main effect;
"the identical teacher grows/degrades" phrasing removed where it implied
an interaction; §7.3 now says explicitly the suite is not powered for
the interaction and that GSM8K's pre-registered P-G2 is the interaction
test (single seed, second seed in flight). Limitations spells out the
evidentiary status. GSM8K's teacher-reduces-pass@k-under-both is now in
Limitations verbatim ("coverage-neutral-to-negative under both
estimators at this budget — the estimator contrast is about the sign of
the mean-accuracy effect, not a coverage rescue").

### B6. "Maze doesn't test the derived utility" — AGREE, disclosed + ablated

Correct: maze teachers use legacy (1-(1-p)^N)(1-p), champion adds ALP.

**Action:** (i) Provenance note added to §7.3. (ii) Measured the gap:
TV between normalized sampling distributions of exact vs legacy forms =
0.013 at N=8 (0.005 at N=16). (iii) New ablation
(`run_utility_forms.py`, 5 seeds, teacher-only skill chain): exact
0.728±.002, legacy 0.733±.013 (within noise), p(1-p) slice 0.642±.028
(−0.09). The claim the maze supports (band vs uniform, and the safety
main effect) survives; "validates the exact functional" is not claimed.

### B7. "Hindsight exactness is conditional" — AGREE, fixed

Correct: the proof gives exactness iff conditional laws match; the
validation testbed satisfies the condition by construction; the +0.22
decomposes into ~83% dose + 0.037 direction.

**Action:** Prop. 4 renamed "Hindsight update, characterized" and
states the biased-otherwise case in the proposition body. The
interpretation says "a property to measure per environment, not a
blanket guarantee" and names where the validation's laws match by
construction. The dose/direction decomposition now ships in §5 AND §7.1
(it was previously only in EVIDENCE.md). Contribution 3 retitled.

### B8. Figure defects — AGREE on all five, all fixed

- **Fig 1:** GRPO curve was √(p(1-p)) rescaled to match MaxRL's max
  (arbitrary). Now the exact finite-N mass (1/N)E√(K(N−K)), K~Bin(N,p),
  MC-verified, same |coefficient|-sum convention for all three
  estimators; caption states the formula. The honest picture is
  *better* for us: the two-sided tail asymmetry (MaxRL √(N−1)× at p→0,
  GRPO √(N−1)× at p→1) is the mechanism of the safety result, now
  annotated on the figure and in Prop. 2's interpretation.
- **Fig 2a:** retracted oracle → corrected panel (see B4).
- **Fig 2d:** "running" cell → final .108; caption states P-G1 null.
- **Fig 3:** maxrl series completed (.091/.097/.108); annotation no
  longer says "teacher helps MaxRL" — says only grpo+teacher regresses.
- **Fig 6:** teacher-vs-uniform attribution: uniform-bins control now
  plotted (cracks the task off-ceiling); caption + §7.4b give the
  honest split (spread = existence, teacher = speed/stability).
- **Fig 7 / abstract:** seed-specific "doubling (.153→.306)" removed
  from the abstract; 3-seed restoration numbers throughout; fig 7
  caption explains why tier-1 B3 shows mean-give-back (destination
  saturation) with the frontier-tier restoration stated alongside.

### B9. Reproducibility gaps — AGREE, partially fixed

- Fig 7 read a sibling-repo JSON outside the public repo → data
  vendored into `paper/figures/data/` (committed), script path fixed.
- GSM8K k-sweep + cell trajectories vendored likewise.
- 86% generation-time claim: traced to ray worker logs
  (timing_s/gen ÷ timing_s/step, n=14 steps, mean 0.870) → corrected
  to "87%, measured across our GSM8K runs"; Reproducibility section
  says where timing numbers live.
- PAPER.md marked superseded by the LaTeX paper with a pointer.
- IsaacLab: prose-summary status now stated in Reproducibility rather
  than implied to be artifact-backed.

### B10. Reinforce-Ada omission — AGREE, fixed

Related work now discusses it: within-prompt adaptive rollout count vs
our across-prompt reallocation at fixed group size; Prop. 3 is the
idealized form of both. Citation added (arXiv:2510.04996).
Population-weight vs realized-gradient vs gradient-norm conflation:
addressed by Remark 1 + the exact fig-1 curves.

---

## The reviewer's questions, answered

**Q1 (fixed objective under q_t):** none — and no longer claimed.
Remark 2: FrontierMax optimizes a reweighted objective; evaluation is
under ρ on held-out pools, so the tilt is charged to the method.

**Q2 (why L1 mass over E‖∇‖² or SNR):** not claimed to predict
improvement better; it is the pre-rollout-knowable factor with exact
zeros (Remark 1). Score-norm-aware utilities need the rollouts the
sampler is trying to avoid spending.

**Q3 (exact finite-N GRPO formula):** (1/N)E[√(K(N−K))], K~Bin(N,p);
in the fig-1 caption, MC-verified in the figure script.

**Q4 (matched multi-seed 2×2):** agreed — this is the top remaining
experiment. The GSM8K grpo+teacher seed 2 is running now; the full
matched-replacement multi-seed 2×2 is queued (NEXT_EXPERIMENTS.md).
F1 (replacement confound) remains open and documented.

**Q5 (exact vs legacy vs p(1-p)):** done — `utility_forms.json`:
0.728±.002 / 0.733±.013 / 0.642±.028. Exact ≈ legacy; the N=2 slice is
the one that costs.

**Q6 (how often do conditional laws match in Countdown):** the relabel
keeps the trajectory's own achieved value (no counterfactual
conditioning), so contract-1 exactness holds always; the law-shift
enters through which expressions get attempted for which targets and is
not directly measurable without paired fresh sampling — stated as open
in Prop. 4's interpretation.

**Q7 (corrected gating × 3 seeds + dose sweep):** corrected-decay B3 is
1 seed; 3-seed replication + pre-registered gate_max_p dose sweep are
the standing follow-up (stated in §7.7).

**Q8 (regenerate everything from committed artifacts):** figures yes
(after this commit); IsaacLab remains prose-summary and is disclosed as
such.

---

## Where we push back

1. **"Proposition 3 is only optimal for the surrogate" — yes, and that
   is the paper's structure, not a flaw:** the theory derives a
   *sampling utility*, the experiments measure *end performance*, and
   the oracle-tie result (§7.1) is exactly the experiment that checks
   whether better allocation of the surrogate buys performance. It
   doesn't (allocation saturates) — which the revised paper now says
   more prominently, because it is a finding, not a concession.
2. **"Uniform MaxRL also gains coverage" does not weaken the safety
   claim** — it sharpens it: the main effect is the *estimator*, and
   that is what the revised text claims. The dangerous configuration is
   curriculum-on-GRPO; that statement survives every run we have.
3. **Figure 6 "uniform finishes higher":** on *mean-across-bins* in the
   demo-budget table, within seed noise; on the official-task currency
   trained to convergence, the gated stack is at ceiling every seed
   while uniform is off-ceiling with variance. The revised figure shows
   both rather than hiding the control.

## Scoreboard

| Finding | Verdict | Status |
|---|---|---|
| B1 mass ≠ signal | agree | fixed (Remark 1) |
| B2 sampler ≠ allocator | agree | fixed (Prop. 3 interp) |
| B3 objective shift | agree | fixed (Remark 2) |
| B4 oracle in PDF/site | agree (worse than cited) | fixed everywhere |
| B5 not factorial | agree | fixed (main effect + powering note) |
| B6 legacy utility in maze | agree | disclosed + TV + ablation |
| B7 conditional exactness | agree | fixed (Prop. 4 + decomposition) |
| B8 figures ×5 | agree | all five fixed |
| B9 reproducibility | agree | data vendored; gaps disclosed |
| B10 Reinforce-Ada | agree | added |
