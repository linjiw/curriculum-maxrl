# Independent review of `docs/paper-draft.pdf`

Date: 2026-08-04

Review target:

- `docs/paper-draft.pdf`
- Built 2026-08-04 21:51 UTC
- SHA-256: `918717573df76726a74cfb91919c69daf551c1523c740f79a93108de40088592`
- 20 pages; the ICLR build has about 15 pages of main text before references

## Overall verdict

There is a real ICLR paper here, but this version is not submission-ready.
My current ICLR-style score is **4/10 (weak reject), confidence 4/5**. The
core coefficient-mass algebra is clean, the paper reports negative results
with unusual honesty, and recycling-induced sharpening is an interesting
observation. The binding problems are that the strongest title-level claim
is still exploratory, the implemented relabel gate does not estimate the
quantity used in its derivation, and the LLM relabel update mixes different
tasks inside one group while borrowing theory that assumes one common task.

ICLR is a much better target than ICRA. In its present form I would expect an
ICRA rejection on fit: the paper is primarily about RLVR and language-model
post-training, while the only robotics result is a one-seed IsaacLab pilot
whose raw logs are not in this repository. An ICRA version would need a
robotics-centered question, substantial multi-seed simulation and preferably
hardware evidence, and direct HER/goal-selection baselines.

## What is strong

1. **The central algebra checks out.** I independently re-derived Lemma 1,
   Proposition 1, its score-contrast factorization, and the MaxRL/RLOO/GRPO
   tail formulas. The repository's mass-enumeration test also passes.
2. **The scope caveats are scientifically mature.** The paper distinguishes
   coefficient mass from gradient norm and expected improvement, acknowledges
   posterior starvation, reports the Jugs null, and retracts earlier claims.
3. **Sharpening is a useful measurement.** The mean@16-up/pass@16-down
   Countdown result is replicated over three seeds and suggests a practical
   evaluation rule: always report coverage beside mean accuracy.
4. **The visual system is coherent.** Figures are legible and the build has no
   unresolved references or overfull-box warnings.
5. **The project has useful verification infrastructure.** The three local
   test entry points pass when run directly, and listed figure-data checksums
   match the manifest.

## Submission-blocking findings

### P0. The implemented gate is not a destination pass-rate gate

The paper says the gate uses an independent estimate of
`p_theta(success | destination g')` and is derived from the high-`p` zero of
`u_N(p)` (`paper/body.tex:118-131`, `963-973`, `1077-1082`). The referenced
verl implementation instead tracks how often an achieved value appears in
the relabel stream, conditional on a requested-task dead group. It does not
roll out the destination-conditioned task and does not observe destination
failures. This is an achieved-goal popularity/recency statistic, not the
fresh-destination pass rate in Proposition 1.

There is a second mismatch: `u_N(p)` has its high endpoint zero at `p=1`,
which would admit every nondegenerate relabel. The useful cutoff is `0.5`,
and Table 3 correctly labels that midpoint as tuned
(`paper/body.tex:1488-1493`). The related-work statement that there is "no
separately tuned goal-selection heuristic" is therefore false.

Required resolution:

- Either implement a genuine task-conditioned estimate of `p(g')`, apply the
  gate to that estimate, and rerun the corrected gate at multiple seeds;
- or rename this an achieved-goal frequency/novelty gate, remove "derived
  gate" and "destination pass rate" claims, and develop an argument for the
  statistic actually used.

### P0. Mixed-target relabel groups do not optimize the stated MaxRL objective

Algorithm 1 relabels each parseable rollout to its own achieved goal while
retaining one group (`paper/body.tex:458-466`). The appendix confirms that one
UID can mix different rewritten tasks (`paper/body.tex:1417-1423`). The MaxRL
derivation assumes `N` i.i.d. rollouts of one common prompt with one pass rate.
Once rows have different prompts, the shared `K` baseline couples unrelated
tasks, and the update is not a `J_{N-1}` gradient for any achieved task.

The main Remark 3 is written for a singular destination and a fresh i.i.d.
group for that destination (`paper/body.tex:472-487`), so its
distribution-matching condition cannot repair the implemented mixed-target
case.

Recommended fix: choose one destination per dead group, rewrite all `N`
rollouts to that same destination, reverify all `N`, and compute `K'` for that
common task. Alternatively regroup rows by destination and define new UIDs.
If per-row relabeling is retained, present it as a separate weighted-SFT
objective and derive that objective instead of calling it MaxRL.

### P0. "The estimator decides" is not causally identified yet

The strongest maze evidence pools heterogeneous runs and reused seed blocks;
the paper itself calls it exploratory. The GSM8K result has two seeds, weak
teacher steering, a sampling-with-replacement confound, and only one of two
seeds shows the preregistered regression shape. The conclusion explicitly
says the decisive experiments are still queued (`paper/body.tex:1112-1123`,
`1209-1212`).

The title-level claim requires, before submission:

1. A balanced `{MaxRL, GRPO} x {uniform, teacher}` neural factorial with at
   least six independent paired seed blocks, identical warmstarts, matched
   generation/optimization budgets, and one prespecified coverage endpoint.
2. A GRPO-native teacher arm. Otherwise the result may mean only that the
   MaxRL-derived teacher is the wrong scheduler for GRPO.
3. A no-standard-deviation/Dr.GRPO arm. This is the sharp ablation separating
   success conditioning from variance normalization.
4. A fixed realized-prompt-schedule neural control, or an equivalent design
   that separates estimator effects from adaptive data selection.

Until those land, the abstract and title should say the algebra **motivates a
hypothesis** about estimator-conditioned coverage, not that it establishes
the interaction.

### P0. The paper is far outside either target venue's normal main-text budget

The ICLR-formatted PDF is 20 pages and reaches limitations on page 15;
references begin afterward. The abstract is about 339 words. A typical ICLR
submission allows roughly nine main-text pages, while ICRA is normally much
shorter and two-column. Check the live CFP, but this is not a final-format
draft for either venue.

This is not solvable by typography. The paper currently contains a theory
paper, a curriculum paper, a hindsight-recycling paper, and a research
postmortem. It needs a narrower claim.

## Major findings

### P1. Proposition 3 is false at its stated boundary

Proposition 3 permits `N_i >= 0` (`paper/body.tex:325-332`), but

`u_0(p) = 1 - (1-p)^0 - p = -p`,

whereas the coefficient mass of an unallocated prompt is zero. A one-rollout
group also has zero MaxRL mass. The displayed marginal argument is valid only
after a mandatory initial group size of at least one. With optional task
activation, the problem has an activation cost and is not the simple
diminishing-return water-filling problem stated.

Best fix: remove Proposition 3 because the deployed teacher does not use it.
If retained, require fixed initial `n_i >= 1`, allocate only additional
rollouts, and state the budget feasibility condition.

### P1. The intended gate has not been validated

All three-seed B3 runs used the faulty decay implementation. The corrected
implementation has one seed and a different effective gate strength
(`paper/body.tex:1008-1027`). The reported three points therefore do not form
a controlled dose response: gate strength is confounded with code version.

Do not call the intended gate validated until the fixed-code sweep is complete
at the claimed operating point and at enough seeds. The current result
supports only: "one bugged frequency-gate configuration restored frontier
coverage in this suite."

### P1. Cross-estimator mass magnitudes depend on arbitrary normalization

The expected-mass formulas are exact for the chosen coefficient conventions,
but an estimator's coefficients can be multiplied by a global constant and
offset with learning-rate calibration. Numerical statements such as
"`(N-1)x` more mass" are implementation-normalization facts, not invariant
properties of an objective.

Emphasize shape within an estimator, normalize curves at a reference pass
rate or equal expected update norm, and include learning-rate/update-norm
sensitivity. The no-std GRPO arm is especially important here.

### P1. The LLM figure visually outruns the evidence

Figure 4d and Figure 6 foreground the registered GSM8K seed. Their error bars
are repeated evaluations of a fixed checkpoint, not training-seed
uncertainty. The captions disclose this, but the graphic still looks like a
method-level uncertainty comparison. The replication does not reproduce the
headline trajectory, and the absolute pass@k harness differs from the trainer
metric by about `3x` (`paper/body.tex:1386-1403`).

Remove GSM8K from the headline figure until the controlled multi-seed result
is complete. If it remains, plot both seeds as paired trajectories and do not
use fixed-checkpoint evaluation noise as method error bars.

### P1. The artifact is not yet self-contained

- `paper/body.tex:1426-1428` cites `verl/utils/hindsight.py`, but that path is
  absent from this Git repository; it exists only in a sibling checkout.
- The figure manifest lists no input for the empirical Gym figure, although
  `paper/figures/fig6_gym.py` reads
  `frontier_rl/examples/gym_convergence.json`.
- Figure 2 and Figure 3 use manually transcribed endpoint tables rather than
  deriving them from seed-level logs.
- Some result literals and annotations remain in plotting scripts despite
  the statement that none are hard-coded.
- IsaacLab raw logs are held by another team.
- The GSM8K pass@k/trainer discrepancy remains unreconciled.

Vendor the exact execution fork or pin it as a submodule/archive, include
execution commit hashes, build every table and figure from raw seed-level
records, and make one clean artifact command reproduce the paper.

### P1. The paper's strongest novelty claims need narrower wording

"First," "no prior work," "one method, no per-domain switches," and "derived
mitigation" are all vulnerable. The implementation already requires
domain-specific relabel maps and destination keys, and Jugs exposed a
domain-specific keying bug. Independently verify the 2026 concurrent-work
claims and state novelty at the exact level supported: measurement of
pass@k loss under a particular exact-verifier relabel loop, plus a tested
admission heuristic.

## Reviewer questions

1. What exact random variable does the gate estimate, and under which
   conditioning distribution is it an estimate of `p(g')`?
2. What objective is optimized when one UID contains several rewritten
   prompts but shares one group mean and one success count?
3. What experiment would falsify the proposed coefficient-mass-to-coverage
   bridge?
4. Why should absolute coefficient mass be compared across estimators without
   matching update norm or learning rate?
5. Does the estimator interaction survive a balanced neural factorial with
   identical warmstarts and realized prompt schedules?
6. What happens when GRPO is scheduled by its own mass functional?
7. Does Dr.GRPO/no-std remove the mastered-tail sharpening signature?
8. Why is a tuned `p_hat <= 0.5` cutoff described as the zero of a function
   whose relevant zero is at `p=1`?
9. Can every headline number be regenerated from raw seed-level artifacts in
   this repository at the cited execution commit?
10. Why is the single-seed registered GSM8K trajectory in the headline figure
    after the replication failed to reproduce its shape?
11. Which one result is the primary confirmatory result, and which results are
    exploratory or post-hoc?
12. Is the intended contribution a curriculum, an estimator diagnosis, or a
    safe hindsight method? What should a reviewer remember as the one claim?
13. For ICRA, where is the robotics-specific evidence showing value over HER,
    HGG, curriculum-guided HER, and standard robot curricula?

## Recommended paper strategy

### Preferred: split the work

**Paper A: estimator-conditioned curricula.**

- Core algebra and estimator mass profiles.
- Balanced maze factorial plus Dr.GRPO and GRPO-native teacher.
- One controlled LLM replication if it is conclusive.
- Main claim: data selection must be evaluated jointly with the estimator.

**Paper B: failure recycling and coverage.**

- A well-defined single-destination relabel objective.
- Recycling-induced sharpening.
- A correctly measured destination-saturation or explicitly named
  achieved-frequency gate.
- Fixed-code multi-seed dose sweep and exact HER/goal-selection baselines.

### If keeping one paper

Center the paper on **estimator-conditioned data interventions**. Keep the
mass identity, one decisive factorial, and recycling as one application.
Move IsaacLab, Gym, Jugs, run-history narration, most negative results, and
telemetry to the appendix. Remove Proposition 3.

Target a 180-220 word abstract with:

1. one exact result;
2. one confirmatory experiment;
3. one practical consequence;
4. one limitation.

Do not submit sentences saying decisive runs are "queued" or "running."

## Priority revision sequence

1. Fix or reframe the gate statistic and the mixed-target group objective.
2. Run the balanced maze factorial, GRPO-native teacher, and Dr.GRPO control.
3. Rerun the corrected gate at multiple seeds and fixed code.
4. Add the generation-budget-matched dead-group skip/resample baseline.
5. Only then spend more compute on the steering-controlled LLM replication.
6. Rebuild all figures from raw seed-level artifacts and vendor the execution
   code.
7. Cut the ICLR main paper to the venue limit and remove exploratory claims
   from the title and abstract unless the decisive experiments land.

## Venue recommendation

**Target ICLR**, after the P0 issues and decisive controls are resolved. The
theory/learning-dynamics/LLM-RLVR framing fits ICLR, and the strongest
potential contribution is methodological rather than robotic.

**Do not target ICRA with this manuscript.** A credible ICRA submission would
need to become a different paper: robot goal-conditioned RL as the center,
multiple manipulation/locomotion tasks, direct hindsight-goal-selection
baselines, several seeds, and preferably real-robot transfer or a compelling
safety/deployment result.

## Verification performed for this review

- Read the complete latest PDF and shared LaTeX body.
- Visually inspected all main-result pages and the appendix layout.
- Re-derived Lemma 1 and Propositions 1-3.
- Ran:
  - `python3 -m curriculum_maxrl.test_mass_formulas`
  - `python3 curriculum_maxrl/test_verl_curriculum.py`
  - `python3 frontier_rl/test_framework.py`
- All direct test entry points passed.
- Checked LaTeX logs: no undefined references/citations and no overfull boxes.
- Checked listed figure-data checksums against `paper/results/manifest.json`.

External novelty and priority claims were not independently literature-searched
for this review.
