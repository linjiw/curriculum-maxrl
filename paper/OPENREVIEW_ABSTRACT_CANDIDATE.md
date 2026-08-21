# OpenReview submission candidate — ICLR 2027

**Prepared:** 2026-08-12. **Regenerated:** 2026-08-20. **Freeze deadline:** Sept 16 (abstract due Sept 18).
**Source of truth:** `paper/main_iclr2027.tex` (title) and `paper/body_iclr.tex` (abstract). If those change, regenerate this file; do not edit the abstract here independently.

## Title

Learnability, Reweighted: Which Tasks the Estimator Makes Active in Verifiable-Reward RL

## Abstract (regenerated verbatim in substance from body_iclr.tex)

Curriculum methods for RL with verifiable rewards summarize each task unit by a pass rate $p$ and apply a learnability curve $f(p)$. That presumes the unit the curriculum scores is the random object the group estimator consumes. It need not be: a unit of uniformly uncertain tasks and a unit mixing mastered with impossible tasks share a mean pass rate, yet only the first yields mixed-outcome groups and non-zero updates. We formalize the mismatch. For a permutation-equivariant binary group estimator, activity is a functional of the success-count law, $\mathcal A(z)=\sum_kP(K=k\mid z)M_{\mathcal E}(k)$, with closed-form $M_{\mathcal E}$ for MaxRL, RLOO and GRPO; for practical MaxRL it is $2\{\Pr(K>0)-\mathbb E[K]/N\}$ under any joint law, and the familiar $2(\mathrm{pass@}N-\mathrm{pass@}1)$ is only its conditionally-i.i.d. atomic slice, which factors as $p(1-p)w_{N-1}(p)$ and recovers $p(1-p)$ at $N=2$. For a coarse unit, plugging in the mean pass rate over-predicts activity by exactly twice its excess all-fail probability. Empirically, where the scored unit is the estimator's own, the deployed-$N$ shape beats $p(1-p)$ by $+.0480$ (95% paired-bootstrap CI $[+.0209,+.0738]$) and replicates on two further platforms. Three preregistered follow-ups then bound it: performance rises past the deployed $N$, the score is starved as a standalone replay priority, and where the curriculum scores a level aggregating heterogeneous tasks the contrast reverses ($-.0032$, CI $[-.0054,-.0011]$), with the coefficient-mass error accounted for to floating point. The estimator defines the coefficient map; the curriculum defines the unit over which that map is averaged, and these operations do not commute. Coefficient activity is a principled, estimator-conditioned source of curriculum hypotheses, not a measure of learning utility.

## Live evidence note — not part of the abstract

The P0 shared-posterior plug-in-versus-count-law intervention is frozen and running blind over 48 paired blocks. Its endpoint is deliberately absent from this candidate until the complete campaign is hash-validated and the frozen analyzer is invoked once.

## Form metadata suggestions

- Primary area: reinforcement learning
- Keywords: reinforcement learning with verifiable rewards; curriculum learning; group-relative estimators; pass@k; task selection; MaxRL
- TL;DR: Mean pass rate is sufficient for a finite-group estimator's coefficient activity only at the right task granularity; the success-count law gives the general score and controlled experiments map where that score does and does not predict learning.
