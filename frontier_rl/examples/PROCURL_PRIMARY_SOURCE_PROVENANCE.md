# ProCuRL primary-source provenance for the Acrobot selection study

This note records what the new experiment borrows from ProCuRL and what it
does not.  It is part of the experiment's source lock.

## Superseded local seal

The first local ProCuRL-selection seal and its engineering/development wave
were invalidated before gate creation because runner and analyzer used
different Python 3.12 floating-point summation algorithms for episode entropy
aggregates.  The exact bytes and outcome-blind incident record are preserved
under `INVALID_ABORTED_PRE_GATE_ENTROPY_SUM_MISMATCH/`.  No arm contrast,
effect direction, confidence interval, p-value, or minimum-effect decision was
inspected.  The exposed quick/development seeds are permanently burned; this
replacement registration uses 21400 and 21300--21302 while leaving the
unopened confirmation block 21000--21079 unchanged.

## Primary sources

- Paper: *Proximal Curriculum for Reinforcement Learning Agents*
  ([OpenReview](https://openreview.net/forum?id=8WUyeeMxMH), especially
  Appendix C.3 and Figure 7; also
  [arXiv:2304.12877](https://arxiv.org/abs/2304.12877)).
- Official repository:
  [machine-teaching-group/tmlr2023_proximal-curriculum-rl](https://github.com/machine-teaching-group/tmlr2023_proximal-curriculum-rl).
- Repository revision inspected: commit
  [`17904f1d7b9b29e089d4f70ae7aadf1da50ba6b2`](https://github.com/machine-teaching-group/tmlr2023_proximal-curriculum-rl/tree/17904f1d7b9b29e089d4f70ae7aadf1da50ba6b2).

The exact source locations used for the independent semantic audit are:

- [`main.py` lines 23--27](https://github.com/machine-teaching-group/tmlr2023_proximal-curriculum-rl/blob/17904f1d7b9b29e089d4f70ae7aadf1da50ba6b2/main.py#L23-L27),
  for the PointMassSparse/default argument tuple, including `beta=20` and
  `noise=0`;
- [`PointMassTeacher.py` lines 4--16](https://github.com/machine-teaching-group/tmlr2023_proximal-curriculum-rl/blob/17904f1d7b9b29e089d4f70ae7aadf1da50ba6b2/envs/pointmass/PointMassTeacher.py#L4-L16),
  for the PointMass teacher's `Npos=5120` cadence;
- [`AbstractTeacher.py` lines 18--27](https://github.com/machine-teaching-group/tmlr2023_proximal-curriculum-rl/blob/17904f1d7b9b29e089d4f70ae7aadf1da50ba6b2/abstract_classes/AbstractTeacher.py#L18-L27),
  for 20 policy-evaluation rollouts per environment; and
- [`AbstractGymWrapper.py` lines 130--135](https://github.com/machine-teaching-group/tmlr2023_proximal-curriculum-rl/blob/17904f1d7b9b29e089d4f70ae7aadf1da50ba6b2/abstract_classes/AbstractGymWrapper.py#L130-L135),
  for the score-to-softmax selection path.

The repository displayed no license file at the inspected revision.  No
upstream source code is copied into this project.  The implementation here was
written independently from the algorithmic description and from a read-only
audit of public configuration and control flow.

## Semantics carried into this study

The ProCuRL arm implements the source experiment's PointMass-s
environment-selection semantics.  The provenance tuple is PointMassSparse,
explicit `procurl-env`, `beta=20`, `noise=0`, teacher cadence `Npos=5120`, and
20 evaluation rollouts.  The command-line default curriculum in the repository
is PLR, not ProCuRL; therefore “default tuple” here means the default
PointMassSparse/environment hyperparameters with the curriculum explicitly set
to the ProCuRL-env arm, consistent with the paper's Appendix C.3/Figure 7
comparison.  It does not mean that invoking the upstream script with no
curriculum argument selects ProCuRL.

The carried semantics are:

1. estimate the current policy's success probability separately on every
   environment using 20 fresh stochastic probe episodes;
2. assign environment logit `20 * p_hat * (1 - p_hat)`;
3. sample the next training environment from the stable softmax of those
   logits; and
4. replace the entire vector of estimates on each scheduled probe sweep.

There is no Bayesian prior, exponential decay, running memory, probability
floor, or additional temperature.  In this Acrobot attachment, the source's
iteration cadence is mapped explicitly to a student-transition clock: refresh
at every crossed multiple of 5,120 student transitions.  Probe transitions are
charged to the paid-compute axis.

The local probe implementation deliberately uses isolated, coordinate-seeded
NumPy action generators and a fresh Gymnasium environment per sweep; evaluation
also uses a separate fresh environment.  The upstream implementation uses a
global Python randomness path.  This RNG engineering is not claimed as copied
source behavior: it is an auditable common-random-number implementation that
leaves the `p_hat -> beta*p_hat*(1-p_hat) -> softmax` selection semantics
unchanged and prevents probes from consuming learner RNG state.

## Deliberate non-equivalence

This is **not** a reproduction of the full ProCuRL training stack or benchmark
suite.  It attaches ProCuRL's selection rule to this project's fixed Acrobot
learner: a task-blind 640-parameter NumPy actor trained by practical `N=16`
dropped-group MaxRL.  The learner, environment pool, optimizer, and evaluation
metric therefore differ from the upstream PPO experiments.  Claims from this
study must be phrased as a source-faithful **selection-semantic comparison on a
common learner**, not as a head-to-head reproduction of the complete ProCuRL
system.

## Registration limitation

The repository does not have an immutable public commit proving that this
protocol predates execution.  The local source/runtime lock and raw artifact
hashes provide mechanical provenance, but they are not equivalent to a public
pre-registration.  This limitation must remain disclosed in any paper or
artifact statement.
