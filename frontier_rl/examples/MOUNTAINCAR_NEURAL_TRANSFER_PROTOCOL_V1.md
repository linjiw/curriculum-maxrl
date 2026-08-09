# MountainCar Neural Curriculum-Transfer Protocol V1

Status: **registration-ready but unsealed; no registered V1 seed has been
run**.  This version implements a deterministic development-only seal command,
but no canonical lock has been created.  Seeds `17000..17002` are unreachable
until an exact source/runtime/protocol lock is supplied.  Confirmatory seeds
remain categorically unreachable and would require a reviewed later version.

## 1. Question and rationale

Sparse policy-gradient learning has two coupled failures.  If training always
requests the final flag, almost every rollout group can fail and practical
MaxRL supplies no update.  Uniform sampling spends equal probability on tasks
that may already be trivial, currently learnable, or still impossible.  The
frontier curriculum instead estimates where a 16-rollout group is likely to
contain the mixed outcomes that make the practical MaxRL gradient nonzero.

MountainCar is a useful independent test because its two-dimensional state,
three discrete actions, and momentum-building dynamics differ materially from
Acrobot.  The nested goals share the same dynamics and require the same skill:
build enough oscillatory momentum to travel farther right.  A task-agnostic
shared policy can therefore transfer updates obtained at easier thresholds to
the final flag.  Disjoint policies cannot transfer an update across threshold
slots.

The study asks two separate questions:

1. On the same shared actor, does frontier-u16 outperform uniform and
   hardest-only sampling at the same number of Gymnasium transitions?
2. Under the same outcome-independent uniform task schedule, does shared
   training outperform both an exact-total-parameter and an
   exact-active-capacity disjoint control on the final native goal?

No hindsight is allowed.  This removes relabeling as an alternative mechanism
and isolates curriculum-mediated transfer.

## 2. Gymnasium task family

Use the official `MountainCar-v0` dynamics, observation bounds, three actions,
and 200-step time limit.  Rewards from Gymnasium are retained for auditing but
do not define the study outcome.  Task `j` succeeds when a post-transition
position first satisfies

\[
  x_t \ge \tau_j,
  \qquad
  \tau=(-0.375,-0.250,-0.125,0,0.125,0.250,0.375,0.500).
\]

This is an evenly spaced `0.125` positional grid whose easiest target remains
strictly to the right of every possible reset position.  The predicates are
nested and change only scoring and early stopping, never
the simulator dynamics.  Training episodes stop at their first requested-goal
crossing.  Evaluation runs to the native horizon (or native termination),
records maximum position, and scores every threshold.  The shared actor uses
the same episode seeds and a fresh per-episode action RNG for all thresholds,
so its evaluated pass rates must be exactly nonincreasing with difficulty.

## 3. Actor and exact capacity controls

The categorical actor is

\[
 h=\tanh(oW_{in}+b), \qquad
 \pi(a\mid o)=\operatorname{softmax}(hW_{out}),
\]

with two inputs, hidden width `H`, three outputs, and no output bias.  A slot
therefore contains exactly

\[
  2H + H + 3H = 6H
\]

trainable parameters.  The shared actor never receives a threshold or task
identifier.

| Architecture | Slots × width | Total parameters | Active per task | Purpose |
|---|---:|---:|---:|---|
| shared H64 | `1 × 64` | 384 | 384 | proposed transfer model |
| disjoint-total H8×8 | `8 × 8` | 384 | 48 | exact total-parameter control |
| disjoint-active H64×8 | `8 × 64` | 3,072 | 384 | exact active-capacity control |

Random input weights, zero hidden biases, and zero output weights make every
architecture's initial policy exactly uniform.  Parameter and action RNGs are
independent.  Updates use a frozen copy of the group policy, sum every score
term without trajectory-length normalization or clipping, and apply plain SGD
ascent with learning rate `3e-4`.  This learning rate is a frozen V1
development choice, not an established MountainCar optimum; changing it after
development requires V2 and leaves all V1 confirmatory seeds untouched.

## 4. Practical MaxRL and frontier-u16

Every requested task produces `N=16` binary rollout outcomes.  With
`K=sum_i r_i`, the exact practical estimator used in all five conditions is

\[
  w_i = \mathbf{1}\{K>0\}\left(\frac{r_i}{K}-\frac1{16}\right).
\]

Both all-fail and all-pass groups have zero aggregate coefficients, while a
mixed group gives a potentially useful update.  No baseline, optimizer state,
hindsight term, or task-dependent reward is added.

For the frozen group policy, the applied ascent step is

\[
  \Delta\theta
  =\eta\sum_{i=1}^{16}w_i\sum_t
    \nabla_\theta\log\pi_\theta(a_{i,t}\mid o_{i,t}).
\]

The curriculum utility is not an unrelated heuristic.  For independent
Bernoulli successes with rate `p`, one half of the expected scalar coefficient
mass is

\[
\begin{aligned}
 \tfrac12\,\mathbb E\lVert w\rVert_1
 &=\mathbb E\!\left[\mathbf 1\{K>0\}
       \left(1-\frac{K}{16}\right)\right] \\
 &=\Pr(K>0)-\mathbb E[K]/16 \\
 &=1-(1-p)^{16}-p.
\end{aligned}
\]

Thus frontier-u16 directly targets tasks expected to supply usable practical-
MaxRL coefficient mass, instead of treating curriculum and estimator as two
separate mechanisms.

For a Thompson draw `p` from a discounted Beta model of the requested task,
the frontier utility is

\[
  u_{16}(p)=\max\{1-(1-p)^{16}-p,0\}.
\]

The MountainCar study uses probabilities proportional to `u16(p)^4`, mixed
with a `0.1` uniform floor.  The exponent is frozen at four because the nested
tasks share a sequential momentum-building skill; it concentrates sampling
without removing guaranteed revisits.  Discount is `0.7`.  Only original
requested-task outcomes update the teacher.

## 5. Five paired conditions

1. `frontier_shared_h64`
2. `uniform_shared_h64`
3. `hardest_shared_h64`
4. `uniform_disjoint_total_h8x8`
5. `uniform_disjoint_active_h64x8`

The first three isolate sampling on an identical shared actor.  The second,
fourth, and fifth isolate sharing under uniform sampling.  Their task at group
index `g` is drawn from one seed-indexed sequence that never observes rewards,
successes, episode lengths, or architecture.  Because first-hit stopping makes
the number of groups consumed by 500,000 transitions policy-dependent, each
cell consumes a possibly different-length prefix; all realized common-prefix
task IDs must be exactly equal.  The complete registered sequence has 31,250
entries, the maximum possible number of 16-episode groups needed to reach the
budget, and its hash is sealed per seed.

Shared versus disjoint-total rules out a total-parameter explanation but
differs in active width; shared versus disjoint-active rules out an
active-capacity explanation but gives the disjoint model more total
parameters.  Superiority to both on the primary final-goal metric is the
intended transfer triangulation.

## 6. Resource axis, evaluation, and primary metric

Development uses a nominal budget of 500,000 training transitions per run.
Only complete 16-episode groups are accepted, so terminal overshoot is at most
3,200 transitions.  Evaluation is triggered after crossing each nominal
100,000-transition checkpoint, with 32 fixed episodes per task.  The evaluated
policy is the right-continuous group-boundary policy using only updates whose
complete rollout group ends at or before that checkpoint: if a group ends
exactly at the checkpoint, evaluate its post-update policy; if it overshoots,
evaluate the saved pre-group parameter snapshot.  Evaluation uses a separate
environment and local per-episode RNGs and must restore the live training
parameters and leave environment-seed RNG and action RNG byte-identical.

Let `h_m,s(x)` be the pass rate at the hardest threshold `0.500` for method `m`
and seed `s`.  The sole primary metric is the piecewise-linear normalized AUC
on the **exact** common resource interval `[0,500000]`:

\[
  Y^{hard}_{m,s}=\frac1{500000}\int_0^{500000}h_{m,s}(x)\,dx.
\]

Target-uniform mean-pass AUC is a secondary, supporting description.  It may
help explain where transfer occurs, but it cannot authorize or rescue a claim
when a hardest-goal primary contrast fails.  Accordingly, claims are limited
to sample efficiency for reaching the native `0.500` goal; they do not imply
uniform improvement at every intermediate threshold.

For each of the four registered comparisons, the primary transfer estimand is
the paired-seed mean difference in `Y^hard`.  Development reports these values
only descriptively; a later locked confirmatory version would apply the family
rule in Section 9.

The curve is stored directly at the nominal coordinates
`[0,100000,...,500000]`.  The actual triggering group endpoints and the exact
evaluated parameter hashes are stored separately.  In particular, an update
from a terminal group ending after 500,000 cannot influence `h(500000)`.
Different episode lengths and complete-group overshoot therefore provide no
extra training information to the primary metric.

## 7. Seed separation and current execution boundary

- Development feasibility: paired seeds `17000..17002`.
- Future confirmatory block, reserved and untouched: `18000..18019`.
- Excluded integration smoke: seed `9053` only.

These blocks do not overlap legacy MountainCar `0..9` or registered Acrobot
training blocks through `16000..16019`.  Actor-action, environment-episode,
teacher, evaluation-episode, and evaluation-action RNG roots add `1,000,000`,
`2,000,000`, `4,000,000`, `6,000,000`, and `7,000,000`, respectively, to the
paired training seed.  These roots are mutually numerically disjoint and do
not overlap any reserved training block, including across consecutive paired
seeds.

The runner defaults to schedule inspection, which executes no environment.
The development-only seal command also creates no environment and does not run
a seed.  It binds the exact eleven-file source manifest (including both test
files), runtime, RNG ledger, protocol, ordered conditions, uniform schedule
hashes, primary estimand, and development seed block.  Development execution
revalidates this exact lock before actor/environment creation and before every
registered run boundary.  There is no acknowledgement bypass.  This V1
implementation cannot authorize a confirmatory seed.  The reusable core also
rejects every reserved seed before `gym.make`; a development seed requires the
seed- and lock-digest-bound internal authorization produced by the validated
runner path.

## 8. Development feasibility gates

The independent analyzer requires all 15 paired runs and verifies:

1. exact development-lock, source/runtime/config/seed/condition identity,
   including both tests, the package initializer, and its transitively imported
   framework modules;
2. complete raw group, transition, task, sampler, and optimizer accounting;
3. every training episode seed, step count, reward sum, prefix maximum,
   pre-final/final position, terminal flags, requested-goal success, and the
   first-hit early-stop invariant;
4. independent replay of every frontier, uniform, and hardest-only draw, plus
   the sealed uniform sequence and exact realized common prefix;
5. exact parameter and active-capacity identities;
6. zero hindsight candidates and zero relabeled groups;
7. evaluation episode/action seeds and every per-task maximum-position sample,
   from which all pass matrices, curves, and shared-actor CRN invariants are
   independently reconstructed;
8. exact hardest-goal primary and mean-pass supporting AUC reconstruction;
9. pooled coverage of all eight tasks in uniform/frontier cells and only task
   seven in hardest-only;
10. at least one applied update in each non-hardest condition (hardest-only is
   deliberately permitted to remain all-fail);
11. pooled presence of dead, mixed, and all-pass regimes; and
12. serial runtime projecting the future 100-run matrix to at most 18 hours.

The three-seed performance contrasts are descriptive only.  No observed sign,
effect size, or p-value is a development gate, and passing development does
not itself authorize a performance claim.

## 9. Reserved confirmatory framework (not implemented for execution)

If development passes and the implementation remains scientifically suitable,
review and seal an immutable source/runtime/schedule lock before any seed in
`18000..18019` is touched.  A future version may then run the same five cells
and test this frozen four-contrast family on hardest-goal AUC:

- frontier shared minus uniform shared;
- frontier shared minus hardest-only shared;
- uniform shared minus uniform disjoint-total; and
- uniform shared minus uniform disjoint-active.

Use exact two-sided paired sign-flip tests over all 20 seeds, Holm correction
at familywise `0.05`, and 20,000-resample paired-seed bootstrap intervals.
Directional support for each claim requires a primary mean improvement of at
least `0.03` plus Holm rejection.  Mean-pass AUC is reported as supporting only
and cannot change that decision.  All 100 runs form an all-or-nothing family:
no seed replacement, interim stopping, outcome-based exclusion, or
unregistered hyperparameter change is permitted.

## 10. Commands (inspection and future development only)

Inspect the schedule without executing Gymnasium:

```bash
python -m frontier_rl.examples.run_mountaincar_neural_transfer_v1 --mode schedule
```

Run an excluded nonregistered integration smoke, if desired:

```bash
python -m frontier_rl.examples.run_mountaincar_neural_transfer_v1 \
  --mode smoke --output /tmp/mountaincar_neural_transfer_v1_smoke.json
```

After review, create the development-only lock without executing Gymnasium:

```bash
python -m frontier_rl.examples.run_mountaincar_neural_transfer_v1 \
  --mode seal-development \
  --lock frontier_rl/examples/MOUNTAINCAR_NEURAL_TRANSFER_V1_LOCK.json
```

Then run development against that exact lock:

```bash
python -m frontier_rl.examples.run_mountaincar_neural_transfer_v1 \
  --mode development \
  --lock frontier_rl/examples/MOUNTAINCAR_NEURAL_TRANSFER_V1_LOCK.json \
  --output frontier_rl/examples/mountaincar_neural_transfer_v1_development.json
```

Independent verification is:

```bash
python -m frontier_rl.examples.analyze_mountaincar_neural_transfer_v1 \
  frontier_rl/examples/mountaincar_neural_transfer_v1_development.json \
  --lock frontier_rl/examples/MOUNTAINCAR_NEURAL_TRANSFER_V1_LOCK.json
```
