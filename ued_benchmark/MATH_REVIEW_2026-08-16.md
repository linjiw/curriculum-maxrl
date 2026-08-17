# Mathematical review: what curriculum-MaxRL actually says, and how it maps onto minimax

Written while the 32x1 sweep runs, before any of its results were read.

## 1. What the theory establishes

The paper's proposition is exact and narrow (`body_iclr.tex:119-144`). For the
**practical MaxRL estimator** — group of `N` i.i.d. binary rollouts, all-fail
groups dropped — the expected absolute coefficient mass is

    A_N(p) = E[ Σ_i |w_i| ] = 2(pass@N − pass@1) = 2{1 − (1−p)^N − p}

and the score is `u_N(p) = A_N(p)/2`. Its content:

- `A_N` is the **expected magnitude of the per-group gradient coefficients**
  the estimator emits on a task with pass rate `p`. It is a statement about
  the *estimator's own algebra*.
- It is zero at `p=0` (all-fail, dropped) and `p=1` (no contrast), and peaks
  at `p*_N = 1 − N^{−1/(N−1)}`.
- The paper is explicit that this is "an estimator-side diagnostic, not a
  theorem of learning progress" (`:9`).

**Coefficient activity is therefore a proxy for "how much gradient signal will
the deployed estimator produce here", derived from that estimator's algebra.**
It is not a proxy for regret, learning progress, or value error.

## 2. What the AMaze student actually is

The minimax student is **PPO with GAE and a learned critic** (`agents/ppo.py`),
not the MaxRL group estimator. It never forms groups of `N` binary rollouts,
never drops all-fail groups, and never uses `w_i`.

So on AMaze, `u_N(p)` is being used as a difficulty-targeting heuristic —
"prefer levels whose success rate is near `p*_N`" — **detached from the
estimator whose algebra derived it.** That is a legitimate curriculum heuristic
(the Acrobot dose–response shows the shape helps), but it is not the paper's
central mechanism, and we should not describe it as such.

## 3. Why MaxMC beats it at n_eval=1: signal, not shape

The winning score is (`ued_scores.py:196-210`):

    MaxMC(level) = mean_t [ max_return_seen(level) − V(s_t) ]

Three properties matter:

| property | MaxMC | coefficient activity |
|---|---|---|
| observations per level per visit | **every timestep** (256) | **one** terminal bit |
| domain of the signal | continuous | `{0, 1}` → posterior takes 2 values |
| memory | remembers best return ever achieved | Beta counts, undecayed |
| what it estimates | value **error** — where the critic is wrong | pass **rate** — where difficulty is moderate |

At `n_eval=1` a Bernoulli score is starved of information: it cannot rank two
unvisited levels, and cannot rank two once-visited levels with the same
outcome. Under rank-based prioritisation that is nearly all ties. MaxMC never
ties, because it reads the critic's disagreement across an entire trajectory.

The 4x8 configuration was an attempt to buy Bernoulli information with level
diversity. The sweep priced that trade at **−0.207 return, 0/5 seeds** — for
MaxMC as well as for us. So the loss was configurational, and the 32x1 test
now running asks whether *accumulated* posterior evidence over revisits can
supply the missing information without the diversity cost.

## 4. What the honest mapping of "MaxRL + curriculum" onto minimax would be

There are three ways to bring the idea in, and they are not equally faithful.

**(a) Level score only, PPO student unchanged — what we have run.**
Faithful to the *shape*, not the *mechanism*. Competes head-on with MaxMC on
signal quality and loses on information per visit. Its best case is parity at
equal diversity.

**(b) Success-rate-conditioned *weighting* of the regret score.**
Multiply the continuous MaxMC signal by `u_N(p̂)`:
`score = MaxMC(level) · u_N(p̂_level)`. This keeps MaxMC's per-timestep
information and continuous ranking while suppressing levels the posterior says
are already mastered (`p̂→1`) or hopeless (`p̂→0`). It is a *gate* on regret,
not a replacement for it. Costs one line; keeps `n_eval=1`. This is the most
promising cheap variant, because it fixes the information deficit instead of
paying for it.

**(c) Change the student to the MaxRL estimator — the faithful mapping.**
Run each level as a group of `N` rollouts (`n_eval=N`), compute binary
success, apply the group-relative MaxRL advantage `w_i` with all-fail dropping,
and let coefficient activity be exactly what the theory says it is: the
expected magnitude of the coefficients *this* estimator will emit. Then the
score and the learner share an algebra and the paper's central promise —
"evaluate data-selection methods with the estimator they ship" — is actually
instantiated on the benchmark. This is a real method change: it replaces GAE
with a group-relative binary-outcome estimator, and on a sparse-reward maze
with a 250-step horizon that is not obviously better for the *student*, only
more coherent for the *curriculum*.

## 5. Recommendation

Run in this order, each cheap and each answering a different question:

1. **32x1 accumulation** (running). Does information-over-time rescue the
   pure activity score? Cheapest; decides whether (a) has any life.
2. **(b) activity-gated MaxMC.** One line. If the gap to `plrMM32` closes or
   inverts, we have a defensible "MaxRL-derived difficulty gating improves
   regret-based replay" result at `n_eval=1`, upstream configs verbatim.
3. **(c) MaxRL-estimator student** as a separate, clearly-labelled study.
   This is the faithful mapping and the interesting science, but it changes the
   learner and so cannot be presented as "our score in their method"; it must
   be "our estimator+score vs their estimator+score", both at matched budgets.

What we must not do: keep re-tuning (a) until a seed lottery produces a win.
Five seeds at 5,000 updates cannot distinguish +0.05 from noise (paired SDs in
this sweep are 0.15–0.37), and every negative arm here is a real result about
where a Bernoulli score is starved.

## 6. Correction to how the lane has been described

Earlier documents in this lane call the AMaze arm "coefficient activity as a
drop-in replacement for the MaxMC PLR score", which is accurate operationally
but obscures that the student is PPO+GAE and the score is detached from its
deriving estimator. Any manuscript text about AMaze must state that plainly.
