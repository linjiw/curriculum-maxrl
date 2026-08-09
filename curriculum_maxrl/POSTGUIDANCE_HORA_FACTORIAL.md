# Post-guidance HORA × MaxRL allocation factorial

## Status and scope

This is a **synthetic, post-guidance CPU experiment**, not a preregistered
confirmation and not a reproduction of the neural experiments in HORA.  It
uses `SkillChainEnv`, a simplified synchronous batch update, practical
group-size-specific MaxRL weights, and no hindsight relabeling.  Its main use
is to stress-test an experiment design that can later be moved to RLVR.

The published-style arm follows the same-step Beta-Binomial marginal in
[Wang et al., *Where to Spend Rollouts*](https://arxiv.org/abs/2605.07114):

\[
M_i^{\mathrm{hit}}(\ell)
= \mathbb{E}[p_i(1-p_i)^\ell \mid c_i].
\]

For practical MaxRL, the expected half coefficient mass at final group size
\(N\) is \(u_N(p)=1-(1-p)^N-p\).  Its one-rollout marginal is
\(p(1-p)^N\).  Because the \(G_0=4\) probes have already been spent and are
pooled into the final group, the mass-aware arm uses

\[
M_i^{\mathrm{mass}}(\ell)
= \mathbb{E}[p_i(1-p_i)^{G_0+\ell} \mid c_i].
\]

Calling this arm “mass-aware HORA” is shorthand for this new allocation
layer; it is not a method proposed or evaluated by the HORA authors.

## Frozen design

- Factorial: sampler `{uniform, u_16}` × allocation
  `{fixed N=16, published-HORA-style hit utility, mass-aware utility}`.
- Phase A: four probes per each of eight sampled tasks.
- Phase B: exactly 96 further rollouts, so every batch uses 128 completions
  and has average group size 16.
- Budget: 51,200 completions per cell, with checkpoints every 2,560.
- Seeds: 16 paired seeds (`0`–`15`).
- Posterior: same-step `Beta(1+c, 1+4-c)`.
- Task sampler: discounted-Beta Thompson draws with a 10% uniform floor;
  evidence decay is normalized by the number of observed completions.
- Update: all rollouts use one behavior-policy snapshot, followed by one
  synchronous sum of eight prompt-level practical-MaxRL gradients.
- Primary performance metric: normalized AUC of exact pass@8 versus sampled
  completions.  Mean-pass AUC is the required secondary safety metric.

Before execution, the frozen directional hypotheses were:

1. Mass-aware > hit utility in realized coefficient mass per completion.
2. Mass-aware > hit utility in pass@8 AUC.
3. `u_16` > uniform in pass@8 AUC, averaged over allocation arms.
4. Exploratory: the mass-aware benefit is smaller under `u_16` than uniform.

No direction was frozen for mean-pass AUC or final mean pass.

## Cell means

| Task sampler | Allocation | Coefficient L1 mass / completion | Mean-pass AUC | Pass@8 AUC | Final mean pass |
|---|---|---:|---:|---:|---:|
| uniform | fixed | 0.014056 | 0.649970 | 0.747888 | 0.966582 |
| uniform | HORA hit | 0.013760 | 0.674188 | 0.770674 | 0.967853 |
| uniform | mass-aware | 0.013535 | 0.679301 | 0.777228 | 0.967459 |
| \(u_{16}\) | fixed | 0.016387 | 0.699261 | 0.769007 | 0.978777 |
| \(u_{16}\) | HORA hit | 0.015841 | 0.710356 | 0.783049 | 0.977392 |
| \(u_{16}\) | mass-aware | 0.015454 | 0.713261 | **0.790005** | 0.976382 |

All accounting checks pass: each of 96 cells uses exactly 51,200
completions, all cells share the same completion checkpoints, all have exact
average \(N=16\), and the fixed arm always uses group size 16.  A serial
repeat is byte-identical to the four-worker result.

## Paired readout

The allocation main effects below average the two sampler arms within each
seed.  Exact sign-test values are descriptive because the hypotheses are
post-guidance.

| Contrast | Metric | Mean paired difference | Positive seeds | Exact two-sided sign p |
|---|---|---:|---:|---:|
| mass-aware − HORA hit | coefficient mass / completion | **−0.000306** | 0/16 | 0.000031 |
| mass-aware − HORA hit | mean-pass AUC | +0.004009 | 10/16 | 0.4545 |
| mass-aware − HORA hit | pass@8 AUC | **+0.006755** | 11/16 | 0.2101 |
| mass-aware − fixed | mean-pass AUC | **+0.021665** | 13/16 | 0.0213 |
| mass-aware − fixed | pass@8 AUC | **+0.025169** | 14/16 | 0.0042 |
| HORA hit − fixed | mean-pass AUC | +0.017656 | 12/16 | 0.0768 |
| HORA hit − fixed | pass@8 AUC | +0.018414 | 12/16 | 0.0768 |
| \(u_{16}\) − uniform, averaged over allocators | mean-pass AUC | **+0.039807** | 15/16 | 0.0005 |
| \(u_{16}\) − uniform, averaged over allocators | pass@8 AUC | **+0.015423** | 14/16 | 0.0042 |

Thus H2 and H3 meet their frozen directions.  H1 fails strongly: despite the
better learning curves, mass-aware allocation produces *less* realized
coefficient mass per completion in all 16 main-effect pairs.  H4 also fails:
the pass@8 interaction is small and opposite in mean sign (+0.000402; 7
positive and 9 negative seeds).

The final pass@8 values saturate near one, so the completion-indexed AUC is
the informative coverage statistic.  Mass-aware versus HORA hit also has a
small negative final-mean-pass difference (−0.000702, 2/16 positive), despite
positive mean-pass AUC; this is an early-learning versus terminal-saturation
tradeoff, not an across-the-board win.

## Interpretation

The defensible conclusion is narrow:

> In this synthetic fixed-budget setting, counting spent probes in the
> MaxRL-mass marginal improves early coverage relative to both fixed groups
> and published-HORA-style hit allocation, but the anticipated realized-mass
> mechanism is falsified.

The mass-aware allocator is substantially more concentrated: its observed
cell-level maximum group size averages about 76, versus about 57 for HORA
hit and 16 for fixed allocation; some tasks remain at the four-rollout probe
minimum.  A plausible **post-hoc** explanation is that this concentration
advances shared frontier skills even while reducing the realized scalar
coefficient mass per completion.  Other possibilities are posterior
miscalibration under a moving policy and a mismatch between the unconditional
\(u_N(p)\) identity and allocation after conditioning on realized probes.
The present experiment does not distinguish those explanations.

This result should be presented as exploratory appendix evidence or as the
motivation for a neural follow-up—not as a fourth confirmed contribution.
A next small-GPU experiment should retain the exact fixed-completion protocol,
report both learning AUC and realized coefficient mass, and include posterior
calibration plus an allocation-cap ablation to test whether extreme groups
drive the tradeoff.

## Reproduction

From the repository root:

```bash
python3 -m unittest -v curriculum_maxrl.test_postguidance_hora_factorial
python3 curriculum_maxrl/run_postguidance_hora_factorial.py --workers 4
```

Structured results are in
`curriculum_maxrl/results_postguidance_hora_factorial.json`.
