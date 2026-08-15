# Results

## Earliest divergence

The initial state is 92/92 records byte-exact.  Task RNGs, selected levels,
level indices, observations, rewards, dones, actions, and minibatch permutation
are also exact.  Sampling is ruled out.

The earliest frozen-tolerance failure is the recurrent LSTM carry after the
warmup rollout: nine elements fail, with maximum absolute error
`0.00012597441673278809`.  The same carry failure appears in the cycle-two PPO
forward pass.  Classification: `forward_or_gemm_recurrent_carry`.

Normalized-advantage and actor elements subsequently fail with maximum error
`0.0014483332633972168`.  Scalar losses, raw gradients, global norm, and
clipping pass.  Both clip factors equal one.

## Explanation of the prior parameter failure

The CPU/GPU `fc_pi_1.bias` gradient absolute sums are respectively
`1.3168447027256391e-09` and `7.101176407786625e-06`.  Both raw gradient trees
still pass the previously frozen GPU tolerance.  First-step Adam turns them
into update absolute sums `3.9505010374085714e-08` and
`0.00020396194759086939`, causing the aggregate gate failure.

An independent CPU calculation of the first-step Adam equation from the saved
post-clipping gradient matches the saved GPU proposal within
`8.185452315956354e-12`.  The diagnostic also checks that the captured clipped
gradient equals the raw gradient times the captured factor.  An active-clipping
contract case (raw norm `5.0`, clipped norm `0.5`) passes and would reject a
formula based on the raw gradient.  This is upstream numerical drift amplified
by Adam, not an Adam arithmetic or clipping mismatch.

## Safety result

- Optimizer applications: 0.
- Parameter mutations: 0.
- Cycle-two runner steps: 0.
- Cycle-two agent updates: 0.
- GPU component captures: 1.
- GPU PPO updates: 0.
- OOD/multiseed/performance/evidence endpoints: 0.
- PID 2786996 present before and after: yes.
