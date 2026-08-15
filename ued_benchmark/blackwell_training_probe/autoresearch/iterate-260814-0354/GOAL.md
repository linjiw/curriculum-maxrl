# Goal

Determine whether the isolated JAX 0.6.2 / CUDA 12.9 Blackwell lane can run
the authored FrontierRL training path without modifying the canonical Frontier
overlay or source-faithful JAX 0.4.31 lane.

The bounded metric is a frozen, exact 4x8 Frontier protocol: all remaining
removed JAX APIs must be mechanically modernized; upstream and project tests
must pass on an asserted CPU backend; JAX 0.6.2 CPU must match the JAX 0.4.31
one-update receipt; and at most one RTX 5090 PPO update may be attempted after
the CPU gate.  Any numerical, structural, API, or checkpoint mismatch closes
the GPU gate.  No OOD evaluation, multiple seeds, longer training, or paper
evidence is allowed.
