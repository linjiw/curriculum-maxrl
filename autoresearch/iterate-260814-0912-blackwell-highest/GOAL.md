# Goal: bounded highest-precision OptimizedLSTMCell compatibility gate

**Started:** 2026-08-14 05:12 America/New_York

Test the smallest checkpoint-preserving Blackwell compatibility intervention:
run only the two `OptimizedLSTMCell` affine dots at highest matmul precision,
then require a frozen JAX 0.6.2 CPU two-cycle/one-update parity gate against
the JAX 0.4.31 reference before permitting at most one RTX 5090 update.

Keep the source-faithful JAX 0.4.31 lane, canonical Frontier overlay, and
Hopper files unchanged. Do not run OOD evaluation, multiple seeds, throughput,
benchmark, performance, long-training, or paper-evidence endpoints. Preserve
GPU PID 2786996 and keep cache/temp writes under `/data`.
