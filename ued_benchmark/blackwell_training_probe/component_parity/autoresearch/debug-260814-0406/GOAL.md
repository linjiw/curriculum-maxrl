# Goal

Find the earliest CPU-versus-RTX-5090 divergence that explains the rejected
Frontier one-update checkpoint, without applying another optimizer update.

Freeze the inputs, stage order, and prior tolerances before GPU execution;
replay the exact initial checkpoint and task/action streams; capture rollout,
PPO forward/loss, raw gradients, clipping, and Adam proposals; and fail closed
on any divergence.  No OOD evaluation, multiple seeds, throughput, benchmark,
or paper evidence is permitted.
