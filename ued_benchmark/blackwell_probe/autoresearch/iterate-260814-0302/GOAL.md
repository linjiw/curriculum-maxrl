# Goal

Determine the minimum isolated compatibility work needed to import the pinned
Frontier-patched `minimax` tree and compile/execute one AMaze reset and one step
on the local RTX 5090 with JAX 0.6.2.

Success metric: `AMAZE_JIT_OK` on the GPU with `PYTHONPATH` unset and XLA
preallocation disabled. Scope excludes training, benchmark evidence, Hopper,
the source-faithful environment, and edits to the canonical overlay.
