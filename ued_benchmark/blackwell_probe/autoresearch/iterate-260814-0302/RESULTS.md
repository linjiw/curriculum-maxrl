# Results

| Iteration | Observation | Decision |
|---|---|---|
| 1 | JAX/JAXlib 0.6.2 detected the RTX 5090 and compiled a tiny operation. | Keep the CUDA 12.9 probe base. |
| 2 | `minimax` import failed first on missing `chex`; the upstream NumPy pin conflicts with JAX 0.6.2. | Install a separately pinned modern import stack; do not alter the faithful lane. |
| 3 | Import and AMaze reset passed, but step failed because JAX 0.6 removed `jax.tree_map`. | Patch only the two calls reached by AMaze auto-reset. |
| 4 | One reset and one step compiled/executed on GPU; 3/3 static contract tests passed. | Keep the two-line modernization patch as an engineering probe. |

Final runtime result:

```text
AMAZE_JIT_OK {"backend":"gpu","device_kind":"NVIDIA GeForce RTX 5090","jax":"0.6.2","jaxlib":"0.6.2","obs_image_shape":[5,5,3],"next_obs_image_shape":[5,5,3],"state_time":1}
```

No training or benchmark evidence was produced. Thirty-three removed
`jax.tree_map` calls remain outside the narrow AMaze path.
