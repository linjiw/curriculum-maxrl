# ICRA BARN evidence loop

- Scope: CPU-only canonical BARN/Gazebo backend, preregistration freeze,
  four-arm paired campaign, mandatory group-size ablation, and frozen gate.
- Keep metric: acceptance tests in `CODEX_GOAL_ICRA_2026-08-11.md`.
- Hard rejects: any CUDA/GPU process, outcome peeking, under-seeded gate,
  post-freeze protocol mutation outside a dated amendment, or scope creep.
- Iteration budget: default autoresearch bound of 25 build/verify decisions.
- Iteration 1: retained canonical UT Austin BARN archive and canonical ROS/ROS2
  verifier semantics; discarded Isaac Lab because the local GPU is embargoed.

