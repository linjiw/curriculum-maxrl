# Adapter audit vs validated design (after the shared-policy bug)

*2026-07-30. The delegated audit agent produced findings across three
runs but died to repeated API timeouts before banking; this file records
the critical checks, re-verified directly. Scope: the design decisions in
EVIDENCE/REPORT/GUIDE vs adapter defaults, prompted by the gym per-bin
parameter bug.*

## The contract-2 question under shared=True (gym_classic) — RESOLVED: NOT vacuous, but changed

Under `shared=True`, `TilePolicy.tile(obs, task_id)` ignores task_id, so
the relabel path's "recompute tiles against the achieved bin"
(`gym_classic.py:156`) produces tiles identical to the originals — the
conditioning-rewrite is a **no-op in feature space**. However contract 2
is not violated, because under a task-blind policy there IS no
conditioning to rewrite: the policy cannot act goal-relative, so
credited-task mismatch cannot corrupt features. What changes is the
INTERPRETATION of gym hindsight results: relabeling here re-credits
trajectories to an achievable bin (pure reward reassignment), not
conditioning-corrected imitation. The gym rung therefore tests relabel
DIRECTION and exactness, not contract 2. Contract 2's measured cost
(0.600 < 0.658) comes from grid_reach, whose policy IS goal-relative —
that attribution in the docs is correct and unaffected.
**Action taken: none needed in code; EVIDENCE/paper wording already
attributes contract-2 evidence to the gridworld. The gym adapter's
relabel comment updated to say "re-credit" would be cosmetic (PENDING).**

## Parameter sharing across bins — verified per adapter

- gym_classic (both spaces): FIXED (shared=True default; per-bin kept as
  the explicit control). Post-fix convergence study validates.
- grid_reach: PASS — theta indexed by goal-relative FEATURES
  (`theta[feat]`), shared across rings by construction; the task enters
  through the goal, exactly the validated pattern.
- skill_chain: PASS — theta indexed by skill (`theta[req]`); tasks
  share exactly the skills they overlap on (the compounding design).
- countdown_llm / cosmos_libero: PASS — single LLM/policy, task in the
  prompt/goal conditioning only.

## Other spot-checks (direct)

- Teacher defaults 0.7/0.1/1.0: PASS in frontier_rl/teacher.py and
  streaming.py; the stale verl_curriculum.py copy (decay 0.9) is
  flagged in VERL_AUDIT F4 — it lives in the maxrl repo, labeled
  legacy, not imported anywhere (grep-verified).
- Posterior hygiene: PASS — trainer.py observes before the relabel
  branch; the gate reads the posterior but never writes it.
- Estimator semantics: PASS — maxrl_weights drop-K=0; positive_part
  maxrl-only (grpo/rloo raise/ignore per config comment).
- Eval honesty: eval_pass_rates rolls FRESH episodes through
  rollout_group — PASS, with the caveat that it consumes env RNG state
  (evaluation advances the same RNG stream as training; a fixed eval
  seed would be cleaner — MINOR, PENDING).

## PENDING (from the lost agent runs, to complete opportunistically)

- verl_integration/ copy vs live maxrl drift (partially covered by
  VERL_AUDIT F4/F10).
- cosmos_live wave-loop edge cases beyond the earlier verification.
- The eval-RNG-stream cleanliness fix above.
