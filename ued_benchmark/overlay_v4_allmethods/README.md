# Overlay v4 — coefficient activity as a first-class UED score

Promotes the curriculum-MaxRL Frontier score from a PLR-runner sidecar into
minimax's shared UED score path, so every runner can use one implementation.

Applies to upstream `d053054c5290a04c1c4cd8b55704d999cad73e30`, on top of the
v3 Frontier overlay. Patch: `allmethods.patch` (3 files, 126 added lines).
Working clone: `/data/robotixx/ued_bench/src/minimax-frontier-v4-allmethods-d053054`.
The verified v3 clone is **not modified**; its contract hash
`5868d346…` still fail-closes as before.

## What changed

| file | change |
|---|---|
| `util/rl/ued_scores.py` | `UEDScore.COEFFICIENT_ACTIVITY = 10`; `compute_coefficient_activity(batch)`; module-level params with `set_coefficient_activity_params()` / `get_coefficient_activity_params()`; one dispatch branch in `_compute_ued_scores` |
| `arguments.py` | frontier knobs added to the PAIRED subparser (`--paired_frontier_*`) |
| `runners/paired_runner.py` | accepts the frontier knobs and configures the score at init |

The score itself is unchanged: `sparse_goal_stream_counts` semantics are
preserved exactly — each evaluation stream contributes at most one Bernoulli,
observed if any `done` occurs and successful if any terminal reward exceeds the
threshold. The params are static Python values baked in at trace time, so they
must be set before the first jitted score call; runners do this at init.

## Why v3's design needed promoting

In v3, `coefficient_activity` was a string compared inside `PLRRunner`
(`use_frontier_activity = ued_score == 'coefficient_activity'`), with
`UEDScore.RETURN` used as a harmless enum placeholder and the scores overridden
later. That works for PLR and, because ACCEL *is* `PLRRunner` with a mutation
function, it works for ACCEL unchanged — verified. But it is invisible to any
other runner, so PAIRED could never reach it.

## PAIRED: attempted, and blocked upstream

The intended method was **activity-targeted environment design** — reward the
teacher for generating levels whose pass rate sits near the score's peak,
rather than for maximal relative regret. The integration is complete and
imports cleanly, but it cannot be run:

**Upstream PAIRED does not support `n_eval > 1`.** Its teacher rollout is sized
to `n_parallel` while the student batch is `n_parallel x n_eval`, so
`compute_ued_scores` returns one score per *level group* rather than per
designed level and the teacher reward shape no longer matches. It fails with

```
vmap got inconsistent sizes ... most axes had size 4 ... one axis had size 32
```

This was confirmed with the **unmodified `relative_regret` baseline** at the
same shape, so it is an upstream structural limit, not a defect in this
overlay.

Consequence: with `n_eval` forced to 1, a success-rate score sees a single
Bernoulli per designed level, so `E[u_N(p)]` collapses to two values — a binary
"student solved / student failed" teacher reward. That is essentially the
existing minimax-adversary (`neg_return`) objective and would not be a new
method. **PAIRED is therefore out of scope** unless the teacher rollout is
modified to support per-level replication, which would make the comparison
non-verbatim against upstream.

## Where the idea does live

| method | host | status |
|---|---|---|
| PLR / robust PLR | `PLRRunner` | works; best arm in the first development sweep |
| ACCEL | `PLRRunner` + mutation fn | works unchanged; verified 200-update smoke |
| Parallel PLR / ACCEL | `PLRRunner`, `use_parallel_eval=True` | expected to work; not yet exercised |
| PAIRED | `PAIREDRunner` | blocked upstream, see above |
| DR | `DRRunner` | no curriculum; it is the control |

This is a reasonable scope rather than a retreat: the replay-based family is
where minimax's strongest maze results live, and a level score is the natural
intervention point there.
