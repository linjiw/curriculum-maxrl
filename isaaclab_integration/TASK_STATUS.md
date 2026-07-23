# Frontier RL × Isaac Lab task integration status

Status date: 2026-07-23. "Integrated" here means the Frontier teacher is
connected to a task-specific difficulty actuator, verifier, checkpoint path,
and unbiased evaluation path. It is stricter than the older
`skills/agentic-rl/isaaclab-task-catalog`, whose "ready" flag refers to that
separate outer-loop manager and knob registry.

| task family | Frontier difficulty axis and verifier | implementation status | evidence |
|---|---|---|---|
| Anymal-C rough velocity locomotion | terrain row; `tile`, `survival`, or endpoint `distance` | Ladder step 1 implemented for RSL-RL: five arms, exact teacher resume, fixed-level probe, analyzer | 16 CPU tests; control-arm simulator retry active, no completed artifact |
| Other manager-based rough velocity robots | terrain row; task-specific locomotion predicate | structurally reusable, but not configured or tested per robot | none |
| Anymal-C flat velocity locomotion | per-env commanded-speed bin; tracking predicate | Ladder step 2 not implemented; requires a custom command term because stock command ranges are global | none |
| Lift/reach/pick/place/stack manipulation | goal distance/object state; explicit task success where available | not integrated; verifier functions in task MDPs are useful inputs, but no Frontier bin assignment, fixed-grid probe, or estimator path exists | none |
| Dexterous/in-hand and factory/contact tasks | object/assembly progress and domain parameters | not integrated; task-specific verifier and observable difficulty axes still need design | none |
| Navigation and aerial/drone tasks | goal/route/wind or obstacle bins | not integrated | none |
| Direct-workflow and classic tasks | task-specific | unsupported by this manager-term integration; separate `frontier_rl` Gym adapters do not make Isaac Lab task IDs integrated | none |

## Curriculum-to-task contract

For every new task, the implementation must provide all of the following:

1. An observable or inferable per-environment difficulty variable that a reset
   or command term can actuate without partitioning the shared policy.
2. A binary task predicate for teacher evidence. Dense shaped reward may remain
   in PPO, but it is not a substitute for the Bernoulli posterior.
3. A fixed-distribution evaluator that gives every checkpoint the same task
   bins and random seed and reports a macro average over bins.
4. Policy and teacher state restored atomically from the same checkpoint.
5. Control, scripted, uniform, stock-adaptive (when available), and teacher
   arms with artifact-based run completion checks.

## Current gate

The next implementation step is not another task family. The active
exclusive-GPU retry must complete all five rough-locomotion arms, followed by
fixed-grid evaluation. Ladder step 2 should begin only after P-A demonstrates
external frontier targeting and the step-1 mechanics produce valid artifacts.
