# curriculum_maxrl — teacher-guided curriculum for MaxRL

Research prototype exploring the integration of curriculum learning
(teacher–student, ZPD/learnability targeting) with the MaxRL objective
(arXiv:2602.02710). Everything here runs on CPU with numpy only.

## Contents

| file | purpose |
|---|---|
| `RESEARCH.md` | verified deep-research synthesis of modern curriculum RL and where MaxRL fits |
| `THEORY.md` | coefficient-mass analysis, derived teacher utility, myopic fixed-p rollout allocation |
| `DESIGN.md` | integration design, hypotheses, and validation results |
| `maze_gpu/` | GPU testbed (tiny transformer on 17×17 mazes, goal-distance curriculum, pass@k eval) |
| `testbed.py` | skill-chain environment (binary verifier rewards, exact score functions) |
| `estimators.py` | REINFORCE / RLOO / GRPO / MaxRL per-group advantage weights |
| `teachers.py` | Uniform / ZPD-band / ALP / MaxRL-frontier teachers + adaptive rollout allocation |
| `run_experiment.py` | teacher × estimator sweep (final performance) |
| `run_speed.py` | learning-speed (AUC, steps-to-frontier) + adaptive-N comparison |
| `verl_curriculum.py` | compatibility re-export of the canonical `verl_integration/` module |
| `test_verl_curriculum.py` | CPU compatibility tests for the verl integration module |

## Quick start

```bash
python3 run_experiment.py --steps 400 --seeds 5   # ~1 min on 8 cores
python3 run_speed.py
python3 test_verl_curriculum.py
```

## Core idea in one line

The MaxRL estimator's own math defines the curriculum: the expected total
scalar coefficient magnitude a prompt receives from a group of N rollouts is
exactly `2·(pass@N − pass@1)`—twice the probability it is solvable within N
attempts but not within one (THEORY.md). The teacher prioritizes prompts using
the corresponding half-mass utility, with Thompson-style draws from discounted
Beta pseudo-counts. Under fixed supplied pass rates, feasible integer bounds,
and a fixed one-step budget, greedy water-filling on `p(1−p)^N` maximizes this
proxy. At N=2 the half-mass utility equals SFL's learnability `p(1−p)`;
`u_1=0`. RLOO's expected coefficient mass is `2p(1−p)`, proportional to
the same score, though the estimators and objectives remain distinct.
