"""frontier_rl — the curriculum-MaxRL training schedule as a reusable framework.

The default packaged loop, environment-agnostic:

  1. TEACHER   — Thompson sampling over a decayed Beta posterior per task,
                 score = practical-MaxRL normalized coefficient half-mass
                 ν_N(p) = pass@N − pass@1, raised to gamma, plus a floor.
  2. ESTIMATOR — practical MaxRL centered/drop advantages by default;
                 raw, full-control-variate, GRPO, and RLOO baselines included.
  3. HINDSIGHT — dense relabeling of all-fail groups to verified sub-goals;
                 the destination update is generally off-policy even when
                 every relabeled reward is semantically exact.
  4. LOOP      — group rollouts → teacher.observe → advantages (+relabels)
                 → user-supplied policy update.

To plug in a new environment (gym task, robotics sim), implement the
`TaskSpace` protocol in `interfaces.py` — the trainer never imports your
simulator. See `adapters/` for three references, from a 40-line toy to a
gym-style continuous-control task.
"""

from frontier_rl.interfaces import TaskSpace, GroupResult, Policy
from frontier_rl.teacher import FrontierTeacher
from frontier_rl.estimators import (
    grpo_weights,
    maxrl_full_cv_weights,
    maxrl_raw_weights,
    maxrl_weights,
    rloo_weights,
)
from frontier_rl.trainer import FrontierTrainer, TrainerConfig

__all__ = [
    "TaskSpace", "GroupResult", "Policy",
    "FrontierTeacher", "FrontierTrainer", "TrainerConfig",
    "maxrl_weights", "maxrl_raw_weights", "maxrl_full_cv_weights",
    "grpo_weights", "rloo_weights",
]
