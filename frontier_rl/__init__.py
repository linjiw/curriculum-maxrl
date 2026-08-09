"""frontier_rl — the curriculum-MaxRL training schedule as a reusable framework.

The validated algorithm (REPORT.md), environment-agnostic:

  1. TEACHER   — Thompson-style draws from discounted Beta pseudo-counts,
                 utility = half the expected scalar coefficient mass
                 u(p)=pass@N-pass@1 (PROOFS.md P2), plus a uniform floor.
  2. ESTIMATOR — practical dropped-group MaxRL weights (effective order N-1);
                 exact order-N variants and GRPO/RLOO are also exposed.
  3. HINDSIGHT — relabel dead groups to verified achieved sub-goals; exactness
                 is guaranteed by the law/moment conditions in Proposition 6.
  4. LOOP      — group rollouts → teacher.observe → advantages (+relabels)
                 → user-supplied policy update.

To plug in a new environment (gym task, robotics sim), implement the
`TaskSpace` protocol in `interfaces.py` — the trainer never imports your
simulator. See `adapters/` for three references, from a 40-line toy to a
gym-style continuous-control task.
"""

from frontier_rl.interfaces import TaskSpace, GroupResult, Policy
from frontier_rl.teacher import FrontierTeacher
from frontier_rl.estimators import (grpo_weights, maxrl_success_weights,
                                    maxrl_unbiased_cv_weights, maxrl_weights,
                                    rloo_weights)
from frontier_rl.trainer import FrontierTrainer, TrainerConfig

__all__ = [
    "TaskSpace", "GroupResult", "Policy",
    "FrontierTeacher", "FrontierTrainer", "TrainerConfig",
    "maxrl_weights", "maxrl_success_weights", "maxrl_unbiased_cv_weights",
    "grpo_weights", "rloo_weights",
]
