"""Branching task pool: the substrate that can separate availability from transfer.

The shipped SkillChainEnv is three LINEAR chains, where task level l requires
skills 1..l. There, the number of downstream tasks a task unlocks is a
deterministic function of its level, and level determines its pass rate --
corr(p, C) = 0.89, and only 2 of 25 pass-rate bins hold more than one distinct
C. So no structural compounding term can carry information u_N(p) lacks, and
the utility audit's "factorization not needed" verdict is a statement about
linear chains, not about compounding.

This pool breaks that confound by construction. Tasks live on a forest whose
nodes have *heterogeneous branching factor*: some skills gate many descendants,
others gate none. Two tasks can therefore require the same number of skills --
hence carry the same pass rate -- while differing by an order of magnitude in
how many downstream tasks their skills unlock.

Design
------
- A forest of `n_roots` trees over skills. Each tree node is one skill.
- Node `s` at depth `d` gets `branch(d, tree)` children, where the branching
  factor differs per tree: "bushy" trees fan out, "spindly" trees do not.
- A task is a root-to-node path: it requires exactly the skills on that path,
  so its pass rate is `prod(q_s)` over the path -- depth alone sets difficulty.
- `C(x)` = number of strict descendants of `x`'s terminal node = the tasks
  whose pass rate rises when `x`'s skills improve. Depth fixes `p`; subtree
  size varies independently, which is exactly the separation the audit needs.

Everything else -- softmax skill policy, rollout, apply_gradient -- is
inherited from SkillChainEnv unchanged, so the estimator, learning rule and
exact `true_pass_rates` are identical to the linear-chain audit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "curriculum_maxrl"))
from testbed import SkillChainEnv  # noqa: E402


class BranchingPoolEnv(SkillChainEnv):
    """Skill forest with heterogeneous branching; API-compatible with SkillChainEnv."""

    def __init__(self, depth: int = 4, n_actions: int = 10,
                 branch_by_tree=(3, 1, 3, 1), init_logit_correct: float = 0.0,
                 seed: int = 0):
        # Build structure first, then hand the parent the sizes it expects.
        self.depth = depth
        self.branch_by_tree = tuple(branch_by_tree)
        self.n_actions = n_actions
        self.init_logit_correct = init_logit_correct
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        parent: list[int] = []          # skill id -> parent skill id (-1 at root)
        node_depth: list[int] = []
        node_tree: list[int] = []
        frontier_by_tree: list[list[int]] = []

        for t, _b in enumerate(self.branch_by_tree):
            root = len(parent)
            parent.append(-1); node_depth.append(1); node_tree.append(t)
            frontier_by_tree.append([root])

        for d in range(2, depth + 1):
            for t, b in enumerate(self.branch_by_tree):
                nxt = []
                for node in frontier_by_tree[t]:
                    for _ in range(b):
                        child = len(parent)
                        parent.append(node); node_depth.append(d); node_tree.append(t)
                        nxt.append(child)
                frontier_by_tree[t] = nxt

        self.parent = np.array(parent)
        self.node_depth = np.array(node_depth)
        self.node_tree = np.array(node_tree)
        self.n_skills = len(parent)

        # one task per node: the root-to-node path
        self.tasks: list[np.ndarray] = []
        for s in range(self.n_skills):
            path, cur = [], s
            while cur != -1:
                path.append(cur); cur = int(self.parent[cur])
            self.tasks.append(np.array(path[::-1]))
        self.task_level = [int(self.node_depth[s]) for s in range(self.n_skills)]
        self.n_tasks = len(self.tasks)

        self.theta = np.zeros((self.n_skills, self.n_actions), dtype=np.float64)
        self.theta[:, 0] = self.init_logit_correct

        self._descendants = self._count_descendants()

    def _count_descendants(self) -> np.ndarray:
        """Strict descendant count per node, computed deepest-first."""
        n = np.zeros(self.n_skills)
        for s in np.argsort(-self.node_depth):
            par = int(self.parent[s])
            if par != -1:
                n[par] += n[s] + 1
        return n

    def compounding(self) -> np.ndarray:
        """C(x): 1 + number of tasks whose pass rate rises if x's skills improve."""
        return 1.0 + self._descendants


def make_branching(seed: int) -> BranchingPoolEnv:
    """Bushy/spindly forest: same depths (hence same difficulty ladder) but
    subtree sizes differing by an order of magnitude at equal depth."""
    return BranchingPoolEnv(depth=4, branch_by_tree=(3, 1, 3, 1), seed=seed)


if __name__ == "__main__":
    env = make_branching(0)
    C = env.compounding()
    d = np.array(env.task_level)
    print(f"tasks={env.n_tasks} skills={env.n_skills} depths={sorted(set(d.tolist()))}")
    for dep in sorted(set(d.tolist())):
        cs = sorted(set(C[d == dep].tolist()))
        print(f"  depth {dep}: n={int((d == dep).sum())}  distinct C at this depth: {cs}")
