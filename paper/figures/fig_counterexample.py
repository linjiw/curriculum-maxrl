#!/usr/bin/env python
"""Figure 1 - the same mean pass rate, two opposite worlds.

Two curriculum units, both with mean pass rate 1/2 at N=16.

  Level A   every task has p=1/2.  Groups come back mixed, so the estimator
            can form success-versus-failure contrast on nearly every one.
  Level B   half the tasks are mastered (p=1), half are impossible (p=0).
            Every group is unanimous: all-pass groups have no contrast and
            all-fail groups are dropped, so realized activity is exactly zero.

Any curriculum that scores a unit by f(mean pass rate) assigns these the same
value.  The estimator does not.

Numbers are exact, not simulated: Level A is Binomial(16, 1/2); Level B is a
half-half mixture of point masses at K=0 and K=16; activity is
A = 2(Pr[K>0] - E[K]/N) in both cases.
"""
import os
from math import comb

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE = "#2a78d6"
RED = "#c1272d"
GRAY = "#555555"

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})

N = 16
k = np.arange(N + 1)
piA = np.array([comb(N, i) * 0.5**N for i in k])          # all tasks p = 1/2
piB = np.zeros(N + 1); piB[0] = 0.5; piB[N] = 0.5          # mastered / impossible


def activity(pi):
    return 2.0 * (pi[1:].sum() - float(pi @ k) / N)


def plugin(pi):
    p = float(pi @ k) / N
    return 2.0 * (1.0 - p - (1.0 - p) ** N)


fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(7.2, 2.25), gridspec_kw={"width_ratios": [1.5, 1.0]})

# ------------------------------------------------------------------ Panel A
w = 0.42
axA.bar(k - w / 2, piA, width=w, color=BLUE, label="Level A: every task $p{=}\\frac{1}{2}$")
axA.bar(k + w / 2, piB, width=w, color=RED,
        label="Level B: half mastered, half impossible")
axA.set_xlabel("successes $k$ in a group of $N{=}16$")
axA.set_ylabel("$P(K{=}k\\mid z)$")
axA.set_xticks([0, 4, 8, 12, 16])
axA.set_ylim(0, 0.58)
axA.legend(loc="upper center", frameon=False, handlelength=1.2,
           borderaxespad=0.1, labelspacing=0.25, fontsize=7.2)
axA.text(8, 0.30, "same mean pass rate $\\bar p_z=\\frac{1}{2}$", fontsize=7.6,
         color=GRAY, ha="center", style="italic")
axA.set_title("A   two units, one mean, two count laws", loc="left", pad=6)

# ------------------------------------------------------------------ Panel B
labels = ["Level A", "Level B"]
truth = [activity(piA), activity(piB)]
naive = [plugin(piA), plugin(piB)]
x = np.arange(2)
axB.bar(x - 0.20, naive, width=0.38, color=GRAY, alpha=0.45,
        label="scored by $f(\\bar p_z)$")
axB.bar(x + 0.20, truth, width=0.38, color=[BLUE, RED],
        label="realized activity")
for xi, (nv, tv) in enumerate(zip(naive, truth)):
    axB.text(xi - 0.20, nv + 0.03, f"{nv:.2f}", ha="center", fontsize=7.2,
             color=GRAY)
    axB.text(xi + 0.20, tv + 0.03, f"{tv:.2f}", ha="center", fontsize=7.2,
             color=[BLUE, RED][xi], weight="bold")
axB.set_xticks(x); axB.set_xticklabels(labels)
axB.set_ylim(0, 1.62)
axB.set_ylabel("coefficient activity $A_N$")
axB.legend(loc="upper center", frameon=False, handlelength=1.2,
           borderaxespad=0.1, fontsize=7.2, ncol=1)
axB.set_title("B   what the estimator actually emits", loc="left", pad=6)

fig.tight_layout(pad=0.35, w_pad=1.3)
fig.savefig(os.path.join(HERE, "fig_counterexample.pdf"))
fig.savefig(os.path.join(HERE, "fig_counterexample.png"), dpi=150)
print(f"Level A: plug-in {naive[0]:.5f}  true {truth[0]:.5f}")
print(f"Level B: plug-in {naive[1]:.5f}  true {truth[1]:.5f}")
print("wrote fig_counterexample.pdf / .png")
