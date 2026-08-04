"""Pilot-0 instruments — the three load-bearing measurements that gate
Phase 1 (COSMOS3_RESPONSE.md II.3).  numpy-only; each instrument consumes
data the live backend already produces, so running Pilot 0 is: collect
rollouts with LiveRolloutBackend, feed them here, read the gates.

  0a  within-group variance   — group-based anything requires K to vary;
                                 a near-deterministic flow sampler collapses
                                 K to {0, N} and starves teacher + estimator
  0b  poison rate             — per-predicate-class precision/recall of the
                                 self-verifier vs oracle, with the
                                 success-enriched probe the mock pilot showed
                                 is mandatory (failure-heavy probes mis-prune
                                 clean classes on base rates alone)
  0c  surrogate fidelity      — cosine between the weighted-CFM update
                                 direction and a reference PG direction on
                                 the same groups (the V1-style probe; gate
                                 against the fresh-group cosine ~0.95)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from frontier_rl.adapters.cosmos_libero import PoisonRateMeter


# ---------------------------------------------------------------------------
# 0a — within-group variance
# ---------------------------------------------------------------------------
@dataclass
class GroupVarianceProbe:
    """Feed every group's (task_id, rewards, action chunks); read the gate.

    Two starvation signals, either one fails the gate:
      - reward degeneracy: K in {0, N} for (almost) every group — no
        contrast, no estimator signal, no posterior resolution;
      - action collapse: near-zero within-group action variance — the
        sampler is effectively deterministic, so identical (task, init)
        rollouts cannot diverge and reward degeneracy is structural, not
        bad luck.  Fix at the sampler (noise/temperature), not the teacher.
    """
    groups: int = 0
    contrasted: int = 0                      # 0 < K < N
    k_histogram: dict = field(default_factory=dict)
    action_stds: list = field(default_factory=list)

    def observe(self, rewards: np.ndarray,
                action_chunks: Optional[Sequence] = None) -> None:
        r = np.asarray(rewards, dtype=float)
        k = int(r.sum())
        self.groups += 1
        self.k_histogram[k] = self.k_histogram.get(k, 0) + 1
        if 0 < k < len(r):
            self.contrasted += 1
        if action_chunks is not None and len(action_chunks) > 1:
            # per-group std of the FIRST chunk across rollouts, mean over
            # dims — first-chunk spread is what distinguishes identical
            # (task, init) episodes before compounding drift kicks in
            first = np.stack([np.asarray(c[0], dtype=float)
                              for c in action_chunks if len(c)])
            self.action_stds.append(float(first.std(axis=0).mean()))

    def report(self, *, min_contrast_frac: float = 0.05,
               min_action_std: float = 1e-3) -> dict:
        contrast = self.contrasted / max(self.groups, 1)
        a_std = float(np.mean(self.action_stds)) if self.action_stds else None
        return {
            "groups": self.groups,
            "contrast_frac": contrast,
            "k_histogram": dict(sorted(self.k_histogram.items())),
            "mean_first_chunk_action_std": a_std,
            "gate_contrast": contrast >= min_contrast_frac,
            "gate_action_variance": (a_std is None
                                     or a_std >= min_action_std),
        }


# ---------------------------------------------------------------------------
# 0b — poison rate (success-enriched probe)
# ---------------------------------------------------------------------------
def run_poison_probe(episodes: Sequence[dict], self_verifier: Callable,
                     *, precision_gate: float = 0.9,
                     min_positive_support: int = 20) -> dict:
    """Score a self-verifier against oracle truth over a probe set.

    episodes: dicts with 'info' (verifier input) and 'oracle_predicates'
    (ground truth, from the sim's BDDL evaluation).  THE PROBE MUST BE
    SUCCESS-ENRICHED: with rare true achievements, false-positive
    opportunities dominate (~65:1 at p≈0.015 in the mock pilot) and clean
    classes fall below any gate on base rates alone.  Enforce that here:
    classes with fewer than `min_positive_support` oracle-positive examples
    are reported as UNMEASURED, not pruned — collecting more successes for
    them (e.g. from demos or easier init states) is Pilot-0b work, not a
    reason to shrink the vocabulary.
    """
    meter = PoisonRateMeter(precision_gate=precision_gate)
    positive_support: dict = {}
    for ep in episodes:
        oracle = set(ep["oracle_predicates"])
        meter.observe(set(self_verifier(ep["info"])), oracle)
        for p in oracle:
            c = PoisonRateMeter.predicate_class(p)
            positive_support[c] = positive_support.get(c, 0) + 1

    precision, recall = meter.precision(), meter.recall()
    all_classes = set(precision) | set(recall) | set(positive_support)
    measured = {c for c in all_classes
                if positive_support.get(c, 0) >= min_positive_support}
    allowed = {c for c in meter.allowed_vocabulary() if c in measured}
    return {
        "precision": precision,
        "recall": recall,
        "positive_support": positive_support,
        "unmeasured_classes": sorted(all_classes - measured),
        "allowed_vocabulary": sorted(allowed),
        "gate_any_class_usable": bool(allowed),
    }


# ---------------------------------------------------------------------------
# 0c — surrogate fidelity
# ---------------------------------------------------------------------------
def surrogate_fidelity(update_pairs: Sequence[tuple],
                       *, gate_cosine: float = 0.8) -> dict:
    """Cosine between surrogate and reference update directions per group.

    update_pairs: (surrogate_grad, reference_grad) flat vectors for the SAME
    group — surrogate = weighted-CFM direction (weights x per-sample CFM-loss
    gradients), reference = an exact-chain PG direction (ReinFlow noise-net,
    or exact score in a toy).  Report per-group cosine spread AND the cosine
    of the mean directions (V1's convention: per-group ~0.95, mean -> 1.000
    on-structure).  The gate is deliberately below V1's fresh-group number:
    0c runs on ~50 groups, and per-group cosine at that sample size wobbles.
    """
    pairs = [(np.asarray(s, float).ravel(), np.asarray(r, float).ravel())
             for s, r in update_pairs]
    pairs = [(s, r) for s, r in pairs
             if np.linalg.norm(s) > 0 and np.linalg.norm(r) > 0]
    if not pairs:
        return {"n": 0, "gate_fidelity": False}
    cos = [float(s @ r / (np.linalg.norm(s) * np.linalg.norm(r)))
           for s, r in pairs]
    mean_s = np.mean([s for s, _ in pairs], axis=0)
    mean_r = np.mean([r for _, r in pairs], axis=0)
    mean_cos = float(mean_s @ mean_r
                     / (np.linalg.norm(mean_s) * np.linalg.norm(mean_r)))
    return {
        "n": len(cos),
        "per_group_cosine_mean": float(np.mean(cos)),
        "per_group_cosine_std": float(np.std(cos)),
        "mean_direction_cosine": mean_cos,
        "gate_fidelity": mean_cos >= gate_cosine,
    }


def pilot0_verdict(report_0a: dict, report_0b: dict, report_0c: dict) -> str:
    """The go/no-go summary, gates in dependency order."""
    lines = ["Pilot 0 verdict:"]
    if not report_0a["gate_action_variance"] or not report_0a["gate_contrast"]:
        lines.append("  0a FAIL — sampler starves groups "
                     f"(contrast {report_0a['contrast_frac']:.2f}, "
                     f"action std {report_0a['mean_first_chunk_action_std']}). "
                     "Fix sampler noise/temperature BEFORE anything else.")
    else:
        lines.append(f"  0a PASS — contrast {report_0a['contrast_frac']:.2f}")
    if report_0b["gate_any_class_usable"]:
        lines.append(f"  0b PASS — relabel vocabulary: "
                     f"{report_0b['allowed_vocabulary']} "
                     f"(unmeasured: {report_0b['unmeasured_classes']})")
    else:
        lines.append("  0b FAIL — no predicate class survives the gate; "
                     "run oracle-relabel arm only, self-verified arm blocked.")
    if report_0c["gate_fidelity"]:
        lines.append(f"  0c PASS — mean-direction cosine "
                     f"{report_0c['mean_direction_cosine']:.3f}")
    else:
        lines.append("  0c FAIL/LOW — weighted-CFM direction diverges from "
                     "reference PG; Phase 1 may proceed but promote ReinFlow "
                     "chain likelihoods to the main line.")
    return "\n".join(lines)
