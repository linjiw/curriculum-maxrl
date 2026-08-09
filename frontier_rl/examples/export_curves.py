"""Export audited, retained training curves for the static project website.

This exporter deliberately reads the corrected result artifacts instead of
launching fresh, small-seed smoke runs.  In particular, the retired CartPole
smoke study is not exported, and MountainCar comes from the corrected ten-seed
shared-policy study.

Output: docs/curves.json

Run: python3 frontier_rl/examples/export_curves.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "curves.json"


DATASETS = {
    "skill_chain": {
        "artifact": "frontier_rl/examples/skill_chain_component_ablation.json",
        "x_key": "checkpoint_steps",
        "curve_key": "mean_pass_curve",
        "x_label": "trainer step (0–400)",
        "metric": "analytic target-uniform mean pass rate",
        "note": (
            "Corrected 12-paired-seed component study. The three displayed "
            "configurations are illustrative; the teacher arm uses gamma=1 and "
            "the full-stack arm uses gamma=4, so their gap is not a pure "
            "hindsight contrast."
        ),
        "cases": {
            "uniform": ("uniform_no_hindsight", "uniform"),
            "teacher": ("teacher_g1_no_hindsight", "exact-mass teacher, gamma=1"),
            "hindsight": (
                "teacher_g4_centered_hindsight",
                "exact-mass teacher, gamma=4 + centered hindsight",
            ),
        },
    },
    "grid_reach": {
        "artifact": "frontier_rl/examples/grid_reach_validation.json",
        "x_key": "x_steps",
        "curve_key": "mean_pass_curve",
        "x_label": "trainer checkpoint",
        "metric": "fixed-evaluation target-uniform mean pass rate",
        "note": "Corrected 10-seed goal-rewrite study.",
        "cases": {
            "uniform": ("uniform + maxrl", "uniform"),
            "teacher": ("teacher + maxrl", "exact-mass teacher"),
            "hindsight": (
                "teacher + maxrl + hindsight",
                "exact-mass teacher + verifier-valid hindsight",
            ),
        },
    },
    "mountaincar": {
        "artifact": "frontier_rl/examples/mountaincar_shared_validation.json",
        "x_key": "x_transitions",
        "curve_key": "mean_pass_curve",
        "x_label": "environment transitions (about 0–500k)",
        "metric": "custom-threshold target-uniform mean pass rate",
        "note": (
            "Corrected 10-paired-seed shared tile-policy study on official "
            "MountainCar-v0 dynamics; this metric is not standard episode return."
        ),
        "cases": {
            "uniform": ("uniform_shared", "uniform, shared tile policy"),
            "teacher": (
                "advmass_shared",
                "exact-mass teacher, gamma=4, shared tile policy",
            ),
            "hindsight": (
                "advmass_shared_hindsight",
                "gamma=4 + centered hindsight, shared tile policy",
            ),
        },
    },
}


def aggregate_case(case: dict, x_key: str, curve_key: str, label: str) -> dict:
    runs = case["runs"]
    curves = np.asarray([run[curve_key] for run in runs], dtype=float)
    xs = np.asarray([run[x_key] for run in runs], dtype=float)
    if curves.ndim != 2 or xs.shape != curves.shape:
        raise ValueError(f"inconsistent retained curve shapes for {label}")
    return {
        "label": label,
        "n_seeds": len(runs),
        "x": np.rint(xs.mean(axis=0)).astype(int).tolist(),
        "mean": np.round(curves.mean(axis=0), 6).tolist(),
        "lo": np.round(curves.min(axis=0), 6).tolist(),
        "hi": np.round(curves.max(axis=0), 6).tolist(),
    }


def main() -> None:
    out = {
        "_meta": {
            "schema": "curriculum-maxrl/website-curves/v2",
            "status": "corrected retained artifacts only",
            "band": "seed min-max envelope",
            "sampling_trace": "not retained in these corrected artifacts",
        }
    }
    for name, spec in DATASETS.items():
        artifact_path = ROOT / spec["artifact"]
        artifact = json.loads(artifact_path.read_text())
        methods = {}
        for method, (case_name, label) in spec["cases"].items():
            methods[method] = aggregate_case(
                artifact["cases"][case_name],
                spec["x_key"],
                spec["curve_key"],
                label,
            )
        out[name] = {
            "artifact": spec["artifact"],
            "status": "corrected retained artifact",
            "metric": spec["metric"],
            "x_label": spec["x_label"],
            "note": spec["note"],
            "sampling_trace": "not retained",
            "methods": methods,
        }
        print(
            f"{name}: {methods['uniform']['n_seeds']} retained seeds, "
            f"{len(methods['uniform']['mean'])} checkpoints"
        )
    unilab_curve_path = (
        ROOT / "frontier_rl" / "examples" / "unilab_stewart_native_v2_curves.json"
    )
    out["unilab_stewart"] = json.loads(unilab_curve_path.read_text())
    print("unilab_stewart: 3 retained native development seeds, 5 checkpoints")
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
