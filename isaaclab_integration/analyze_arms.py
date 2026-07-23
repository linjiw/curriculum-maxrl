"""Analyze the 5-arm frontier-curriculum experiment from TensorBoard event files.

Pre-registered readout (ISAACLAB_DESIGN.md §2 + INTEGRATION.md §4):
  P-A mechanism gate (teacher arm only): ZPD targeting ratio — fraction of
      sampling mass on bins with p̂∈[0.2,0.8] vs the uniform share, from the
      logged dead/mastered/effective-bins telemetry + teacher_state.json.
  P-B outcome: terrain-level progression (Curriculum/*/mean_bin AUC + final)
      and tracking-reward AUC, per arm, wall-clock- and iteration-matched.
  P-C retention: Episode_Reward/track_lin_vel_xy_exp at the end vs mid-run —
      regressions beyond the noise band flag frontier-sampling forgetting.

Needs tensorboard — run with the container's kit python:
  docker exec isaac-lab-base /isaac-sim/python.sh \
      /workspace/isaaclab/scripts/curriculum-maxrl/isaaclab_integration/analyze_arms.py \
      --log_root /workspace/isaaclab/logs/rsl_rl/anymal_c_rough
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError as e:
    raise SystemExit("tensorboard not importable — run in-container with /isaac-sim/python.sh") from e

ARMS = ("control", "greedy", "scripted", "uniform", "teacher")
MEAN_BIN_TAGS = ("Curriculum/terrain_levels/mean_bin", "Curriculum/terrain_levels")
TRACK_TAG = "Episode_Reward/track_lin_vel_xy_exp"
REW_TAG = "Train/mean_reward"


def load_scalars(run_dir: str) -> dict[str, list[tuple[int, float]]]:
    acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    acc.Reload()
    return {tag: [(e.step, e.value) for e in acc.Scalars(tag)] for tag in acc.Tags()["scalars"]}


def auc(series: list[tuple[int, float]]) -> float:
    """Step-weighted mean value (trapezoid AUC / total steps)."""
    if len(series) < 2:
        return series[0][1] if series else float("nan")
    total, span = 0.0, series[-1][0] - series[0][0]
    for (s0, v0), (s1, v1) in zip(series, series[1:]):
        total += 0.5 * (v0 + v1) * (s1 - s0)
    return total / span if span else series[-1][1]


def first_tag(scalars: dict, candidates) -> list[tuple[int, float]]:
    for tag in candidates:
        if tag in scalars:
            return scalars[tag]
    return []


def mean_present(values) -> float:
    vals = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(vals) / len(vals) if vals else float("nan")


def parse_run(run_dir: str):
    """Arm from params/arm.yaml (authoritative), seed from params/agent.yaml."""
    arm_yaml = os.path.join(run_dir, "params", "arm.yaml")
    if not os.path.exists(arm_yaml):
        return None  # not one of our runs
    with open(arm_yaml) as f:
        arm_match = re.search(r"^arm:\s*(\w+)", f.read(), re.M)
    seed = -1
    agent_yaml = os.path.join(run_dir, "params", "agent.yaml")
    if os.path.exists(agent_yaml):
        with open(agent_yaml) as f:
            seed_match = re.search(r"^seed:\s*(\d+)", f.read(), re.M)
        if seed_match:
            seed = int(seed_match.group(1))
    return (arm_match.group(1), seed) if arm_match else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_root", default="logs/rsl_rl/anymal_c_rough")
    ap.add_argument("--retention_tau", type=float, default=None,
                    help="Absolute fixed-grid easy-level regression tolerance.")
    args = ap.parse_args()

    rows = defaultdict(list)  # arm -> [(seed, metrics dict)]
    for run_dir in sorted(glob.glob(os.path.join(args.log_root, "*_*"))):
        parsed = parse_run(run_dir)
        if not parsed or not glob.glob(os.path.join(run_dir, "events.out.tfevents.*")):
            continue
        arm, seed = parsed
        s = load_scalars(run_dir)
        mean_bin = first_tag(s, MEAN_BIN_TAGS)
        track = s.get(TRACK_TAG, [])
        metrics = {
            "bin_auc": auc(mean_bin),
            "bin_final": mean_bin[-1][1] if mean_bin else float("nan"),
            "track_auc": auc(track),
            "track_final": track[-1][1] if track else float("nan"),
            "track_mid": track[len(track) // 2][1] if track else float("nan"),
            "rew_final": s.get(REW_TAG, [(0, float("nan"))])[-1][1],
        }
        # P-A inputs from the teacher arm's telemetry
        if arm == "teacher":
            state_path = os.path.join(run_dir, "curriculum_teacher", "teacher_state.json")
            teacher_state = None
            if os.path.exists(state_path):
                with open(state_path) as f:
                    teacher_state = json.load(f)
                if "pass_rates" in teacher_state:
                    metrics["final_pass_rates"] = [float(x) for x in teacher_state["pass_rates"]]
                if "sampling_probs" in teacher_state:
                    metrics["final_sampling_probs"] = teacher_state["sampling_probs"]
            dead = s.get("Curriculum/terrain_levels/dead_frac", [])
            metrics["dead_frac_final"] = dead[-1][1] if dead else float("nan")
            eff = s.get("Curriculum/terrain_levels/effective_bins", [])
            metrics["effective_bins_final"] = eff[-1][1] if eff else float("nan")
            frontier = s.get("Curriculum/terrain_levels/frontier_bin", [])
            if frontier:
                metrics["frontier_start"] = frontier[0][1]
                metrics["frontier_final"] = frontier[-1][1]
            # Internal sampler-conformance trajectory. This is not sufficient
            # to establish frontier alignment because both p_hat and sampling
            # probabilities come from the teacher itself; fixed-grid eval below
            # supplies the external calibration check.
            zm = s.get("Curriculum/terrain_levels/zpd_mass", [])
            zb = s.get("Curriculum/terrain_levels/zpd_bins", [])
            n_bins_series = s.get("Curriculum/terrain_levels/n_bins", [])
            n_bins = (
                float(n_bins_series[-1][1])
                if n_bins_series
                else float(len(metrics.get("final_pass_rates", [])))
            )
            if zm and zb:
                zpd_bins_by_step = dict(zb)
                ratios = [
                    (step, mass / max(zpd_bins_by_step[step] / n_bins, 1e-9))
                    for step, mass in zm
                    if step in zpd_bins_by_step and zpd_bins_by_step[step] > 0 and n_bins > 0
                ]
                metrics["internal_zpd_ratio_mean"] = (
                    sum(r for _, r in ratios) / len(ratios) if ratios else float("nan"))
                metrics["internal_zpd_ratio_final"] = ratios[-1][1] if ratios else float("nan")
        # fixed-grid eval results (written by eval_arms.py), if present
        eval_records = []
        for ev in sorted(glob.glob(os.path.join(run_dir, "eval_frontier_*.json"))):
            with open(ev) as f:
                e = json.load(f)
            it = int(re.search(r"eval_frontier_(\d+)", ev).group(1))
            metrics[f"eval@{it}_mean_pass"] = e["mean_pass"]
            metrics[f"eval@{it}_per_level"] = e["per_level_pass"]
            eval_records.append((it, e))
        eval_records.sort()
        if len(eval_records) >= 2:
            mid_it, mid_eval = eval_records[0]
            final_it, final_eval = eval_records[-1]
            n_levels = len(final_eval["per_level_pass"])
            easy_count = max(1, math.ceil(0.2 * n_levels))
            easy_mid = mean_present(mid_eval["per_level_pass"][:easy_count])
            easy_final = mean_present(final_eval["per_level_pass"][:easy_count])
            metrics["eval_easy_mid"] = easy_mid
            metrics["eval_easy_final"] = easy_final
            metrics["eval_easy_delta"] = easy_final - easy_mid
            metrics["eval_mid_iter"] = mid_it
            metrics["eval_final_iter"] = final_it
        if arm == "teacher" and eval_records:
            final_eval = eval_records[-1][1]
            true_p = final_eval["per_level_pass"]
            p_hat = metrics.get("final_pass_rates")
            probs = metrics.get("final_sampling_probs")
            if p_hat and len(p_hat) == len(true_p):
                paired = [
                    (float(estimate), float(truth))
                    for estimate, truth in zip(p_hat, true_p)
                    if truth is not None
                ]
                if paired:
                    metrics["posterior_mae_fixed_eval"] = sum(
                        abs(estimate - truth) for estimate, truth in paired
                    ) / len(paired)
            if probs and len(probs) == len(true_p):
                zpd = [
                    idx for idx, truth in enumerate(true_p)
                    if truth is not None and 0.2 < float(truth) < 0.8
                ]
                if zpd:
                    metrics["external_zpd_ratio"] = (
                        sum(float(probs[idx]) for idx in zpd) / (len(zpd) / len(probs))
                    )
        rows[arm].append((seed, metrics))

    if not rows:
        raise SystemExit(f"No parseable runs under {args.log_root}")

    def agg(arm, key):
        vals = [m[key] for _, m in rows[arm] if key in m]
        if not vals:
            return "  --  "
        mean = sum(vals) / len(vals)
        if len(vals) > 1:
            sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
            return f"{mean:6.3f}±{sd:.3f}"
        return f"{mean:6.3f}"

    print(f"{'arm':>10} | {'n':>2} | {'bin AUC':>12} | {'bin final':>12} | {'track AUC':>12} | {'track final':>12}")
    print("-" * 80)
    for arm in ARMS:
        if arm not in rows:
            continue
        print(f"{arm:>10} | {len(rows[arm]):>2} | {agg(arm, 'bin_auc'):>12} | "
              f"{agg(arm, 'bin_final'):>12} | {agg(arm, 'track_auc'):>12} | {agg(arm, 'track_final'):>12}")

    if "teacher" in rows:
        print("\nP-A mechanism gate (teacher arm):")
        for seed, m in rows["teacher"]:
            internal = m.get("internal_zpd_ratio_mean", float("nan"))
            external = m.get("external_zpd_ratio", float("nan"))
            calibrated = math.isfinite(external)
            verdict = "PASS" if calibrated and internal > 1.0 and external > 1.0 else (
                "UNVERIFIED" if not calibrated else "FAIL"
            )
            print(f"  seed {seed}: internal_ratio mean={internal:.2f} "
                  f"final={m.get('internal_zpd_ratio_final', float('nan')):.2f}; "
                  f"fixed-eval ratio={external:.2f} -> {verdict}")
            print(f"           dead_frac={m.get('dead_frac_final')}, "
                  f"effective_bins={m.get('effective_bins_final')}, "
                  f"posterior_MAE={m.get('posterior_mae_fixed_eval')}, "
                  f"frontier={m.get('frontier_start')}->{m.get('frontier_final')}")

    print("\nP-C fixed-grid easy-level retention (lowest 20%, mid -> final):")
    for arm in ARMS:
        if arm not in rows:
            continue
        for seed, m in rows[arm]:
            if "eval_easy_delta" not in m:
                print(f"  {arm:>10} seed {seed}: UNVERIFIED (run eval_arms.py on mid,final)")
                continue
            delta = m["eval_easy_delta"]
            if args.retention_tau is None:
                verdict = "NO_TAU"
            else:
                verdict = "PASS" if delta >= -args.retention_tau else "FAIL"
            print(f"  {arm:>10} seed {seed}: {m['eval_easy_mid']:.3f} -> "
                  f"{m['eval_easy_final']:.3f} (delta={delta:+.3f}) {verdict}")

    print("\nTraining-distribution tracking reward (biased diagnostic only, mid -> final):")
    for arm in ARMS:
        if arm in rows:
            for seed, m in rows[arm]:
                print(f"  {arm:>10} seed {seed}: {m['track_mid']:.4f} -> {m['track_final']:.4f}")

    # fixed-grid eval comparison (the honest P-B), if eval_arms.py has run
    eval_keys = sorted({k for arm in rows for _, m in rows[arm] for k in m if k.endswith("_mean_pass")})
    if eval_keys:
        print("\nFixed-grid eval (same per-level grid for every arm — the honest P-B):")
        for arm in ARMS:
            if arm not in rows:
                continue
            for seed, m in rows[arm]:
                for k in eval_keys:
                    if k in m:
                        lvl = m[k.replace("_mean_pass", "_per_level")]
                        print(f"  {arm:>10} seed {seed} {k}: {m[k]:.3f}  per-level={lvl}")


if __name__ == "__main__":
    main()
