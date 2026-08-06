#!/usr/bin/env python3
"""Extract reviewer-arm results from ray worker logs + compute verdicts.

Usage: python3 extract_arm_results.py [tag ...]
  default tags: all ARM A/B cells. Writes arm{A,B}_*.json next to this
  file and recomputes reviewer_arms_verdicts.json whenever all inputs
  for a verdict are present.

P-R1 / P-R2 preregistered in smollm/run_reviewer_arms.sh (79473b2).
References: b_scoreboard_3seed.json (B1/B2, same pool/protocol/seeds).
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

TAGS = {
    "cdb3fix_s1": "armA_b3fix_s1.json",
    "cdb3fix_s2": "armA_b3fix_s2.json",
    "cdb3fix_s3": "armA_b3fix_s3.json",
    "cdreplay_s1": "armB_replay_s1.json",
    "cdreplay_s2": "armB_replay_s2.json",
    "cdreplay_s3": "armB_replay_s3.json",
}


def extract(tag):
    """Find the ray session whose logs carry `tag`; pull per-tier val + hindsight telemetry."""
    for sess in sorted(glob.glob("/tmp/ray/session_2026-*"), reverse=True):
        hit = None
        for f in glob.glob(f"{sess}/logs/worker*.out"):
            try:
                if tag in open(f, errors="ignore").read(30000):
                    hit = sess
                    break
            except OSError:
                continue
        if not hit:
            continue
        val, hs = [], []
        for f in glob.glob(f"{hit}/logs/worker*.out"):
            for line in open(f, errors="ignore"):
                if not line.startswith("step:"):
                    continue
                d = {}
                for tok in line.strip().split(" - "):
                    k, _, v = tok.partition(":")
                    d[k] = v
                step = int(d.get("step", -1))
                row = {"step": step}
                found = False
                for tier in range(3):
                    mk = f"mean_accuracies/countdown_tier{tier}/reward/mean@16"
                    pk = f"pass@16_accuracies/countdown_tier{tier}/reward/best@16/mean"
                    if mk in d:
                        row[f"t{tier}_mean16"] = float(d[mk])
                        found = True
                    if pk in d:
                        row[f"t{tier}_pass16"] = float(d[pk])
                if found:
                    val.append(row)
                elif "hindsight/relabeled_rollouts" in d:
                    hs.append({"step": step,
                               "relabeled": float(d["hindsight/relabeled_rollouts"]),
                               "gated": float(d.get("hindsight/gated_saturated", 0))})
        val.sort(key=lambda r: r["step"])
        hs.sort(key=lambda r: r["step"])
        return {"cell": tag, "session": hit.split("/")[-1], "val": val, "hindsight": hs}
    return None


def complete(path):
    if not os.path.exists(path):
        return False
    d = json.load(open(path))
    return bool(d["val"]) and d["val"][-1]["step"] == 60


def final_t1(path):
    d = json.load(open(path))
    last = d["val"][-1]
    assert last["step"] == 60, f"{path}: last step {last['step']} != 60 (run incomplete?)"
    return last["t1_mean16"], last["t1_pass16"]


def verdicts():
    sb = json.load(open(os.path.join(HERE, "b_scoreboard_3seed.json")))
    b1_mean, _, b1_pass, _ = sb["B1_t1"]
    b2_mean, _, b2_pass, _ = sb["B2_t1"]
    out = {}

    a_files = [os.path.join(HERE, f"armA_b3fix_s{s}.json") for s in (1, 2, 3)]
    if all(complete(f) for f in a_files):
        pts = [final_t1(f) for f in a_files]
        m = sum(p[0] for p in pts) / 3
        c = sum(p[1] for p in pts) / 3
        kept = (m - b1_mean) / (b2_mean - b1_mean)
        out["P_R1"] = {
            "arm": "ARM A designed-gate B3, 3 seeds",
            "t1_mean16": [p[0] for p in pts], "t1_pass16": [p[1] for p in pts],
            "mean_kept_fraction": round(kept, 3),
            "window": {"mean_kept": [0.0, 0.60], "coverage": [0.541, 0.571]},
            "verdict": "REFUTED" if not (0 <= kept <= 0.60 and 0.541 <= c <= 0.571) else "CONFIRMED",
        }

    b_files = [os.path.join(HERE, f"armB_replay_s{s}.json") for s in (1, 2, 3)]
    have = [f for f in b_files if complete(f)]
    if have:
        pts = [final_t1(f) for f in have]
        m = sum(p[0] for p in pts) / len(pts)
        c = sum(p[1] for p in pts) / len(pts)
        kept = (m - b1_mean) / (b2_mean - b1_mean)
        rec = {
            "arm": f"ARM B replay ppo_epochs=2, {len(have)}/3 seeds",
            "t1_mean16": [p[0] for p in pts], "t1_pass16": [p[1] for p in pts],
            "mean_captured_fraction_of_B2_gain": round(kept, 3),
            "coverage_vs_B1": round(c - b1_pass, 3),
        }
        if len(have) == 3:
            # P-R2: replay captures >= half of B2's mean gain with no pass@16 loss;
            # "captures ~all" branch => 6.8 reduces recycling's case to the direction term
            per_seed_ok = [(p[0] - b1_mean) / (b2_mean - b1_mean) >= 0.5 and p[1] >= b1_pass
                           for p in pts]
            rec["verdict"] = ("CONFIRMED-STRONG (captures >all of B2's gain, no coverage loss)"
                              if kept >= 1.0 and c >= b1_pass and all(per_seed_ok)
                              else "CONFIRMED" if kept >= 0.5 and c >= b1_pass
                              else "REFUTED")
            rec["per_seed_meets_window"] = per_seed_ok
        else:
            rec["verdict"] = "INTERIM"
        out["P_R2"] = rec

    path = os.path.join(HERE, "reviewer_arms_verdicts.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    tags = sys.argv[1:] or list(TAGS)
    for tag in tags:
        dest = os.path.join(HERE, TAGS[tag])
        if os.path.exists(dest):
            d = json.load(open(dest))
            if d["val"] and d["val"][-1]["step"] == 60:
                print(f"{tag}: already extracted (step 60)")
                continue
        rec = extract(tag)
        if rec is None:
            print(f"{tag}: session not found")
            continue
        json.dump(rec, open(dest, "w"), indent=1)
        last = rec["val"][-1] if rec["val"] else {}
        print(f"{tag}: extracted, last step {last.get('step')}")
    verdicts()
