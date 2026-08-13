#!/usr/bin/env python3
"""GATE-DR analysis — implements hopper/GATE_DR_PREREG.md exactly.

Inputs: logs/gatedr_<task>_<jobid>.out (successful runs only; failed first-attempt
logs contain no step-60 line and are skipped by construction) and
accounting/gatedr_<arm>_s<seed>_dose.jsonl.

Terminology: t*_proxy16 is the VERL bootstrap best@16 coverage proxy
(pass@16_accuracies/.../best@16/mean), never standard unbiased pass@16.
"""
import json, os, re, sys
from glob import glob

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["b1h", "g0", "g085", "g070"]
SEEDS = [1, 2, 3]
TASK_TO_RUN = {i: (ARMS[i // 3], i % 3 + 1) for i in range(12)}

def load_runs():
    """Amendment 2026-08-13b: endpoints come from the sealed checkpoint
    evaluation (eval/gate_dr_eval_master.json), not from training logs —
    in-run validation never executed (test_freq=-1 suppresses val_on_last_step).
    t*_proxy16 maps to the legacy bootstrap proxy; standard pass@16 is carried
    alongside descriptively."""
    master = json.load(open(os.path.join(HERE, "eval", "gate_dr_eval_master.json")))
    runs = {}
    for run_id, rec in master["results"].items():
        arm, s = run_id.replace("gatedr_", "").rsplit("_s", 1)
        t = rec["tiers"]
        runs[(arm, int(s))] = {
            "t1_mean16": t["t1"]["mean16"], "t1_proxy16": t["t1"]["proxy16_bootstrap"],
            "t1_pass16_standard": t["t1"]["pass16_standard"],
            "t2_mean16": t["t2"]["mean16"], "t2_proxy16": t["t2"]["proxy16_bootstrap"],
            "t2_pass16_standard": t["t2"]["pass16_standard"],
        }
    return runs

def reject_fraction(arm, seed):
    p = os.path.join(HERE, "accounting", f"gatedr_{arm}_s{seed}_dose.jsonl")
    if not os.path.exists(p):
        return None
    gated = relab = 0.0
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            gated += float(d.get("hindsight/gated_saturated", 0))
            relab += float(d.get("hindsight/relabeled_groups", 0))
    denom = gated + relab
    return {"gated": gated, "relabeled_groups": relab,
            "reject_frac": (gated / denom) if denom else None}

def main():
    runs = load_runs()
    missing = [(a, s) for a in ARMS for s in SEEDS if (a, s) not in runs]
    assert not missing, f"incomplete matrix: missing {missing}"

    doses = {f"{a}_s{s}": reject_fraction(a, s) for a in ARMS[1:] for s in SEEDS}

    res = {"_prereg": "hopper/GATE_DR_PREREG.md (frozen 2026-08-13, commit 16b95b7)",
           "_metric_provenance": "t*_proxy16 = VERL bootstrap best@16 coverage proxy, "
                                 "not standard unbiased pass@16",
           "runs": {f"{a}_s{s}": runs[(a, s)] for a in ARMS for s in SEEDS},
           "dose_manipulation_check": doses}

    def paired(arm, key):
        return [runs[(arm, s)][key] - runs[("b1h", s)][key] for s in SEEDS]

    d_mean_g0 = paired("g0", "t1_mean16")
    transfer_pairs_pos = sum(1 for d in d_mean_g0 if d > 0)
    res["transfer_gate"] = {
        "delta_mean_g0_per_seed": d_mean_g0,
        "positive_pairs": transfer_pairs_pos,
        "passes": transfer_pairs_pos >= 2,
    }

    if transfer_pairs_pos < 2:
        res["verdict"] = ("INCONCLUSIVE-BY-TRANSFER: ungated recycling did not beat "
                          "no-recycling on >=2/3 seed pairs in this environment; "
                          "no gate conclusions drawn (prereg rule 1).")
    else:
        g0_gain = sum(d_mean_g0) / 3
        settings = {}
        for arm in ("g085", "g070"):
            dm, dp = paired(arm, "t1_mean16"), paired(arm, "t1_proxy16")
            kept = [x / g0_gain for x in dm] if g0_gain else None
            mean_kept = (sum(dm) / 3) / g0_gain if g0_gain else None
            rule2a = mean_kept is not None and mean_kept >= 0.40 and all(x > 0 for x in dm)
            rule2b = (sum(dp) / 3) >= -0.005 and sum(1 for x in dp if x >= 0) >= 2
            settings[arm] = {
                "delta_mean_per_seed": dm, "delta_proxy_per_seed": dp,
                "mean_kept_fraction": mean_kept, "kept_per_seed": kept,
                "rule2a_mean": rule2a, "rule2b_coverage": rule2b,
                "reproduces_useful_point": rule2a and rule2b,
            }
        res["settings"] = settings
        winners = [a for a in settings if settings[a]["reproduces_useful_point"]]
        rejects = [doses[f"{a}_s{s}"]["reject_frac"] for a in ("g085", "g070")
                   for s in SEEDS if doses[f"{a}_s{s}"]["reject_frac"] is not None]
        if winners:
            res["verdict"] = (f"USEFUL OPERATING POINT REPRODUCED at {winners} "
                              "(prereg rule 2).")
        elif rejects and min(rejects) > 0.85:
            res["verdict"] = ("CORRECTED GATE EFFECTIVELY BINARY: no setting meets "
                              "rule 2 and all reject fractions exceed .85 "
                              "(prereg rule 3).")
        else:
            res["verdict"] = ("GRADED DOSE WITHOUT USEFUL OPERATING POINT: rejection "
                              "is graded but no setting meets rule 2 (prereg rule 4).")

    out = os.path.join(HERE, "gate_dr_analysis.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("transfer_gate", "verdict")}, indent=1))
    print("full analysis ->", out)

if __name__ == "__main__":
    main()
