"""E-LLM-3 verdict analyzer (PREREG_E_LLM3.md, commit 728d7aa).

Extracts per-tier val trajectories (mean@16 / pass@16) from the ray
session logs of each cell, persists them as JSON artifacts (the /tmp
logs are volatile — extract early, extract often), and applies the
pre-registered decision rules against the measured noise floor.

Key format (validated on E-LLM-2 Countdown logs):
  step:N - ... mean_accuracies/jugs_tierX/reward/mean@16:V
           ... pass@16_accuracies/jugs_tierX/reward/best@16/mean:V
Hindsight telemetry: hindsight/relabeled_rollouts, hindsight/
gated_saturated, hindsight/relabel_yield.

Usage:
  python3 analyze_e_llm3.py            # extract + verdicts on what exists
  python3 analyze_e_llm3.py --extract  # extraction only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CELLS_DIR = HERE / "cells"
TIERS = [f"jugs_tier{i}" for i in range(5)]
ARMS = {"B1": "hsfalse", "B2": "hstrue", "B3": "hstrue_gated"}
SEEDS = (1, 2, 3)


def parse_session(session_dir: str) -> dict | None:
    """Parse one ray session's worker logs into a cell record."""
    exp = None
    seed = None
    val_rows, hs_rows = [], []
    for f in glob.glob(os.path.join(session_dir, "logs", "worker*.out")):
        for line in open(f, errors="ignore"):
            if exp is None and "'experiment_name':" in line:
                m = re.search(r"'experiment_name': '([^']+)'", line)
                if m:
                    exp = m.group(1)
            if seed is None and "'default_local_dir':" in line:
                m = re.search(r"'default_local_dir':\s*'[^']*_s(\d+)'", line)
                if m:
                    seed = int(m.group(1))
            if not line.startswith("step:"):
                continue
            d = {}
            for tok in line.strip().split(" - "):
                k, _, v = tok.partition(":")
                d[k] = v
            step = int(d.get("step", -1))
            row = {"step": step}
            found_val = False
            for t in TIERS:
                mk = f"mean_accuracies/{t}/reward/mean@16"
                pk = f"pass@16_accuracies/{t}/reward/best@16/mean"
                if mk in d:
                    row[f"{t}/mean16"] = float(d[mk])
                    found_val = True
                if pk in d:
                    row[f"{t}/pass16"] = float(d[pk])
            if found_val:
                val_rows.append(row)
            elif "hindsight/relabeled_rollouts" in d:
                hs_rows.append({
                    "step": step,
                    "relabeled": float(d["hindsight/relabeled_rollouts"]),
                    "dead_groups": float(d.get("hindsight/dead_groups", 0)),
                    "gated": float(d.get("hindsight/gated_saturated", 0)),
                    "yield": float(d.get("hindsight/relabel_yield", 0)),
                })
    if exp is None or not val_rows:
        return None
    return {"experiment": exp, "val": sorted(val_rows, key=lambda r: r["step"]),
            "hindsight": sorted(hs_rows, key=lambda r: r["step"]),
            "session": os.path.basename(session_dir)}


def extract_all():
    CELLS_DIR.mkdir(exist_ok=True)
    for sess in sorted(glob.glob("/tmp/ray/session_2026-*")):
        rec = parse_session(sess)
        if rec is None or "jugs" not in rec["experiment"]:
            continue
        # seed comes from the ckpt dir naming (…_s{seed}); experiment_name
        # lacks it, so key on experiment+session and dedupe on content
        out = CELLS_DIR / f"{rec['experiment']}_{rec['session']}.json"
        json.dump(rec, open(out, "w"), indent=1)
        last = rec["val"][-1]
        print(f"extracted {out.name}: {len(rec['val'])} val rows, "
              f"last step {last['step']}")


def cell_files(arm: str) -> list[Path]:
    # experiment_name = jugsv1_maxrl_curfalse_{hs...}_s{seed}_16r
    pat = f"jugsv1_maxrl_curfalse_{ARMS[arm]}_s*_16r_*"
    exact = []
    for p in sorted(CELLS_DIR.glob(pat)):
        # B2's pattern is a prefix of B3's; disambiguate
        if arm == "B2" and "_gated_" in p.name:
            continue
        exact.append(p)
    # keep one file per seed (latest session wins — resumes/repeats)
    by_seed = {}
    for p in exact:
        m = re.search(r"_s(\d+)_16r", p.name)
        if m:
            by_seed[int(m.group(1))] = p
    return [by_seed[s] for s in sorted(by_seed)]


def series(rec: dict, tier: str, meter: str) -> list[tuple[int, float]]:
    key = f"{tier}/{meter}"
    return [(r["step"], r[key]) for r in rec["val"] if key in r]


def verdicts():
    floor_path = HERE / "jugs_noise_floor.json"
    floors = None
    if floor_path.exists():
        floors = json.load(open(floor_path))["floors"]

    arms_data = {}
    for arm in ARMS:
        recs = [json.load(open(p)) for p in cell_files(arm)]
        arms_data[arm] = recs
        print(f"{arm}: {len(recs)} cells")

    out = {"n_cells": {a: len(r) for a, r in arms_data.items()},
           "floors_loaded": floors is not None}

    def finals(arm, tier, meter):
        vals = []
        for rec in arms_data[arm]:
            s = series(rec, tier, meter)
            if s:
                vals.append(s[-1][1])
        return np.array(vals)

    # P-J1: B2 vs B1 on t0 — mean up, pass down, >=2/3 seeds
    if arms_data["B1"] and arms_data["B2"]:
        n = min(len(arms_data["B1"]), len(arms_data["B2"]))
        m1, m2 = finals("B1", "jugs_tier0", "mean16")[:n], \
            finals("B2", "jugs_tier0", "mean16")[:n]
        p1, p2 = finals("B1", "jugs_tier0", "pass16")[:n], \
            finals("B2", "jugs_tier0", "pass16")[:n]
        mean_up = int((m2 > m1).sum())
        pass_down = int((p2 < p1).sum())
        out["P-J1"] = {
            "n_pairs": n,
            "t0_mean16": {"B1": m1.tolist(), "B2": m2.tolist(),
                          "delta": float((m2 - m1).mean()),
                          "up_count": mean_up},
            "t0_pass16": {"B1": p1.tolist(), "B2": p2.tolist(),
                          "delta": float((p2 - p1).mean()),
                          "down_count": pass_down},
            "verdict": ("CONFIRMED" if n >= 3 and mean_up >= 2
                        and pass_down >= 2 else
                        ("PENDING" if n < 3 else "NOT CONFIRMED"))}
        if floors:
            f = floors.get("jugs_tier0", {})
            out["P-J1"]["noise_note"] = {
                "mean16_2sd": 2 * f.get("mean16_sd", float("nan")),
                "pass16_2sd": 2 * f.get("pass16_sd", float("nan"))}

    # P-J2: B3 recovers >=50% of B2's t0 pass16 loss, keeps >=40% mean gain
    if all(arms_data[a] for a in ("B1", "B2", "B3")):
        n = min(len(arms_data[a]) for a in ("B1", "B2", "B3"))
        b1p = finals("B1", "jugs_tier0", "pass16")[:n].mean()
        b2p = finals("B2", "jugs_tier0", "pass16")[:n].mean()
        b3p = finals("B3", "jugs_tier0", "pass16")[:n].mean()
        b1m = finals("B1", "jugs_tier0", "mean16")[:n].mean()
        b2m = finals("B2", "jugs_tier0", "mean16")[:n].mean()
        b3m = finals("B3", "jugs_tier0", "mean16")[:n].mean()
        loss = b1p - b2p
        gain = b2m - b1m
        out["P-J2"] = {
            "n": n, "t0_pass16": [b1p, b2p, b3p],
            "t0_mean16": [b1m, b2m, b3m],
            "coverage_recovered_frac":
                float((b3p - b2p) / loss) if loss > 0 else None,
            "mean_kept_frac": float((b3m - b1m) / gain) if gain > 0 else None,
        }
        if loss > 0 and gain > 0:
            ok = (b3p - b2p) / loss >= 0.5 and (b3m - b1m) / gain >= 0.4
            out["P-J2"]["verdict"] = "CONFIRMED" if ok else "NOT CONFIRMED"
        else:
            out["P-J2"]["verdict"] = ("VACUOUS (P-J1 pattern absent: "
                                      f"loss={loss:.3f} gain={gain:.3f})")

    # P-J3: t2 ignition — B2/B3 pass16 > 0 in >=2/3 while B1 == 0
    for arm in ("B1", "B2", "B3"):
        if arms_data[arm]:
            t2 = finals(arm, "jugs_tier2", "pass16")
            out.setdefault("P-J3", {})[arm] = {
                "t2_final_pass16": t2.tolist(),
                "ignited_count": int((t2 > 0).sum())}
    if "P-J3" in out and all(a in out["P-J3"] for a in ("B1", "B2")):
        b1_zero = out["P-J3"]["B1"]["ignited_count"] == 0
        b2_ign = out["P-J3"]["B2"]["ignited_count"]
        n2 = len(out["P-J3"]["B2"]["t2_final_pass16"])
        out["P-J3"]["verdict"] = (
            "PENDING" if n2 < 3 else
            ("CONFIRMED" if b1_zero and b2_ign >= 2 else "NOT CONFIRMED"))

    # P-J4: relabel yield (report only)
    yields = []
    for arm in ("B2", "B3"):
        for rec in arms_data[arm]:
            ys = [r["yield"] for r in rec["hindsight"] if r["yield"] > 0]
            if ys:
                yields.append({"cell": rec["experiment"],
                               "session": rec["session"],
                               "mean_yield": float(np.mean(ys))})
    out["P-J4_report"] = yields

    path = HERE / "e_llm3_verdicts.json"
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items()
                      if k.startswith("P-")}, indent=1)[:2000])
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    args = ap.parse_args()
    extract_all()
    if not args.extract:
        verdicts()
