#!/usr/bin/env python
"""Record the training state actually stored in each checkpoint.pkl.

minimax checkpoints on `tick % checkpoint_interval == 0` and has no post-loop
save, so `checkpoint.pkl` is not necessarily the final model.  The 2026-08-17
execution of AMAZE_GATE_PREREG.md evaluated six of ten seeds at ~50% of the
budget because of that.  This script reads each checkpoint's own stored
`n_updates` and writes them to a sidecar the analyzer requires, so the failure
cannot recur silently.

Run under the minimax environment (needs jax to unpickle the runner state).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def stored_updates(obj, depth=0):
    if depth > 6:
        return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "n_updates":
                return int(np.asarray(v).ravel()[0])
            r = stored_updates(v, depth + 1)
            if r is not None:
                return r
    elif hasattr(obj, "_fields"):
        for k in obj._fields:
            v = getattr(obj, k)
            if k == "n_updates":
                return int(np.asarray(v).ravel()[0])
            r = stored_updates(v, depth + 1)
            if r is not None:
                return r
    elif isinstance(obj, (list, tuple)):
        for v in obj[:6]:
            r = stored_updates(v, depth + 1)
            if r is not None:
                return r
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results")
    ap.add_argument("--output", default=None,
                    help="default: <results>/ckpt_budget.json")
    args = ap.parse_args(argv)
    root = Path(args.results)
    out = Path(args.output) if args.output else root / "ckpt_budget.json"

    rec = {}
    for d in sorted(root.glob("arm-*-u30000")):
        ck = d / "checkpoint.pkl"
        if not ck.is_file():
            print(f"MISSING {d.name}")
            continue
        with ck.open("rb") as f:
            rec[d.name] = stored_updates(pickle.load(f))
        print(f"{d.name:<28} n_updates={rec[d.name]}")
    out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out}  ({len(rec)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
