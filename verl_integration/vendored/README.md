# Vendored execution-fork files

Verbatim copies of the files the paper's LLM-scale experiments executed,
vendored so the paper repository is self-contained (2026-08-04 draft
review, "artifact is not yet self-contained").

| file | source | execution commit | last change |
|---|---|---|---|
| `hindsight.py` | `maxrl/verl/utils/hindsight.py` (sibling execution fork) | `0ad11b1` (fork HEAD at vendor time) | `df8b2cf` 2026-08-03 (gate granularity fix: task-keyed posterior) |

md5 at vendor time: `c16aad224433d85bf8f83eb15dcdc5dc`.

Provenance notes:

- The three-seed gated Countdown runs (B3, §6.9) executed an EARLIER
  revision with the decay bug disclosed in §6.9 (one-sided pseudo-miss
  decay; effective threshold ~0.64). This vendored copy is the CORRECTED
  implementation (both-sides decay, F5) used by the single-seed
  full-strength rerun and all subsequent runs.
- The gate statistic in this file is a decayed achieved-destination
  frequency over the relabel stream (see module docstring), NOT a
  fresh-rollout pass-rate estimate of the destination task — the paper
  names it accordingly.
