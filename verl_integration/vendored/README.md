# Vendored execution-fork files

Copies of the files the paper's LLM-scale experiments executed, vendored so
the paper repository is self-contained (2026-08-04 draft review, "artifact is
not yet self-contained"). Local post-review changes are documented below.

| file | source | execution commit | last change |
|---|---|---|---|
| `hindsight.py` | `maxrl/verl/utils/hindsight.py` (sibling execution fork) | `2700198` (fork HEAD at vendor time) | `2700198` 2026-08-05 (one_target_per_group ablation mode, draft-review P0-2; runs to date all used per-row mode) |

md5 at vendor time: `33b75833cb7fe6895c437d349055efba`. The revision
all published runs executed (per-row mode, before the ablation flag)
was fork commit `0ad11b1`, md5 `c16aad224433d85bf8f83eb15dcdc5dc` —
the flag defaults to False, so behavior is identical.

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
- On 2026-08-09, the rebuilt clean-SFT path made response reasoning
  target-agnostic. The relabeler now leaves responses unchanged instead of
  applying unsafe blanket number substitution, and emits per-step accepted
  group/token accounting for the preregistered matched-replay study. The
  hashes above remain provenance for the historical copies, not the modified
  working file.
- The same local research iteration added fixed-slot, dose-matched live-group
  replay. Replay consumes B2's immutable accepted-slot schedule and matches
  cumulative auxiliary and total optimizer response-token dose. The original
  current-batch-only arm stopped at its frozen treatment-delivery gate; the
  separately preregistered E2b follow-up may also draw exact informative group
  snapshots from a 64-group, eight-step recent buffer. These are prospective
  controls for this iteration, not behavior of the historical execution fork.
- E2c (preregistered 2026-08-10) adds a third, separate source mode: an
  immutable, checksummed reservoir of informative train-only groups generated
  from the frozen clean-SFT checkpoint with learning rate zero. When configured,
  reservoir sources are exclusive—current-policy and recent-buffer groups are
  never mixed into the arm. Static provenance/token-support checks and a
  three-seed delivery audit must pass before held-out endpoints are generated.
