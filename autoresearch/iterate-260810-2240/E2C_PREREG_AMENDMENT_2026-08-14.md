# E2c preregistration amendment — 2026-08-14

**Type:** mechanical repair of the orchestrator, outcome-blind, pre-data.
**Amends:** `E2C_PREREG.md` (frozen 2026-08-10), section
"Frozen input fingerprints and executable lock".
**Scientific protocol changed:** none. No hypothesis, arm, seed, budget, gate,
dose, estimator, reservoir rule, matcher, generation setting, endpoint, or
decision branch is altered by this amendment.

## Why this amendment exists

The frozen orchestrator `verl_integration/run_e2c_rtx5090.sh` could never
execute a training stage. The defect was present at freeze time and had never
been exercised: between the freeze on 2026-08-10 and 2026-08-14 the shared
RTX 5090 never fell below the 4,096 MiB launch ceiling, so no launch path ever
ran. Only the readiness path, which does not reach the defect, had been used.

At `2026-08-14T22:35:57-04:00` the GPU dropped to 1,280 MiB, the standing
watcher observed `launch_authorized_now: true`, and the unchanged frozen driver
was invoked. It aborted immediately:

```
verl_integration/run_e2c_rtx5090.sh: line 156: PYTHON_BIN: readonly variable
driver exited rc=1
```

### Root cause

The driver runs under `set -euo pipefail` (line 4). It declares

```bash
readonly PYTHON_BIN=...      # line 19
readonly TRAIN_DATA=...      # line 23
readonly MODEL_PATH=...      # line 25
readonly STEPS=60            # line 27
```

and then re-states those same four names as **command-prefix assignments** in
each of the three launch blocks, in order to export them to the child
`countdown_rtx5090.sh`:

| Variable | Prefix-assignment sites |
|---|---|
| `PYTHON_BIN` | 157, 209, 288 |
| `MODEL_PATH` | 157, 209, 288 |
| `TRAIN_DATA` | 159, 212, 290 |
| `STEPS` | 162, 215, 293 |

Bash rejects a command-prefix assignment to a `readonly` name. Under `set -e`
the script aborts **before the trainer is executed**. Verified in isolation on
GNU bash 5.1.16:

```
$ bash -c 'set -euo pipefail
  readonly PYTHON_BIN=/usr/bin/python3
  PYTHON_BIN="$PYTHON_BIN" bash -c "echo EXECUTED" | tee probe.log'
bash: line 4: PYTHON_BIN: readonly variable
rc=1        # "EXECUTED" never printed; probe.log created and empty
```

A sweep of the driver found exactly these four names reassigned, at exactly
these twelve sites, all inside the three launch blocks. All twelve are
**value-identical self-assignments** (`VAR="$VAR"`), except line 215, which is
the literal `STEPS=60`, equal to the frozen value.

### Consequences established before repair

1. No E2c training stage has ever started. `heldout_artifact_count` is 0 and no
   E2c checkpoint, reservoir, or endpoint exists.
2. The four completed comparator arms (`b1`/`b2` × seeds 1, 2) predate this
   driver; it only *verifies* and skips them. Their provenance is untouched.
3. The driver file was byte-identical to its preregistered fingerprint
   immediately before repair, confirming it had not been silently edited:
   recorded `729447c426944f060b88cae272d537fe78a89e61bc0db3c1b6467daebc2cd4b9`,
   15,676 bytes — observed identical.

## The repair

A single `env` command token is inserted at the head of each of the three launch
blocks (lines 156, 208, 287). Nothing else in the driver changes.

```diff
-  RUNTIME_ROOT="$RUNTIME_ROOT" MAXRL_ROOT="$RUNTIME_ROOT/maxrl" \
+  env RUNTIME_ROOT="$RUNTIME_ROOT" MAXRL_ROOT="$RUNTIME_ROOT/maxrl" \
```

`env` receives the `VAR=VAL` pairs as its own argv and applies them to the
environment of the command it execs. The shell therefore performs **no variable
assignment at all**, so the `readonly` conflict cannot arise.

### Why this is the correct repair and not the alternatives

- **All seventeen `readonly` guards are preserved**, including on the four
  prereg-frozen constants `PYTHON_BIN`, `TRAIN_DATA`, `MODEL_PATH`, `STEPS`.
  Dropping `readonly` from lines 19/23/25/27 would also have worked, but would
  have removed the in-script immutability lock on exactly the four values the
  preregistration freezes. Rejected.
- **The self-assignments are retained.** Deleting them would be silently
  destructive: `verl_integration/countdown_rtx5090.sh:24` is
  `STEPS=${STEPS:-1}`, so a dropped `STEPS` would produce **1-step runs** in
  place of the frozen 60-step budget, and would do so without error. Rejected.
- The values reaching `countdown_rtx5090.sh` are byte-for-byte those the frozen
  driver declared.

Verified after the edit:

- `bash -n verl_integration/run_e2c_rtx5090.sh` — clean.
- Isolated semantic proof on bash 5.1.16: with `readonly PYTHON_BIN` and
  `readonly STEPS` in force, `env PYTHON_BIN="$PYTHON_BIN" STEPS="$STEPS" …`
  execs the child, the child observes the correct values, the parent returns
  rc=0, and a subsequent attempt to assign `PYTHON_BIN` in the parent is still
  rejected as readonly.
- Diff is exactly three lines, one per launch block.

### Scope relative to the executable lock

`run_e2c_rtx5090.sh` is **not** one of the 31 files in `E2C_CODE_MANIFEST.json`
(SHA-256 `0e46b89f…`), which is unchanged by this amendment. The manifest covers
the scientific surface, including the inner training script
`verl_integration/countdown_rtx5090.sh`, which is **not modified**. The
preregistration separately records the orchestrator's own fingerprint at
`E2C_PREREG.md:110-112`, which this amendment supersedes:

| | SHA-256 | Bytes |
|---|---|---|
| Preregistered 2026-08-10 | `729447c426944f060b88cae272d537fe78a89e61bc0db3c1b6467daebc2cd4b9` | 15,676 |
| Amended 2026-08-14 | `ac4148dbfa38a8cc3bc7778fd632e2dd761cb9e652f43f65944886c349693e69` | 15,688 |

## Second repair: removal of a zero-byte log

The aborted 22:35:57 attempt left an empty log at
`autoresearch/iterate-260810-2240/e2c_logs/e2_clean_b1_s3_260809.log`. It was
created by the `tee` on the right-hand side of the pipeline, which opened and
truncated the file before the left-hand side failed.

Its presence made the readiness audit hard-fail with
`e2_clean_b1_s3_260809: artifacts exist without .complete`, setting
`integrity_status: fail`, `next_stage: repair_integrity_failure`, and
`launch_authorized_now: false`. This is the audit behaving correctly: it cannot
distinguish an aborted launch from a partially-written one, and the driver's own
guard at line 146 refuses to append to an incomplete comparator.

The file was removed. It is provably empty and therefore carries no outcome:

- size 0 bytes;
- SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`,
  the SHA-256 of the empty string;
- no corresponding checkpoint directory was created
  (`/data/robotixx/curriculum-maxrl-runtime/checkpoints/e2_clean_b1_s3_260809`
  did not exist).

Nothing else required removal. `/data/robotixx/curriculum-maxrl-runtime/e2c_gpu.lock`
is the `flock` target only and is not inspected by any guard.

## Outcome-blindness statement

This amendment was written and applied before any E2c training step, any
reservoir, any replay run, and any held-out evaluation. No E2c outcome, metric,
endpoint, or partial result existed at the time of the repair, and none was
inspected. `heldout_artifacts_inspected` was `false` and
`heldout_artifact_count` was 0 throughout.

The direction test, its analysis rule, and the ordered seed tuple `(1, 2, 3)`
are unchanged.

## Operational note: the standing watcher

The watcher at `/home/robotixx/.claude/jobs/ca9ae5b6/tmp/e2c_watch.sh`
(PID 2842583, started 2026-08-12) polled every 600 s and, from 22:46 onward,
logged `blocked` — reporting GPU memory in its message even though the actual
cause was the readiness failure above. Its attempt counter increments only when
readiness reports `launch_authorized_now: true`, so it consumed only one of its
three attempts (the 22:35:57 crash).

It was stopped at `2026-08-14T23:51:03-04:00` before this repair was applied, to
remove the risk of it launching a partially-edited driver. The launch is
performed deliberately instead.

## Post-repair state

Readiness after repair reports `integrity_status: pass`, `issues: []`,
`launch_authorized_now: true`, `next_stage: train_e2_clean_b1_s3_260809`,
under the unchanged 4,096 MiB ceiling.

## Effect on the Aug 28 training stop

The Aug 28 decision rule in `CODEX_GOAL_ICLR_2026-08-12.md` is unchanged. This
amendment does not extend it. It records that the runway lost between 2026-08-10
and 2026-08-14 was attributable to GPU occupancy, and that the delay from
22:35 on 2026-08-14 onward was attributable to this defect rather than to
occupancy.
