# Do not restart the AMaze driver against a live campaign directory

**Applies to:** `ued_benchmark/scripts/run_gate_confirmatory.sh` while the
2026-08-19 campaign is running. Written 2026-08-19, campaign in progress.

## The hazard

`launch()` treats the presence of `checkpoint.pkl` as "this run already
finished":

```bash
if [[ -f "$OUT/$xpid/checkpoint.pkl" ]]; then
  echo "SKIP   $xpid (checkpoint present)" >> "$OUT/driver.log"; return
fi
```

That was safe when `checkpoint_interval` was 30,000 ticks, because the file
appeared only late. With the 2026-08-19 fix it is written every 100 ticks, so a
checkpoint exists **about a minute into every run**. Restarting the driver
against a directory with runs in flight (or killed mid-run) would therefore skip
them and declare the campaign complete with partially trained cells — the same
class of silent truncation the 2026-08-19 amendment was written about, arriving
by a different route.

## Why it is not being patched right now

The driver is executing this file. bash reads scripts incrementally, so editing
it in place can make a running shell resume at a shifted byte offset. The
correct time to apply the patch is after the campaign is terminal.

## A second thing the same change broke

Progress monitoring. Counting `arm-*/checkpoint.pkl` used to approximate
"runs finished"; it now counts every run that has been alive for a minute. A
monitor built on it reported "10/20 runs finished" when 8 were complete and 2
were in flight. Completion counts must come from the driver's `^OK` lines, not
from file presence — the same distinction the patch below makes.

## Backstops that are already active

1. The monitor reports `AMAZE DRIVER GONE ... campaign incomplete` rather than
   silently finishing.
2. `analyze_gate_confirmatory.py` requires `ckpt_budget.json` and refuses any
   cell whose evaluated checkpoint holds fewer than 29,900 updates, so a
   partially trained cell cannot reach a verdict.

## The patch to apply once the campaign is terminal

Replace the presence check with an explicit completion marker:

```bash
  if [[ -f "$OUT/$xpid/DONE" ]]; then
    echo "SKIP   $xpid (completed)" >> "$OUT/driver.log"; return
  fi
  ...
  if bash "$ROOT/ued_benchmark/scripts/run_arm.sh" ... ; then
    touch "$OUT/$xpid/DONE"
    echo "OK     $xpid" >> "$OUT/driver.log"
```

If the driver has to be restarted **before** that patch lands, delete the
directories of any run that is not terminal first, or pass a fresh `$OUT`.
