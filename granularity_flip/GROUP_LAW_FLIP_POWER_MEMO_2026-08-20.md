# P0 group-law flip — outcome-blind power and block-count memo

**Inputs opened:** the already published MAZE-SCORE design SD range
`0.0077–0.0135`, its measured full-arm engineering cost of 1,337 seconds, and
the P0 draft's `+.005` SESOI. No P0 run exists; no P0 endpoint was opened.

## Decision

Use **48 paired blocks, seeds 3001–3048**. The study has one primary contrast,
`grouplaw - plugin`, so there is no Holm family. The final analyzer retains the
exact two-sided sign-flip test and a 20,000-resample paired percentile-bootstrap
interval.

The sample-size proxy simulates the full conjunction except that, as in the
MAZE-SCORE power memo, a paired t-test and t interval stand in for the exact
sign-flip and bootstrap procedures under symmetric Normal paired differences.
`power_group_law_flip.py` uses 100,000 replications per cell, seed 20260820.

At the pessimistic paired SD `0.0135`, estimated support probability is:

| true effect | n=20 | n=30 | n=40 | n=48 |
|---:|---:|---:|---:|---:|
| +.0050 (SESOI) | .342 | .460 | .497 | .503 |
| **+.0075 (powered-for)** | .653 | .815 | .878 | **.901** |
| +.0100 | .881 | .971 | .990 | .995 |
| +.0125 | .975 | .998 | 1.000 | 1.000 |

At the optimistic SD `0.0077`, n=48 has `.988` power at `+.0075`. A true
effect exactly at `+.005` remains a coin flip because the decision rule
requires the observed mean itself to reach `+.005`; the SESOI is a reporting
threshold, not the powered-for effect.

## Why not keep twenty blocks

The draft's n=20 had no prospective power calculation. At the pessimistic SD
it gives only `.653` power at the effect the design means to detect. Forty-eight
blocks reach `.901` while keeping the exact test feasible with the already
verified meet-in-the-middle implementation. A full 48-block synthetic campaign
completed in 5.0 seconds with 572 MB peak RSS on the local pre-freeze test
host. No endpoint or development contrast from P0 influenced this change.

## Cost bound

The original full-arm smoke took 1,337 seconds and included SFT preparation.
Conservatively charging that full time to each of P0's two arms gives
`2 × 1,337 s = 0.7428` MIG-slice-hours per block and **35.7 MIG-slice-hours**
for 48 blocks. At the frozen `%5` throttle this is at most about **7.4 hours**
wall time, plus queueing and retrieval. This bound deliberately double-counts
the once-per-block SFT preparation.

Hopper's 40-submitted-job cap requires two non-overlapping 24-seed array
chunks. Their union remains the one frozen 48-block campaign; the second chunk
is submitted only when queue capacity permits.
