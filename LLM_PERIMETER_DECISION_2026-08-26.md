# P1 — LLM perimeter decision

**Status:** OPTION (b), DE-SCOPE, bound 2026-08-26 under the pre-committed
default. The rule below was recorded 2026-08-19, before any new LLM run was
launched or inspected.

## Decision record — 2026-08-26

No qualifying frozen-checkpoint smoke was recorded by the deadline showing
that the coarse-state treatment moves the delivery diagnostics. The default is
therefore option (b): no 40--80 A10G-hour confirmatory rerun is authorized, and
the LLM teacher-by-estimator interaction remains open and outside the ICLR
claim perimeter. The existing delivery finding stays in the appendix at its
earned tier. This scope decision creates no new scientific evidence and does
not reinterpret the prior gate failure.

## The rule

The charter's instruction is "pre-commit now: a second delivery-gate failure
auto-triggers (b)." That is binding from this file's commit:

> **If option (a) runs and its treatment-delivery gate fails, the LLM
> interaction leaves the claim perimeter immediately and permanently. No third
> attempt, no gate relaxation, no post-hoc reweighting of which gate component
> matters.**

This is the same discipline that makes the existing `.60148` failure an asset.
A gate that can be re-argued after the fact was never a gate.

## The gate, carried forward unchanged

The prior registered delivery gate (P-S1) was conjunctive on the dead-prompt
fraction `train_all_datasets_binning/fraction_of_prompts_in_[0.0, 0.0]`:

| component | threshold | prior run |
|---|---|---|
| minimum over steps | `< 0.50` | 0.413 — passed |
| run mean over steps | `< 0.60` | **0.60148 — failed by .00148** |

Option (a) inherits both components at the same thresholds. The only change is
the treatment: **tier/bucket-level posteriors instead of per-prompt**, which is
the structural fix the paper's own diagnosis names — a per-prompt Beta posterior
needs multiple visits, and a realistic RLVR pool gives roughly one epoch, so
the teacher knows but has no budget to act. Coarsening the posterior is the one
change that addresses that diagnosis rather than working around the gate.

Arms: GRPO ± teacher. Budget ≈40–80 A10G-hours.

## Both branches, drafted before data

**(a) passes the gate.** The LLM rung re-enters the perimeter, and its endpoint
is reported with whatever verdict its own registered primary returns —
including a negative one. Tier assignment follows that verdict; passing a
delivery gate certifies only that the treatment was delivered, never that it
worked.

**(a) fails the gate, or (b) is chosen now.** §6.7 compresses to the delivery
finding — that per-prompt posteriors are not deliverable at this pool size and
budget — and the claim perimeter ends at the neural maze. The LLM interaction
is named as open in the contribution list and confined to the appendix, which
is already where it sits.

## Why (b) is not a retreat

As of this week the perimeter is self-contained without the LLM rung: a proved
group-law identity with its granularity corollary, a registered positive
reproduced on three platforms using one seed cohort, three registered
boundaries, and a preregistered MAZE-SCORE intervention confirming that the
count-law correction is causally relevant on that substrate. A paper that
claims less and demonstrates all of it
outscores one gesturing at LLM scale through gated-out runs.

## Recommendation to the PI, due 2026-08-26

Take **(b)** unless 40–80 A10G-hours are free *and* the coarse-state treatment
can be smoked to confirm it moves the dead-prompt fraction before the
confirmatory run starts. Option (a) spends real compute on a gate that has
already failed once by .00148; the diagnosis says the fix is coarsening, but
the fix is untested, and an untested fix in front of a binding conjunctive gate
is a coin flip whose downside is a second public failure. A smoke that shows
the dead-prompt fraction moving is cheap and converts (a) from a gamble into a
decision.
