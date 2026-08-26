# AMaze gate confirmatory closure record — 2026-08-26

**Status at authoring:** outcome-blind; analyzer unspent.

The 2026-08-19 rerun is terminal at 20/20 training cells and 20/20 shipped
evaluations. This record adds no scientific rule. It binds the operational
closure required by the 2026-08-19 preregistration amendment before any
evaluation value is opened.

The closure entrypoint is:

```bash
ued_benchmark/scripts/close_gate_confirmatory.sh --preflight-only
ued_benchmark/scripts/close_gate_confirmatory.sh
```

It pins the frozen preregistration, checkpoint-budget verifier, and analyzer;
requires the exact 2×10 jobs matrix, 20 distinct training receipts, 20 distinct
evaluation receipts, zero failure receipts, and all expected nonempty files;
then reads only each checkpoint's stored `n_updates`. Every checkpoint must be
in `[29900,30000]`. Only after that conjunction passes does it write the
canonical `ckpt_budget.json`, backfill explicit `DONE` markers, and invoke the
frozen analyzer once. It refuses to run if either canonical single-use output
already exists.

The three frozen verdict branches remain exactly those in
`AMAZE_GATE_PREREG.md`: `gate_beats_upstream`,
`gate_does_not_beat_upstream`, or `inconclusive_at_n10`. This record neither
changes those branches nor creates evidence.
