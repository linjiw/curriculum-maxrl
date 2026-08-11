# Iteration goal

Build and verify the first outcome-blind ICRA 2027 navigation-campaign slice:
choose the domain, freeze the four-arm protocol and analysis boundary, add the
missing teacher/allocation/evaluation plumbing, and produce one explicitly
non-evidentiary end-to-end smoke artifact without touching the shared RTX 5090.

## Keep criteria

1. Existing `frontier_rl` tests remain green.
2. New campaign tests prove fixed held-out evaluation is repeatable and does
   not mutate training RNG/counters.
3. The four primary arms complete one CPU smoke run and emit dual-budget,
   per-difficulty, retention, dead-group, and calibration fields.
4. The analysis refuses to make an August 24 decision from fewer than five
   full-domain seeds.
5. Every new result is labeled engineering smoke, not robotics evidence.

