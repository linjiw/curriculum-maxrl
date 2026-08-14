# Results

**Snapshot:** 2026-08-14 03:58 America/New_York
**Scientific evidence launched:** no

## Iteration decisions

1. **Discard as ICRA progress:** `autoresearch/iterate-260813-2348` targets
   ICLR/MAZE-SCORE Hopper work explicitly excluded by the governing goal.
   Its queued job is preserved but not expanded.
2. **Keep:** the candidate 300-course BARN manifest reproduces at SHA-256
   `1015a6a48ef44add7224200da2ace1cd6c8d7780275b30d7266a44dc88e9ec61`;
   all 900 adapter-consumed assets reverify.
3. **Keep:** the deterministic split reproduces at SHA-256
   `c0ed1d7024ebc240d96a023efb6a124e879fdb06d0342a5e5de7b6d6ed07d7d7`.
   All prior engineering used `barn-299`, which is in TRAIN.
4. **Keep with declared limitation:** fixed-seed BARN success repeated, but
   exact trajectories/step counts did not. The protocol now treats simulator
   transitions as primary, records the nondeterminism before evidence, and
   retains the training seed as the independent unit.
5. **Keep:** Hopper fingerprint job 9366688 completed CPU-only with container
   SHA-256
   `cd6620e33c0822f7d6a03c6de6ea9dd4304f0927e8d7997c003560f5b4781be0`.
6. **Keep:** production runner, machine protocol, balanced orders, blind
   merger/selector, strict analyzer, immutable staging, dataset-preparation
   job, engineering smoke, evidence sbatch, source-bound submission, exact
   ledger finalization, and sealed all-cell postprocessing passed their local
   fail-closed gates. These remain engineering artifacts only until the freeze
   and evidence gates pass.
7. **Discard as engineering passes, retain diagnostics:** dataset preparation
   jobs 9366805 and 9366814 failed before publication because the compute node
   lacked `/usr/bin/time` and Hopper scratch rejected directory
   `renameat2(RENAME_NOREPLACE)`, respectively. Bash timing and an atomic
   canonical-directory claim with `COMPLETE` hard-linked last fixed the two
   environmental assumptions.
8. **Keep:** dataset preparation job 9366817 completed CPU-only in 42 seconds
   and published the checksum-closed canonical package. Its exact receipt
   SHA-256 is
   `216408ddfb6ef95c6d7cc912608aac0428240d09a562f20b03069408b1a9d76f`.
9. **Discard as a smoke pass, retain diagnostics:** job 9366819 built the exact
   Gazebo stepper/plugin and passed the no-asset guard, then failed because the
   launcher replaced ROS's `PYTHONPATH` and made `rclpy` unavailable. The fix
   prepends repository paths while retaining the container environment.
10. **Keep as runtime validation:** corrected train-only smoke job 9366821
    completed CPU-only in 8 minutes 28 seconds, without held-out reads or a
    retained metric artifact. Its receipt lacked simulator-step and phase
    timing counters, so it was not used as the final feasibility receipt.
11. **Keep as the outcome-blind feasibility measurement:** repeated smoke job
    9366831 completed CPU-only in 7 minutes 15 seconds of scheduler elapsed
    time (422 seconds in its receipt) and reported only resource counters:
    50,570 training simulator steps, 16 training episodes, 330.497 training
    seconds, and 62.233 seconds for two train-course evaluations. It read no
    held-out course, retained no internal metric artifact, and emitted no paper
    endpoint. Receipt SHA-256:
    `d9d251c819bbf602dae6c829e3c6755b514639f2fa1c3c9f83cd5b13d21c8738`.
12. **Discard the incomplete campaign, retain the audit trail and fix:** the
    first primary transaction created held job 9366866, but its remote ledger
    install/acknowledgement did not complete. The job stayed
    `PENDING|JobHeldUser`, never received a compute allocation, and was
    canceled. No endpoint was opened. The source-bound submitter now reuses an
    exact staged-ledger upload on resume, with a network-free interruption
    regression. Evidence will restart under a new campaign ID and source SHA.

## Outcome-blind design decisions

- Primary budget: 1,000,000 Gazebo physics steps per arm; wall time descriptive.
- Primary cell: four arms, paired seeds 1--5, N=8.
- Mandatory N sweep: fresh two-arm cells at N=2,4,16; reuse the identical
  primary ours/learnability rows at N=8.
- Gate: ours directionally at least uniform and learnability; staged cannot
  pass or veto it.
- Engineering smoke: exactly one N=8 trainer update on train course barn-299;
  no held-out course and no retained metric artifact.
- Resource-only projection from job 9366831: 153.012 training simulator steps
  per second; 1.815 hours per one-million-transition training phase; 31.116
  seconds per evaluation episode; 3.112 hours for six checkpoints over 60
  courses; 4.927 hours per arm. The four-arm primary cell is 19.708 hours
  nominal and 23.650 hours with 20% padding; a two-arm ablation cell is 9.854
  hours nominal.
- **Kept in the frozen package:** request 36 hours per evidence array task. A 24-hour
  request leaves only 0.350 hours beyond the padded primary estimate, whereas
  36 hours leaves 12.350 hours. This is scheduler headroom only and does not
  alter the frozen scientific budgets.

No scientific comparison, gate result, or paper claim is available yet. The
outcome-blind package was frozen in milestone commit
`23dacb88cf7b1f46dddf9d2453dbd7e0bcbbbf33` and now carries the dated
pre-execution operational amendment above. No BARN seed task has run.
