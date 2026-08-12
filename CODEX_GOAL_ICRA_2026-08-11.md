# Goal statement for codex — ICRA 2027 track, phase 2 (evidence phase)

**Date issued:** 2026-08-11 (evening)
**Prior phase:** `ICRA2027_PROGRESS_REPORT_2026-08-11.md` (scaffolding complete, committed at `08c3726`)
**Governing plan:** `claude-fable-plan.md` Track B; protocol frozen in draft at `icra2027/prereg_icra.md`
**Deadline chain:** first full-domain seed by **Aug 17** → go/no-go gate **Aug 24** → ICRA submission **Sept 14** (deadline Sept 15, 11:59 PST; 8 pages TOTAL including references, double-anonymous)

## What is already done (do not redo)

- Four sampler arms (`ours_uN`, `uniform`, `learnability p(1−p)`, `staged`) implemented in `frontier_rl/teacher.py` with shared posterior/Thompson/floor machinery; exact water-filling on `p(1−p)^N`; all tests pass (5/5 campaign, 21/21 frontier_rl with `PYTHONPATH=curriculum_maxrl`).
- Outcome-blind campaign runner (`icra2027/navigation_campaign.py`), stratified split freezer (`icra2027/freeze_pool_split.py`), dual-budget paired analyzer (`icra2027/analyze_campaign.py`, frozen SHA-256 `4017958…7efdf3`).
- CPU grid smoke ran end-to-end; stamped `engineering_smoke_not_paper_evidence`. It proves plumbing, nothing else.
- Preregistration draft with all decision rules; two placeholders remain: **BARN asset manifest SHA-256** and **container/image digest**.

## The single goal of this phase

**Produce real navigation-domain evidence: complete one end-to-end full-domain seed of all four arms by Aug 17, then run the 4-arm × ≥5-paired-seed matrix plus the mandatory N ∈ {2,4,8,16} ablation, and apply the Aug 24 gate exactly as preregistered.** Everything below serves that goal; nothing else is in scope.

## Unblocking facts discovered 2026-08-11 (corrects the progress report)

The progress report said "no BARN assets or Isaac Lab installation were found." Both halves are now softened:

1. **BARN assets are publicly downloadable.** The BARN dataset (300 procedurally generated Gazebo `.world` courses with published difficulty metrics) is distributed via UT Austin ("Benchmark for Autonomous Robot Navigation", Xiao et al.) and the ICRA BARN Challenge repos (e.g. `github.com/Daffan/nav-competition-icra2022`, which vendors worlds + Jackal launch files for ROS + Gazebo). ROS 2 Humble, Nav2, and `/opt/ros` are already installed on this machine; a `colcon_ws` exists at `/home/robotixx/colcon_ws`. First action: fetch the world set, build/verify a Gazebo (Classic or new Gazebo) + Jackal or generic diff-drive lidar robot pipeline, and emit the course JSONL manifest that `freeze_pool_split.py` consumes (fields: immutable ID, scalar difficulty, asset path, checksum).
2. **Isaac Lab is installed locally** at `/home/robotixx/.holosoma_deps/IsaacLab`, `/home/robotixx/Documents/IsaacLab`, `/home/robotixx/reward_research/IsaacLab` (plus `cour-isaaclab`). The fallback does not require a new install. **However — see the GPU constraint below: Isaac Lab needs the RTX 5090, which is embargoed.** The Isaac Lab fallback is therefore only usable on a different machine, or after the E2c campaign fully completes. Plan accordingly: **the primary path (Gazebo/BARN) must be CPU-runnable.**

## Hard constraints (violating any of these invalidates the campaign)

1. **GPU embargo:** this machine's only GPU is the shared RTX 5090 reserved for E2c under a frozen 4,096 MiB occupancy gate (currently blocked by an unrelated `cosmos-framework` process at ~7.3 GiB). Do not place ANY ICRA job on it. The BARN/Gazebo campaign runs CPU-only (the policy is a small lidar-to-velocity net; this is fine). If CPU throughput makes 5 seeds × 4 arms infeasible by Aug 24, report that honestly rather than touching the GPU — do not silently shrink the protocol.
2. **Freeze before evidence:** fill both SHA-256 placeholders in `icra2027/prereg_icra.md`, generate and inspect `barn_split.json`, and commit prereg + split + analyzer TOGETHER **before the first full seed finishes**. After that commit, the prereg is immutable except via a dated amendment section.
3. **Budget convention:** if exclusive like-for-like hardware cannot be guaranteed for wall-clock timing (likely, since this is a shared box), invoke the prereg's own escape hatch — transition-matched AUC becomes primary, wall-clock descriptive — and record that switch in the prereg BEFORE unblinding, not after.
4. **No peeking:** do not inspect partial full-domain endpoints to tune score shape, split, promotion threshold, or seed list. The analyzer refuses <5 paired seeds for the gate; do not work around it.
5. **Independent unit = training seed.** Courses and repeated samplers within a seed are repeated measurements. Report paired deltas, 95% paired bootstrap CI, per-seed deltas, sign-flip p.
6. **Terminology:** arm 3 is "learnability," never "ALP-GMM." Never call the historical Countdown proxy "pass@16" (it is the "VERL bootstrap best@16 coverage proxy"). ARM B (ICLR side) is "higher-dose replay control," never "dose-matched."
7. **No scope creep:** no PLR arm, no HER arm, no physical-robot runs until the 4×5 matrix is complete or safely running. GCL/GACL comparison is a stretch goal AFTER the matrix, not before.
8. **Aug 24 gate is binding:** continue toward ICRA only if `ours_uN` is directionally ≥ both `uniform` and `learnability` on the full domain at ≥5 paired seeds. Otherwise preserve everything for RA-L and stop deadline-driven expansion. Do not force the result either way — a clean negative is publishable at RA-L.

## Milestones and acceptance criteria

| Date | Milestone | Acceptance test |
|---|---|---|
| Aug 12–13 | BARN backend adapter | A `NavigationSpace`-compatible adapter runs one course end-to-end in Gazebo headless on CPU; returns course ID, difficulty, binary success (Nav2 or timeout/collision verifier), simulator steps, trajectory for the policy update. Deterministic under fixed seed. |
| Aug 13 | Prereg freeze | Manifest SHA-256 + container digest filled; `barn_split.json` generated (seed 20270811, 10 strata, 80/20) and eyeballed; prereg + split + analyzer committed in one commit. |
| Aug 17 | First full seed, all four arms | Runner completes on the real domain; analysis JSON produced; artifacts NOT stamped as smoke; per-arm wall-clock and step budgets recorded. If this date slips, escalate immediately — the Isaac-Lab-on-another-machine fallback needs lead time. |
| Aug 17–23 | 5-seed matrix + N-ablation | ≥5 paired seeds × 4 arms; N ∈ {2,4,8,16} ablation at matched rollout budget (confirmatory contrast: deployed `u_N` vs `p(1−p)` at each N). Parallelize across CPU cores, one seed per process group; keep the eval stream isolated per prereg. |
| Aug 24 | Gate decision | Run `analyze_campaign.py` on the full matrix; write `icra2027/GATE_DECISION_2026-08-24.md` quoting the frozen rule and the numbers; decision is ICRA-continue or RA-L-pivot. |
| Aug 25–Sept 10 | Paper (only if gate passes) | 8-page IEEE double-column draft INCLUDING references, double-anonymous, AI-content disclosure in acknowledgments. Related-work skeleton per `LITERATURE_POSITIONING.md` §5 (GCL ICRA'25 and GACL IROS'25 are the BARN-native baselines to cite and differentiate: our score is derived from the deployed estimator's algebra and is N-aware; theirs are not). |

## Reporting requirements

After each milestone, append a dated section to `ICRA2027_PROGRESS_REPORT_2026-08-11.md` (or a successor dated report) with: what ran, artifact paths + SHA-256, tests passed, and any deviation from this document with justification. Commit at each milestone boundary; never commit `.codex/`. Do not switch branches in this checkout; never merge the ICLR release branch.

## Explicitly out of scope for codex this phase

- Anything touching the ICLR compact draft, `paper/body_iclr.tex`, or the release branch (reconciliation is a separate task with its own rules in `autoresearch/iterate-260810-2240/BRANCH_RECONCILIATION.md`).
- Launching E2c training (its driver self-authorizes via `E2C_LAUNCH_READINESS.json`; keep polling read-only if idle, never force).
- Website updates, arXiv posting, or any external publication.

---

# Addendum 2026-08-12 — milestone review and next-step directives

## Review verdict on work completed since issue

The **Aug 12–13 "BARN backend adapter" milestone is substantially achieved a day early** and is accepted: a 300-course manifest (`icra2027/barn_manifest.jsonl`) with per-asset SHA-256, published optimal-traversal-time difficulty, grid and path checksums; a CPU-only seeded `gzserver` runtime with ROS 2 pause/unpause, bumper collision detection, and sim-clock accounting (`frontier_rl/adapters/barn_gazebo.py`); a diff-drive robot SDF; and the correct rejection of Isaac Lab under the GPU embargo. This is exactly the unblock the original report said was impossible — good.

## Directives before the prereg freeze (do these NOW, in order)

The prereg is still an outcome-blind draft, so these are ordinary edits, not amendments. After step 4 it becomes immutable.

1. **Pin the teacher unit.** The adapter makes the teacher's task unit a frozen difficulty *stratum* (one course sampled uniformly within it per group), not an individual course. This is an acceptable design (denser posteriors at a 5-seed budget) but it changes what all four arms sample over, so it must be written into `prereg_icra.md` explicitly: state that all arms share the same frozen stratum set, give the stratum count, state that the staged arm's "published difficulty order" means stratum order, and record the rejected alternative (course-level teacher) with the posterior-density rationale. Do not leave this implicit in code.
2. **Pin the difficulty semantics.** Difficulty = published optimal traversal time in seconds (longer = harder). Say so in the prereg, and confirm the staged arm unlocks from short-time strata upward and the "easy decile" secondary uses the same metadata.
3. **Invoke the budget escape hatch now.** This is a shared CPU box; exclusive like-for-like hardware cannot be guaranteed. The prereg's own clause requires the switch be made before unblinding: transition-matched AUC becomes primary, wall-clock descriptive. Write it in.
4. **Freeze and commit the package together:** (a) manifest fingerprint = SHA-256 of `barn_manifest.jsonl` plus the acquisition receipt (source URL and archive SHA-256) for the 300-world dataset root, recorded in the repo even if the worlds themselves are not committed; (b) environment fingerprint — there is no container, so record and hash the exact Gazebo/ROS/dpkg versions and a `pip freeze`, and state in the prereg that this fingerprint stands in for the container digest; (c) `barn_split.json` from `freeze_pool_split.py --seed 20270811 --n-strata 10`, inspected; (d) prereg + split + analyzer + adapter, one commit. Also commit the currently untracked backend files — uncommitted evidence infrastructure is a provenance hole.

## Directives for the first full seed (target unchanged: Aug 17)

5. **Determinism check first:** one test that two runs of the same course, seed, and policy state produce identical outcome bits and step counts end-to-end through gzserver+ROS. If message-timing nondeterminism is found, document it in the prereg and rely on the seed-paired design and fixed evaluation streams — do not silently accept it.
6. **Throughput budget before committing to the matrix:** measure single-episode wall time headless, then compute whether 4 arms × 5 seeds fits by Aug 23 with parallel isolated `gzserver` instances (separate `ROS_DOMAIN_ID`/partition/ports, one seed-arm per process group). Physics step size stays frozen and identical across arms — never tune it per arm. If the arithmetic does not close, report it immediately rather than shrinking the protocol silently.
7. Run the full four-arm seed with the real adapter; artifacts are evidence-grade only if the freeze (step 4) landed first.

## Standing constraints (unchanged, restated because they bind harder now)

- The RTX 5090 remains embargoed for E2c — note an automatic E2c launch watcher now polls it from the ICLR side, so the GPU may spin up a Countdown training job at any moment without warning. Plan CPU capacity assuming that.
- Aug 24 gate exactly as frozen; ≥5 paired seeds; no peeking; RA-L is the honest exit.
- Milestone-boundary commits with dated progress sections; never commit `.codex/`.
