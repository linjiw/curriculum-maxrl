# Integrated Status Review and Improvement Plan

> ## Execution status — updated 2026-08-15 00:45 EDT
>
> The review below was produced read-only. Most of §5 has since been executed.
> **E2c has run and is now CLOSED** — see the second box.
>
> | §5 item | State | Evidence |
> |---|---|---|
> | 2. E2c amendment written before the edit | **DONE** | `autoresearch/iterate-260810-2240/E2C_PREREG_AMENDMENT_2026-08-14.md` |
> | 3. Minimal driver fix (`env` token, three sites) | **DONE** | driver `729447c4…` → `ac4148db…`; all 17 `readonly` guards intact |
> | 4. Delete the 0-byte log, then launch | **DONE — and E2c has since CLOSED** | `E2C_CLOSURE_2026-08-15.md` |
> | 5. Selective clean commit | **DONE (by the PI, 00:12–00:13)** | 14 commits; tree now clean, which unblocks `stage_maze_score.sh evidence` |
> | 6. BARN scheduler poll | **DONE — campaign healthy** | all 20 tasks RUNNING on hop064–073, 17:47:33 elapsed of a 36 h limit, 0 failures. Metadata only; no log or endpoint opened |
> | 8. MAZE-SCORE power memo | **DONE — recommendation changed, see below** | `hopper/MAZE_SCORE_POWER_MEMO_2026-08-15.md` |
> | §4 Cut 1. UED lane closure note | **DONE** | `ued_benchmark/LANE_CLOSURE_2026-08-15.md` |
> | P0-A canonicalization + parity test | **DONE** | `curriculum_maxrl/test_score_contract.py`, 13 tests |
> | 9. Reap four orphaned `gzserver` PIDs | **NOT DONE — one command for a human** | `kill 2874122 2900798 2903603 2905340 && rm -rf /tmp/icra_barn_adapter_test*` (11.1 CPU-hours burned; verified unrelated to campaign 003) |
> | 1. Repo private + push | **NOT DONE — needs a human** | now **24** unpushed commits; account-level action, see §7.1 |
> | 7. `docs/index.html` retracted claim | **NOT DONE** | resolves itself if the repo goes private |
>
> ### E2c: closed, INCONCLUSIVE by preregistered rule
>
> The repaired driver ran the frozen order through `b1_s3`, `b2_s3`, the frozen
> reservoir, and the static preflight — all passing. Replay seed 1 then stopped
> at step 39 on **runtime validity gate 7**: cumulative optimizer-token mismatch
> 5.7856% over the 5% tolerance. Per the frozen rule this makes the three-seed
> direction test **inconclusive**, and gates/reservoir/matcher/budget are not
> changed under the E2c label. Seeds 2–3 were not attempted. No endpoint exists.
>
> The drift is structural, not transient: monotone and accelerating from ~step 25
> (0.45% → 1.44% → 2.35% → 3.40% → 4.67% → 5.79%), with `fallback_slots = 0` at
> all 38 steps. Cause: **source diversity collapses**. Late steps draw as many as
> seven replacement groups from a single reservoir entry while `dead_slots` sits
> at 5–8, so one fixed response length is substituted for varying targets.
>
> Recorded finding: *dose-matched replay from a frozen immutable reservoir
> degrades as the policy sharpens, because the pool of distinct, length-compatible
> informative sources shrinks exactly when demand for replacements grows.*
> This strengthens §2.5 — E2c was never the critical path — and it satisfies the
> Aug 28 stop 13 days early. **The RTX 5090 is free.**
>
> ### §2.2 is ANSWERED: peak-location specificity is not supported
>
> The u₆₄ arm §2.2 called "the only new experiment whose result changes the
> paper's central claim" ran, plus a stronger dose–response that §2.2 did not
> propose. Both preregistered, both analyzed exactly once.
>
> | contrast | Hopper (Xeon 6240R) | local (Ultra 7 265K) |
> |---|---|---|
> | u16 − u64 | −.01127, p .2349, 10/20 | −.01279, p .1307, 7/20 |
> | u16 − p(1−p) | **+.03217, p .000105, 17/20** | **+.03074, p .000507, 16/20** |
>
> Dose–response with the deployed estimator pinned at N=16 throughout:
> uniform .658 | u2 .644 | u4 .668 | u8 .665 | **u16 .675** | u32 .679 |
> **u64 .688 (argmax)** | u128 .681 — **Spearman(exponent, mean) = +0.929**.
>
> **The curve rises past the deployed N.** A score peaked at p\*=.064 does at
> least as well as one peaked at the "matched" .169, on both platforms. So
> D4 resolves to its second branch: the replicated finding is *"harder-peaked
> beats p(1−p)"*, and deployed-N peak-location specificity **must not be
> claimed**. Risk-register item 2 is closed by evidence rather than hedging.
>
> **What got stronger:** the V2 primary now replicates on two further platforms
> with a *tighter* p than V2's own — the first reproduction outside the machine
> that produced it. That is precisely the §2.1 reframe's load-bearing claim, now
> evidence-backed. Caveats that bound it are in
> `acrobot_nsweep/FINDINGS_2026-08-15.md`: the two new campaigns share seeds so
> their agreement is portability not extra n; and magnitude is platform-sensitive
> (+.048 arm64 vs +.031/.032 x86 on identical seeds).
>
> The algebra is untouched — A_N(p)=2(pass@N−pass@1), T=N−1, and the closed-form
> Beta posterior all stand. What fails is a claim about where the score should
> peak, not about the algebra producing it.
>
> ### MAZE-SCORE: §2.4's recommendation is superseded
>
> §2.4 recommends 72 blocks + Monte-Carlo. Simulating the *actual* conjunction at
> `analyze_maze_score.py:529` changes that in three ways:
>
> 1. **Sample size cannot rescue an effect at the SESOI.** The rule requires the
>    *observed* mean to clear +.005, so a true effect of exactly +.005 is a coin
>    flip at every n (45.7% at n=30, 50.2% at n=72). §2.4's "50% → 88%" silently
>    assumed a larger true effect. The prereg must state a powered-for effect
>    (≥ +.0075).
> 2. **The curve knees at 40 and flattens past 48**, so 48 → 72 costs +26.7
>    MIG-slice-h for +4.1 points.
> 3. **`MAX_EXACT_SIGN_FLIP_N = 40` is a memory guard, not a wall.** The test is
>    meet-in-the-middle: 268 MB at n=48, 1.1 GB at 52, 17 GB at 60, 1.1 TB at 72.
>
> **Revised recommendation: 48 blocks (seeds 20–67), cap 40 → 48, keeping the
> exact test.** That is 90.0% of the 94.1% power 72 would give, at two-thirds the
> compute, and it avoids swapping a preregistered *exact* randomization test for a
> sampled approximation — a change of instrument, not of sample size.
> `test_sample_size_contract.py` now cross-checks the four encodings of N that
> nothing previously compared.
>
> **The root cause found tonight is not in the review below and supersedes its
> E2c framing.** The frozen driver had never been able to launch *any* training
> stage. Lines 19/23/25/27 declare `PYTHON_BIN`, `TRAIN_DATA`, `MODEL_PATH`,
> `STEPS` `readonly`; the three launch blocks then re-state those same names as
> command-prefix assignments, which bash rejects under `set -euo pipefail`,
> aborting rc=1 before the trainer execs. Both the defect and the recorded
> orchestrator hash date from the single commit `161f335` that froze the driver,
> so the GPU wait from Aug 10–14 was moot: the run would have failed identically
> on any earlier night. This makes the §2.5 conclusion stronger, not weaker —
> E2c was never the critical path, and it has now cost four days of runway to a
> three-token shell defect.
>
> The repair uses the `env` command token rather than removing `readonly`,
> preserving the immutability lock on all four preregistration-frozen constants,
> and retains the self-assignments because `countdown_rtx5090.sh:24` is
> `STEPS=${STEPS:-1}` — dropping them would have produced silent 1-step runs.
>
> The standing watcher (PID 2842583) was stopped at 23:51 before the edit so it
> could not launch a partially-written driver; the launch was then done
> deliberately. **Risk-register item 8 (partial-log hard block) is now live and
> unmitigated for the remaining unattended stages** — if the driver dies
> mid-stage it will leave a partial tee log that hard-blocks both the driver and
> `--readiness-only`, exactly as tonight's 0-byte log did. Recovery is to remove
> that stage's log after confirming no `.complete` marker.

**Prepared:** 2026-08-14 23:00 EDT · **Horizon:** ICLR 2027 full paper, Sept 25 AOE (42 days) · **Scope:** all six lanes
**Author role:** research lead, read-only pass. No file was modified, no job submitted, no endpoint opened.

**Verification note.** Facts marked **[V]** were re-checked directly in this session (git, filesystem, process table, receipt JSON, source line numbers). Facts marked **[S]** come from the lane surveys and were not independently re-verified. Facts marked **[I]** are my inference from **[V]**/**[S]** inputs. Anything I could not establish is written `UNKNOWN` with the exact method to obtain it.

---

## 1. Where the project actually stands

### The one-sentence version

The project has **one proved identity, one frozen positive score result at 640 parameters, one clean negative, and roughly 90,000 lines of preregistration/packaging scaffolding that has produced zero additional paper numbers** — while the two cheapest, highest-value actions available (a rewrite that costs no compute, and a ~1-hour repo/provenance fix) have not been taken.

### Lane by lane

**PAPER — the only lane with a deliverable, and it is nearly finished as an artifact and mis-aimed as a claim.**
Real: the compact 15-page PDF is deterministically reproducible to SHA-256 `36a6c1fb…` under pinned Tectonic 0.16.9; the independent build audit returns GO with P0=P1=P2=0; claim tracing is 62/62 with 0 UNTRACED; all three adversarial P0s are repaired **[S]**. Scaffolding-with-a-hole: the frozen artifact is **entirely uncommitted** — `git status --porcelain -uall` returns **280 paths [V]**, and two hash-bound build inputs, `paper/FIGURE_TOOLCHAIN.json` and `paper/requirements-figures.lock`, are **untracked [V]**. A clone of HEAD cannot reproduce the frozen build. The paper claims `reproduce.sh` builds `main_iclr2027.pdf`; `ls paper/*.pdf` shows only `main_iclr.pdf` and `main.pdf` **[V]**. The pinned build engine does exist and is correct — `/data/robotixx/snmr-tools/bin/tectonic` hashes to `397efac4…` **[V]** — but it lives in a shared `/data` tree named for an unrelated toolset, outside the repo, with no bootstrap.
Honest read: **the paper is a strong, unusually candid diagnostic paper wearing a method paper's clothes.** Its stated ceiling — no positive score evidence above 640 parameters — is real, disclosed in three independent internal documents, and is exactly what a hostile AC will lead with.

**E2c / Countdown — unblocked by hardware, blocked by three shell tokens, and worth less than the plan assumes.**
The RTX 5090 is free: readiness receipt regenerated **22:56:08 EDT** reports `gpu.memory_used_mib = 1280`, `compute_processes = []`, `gate_pass = true` **[V]**. But the same receipt reports `integrity_status = "fail"`, `issues = ["e2_clean_b1_s3_260809: artifacts exist without .complete"]`, `next_stage = "repair_integrity_failure"`, `launch_authorized_now = false` **[V]** — the orchestrator's "authorized: true" reading was taken *before* the 22:35:57 crash. Two mechanical defects, both confirmed: (a) `readonly PYTHON_BIN/TRAIN_DATA/MODEL_PATH/STEPS` at lines 19/23/25/27 are reassigned as command-prefix assignments at lines 157/209/288 **[V]**, so bash returns rc=1 before exec at all three GPU launch blocks; (b) the resulting 0-byte `autoresearch/iterate-260810-2240/e2c_logs/e2_clean_b1_s3_260809.log` (mtime Aug 14 22:35) **[V]** makes the readiness audit hard-fail, blocking even `--readiness-only`.
Real evidence produced by this lane to date: **zero** (`heldout_artifact_count = 0` **[V]**). And its marginal value is smaller than the goal document assumes: GATE-DR already ships **12 tracked per-arm/seed eval JSONs with per-task 16-bit outcomes** and already reports the paper's measurement ambiguity in *standard* observed-set pass@16 (`.656 → .414`) rather than the VERL bootstrap proxy **[S]**. E2c adds only the direction-isolation contrast, which the current abstract does not cite.

**MAZE-SCORE — the best experiment in the project, one commit from being permanently spoiled.**
Real: engineering ladder rungs 0–5 complete, three Hopper jobs COMPLETED exit 0, measured full-arm cost 1,337 s at peak 39,672 MiB on 3g.40gb **[S]**. The arms are the cleanest contrast in the whole record: `un` = u₃₂, `learn` = p(1−p), `unif` = uniform, with identical posterior, Thompson mechanism, 0.15 floor and level set, so the *only* algorithmic difference between `un` and `learn` is the utility function **[S]**.
Not real yet: the prereg is `**Status:** DRAFT v2 — NOT AUTHORIZED FOR EVIDENCE SUBMISSION` **[V]**; the worktree is dirty so `stage_maze_score.sh evidence` refuses **[V]**; and there is a four-way sample-size mismatch — `EXPECTED_SEEDS = tuple(range(20,50))` at `analyze_maze_score.py:34`, `--array=20-49%5` at `maze_score_array.sbatch:12`, the regex `^(2[0-9]|3[0-9]|4[0-9])$` at `:88`, and a DRAFT prereg calling 30 "a candidate" — plus `MAX_EXACT_SIGN_FLIP_N = 40` at `analyze_maze_score.py:51`, which raises `AnalysisError` above 40 pairs at `:464-467` **[all V]**.

**UED / AMaze — the clearest resource sink in the project.**
~44,400 LOC, ~37 h of continuous agent iteration, 11 telemetry build/verify rounds, 2 bundle freezes, 3 independent audits **[S]** — against **3.25 minutes of real GPU compute, exactly one PPO update, and zero numbers in any manuscript [S]**. Its only preregistered comparison self-declares "engineering/development selection only; never paper evidence" and "A positive development result is a selection signal, not evidence that Frontier beats robust PLR, ACCEL, or a published number" **[S]**. Its frozen protocol needs ~492M transitions/run against a documented 1-day gpuq cap and an explicit no-resume rule **[S]**. Five dev seeds cap the exact two-sided sign-flip p at .0625, so even a perfect result is inferentially empty **[S]**. Every audit round has ended by discovering more items and none has ended in a remote action.

**ICRA / BARN — a genuinely well-run sealed campaign aimed at the wrong venue.**
Real: campaign `barn-icra2027-20260814-003`, 20 tasks, all RUNNING at 2026-08-14T10:37:04Z, byte-locked protocol, two dated outcome-blind amendments, nominal terminal 2026-08-15T06:19Z **[S]**. Two structural limits that no amount of execution fixes: the teacher pools a Beta posterior over a 24-course stratum while each group runs on one sampled course, so it computes u_N(E[p_c]) and — since u_N is strictly concave — **systematically overstates true activity by Jensen [S/I]**; and the backend is documented non-deterministic at fixed seed (`fixed_seed_exact_match: false`), so arms are seed-paired but **not trajectory-paired [S]**. The gate is directional-only on 5 seeds with no effect-size floor, so the modal outcome is an ambiguous near-tie that authorizes a deadline it cannot support.

**CODE CONTRACTS — the P0 is real, mostly repaired, and empirically smaller than it looks.**
Canonical `coefficient_activity(p,N)` and `legacy_frontier_activity(p,N)` now exist in `curriculum_maxrl/estimators.py`, but exactly **one** production module imports them (`maze_gpu/train.py:39`); `teachers.py:148`, `verl_curriculum.py:95`, `verl_integration/curriculum.py`, `frontier_rl/*` all still reimplement inline **[S]**. Committed HEAD of `maze_gpu/train.py:77` still documents the *shifted* u_{N+1} form as "u_N(p)" **[S]**. **Crucially undervalued:** `curriculum_maxrl/un_form_verdicts.json` already measures the difference — `frontier_legacy` AUC .21348±.01042 vs `frontier_un` .21384±.00591 **[S]** — i.e. the two forms are empirically indistinguishable on the maze. **This is a naming and provenance defect, not a numerical one, and it should be defended that way.**

### Status table

| Lane | Paper-usable evidence today | Scaffolding built | Blocking item | Can it produce a new paper number by Sept 25? |
|---|---|---|---|---|
| Paper | Compact PDF, GO audit, 62/62 trace | reproduce.sh, manifest, claim trace | 280 uncommitted paths **[V]** | N/A — it *is* the deliverable |
| E2c | None (`heldout_artifact_count=0` **[V]**) | 15,676-B hash-pinned driver, 31-file manifest, 7 delivery gates | 0-byte log + readonly-prefix rc=1 **[V]** | Yes, in 3.7 h — but low value |
| MAZE-SCORE | None | 20 tests, analyzer, fail-closed sbatch, cost receipt | DRAFT prereg + dirty tree + 4-way N mismatch **[V]** | **Yes — highest value** |
| UED/AMaze | None | ~44,400 LOC, 8 open blockers | 492M transitions vs 1-day cap **[S]** | **No** |
| ICRA/BARN | Sealed, unopened | 93 tests, 2 amendments, ledger chain | Gate Aug 24; venue mismatch | Yes for RA-L; not usefully for ICRA |
| Code contracts | `un_form_verdicts.json` (unused) | canonical helpers + 20 tests | 7 adapters still inline **[S]** | N/A — it is a defensibility fix |

---

## 2. The five things that matter

### 1. The paper is selling the wrong claim, and a four-study replication is already in hand at zero compute cost

**Evidence.** The same directional contrast — deployed-N score shape beats its N=2 slice — appears in four independent studies, two estimators, three environments, all with the same sign **[S]**:

| Study | Contrast | Effect | Paired support | Status of that contrast |
|---|---|---|---|---|
| N-sweep (N=4/8/16/32) | u_N − p(1−p) | +.0307/+.0920/+.1526/+.1909 | 8/8 at every N>2 | post-guidance, descriptive |
| Acrobot V2 | u₁₆ − p(1−p) | **+.0480**, CI [+.0209,+.0738], p=.0034 | 15/20 | **frozen confirmatory primary** |
| Digits, MaxRL | u₈ − p(1−p) | **+.20842** | 23/24 | secondary read of a frozen design |
| Digits, RLOO | u₈ − matched | **+.17665** | 24/24 | secondary read of a frozen design |

And the boundary replicates just as cleanly in the *other* direction: u_N vs uniform is −.0106/−.0032 at N=2/4 in the sweep, +.0419 on Acrobot, −.11279 and −.37581 in Digits **[S]**. The largest, cleanest effect in the entire manuscript (+.208, 23/24, in the paper's only exact-probability environment) is currently printed once, in the appendix, under a header that calls the study a negative.

**Consequence.** Reframing converts the Digits study from the paper's self-inflicted wound into its strongest confirmation; makes 640 parameters one of four replications rather than the sole support; and removes the *mechanical* objection that a method paper with no baselines is below bar, because a shape characterisation owes no baseline. Estimated by the hostile-AC pass at +0.8 to +1.2 mean reviewer score, P(accept) 8–12% → 22–28% **[S]**. Cost: zero compute, roughly two to three days of writing.

**Guardrail the critiques did not state, and which must be in the paper or a sharp reviewer kills the reframing in one line:** only *one* of those four legs (Acrobot) is a frozen confirmatory primary. The N-sweep is explicitly post-guidance and not preregistered; the two Digits legs are secondary reads of a design whose frozen primary was the interaction. **Sell it as "one frozen primary plus three concordant supporting measurements," never as "four preregistered replications."** Put exactly that in the claim-to-evidence contract table.

### 2. Every positive result in the paper is confounded with peak hardness, and one ~20-run CPU arm decides it

**Evidence.** p*_N = 1 − N^(−1/(N−1)) falls .5 → .37 → .257 → .169 → .106 as N goes 2 → 4 → 8 → 16 → 32. Every winning arm in every study is the harder-peaked arm **[S/I]**. The paper concedes it verbatim at `body_iclr.tex:523-526`: "No arm pairs a mismatched-N score with a fixed deployed N, so the tournament supports rollout-aware difficulty targeting rather than peak-location specificity" **[S]**. A hand-set difficulty target of ≈.17 reproduces every number in the paper.

**Consequence.** The replicated finding, stated honestly, is *"harder-peaked scores beat p(1−p)"* — not *"the deployed-N peak is correct."* The only claimed advance over the ProCuRL/SFL/LILO p(1−p) literature is peak-location specificity, and nothing tests it. **The decisive experiment is one over-shooting arm on the existing Acrobot scaffold: score by u₆₄ (p*=.0639) while the deployed estimator stays at N=16, 20 paired seeds, same 2M-transition budget, same 60-run harness.** If u₁₆ beats both u₂ and u₆₄, peak-location specificity is proved and the confound dies. It is CPU-only, it is 1/3 of the V2 tournament's run count and 1/16 of the already-executed 320-run paid-probe family, and **it is not planned anywhere** — not in MAZE-SCORE, not in DESIGN_IMPROVEMENT_PLAN P0–P4.

Per-run wall clock is `UNKNOWN`. Obtain it from the timing fields in `curriculum_maxrl/` Acrobot tournament analysis JSONs / `ACROBOT_CURRICULUM_TOURNAMENT_RESULTS.md`, or from the paid-probe family's recorded wall clock, before committing to the arm.

**This is the only new experiment in the whole project whose result changes the paper's central claim rather than replicating it.** Note also that MAZE-SCORE reproduces this confound at 1.26M parameters (u₃₂ vs p(1−p), p* .106 vs .5) rather than removing it — which is fine and honest *under the reframed claim*, but means the two experiments are complementary, not substitutes.

### 3. The repository is a public, single-disk single point of failure for two double-blind submissions

**Evidence, all [V] this session.** `git rev-list --count origin/main..main` = **6**. `origin/main` tip is `2fe4481`, i.e. the six unpushed commits are the entire preregistration-freeze provenance chain for the ICRA campaign **that is running sealed right now**. `git remote -v` identified the author's account in the repository URL. `docs/paper-iclr.pdf` is 220,740 bytes, byte-identical to the double-blind submission PDF `paper/main_iclr.pdf` **[V]**. `docs/index.html` asserts the **retracted** zero-exception maze claim at **two** places: line 287 ("grew coverage under MaxRL in every seed, while GRPO decayed coverage in every seed") and line 583 ("GRPO's pass@8 decays in every seed") **[V]**, while retracting it 70 lines earlier. ICLR 2027 is double-blind; ICRA 2027 is double-anonymous. Anonymity policy appears nowhere in the repo **[S]**. Untracked-but-irreplaceable work totals 3.2 MB across ~44k LOC **[S]**.

**Consequence.** The paper already concedes that its maze/Acrobot locking commits are not externally verifiable (`body_iclr.tex:540-546`); the BARN freeze is now reproducing that failure *live and self-inflicted*, on one disk. And a title search de-anonymizes both submissions. **A single action fixes all of it: make the GitHub repository private, then push.** That preserves a server-side third-party commit timestamp (the freeze anchor), removes the de-anonymization vector, takes the Pages site with its live retracted claim offline, and eliminates the loss exposure — in one step, needing no content edits.

### 4. MAZE-SCORE is one commit away from being permanently underpowered *and* unanalyzable

**Evidence [V].** Four independent encodings of N all say 30, and the analyzer refuses more than 40 pairs. Power at SESOI +.005 with the pessimistic paired SD .0135 **[S]**:

| Blocks | Power @ α=.05 | Power @ Holm .025 | MIG-slice-h | Wall @ %5 throttle |
|---|---|---|---|---|
| 30 (current default) | 50–53% | 38–42% | 33.4 | 6.7 h |
| 40 (largest exact-test-legal) | ~65% | ~54% | 44.6 | 8.9 h |
| 60 | 81–82% | 71–74% | 66.9 | 13.4 h |
| **72 (recommended)** | **87–88%** | **80–82%** | **80.2** | **16.7 h** |

Arithmetic: block cost ≤ 3 × 1,337 s = 4,011 s = 1.1142 h (conservative — it triple-counts the one-time SFT prep); waves = ⌈N/5⌉. The entire difference between a coin-flip and a well-powered version of the project's most important experiment is **+46.8 MIG-slice-hours and +10.0 h of unattended wall clock**, against 42 days = 1,008 h of calendar. That is **1.0% of the remaining calendar**.

**Consequence.** The prereg forbids re-running, extending seeds, or substituting metrics after freeze **[S]**. Choose 72 and replace the exact sign-flip with a seeded Monte-Carlo sign-flip **while the prereg is still DRAFT** — that is a free edit today and an impossible one tomorrow. If Monte-Carlo substitution is refused on principle, take **40 and label it a precision compromise in writing**. **Never ship 30**, which is what the code does if nobody intervenes.

### 5. The declared critical path is not the critical path

**Evidence.** `CODEX_GOAL_ICLR_2026-08-12.md:20-22` names E2c "the sole remaining ICLR experiment and the critical path" **[S]**. But: the current abstract contains no Countdown sentence at all **[S]**; GATE-DR already ships 12 tracked raw-outcome LLM pass@16 runs including the project's sharpest non-proxy measurement instance **[S]**; and E2c changes only an appendix measurement caution. Meanwhile the two lanes consuming the most agent-hours — UED (~37 h in 37 wall-hours, 0 numbers) and a prospective ICRA manuscript (Aug 25 – Sept 14) — **cannot reach a paper number by Sept 25 and Sept 15 respectively** [see §4].

**Resource arithmetic that settles it [V]/[I].** Compute utilization to the binding deadlines: local RTX 5090 = 3.32 h of training / 337.1 h to the Aug 28 hard stop = **0.99%**. Hopper GPU = 80.2 MIG-slice-h compressed to 16.7 h wall / 1,008 h to Sept 25 = **1.7%**. **Compute is not the constraint. A single serialized agent is.** At ~6 productive h/day × 42 days ≈ 252 agent-hours, the executable plan in §3 consumes ≈95 h (38%); the UED lane alone, continued at its observed burn rate, would consume all of it.

**Consequence.** Reclassify E2c as *cheap opportunistic upside, time-boxed to 2 human-hours*; kill UED and the ICRA manuscript outright; and spend the reclaimed agent-hours on the §2.1 reframe and the §2.2 experiment.

---

## 3. Critical path to ICLR Sept 25

**Master arithmetic.** Now = 2026-08-14 23:00 EDT. To Sept 25 AOE = **42 days / 1,008 h**. To Sept 18 abstract = 35 d. To internal title/abstract freeze Sept 16 = 33 d. To Aug 28 E2c training stop = **337.1 h** (Aug 14 22:53 → Aug 28 23:59). To Aug 24 ICRA gate = 10 d. Total compute demanded by the entire plan below: **3.32 h local GPU + 80.2 MIG-slice-h Hopper GPU + UNKNOWN (≤ 1/3 of the V2 tournament) CPU** — i.e. under 2% of available compute on every machine. **The schedule is agent-hour-bound, and the plan below is sized at ≈95 agent-hours of ≈252 available.**

### Dependency graph (critical path in bold)

```
[make repo private + push] ──┐
                             ├──> BARN freeze provenance secured
[E2c amend+fix+launch] ──> E2c endpoint (Aug 15) ──> E2c writeup (Aug 25-28)

**[selective clean commit] ──> [P0-A canonicalization] ──> [MAZE-SCORE power memo + N reconcile]
     ──> [prereg FROZEN commit] ──> [stage evidence] ──> [submit array] ──> [fetch]
     ──> [analyze once] ──> D3 branch ──> **[paper reframe integrating result]**
                                              ^
[u64 Acrobot prereg + launch] ──> D4 ─────────┘
                                              |
[two tables + boxed algorithm + artifact repairs] ──┘
     ──> [single re-freeze] ──> [adversarial review Sep 6-12] ──> [freeze Sep 16] ──> [submit Sep 25]
```

### Dated plan

| Window | Actions | Resource | Arithmetic / fit |
|---|---|---|---|
| **Aug 14 23:00 – Aug 15 04:00** | Repo private + push (§5.1); E2c amendment → `env` fix → delete stray log → launch (§5.2–5.4); selective clean commit while E2c runs (§5.5) | Agent ~3 h; 5090 unattended 3.7 h | E2c training 3.32 h finishes ≈02:30; endpoint ≈02:55. 13.9 d of slack on the Aug 28 stop |
| **Aug 15** | BARN: poll scheduler metadata; seal when all 20 terminal (nominal 06:19Z = 02:19 EDT) + 30-min sealer. P0-A canonicalization + cross-adapter parity test. MAZE-SCORE outcome-blind power memo | Agent ~6 h; Hopper CPU (already running) | Gate Aug 24 is 209 h after nominal terminal ⇒ ~9 whole-cell retries still fit |
| **Aug 16** | Reconcile all four N sites + `MAX_EXACT_SIGN_FLIP_N`; record campaign receipt; flip prereg to **FROZEN** in a clean commit; `stage_maze_score.sh evidence`; submit array | Agent ~5 h; Hopper GPU 16.7 h unattended | 72 blocks × 1.1142 h = 80.2 MIG-h; 15 waves at %5 = 16.7 h wall |
| **Aug 17** | MAZE-SCORE terminal + fetch. BARN gate doc drafted **outcome-blind** with RA-L pre-commitment. u₆₄ Acrobot preregistration written and **publicly timestamped** | Agent ~5 h | Array submitted Aug 16 ~12:00 + 16.7 h + queue ⇒ terminal Aug 17 |
| **Aug 18** | **D3:** run `analyze_maze_score.py` exactly once. Launch u₆₄ Acrobot arm (20 paired seeds, CPU) | Agent ~3 h; local CPU (~12 free cores of 20) | Per-run wall UNKNOWN; ≤1/3 of V2 tournament |
| **Aug 19 – Aug 21** | **Paper reframe: the big writing job.** New abstract, two-part claim, Digits promoted to main text, cross-study figure, maze factorial demoted to context, PLR row deleted | Agent ~24 h | 3 days × 8 h |
| **Aug 22 – Aug 24** | PLR/robust-PLR/MaxMC/ACCEL/minimax comparison table; claim-to-evidence contract table at head of Evidence; boxed algorithm + convention table. **D5: ICRA gate Aug 24 (binding)** | Agent ~12 h | ~1 page of freed main-text budget exists (conclusion moved p9→p8) |
| **Aug 25 – Aug 28** | E2c closure per its frozen decision rule (whatever the gates say). Artifact repairs: `fig_maze_block_contrasts.py`, manifest rebuild, `reproduce.sh` verify-path fallback, `main_iclr2027.pdf` naming, OPENREVIEW abstract regeneration, Jugs limitation sentence, `CURRENT_STATUS.md` | Agent ~14 h | **Aug 28 = E2c training hard stop; enforce it literally** |
| **Aug 29 – Sept 5** | **D6:** select the go/no-go branch from MAZE-SCORE + u₆₄; integrate both results; full consistency pass; **one** re-freeze (pinned rebuild + receipt hash refresh + re-audit) | Agent ~16 h | Batch every figure/manifest change into this single re-freeze so the PDF hash changes exactly once |
| **Sept 6 – Sept 12** | Independent adversarial review. **Zero new engineering.** | Agent ~16 h | Protected window |
| **Sept 13 – Sept 16** | Review fixes; **D7: freeze title + abstract Sept 16** | Agent ~8 h | 2 days ahead of the Sept 18 abstract deadline |
| **Sept 17 – Sept 24** | Final build, claim trace regeneration, artifact bundle, buffer | Agent ~8 h | 8 days of pure buffer |
| **Sept 25** | Submit | — | — |

### Decision points and fallbacks

| ID | Date | Decision | Fallback if it goes the other way |
|---|---|---|---|
| **D1** | Aug 15 06:00 | E2c training complete? | If the GPU was re-occupied: the driver stops cleanly at a stage boundary. Retry only while the ceiling holds. If incomplete at Aug 28: write the dated closure note, keep recycling out of the contribution ladder (it already is), cite GATE-DR's 12 tracked runs for the raw-outcome recommendation. **Do not extend the hard stop.** |
| **D2** | Aug 16 | MAZE-SCORE: 72 blocks + Monte-Carlo sign-flip? | If quota refuses 72 → 60 (+MC). If MC substitution is refused → **40 + exact test, labelled a precision compromise in the prereg**. Never 30. |
| **D3** | Aug 18 | MAZE-SCORE verdict | *Supported*: it becomes the second frozen confirmatory leg of the reframed claim at 1.26M params — lead the Evidence section with it. *Practically ruled out* (CI upper < +.005): report at equal prominence, narrow to "the shape advantage does not survive to 1.26M parameters," which is still a publishable boundary. *Inconclusive*: show the interval, do not phrase non-significance as failed transfer, and the paper ships on the reframe alone. |
| **D4** | ~Aug 21 | u₆₄ Acrobot result | If u₁₆ > u₂ *and* u₁₆ > u₆₄: claim peak-location specificity — the paper's strongest possible outcome. If u₆₄ ≥ u₁₆: **state plainly that the replicated finding is "harder-peaked beats softer-peaked" and that deployed-N peak specificity is not supported.** That is a *better* paper than silently keeping the confound. |
| **D5** | Aug 24 | ICRA gate | Pre-commit **before unblinding**: a directional pass whose 95% paired bootstrap CI spans zero is an **RA-L** outcome, not an ICRA go. See §7.6. |
| **D6** | Sept 5 | Final paper branch | If neither MAZE-SCORE nor u₆₄ helps: the reframed manuscript is already true under that branch. Ship the diagnostic/boundary paper. |
| **D7** | Sept 16 | Title + abstract freeze | Regenerate `OPENREVIEW_ABSTRACT_CANDIDATE.md` from `main_iclr2027.tex` + `body_iclr.tex` as the file itself instructs; it is currently stale (old title, a Countdown sentence the abstract no longer contains) **[S]**. |

---

## 4. What to cut

**Cut 1 — the entire UED/AMaze lane, today, with a written closure note.**
Its own preregistration says it can never be paper evidence; its protocol needs ~492M transitions/run against a 1-day queue cap with no resume path; five dev seeds cap the exact p at .0625; no ACCEL/PAIRED/robust-PLR arm exists in `ued_benchmark/configs/`; four of its eight open blockers are provably unclosable without remote authorization, so local work on them is unfalsifiable **[S]**. Freeze the lane in place, record the pin (`minimax d053054c…`), the v3/v4 contract hashes, and the unmeasured-throughput blocker in a dated note, and stop **all** v4 remote-hardening, telemetry, and audit work. *Optional, post-submission only:* one ≤1 GPU-h, ~200-update steady-state throughput probe on the already-Hopper-proven v3 bundle to replace the compile-contaminated 248.97 tr/s figure for a future paper. Not before Sept 25.

**Cut 2 — the ICRA manuscript writing window (Aug 25 – Sept 14). Keep the campaign; change the venue.**
The campaign is nearly free (already running, CPU-only, 30-minute sealer). The manuscript is not: it consumes exactly the Aug 29 – Sept 5 branch-selection and Sept 6–12 adversarial-review windows that the plan itself calls the critical path. What ICRA reviewers will demand and what cannot exist by Sept 15: a BARN-native curriculum baseline (GCL/GACL), a PLR/HER arm, a hardware run — all blocked pre-gate by the frozen scope rule and post-gate by the calendar. **Go RA-L** (no deadline; the hierarchical per-course q_s repair, the competitor arms, and a Jackal run can all be added there) and fold a compact BARN paragraph + one figure into ICLR as a third domain, noting explicitly that the plan's prohibition covers the ICRA navigation *smoke* result, not the sealed preregistered campaign.

**Cut 3 — the 30-block MAZE-SCORE default.** See §2.4. This is a cut in the sense that "do nothing" ships it.

**Cut 4 — the corrected-code saturation-gate replication.** Technically feasible (3 seeds × ~30 min on the 5090 after E2c), but its entire payoff is upgrading one appendix's "provisional mitigation" wording, and the project's own frozen prohibition list says "do not reinterpret GATE-DR again" **[S]**. Keep the provisional wording.

**Cut 5 — inside the manuscript: the Coefficient-Activity PLR contribution row, and the maze factorial as a contribution rung.** The PLR row is labelled ENGINEERING ONLY / HOLD with no evidence anywhere; it hands a reviewer the sentence "the authors say their replay method has no evidence" for free, and deleting it plus its Related-Work paragraph recovers ~1/3 page toward the comparison table. The maze factorial is confounded by a shared, per-estimator-untuned learning rate and the paper already concedes it "does not test the acquisition score" — demote it to one context paragraph.

**Cut 6 — P3 (1.5B LLM curriculum benchmark) and all P4 algorithm extensions.** Explicitly post-ICLR. P3 is unstarted, has no protocol, no pool, no allocation, and would contend for the only free GPU.

**Cut 7 — housekeeping.** Reap the four orphaned `gzserver` processes (PIDs 2874122 / 2900798 / 2903603 / 2905340, all PPID 1, started Aug 12, pointed at `/tmp/icra_barn_adapter_test*` paths that appear nowhere in the repo and at the permanently-non-evidentiary course `barn-299`) **[V]** — unrelated to campaign 003, which runs exclusively on Hopper. Also delete or license-and-relocate `config/configs/maze/` (12 files **[V]**, byte-identical to upstream Apache-2.0 minimax configs, untracked and *not* in `.gitignore`).

---

## 5. Immediate actions (next 24 hours)

> **Ordering is load-bearing.** The watcher `/home/robotixx/.claude/jobs/ca9ae5b6/tmp/e2c_watch.sh` (PID 2842583, running 70.6 h) **[V]** polls every 600 s and its `attempts` counter increments **only** when readiness reports `launch_authorized_now: true` (script lines 11–13 **[V]**). Because readiness currently reports `false`, it logs "blocked" forever and burns nothing. **The moment the 0-byte log is deleted, the next tick (≤600 s) runs the driver.** So: amend → fix → *then* delete the log. Deleting first re-crashes the driver, recreates the log, and burns attempts 2 and 3.

**1. Decide repo visibility, then push.** *(needs human authorization — account-level)*
```bash
# 1a. confirm current visibility in a browser at the private author-owned remote
# 1b. RECOMMENDED: set repository to Private in GitHub settings (this also takes the
#     Pages site, with its live retracted claim and the byte-identical submission PDF, offline)
git -C /home/robotixx/curriculum-maxrl push origin main
```
Risk: pushing to a *public* repo adds the ICRA preregistration text under the author's name — a de-anonymization vector for a double-anonymous venue. Pushing to a *private* repo still records a server-side commit timestamp, which is exactly the external freeze anchor the project needs. **Do 1b before 1b's push if the repo is public.** Not pushing at all leaves six commits — the entire BARN freeze provenance for a live sealed campaign — on one disk.

**2. Write the dated, outcome-blind E2c amendment BEFORE touching the driver.** *(needs human authorization — `CODEX_GOAL_ICLR_2026-08-12.md:36` says run the driver "unchanged")*
Append to `autoresearch/iterate-260810-2240/E2C_PREREG.md`, mirroring the existing precedent at its own lines 7–19 and the BARN precedent: (a) the defect and its evidence (bash `readonly` command-prefix rc=1; 0-byte tee log; `integrity_status=fail` receipt); (b) the exact change — three `env` tokens at lines 156/208/287 — and the explicit sentence that it changes **no arm, seed, gate, dose, data, generation setting, endpoint, or decision branch**; (c) the OLD orchestrator identity (15,676 bytes, SHA-256 `729447c426944f060b88cae272d537fe78a89e61bc0db3c1b6467daebc2cd4b9`, recorded at `E2C_PREREG.md:110-112`) and the NEW hash after the edit; (d) confirmation that `E2C_CODE_MANIFEST.json` (`0e46b89f…`) is unaffected because the driver is not one of its 31 entries; (e) the test evidence. Then log it in a dated `ICLR_PROGRESS_REPORT_2026-08-15.md`.

**3. Apply the minimal driver fix.** *(covered by the same authorization)*
Insert the `env` command token at the head of the three command-prefix blocks — `verl_integration/run_e2c_rtx5090.sh` lines **156, 208, 287** (the blocks whose next line is `PYTHON_BIN="$PYTHON_BIN" MODEL_PATH="$MODEL_PATH" \` at 157/209/288 **[V]**). `env` receives `VAR=VAL` as its own argv, so bash never performs an assignment; **all four `readonly` guards and all four prereg-frozen values are preserved byte-for-byte.**
Then verify:
```bash
bash -n verl_integration/run_e2c_rtx5090.sh
python3 -m pytest curriculum_maxrl/countdown/ -q      # the 17 countdown protocol tests
sha256sum verl_integration/run_e2c_rtx5090.sh        # record the NEW hash into the amendment
```
Risk: **do not** instead drop `readonly` from lines 19/23/25/27 — that weakens the immutability lock on exactly the four prereg-frozen constants. **Do not** delete the self-referential assignments — `countdown_rtx5090.sh:24` defaults `STEPS` to 1, which would silently produce 1-step runs **[S]**.

**4. Delete exactly one file, then let the watcher launch (or launch manually).** *(low risk once 2+3 are done)*
```bash
rm /home/robotixx/curriculum-maxrl/autoresearch/iterate-260810-2240/e2c_logs/e2_clean_b1_s3_260809.log
bash verl_integration/run_e2c_rtx5090.sh --readiness-only   # expect integrity_status=pass,
                                                            # next_stage=train_e2_clean_b1_s3_260809
```
Nothing else needs removing: no `e2_clean_b1_s3_260809` checkpoint dir was ever created, and `/data/robotixx/curriculum-maxrl-runtime/e2c_gpu.lock` (0 bytes, 22:35 **[V]**) is only the flock target and is never inspected by any guard **[S]**. The watcher will pick it up within 600 s; or run the driver directly. Expect training complete ≈+3.32 h, endpoint ≈+3.67 h.

**5. Selective clean commit — NOT `git add -A`.** *(low risk; needs care)*
```bash
git -C /home/robotixx/curriculum-maxrl status --porcelain --untracked-files=all   # currently 280 [V]
```
Commit in 3–4 logical commits: **(a)** the frozen paper set **including** the two untracked hash-bound inputs `paper/FIGURE_TOOLCHAIN.json` and `paper/requirements-figures.lock` **[V]**; **(b)** the estimator/formula surface (`curriculum_maxrl/estimators.py`, `maze_gpu/train.py`, `maze_gpu/model.py`, `test_mass_formulas.py`, `maze_gpu/test_train_protocol.py`, `curriculum_maxrl/maze_score/`); **(c)** hopper scripts + `HOPPER_STATUS.md` + `DESIGN_IMPROVEMENT_PLAN.md` + `autoresearch/` receipts; **(d)** `ued_benchmark/` as an explicit frozen-closure commit.
**Explicit exclusions:** `.codex/` (the goal document forbids committing it), `config/` (12 untracked byte-identical Apache-2.0 upstream files with no LICENSE/NOTICE **[V]** — move to `/tmp` and add to `.gitignore`), all `__pycache__`. After committing, re-verify that the `PAPER_BUILD_RECEIPT` hashes still hold at HEAD.

**6. BARN: poll scheduler metadata only.** *(needs remote access — I am prohibited; the human or an authorized agent must run it)*
```bash
./hopper/hopper.sh status 9367009 9367011 9367020 9367022
```
The prereg explicitly permits scheduler state, accounting, hashes and non-metric completion markers to be inspected while jobs run. **Never** `hopper.sh logs`, **never** `--allow-endpoints`. If all 20 are terminal, run `finalize_barn_ledger.sh` then `finalize_barn_campaign.sh` and treat `BARN_CAMPAIGN_SEALED` as the only retrieval signal. Last observed state is 2026-08-14T10:37:04Z — now ~16.5 h stale.

**7. Fix the public retracted claim (or take the site down with §5.1b).** *(low risk)*
`docs/index.html` lines **287** and **583** both assert the retracted zero-exception cohort claim **[V]**. If the repo stays public: strike both sentences, refresh `docs/paper-draft.pdf` (863,510 B, Aug 9) from `paper/main.pdf` (585,608 B) **[V]**, and add a dated currency banner. If the repo goes private, this resolves itself.

**8. Write the outcome-blind MAZE-SCORE power memo.** *(no authorization needed; zero endpoint exposure)*
Use **only** job 9366552's cost receipt (1,337 s/arm-block) and the pre-existing SD range .0077–.0135. Recommend **72 blocks (seeds 20–91) + seeded Monte-Carlo sign-flip**. Also copy job-9366552's `meta/profile.tsv`, `SHA256SUMS` and `COMPLETE` from `/data/robotixx/maze_score/hopper_cost_audit/` into `autoresearch/iterate-260813-2348/hopper_full_arm/job-9366552/` and commit them with the memo, so the sample-size justification is auditable from the repo alone.

**9. Reap the four orphaned `gzserver` processes.** *(trivial; verify first)*
PIDs **2874122, 2900798, 2903603, 2905340** — all PPID 1, started Aug 12 00:48–01:07, all pointed at `/tmp/icra_barn_adapter_test*/launch_000000_barn-299/active_course.world` **[V]**. `SIGTERM` them and `rm -rf /tmp/icra_barn_*`. Campaign 003 runs exclusively on Hopper, so there is no interaction.

---

## 6. Week-by-week plan through Sept 25

| Week | Goal | Resource | Deliverable | Gate |
|---|---|---|---|---|
| **Aug 15 – Aug 17** | Unblock and launch everything cheap; secure provenance | Agent ~14 h; 5090 3.7 h; Hopper GPU 16.7 h; Hopper CPU (running) | Repo private + pushed; E2c endpoint; BARN sealed; P0-A canonical + parity test; MAZE-SCORE **FROZEN** at 72 and submitted; u₆₄ prereg publicly timestamped | **MAZE-SCORE frozen and submitted by end of Aug 17.** If not, escalate — it is the only experiment that repairs the paper's stated weakest link |
| **Aug 18 – Aug 24** | Get the two new numbers; begin the rewrite; close ICRA | Agent ~30 h; local CPU (u₆₄) | MAZE-SCORE analyzed once (D3); u₆₄ arm running; reframed abstract + Introduction + Evidence skeleton; BARN gate decision doc | **D5 Aug 24 ICRA gate, binding.** RA-L pre-commitment recorded *before* unblinding |
| **Aug 25 – Aug 28** | Finish the rewrite; close E2c; repair the artifact | Agent ~26 h | Full reframed draft; both new tables + boxed algorithm; `fig_maze_block_contrasts.py` + manifest rebuild; `reproduce.sh` verify-path fallback; Jugs limitation sentence; `CURRENT_STATUS.md`; E2c writeup | **Aug 28 E2c training hard stop enforced literally.** Any incomplete stage → dated closure note, no extension |
| **Aug 29 – Sept 5** | Branch selection and the single re-freeze | Agent ~16 h | Final claim set chosen (D6); all results integrated; **one** pinned rebuild + receipt hash refresh + independent re-audit | GO audit must return P0=P1=P2=0 at **HEAD**, not in a dirty tree |
| **Sept 6 – Sept 12** | Independent adversarial review | Agent ~16 h | Review report + triaged fix list. **Zero new engineering, zero new experiments** | No P0 findings survive into Sept 13 |
| **Sept 13 – Sept 16** | Review fixes; freeze identity | Agent ~8 h | Final title + abstract; `OPENREVIEW_ABSTRACT_CANDIDATE.md` regenerated from its declared sources | **D7 Sept 16 title/abstract freeze** |
| **Sept 17 – Sept 24** | Final build and artifact bundle; buffer | Agent ~8 h | Sept 18 abstract submitted; final PDF + hashes + claim trace + artifact | Claim trace regenerated, 0 UNTRACED, page bound asserted |
| **Sept 25** | Submit | — | ICLR 2027 submission | — |

Total ≈ 118 agent-hours planned against ≈252 available (42 d × 6 h) = **47% loaded**, leaving genuine slack for one MAZE-SCORE campaign retry (≈20 h) and one unforeseen defect class.

---

## 7. Open questions requiring a human decision

1. **Repository visibility and anonymity policy.** *Blocks §5.1 and §5.7.* Is the author-owned remote public? It hosts the byte-identical double-blind submission PDF and a Pages site stating every headline number under the author's name, against a double-blind ICLR and a double-anonymous ICRA. I could not check (no network). Proceeding either way under an assumption is unsafe: pushing to a public repo worsens de-anonymization; not pushing leaves the ICRA freeze provenance on one disk. **Recommendation: private, then push.**

2. **Authorization to edit the hash-pinned E2c orchestrator under a dated amendment.** `CODEX_GOAL_ICLR_2026-08-12.md:36` says run it "unchanged," and `E2C_PREREG.md:110-112` records its byte size and SHA-256. The same document's item 4 means the alternative to fixing is that **E2c never runs at all**. The repo has clear precedent for exactly this pattern (`E2C_PREREG.md:7-19`; the BARN operational amendments). **Recommendation: authorize, with the amendment written before the edit.**

3. **The paper's central claim: reframe or not.** Moving from "coefficient activity is a rollout-aware acquisition hypothesis" to "the deployed-N score shape dominates its N=2 slice across every environment and estimator tested, while its advantage over uniform does not replicate" is the single highest-leverage change available and costs no compute — but it is a genuine research judgment, it rewrites a manuscript that was already refocused on 2026-08-14, and it must carry the honesty guardrail in §2.1 (one frozen primary + three concordant supporting measurements, **not** four preregistered replications) and the peak-hardness caveat in §2.2. **Recommendation: reframe.**

4. **MAZE-SCORE inference design.** 72 blocks requires replacing the preregistered *exact* two-sided sign-flip with a seeded Monte-Carlo sign-flip (`MAX_EXACT_SIGN_FLIP_N = 40` **[V]**). That is a free edit while the prereg is DRAFT and an impossible one after FROZEN. Also still open and required before the smoke receipt freezes: the fresh primary evaluation panel (the repeatedly-used seed-12345 panel can only be a descriptive continuity check) and mazes-per-level.

5. **Whether to run the u₆₄ over-shooting Acrobot arm.** It is a *new* experiment requiring its own preregistration and, given the provenance criticism, a public immutable timestamp. Its per-run wall clock is `UNKNOWN` — obtain from the Acrobot tournament analysis JSONs / `ACROBOT_CURRICULUM_TOURNAMENT_RESULTS.md` timing fields before committing. It is the only experiment in the project that converts "harder peaks win" into "the deployed-N peak wins."

6. **ICRA vs RA-L, pre-committed before unblinding.** The gate as frozen is directional-only with no effect-size floor, so it can pass on evidence that supports no defensible claim. The pre-commitment — *a directional pass whose 95% paired bootstrap CI spans zero is an RA-L outcome* — must be recorded in writing **before** any endpoint is opened, or it is not a pre-commitment.

7. **How to spend the ~1 page of freed main-text budget.** Four candidates now compete for it: the PLR/ACCEL comparison table (open adversarial P1), the claim-to-evidence contract table (open adversarial P1), the cross-study replication figure (the reframe's centrepiece), and the boxed algorithm + convention table (last open manuscript-redesign item). They do not all fit. **My ranking: cross-study figure > claim-to-evidence table > comparison table > boxed algorithm**, with the last two compressed or moved to the appendix.

8. **Whether to downgrade the word "preregistered" throughout.** The paper concedes in three places that no study's locking timestamp is externally verifiable. Consistently using "internally locked before execution; timing not independently verifiable" is more defensible than the current mixed usage, but it visibly weakens the evidence ladder. This is a values call.

---

## 8. Risk register

| # | Risk | Likelihood | Impact | Mitigation | Trigger |
|---|---|---|---|---|---|
| 1 | Reviewer summarizes the paper as "a short algebraic identity, one 640-parameter positive family, one contextual-bandit counterexample, and an unrelated neural estimator comparison" | **High** — extractable from the paper's own Limitations; independently reached by three internal reviews | **Severe** — the difference between a 4 and a 6 | Execute the §2.1 reframe; land MAZE-SCORE as a second frozen leg; run u₆₄ | Any reviewer quoting the Limitations section back at the claim |
| 2 | Peak-hardness confound is named by a reviewer and the paper has no answer | **Medium-high** — the paper concedes it verbatim at `body_iclr.tex:523-526` | **High** — collapses the only claimed advance over the p(1−p) literature | Run the u₆₄ arm (§2.2); if it is not run, state the confound as the paper's own scope limit rather than leaving it in Limitations only | D4, ~Aug 21 |
| 3 | Repo is public → submission de-anonymized at one or both venues | **Medium** — public status unconfirmed **[UNKNOWN]** | **Severe** — desk rejection | Make private (§5.1); decide artifact-statement wording that does not name the repo | Browser check, tonight |
| 4 | Single-disk loss of the 6 unpushed commits **[V]** and 3.2 MB of untracked work | **Low-moderate** | **Severe** — loses the ICRA freeze provenance for a live sealed campaign | Push tonight (§5.1); selective commit (§5.5) | Any disk event; also `/data/robotixx/curriculum-maxrl-runtime` is 43 GB with no stated backup |
| 5 | MAZE-SCORE ships at 30 blocks and returns "inconclusive" | **Moderate** — 30 is what all four code sites say today **[V]** | **Critical** — the prereg forbids re-running or extending; the neural anchor is permanently spent | Freeze at 72 + Monte-Carlo, or 40 + exact (§2.4) | D2, Aug 16 |
| 6 | 60/72 blocks chosen without amending the test → `AnalysisError` **after** the campaign runs | **Medium** — the plan recommends 72, the analyzer caps at 40, nothing cross-checks them **[V]** | **High** — completed, unanalyzable campaign; any post-hoc substitution is a protocol violation | Reconcile all four N sites + `MAX_EXACT_SIGN_FLIP_N` in one pre-freeze commit; run the 13+7 tests and `bash -n` | D2, Aug 16 |
| 7 | RTX 5090 is re-occupied by out-of-scope Cosmos/OpenPI work before E2c launches | **Moderate** — those workloads held it continuously Aug 11→14 and are outside project authority | **High before launch** (window closes); **Low after** (`require_gpu_clear` stops cleanly at a stage boundary, before any tee log is created) | Launch tonight; do not touch the occupying processes | Any `gpu.memory_used_mib` ≥ 4096 in the readiness receipt |
| 8 | E2c crashes mid-training and leaves a partial tee log, re-triggering the same hard block on both the driver and `--readiness-only` | **Moderate** over a 4-stage, 3.3 h unattended run | **Medium** — recoverable, but silently halts the run | Bundle a partial-log hardening (treat a zero-byte active log as absent, or write `.partial` and rename on success) into the same amendment | Any non-zero driver rc in `e2c_watch.log` |
| 9 | Agent-hours are consumed by the UED lane or ICRA writing instead of the reframe | **Moderate-high** — 645 of the plan's 1,185 lines are UED; pressure rises near the freeze | **High** — displaces the only work that can reach a paper number | Write the §4 kill list into a single dated decision note today so it cannot silently resurface | Any new UED audit round, any ICRA manuscript draft after Aug 24 |
| 10 | Reviewer greps the repo, finds `MaxRLFrontierTeacher` computing (1−(1−p)^N)(1−p) = u_{N+1} under a name matching the paper's method | **Moderate-high** — `teachers.py:148` and `verl_curriculum.py:95` are unmodified and reachable from the READMEs **[S]** | **Medium-high** — attacks the central exact-mapping promise, though no ICLR number depends on those files | Route through the canonical helpers or mark them historical; rename; add the cross-adapter parity test; **and cite `un_form_verdicts.json`** (.21384 vs .21348) to show the forms are empirically indistinguishable | Aug 15, before the MAZE-SCORE source freeze |
| 11 | The reframing is attacked as "four preregistered replications" when three legs are not frozen primaries | **High if the guardrail is omitted** | **Severe** — a bigger overclaim than the current draft | State the status of each leg explicitly in the claim-to-evidence contract table (§2.1) | Drafting, Aug 19–21 |
| 12 | Ambiguous ICRA directional pass (CI spans zero) pulls three weeks into a likely-reject submission | **Moderate-high** — the gate has no effect-size floor and the backend is not trajectory-paired **[S/V]** | **High** — collides head-on with ICLR branch selection and adversarial review | Record the RA-L pre-commitment before unblinding; run the within-stratum homogeneity diagnostic (zero extra compute) | D5, Aug 24 |
| 13 | Artifact fails on a reviewer's machine: `reproduce.sh` verify path exits 1 without the pinned interpreter (`4627a60c…`), and its second derivation check is structurally unfailable (`[ -d ../maxrl ]`; `/home/robotixx/maxrl` does not exist **[V]**) | **Medium-high** — trivially checkable | **Medium** — undercuts the paper's principal rhetorical asset | Two-tier verify path (strict byte-compare when pinned, labelled fallback otherwise); vendor or delete the fig2c check; rebuild `manifest.json` around the compact paper's actual three figures | Aug 25–28 artifact window |
| 14 | Build toolchain lives outside the repo in a shared tree (`/data/robotixx/snmr-tools/bin/tectonic` = `397efac4…` **[V]**, an unrelated project's directory) | **Low-moderate** | **Medium** — frozen build becomes unreproducible on this host | Copy the pinned binary + the 483-member bundle cache to a project-owned path and record the hash; publish a checksum-bound bootstrap for release | Any change to `/data/robotixx/snmr-tools` |
| 15 | Known internal counter-example (Jugs E-LLM-3: plain MaxRL collapsing pass@k, making the coverage ordering pool-conditional) is found by a reviewer in the extended draft while the compact Limitations is silent | **Low-moderate** | **High** — a concealed-negative accusation is unrecoverable for a paper whose asset is candor | Add one sentence to the compact Limitations (§5, Aug 25–28 window); cost ≈2 lines | Drafting |

---

**Single highest-value action if only one thing happens: execute the §2.1 reframe.**
**If two: reframe plus the u₆₄ Acrobot arm.**
**If three: add MAZE-SCORE at 72 blocks, which under the reframe becomes the second frozen confirmatory leg at 1.26M parameters rather than a third-priority scale check.**
**Tonight, regardless: push (privately), fix and launch E2c, and land the clean commit.**
