# Adversarial review of the compact ICLR manuscript

**Review date:** 2026-08-14

**Decision:** **HOLD as a competitive method paper; GO for an immediate focused rewrite**

**Canonical submission source reviewed:** `paper/main_iclr2027.tex` ->
`paper/body_iclr.tex`

## Scope and evidence boundary

This is a read-only scientific and artifact review of the compact manuscript,
its wrappers and claim trace, the current project status, the checked-in AMaze
method contracts, and the pinned `minimax` documentation. No held performance
endpoint, BARN artifact, partial Hopper log, or sealed result was opened. The
only file created is this report.

The paper has a defensible core: the practical-MaxRL coefficient identity, the
`T=N-1` implementation-convention correction, the posterior-integrated score,
and unusually candid negative-result accounting. It is not yet a strong
competitive method paper because the only direct positive test of the proposed
score is a small Acrobot family. The neural maze result compares estimators, not
the score; the source-faithful AMaze score test has no endpoint; and no direct
matched ACCEL comparison exists. The strongest outcome-independent action is to
make the manuscript say exactly that, while preparing a clean branch that can
promote AMaze only if the frozen evidence supports it.

## Highest-leverage findings

### P0. The advertised reproducibility path can build the wrong paper and makes a registry claim this checkout cannot satisfy

**Evidence.** The intended ICLR-2027 wrapper explicitly loads the compact body
(`paper/main_iclr2027.tex:41-43`). The reproduction script instead compiles
`main_iclr.tex` and `main.tex` (`reproduce.sh:96-103`), and the former loads the
legacy `body.tex` (`paper/main_iclr.tex:40-42`). Nevertheless, the compact paper
says `bash reproduce.sh --build` validates inputs and builds both PDFs
(`paper/body_iclr.tex:1054-1056`). It also says the registry has 562 records
(`paper/body_iclr.tex:1056-1062`), while the checked-in
`curriculum_maxrl/run_registry.json:4` declares `n_rows: 53`. The claim trace
itself acknowledges that the 562-row object exists only on the release branch
and must not overwrite the different local registry
(`paper/CLAIM_TRACE_ICLR.md:96-106`). Finally, the trace announces a final
58/58 count (`paper/CLAIM_TRACE_ICLR.md:115`) before appending additional GATE-DR
and posterior claims (`paper/CLAIM_TRACE_ICLR.md:117-129`), so even its headline
count is no longer a complete inventory.

**Risk.** A reviewer or artifact evaluator following the stated command can
compile a different manuscript and cannot reproduce a central accounting
claim. This is a credibility failure independent of experimental outcomes.

**Edit now.** Make `main_iclr2027.tex` the sole submission build target; give
the 53-row and 562-row registries distinct canonical names; either vendor the
claimed object or remove the 562-row claim; regenerate the claim inventory from
structured assertions; and narrow the reproducibility statement to checks the
command actually executes. This is a submission P0.

### P0. The contribution ladder still outruns the evidence ladder

**Evidence.** The introduction presents three contributions, including a
positive/negative score result and neural estimator-conditioned coverage
(`paper/body_iclr.tex:53-79`). The Evidence section then correctly admits that
the score's positive support is small-scale, the neural maze study does not test
the score, and Countdown is only a reporting caution
(`paper/body_iclr.tex:297-305`). Current project status is stricter still: no
AMaze development performance run is authorized on the biased v1 ranking path
(`autoresearch/iterate-260814-0112/STATUS.md:127-142`), and both tie-aware v4 and
the `N={2,4,8}` package are DRAFT, engineering-only, and endpoint-ineligible
(`autoresearch/iterate-260814-0112/STATUS.md:143-155`).

**Risk.** A skeptical reviewer can summarize the paper as “a short algebraic
identity, one 640-parameter positive family, one contextual-bandit
counterexample, and an unrelated neural estimator comparison.” The present
three-contribution framing does not refute that reading.

**Edit now.** State one central contribution: **coefficient activity is a
rollout-aware acquisition hypothesis derived from the deployed estimator**.
Make the theory, the direct controlled score test, and the boundary test three
rungs supporting that one claim. Describe the current maze factorial only as
evidence that estimator choice matters for coverage measurement, not as neural
validation of the acquisition score. Keep an outcome-contingent manuscript
branch ready to promote AMaze only after a frozen direct score endpoint exists.

### P0. The manuscript's `FrontierMax` sampler is not the planned AMaze method

**Evidence.** The paper defines direct task probabilities using a Thompson draw,
a uniform floor, and proportional `[u_N]^gamma` weighting
(`paper/body_iclr.tex:243-254`); its appendix again describes a posterior draw
mapped through `u_N` and mixed with a floor (`paper/body_iclr.tex:853-858`). The
AMaze overlay instead replaces the score inside robust PLR while retaining the
buffer, replay gate, rank transform, staleness mixture, and with-replacement
sampling (`ued_benchmark/README.md:3-10`, `ued_benchmark/README.md:19-48`). Its
default score is deterministic posterior expected activity, not Thompson
sampling (`ued_benchmark/README.md:50-81`). Exact activity also requires the
4-by-8 grouped layout, whereas the official robust-PLR reference is 32-by-1
(`ued_benchmark/README.md:83-104`). Tie-aware v4 changes replay probabilities
for exact score ties and is shared with the matched MaxMC control; it is not
merely the direct sampler in the displayed `q(x)` definition.

**Risk.** If an AMaze result is inserted under the existing method name, the
paper will appear to claim evidence for an algorithm it did not run. Conversely,
calling v4 “score-only” without disclosing the common tie-aware transform hides
a material replay-policy change.

**Edit now.** Add a convention/implementation table with two explicitly named
instantiations:

1. **Direct coefficient-activity teacher:** direct pool sampling, floor,
   concentration, and the posterior rule actually used by the existing
   controlled experiments.
2. **Coefficient-Activity PLR:** robust-PLR replay with MaxMC replaced by the
   analytic `E[u_N(p)]`, exact `N=n_eval` grouping, inverse-rank priority,
   staleness mixing, replay gate, buffer capacity, and with-replacement draws.

Keep “FrontierMax” for at most one of these. Treat tie-aware ranking as a shared
experimental control/engineering correction unless the paper separately
formalizes and evaluates it as a contribution. Do not cite v4 performance; no
such result exists.

### P1. Novelty versus PLR, robust PLR, ACCEL, and `minimax` is underdeveloped

**Evidence.** PLR receives one sentence alongside ALP-GMM, and ACCEL/PAIRED are
dismissed as task generators without an Acrobot analogue
(`paper/body_iclr.tex:537-543`). Yet the new implementation is explicitly a PLR
overlay, and the pinned benchmark supplies PLR, robust PLR, ACCEL, parallel
variants, and AMaze (`/data/robotixx/ued_bench/src/minimax-d053054/README.md:54-70`,
`/data/robotixx/ued_bench/src/minimax-d053054/README.md:133-148`). The local
method contract itself says PPO, replay, buffer admission, staleness, and
sampling remain PLR semantics (`ued_benchmark/README.md:3-10`). The bibliography
cites PLR, PAIRED, and ACCEL but does not cite the `minimax` benchmark paper
(`paper/body_iclr.tex:704-711`).

**Risk.** The closest reviewer objection is not “this resembles ProCuRL”; it is
“this is PLR with a new replay score.” The current paper does not explain why
that score is theoretically or empirically preferable to MaxMC, value-loss/TD
scores, staleness, or ACCEL's mutation loop.

**Edit now.** Add a compact comparison table with columns for level source,
replay score, posterior state, dependence on deployed `N`, rank/staleness
transform, grouping, and mutation. State the narrow novelty bundle:

- exact unnormalized coefficient activity for the released drop-all-fail MaxRL
  convention;
- the `T=N-1` deployment correction and analytic posterior expectation; and
- using that estimator- and rollout-conditioned quantity as a level-acquisition
  score.

Do not claim novelty for Bayesian posteriors, Thompson sampling, intermediate
difficulty, PLR replay, or generic curriculum learning. Cite `minimax` as the
implementation and benchmark substrate once AMaze enters the paper.

### P1. Failure recycling, Countdown, and the paid-probe detour dilute the paper after being demoted from the contributions

**Evidence.** The introduction still opens with both curricula and failure
recycling (`paper/body_iclr.tex:24-40`), the method retains a recycling section
(`paper/body_iclr.tex:276-284`), and the main text spends 485 raw words on the
historical Countdown record (`paper/body_iclr.tex:440-501`). The conclusion
still recommends how “curricula and recyclers” should be evaluated
(`paper/body_iclr.tex:644-647`). This persists even though the contribution list
no longer claims recycling and the Evidence preamble labels Countdown only a
reporting caution (`paper/body_iclr.tex:297-305`). The ProCuRL section also
primarily establishes that a chosen cadence spent 93.2% of its budget on probes,
not competitive selector quality (`paper/body_iclr.tex:415-438`).

**Risk.** These branches consume attention needed to explain the actual method
and its closest baselines. They make the work look like a project anthology,
not one ICLR argument.

**Edit now.** Move the recycling method, Countdown result, gate history, and
detailed paid-probe study to the appendix. Retain one main-text sentence:
raw per-task outcomes are required to distinguish mean accuracy from standard
pass@k. Keep the paid-probe result as a cost-accounting limitation, never as a
named-method superiority comparison.

### P1. Several sentences overstate what was predicted or what `u_2` means in the Acrobot comparison

**Evidence.** The figure caption says coefficient mass is sign-blind and only
motivates the empirical coverage ordering (`paper/body_iclr.tex:169-176`), and
the prediction paragraph likewise says the curves motivate but do not prove an
ordering (`paper/body_iclr.tex:286-292`). The conclusion nevertheless says the
coverage ordering “it predicted persisted” (`paper/body_iclr.tex:635-638`). The
Acrobot experiment holds the deployed estimator at `N=16` while comparing the
`u_16` score with the `u_2=p(1-p)` score (`paper/body_iclr.tex:367-378`), yet the
conclusion calls the latter the method's “N=2 slice” without restating that the
estimator still uses 16 rollouts (`paper/body_iclr.tex:639-640`). The abstract's
“also beats uniform sampling” is true only inside that fixed eight-threshold
family (`paper/body_iclr.tex:10-12`).

**Risk.** “Predicted” suggests a directional learning theorem the paper
explicitly denies. “N=2 slice” can be read as an estimator group-size ablation,
which was not run.

**Edit now.** Replace the conclusion language with “a separately specified
estimator-conditioned ordering was observed.” Call the Acrobot comparator
“`u_2` scoring under the same deployed `N=16` estimator,” and qualify the
uniform result as occurring “in this fixed eight-threshold family.” Replace
“the positive-negative pair is itself the contribution”
(`paper/body_iclr.tex:60-71`) with the concrete boundary it establishes; reviewers
rarely accept outcome polarity itself as novelty.

### P1. The statistical hierarchy is candid but not yet organized into a persuasive evidence contract

**Evidence.** The maze primary is two sampler-specific six-block sign tests at
their exact granularity floor, while the inferential anchor is a post-hoc
Student-t interval (`paper/body_iclr.tex:344-353`). The locking commit is not
vendored (`paper/body_iclr.tex:323-329`), checkpoint trajectories needed for the
robustness multiverse remain external (`paper/body_iclr.tex:355-363`), and the
four evidence families have separately frozen alpha-0.05 primaries with no
paper-level error-rate claim (`paper/body_iclr.tex:589-598`). Acrobot's 20-seed
count came from a precursor rather than power analysis, and the exact paired
test assumes sign exchangeability (`paper/body_iclr.tex:589-596`).

**Risk.** The paper contains careful caveats, but reviewers must reconstruct
which claims are confirmatory, internally frozen, externally recorded,
post-guidance, post-hoc, or merely descriptive. Caveats buried in prose do not
create a clear inferential hierarchy.

**Edit now.** Put a claim-to-evidence table at the start of Evidence with:
claim, study, intervention, independent unit, endpoint, freeze/timing status,
uncertainty procedure, multiplicity family, result, and allowed scope. Label the
maze factorial secondary until a direct neural score test exists. Do not pool
sampler observations, tasks, episodes, or checkpoints as independent units.
Avoid “confirmed” for any study whose public pre-execution timing object cannot
be audited; “externally specified fresh wave” is accurate.

### P1. The rollout-count story needs an identifiability firewall before the factorial result exists

**Evidence.** The paper's existing `N` evidence is a post-guidance synthetic
fixed-completion sweep (`paper/body_iclr.tex:193-226`). It already admits that
Acrobot has no mismatched-score/deployed-`N` arm and therefore does not identify
peak-location specificity (`paper/body_iclr.tex:589-596`). The planned AMaze
factorial jointly changes `N`, `n_eval`, `n_parallel`, and buffer size; its
machine-readable protocol explicitly says cross-`N` effects are joint-layout
effects, not pure estimator-`N` effects
(`ued_benchmark/analysis/development_protocol_v3_n_factorial_tie_aware_draft.json:53-83`).
The same protocol forbids endpoint access, Hopper/GPU submission, production
scheduling, and paper-evidence labels
(`ued_benchmark/analysis/development_protocol_v3_n_factorial_tie_aware_draft.json:3-18`).

**Risk.** A future positive contrast-of-contrasts could easily be narrated as
validation that increasing rollout count improves coefficient activity, even
though level diversity and replay-buffer capacity change simultaneously.

**Edit now.** Preserve the current `N` sweep as descriptive, explicitly call
Acrobot a score-shape ablation at fixed estimator `N=16`, and prewrite the AMaze
estimand as: “within each layout, Frontier-minus-MaxMC isolates score choice;
across layouts, contrast modification is joint in grouping and buffer design.”
The five development seeds cannot support an exact two-sided sign claim
(minimum nonzero `p=.0625`; `autoresearch/iterate-260814-0112/STATUS.md:150-155`).
Any paper-level `N` mechanism claim needs untouched confirmatory seeds and the
predeclared multiplicity family.

### P1. “Beat PLR/minimax/ACCEL” and robotics transfer are correctly still HOLD and must remain outside the manuscript

**Evidence.** The source target is a full AMaze OOD comparison after 30,000 PPO
updates; published robust-PLR and ACCEL LSTM means are `.82±.02` and
`.83±.02`, respectively
(`autoresearch/iterate-260814-0112/RESULTS.md:245-250`). The project already
requires a direct current-code matched ACCEL rerun and a positive paired
interval on the same twelve-maze panel; an unpaired point estimate above `.83`
or a three-maze development score is explicitly insufficient
(`autoresearch/iterate-260814-0112/RESULTS.md:245-250`). The BARN campaign is a
separate sealed workflow, no endpoint has been inspected here, and its
stratum-level priority is not source-faithful PLR
(`autoresearch/iterate-260814-0112/STATUS.md:159-163`).

**Risk.** Adding an early AMaze point estimate or any BARN smoke result would
convert a carefully scoped paper into an invalid state-of-the-art or robotics
transfer claim.

**Experiment HOLD.** The minimum competitive matrix is source-faithful robust
PLR, direct matched ACCEL, and the selected coefficient-activity arm under the
same 30k-update accounting and full twelve-maze evaluator; DR/PAIRED should be
included for benchmark context if compute permits. The training seed is the
inferential unit. Report the seed-by-maze matrix, paired uncertainty, simulator
steps, PPO updates, optimizer applications, wall time, and GPU allocation. A
superiority claim requires the predeclared paired interval to exclude zero in
the favorable direction. Keep BARN/robotics in the separate ICRA track until
course-level priority semantics and sealed evidence are valid.

## Recommended paper structure that does not require new endpoints

1. **Problem and convention.** Define the exact practical-MaxRL estimator,
   coefficient activity, event interpretation, `T=N-1`, and posterior-integrated
   priority. State immediately that activity is not learning progress.
2. **Two method instantiations.** Separate the direct task teacher from the PLR
   replay-score implementation in a convention table and compact algorithm.
3. **Controlled evidence.** Present the synthetic `N` sweep, Acrobot score-shape
   test, and Digits boundary as one coherent ladder, with inferential status
   visible in a table.
4. **Neural supporting evidence.** Keep the existing maze estimator factorial as
   scoped context only. Insert AMaze here later only under its frozen
   outcome-dependent branch.
5. **Closest comparisons.** Explain PLR, robust PLR, MaxMC, ACCEL, and `minimax`
   before broader RLVR selector literature.
6. **Limitations and measurement.** Retain the raw-outcome/pass@k recommendation;
   move recycling, Countdown, probe-cadence detail, gates, and registry mechanics
   to the appendix/artifact.

## GO/HOLD decision table

| Action or claim | Decision | Reason |
|---|---|---|
| Fix wrapper, registry, claim trace, and reproduction assertions | **GO now / P0** | Outcome-independent correctness issue. |
| Rewrite around one coefficient-activity acquisition thesis | **GO now** | Matches the evidence currently available. |
| Separate direct teacher from Coefficient-Activity PLR | **GO now** | Prevents method/result identity drift. |
| Add PLR/robust-PLR/ACCEL/`minimax` comparison table | **GO now** | Required to make novelty legible. |
| Move recycling/Countdown and most paid-probe detail to appendix | **GO now** | Improves coherence without suppressing negative evidence. |
| Retain existing maze factorial as scoped estimator evidence | **GO, secondary only** | It does not validate `u_N` and has external provenance/raw-data limits. |
| Claim tie-aware ranking improves AMaze performance | **HOLD** | v4 is engineering-green but DRAFT and endpoint-ineligible. |
| Claim a pure neural rollout-count mechanism from the `N` factorial | **HOLD** | No endpoint; the factor jointly changes layout and buffer capacity. |
| Claim superiority over robust PLR, ACCEL, or published `.83` | **HOLD** | Requires direct matched full-panel paired confirmation. |
| Claim robotics/BARN transfer | **HOLD / separate paper** | Sealed endpoint and task-granularity issues remain. |

## Outcome-contingent publication branches

- **AMaze Frontier > group-matched MaxMC, and Frontier > matched ACCEL with a
  positive paired interval:** promote Coefficient-Activity PLR as the empirical
  anchor and make the existing estimator maze study supporting context.
- **Frontier > MaxMC but not ACCEL:** claim an improved PLR priority in the
  frozen layout, not benchmark superiority; ACCEL remains the competitive
  ceiling.
- **Frontier is negative or inconclusive:** keep the paper as a rigorous
  estimator-diagnostic and boundary paper, remove neural method-effectiveness
  language, and do not force a state-of-the-art story.
- **The `N` factorial is positive:** report only within-layout score contrasts
  and joint-layout effect modification unless a later design isolates `N`.
- **BARN is positive:** evaluate it under its independently frozen ICRA contract;
  do not use it as cross-domain corroboration in this ICLR manuscript.

## Bottom line

The highest-value improvement is not another loosely connected domain. It is a
one-to-one alignment among theory, named method, direct intervention, closest
baseline, and claim. That alignment can be repaired in the manuscript now. The
stronger empirical version remains contingent on a tie-aware, production-frozen
AMaze comparison against group-matched MaxMC and direct matched ACCEL, with the
full OOD panel and seed-level paired inference.

## Static focus-rewrite follow-up (2026-08-14)

The outcome-independent rewrite was implemented without inspecting or changing
an experiment endpoint. The main narrative now contains only the proved
coefficient identities, the direct fixed-pool score evidence and Digits boundary,
and the maze estimator-conditioned measurement. One three-row
contribution--evidence table prevents evidence transfer from the theorem or direct
teacher to the engineering-only PLR overlay. Paid-probe, recycling, gate, and
Countdown detail --- including the valid Countdown figure and numerical record ---
is preserved in the appendix. AMaze, robotics, and performance against PLR,
ACCEL, or another `minimax` baseline remain explicitly **HOLD**.

Static verification passed: balanced LaTeX environments and braces, unique labels,
no unresolved references, a complete 62-row claim inventory, `git diff --check`,
and the mass-formula enumeration script. No TeX compiler is installed in this
environment, so PDF compilation was not available.
