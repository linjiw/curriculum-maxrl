# Claim-to-artifact trace table for `paper/body_iclr.tex`

**Purpose.** This bounded inventory traces 92 quantitative claim rows: 58
base-table rows plus 34 dated addendum rows, covering every quantitative claim
in the main ICLR narrative and the previously audited secondary rows retained
in the appendix. It is not an inventory of every appendix number.
Compiled 2026-08-12 and re-audited 2026-08-14 after two static-only focus passes.
Those passes changed no quantitative value or supporting artifact. The paid-probe,
Countdown, recycling, and gate details now appear only in the appendix; their trace
rows remain below so moving a claim cannot erase its provenance.

**Provenance note (important).** Several inputs were originally read from the remote
release branch `origin/codex/curriculum-maxrl-research`; rows still marked
**[branch]** retain that provenance. Seven compact supporting artifacts were later
vendored additively into this working tree, as detailed below. The release branch's
562-row registry is intentionally not copied over the distinct 55-row compact
registry in this checkout. The current manuscript claims the local 55-row count.
Every cited object was opened and matched; nothing below is inferred from an
unopened file.

**Status legend.** TRACED = exact match (counts, exact p-values, config constants, or
values reproduced at full precision). TRACED-ROUNDED = matches the artifact's
higher-precision value after rounding to the manuscript's precision. Derivations
(e.g. population-SD to sample-SD, or a difference of two stored means) are noted inline.

**Location note.** Line-number hints in the base table predate the final focus
rewrite. Rows 37--55 and 59--61 are now appendix-only; row 57 is split between
the main reproducibility statement (60-run tournament) and appendix
(320 paid-probe runs). The main narrative now presents the direct Acrobot
rows 24--36 before the secondary maze rows 14--23; the trace-table row order
still reflects the original extraction. The claims and supporting values are
unchanged.

## Trace table

| # | Claim (verbatim-ish, approx. line in body_iclr.tex) | Artifact path | Value in artifact | Status |
|---|---|---|---|---|
| 1 | N-sweep `u_N - p(1-p)`: .0000, +.0307, +.0920, +.1526, +.1909 (Table 1, L207) | curriculum_maxrl/results_fixed_budget_n_sweep.json **[branch]** (`by_n/*/paired_contrasts/u_n_minus_learnability/metrics/normalized_auc_mean_pass`) | 0.0, 0.03072, 0.09195, 0.15256, 0.19089 | TRACED-ROUNDED |
| 2 | N-sweep `u_N - uniform`: -.0106, -.0032, +.0309, +.0453, +.0836 (Table 1, L208) | same **[branch]** (`u_n_minus_uniform`) | -0.01064, -0.00318, 0.03088, 0.04528, 0.08359 | TRACED-ROUNDED |
| 3 | wins vs `p(1-p)`: identical at N=2, then 8/8 at each N (Table 1, L209; "reproduces bit-for-bit" L214) | same **[branch]** (`positive_seeds`; `checks/u_2_equals_p_times_one_minus_p_pairwise`) | N=2 contrast exactly 0.0 (0 pos, 0 neg; pairwise-identical check true); N=4..32 positive_seeds=8, negative=0 | TRACED |
| 4 | "only 5/8 seed trajectories are monotone" (L217-218) | curriculum_maxrl/FIXED_BUDGET_N_SWEEP.md **[branch]** | "only 5/8 individual seed trajectories are monotone across N={4,8,16,32}" | TRACED |
| 5 | paired positive-seed counts vs uniform "3/8, 3/8, 8/8, 7/8, and 7/8" (L219) | curriculum_maxrl/results_fixed_budget_n_sweep.json **[branch]** (`u_n_minus_uniform/positive_seeds`) | 3, 3, 8, 7, 7 (of 8) | TRACED |
| 6 | "36-task CPU skill-chain testbed with N in {2,4,8,16,32}, eight paired seeds ... exactly 51,200 sampled completions per cell" (L190-192) | same **[branch]** (`config`, `protocol`) | 3 nested chains x 12 levels (=36 tasks); n_values [2,4,8,16,32]; seeds 0-7; total_completions 51200; hindsight false | TRACED |
| 7 | Digits interaction "+.01589" (L231; App L850) | curriculum_maxrl/digits_factorial/analyses/confirmation_registered_v1/confirmation_analysis.json **[branch]** (`tuned/contrasts/interaction/mean`) | 0.015885210445282034 | TRACED-ROUNDED |
| 8 | Digits interaction "95% CI [-.01686,+.04712]" (L231-232) | same **[branch]** (`bootstrap_percentile_95`) | [-0.016860614132443552, 0.04712022627021061] | TRACED-ROUNDED |
| 9 | Digits "exact sign-flip p=.350" (L232; abstract-adjacent) | same **[branch]** (`exact_two_sided_sign_flip_p`) | 0.34955739974975586 | TRACED-ROUNDED |
| 10 | Digits "15/24 positive" (L232) | same **[branch]** (`positive_blocks`, `n_blocks`) | 15 of 24, 0 ties | TRACED |
| 11 | "MaxRL favors u8 over p(1-p) by +.20842" (L233; App L853: CI [+.16791,+.24744]; 23/24) | same **[branch]** (`maxrl_u8_minus_p1mp`) | mean 0.20842053932699797; CI [0.1679053, 0.2474355]; positive_blocks 23/24 | TRACED-ROUNDED |
| 12 | "fresh 24-block Digits" counter-test (L227-228; abstract L16) | same **[branch]** (`n_blocks`, `n_complete_blocks`) | 24 complete blocks (seeds 32000-32023) | TRACED |
| 13 | (main-text implication L233-234; numbers in App L854-856) RLOO reversal "-.17665", below-uniform "-.11279" and "-.37581" | same **[branch]** (`rloo_p1mp_minus_u8/mean`, `maxrl_u8_minus_uniform/mean`, `rloo_p1mp_minus_uniform/mean`) | -0.17665011843643388; -0.11278554915649219; -0.37580823451272055 | TRACED-ROUNDED |
| 14 | "1.26M-parameter transformer trained on ... 17x17 mazes" (L286-288) | GPU_EXPERIMENT_HANDOFF.md **[branch]** (design table L216-217) | TinyTransformer d_model=128, six layers, ~1.26M parameters; fresh 17x17 Prim mazes, 13 BFS levels | TRACED |
| 15 | "N=32, eight prompt groups per training step, and 250 steps. Evaluation ... every 25 steps on 16 held-out mazes per level" (L295-297) | GPU_EXPERIMENT_HANDOFF.md **[branch]** (L220-225); curriculum_maxrl/maze_gpu_factorial/run_factorial_wave2.sh (header) | N=32/group; eight groups/update; 250 completed updates; eval at 0,25,...,250; fixed 16 tasks/level | TRACED |
| 16 | wave-1 endpoint failure: "only 3/6 uniform blocks and 4/6 FrontierMax blocks" (L303-304) | curriculum_maxrl/maze_gpu_factorial/FACTORIAL_VERDICT.md (L17-18); run_factorial_wave2.sh header | "uniform: 3/6 positive", "frontier_un: 4/6 positive" | TRACED |
| 17 | wave-2 primary "positive in 6/6 blocks under each sampler" (L313-314, L322-324; abstract L18-19) | curriculum_maxrl/maze_gpu_factorial/block_reanalysis.json (`registered_wave2_readout/P-F2`) | uniform_positive 6/6; frontier_un_positive 6/6; status registered_bar_met | TRACED |
| 18 | "exact two-sided sign p=.03125 per sampler" (L314, L324) | same (`exact_two_sided_sign_p_per_sampler`) | 0.03125 | TRACED |
| 19 | wave-2 sampler-averaged "mean +.0195" (L315, L326) | same (`P-F2/block_level_cov_auc/mean`) | 0.01949786324786326 | TRACED-ROUNDED |
| 20 | wave-2 "post-hoc 95% t interval [+.0115,+.0275]" (L315-316, L326-327) | same (`block_level_cov_auc/ci95_t`) | [0.011475511743969042, 0.027520214751757477] | TRACED-ROUNDED |
| 21 | "all 12 block averages are positive" across both waves / "12/12 ... descriptive" (L316, L328) | same (`cross_wave_exploratory/cov_auc`) | positive 12, negative 0, ties 0 (n_independent_seed_blocks 12) | TRACED |
| 22 | easy band: "four positive, one tie, one negative ... interval containing zero" (L317-318, L334-336) | same (`registered_wave2_readout/P-F3/block_level_easy_band`) | positive 4, ties 1, negative 1; ci95_t [-0.0032959, +0.1699626] | TRACED |
| 23 | "every leave-one-block-out interval remains above zero" (L327) | paper/results/maze_factorial_block_analysis.json **[branch]** (`wave2_registered_confirmation/leave_one_block_out`) | all_leave_one_out_t_intervals_exclude_zero: true (all remaining means positive) | TRACED |
| 24 | Acrobot design: eight nested predicates, H64 actor "(640 parameters)", MaxRL N=16, lr 3e-4, two million paid transitions (L343-348) | frontier_rl/examples/ACROBOT_CURRICULUM_TOURNAMENT_PROTOCOL.md **[branch]** (L54, L137); ACROBOT_CURRICULUM_TOURNAMENT_LOCK.json **[branch]** | "640 trainable parameters, learning rate 3e-4"; 2,000,000 actual transitions; lock: n_rollouts 16, learning_rate 0.0003, shared_h64 architecture | TRACED |
| 25 | "Twenty fresh paired seeds" (L348; table caption L358) | frontier_rl/examples/acrobot_curriculum_tournament_analysis.json **[branch]** (`primary/n_paired_seeds`); LOCK confirmatory_seeds | 20 (seeds 20000-20019) | TRACED |
| 26 | primary u16-p(1-p): "+.0480", CI "[+.0209,+.0738]", "p=.0034" (Table 2 L366; abstract L13-14; App L885-886: +.04803, [+.02094,+.07385], p=.003361) | same **[branch]** (`primary`) | mean 0.04803368836792599; CI [0.020936667560029185, 0.07384856540281254]; exact sign-flip p 0.003360748291015625; SESOI 0.01 | TRACED-ROUNDED |
| 27 | p(1-p)-uniform: "-.0062", "[-.0226,+.0122]", p ".5078", Holm ".5078" (Table 2 L367) | same **[branch]** (`secondary_uniform_auc_tests/p1mp_minus_uniform`) | -0.006159983362081301; [-0.022643721855486, 0.012195497121435254]; raw p 0.507843017578125; Holm 0.507843017578125 | TRACED-ROUNDED |
| 28 | u16-uniform: "+.0419", "[+.0218,+.0606]", p ".0008", Holm ".0016" (Table 2 L368) | same **[branch]** (`secondary_uniform_auc_tests/u16_minus_uniform`) | 0.04187370500584468; [0.021823939626013727, 0.06058598527620792]; raw p 0.0008087158203125; Holm 0.001617431640625 | TRACED-ROUNDED |
| 29 | "arm means are .64523 (uniform), .63907 (p(1-p)), and .68711 (u16)" (L373) | frontier_rl/examples/ACROBOT_CURRICULUM_TOURNAMENT_RESULTS.md **[branch]** (L39-41) | ".68711 vs .63907" (primary row), ".63907 vs .64523", ".68711 vs .64523" | TRACED |
| 30 | "15/20 paired differences are positive" (L374) | acrobot_curriculum_tournament_analysis.json **[branch]** (`primary/paired_differences`) | 15 of the 20 stored differences > 0 (5 negative) | TRACED (derived count) |
| 31 | "descriptively positive on sampled-group (+.05294)" (L379-380) | same **[branch]** (`secondary_descriptive_metrics/target_uniform_sampled_group_auc/.../u16_minus_p1mp/mean_paired_difference`) | 0.05294346776675794 | TRACED-ROUNDED |
| 32 | "and optimizer-update (+.05366) axes" (L380) | same **[branch]** (`target_uniform_optimizer_update_auc/.../u16_minus_p1mp`) | 0.05366310409977665 | TRACED-ROUNDED |
| 33 | "2.40 fewer groups and 3.42 fewer updates per million transitions" (L381-382) | same **[branch]** (`sampled_groups_per_million_transitions` and `optimizer_updates_per_million_transitions`, `u16_minus_p1mp`) | -2.4009389435384305 groups/M; -3.416178041934711 updates/M | TRACED-ROUNDED |
| 34 | "Native-success transition-AUC is .31085, .30521, and .37615" (L384) | same **[branch]** (`native_success_auc/arm_means`) | 0.310847230632091, 0.30521467542437297, 0.3761516195349437 | TRACED-ROUNDED |
| 35 | "mass means ... (.7344 versus .7490 per group; 115.57 versus 115.90 per million transitions)" (L385-386) | same **[branch]** (`coefficient_mass_per_group/arm_means`, `coefficient_mass_per_million_transitions/arm_means`) | 0.7343878949483179 vs 0.749039739344201; 115.57364891306358 vs 115.9003918114901 | TRACED-ROUNDED |
| 36 | "both descriptive paired intervals include zero" (L387) | same **[branch]** (u16_minus_p1mp bootstrap CIs for the two mass metrics) | [-0.0191079, +0.0499512] and [-3.6078814, +4.4416321] — both straddle 0 | TRACED |
| 37 | paid-probe: "80 paired seeds" (L394; L901 seeds 21000-21079) | frontier_rl/examples/acrobot_procurl_selection_analysis.json **[branch]** (`primary/n_pairs`); ACROBOT_PROCURL_SELECTION_LOCK.json **[branch]** | n_pairs 80; seeds 21000-... present in lock | TRACED |
| 38 | probed arms "refresh 20-episode-per-task estimates every 5,120 student transitions ... two-million paid budget" (L397-399) | ACROBOT_PROCURL_SELECTION_LOCK.json **[branch]**; arm names `*_b20_f5120` in analysis JSON | 5120 cadence, b20 probes, 2,000,000 budget present | TRACED |
| 39 | "u16-ProCuRL is +.00489 (paired t79=1.977, p=.0515 ...)" (L400; App L902-903: +.004894, SD .022139, t=1.9773, p=.05149) | acrobot_procurl_selection_analysis.json **[branch]** (`primary`) | mean 0.004894235861048817; sample_std 0.022138618105654644; t 1.9773310205711703; df 79; p 0.05149237843697304 | TRACED-ROUNDED |
| 40 | "percentile-bootstrap 95% CI [+.00011,+.00973]" (L401; App: [+.000110,+.009727]) | same **[branch]** (`primary/mean_ci95_paired_seed_bootstrap`) | [0.00011018286171968657, 0.009727194706612728] (20,000 resamples) | TRACED-ROUNDED |
| 41 | "point estimate is below the .02 SESOI" (L401-402) | same **[branch]** (`primary/sesoi`, `supported`) | sesoi 0.02; supported false | TRACED |
| 42 | "Neither adaptive-versus-sham secondary rejects after Holm (adjusted p=.4568 and .4547)" (L403-404) | same **[branch]** (`secondary_holm_family/procurl_minus_sham`, `u16_minus_sham`) | Holm 0.4567812019012931; Holm 0.45468238514441794 | TRACED-ROUNDED |
| 43 | "Probes consume about 93.2% of paid transitions" (L405; also L571; App L913: 93.19-93.21%) | same **[branch]** (`arm_descriptives/*/probe_fraction_of_paid`) | 0.932002 (ProCuRL), 0.932135 (sham), 0.931938 (u16) | TRACED-ROUNDED |
| 44 | "three probed arms trail ordinary uniform by -.314, -.309, and -.312 AUC ... (all Holm-rejected)" (L405-407; App L906-909 at higher precision) | same **[branch]** (`secondary_holm_family/{procurl,u16,sham}_minus_ordinary`) | -0.31377426306862766 (Holm 3.47e-66), -0.30888002720757884 (Holm 9.48e-67), -0.312070021671206 (Holm 4.06e-68); all reject | TRACED-ROUNDED |
| 45 | (App. paid-probe result) mean fixed-paid AUCs .33771 / .33942 / .65149 / .34261 | same **[branch]** (`arm_descriptives/*/auc_target_uniform_mean_success_fixed_paid_budget/mean`) | 0.337714, 0.339419, 0.651489, 0.342609 | TRACED-ROUNDED |
| 46 | Countdown design: "SmolLM2-360M ... N=16 ... three-seed aggregate on a fixed 128-task tier-1 evaluation set at step 60" (L415-418) | curriculum_maxrl/data_integrity_check.json (`test_tier_counts/countdown_tier1`); paper/figures/data/b_scoreboard_3seed.json (3-seed rows); curriculum_maxrl/countdown_reviewer_arms/reviewer_arms_verdicts.json | tier-1 eval 128 unique tasks; scoreboard stores 3-seed [mean, sd] tuples; reviewer arms list 3 seeds each | TRACED |
| 47 | "1,000 with-replacement resamples of size 16" bootstrap best@16 (L419-420) | paper/figures/data/fig9_bestk_proxy.json (`metric`); reviewer_arms_verdicts.json (`_metric_provenance`) | "1000 with-replacement resamples, not standard unbiased pass@k"; "VERL bootstrap best@16 coverage proxy" | TRACED |
| 48 | "zero SFT overlap for tier 1, and 27/128 exposed tier-0 tasks" (L426-427; also L633-634, App L992-994) | curriculum_maxrl/data_integrity_check.json (`sft_evaluation_overlap/tiers`) | tier1 sft_overlap_unique_tasks 0/128; tier0 27/128 (overlap_fraction 0.2109375, clean 101) | TRACED |
| 49 | "clean 101-task tier-0 analysis cannot be reconstructed" (L597-598) | same (`countdown_tier0/clean_unique_tasks`; `clean_tier0_reanalysis/status`) | 101; "blocked_missing_per_task_outcomes" | TRACED |
| 50 | "tier-1 mean@16 from .278±.066 to .324±.014" (L448-449) | paper/figures/data/b_scoreboard_3seed.json (`B1_t1`, `B2_t1`; stored SDs are population SDs, x sqrt(3/2) gives 3-seed sample SD) | 0.27767 (pop SD 0.05369 -> sample SD 0.0658) and 0.32433 (0.01161 -> 0.0142) | TRACED-ROUNDED (SD derived) |
| 51 | "bootstrap best@16 moves from .541±.024 to .492±.013" (L449-450) | same (`B1_t1[2:4]`, `B2_t1[2:4]`) | 0.54067 (pop SD 0.01991 -> sample SD 0.0244) and 0.49167 (0.01087 -> 0.0133) | TRACED-ROUNDED (SD derived) |
| 52 | replay control "reaches mean@16 .478±.021 and bootstrap best@16 .635±.046" (L457-458) | curriculum_maxrl/countdown_reviewer_arms/reviewer_arms_verdicts.json (`P_R2/t1_mean16`, `t1_pass16`) | [.459,.475,.5] -> mean .478, sample SD .0207; [.585,.674,.646] -> mean .635, sample SD .0455 | TRACED-ROUNDED (derived from per-seed values) |
| 53 | "all three replay endpoints vendored" (L458) | curriculum_maxrl/countdown_reviewer_arms/armB_replay_s{1,2,3}.json | three per-seed endpoint files present in this working tree | TRACED |
| 54 | "at most 12 auxiliary groups per 64 requested groups (18.75%)" (L460-461) | COUNTDOWN_ANALYSIS.md (cap=12 per amendment A2; "the dose rode its cap (12/12)"); GPU_EXPERIMENT_HANDOFF.md **[branch]** ("recycling affects at most 12 of 64 requested groups") | cap 12; 12/64 = 0.1875 exactly | TRACED (derived ratio) |
| 55 | GSM8K cell "missed its ... treatment-delivery gate by .00148" (L580-581; App L999: ".601480 versus <.60") | curriculum_maxrl/run_registry.json **[branch]** (row `gsm8k-steering-controlled-g3p`); FINAL_ICLR_REVIEW_AND_COMPLETION_GUIDE_2026-08-07.md (L205-210, local) | run_mean_dead_sampled 0.60148 vs gate <0.60 -> miss 0.00148; guide records 0.601480 and "fails ... by 0.00148" | TRACED |
| 56 | "55-row compact registry" with 35 maze, 11 Countdown, 7 GSM8K, one P0 analysis artifact, and one AMaze-gate analysis artifact (Reproducibility; App artifact accounting) | `curriculum_maxrl/run_registry.json` (`n_rows`, `rows[*].suite`) | n_rows 55; actual rows 55; by_suite {maze 35, countdown 11, gsm8k 7, group_law_flip 1, amaze_gate 1} | TRACED |
| 57 | source-locked 60-run Acrobot tournament (Reproducibility) and 320 paid-probe runs (Appendix) | `frontier_rl/examples/ACROBOT_CURRICULUM_TOURNAMENT_RESULTS.md` (all 60 completed); `frontier_rl/examples/acrobot_procurl_selection_analysis.json` (`primary/n_pairs`, `arm_descriptives`) | 60 tournament runs; 80 paired seeds x 4 arms = 320 paid-probe runs | TRACED |
| 58 | wave-2 AUC-multiverse anchors "(uniform 6/6, mean +.01496; FrontierMax 6/6, +.02404; sampler-averaged 6/6, +.01950)" (App L980-981; restates main-text L315/L322-326 numbers per sampler) | curriculum_maxrl/maze_gpu_factorial/block_reanalysis.json (`waves/wave2/repeated_sampler_contrasts`) | uniform cov_auc mean 0.01495726495726496 (6/6); frontier_un 0.024038461538461554 (6/6); block-level 0.01949786 (6/6) | TRACED-ROUNDED |

## Untraced claims requiring attention

**None.** All 58 base-table claims retained across the main text and explicitly
audited secondary appendix were matched at the manuscript's stated precision; the
four appended claim rows below are also traced.

Two caveats deserve attention even though no claim is untraced:

1. **Branch location.** 24 of the 58 rows originally traced only to files on
   `origin/codex/curriculum-maxrl-research`. **Update 2026-08-12:** the seven
   artifacts carrying those rows (`results_fixed_budget_n_sweep.json`,
   `FIXED_BUDGET_N_SWEEP.md`, Digits `confirmation_analysis.json`, both Acrobot
   analysis JSONs plus `ACROBOT_CURRICULUM_TOURNAMENT_RESULTS.md`, and
   `paper/results/maze_factorial_block_analysis.json`) were vendored additively
   from that branch into `main`, so those numbers are now auditable from this
   tree. The release branch's 562-row registry remains a distinct, branch-only
   audit object. It was not copied over the local 55-row compact registry; the
   manuscript and row 56 now use only the verified local count.
2. **Derived presentation conventions.** The Countdown ±SDs in the appendix are 3-seed
   *sample* SDs; `b_scoreboard_3seed.json` stores *population* SDs (conversion
   factor sqrt(3/2) confirms every value), and the ARM-B ±SDs are computed from the
   three per-seed endpoints in `reviewer_arms_verdicts.json`. The 18.75% recycling
   dose and the .00148 GSM8K gate miss are exact arithmetic on stored values
   (12/64; 0.60148 - 0.60). These are all consistent, but no single artifact stores
   the printed number verbatim.

**Base-table subtotal: 58/58 traced** (31 TRACED, 27 TRACED-ROUNDED;
0 UNTRACED).

## Addendum 2026-08-13: GATE-DR statistics (added to draft after the dose-response study)

| # | Claim (location) | Artifact | Status |
|---|---|---|---|
| 59 | "72--93\% rejection across settings" (Appendix negative branches) | `curriculum_maxrl/gate_dr/gate_dr_analysis.json` `dose_manipulation_check` (.721–.804 at .85; .883–.896 at .70; ARM A .934–.944 from `countdown_reviewer_arms/PROVENANCE.md`) | TRACED |
| 60 | "no operating point retaining the frozen fraction of the ungated mean gain" | same file, `settings.*.reproduces_useful_point` = false; verdict rule 4 | TRACED |
| 61 | Appendix: transfer 3/3 seeds; rejection grades 0/.72–.80/.88–.90/.93; mean@16 up while standard pass@16 falls .656→.414 (seed 1) | same file `transfer_gate` + `runs` (b1h_s1 `t1_pass16_standard` .6562, g0_s1 .4141) | TRACED |

## Addendum 2026-08-14: analytic Beta-posterior priority

| # | Claim (location) | Artifact | Status |
|---|---|---|---|
| 62 | $\mathbb E[u_N(p)]=1-(b)_N/(a+b)_N-a/(a+b)$ for $p\sim\operatorname{Beta}(a,b)$, $\mathbb E[u_N(p)]\leq u_N(\mathbb E[p])$, and the printed closed-form gap (task-sampling method paragraph) | `curriculum_maxrl/test_mass_formulas.py` (`beta_expected_activity`, six independent 256-point Gauss--Legendre checks, Jensen assertions, and exact gap checks) | TRACED |

**Cumulative subtotal through 2026-08-14: 62/62 traced** (35 TRACED, 27 TRACED-ROUNDED;
0 UNTRACED).

---

## Addendum 2026-08-18 — MAZE-SCORE rows (63–75)

Added after the frozen 48-block campaign `maze-score-v2-20260816-001` was
retrieved and analysed once. Campaign digest `1f9eb70447b212b1…`; all 48
per-cell `SHA256SUMS` verified; source manifest
`d98fe3ed02acbbeb7c1e29d9…`. Analyzer SHA-256 `197f1254…7bd5`, byte-identical
to the value frozen in `hopper/MAZE_SCORE_PREREG.md` before launch.

| # | claim in `body_iclr.tex` | value | artifact | status |
|---|---|---|---|---|
| 63 | primary `u_32 − p(1−p)` | −.00324 | `hopper/MAZE_SCORE_ANALYSIS.json` `contrasts.primary_un_minus_learn.mean` | TRACED |
| 64 | its 95% bootstrap CI | [−.00543, −.00111] | same, `.bootstrap_ci_95` | TRACED |
| 65 | its exact sign-flip p | .0054 | same, `.sign_flip_p_two_sided_exact` (.005416) | TRACED-ROUNDED |
| 66 | positive pairs | 15/48 | same, `.positive_pairs` / `.n` | TRACED |
| 67 | secondary `u_32 − uniform` | +.00888, [+.00657,+.01115], 41/48 | same, `contrasts.secondary_un_minus_unif` | TRACED |
| 68 | practically-ruled-out verdict | CI upper < +.005 SESOI | same, `.decision`, `.sesoi`; rule in prereg §"Decision rule" | TRACED |
| 69 | group draws used for calibration | 288,000 | `hopper/MAZE_SCORE_CALIBRATION.json` `.n_group_draws_used` | TRACED |
| 70 | binned predicted-vs-observed correlation | r = .90 | same, `.pearson_r_binned` (.8978) | TRACED-ROUNDED |
| 71 | silent groups at p̂≈.11 | 2.2% predicted, 51.2% observed | same, `bins[1]` `.predicted_dead_fraction_binomial` / `.observed_dead_fraction` | TRACED-ROUNDED |
| 72 | silent groups at p̂≈.22 | 0.03% predicted, 31.6% observed | same, `bins[2]` | TRACED-ROUNDED |
| 73 | realization ratios .43 / .78 / .93 | observed÷predicted at p̂ ≈ .11 / .45 / .73 | same, `bins[1]`, `bins[5]`, `bins[6]`; ratio derived from the two stored means | TRACED (derived) |
| 74 | per-arm predicted .81 vs .88, realized .44 vs .60 | `un` vs `learn` | `hopper/MAZE_SCORE_ARM_REALIZATION.json` | TRACED-ROUNDED |
| 75 | silent-group share 60.7% vs 32.2% | `un` vs `learn` | same, `.observed_dead_fraction` | TRACED |

Rows 69–75 are **post-hoc descriptive**: computed after the frozen primary by
`curriculum_maxrl/maze_score/calibration.py`, which is *not* the frozen
analyzer and computes no preregistered quantity. The manuscript labels them as
such at the point of use (§"Why: the i.i.d. assumption fails where the score
aims").

**Structural note.** This addendum accompanies a layout change, not a value
change: the maze factorial moved from a subsection to a paragraph, and
`tab:instantiations` plus the estimator-comparison figure moved to the
appendix. Every trace row above and in the base table still resolves; no
quantitative value was edited. One factual correction was made while adding
row 63's surrounding text: the MAZE-SCORE setup had been drafted as
"17×13-level mazes" and is now "17×17 mazes across 13 goal-distance levels".

---

## Addendum 2026-08-18b — group-law correction (rows 76–82)

Added after a direct audit of `curriculum_maxrl/maze_gpu/train.py` overturned
the mechanism wording of rows 69–75's surrounding prose. **Rows 63–75 and every
frozen endpoint are unchanged**; what changed is the explanation and the theory
that carries it. See `PI_CORRECTION_GROUPLAW_GRANULARITY_2026-08-18.md`.

| # | claim in `body_iclr.tex` | value | artifact | status |
|---|---|---|---|---|
| 76 | group semantics: one concrete maze per group, repeated $N$ times; posterior pools at the level | — | `curriculum_maxrl/maze_gpu/train.py` L680–689, L704 | TRACED (source) |
| 77 | Prop. 1, arbitrary group law: $A_N(Q)=2(\Pr(K{>}0)-\mathbb E[K]/N)$ | exact | `curriculum_maxrl/test_group_law.py::test_mass_identity_holds_for_arbitrary_group_laws` (dependent, anti-correlated, heterogeneous, random dense laws; $N=2..5$) | TRACED (proved + tested) |
| 78 | i.i.d. reduction parity to $2\{1-p-(1-p)^N\}$ | exact | same, `::test_iid_reduction_matches_closed_form` | TRACED |
| 79 | Cor. 2 granularity gap $=2[\Pr(K{=}0\mid z)-(1-\bar p_z)^N]\ge0$ for a mixture of conditionally-i.i.d. atomic tasks (the sign is not asserted for an arbitrary count law) | exact | same, `::test_granularity_gap_equals_twice_excess_all_fail`, `::test_granularity_gap_vanishes_without_heterogeneity`; scope guards `curriculum_maxrl/test_count_law_stats.py::test_gap_is_nonnegative_for_binomial_mixtures` and `::test_gap_inverts_under_anticorrelated_groups` | TRACED |
| 80 | both identities hold on the campaign to $<5\times10^{-16}$ over 41,101 / 18,497 / 9,355 cells at windows 10/25/50 | 2.8e-16, 4.4e-16 | `hopper/MAZE_SCORE_GROUPLAW_AUDIT.json` `.identity_checks` | TRACED |
| 81 | seed-clustered realization ratios .580 [.570,.590] vs .703 [.691,.715]; paired −.123, 48/48 negative | — | same, `.arm_un`, `.arm_learn`, `.paired_realization_ratio_un_minus_learn` | TRACED |
| 82 | silent-group shares 60.8% vs 32.1% (seed-clustered) | — | same, `.silent_group_share_mean` | TRACED |

**Supersedes.** Rows 74–75 quoted arm-level realization figures computed by
`calibration.py` with uncertainty implicitly at the group-draw level (.81/.88
predicted, .44/.60 realized, 60.7%/32.2% silent). Rows 81–82 recompute the same
quantities with uncertainty clustered on the 48 seed blocks, which is the
correct independent unit; the manuscript now cites the clustered figures
(.580/.703 ratios, 60.8%/32.1% silent). The small differences are the change of
aggregation unit, not of data. No frozen quantity is affected.

## Addendum 2026-08-26 — GROUP-LAW-FLIP rows (83–88)

Added after the frozen 48-block campaign `group-law-flip-v1-20260820-001`
was retrieved, hash-validated, and analyzed once. Source manifest
`b0cf3d2d...f3a2c95a`; frozen analyzer SHA-256 `9d88d6d...2186603`;
analysis artifact SHA-256 `c1e6dc3a...d952e9`.

| # | claim in `body_iclr.tex` | value | artifact | status |
|---|---|---|---|---|
| 83 | primary count-law minus plug-in cov-AUC | +.00666 | `curriculum_maxrl/group_law_flip/GROUP_LAW_FLIP_ANALYSIS.json` `primary_grouplaw_minus_plugin.mean` (.006655649...) | TRACED-ROUNDED |
| 84 | paired-bootstrap 95% CI | [+.00441,+.00887] | same, `.bootstrap_ci_95` [.004407051..., .008869191...] | TRACED-ROUNDED |
| 85 | exact paired sign-flip p | 9.56e-7 | same, `.sign_flip_p_two_sided_exact` (9.5580688e-7) | TRACED-ROUNDED |
| 86 | positive paired blocks | 40/48 | same, `.positive_pairs` / 48 seed blocks; 8 negative, 0 ties | TRACED |
| 87 | delivered mean visit TV | .33597, gate ≥.05 passed | same, `treatment_delivery.mean_tv`, `.threshold`, `.passed` | TRACED-ROUNDED |
| 88 | preregistered verdict | supported; observed mean ≥+.005 SESOI, CI lower >0, exact p≤.05, delivery passed | same, `primary_grouplaw_minus_plugin.decision`; frozen conjunction in `granularity_flip/GROUP_LAW_FLIP_PREREG.md` | TRACED |

The intervention establishes causal relevance of the count-law correction on
this substrate. It does not show that Corollary 2 predicted the downstream
sign, that the correction alone mediates the earlier MAZE-SCORE contrast, or
that either arm beats `p(1-p)`.

## Addendum 2026-08-26b — AMaze gate rows (89–92)

Added after all 20 checkpoint-budget receipts passed and the frozen analyzer
was invoked once. Analyzer SHA-256 `aaf54f22...51d755`; analysis artifact
SHA-256 `c0162e99...c86774e`.

| # | claim in `body_iclr.tex` | value | artifact | status |
|---|---|---|---|---|
| 89 | full-budget gate minus upstream solved rate | +.0633 | `ued_benchmark/AMAZE_GATE_ANALYSIS.json` `primary_mean_solved_rate.mean_paired_difference` (.063333334...) | TRACED-ROUNDED |
| 90 | paired-bootstrap 95% CI | [+.0003,+.1410] | same, `.paired_bootstrap_ci95` [.000333334..., .141000001...] | TRACED-ROUNDED |
| 91 | exact paired sign-flip p and positive pairs | .1562; 5/10 | same, `.exact_two_sided_sign_flip_p` (.15625), `.positive_pairs` (5), `.n` (10) | TRACED-ROUNDED |
| 92 | frozen verdict | `inconclusive_at_n10`; point ≥+.02 but p>.05, CI upper >+.02 | same, top-level `.verdict`; frozen conjunction in `ued_benchmark/AMAZE_GATE_PREREG.md` | TRACED |

This Tier-4 result neither promotes the development gate observation nor
changes the separate registered standalone-priority negative.

## Addendum 2026-08-26c — P0 arm means and frozen secondary (rows 93–96)

Added after the PI-requested completeness pass over the already-unsealed P0
analysis. These rows report existing frozen outputs; they do not add an
endpoint, threshold, hypothesis test, seed, or analysis.

| # | claim in `body_iclr.tex` | value | artifact | status |
|---|---|---|---|---|
| 93 | P0 arm mean cov-AUC changes and common post-SFT coverage | plug-in −.0043; count law +.0024; post-SFT .280 | `curriculum_maxrl/group_law_flip/GROUP_LAW_FLIP_ANALYSIS.json` `cells/*/{plugin,grouplaw}/cov_auc_delta` and `post_sft_cov8`, averaged over the frozen 48 blocks | TRACED-ROUNDED |
| 94 | frozen descriptive per-level Spearman correlation | ρ=.157 | same, `descriptive_secondary/spearman_gap_vs_coverage_difference` (.1568682839); status `descriptive_only` | TRACED-ROUNDED |
| 95 | levels 2–4 activity gaps and coverage contrasts; level-5 reversal | gaps [.239,.600,.966], contrasts [+.041,+.035,+.017]; level 5 gap .864, contrast −.023 | same, `descriptive_secondary/per_level/{2,3,4,5}` | TRACED-ROUNDED |
| 96 | levels 8–12 near-zero contrast; levels 10–12 exact zeros | max absolute contrast .0001303 over 8–12; levels 10–12 0 exactly | same, `descriptive_secondary/per_level/{8,9,10,11,12}/mean_cov_auc_difference_grouplaw_minus_plugin` | TRACED-ROUNDED |

The secondary is Tier 2′ descriptive. Its weak rank association is consistent
with, not evidence of, mediation; no p-value or decision rule is attached.

**Current total through the 2026-08-26c addendum: 96/96 traced; 0 untraced.**
