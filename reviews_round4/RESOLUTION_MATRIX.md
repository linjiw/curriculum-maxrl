# Review round 4 — resolution audit (2026-08-05)

Legend: FIXED (text, committed) / QUEUED (GPU, pre-registered) /
RUNNING / DEFERRED (with reason) / DISPUTED (with counter-evidence).

## R1 (theory, 6)
| Item | Status |
|---|---|
| Partition decorative; foreground tail ratios | PARTIAL — hypothesis reframe done (6.4); intro reframe deferred to the post-verdict restructure |
| Mass->coverage bridge = hypothesis | FIXED (6.4 restated w/ sign-structure sketch) |
| Dr.-GRPO/no-std ablation | RESOLVED: P-G0c FAILED at neural scale (no-SD loses easy band 6/6, contradicting exact rung) — mechanism claim bounded in 6.3b/6.4 |
| Prop 3 orphaned | DEFERRED to restructure — candidate: move to App with the oracle tie as its only use |
| Lemma 1 magnitude | FIXED (shortfall table in interp: 5.8%/1.1%/12%) |
| Step-matched uniform-only GRPO arm (Q7) | PARTIAL — existing step-matched analysis covers teacher arms; uniform GRPO pass@8 loss at matched steps still open (needs run) |
| Writing density | DEFERRED to post-verdict restructure |

## R2 (practitioner, 4)
| Item | Status |
|---|---|
| W1 scale | DISPUTED-in-part: cannot run 7B on one A10G; honesty paragraphs now scope every LLM claim. Remaining exposure acknowledged |
| W2 abstract 2/2 seeds | FIXED (abstract leads with registered outcome; body separates -.007-within-noise from -.026-at-1.5x-floor) |
| W3 posterior starvation generic | FIXED (deployment-limits paragraph names bucket/predictor/warm-start paths; rho=-0.17 kills cold-start) |
| W4 throughput non-transfer | FIXED; ARM B (dose-matched replay) 2/3 seeds: replay exceeds recycling on both axes — 6.8 carries it; s3 pending |
| W5 recycling under-specified | FIXED (App B wiring contract: placement, rewrite scope, group semantics, KL) |
| W6 gate designed-point n=1 | RESOLVED 2026-08-06: ARM A ran (3 seeds); P-R1 REFUTED — designed point = recycling-off; dial claim dropped, Fig 7a redrawn as scatter |
| W7 GRPO-native teacher | RESOLVED: P-G0a confirmed 5/6 (grpo_mass does not rescue GRPO, both rungs) — in 6.3b controls |
| W8 pool-conditional | Already in paper (Jugs paragraph) — R2 counts it as honest scope erosion; stands |
| W9 estimator set thin | PARTIAL: +grpo_nostd queued; RLOO/PPO-critic arms not planned (stated scope) |
| W10 harness 3x | QUEUED (ARM C w/ decision tree) |

## R3 (methodology, 5)
| Item | Status |
|---|---|
| Floor p-values | FIXED (stated as floors, one-sided, zero-exceptions framing) |
| Permutation null validity | FIXED WITH DATA (stratified p=.025 still floor-perfect; all shared-warmstart pairs positive; artifact committed) |
| Run accounting | FIXED (permutation_reanalysis.json tabulates 5v4 and why) |
| Prereg unverifiable | PARTIAL: commit hashes to be embedded in camera-ready; can't retro-timestamp. DISPUTED-in-part: repo history IS append-only evidence, but R3's standard (third-party) is correct for confirmatory language |
| Single-seed anchors | RESOLVED: ARM A showed the 1-seed full-strength point does NOT replicate (inside 3-seed spread); removed as an anchor |
| Outcome switching | FIXED (abstract) |
| Noise-floor inconsistency (Countdown +.046) | FIXED (requalified as directional) |
| 6/6 tally | FIXED |
| IsaacLab n=1 framing (Q7) | FIXED ('consistent with the null; underpowered alone at one seed') |

## R4 (HER, 5)
| Item | Status |
|---|---|
| Prop 4 definitional | FIXED (-> attributed Remark, marginal-vs-conditional distinction stated) |
| Sharpening rebranding | FIXED (scoped to first-measurement-in-RLVR, lineage named) |
| Gate prior art | FIXED (GoalGAN/HGG/C-HER/Skew-Fit block; novelty = derivation) |
| Placebo accounting / corner case | RESOLVED: ARM B interim (2/3 seeds) shows the replay placebo exceeds recycling at LLM scale on both axes; final on s3 |
| Gate stats | RESOLVED: 3-seed designed-strength stats in armA_b3fix_s{1,2,3}.json + appendix 6.9 secondary reads |
| Exactness instrument | OPEN — honest sentence exists; no new instrument proposed (acknowledge in rebuttal) |
| Granularity (Q2) | FIXED (Alg 1 line 5) |
| Gate cold-start (Q3) | FIXED (Alg 1 line: Beta(1,1) prior p_hat=.5, admitted at default threshold) |

## R5 (AC, 6)
| Item | Status |
|---|---|
| Abstract inversion | FIXED |
| GRPO-own-teacher | RESOLVED (same as W7: P-G0a confirmed) |
| Theory-scope vs method tension (flat-over-band arm) | FIXED WITH DATA (part J: indicator-over-band loses 0/10, .681 vs .743 — zeros alone don't suffice; Remark updated with u_N's refined standing) |
| Compression damage / 6.7 | DEFERRED to post-verdict restructure |
| n=1 links | RESOLVED: ARM A 3-seed (n=1 point superseded/faded in figs) |
| 'Closed-form curriculum' oversold | RESOLVED 2026-08-06: contribution 1 retitled 'closed-form learnability functional' + explicit caveat that the sampler still estimates p and keeps knobs |

## Verdict on reviewer validity
All five reviews were substantively valid. Two pushbacks recorded:
(1) R2-W1: 7B evidence is out of scope for one A10G; the fix is claim
scoping (done), not hardware we don't have. (2) R3's prereg standard:
correct for confirmatory language; repo-internal timestamps are still
evidence, and camera-ready will embed commit hashes.
