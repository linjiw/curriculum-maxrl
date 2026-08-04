# Round-2 audit: REPORT.md and GUIDE.md

Audit date: 2026-07-22/23. Method: every quantitative/checkable claim was tested
against primary artifacts — maze JSONL logs re-analyzed with
`maze_gpu/analyze.py` plus direct log parsing, `efficiency.json` /
`efficiency_p1.json` k* recomputation, `v7_oracle_result.json`,
`results_speed.json` / `results_fixed_n.json`, source defaults in
`verl/utils/curriculum.py`, `frontier_rl/teacher.py`, `maze_gpu/train.py`, and
fresh closed-form / Monte-Carlo checks of every math identity (numpy, CPU,
niced). CPU-doc-only numbers with no persisted artifact (per the EVIDENCE.md
round-1 audit) are marked "consistent-with-docs" rather than re-run.

Verdicts: CONFIRMED (matches primary evidence), STALE (evidence exists, value
or status differs), WRONG (contradicted), UNVERIFIABLE (no primary artifact).

---

## REPORT.md verdict table

| claim | location | verdict | evidence | suggested fix |
|---|---|---|---|---|
| P1 identity E[Σ\|w\|] = 2(pass@N − pass@1), exact | §2 table row P1 | CONFIRMED | Fresh 200k-trial MC matches to 3 decimals at p ∈ {0.005…0.95}, N=32; matches PROOFS.md P1 | — |
| Dead groups 5.8→3.4/8 (teacher vs uniform) | §2 row P1; F1 | CONFIRMED (values) / needs qualifier | analyze.py on matched logs: uniform 5.79, frontier_alp 3.42. BUT verified in train.py that the dead counter is `not np.any(w != 0)`, which under MaxRL pools K=0 with K=N groups — same mechanism-open caveat the EVIDENCE.md audit added | Keep numbers; add the "historical zero-weight-group counter, pools K=0 with K=N, cannot isolate dead-group waste" qualifier used in EVIDENCE.md |
| V2: AUC 0.700 vs 0.688 (ties hand-tuned ZPD) | §2 row P1 | CONFIRMED (mixed sources) | 0.688 = zpd+maxrl 0.6881 in `results_speed.json`; 0.700 = V2 Thompson (decay 0.9) in VALIDATION.md (doc-only, no JSON) | Optionally cite v7_oracle_result.json teacher_thompson 0.728 (decay 0.7) as the current-default number |
| P2 peak p* ≈ ln N/N, strictly concave | §2 row P2 | CONFIRMED | Closed form p* = 1 − N^(−1/(N−1)) verified: N=16→0.169 (lnN/N 0.173), N=32→0.106, N=128→0.0375; concavity per PROOFS P2 | — |
| CPU: posterior tracks true p within ±0.03 | §2 row P2 | UNVERIFIABLE | No artifact and no matching statement found in VALIDATION.md/DESIGN.md; not persisted | Cite a source or soften to "posterior tracks true p closely (V2)" |
| GPU: teacher p̂ matches eval per level; 60% of mass on true frontier | §2 row P2 | STALE (minor) | sweep_frontier_maxrl_s0.jsonl: final p̂ within ≤0.09 per level of eval (good match); mass on levels 2–5 is ~0.70 mean (0.32 first record, 0.70–0.80 thereafter), not 60% | "~70% of sampling mass on levels 2–5" (understatement as written; benign) |
| P3 greedy water-filling: +18% mass vs uniform split | §2 row P3 | UNVERIFIABLE (plausible) | No persisted artifact. Fresh re-derivation on mixed-difficulty pools gives +15–25% depending on pool, so the number is plausible | Mark as unpersisted or persist the experiment |
| P4: RLOO mass = 2p(1−p) ≡ SFL learnability, exact MC match | §2 row P4 | CONFIRMED | Fresh MC: N=8,p=0.3 → 0.41994 (2p(1−p)=0.42000); N=32,p=0.01 → 0.01977 vs 0.01980 | — |
| P5: MaxRL mass ≈ N× RLOO's as p→0 | §2 row P5 | STALE | Exact limit ratio is N−1, not N (fresh MC at N=32: ratio → 30.995 = N−1; PROOFS.md P5 itself says "→ N−1 (≈ N)") | Write "≈ (N−1)×" or "→ N−1 (≈N)" |
| P6 V1: per-group cosine 0.956 vs 0.958, mean cosine 1.000 | §2 row P6 | UNVERIFIABLE (consistent-with-docs) | Matches VALIDATION.md V1 table exactly; no persisted JSON | — |
| Dense hindsight GPU champion: final 0.258, best 0.269 | §2 row P6; F5 | CONFIRMED | matched_falp_maxrl_hsdense_s0.jsonl via analyze.py: 0.258 / 0.269 | — |
| P7 defaults decay 0.7, floor 0.1 shipped in verl module | §2 row P7 | CONFIRMED | verl/utils/curriculum.py: decay=0.7, floor=0.1; frontier_rl/teacher.py same | — |
| V2b decay 0.7 closes ~19% of oracle gap; V3 floor flat 0–0.4 | §2 row P7 | UNVERIFIABLE (consistent-with-docs) | Matches VALIDATION.md V2b/V3 ((0.728−0.700)/(0.851−0.700)=18.5%); no artifact | — |
| V6: γ=4 AUC 0.782 vs 0.728 | §2 row V6; F4 | CONFIRMED (0.728 by artifact) | 0.728 = teacher_thompson auc_mean 0.7278 in v7_oracle_result.json; 0.782 doc-only (VALIDATION V6) | — |
| γ power knob "(not yet run on GPU)" | §2 row V6; §5 item 2 | STALE (major) | matched_falp_p4_hsdense_s0.jsonl exists (2026-07-22): AUC 0.231 / best 0.254 vs γ=1's 0.236/0.269 — γ=4 did NOT transfer (EXPERIMENTS.md F2 verdict, as the V6b ODE model pre-registered) | Replace with the F2 result: γ=4 run on GPU, negative as pre-registered; γ stays 1 as GPU default |
| Adaptive-T hurts: AUC 0.698 vs 0.704 | §2 broke-chain item 1 | UNVERIFIABLE (consistent-with-docs) | Matches THEORY.md §4; no artifact | — |
| Hindsight→teacher feedback: p̂ 0.81 vs true 0.47 at level 2, worse final | §2 broke-chain item 2 | CONFIRMED | matched_falp_maxrl_hsdense_tt_s0.jsonl: step 575 p̂[2]=0.81, eval[2]=0.47; final 0.242 < 0.258 (B) | — |
| F1: AUC 0.221–0.223 vs 0.211–0.216; 6/6 paired deltas positive | F1 | CONFIRMED (one endpoint imprecise) | Recomputed 3-seed means: falp 0.221, frontier+hs 0.223 vs uniform maxrl 0.211, uniform grpo 0.213. Per-seed deltas +0.019/+0.004/+0.006 and +0.020/+0.002/+0.013 — 6/6 positive. The "0.216" is uniform+grpo seed-0, not its multi-seed mean (0.213) | Use 0.211–0.213 for the multi-seed uniform range |
| F1: steps/sec rises ~30% | F1 | CONFIRMED | Matched logs: 583 steps (uniform) vs 713–787 (teachers) = +22–35% | — |
| F2 (H6 reversed): frontier+GRPO pass@8 0.332→0.269 vs uniform+GRPO 0.351→0.312; MaxRL grew 0.316→0.348 every seed | F2 | CONFIRMED (values) / needs single-seed qualifier | Logs: frontier_grpo_s0 0.332→0.269 ✓; uniform_grpo_s0 0.351→0.312 ✓; frontier+hs 3-seed mean 0.316→0.348 ✓, grew in each seed ✓; GRPO decayed in every seed (0.351→0.312, 0.308→0.240, 0.264→0.260). BUT frontier+grpo exists only at seed 0 — the teacher-AMPLIFIED collapse is a single-seed arm (EVIDENCE.md finding b) | Add "(teacher-amplified collapse observed in the single frontier+grpo seed; GRPO's decay itself is every-seed)" |
| F2: GRPO lost easy retention 0.62 vs 0.75 | F2 (implicit), GUIDE M6 | CONFIRMED | Recomputed min easy pass over last half: frontier_grpo 0.62, uniform_grpo 0.75 | — |
| F3: oracle CPU AUC 0.851; Thompson+γ=4+hindsight 0.890 | F3 | CONFIRMED | v7_oracle_result.json: oracle_gamma1 0.8511 ± 0.002, full_stack_gamma4_hs 0.8895 ± 0.002 (also beats oracle_gamma4 0.8836) | — |
| F3 categorical: frontier-heavy (max p=10⁻⁵) uniform/DAPO/teacher exactly 0, teacher+hindsight 0.98; ignites within ~400 groups | F3 | UNVERIFIABLE (consistent-with-docs) | Matches VALIDATION.md V5 (AUC 0.928, final 0.981; 257 live groups by draw 400); run_baselines.py does not persist JSON | Persist V5 output like V7 was persisted |
| F4: γ=4 0.782 vs 0.728, beats hard top-k | F4 | CONFIRMED (0.728) / consistent-with-docs (0.782, 0.771) | See V6 row above | — |
| F5 leaderboard (all 5 rows: final/best/AUC/pass@8) | F5 table | CONFIRMED | analyze.py reproduces every cell: 0.258/0.269/0.236/0.361; 0.244/0.257/0.233/0.361; 0.230/0.256/0.234/0.356; 0.225/0.233/0.214/—; 0.230/0.237/0.216/0.312 | — |
| Goal 3: +0.02–0.03 final / +0.02 AUC over uniform+MaxRL, consistent across seeds | §4 achieved-3 | STALE (minor) | Multi-seed: final +0.016 (falp) to +0.022 (champion); AUC +0.010–0.012 mean (+0.02 is seed-0 only). Report's own §6 acknowledges ~5–10% margin | "+0.016–0.022 final / +0.010–0.012 AUC (multi-seed means; ~+0.02 AUC at seed 0)" |
| "Level 6 is a duration question… that claim itself needs a long-run to verify" | §4 not-achieved-1; §5 item 1 | STALE (major) | The long run EXISTS and answered it: long_falp_hsdense_s0.jsonl (9600 s, 2381 steps, 2026-07-22): mean 0.269, level 5 doubles, level 6 stays 0.01–0.02 → EXPERIMENTS.md F1 verdict: NOT (just) a duration question; depth-mechanism study diagnoses a per-step-legality (q≈0.87) ceiling, and coverage@64 shows L6 0.125→0.438 (wide) | Rewrite: long-run complete, duration hypothesis refuted, stall diagnosed as per-step-accuracy ceiling; coverage currency shows the frontier moving (0.188→0.312→0.438 @k=64) |
| Efficiency study "is running now" | §4 not-achieved-3; §5 item 3 | STALE | efficiency.json (2026-07-22) complete: ours vs GRPO k* speedups 1.2× (L2), 2.7× (L3), 0.5× (L4 reversal), 11× (L5) — all recomputed from the JSON and matching EXPERIMENTS.md E4 | Replace with the completed E4 table (including the honest L4 0.5× reversal) |
| Champion margin "is a single-seed number" (§5 item 4); "Maze GPU results are seed-0 except the confirmed multi-seed round" | §5 item 4; §6 | STALE | Champion seeds 1–2 exist (matched_falp_maxrl_hsdense_s1/s2.jsonl, 2026-07-22): final 0.252±0.005, AUC 0.229±0.009, 6/6 paired deltas positive vs frontier_alp (recomputed: ΔAUC +0.003/+0.006/+0.014, Δfinal +0.014/+0.002/+0.001) | Update to the F3/F4 multi-seed verdict: champion edge survives on AUC every seed; final margin mostly one seed |
| 1.26M-param transformer, 17×17 mazes | §2 intro | CONFIRMED | TinyTransformer() = 1,258,496 params (counted); EXPERIMENTS.md header | — |
| "Both CPU pre-registrations that made GPU predictions were confirmed" | §2 | STALE (now understated) | V4-feedback prediction confirmed (C≤B ✓); the γ/ODE prediction was ALSO confirmed by the (post-REPORT) γ=4 GPU run — EXPERIMENTS.md: "third CPU→GPU transfer test; the ODE model correctly predicted which one" fails | Update count: three CPU→GPU pre-registrations tested, all resolved as predicted |

### REPORT.md health summary

The experimental numbers in REPORT.md are in excellent shape: every cell of the
F5 leaderboard, the H6 reversal numbers, the multi-seed 6/6 paired deltas, the
teacher-feedback inflation signature, and the V7 oracle-beating result
reproduce exactly from the JSONL logs and v7_oracle_result.json, and all core
math identities (P1, P2, P4, marginal-mass, GRPO finite-sample throttling)
pass fresh MC verification. The document's problem is staleness of *status*,
not correctness of *data*: four load-bearing "open items" in §4–§5 (long-run
duration test, γ=4 GPU test, efficiency study, champion multi-seed) have all
since completed — two of them resolving *against* the stated expectation
(level-6 is not a duration question; γ=4 does not transfer) — so §4's
"not yet achieved" list and §5's ranked next-push list actively misdirect a
reader. Smaller fixes: P5's "≈N×" should be "→N−1", the dead-group counter
needs the pooled-K=N qualifier, F2 needs the single-seed-arm qualifier on the
amplified-collapse claim, and the CPU "±0.03" tracking figure is uncited.

---

## GUIDE.md verdict table

| claim | location | verdict | evidence | suggested fix |
|---|---|---|---|---|
| MaxRL estimator "unbiased for the T=N-truncated ML objective" | §1 ¶1 | WRONG (per repo's own correction) | PROOFS.md P1 correction (verified 2026-07-23, 4M MC) + fresh MC here: the practical drop-K=0 Algorithm-1 estimator is unbiased for T=N−1 (MC 5.21842 = w_{N−1} 5.21703 ≠ w_N 5.69533 at N=8, p=0.1) | "unbiased for the (N−1)-truncated objective (T=N only if the K=0 control variate is retained; PROOFS.md P1 correction)" |
| M1: mass = 2(pass@N − pass@1), peaks p* ≈ ln(N)/N, MC-verified 200k trials | M1 | CONFIRMED | Fresh MC to 3 decimals; peak closed form verified | — |
| M1: CPU AUC 0.704 vs 0.712 heuristic-frontier vs 0.688 ZPD | M1 | CONFIRMED (0.712, 0.688) / consistent-with-docs (0.704) | results_speed.json: maxrl_frontier+maxrl 0.7115, zpd+maxrl 0.6881; 0.704 advmass is doc-only (THEORY.md) | — |
| M1: teacher p̂ tracks true eval pass rates to ±0.03 (GPU maze) | M1 | WRONG | sweep_frontier_maxrl_s0.jsonl: final per-level \|p̂−eval\| up to 0.09 (0.07/0.09/0.08/0.00/0.03/0.06 on levels 0–5); median per-record max discrepancy ≈0.16 | "to within ~±0.1 per level at convergence" or drop the number |
| M1: concentrates 60% of sampling mass on the true frontier band | M1 | STALE (minor) | Log: mass on levels 2–5 averages ~0.70–0.72 (first record 0.32) | "~70%" (current text understates) |
| M1: dead groups 5.2/8 uniform → 3.9 frontier / 2.6 learnability | M1 | CONFIRMED (+ counter qualifier) | sweep_*.jsonl recomputed: 5.15 / 3.92 / 2.62. Same pooled K=0/K=N counter caveat as REPORT | Add the mechanism-open counter qualifier |
| M1 status: "🔄 GPU sweep running (6 configs × 2400 s)" | M1 header + Open | STALE | Sweep complete; results are in §2b of this same document | Change to ✅ complete, point to §2b |
| M2: +18% total advantage mass vs uniform split | M2 | UNVERIFIABLE (plausible) | No artifact; fresh re-derivation gives +15–25% on mixed pools | Persist or mark unpersisted |
| M3: RLOO mass exactly 2p(1−p) ≡ SFL learnability (up to constant) | M3 | CONFIRMED | Fresh MC 0.41994 vs 0.42000 (N=8, p=0.3); exact for all N tested | — |
| M3: GRPO realized 0.100 vs population 0.199 at p=0.01, N=32; 72% all-fail | M3 | CONFIRMED | Exact binomial sum: 0.1017 vs 2√(p(1−p))=0.1990; P(K=0)=0.725 | — |
| M4: c_{N,N}(K)=1/K recovers Algorithm 1; E[c·K]=1−(1−p)^T to 4 decimals | M4 | CONFIRMED | Standalone numpy reimplementation of eq. 51: c_{N,N}(K)=1/K exact; worst MC-vs-closed-form diff 0.00046 over T∈{1,4,16}, p∈{0.05,0.3,0.7}, N=16 | — |
| M4: adaptive-T underperforms (0.698 vs 0.704; 0.641 vs 0.653) | M4 | UNVERIFIABLE (consistent-with-docs) | Matches THEORY.md §4; no artifact | — |
| M7 CPU table (0.966/0.653, 0.978/0.878, 0.979/0.704, 0.984/0.883) | M7 | CONFIRMED (uniform row) / consistent-with-docs (rest) | uniform+maxrl final 0.9659 (results_fixed_n.json), AUC 0.6528 (results_speed.json); hindsight rows match THEORY.md §5 but have no artifact | — |
| M7: weight-scale monotone 0.805→0.943 at 0.25→8; uniform+hs 0.970 > advmass 0.961; stack 0.978 | M7 ablations | UNVERIFIABLE (consistent-with-docs) | Matches THEORY.md numbers | — |
| M7: default 1.0 = natural K=1 group weight | M7 | CONFIRMED | train.py: `w_hs = scale * (1 − 1/N)` — exactly the K=1 MaxRL success weight | — |
| M7 GPU A/B/C: sparse ties baseline (0.234 vs 0.233); dense champion 0.258/0.269; shallow levels 0.99/0.87/0.68/0.45; 3.6→16 relabels/step; C: p̂ 0.81 vs eval 0.47 (B tracks 0.54 vs 0.64) | M7 | CONFIRMED | analyze.py: falp+hs 0.234 vs falp 0.233 ✓; dense 0.258/0.269 ✓; per-level final [0.99 0.87 0.68 0.45] exact ✓; relabels: sparse 3.7–3.9 mean, dense 16.0 (cap) ✓; tt log step 575: p̂[2]=0.81 eval 0.47 ✓; B final p̂[2]=0.54 eval 0.64–0.68 ✓ | — |
| M7 status: "GPU sweep queued" / maze hindsight "queued behind sweep 1" | M7 header + bias caveat | STALE | The same section already reports the completed GPU A/B/C; logs exist | Change to ✅ complete; remove "queued" |
| M6 (H6): frontier+grpo 0.332→0.269 vs uniform+grpo 0.351→0.312; easy retention 0.62 vs 0.75; MaxRL 0.327→0.351 | M6 | CONFIRMED (+ single-seed qualifier) | All values reproduce from seed-0 logs (frontier_maxrl_s0: 0.327→0.351 ✓). frontier+grpo is single-seed (finding b) | Add single-seed-arm qualifier for the amplified collapse |
| §2b matched-clock table (8 configs: steps/dead/AUC/pass@8) | §2b | CONFIRMED | Every cell reproduces via analyze.py (steps 583/527/751/651/713/787/765/694; dead 5.8/6.0/5.7/5.0/3.9/3.3/3.4/3.8; AUC 0.214–0.234; pass@8 0.312–0.361) | — |
| §2b: frontier_alp level-2 pass 0.62 vs 0.54 | §2b | CONFIRMED | Final eval level 2: frontier_alp 0.62, frontier 0.54 | — |
| §2b: GPU hindsight +0.01 vs CPU +0.22 AUC | §2b | CONFIRMED (GPU) / consistent-with-docs (CPU) | +0.011 (0.234 vs 0.223), +0.008 (0.222 vs 0.214) from logs | — |
| Multi-seed: 6/6 paired deltas (+0.019/+0.004/+0.006, +0.020/+0.002/+0.013); falp final 0.246±0.002 vs 0.230±0.015; GRPO 0.308→0.271 vs teacher+MaxRL 0.316→0.348 | §2b | CONFIRMED | All recomputed exactly from matched_*_s{0,1,2}.jsonl | — |
| §2c V5 regime table (0.946/0.929/0.953/0.975; 0.734/0.825/0.784/0.931; 0/0/0/0.928) | §2c | UNVERIFIABLE (consistent-with-docs) | Matches VALIDATION.md V5 verbatim, incl. DAPO easy-heavy 0.929 (correctly NOT "≥0.93" — finding d already applied); run_baselines.py persists nothing | Persist V5 results JSON |
| §3 H3: 0.871 / 0.847 alone vs 0.961 combined | §3 | UNVERIFIABLE (consistent-with-docs) | Matches DESIGN.md hard-regime table; no artifact for 16-level regime | — |
| §3 H2: teacher lifts GRPO +0.23/+0.42, MaxRL +0.01/+0.09 | §3 | CONFIRMED (moderate regime) / consistent-with-docs (hard) | results_fixed_n.json: grpo 0.7561→0.9888 (+0.233), maxrl 0.9659→0.9772 (+0.011) ✓; +0.42/+0.09 from DESIGN.md doc-only | — |
| §3: teachers run ~2× more steps per wall-clock second | §3 | CONFIRMED | sweep-1 logs: 2970 s vs 1477/1245 s for 300 steps (2.0–2.4×) | — |
| §3: maze-size cliff 0/1024 on 9×9; distance transfers smoothly | §3 | UNVERIFIABLE (consistent-with-docs) | Matches EXPERIMENTS.md pilot notes; pilot logs not retained | — |
| §4 testbed: 1.26M-param transformer | §4 | CONFIRMED | 1,258,496 params counted | — |
| §5 roadmap item 1: "Now: finish matched-clock GPU sweep → test H6/H7" | §5 | STALE | Done (results in §2b of the same doc); H6 refuted, H7 supported | Advance roadmap to current state (GSM8K, per-prompt N) |
| §5 item 2: "Multi-seed (≥3) confirmation of GPU winner configs" | §5 | STALE | Done for falp, frontier+hs, champion (seeds 0–2) | Mark done |
| §6: "Matched GPU results so far are single-seed" | §6 | STALE / self-contradictory | §2b of the same document reports the 3-seed confirmation | Delete or scope to the not-yet-multi-seeded arms (frontier+grpo, learnability, γ=4, wide) |

### GUIDE.md health summary

GUIDE.md's numbers are almost uniformly accurate — every GPU table cell
(§2b matched sweep, multi-seed block, A/B/C hindsight results, H6 numbers)
reproduces exactly from the JSONL logs, the M3/M4 math checks pass fresh MC
verification, and the V5/M7-CPU tables are faithful to VALIDATION/THEORY (though
those CPU artifacts remain unpersisted). Two substantive errors need fixing:
§1 states the practical estimator is unbiased for the T=N-truncated objective,
which the repo's own PROOFS.md correction (and this audit's MC) shows is T=N−1;
and M1's "p̂ tracks eval to ±0.03" overclaims — the logs show per-level final
deviations up to ~0.09 with mid-run excursions near 0.16. The rest is status
rot: three "running/queued" markers and two roadmap/limitations items describe
work that is complete and reported elsewhere in the same document, including
the §6 "single-seed" limitation that §2b itself contradicts. The known
finding-(d) fix (DAPO 0.929) is correctly applied; findings (a) and (c) do not
appear; findings (b) and (e) qualifiers are missing at M6/M1 and should be
copied from EVIDENCE.md.
