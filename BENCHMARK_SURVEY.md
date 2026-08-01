# Benchmark survey for the next RLVR experiment (2026-08-01)

Question: which benchmark best tests whether recycling-induced sharpening
and the derived gate generalize beyond Countdown, and whether the frontier
teacher pays where pools have unlearnable-at-budget tiers? Constraints:
one A10G, 360M–1.7B models, verl, exact verifier, and — the differentiator
— an **exact relabel map** for hindsight recycling.

## The structural criterion

A relabel map exists only for **constructive-artifact tasks**: the answer
is an object (expression, move sequence, assignment, coloring) from which
an achieved goal g(artifact) is computable and exactly verifiable. A
failed artifact is then a verified success of task g(artifact). This
disqualifies evaluation-direction pools for the recycling arm — final-
answer math (Big-Math, DeepScaleR, POLARIS, MATH, GSM-Symbolic), SAT
yes/no classification, trace prediction — a wrong number is a success of
nothing. Those pools remain fine for curriculum-only baselines. The
relabel must also keep the whole trajectory coherent under the new goal
(Countdown's equation IS the solution; math CoT with a swapped final
answer is formally exact but semantically incoherent).

## Ranked candidates (full notes from the survey)

1. **Jugs (reasoning-gym `games/jugs`, vendored)** — exact move-sequence
   simulator; failed sequence's final state yields relabels for every jug
   amount ever achieved (prefix truncation → multiple relabels per
   failure); built-in curriculum num_jugs {3,4,5,7} × min-moves
   {5,10,15,20} with plausibly unlearnable top tiers. Countdown-isomorphic
   but *stateful sequential planning* — a real domain jump. Zero
   integration cost (vendored, curriculum + verifier APIs).
2. **Blocksworld (E2H-Reasoner pool, arXiv:2506.06632)** — the canonical
   HER setting in tokens: goals are on(x,y) conjunctions, any valid failed
   prefix reaches a relabelable state. Only pool with a documented 1.5B
   curriculum-RL success (Qwen2.5-1.5B). Strongest reviewer narrative;
   moderate integration effort.
3. **3-SAT assignment production (DIY on SATBench's CNF generator)** —
   produce a satisfying assignment; failed assignment satisfies a clause
   subset → relabel to that sub-formula. Cleanest difficulty arithmetic
   (variables × clauses) for placing tiers exactly at/beyond budget;
   ~1 day to build; zero contamination.
4. Graph Coloring (vendored) — ~100% relabel coverage (drop conflict
   edges), relabels flow downward into easier tiers; near-pool caveat.
5. Word Ladder (vendored) — truncate at last valid word; thin tier axis.
6. Quantum Lock (vendored) — Countdown-on-a-state-machine; coarse tiers.
7. CodeIO input-prediction (vendored) — failed input i still produces
   P(i) → relabel output to P(i); code domain; JSON output risky at 360M.
8. Knights & Knaves (DUMP's pool) — great curriculum-only arm, weak
   relabel map (consistent-subset search, underdetermined subpuzzles).

## Literature placement

SEC (2505.14970) is the closest precedent — reasoning-gym pools at 4
procedural levels, Qwen2.5-3B; we would be the first below 3B. E2H used
Blocksworld/Countdown at 1.5B with tiered difficulty. DUMP used K&K at
3B+. AdaRFT used DeepScaleR + solve-rate difficulty at 1.5B (math-only,
no relabel). TinyZero documents Qwen2.5-0.5B *failing* Countdown — the
360M end of any new pool needs genuinely easy tiers. External solve-rate
difficulty columns (Big-Math, DeepScaleR_Difficulty, SYNTHETIC-2) are all
7–8B-calibrated; recalibrate before use.

## Decision

**Run Jugs first** (vendored, days not weeks): E-LLM-3 = Jugs 2×2
(recycling × gate) at 3–4 tier grid, pre-registered predictions:
(P-J1) sharpening replicates — ungated recycling lifts mean@k and loses
pass@k on tiers whose relabel destinations saturate; (P-J2) the utility
gate restores coverage at ≥50% mean retention (same gate_max_p, no
per-domain tuning — the transparency claim); (P-J3) the frontier teacher
pays iff the top tier is unlearnable-at-budget for the base model
(regime rule). Graph Coloring is the fallback if Jugs' low tiers are too
easy; Blocksworld is the scale-up if Jugs confirms.

Model note: SmolLM2-360M-Instruct may sit below Jugs' floor (TinyZero
0.5B-fails-Countdown precedent); check base pass@1 per tier first — if
tier 1 < 1%, either SFT-prime the format (as Countdown did) or move to
Qwen2.5-0.5B/1.5B-Instruct, whichever fits the A10G at N=16.
