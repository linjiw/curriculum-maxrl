# Reviewer 1: RL theory — Rating 6/10, Confidence 4/5

## Verified: all identities correct by hand (Prop 1-3, Lemma 1, GRPO tails).

## Sharpest catches
1. PARTITION IS DECORATIVE, not load-bearing: u_N > 0 on (0,1); "zones"
   are asymptotic regions; every estimator shares the endpoint zeros —
   all estimator-specific content lives in Prop 2's TAIL RATIOS. "Only
   channel into the dead zone" is near-definitional.
2. MASS->COVERAGE BRIDGE IS HEURISTIC: mass is sign-blind; no derivation
   that GRPO's mastered-tail mass SHARPENS rather than "spends." Should
   be stated as an explicit hypothesis. (R1-Q2: expected entropy change
   per update as p->1 under each estimator would discriminate.)
3. MISSING DECISIVE ABLATION (R1-Q3): Dr.-GRPO / no-std variant has mass
   ~ RLOO's => theory predicts it should NOT liquidate easy-band
   coverage. Sharpest falsifiable consequence of the whole framework;
   untested. Also disambiguates success-conditioning vs variance
   normalization as "the estimator."
4. Prop 3 orphaned (matches R5-Q6); 6.6's shared-parameters finding
   undercuts the separable accounting behind it.
5. Lemma 1 magnitude never quantified at deployed N (R1-Q5).
6. Step-matched uniform-only GRPO arm (R1-Q7) — isolate estimator from
   throughput confound.
7. Writing: compressed/aphoristic; 6.3 nearly unreadable without appendix.

## Score movement: UP with Dr.-GRPO ablation + multi-seed LLM cell;
   DOWN if queued cell fails to replicate the interaction.
