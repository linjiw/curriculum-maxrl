# Critical Review of the Latest Draft

**Received:** 2026-08-20
**Status:** advisory review; not an evidence source and not a preregistration amendment
**Saved content:** the review text below is preserved as received, with only
Markdown line wrapping and heading normalization. The
repository contract and frozen P0 preregistration govern where its proposed
wording differs from the registered claim perimeter.

The draft represents an exceptionally rigorous, mathematically grounded piece
of scholarship. By moving away from over-claiming and centering the narrative
on the core theoretical identity
($A_N(Q) = 2(\operatorname{Pr}(K>0) - \mathbb{E}[K]/N)$) and its breakdown
under coarse pooling (Corollary 2), the paper has achieved a rare level of
scientific integrity.

```text
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ Theoretical Core (Exact)                                                                  │
│  - Estimator defines group coefficient mass: A_N(Q) = 2(Pr(K>0) - E[K]/N)[cite: 2]       │
│  - Under i.i.d.: u_N(p) = p(1-p) · w_{N-1}(p) = (1-p)(1 - (1-p)^{N-1})[cite: 2]          │
└─────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                              │
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ The Commutation Breakdown (Corollary 2 & MAZE-SCORE)                                      │
│  - A_N(p̄_z) - 2 E_X[u_N(p_X)] = 2 [Pr(K=0|z) - (1-p̄_z)^N] ≥ 0[cite: 2]                  │
│  - Estimator map and curriculum averaging DO NOT commute[cite: 2]                        │
└─────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
┌──────────────────────────────────────────┐     ┌──────────────────────────────────────────┐
│ Confirmed Positives (Task = i.i.d. Unit) │     │ Measured Boundaries & Refutations        │
│  - Acrobot V2 (+.0480, replicated)       │     │  - Peak location specificity rejected    │
│  - Digits (+.208, bandit)                │     │  - AMaze: Starved Bernoulli replaces TD  │
│  - CPU Skill Chain (compounding steps)   │     │  - Aggregated Unit: Sign reverses (-.003)│
└──────────────────────────────────────────┘     └──────────────────────────────────────────┘
```

## Major Strengths

* **Uncompromising Epistemic Rigor:** Preregistering confirmatory hypotheses
  and keeping negative results/refutations (peak location sweep, AMaze
  replacement, coarse unit inversion) as first-class citizens elevates the
  work far above standard heuristic benchmark chasing.

* **Exact Mathematical Anchoring:** Deriving that standard $p(1-p)$
  learnability is strictly the $N=2$ (REINFORCE/RLOO) atomic slice and proving
  the $T = N-1$ practical truncation order eliminates ambiguity surrounding
  empirical curriculum curves.

* **Explanatory Power of Corollary 2:** Tracing the $-.0032$ reversal in
  MAZE-SCORE directly to Jensen’s gap and excess silent groups
  ($\operatorname{Pr}(K=0\vert{}z) - (1-\bar{p}_z)^N$) precisely explains why
  group-aware curricula collapse when evaluated on heterogeneous task buckets.

## Vulnerabilities & Reviewer Attack Vectors

* **The "Resolution vs. Scale" Confound:** Although Section 3 argues that
  granularity rather than neural scale explains the divergence across studies,
  MAZE-SCORE altered model capacity, task diversity, and curriculum granularity
  simultaneously. Until the blind-running $P_0$ Hopper experiment finishes,
  this attribution remains post hoc.

* **Auditing Provenance Bottleneck:** Section 5 and Appendix D openly disclose
  that certain replay payloads (Digits 5.08 GB, paid-probe raw 1.37 GB) and
  checkpoint trajectories remain external. While transparent, top-tier
  reviewers may scrutinize the lack of immutable public pre-execution commits
  for the tournament locks.

* **Limited LLM Confirmation:** The main text relies heavily on Acrobot and
  Digits; LLM results (GSM8K and Countdown) are confined to App. C and
  characterized by small effect sizes or treatment delivery issues.

---

## Step-by-Step Research Guidance: Next Milestones

```text
                     ┌─────────────────────────────────────────────────────────┐
                     │ WEEK 1 (Immediate): P0 Hopper Unblinding & Ingestion    │
                     │  - Reconcile 48-block frozen trial                      │
                     │  - Verify if count-law functional fixes plug-in loss    │
                     └────────────────────────────┬────────────────────────────┘
                                                  │
                                                  ▼
                     ┌─────────────────────────────────────────────────────────┐
                     │ WEEK 2: Artifact Hardening & Open Data Auditing         │
                     │  - Mint Zenodo/OSF DOI with immutable pre-commit hashes │
                     │  - Reconstruct missing manifests / reproducible scripts │
                     └────────────────────────────┬────────────────────────────┘
                                                  │
                                                  ▼
                     ┌─────────────────────────────────────────────────────────┐
                     │ WEEK 3: Section 3.4 & Practical Playbook Polish         │
                     │  - Integrate P0 Hopper outcome (Pass or Fail branch)    │
                     │  - Provide concrete pseudo-code for q̂_z - p̄_z estimator│
                     └─────────────────────────────────────────────────────────┘
```

### 1. Ingesting the $P_0$ Count-Law Hopper Intervention

The 48-block Hopper campaign ($N=32$, seeds 3001–3048) is the single most
decisive piece of evidence needed before submission:

* **Protocol Check:** The trial holds the substrate, policy architecture,
  budget, and 4-moment count-law posterior fixed while testing the plug-in
  $u_N(\bar{p}_z)$ against the realized count-law activity
  $\hat{q}_z - \bar{p}_z$.

* **Branch A (P0 Supported):** If realized count-law scoring reverses the
  $-.0032$ deficit and outperforms uniform/$p(1-p)$, promote this result
  directly into Section 3.4 as the experimental validation of Corollary 2.

* **Branch B (P0 Null/Negative):** If the deficit persists, state clearly that
  while Corollary 2 accounts for coefficient mass underprediction, downstream
  optimization dynamics require explicit task-level separation rather than
  posterior-level count-law averaging.

### 2. Resolving Open Artifact and Registration Gaps (Appendix D)

To preempt reviewer pushback on provenance:

* **Publish Local Hash Bundles to Public Ledgers:** Upload the SHA-256 manifest
  (`manifest.json`), the Acrobot tournament lock, and the Digits replay
  descriptors to an immutable public repository (e.g., OSF or Zenodo) to
  establish an externally verifiable timestamp before the September deadline.

* **Audit Manifest Completeness:** For Countdown and GSM8K, regenerate
  task-level binary outcome matrices so that unbiased
  $\operatorname{pass}@k$ curves can be verified without relying on bootstrap
  proxies.

### 3. Formalizing the Practical Deployment Rule

In Section 3 (Takeaways), provide an explicit recipe for practitioners
deploying GRPO/MaxRL on mixed datasets:

* **The "Zero-Cost" Estimator Rule:** When tasks cannot be uniquely tracked
  (e.g., continuous domains or broad topic buckets), log the empirical
  non-all-fail fraction
  $\hat{q}_z = \frac{1}{B}\sum_{b=1}^B \mathbf{1}\{K_b > 0\}$ and mean pass
  rate $\bar{p}_z = \frac{1}{B N}\sum_{b=1}^B K_b$.

* **Sampling Criterion:** Score batches using:

  $$\text{Priority}(z) = \left( \hat{q}_z - \bar{p}_z \right)_+^\gamma + \epsilon_{\text{floor}}$$

  This directly computes realized coefficient mass without suffering from
  Jensen's overestimation penalty.

---

## Operational reconciliation with the frozen record

The following constraints govern implementation; they annotate but do not
alter the received review above.

1. P0 remains Tier 4 and blind until all 48 paired blocks are terminal, the
   complete 2×48 matrix and hashes pass, the canonical campaign is retrieved,
   and the frozen analyzer is invoked exactly once.
2. P0 contains only `plugin` and `grouplaw` arms. It cannot establish
   superiority to uniform or $p(1-p)$, because neither is present.
3. A supported P0 result establishes causal relevance of the count-law
   correction on this substrate. Corollary 2 predicts coefficient-activity
   calibration, not the downstream learning sign, and therefore must not be
   described as experimentally validated by an endpoint improvement.
4. A nonsupported result must use the frozen branch: `practically_ruled_out`,
   `inconclusive`, or `treatment_not_delivered`. It cannot be diagnosed as
   requiring task-level separation unless separately supported.
5. The aggregation-gap sign is restricted to the stated mixture of
   conditionally-i.i.d. atomic tasks. It is not asserted for arbitrary count
   laws.
6. Historical Countdown numbers remain VERL bootstrap best@k proxies unless
   raw per-task binary outcomes are actually retained; they must not be
   relabeled as standard pass@k.
