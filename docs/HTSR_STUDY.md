# HTSR Lineage And Performance Study

Protocol: `htsr-combined-study-v4`

Retained run: `research_runs/htsr_study/runs/htsr_7f739673cde388be2d4c`

Local paper snapshot: `archive/reference-work/Spectral Signatures of Large Language Models.html`

## Scope

The workflow answers two separate questions from the same immutable artifacts:

1. Can spectral structure route a model to a known root/backbone family and expose
   unknown-family uncertainty?
2. For a held-out ancestry component, do HTSR summaries improve benchmark
   prediction beyond structural and non-HTSR spectral features?

It does not infer causal ancestry or claim exact parent reconstruction. All
target-dependent transforms and Ridge tuning occur inside ancestry-grouped folds.
The external ablation collection is descriptive because it informed prior method
development. Those superseded study outputs are retained only under
`archive/research_runs/` as provenance for that evidence-role decision.

## Feature Blocks

- `M`: matrix dimensions, layer count, canonical-role coverage, and parameter scale
- `S`: non-HTSR log RMS weight scale, log spectral norm, normalized entropy, and
  stable-rank-fraction trajectories
- `H`: stored DKS alpha, alpha-weighted norm, and `log_alpha_norm` trajectories

Each quantity is summarized by canonical transformer role using median, IQR,
normalized-depth slope, and missingness. The seven roles are Q, K, V, attention
output, MLP gate, up, and down projections. Unsupported/fused schemas are reported
rather than silently coerced.

The stored alpha values use WeightWatcher-style DKS power-law fitting. They are
valid HTSR features, but they are not the paper's fixed-tail Hill estimator; the
saved artifacts do not contain enough eigenspectrum information to reconstruct
that estimator.

## Evaluation

Lineage uses a predeclared method ladder: configuration, non-HTSR spectral shape,
paper-style depth-aligned DKS-alpha similarity, and their combination. It reports
root-gallery retrieval, leave-one-family-out rejection, cross-fitted observed-family
routing, label-independent clustering, and an explicitly secondary exact-parent
diagnostic.

Performance uses nested ancestry-grouped Ridge evaluation on identical populations.
The primary contrast is

`Delta MAE = MAE(M+S+H) - MAE(M+S)`,

with component-weighted MAE, a component bootstrap interval, and a paired sign-flip
test. Negative values favor adding `H`. A sensitivity analysis repeats a 70/30
ancestry-component split 20 times on the Phase-2 population, tuning only within
each training partition and comparing both feature blocks on identical test rows.
Repeated observations are averaged within ancestry component before bootstrapping.

## Current Evidence

- Performance, 302 models in 79 ancestry components: `M` MAE 3.823, `M+S` 3.597,
  and `M+S+H` 3.468.
- The HTSR increment was -0.129 MAE (3.59% relative improvement), 95% CI
  `[-0.365, 0.092]`, sign-flip `p=0.278`. The point estimate favors `H`, but the
  frozen improvement and interval criteria did not pass.
- Across 20 ancestry-disjoint 70/30 Phase-2 holdouts, the increment shrank to
  -0.016 MAE, 95% component-bootstrap CI `[-0.236, 0.209]`; `H` was better in
  only 9/20 splits. This is internal robustness evidence, not an external test.
- Observed-family routing covered 174 queries in 11 audited families.
  `spectral_shape` and `combined` both reached 0.96884 macro top-1; alpha did not
  add a measurable top-1 gain there.
- Label-independent clustering covered 7,534 models with `K=62`: ARI 0.850, NMI
  0.905, and Hungarian-matched macro family recall 0.847 on the labeled subset.
- Exact-parent top-1 was 0.065. The signatures support family routing, not direct
  parent recovery.
- Canonical-root top-1 was 1.000 on only 31 queries and 3 families, so it remains
  descriptive.

## Decision

The retained v4 result remains the frozen confirmatory record. The scaled study at
`research_runs/htsr_scaleup/runs/scaleup_f500d13ec545609d70b5` adds exploratory
evidence over the larger available populations:

- Direct component-distinct alpha-profile kNN reached MAE 3.082 versus 3.455 for
  `M+S+H` on an identical 649-model, 139-component modern population. The paired
  improvement was 0.373 MAE, 95% CI `[0.037, 0.727]`, `p=0.037`.
- On 348 ancestry-supported legacy models, direct alpha-profile kNN improved the
  six-task average by 1.328 MAE, 95% CI `[0.554, 2.104]`, `p=0.0013`. Five of six
  individual tasks survived Holm correction.
- Nested residual correction did not reliably improve `M+S+H` on either target
  generation. Alpha profiles work better here as a direct local predictor.
- The 15,000-model Phase-2 ledger contains 8,892 HTSR artifacts. Keep the 7,534
  strict-QC lineage result as the default. The 8,284-model broad clustering is a
  coverage sensitivity and its family routes are manual-triage candidates only.

Use direct alpha-profile kNN as the leading scaled performance method, while
retaining `M+S+H` as the parametric baseline. Treat the scaled gains as internal
grouped evidence until a genuinely prospective model collection is available.
For lineage, keep `spectral_shape` routing and strict clustering as the primary
views; do not convert broad clusters or inverse-distance scores into ancestry
ground truth.
