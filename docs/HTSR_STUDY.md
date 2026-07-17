# HTSR Lineage And Performance Study

Protocol: `htsr-combined-study-v3`  
Retained run: `research_runs/htsr_study/runs/htsr_b41e5fbcb8abcc600bce`

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
test. Negative values favor adding `H`.

## Current Evidence

- Performance, 302 models in 79 ancestry components: `M` MAE 3.823, `M+S` 3.597,
  and `M+S+H` 3.468.
- The HTSR increment was -0.129 MAE (3.59% point improvement), 95% CI
  `[-0.365, 0.092]`, sign-flip `p=0.278`. The point estimate favors `H`, but the
  frozen improvement and interval criteria did not pass.
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

Use grouped `M+S` Ridge as the defensible performance baseline and retain `M+S+H`
as the best point-estimate sensitivity until it replicates on a prospectively
untouched source. Use `spectral_shape` as the minimal family-routing method, with
the combined/alpha views retained for diagnostics. More model complexity is not
the present bottleneck; independent lineage breadth and prospective evaluation are.
