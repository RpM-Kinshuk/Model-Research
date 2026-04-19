# Preliminary Research Draft

## North Star

The project goal is to manage a large zoo of models using signals that can be extracted from the models themselves.

In practice, that means three things have to stay true as the work scales:

- the method should not depend on rich catalog metadata being present
- the method should be useful for both homogeneous and heterogeneous mixtures of models
- performance prediction matters, but only if it grows naturally out of the same zoo representation

So the center of the research is not “one scalar predicts everything,” and it is not “solve every model-analysis problem at once.” The center is that model-intrinsic spectral structure may provide a practical way to place, compare, and reason about models inside one large collection.

## Why We Started Narrow

The preliminary study began with a strict homogeneous benchmark for a simple reason: we needed to know whether the spectral signal was real before testing it in a broader and messier setting.

If we had started immediately with the whole mixed pool, we would have mixed together clean family recovery, weak labels, derived-model effects, merges, quantization, schema mismatch, and metadata noise. Then a positive or negative result would have been hard to interpret.

So the strict benchmark was a control, not the destination. It was meant to answer one clean first question:

`Can module-aware spectral fingerprints recover useful family structure from model weights alone?`

That question now has a clear answer: yes.

## What We Built

The preliminary system now has two benchmark layers, one retrieval method ladder, and two secondary analysis branches.

- A strict homogeneous control benchmark. This is the cleanest audited setting and the place where the strongest first retrieval result comes from.
- A lean mixed-zoo benchmark. This is the broader probe that matters more for the actual project goal. It removes architecture and parameter-count shortcuts from the baseline and includes a real heterogeneous supported slice.

The retrieval method ladder is:

- `full_module_trace`
- `role_aggregated_trace`
- `attention_trace`
- `pl_alpha_scalar`
- `shared_role_trace`

The secondary branches are:

- a neighborhood-based prediction pipeline that tests whether the same zoo representation carries performance signal
- a descriptive trend-analysis branch that checks whether simple absolute or lineage-relative spectral trends correlate with benchmark behavior

This is enough to support a coherent preliminary story without pretending that every branch is equally mature.

## What Holds Up

The strongest result is still retrieval.

On the strict control benchmark, `full_module_trace` reaches top-1 `0.918` and MRR `0.950`. `attention_trace` is almost identical at top-1 `0.918` and MRR `0.946`. Even the simpler `pl_alpha_scalar` baseline remains strong at top-1 `0.857`, but the richer traces are better. That is the clearest proof that the core spectral signal is real and that a structured trace is more useful than a compressed scalar view.

The second strong result is that mixed-zoo retrieval is no longer hypothetical. On the lean benchmark, the heterogeneous supported slice is real: `45` heterogeneous supported rows before split and `14` heterogeneous test queries after split. On that slice, `role_aggregated_trace` is the best current method at top-1 `0.929` and MRR `0.964`. `full_module_trace` remains strong at top-1 `0.857` and MRR `0.907`, and both beat the structural baseline.

This matters because it means the project is no longer trapped inside the easy common-schema regime. We now have evidence that a simple heterogeneity-aware spectral representation can work on a broader model mixture.

The third result that holds up is that spectral neighborhoods carry real prediction signal, even though prediction is not yet solved. In the stronger buckets, neighborhood prediction using the richer traces is clearly useful. For example, `full_module_trace` reaches R² `0.831` on `MMLU-PRO` for `supported_candidate`, and R² `0.806` on `MMLU-PRO` for `unsupported_special`. The right conclusion is not that prediction is solved. The right conclusion is that the same neighborhood structure used for retrieval also contains nontrivial performance information.

## What Stayed Secondary Or Weak

Several branches stayed secondary.

Rejection improved, but it stayed weak. The best simple confidence rule combines the top score with the top-two margin, and it helps a little. But the improvement is not large enough to support a strong open-world claim. Rejection should stay outside the main paper claim for now.

Metadata-heavy controls turned out to be a warning rather than a destination. On the narrow strict benchmark, once architecture and size metadata were fully backfilled, metadata alone became extremely strong. That does not kill the spectral story, but it shows that narrow clean benchmarks can be partly solved by catalog-like fields. For the actual zoo goal, those controls are diagnostic, not the real method.

Trend-only signals also stayed secondary. Absolute trend features do show moderate association with the main targets on the managed slice, but exact-parent evidence is still small-sample and family-relative evidence is weak. So trend summaries are worth keeping only as supporting features.

The hardest remaining weak point is heterogeneous prediction. Retrieval now has a credible mixed-zoo result. Prediction does not yet have an equally strong heterogeneous story.

## Current Method Hierarchy

The current method ranking should stay disciplined.

Primary methods:

- `full_module_trace`
- `role_aggregated_trace`

These are the methods the project should carry forward into scale-up as the main spectral representations.

Secondary methods:

- `attention_trace`
- `pl_alpha_scalar`
- spectral-neighborhood prediction
- trend-style descriptive features

These matter, but mostly as comparators, supporting evidence, or secondary extensions.

Side-result only:

- rejection
- metadata-heavy controls

These should remain in the record, but they should not drive the framing.

## Canonical Preliminary Artifact Set

The branch now has many artifacts, but the preliminary story should keep referring back to a small canonical set.

- consolidated findings note:
  [2026-04-18-spectral-family-retrieval-preliminary-findings.md](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/docs/superpowers/specs/2026-04-18-spectral-family-retrieval-preliminary-findings.md)
- strict control benchmark:
  [report.md](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/spectral_family_retrieval_strict_headline_v2/report.md)
- lean mixed-zoo benchmark:
  [report.md](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/lean_mixed_zoo_benchmark/report.md)
- performance prediction follow-up:
  [report.md](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_prediction_cycle2/report.md)
- trend diagnostics:
  [report.md](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_trend_diagnostics/report.md)

These are the documents and runs that should anchor the preliminary phase.

## What Needs To Happen Before Scale-Up

The remaining work before scale-up should stay narrow.

First, the preliminary narrative needs to be stable in writing. That is the purpose of this draft and the outline behind it. The project should not add more side experiments until the current evidence is written in a clean and disciplined form.

Second, the experiment family should stay fixed. The larger-scale rerun should preserve the same core structure:

- strict homogeneous control
- lean mixed-zoo retrieval
- heterogeneity-aware retrieval comparison
- neighborhood-based performance prediction
- descriptive trend diagnostics as secondary features

Third, the method hierarchy should not expand without a strong reason. The project already knows which methods are primary and which ones are only supporting evidence.

Fourth, the current limits need to stay explicit even during scale-up:

- we do not solve open-world model identification
- we do not solve robust rejection
- we do not solve general performance prediction
- we do not claim that one scalar metric explains the zoo

The preliminary phase is successful because it established a believable center:

`model-intrinsic spectral structure is useful for organizing a mixed zoo of models`

What comes next is not a new research direction. It is the larger-scale rerun of the same disciplined experiment family.
