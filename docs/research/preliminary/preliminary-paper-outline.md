# Preliminary Paper Outline

## Working Framing

`Model-intrinsic spectral structure is useful for organizing a mixed zoo of models.`

The preliminary study should be written as a retrieval-first paper with a mixed-zoo motivation. The center is not generic model fingerprinting, and it is not benchmark prediction by itself. The center is that weight-derived spectral structure gives a practical way to organize, compare, and route models in a large open model collection, and that this remains useful when the zoo contains both homogeneous and heterogeneous schemas.

## 1. Introduction

The paper should open from the model-zoo problem, not from a narrow benchmark. Public model collections are growing faster than they can be manually organized, and catalog metadata is often incomplete, noisy, or absent. That makes model-intrinsic signals attractive. The motivating question is whether spectral structure extracted directly from weights can serve as a usable organizing signal for a large model zoo.

The introduction should then narrow quickly to the preliminary contribution: we do not solve the full zoo problem yet, but we show that module-aware spectral traces are already useful for family-level retrieval, and that a simple heterogeneity-aware variant remains effective when schemas differ.

## 2. Problem Setup

The paper should define two regimes clearly.

The first is a strict homogeneous control benchmark. This is the clean benchmark used to test whether there is real retrieval signal at all. It is intentionally narrower so that the first result is interpretable.

The second is a lean mixed-zoo benchmark. This is the broader probe that matters more for the actual research goal. It includes a wider supported pool, real heterogeneous models, and unsupported buckets that better resemble a practical zoo setting.

This section should explain that the preliminary study starts narrow for clarity, but the destination is mixed-zoo organization rather than root/base-only bookkeeping.

## 3. Methods

The methods section should keep a clear hierarchy.

Primary methods:
- `full_module_trace`: the main module-aware spectral representation
- `role_aggregated_trace`: the main heterogeneity-aware representation

Secondary methods:
- `attention_trace`: a leaner trace comparator
- `pl_alpha_scalar`: a simple spectral baseline
- `structural_basics_only`: a model-intrinsic non-spectral control in the lean benchmark

The text should emphasize that `full_module_trace` is the default representation when schema alignment is straightforward, while `role_aggregated_trace` is the simplest useful fallback when exact module matching is too brittle. Trend-style summaries should be mentioned only as secondary descriptive features, not as the main predictive method.

## 4. Benchmarks

This section should separate the two main preliminary benchmarks.

The strict benchmark is the clean homogeneous control. Its strongest headline result is that `full_module_trace` and `attention_trace` both reach top-1 `0.918`, with MRR `0.950` and `0.946` respectively, while `pl_alpha_scalar` remains strong but lower at top-1 `0.857`.

The lean benchmark is the broader mixed-zoo probe. It removes architecture and parameter-count shortcuts from the baseline and yields a real heterogeneous supported slice: `45` heterogeneous supported rows before split and `14` heterogeneous test queries after split. This is the benchmark that should carry the broader-sense claim.

## 5. Results

### 5.1 Homogeneous Control

The first result section should establish that the main signal is real. On the strict control benchmark, module-aware traces perform strongly and consistently. This is the cleanest evidence that the weight-derived spectral representation is not just noise.

### 5.2 Mixed-Zoo And Heterogeneous Retrieval

The second result section should shift to the broader goal. On the heterogeneous slice of the lean benchmark, `role_aggregated_trace` is the best current method, reaching top-1 `0.929` and MRR `0.964`. `full_module_trace` remains strong at top-1 `0.857`, MRR `0.907`, and both beat `structural_basics_only` at top-1 `0.714`, MRR `0.845`.

This is the key mixed-zoo result: spectral retrieval remains useful beyond the clean common-schema slice, and broad role aggregation is the most promising current way to handle heterogeneity without overcomplicating the method.

### 5.3 Secondary Prediction Result

Performance prediction should appear as a secondary result, not the main one. The best preliminary signal comes from neighborhood-based prediction using the richer spectral traces. On stronger slices, this signal is meaningful: for example, `full_module_trace` reaches R² `0.831` on `MMLU-PRO` for `supported_candidate`, and R² `0.806` on `MMLU-PRO` for `unsupported_special`.

The paper should also be clear that this signal is uneven. Heterogeneous prediction is still hard, and the best heterogeneous rows remain weak or negative in R² despite some improvement in Cycle 2.

### 5.4 Trend Diagnostics

Trend diagnostics should be framed as descriptive support only. On the managed `supported_candidate` slice, absolute trend signals show moderate association with the main targets, including structural-adjusted Spearman values around `0.604` for `Average ⬆️`, `0.646` for `MMLU-PRO`, and `0.649` for `MATH Lvl 5`.

Exact-parent trend evidence exists but is still small-sample, and family-relative trend evidence is weak. These results justify keeping trend-style summaries as secondary features, not as the main predictor family.

## 6. What Holds Up

The discussion section should make four points.

- Spectral structure is useful for organizing a zoo of models.
- Rich module-aware traces are the strongest current retrieval representation.
- Role aggregation helps when schema mismatch becomes a real problem.
- Spectral neighborhoods carry some performance signal, but prediction remains a secondary and still uneven direction.

## 7. Limits

The limitations section should stay explicit.

- We do not solve open-world model identification.
- Rejection improves slightly but is not strong enough to lead with.
- We do not solve general performance prediction.
- Trend evidence is secondary and mostly descriptive.
- The heterogeneous slice is now real, but it is still smaller than the common-schema regime.

## 8. Next Step

The paper should close the preliminary phase cleanly. The next step is not to invent many new methods. It is to scale the same experiment family to a larger model pool while preserving the method hierarchy that already survived the preliminary study:

- strict homogeneous control
- lean mixed-zoo retrieval
- heterogeneity-aware retrieval comparison
- neighborhood-based performance prediction
- descriptive trend diagnostics as secondary features

That preserves the main research direction: zoo management first, heterogeneous support as a real requirement, and performance prediction as the next layer rather than the opening claim.

## Canonical Supporting Artifacts

- Consolidated findings: [2026-04-18-spectral-family-retrieval-preliminary-findings.md](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/docs/superpowers/specs/2026-04-18-spectral-family-retrieval-preliminary-findings.md)
- Strict control benchmark: [report.md](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/spectral_family_retrieval_strict_headline_v2/report.md)
- Lean mixed-zoo benchmark: [report.md](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/lean_mixed_zoo_benchmark/report.md)
- Prediction Cycle 2: [report.md](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_prediction_cycle2/report.md)
- Trend diagnostics: [report.md](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_trend_diagnostics/report.md)
