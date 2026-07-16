# Scripts

This directory keeps the supported current entrypoints for the active preliminary workflows, plus a small set of preserved exploratory helpers. Some of these entrypoints write support or intermediate runs; the canonical artifact set is defined in `docs/research/preliminary/`.

## Supported current entrypoints

### Retrieval

- `python -m spectral_research.pipeline`
  - Strict homogeneous control default output: `research_runs/preliminary/retrieval/spectral_family_retrieval_strict_headline_v2`
- `mixed_zoo_retrieval_research.py`
  - Strict default output: `research_runs/preliminary/retrieval/mixed_zoo_retrieval_cycle1`
  - Lean default output when `--output-dir` is omitted: `research_runs/preliminary/retrieval/lean_mixed_zoo_benchmark`

### Rejection

- `rejection_confidence_study.py`
  - Default output: `research_runs/preliminary/rejection/rejection_confidence_study_strict_v1`
  - Canonical artifact root: `research_runs/preliminary/rejection/`

### Prediction

- `spectral_prediction_research.py`
  - Default output: `research_runs/preliminary/prediction/performance_prediction_cycle2`
  - Canonical artifact root: `research_runs/preliminary/prediction/`
- `performance_trend_diagnostics.py`
  - Default output: `research_runs/preliminary/prediction/performance_trend_diagnostics`
  - Support or intermediate diagnostics under the prediction area

### Spectral atlas

- `streamlit run app.py`
  - Interactive dashboard for clustering, alpha profiles, and prediction-neighborhood visualizations.
  - Defaults to `output/results/metrics` with the curated phase-2 metadata.
- `build_ablation_dashboard_cache.py`
  - Precomputes the persistent atlas cache used by the dashboard.
  - Default output: `research_runs/preliminary/results/dashboard_cache`
- `ablation_spectral_analysis.py`
  - Coverage, metadata, and scalar spectral-feature summary for the ablation collection.
  - Default output: `research_runs/preliminary/ablation/ablation_spectral_analysis_20260708`
- `ablation_method_study.py`
  - Minimal comparison of paper-faithful Spearman, existing alpha baselines, rank-shape variants, a role-wise alpha Wasserstein check, and a structural leakage control.
  - Default output: `research_runs/preliminary/ablation/ablation_method_study_20260708`

## Preserved exploratory or legacy helpers

- `alpha_leaderboard_correlation.py`
- `alpha_near_two_hypothesis_check.py`
- `common_schema_performance_research.py`
- `find_merge_models_with_valid_analysis.py`
  - Preserved helper defaults to `research_runs/legacy/merge_models_with_valid_analysis.csv`
- `spectral_similarity_research.py`
  - Preserved helper defaults to `research_runs/legacy/spectral_similarity_research`
