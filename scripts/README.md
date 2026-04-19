# Scripts

This directory keeps both the current canonical experiment CLIs and a small set of preserved exploratory helpers. If you are running the active preliminary pipeline, start with the canonical entrypoints below.

## Canonical current entrypoints

### Retrieval

- `python -m spectral_research.pipeline`
  - Strict homogeneous control default output: `research_runs/preliminary/retrieval/spectral_family_retrieval_strict_headline_v2`
- `mixed_zoo_retrieval_research.py`
  - Strict default output: `research_runs/preliminary/retrieval/mixed_zoo_retrieval_cycle1`
  - Lean default output when `--output-dir` is omitted: `research_runs/preliminary/retrieval/lean_mixed_zoo_benchmark`

### Rejection

- `rejection_confidence_study.py`
  - Default output: `research_runs/preliminary/rejection/rejection_confidence_study_strict_v1`

### Prediction

- `spectral_prediction_research.py`
  - Default output: `research_runs/preliminary/prediction/performance_prediction_cycle2`
- `performance_trend_diagnostics.py`
  - Default output: `research_runs/preliminary/prediction/performance_trend_diagnostics`

## Preserved exploratory or legacy helpers

- `alpha_leaderboard_correlation.py`
- `alpha_near_two_hypothesis_check.py`
- `common_schema_performance_research.py`
- `find_merge_models_with_valid_analysis.py`
  - Preserved helper defaults to `research_runs/legacy/merge_models_with_valid_analysis.csv`
- `spectral_similarity_research.py`
  - Preserved helper defaults to `research_runs/legacy/spectral_similarity_research`
