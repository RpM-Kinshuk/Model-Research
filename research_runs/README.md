# Research Runs

This directory stores experiment artifacts.

## Canonical layout

Current preliminary work uses grouped canonical paths under `research_runs/preliminary/`:

- `research_runs/preliminary/retrieval/`
- `research_runs/preliminary/rejection/`
- `research_runs/preliminary/prediction/`
- `research_runs/preliminary/optimization/`
- `research_runs/preliminary/exploratory/`

## Legacy compatibility area

Preserved helper chains that still write historical-style outputs should use `research_runs/legacy/`.

- `research_runs/legacy/alpha_leaderboard/`
- `research_runs/legacy/common_schema_performance_research/`
- `research_runs/legacy/alpha_near_two_check/`
- `research_runs/legacy/spectral_similarity_research/`

The old flat layout has been retired for current preliminary work, but these legacy compatibility paths remain available for preserved helper workflows.
