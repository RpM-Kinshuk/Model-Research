# Model Research

This repository contains the current phase-1 preliminary research pipeline for organizing a mixed zoo of models with spectral structure. The active codepaths, scripts, and saved outputs now follow the grouped `research_runs/preliminary/...` layout.

## Start here

- Reader-facing research summary: `docs/research/preliminary/`
- Active package: `spectral_research/`
- Active data/loading utilities: `alpha_cluster_dashboard/`
- Canonical experiment entrypoints: `scripts/`
- Saved experiment outputs: `research_runs/preliminary/`

## Active areas

- `spectral_research/`: retrieval, prediction, rejection, heterogeneity, evaluation, and trend analysis logic
- `alpha_cluster_dashboard/`: data loading and shared dashboard-facing helpers still used by the active pipeline
- `scripts/`: canonical experiment CLIs plus preserved exploratory helpers
- `data/`: sampled metadata, leaderboard inputs, and manual audit freezes
- `svd_results/`: spectral metric sources consumed by the experiments
- `tests/`: unit and regression coverage for the active research code
- `research_runs/preliminary/`: grouped preliminary retrieval, rejection, and prediction outputs

## Notes

- `docs/research/preliminary/` is the current narrative for the preliminary phase.
- `docs/superpowers/` contains internal design and planning history.
- Some older helper scripts remain in `scripts/`, but the current supported experiment surface is documented in `scripts/README.md`.
