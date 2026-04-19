# Fetch Data

This directory now has one active phase-2 entrypoint:

- `build_phase2_zoo.py`

It builds the unified candidate zoo for scale-up:

- ingests the current 5k sample
- collapses duplicate leaderboard rows
- expands from `large_nlp_data.json`
- applies soft balancing
- runs lightweight Hugging Face preflight checks
- writes canonical outputs under `data/curated/`

Preserved helper scripts:

- `atlas_subsample.py`
- `check_diversity.py`
- `fetch_eval_data.py`
- `load_model.py`

These remain useful references, but they are not the main phase-2 pipeline.
