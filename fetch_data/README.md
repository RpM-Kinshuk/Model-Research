# Fetch Data

This directory now has one active phase-2 entrypoint:

- `build_phase2_zoo.py`

It builds the unified candidate zoo for scale-up:

- ingests the current 5k sample
- collapses duplicate leaderboard rows
- expands from `large_nlp_data.json`
- can optionally supplement from the Open LLM Leaderboard dataset
- can optionally supplement from small Hugging Face search passes for underfilled categories
- applies soft balancing
- runs lightweight Hugging Face preflight checks
- writes canonical outputs under `data/curated/`

Useful flags:

- `--supplement-leaderboard`
- `--supplement-hf`
- `--supplement-per-query-limit`
- `--skip-hub-probes`

Preserved helper scripts:

- `atlas_subsample.py`
- `check_diversity.py`
- `fetch_eval_data.py`
- `load_model.py`

These remain useful references, but they are not the main phase-2 pipeline.
