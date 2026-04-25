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

## Ablation Dataset Curation

`build_ablation_dataset.py` builds a second, phase-2-disjoint candidate set for
Wasserstein/raw-eigenvalue and ESD ablations. It prioritizes Open LLM
Leaderboard eval-labelled models, caps missing eval rows with
`--max-eval-missing-fraction`, and rejects phase-2 duplicates using normalized
repo ids, canonical keys, and model sha.

The curation step can merge local Hugging Face Hub metadata with
`--metadata-path` before type and suitability labeling. It labels
`esd_suitability` as `good`, `usable_with_caution`, or `separate_or_exclude`.
Outputs are written under `data/ablation/`.

Prerequisite: `data/curated/model_zoo_phase2.csv` must already exist. If it is
missing, produce it first by running `fetch_data/build_phase2_zoo.py`.

Safe local-file run:

```bash
python fetch_data/build_ablation_dataset.py \
  --contents-path data/sampled_leaderboard.csv \
  --metadata-path data/sampled_metadata.csv \
  --phase2-path data/curated/model_zoo_phase2.csv \
  --output-dir data/ablation \
  --target-size 5000 \
  --max-eval-missing-fraction 0.15
```

Network-backed Open LLM Leaderboard run:

```bash
python fetch_data/build_ablation_dataset.py \
  --fetch-contents \
  --phase2-path data/curated/model_zoo_phase2.csv \
  --output-dir data/ablation \
  --target-size 5000 \
  --max-eval-missing-fraction 0.15
```

Expected outputs:

- `data/ablation/ablation_candidates.csv`
- `data/ablation/ablation_selected.csv`
- `data/ablation/ablation_rejected.csv`
- `data/ablation/ablation_overlap_audit.csv`
- `data/ablation/ablation_summary.json`
