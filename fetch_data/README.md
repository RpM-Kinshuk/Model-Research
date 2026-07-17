# Model-Zoo Data Production

These files are the retained curation and Hugging Face ingestion path. They are
independent of the HTSR analysis code and must not be removed during study cleanup.

## Phase-2 Zoo

`build_phase2_zoo.py` combines sampled metadata, leaderboard rows, the Model Atlas
graph, optional Open LLM Leaderboard rows, and optional Hugging Face Hub searches.
Its support modules are `phase2_schema.py`, `leaderboard_supplement.py`,
`hf_supplement.py`, and `hub_preflight.py`.

To write the table consumed by Model-Zoo explicitly:

```bash
python fetch_data/build_phase2_zoo.py \
  --metadata-path data/sampled_metadata.csv \
  --leaderboard-path data/sampled_leaderboard.csv \
  --atlas-json-path fetch_data/large_nlp_data.json \
  --svd-dir svd_results \
  --output-dir Model-Zoo/data/curated
```

Add `--supplement-leaderboard` for `open-llm-leaderboard/contents`,
`--supplement-hf` for targeted Hub searches, or `--skip-hub-probes` for an offline
curation pass.

## Ablation Zoo

`build_ablation_dataset.py` and `ablation_schema.py` construct a phase-2-disjoint
ablation table. A local-input run is:

```bash
python fetch_data/build_ablation_dataset.py \
  --contents-path data/sampled_leaderboard.csv \
  --metadata-path data/sampled_metadata.csv \
  --phase2-path Model-Zoo/data/curated/model_zoo_phase2.csv \
  --output-dir Model-Zoo/data/ablation
```

Use `--fetch-contents` instead of `--contents-path` to load the Open LLM
Leaderboard dataset directly. `atlas_subsample.py` is retained as a graph sampling
utility. Tests under `tests/test_*supplement.py`, `tests/test_phase2_schema.py`, and
`tests/test_*ablation*.py` protect these production contracts.
