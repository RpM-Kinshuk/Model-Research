# Model Research

This repository has one supported analysis workflow: the frozen HTSR v3 study for
model-family routing, label-independent clustering, and grouped benchmark
prediction. Data curation and ESD extraction remain separate, read-only inputs to
the study.

## Run The Study

The convenience wrapper uses the checked-in protocol and persistent,
source-fingerprinted feature cache:

```bash
DEVELOPMENT_STATS=/path/to/development/stats \
EXTERNAL_STATS=/path/to/ablation/stats \
scripts/build_htsr_study.sh
```

Use `python scripts/htsr_combined_study.py --help` for fully explicit paths. New
runs are content-addressed under `research_runs/htsr_study/runs/`; cached features
are reused only while their source fingerprints match.

## Open The Dashboard

```bash
HTSR_STUDY_RUN=research_runs/htsr_study/runs/htsr_b41e5fbcb8abcc600bce \
streamlit run app.py
```

The dashboard reads validated frozen artifacts and does not fit or tune models.

## Repository Map

- `spectral_research/study/`: active feature, lineage, performance, and artifact logic
- `scripts/build_htsr_study.sh`: canonical convenience entrypoint
- `scripts/htsr_combined_study.py`: explicit lower-level CLI
- `configs/htsr_combined_study_v3.json`: frozen evaluation protocol
- `alpha_cluster_dashboard/`: read-only dashboard and persistent display cache
- `docs/HTSR_STUDY.md`: method, limitations, and current findings
- `fetch_data/`: Model-Zoo curation and Hugging Face leaderboard ingestion
- `Model-Zoo/`: ESD extraction and operational analysis checkout
- `data/`, `svd_results/`: source inputs; do not mutate during analysis
- `research_runs/htsr_study/`: retained run artifacts and source-bound caches

Older `spectral_research` modules and the explicitly labeled compatibility CLIs
remain for reproducibility, but they are not part of the supported evidence path.
See `fetch_data/README.md` and `Model-Zoo/docs/operations/` for data production.
