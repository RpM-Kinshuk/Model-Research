# Model Research

This repository has one supported analysis workflow: the frozen HTSR v4 study for
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
streamlit run app.py
```

The dashboard automatically selects the newest complete retained study and scale-up
run available under `research_runs/`. Explicit paths still take precedence:

```bash
HTSR_STUDY_RUN=/path/to/retained-run \
HTSR_SCALEUP_RUN=/path/to/scaleup-run \
streamlit run app.py
```

The dashboard reads validated artifacts and does not fit or tune models. The
scale-up overlay defaults to the full Phase-2 ledger, strict-QC lineage clusters,
and direct alpha-profile predictions.

## Repository Map

- `spectral_research/study/`: active feature, lineage, performance, and artifact logic
- `scripts/build_htsr_study.sh`: canonical convenience entrypoint
- `scripts/htsr_combined_study.py`: explicit lower-level CLI
- `configs/htsr_combined_study_v4.json`: frozen evaluation protocol
- `alpha_cluster_dashboard/`: read-only dashboard and persistent display cache
- `docs/HTSR_STUDY.md`: method, limitations, and current findings
- `docs/HTSR_PLAIN_LANGUAGE.md`: concise non-technical explanation and scaled results
- `fetch_data/`: Model-Zoo curation and Hugging Face leaderboard ingestion
- `Model-Zoo/`: ESD extraction and operational analysis checkout
- `data/`, `svd_results/`: source inputs; do not mutate during analysis
- `research_runs/htsr_study/`: retained run artifacts and source-bound caches

Older `spectral_research` modules and the explicitly labeled compatibility CLIs
remain for reproducibility, but they are not part of the supported evidence path.
See `fetch_data/README.md` and `Model-Zoo/docs/operations/` for data production.
