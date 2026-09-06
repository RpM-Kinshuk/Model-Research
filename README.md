# Model Research

This repository has one supported reporting workflow: weight-only spectral
clustering, supplied-base routing, and benchmark prediction. Model metadata is
opened only after predictions are frozen, for validation and descriptive audits.
Data curation and empirical spectral density (ESD) extraction remain separate,
read-only inputs.

## Run The Study

First build the frozen spectral tables with the checked-in protocol and
source-fingerprinted cache:

```bash
DEVELOPMENT_STATS=/path/to/development/stats \
EXTERNAL_STATS=/path/to/ablation/stats \
scripts/build_htsr_study.sh
```

Then run the metadata-isolated analysis:

```bash
python scripts/htsr_weight_only_study.py \
  --study-run research_runs/htsr_study/runs/<run> \
  --validation-run research_runs/htsr_scaleup/runs/<validation-run>
```

The validation run supplies cleaned benchmark targets and labels used only after
the prediction freeze. New results are content-addressed under
`research_runs/htsr_weight_only/runs/`.

## Open The Dashboard

```bash
streamlit run app.py
```

The dashboard automatically selects the newest complete retained study and
weight-only run. Explicit paths still take precedence:

```bash
HTSR_STUDY_RUN=/path/to/retained-run \
HTSR_SCALEUP_RUN=/path/to/scaleup-run \
streamlit run app.py
```

The dashboard reads validated artifacts and does not fit or tune models. Its main
views use a complete-role clustering core, masking-checked partial assignments,
observed representative models, and held-out score predictions. Validation
labels remain separate.

## Repository Map

- `spectral_research/study/`: active feature, lineage, performance, and artifact logic
- `scripts/build_htsr_study.sh`: canonical convenience entrypoint
- `scripts/htsr_combined_study.py`: explicit lower-level CLI
- `scripts/htsr_weight_only_study.py`: metadata-isolated analysis and validation
- `configs/htsr_combined_study_v4.json`: frozen evaluation protocol
- `alpha_cluster_dashboard/`: read-only dashboard and persistent display cache
- `docs/HTSR_STUDY.md`: method, limitations, and current findings
- `docs/HTSR_PLAIN_LANGUAGE.md`: concise non-technical explanation and scaled results
- `fetch_data/`: Model-Zoo curation and Hugging Face leaderboard ingestion
- `Model-Zoo/`: ESD extraction and operational analysis checkout
- `data/`, `svd_results/`: source inputs; do not mutate during analysis
- `research_runs/htsr_study/`: retained run artifacts and source-bound caches
- `research_runs/htsr_weight_only/`: current frozen predictions and reports

Older `spectral_research` modules and the explicitly labeled compatibility CLIs
remain for reproducibility, but they are not part of the supported evidence path.
See `fetch_data/README.md` and `Model-Zoo/docs/operations/` for data production.
