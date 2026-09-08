# Model Research

This repository studies model grouping, supplied-base matching, and benchmark
prediction from saved weight measurements. The current reassessment keeps the
earlier study for comparison. Neither uses annotated family or architecture
labels as predictor inputs. Collection metadata defines the datasets and some
evaluation splits; benchmark training scores are required for score prediction.

Read [the current findings](docs/HTSR_PLAIN_LANGUAGE.md) or
[the method and limitations](docs/HTSR_STUDY.md). Results are exploratory: saved
analyses do not record the resolved checkpoint commit needed to verify score joins.

## Run The Study

The dependency order is combined tables, validation tables, the earlier weight-
only study, then reassessment. Existing complete inputs can be reused.
First build the retained combined-study tables:

```bash
DEVELOPMENT_STATS=/path/to/development/stats \
EXTERNAL_STATS=/path/to/ablation/stats \
scripts/build_htsr_study.sh
```

Prepare validation tables, then run the weight-only analysis:

```bash
python scripts/htsr_scaleup_study.py \
  --study-run research_runs/htsr_study/runs/<run>

python scripts/htsr_weight_only_study.py \
  --study-run research_runs/htsr_study/runs/<run> \
  --validation-run research_runs/htsr_scaleup/runs/<validation-run>
```

The validation run supplies cleaned benchmark targets and lineage labels.
Training scores may be used during fitting; held-out scores and lineage labels
are used only for evaluation. Reassess the saved measurements:

```bash
python -B -m spectral_research.study.reassessment_lineage \
  research_runs/htsr_weight_only/runs/<run> \
  research_runs/htsr_reassessment/lineage_check_new

python scripts/htsr_reassessment.py \
  --run research_runs/htsr_weight_only/runs/<run> \
  --lineage-check research_runs/htsr_reassessment/lineage_check_new
```

The lineage destination must be new or empty. The reassessment uses fixed model
settings, shared-feature grouping checks, spectral controls, individual-task
checks, and a role-free comparison. `--reuse-run` can reuse validated aggregate
predictions from the same parent. All analysis output stays in `research_runs/`;
collection inputs are not changed.

After a reassessment completes, generate its independent audit:

```python
from spectral_research.study.reassessment_audit import run_reassessment_audit
run_reassessment_audit("path/to/weight-only-run", "path/to/reassessment-run")
```

The audit checks identities, overlaps, collection directions, range failures,
and tied base candidates without refitting. The presentation build requires it.

## Open The Dashboard

```bash
streamlit run app.py
```

The dashboard automatically selects the newest complete retained study and
weight-only run. Explicit paths still take precedence:

```bash
HTSR_STUDY_RUN=/path/to/retained-run \
HTSR_SCALEUP_RUN=/path/to/weight-only-run \
streamlit run app.py
```

The dashboard reads validated artifacts and does not fit or tune models. Its main
views use a complete-role clustering core, masking-checked partial assignments,
observed representative models, and held-out score predictions. Validation
labels remain separate.
The **Reassessment** view compares methods and test designs without replacing
the prior galaxy or silently relabeling its predictions. It also shows coverage,
root-mean-square error, and predictions outside the score range.

## Repository Map

- `spectral_research/study/`: active feature, lineage, performance, and artifact logic
- `scripts/build_htsr_study.sh`: retained combined-study builder
- `scripts/htsr_combined_study.py`: explicit lower-level CLI
- `scripts/htsr_weight_only_study.py`: metadata-isolated analysis and validation
- `scripts/htsr_reassessment.py`: current fixed-method comparisons and controls
- `configs/htsr_combined_study_v4.json`: frozen evaluation protocol
- `alpha_cluster_dashboard/`: read-only dashboard and persistent display cache
- `docs/HTSR_STUDY.md`: method, limitations, and current findings
- `docs/HTSR_PLAIN_LANGUAGE.md`: concise non-technical explanation and scaled results
- `fetch_data/`: Model-Zoo curation and Hugging Face leaderboard ingestion
- `Model-Zoo/`: ESD extraction and operational analysis checkout
- `data/`, `svd_results/`: source inputs; do not mutate during analysis
- `research_runs/htsr_study/`: retained run artifacts and source-bound caches
- `research_runs/htsr_weight_only/`: earlier frozen predictions and reports
- `research_runs/htsr_reassessment/`: current comparisons and input-bound audits

Older `spectral_research` modules and the explicitly labeled compatibility CLIs
remain for reproducibility, but they are not part of the supported evidence path.
See `fetch_data/README.md` and `Model-Zoo/docs/operations/` for data production.
