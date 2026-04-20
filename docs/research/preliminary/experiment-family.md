# Preliminary Experiment Family

This preliminary phase keeps one fixed experiment family:

1. strict homogeneous control retrieval
2. lean mixed-zoo retrieval
3. rejection mini-study as side evidence
4. performance prediction cycles
5. trend diagnostics as descriptive-only support

Out-of-scope for this phase:

- data-curation redesign in `Model-Research`
- output-contract redesign in `Model-Research`
- replacing `alpha_cluster_dashboard/`
- new large method families beyond the current ones

## Grouped Artifact Layout

The preliminary runs are organized by purpose under these canonical roots and support areas.
The stable scale-up inputs are:

- curated table: `Model-Zoo/data/curated/model_zoo_phase2.csv`
- analysis output pattern: `Model-Zoo/analysis_runs/phase2/<run_name>/`

`Model-Research` stays the phase-3 experimental consumer and report layer.
The operational inputs for the larger curated pool live in `Model-Zoo/data/curated/`, and the operational analysis outputs live in `Model-Zoo/analysis_runs/phase2/`.

Canonical preliminary roots:

- `research_runs/preliminary/retrieval/`
- `research_runs/preliminary/rejection/`
- `research_runs/preliminary/prediction/`

Support areas:

- `research_runs/preliminary/optimization/`
- `research_runs/preliminary/exploratory/`
