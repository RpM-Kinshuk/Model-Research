# Scale-Up Research

This directory connects the completed spectral study to the next collection and
model-selection experiments.

- [October 23 research plan](model_zoo_frontier_research_plan_2026-10-23.md):
  task-specific selection, weight measurements, evaluation budgets, and proposed
  collection milestones.
- [Completed study](../../HTSR_STUDY.md): September 2026 results and limitations.
- [Presentation and PDF](../../../deliverables/htsr_model_zoo_deck/README.md):
  measured results with a separate appendix for the proposed scale-up.

At a glance:

- Phase 1 curation and Phase 2 analysis are operationalized in `Model-Zoo`
- Phase 3 experiments and reports live in `Model-Research`

Operational source of truth:

- [curated tables](/jumbo/yaoqingyang/kinshuk/Model-Zoo/data/curated/)
- [analysis outputs](/jumbo/yaoqingyang/kinshuk/Model-Zoo/analysis_runs/phase2/)
- [collector operation and output schema](/jumbo/yaoqingyang/kinshuk/Model-Zoo/docs/operations/analysis.md)

`Model-Research` owns the retained combined study, weight-only grouping, and
reassessment reports. Those experiments consume historical Phase 2 and ablation
measurements. The newer collector's revision-pinned records and full computed
spectra are a separate collection; its improved provenance does not repair the
older benchmark joins.

The scale-up plan's model-selection system, hybrid comparison, and 50k measured
artifact release are proposed work. The plan is a dated research note, not a
record of completed experiments or an implementation specification.

The production catalog checked October 24 contains 46 valid successful
measurement records across 44 repository IDs: 43 with complete matrix coverage
and three with partial coverage. Records use different layer-selection versions.
These counts describe collection output, not additional prediction results.

Use this directory for scale-up-facing research notes, summaries, and report links. Do not treat it as the canonical home for curated tables or analysis outputs.
