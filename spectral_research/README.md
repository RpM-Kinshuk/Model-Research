# spectral_research

The supported workflow is `spectral_research.study`:

- `weight_only.py` and `partial_roles.py`: weight-derived grouping, supplied-base
  matching, and comparable-neighbor score prediction
- `reassessment_lineage.py`: norm-based base matching beside the alpha comparator
- `reassessment_audit.py`: identity, overlap, range, transfer, and tie checks
- `contracts.py`: immutable inputs, model identity, graph grouping, and provenance
- `features.py`: canonical-role `M`, `S`, and `H` feature extraction
- `lineage.py`: layer sequences and retained lineage comparison methods
- `performance.py`: retained combined-study regression and evaluation controls
- `artifacts.py`: frozen artifact manifest and validation
- `protocol.py`: serialized protocol validation
- `runner.py`: content-addressed orchestration and reporting

`scripts/htsr_reassessment.py` runs the current fixed-method comparisons.
See [the study](../docs/HTSR_STUDY.md) for the September results and
[the scale-up plan](../docs/research/scaleup/model_zoo_frontier_research_plan_2026-10-23.md)
for proposed model-selection experiments. The proposal does not change the
predictor inputs or results of the completed study.

The package-level modules outside `study/` are retained compatibility code for
pre-HTSR experiments. They are not imported by the supported study path.
