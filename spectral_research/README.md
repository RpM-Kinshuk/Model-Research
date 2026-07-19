# spectral_research

The supported workflow is `spectral_research.study`:

- `contracts.py`: immutable inputs, model identity, graph grouping, and provenance
- `features.py`: canonical-role `M`, `S`, and `H` feature extraction
- `lineage.py`: family routing, open-set diagnostics, and label-free clustering
- `performance.py`: ancestry-grouped nested Ridge, repeated holdouts, and paired inference
- `artifacts.py`: frozen artifact manifest and validation
- `protocol.py`: serialized protocol validation
- `runner.py`: content-addressed orchestration and reporting

The package-level modules outside `study/` are retained compatibility code for
pre-HTSR experiments. They are not imported by the supported study path.
