# Scripts

## Supported

- `htsr_weight_only_study.py`: primary metadata-isolated study. It fits clusters
  on complete seven-role summaries, adds partial summaries only after a masking
  check, routes queries against a separately supplied base gallery, and predicts
  benchmark scores from three comparable training models. Descriptive metadata
  is loaded only after predictions are frozen.
- `build_htsr_study.sh`: canonical HTSR v4 build using the frozen protocol and a
  persistent cache keyed by source content. Set `DEVELOPMENT_STATS` and
  `EXTERNAL_STATS` to select metric collections.
- `htsr_combined_study.py`: lower-level CLI with explicit source, cache, output,
  and protocol paths.
- `htsr_scaleup_study.py`: prepares the larger Phase 2 and ablation populations
  and validation tables consumed by the metadata-isolated study. Its old
  older lineage experiments are compatibility outputs, not the primary method.
- `streamlit run app.py`: read-only dashboard with automatic newest-complete local
  defaults. Override run paths with the environment variables shown in the app's
  sidebar.

Run the current study from retained feature and validation inputs:

```bash
python scripts/htsr_weight_only_study.py \
  --study-run research_runs/htsr_study/runs/htsr_7f739673cde388be2d4c \
  --validation-run research_runs/htsr_scaleup/runs/scaleup_05d8fb3a8bea1b39f713
```

Outputs are content-addressed under `research_runs/htsr_weight_only/runs/`.
Profile caches persist under `research_runs/htsr_weight_only/cache/` and are
reused until source content or method settings change.

## Compatibility

The remaining pre-HTSR research CLIs support older tests and reproducibility only.
They are not used to produce the current study or dashboard evidence. Model-Zoo
curation and Hugging Face leaderboard ingestion live in `fetch_data/`, not here.
