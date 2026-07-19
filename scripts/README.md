# Scripts

## Supported

- `build_htsr_study.sh`: canonical HTSR v4 build using the frozen protocol and a
  persistent, source-fingerprinted cache. Set `DEVELOPMENT_STATS` and
  `EXTERNAL_STATS` to select metric collections. Performance evaluation includes
  grouped 5-fold CV and a 20-repeat ancestry-disjoint 70/30 sensitivity.
- `htsr_combined_study.py`: lower-level CLI with explicit source, cache, output,
  and protocol paths.
- `htsr_scaleup_study.py`: evaluates a retained run across its full Phase-2 and
  ablation populations. Inputs and statistical settings come from the retained
  run's provenance; generated outputs are content-addressed under
  `research_runs/htsr_scaleup/`.
- `streamlit run app.py`: read-only dashboard with automatic newest-complete local
  defaults. Override the retained base with `HTSR_STUDY_RUN` and the optional
  large-scale overlay with `HTSR_SCALEUP_RUN`, or edit either path in the sidebar.

Run the scale-up against any retained study directory:

```bash
python scripts/htsr_scaleup_study.py --study-run "$HTSR_STUDY_RUN"
```

## Compatibility

The remaining pre-HTSR research CLIs support older tests and reproducibility only.
They are not used to produce the current study or dashboard evidence. Model-Zoo
curation and Hugging Face leaderboard ingestion live in `fetch_data/`, not here.
