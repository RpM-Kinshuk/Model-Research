# Scripts

## Supported

- `build_htsr_study.sh`: canonical HTSR v3 build using the frozen protocol and a
  persistent, source-fingerprinted cache. Set `DEVELOPMENT_STATS` and
  `EXTERNAL_STATS` to select metric collections.
- `htsr_combined_study.py`: lower-level CLI with explicit source, cache, output,
  and protocol paths.
- `streamlit run app.py`: read-only dashboard over one validated study run selected
  with `HTSR_STUDY_RUN` or the sidebar.

## Compatibility

The remaining pre-HTSR research CLIs support older tests and reproducibility only.
They are not used to produce the current study or dashboard evidence. Model-Zoo
curation and Hugging Face leaderboard ingestion live in `fetch_data/`, not here.
