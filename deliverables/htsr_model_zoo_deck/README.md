# Metadata-isolated Model-Zoo HTSR deck

Sources:

- `model_zoo_htsr_study.js`: PowerPoint source
- `make_charts.py`: regenerates figures and `assets/chart_data.json`
- `method.md`: claim and number guide

The chart build reads the newest manifest-validated run under
`research_runs/htsr_weight_only/runs/`. Set `HTSR_WEIGHT_ONLY_RUN` to select a
specific complete run. The build never changes source data.

```bash
cd deliverables/htsr_model_zoo_deck
python3 make_charts.py
node model_zoo_htsr_study.js
```

`make_charts.py` requires NumPy, pandas, and Matplotlib. It also writes
`htsr_study_data.xlsx` when OpenPyXL is available. The deck generator requires
`pptxgenjs`; install it in the parent environment rather than adding a local
frontend project.

The deck uses UC Berkeley navy and gold, Dartmouth green, and Booking.com blue
and cyan. It contains a 14-slide main narrative and a 5-slide appendix. PDF
rendering and visual QA are separate steps and are intentionally not part of
this source update.
