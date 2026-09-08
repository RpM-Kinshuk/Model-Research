# Metadata-isolated Model-Zoo HTSR deck

Sources:

- `model_zoo_htsr_study.js`: PowerPoint source
- `make_charts.py`: regenerates figures and `assets/chart_data.json`
- `method.md`: claim and number guide

The chart build reads the newest manifest-validated run under
`research_runs/htsr_weight_only/runs/` plus its newest completed reassessment.
Set `HTSR_WEIGHT_ONLY_RUN` and `HTSR_REASSESSMENT_RUN` to select specific complete
runs. The build never changes source data.

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
and cyan. It contains a 16-slide main narrative and a 5-slide appendix. The PPTX
and PDF are rebuilt from the same chart data; the PDF is rendered for visual
inspection. Findings and scope are defined in `docs/HTSR_STUDY.md`, not inferred
from the decorative cover or galaxy layout.
