# Deck evidence and claim guide

The main methodology and findings are in [HTSR_STUDY.md](../../docs/HTSR_STUDY.md).
`make_charts.py` reads manifest-validated runs and saves the numbers in
`assets/chart_data.json` and `htsr_study_data.xlsx`. Do not copy older slide
numbers over newer results without checking the population and test design.

## Evidence boundary

- Grouping uses weights only; validation labels are revealed after assignments.
- Base matching assumes an operator-supplied candidate list. It does not discover
  historical bases from nothing or infer training direction.
- Score prediction uses training benchmark scores. Held-out scores are excluded
  from fitting. Descriptive metadata is not a numeric feature.
- Metadata shapes collection and collection-transfer splits. Thus the predictor
  is metadata-isolated, but the experiment is not knowledge-free.
- Current analyses lack a resolved weight checkpoint commit. Repository-based
  score joins are not checkpoint-verified. All results remain exploratory.

## What each comparison supports

| Comparison | Reading |
| --- | --- |
| Retained layer-wise groups versus broad-family labels | Useful broad grouping, not exact parent recovery |
| All-matrix quantile groups | Wider coverage but much poorer family agreement; do not replace the retained groups |
| Alpha versus Trace on common queries and candidate bases | Higher full-set accuracy for norms, but exact-tie exclusion reverses overall ordering on a smaller subset; not a robust winner or paper replication |
| Near-match groups held out together | A stronger local prediction check than exact full-vector hash splits |
| Matrix-shape, non-alpha, and combined spectral controls | Spectral signal beyond shapes; no demonstrated alpha-specific gain in the fixed tree comparison |
| Whole weight cluster held out | Harder transfer; linear spectral signal remains, but this is not unseen-family validation |
| Phase 2 to ablation and the reverse | Separate collection-transfer directions, not independent ancestry tests |

MAE means mean absolute error. RMSE means root-mean-square error and makes large
mistakes more visible. Score metrics refer to the explicitly stated target and
population. A six-task average does not have the same error as every task.
Out-of-range predictions must remain visible, not silently clipped.

## Collection and extraction

The Phase 2 list combines graph candidates, sampled metadata, leaderboard rows,
and targeted Hub searches. Ablation is a score-enriched list screened for source
overlap, access, and duplicates. Historical type categories are heuristic
selection buckets. A declared source link is not verified lineage.

Workers reserve allowed GPUs inside the runner and check memory use. This reduces
overlap risk; it is not a cluster-wide reservation against unrelated schedulers.
Independent matrix decompositions can run across GPUs for one model. This may
reduce wall-clock time but does not guarantee fewer GPU-seconds or lower energy.

Per-worker logs, heartbeat and final status support diagnosis. Signals change
allocation settings, drain pending work, or stop workers. Retry rules separate
retryable failures from access and unsupported-model failures. Per-task model
caches are removed on exit, trading more downloads for bounded disk usage.

The collector supports decomposition choices, tail-fitting rules, and optional
eigenvalue output. Current saved alpha uses a custom DKS-style fit, not fixed-tail
Hill. Saved `xmin` is the smallest retained eigenvalue, not the fitted cutoff.
The Trace comparison uses the square root of the saved eigenvalue sum, which can
omit eigenvalues filtered during collection.

## Design and verification

Keep the Berkeley, Dartmouth, and Booking.com palette, the dotted galaxy cover,
phase diagrams, dataset curation/distribution slides, and appendix. Charts show
measured values; decorative galaxies do not encode ancestry distances. Render
the finished PDF and inspect contrast, text fit, diagram connections, and charts.
