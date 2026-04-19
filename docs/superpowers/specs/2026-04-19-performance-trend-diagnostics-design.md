# Performance Trend Diagnostics Design

## Goal

Run one small diagnostic cycle that asks:

`do simple spectral trend signals correlate with benchmark performance in a way that is useful for mixed-zoo analysis?`

This is not a new prediction stack.
This is a compact diagnostic that checks whether trend-style signals are promising enough to justify later use in prediction work.

## Why This Is The Right Next Step

The current prediction work already showed:

- richer spectral neighborhoods carry real signal on several slices
- heterogeneous prediction is still the weak point
- one more large prediction-method cycle would likely turn into side work

The user also asked to remember two specific directions:

- `absolute trend`
- `lineage-relative trend`

So the next step should test those directly and separately, without folding them into another large model.

## Scope

Included:

- one small `absolute trend` diagnostic
- one small `exact-parent trend` diagnostic
- one small `family-relative trend` diagnostic
- the same three targets used in the current prediction work
- simple correlation and tiny linear-fit checks only

Excluded:

- any new large predictor stack
- deep models
- broad benchmark sweeps
- many new feature families
- claim inflation around general performance prediction

## Data

Use the same joined scored dataset already used in the prediction cycles.

That table already carries the minimal fields needed here:

- `model_id`
- `source_model`
- `base_model_relation`
- `num_layers`
- `raw_module_count`
- `common_schema`
- `schema_name`
- `audited_base_family`
- `query_bucket`
- `Average ⬆️`
- `MMLU-PRO`
- `MATH Lvl 5`

This keeps the trend analysis directly tied to the current zoo benchmark rather than creating a new data path.

## Targets

Keep the same three targets:

- `Average ⬆️`
- `MMLU-PRO`
- `MATH Lvl 5`

Interpret them the same way as before:

- `Average ⬆️` is a supporting sanity target
- `MMLU-PRO` is the main broad target
- `MATH Lvl 5` is the main specialized target

## Trend Signals

Keep the feature set intentionally small and based on information we can already derive from current spectral summaries.

### 1. Absolute Trend Features

Use only a compact set of per-model spectral summaries:

- overall level
- start-of-depth level
- end-of-depth level
- end-minus-start

These should be computed at the smallest useful granularity:

- global or across-module aggregate
- optionally broad role-group aggregates like `attention` and `mlp` if they are already easy to derive from current canonical summaries

The rule is:

`prefer existing simple summaries over inventing new features`

### 2. Exact-Parent Relative Trend

For models with a trustworthy direct source model in the scored set:

- compute feature delta relative to the direct parent
- compute target delta relative to the direct parent

This is the strictest lineage-relative analysis.

It asks:

`when a model changes relative to its direct ancestor, does performance tend to change in step with simple spectral trend changes?`

### 3. Family-Relative Trend

When exact parent is missing, use a broader family-relative fallback.

For each family:

- build a simple family reference point from scored members
- use a stable reference like the family median for each trend feature and each target

Then compute:

- feature delta from the family reference
- target delta from the family reference

This is broader and noisier than exact-parent analysis, so it must be reported separately.

## Analysis Structure

Run three separate diagnostics.

### A. Absolute-Trend Analysis

For each absolute trend feature and each target:

- compute simple rank correlation
- compute a tiny one-feature linear fit
- optionally compute a structure-adjusted version using only the same lightweight structural controls already available from the joined table

This analysis asks whether absolute spectral position or direction carries performance signal.

### B. Exact-Parent Trend Analysis

For each model-parent pair that passes the lineage filter:

- compute feature delta
- compute target delta

Then report correlation and tiny one-feature fit results on these deltas.

This analysis should stay strict:

- only use exact parent links that are actually available
- do not silently fall back to family-relative inside this analysis

### C. Family-Relative Trend Analysis

For each scored model with a usable family reference:

- compute delta from family reference for each feature
- compute delta from family reference for each target

Then report the same compact statistics.

This is not a replacement for exact-parent analysis.
It is the broader zoo-style fallback when precise lineage is unavailable.

## Buckets

Do not create a huge reporting grid.

Primary reporting buckets:

- `all_scored`
- `heterogeneous`

Optional third bucket only if counts are healthy:

- one broad non-heterogeneous bucket, likely `supported_candidate` or pooled non-heterogeneous scored rows

If a bucket is too small for a stable statistic, report the count and skip the metric rather than forcing a noisy number.

## Statistics

Keep the statistics simple and interpretable.

Primary statistics:

- `Spearman`
- `R²`
- `MAE` only if it adds value in the tiny fits
- pair counts / sample counts

The emphasis should be:

- direction of association
- stability across the main targets
- whether the signal survives in the heterogeneous bucket

This is a diagnostic cycle, so interpretability matters more than squeezing out small numeric gains.

## Controls

Controls should stay minimal.

Allowed controls:

- no-control raw trend analysis
- one lightweight structure-adjusted variant using fields like `num_layers`, `raw_module_count`, `common_schema`, and `schema_name`

Disallowed:

- heavy multivariate stacks
- adding retrieval neighborhoods into this cycle
- mixing trend signals into the current prediction models

## Success Criteria

This cycle is useful if at least one simple trend signal:

- shows a stable directional relationship with `MMLU-PRO` or `MATH Lvl 5`
- and does so in either:
  - the strict exact-parent analysis
  - or the broader family-relative analysis

Better outcome:

- the same trend signal is useful in both the strict and broad analyses
- and the heterogeneous bucket does not collapse completely

Negative but still useful outcome:

- no simple trend signal is stable enough

That would tell us not to spend more time on trend-style prediction features right now.

## Keep / Kill Rules

Keep:

- any trend feature that is directionally stable on one of the two main targets
- exact-parent and family-relative results as separate evidence streams

Kill:

- any attempt to merge exact-parent and family-relative into one pooled headline metric
- any rapid expansion into many extra feature summaries
- any move to turn this into a new predictor stack before the diagnostics show real promise

## Outputs

This cycle should produce:

- one compact table of the strongest absolute-trend rows
- one compact table of the strongest exact-parent delta rows
- one compact table of the strongest family-relative delta rows
- one short report interpreting whether trend-style signals look promising enough for later prediction work

## Expected Interpretation

If the cycle works:

- we learn that simple spectral trend summaries carry useful performance information
- we can later consider adding one or two of them into a small prediction follow-up

If the cycle does not work:

- we keep the current neighborhood-based prediction direction as the stronger path
- we record trend diagnostics as a negative or weak result and move on

## Decision After This Cycle

If the results are promising:

- do one small follow-up that adds only the strongest trend signal into a later prediction comparison

If the results are weak:

- stop the trend branch
- keep the center on zoo organization and neighborhood-based prediction

That keeps the research aligned with the main goal:

`understand and manage a mixed zoo of models without turning every idea into a side project`
