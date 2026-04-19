# Performance Prediction Cycle 1 Design

## Goal

Run a small, honest pilot on performance prediction using the current model-zoo representation.

The point is not to build a full prediction system yet.
The point is to test whether the current zoo neighborhoods carry useful benchmark-signal beyond simple structural confounds.

## Why This Is The Right Next Step

The retrieval work has now reached a useful state:

- strict benchmark gives a clean homogeneous control
- lean benchmark gives a broader mixed-zoo probe
- mixed-zoo neighborhoods are strong enough to justify a tiny prediction pilot

So the next question is:

`Do neighborhoods induced by structural or spectral model similarity help predict benchmark performance?`

That is aligned with the north star:

- manage a large zoo of models
- support both homogeneous and heterogeneous mixtures
- move toward benchmark completion or performance estimation

## Scope

This cycle stays deliberately small.

Included:

- one clean score-join dataset
- three targets only
- a small method ladder
- grouped cross-validation
- out-of-fold evaluation

Excluded:

- large multi-target sweeps
- deep supervised models
- complicated hybrids
- claim inflation around benchmark completion

## Dataset

Use the current local leaderboard scores in:

- [sampled_leaderboard.csv](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/data/sampled_leaderboard.csv)

Join them to the current audited mixed-zoo table using:

- primary join: `display_name -> atlas_repo_id`
- fallback join if needed: `display_name -> fullname`

Current local overlap is:

- `388` joined models

Current label mix inside the joined set:

- `41` confirmed
- `119` provisional
- `228` unknown

Current query-bucket mix inside the joined set:

- `41` supported_candidate
- `21` heterogeneous
- `98` unsupported_special
- `228` unsupported_unknown

This is acceptable for Cycle 1 because prediction is no longer tied to the strict retrieval benchmark alone.
The pilot should use the broader joined scored set, while still reporting bucket breakdowns.

## Targets

Use exactly these three targets:

- `Average ⬆️`
- `MMLU-PRO`
- `MATH Lvl 5`

Reason:

- `Average ⬆️` is the broadest coarse target
- `MMLU-PRO` is a strong general capability target
- `MATH Lvl 5` is a useful sharper target because earlier spectral work already hinted at math-related signal

All three are available on all `388` joined rows in the current local file.

## Methods

### 1. Structural Control

Use a simple non-spectral control based on model-intrinsic structure only:

- `num_layers`
- `raw_module_count`
- `common_schema`
- `schema_name`

This is the same spirit as `structural_basics_only`.

### 2. Neighborhood Prediction Main Path

Main prediction method:

- kNN or similarity-weighted neighborhood prediction using the current model representations

Representations to compare:

- `structural_basics_only`
- `pl_alpha_scalar`
- `full_module_trace`
- `role_aggregated_trace`

Prediction rule:

- for each held-out model, find nearest training neighbors
- predict the target as a weighted or unweighted average of neighbor target values

Keep this simple.
Only test a very small set of neighborhood rules, for example:

- `k = 3`
- `k = 5`
- inverse-distance weighted average if clearly stable

### 3. Small Regression Control

Add one simple regression-style control:

- ridge or linear regression on structural features
- optionally plus scalar spectral features

This is not the main story.
It is there to check whether neighborhood prediction is actually useful, or whether a very simple supervised baseline already does as well.

## Evaluation

Use grouped cross-validation.

Primary grouping rule:

- group by `audited_base_family` when available and not empty

Fallback grouping rule:

- if family is unavailable, fall back to `candidate_family`
- if that is also unavailable, fall back to model identity

This keeps the pilot honest by reducing easy leakage across near-duplicates.

Primary metrics:

- `R²`
- `MAE`
- `Spearman`

Report:

- overall results by target
- bucket breakdowns for:
  - `supported_candidate`
  - `heterogeneous`
  - `unsupported_special`
  - `unsupported_unknown`

Do not overcomplicate the analysis beyond that.

## Main Questions

Cycle 1 is successful if it answers these:

1. Does spectral neighborhood prediction beat the structural control?
2. Does `role_aggregated_trace` help on the broader heterogeneous or unsupported buckets?
3. Is there any target where the zoo representation carries useful predictive signal at all?

## Keep / Kill Rules

Keep:

- any spectral neighborhood method that clearly beats the structural control on at least one target
- `role_aggregated_trace` if it remains strongest or most stable outside the clean supported slice

Kill:

- any method that only wins by tiny noise-level margins
- complicated weighting rules that do not beat simple kNN cleanly
- any large hybrid stack that is hard to justify

## Expected Outcomes

Best-case useful outcome:

- `Average ⬆️` and maybe `MMLU-PRO` show modest but real predictive signal
- `MATH Lvl 5` may show the strongest spectral gain

Possible negative outcome:

- prediction is weak across all methods

That is still useful.
It would mean:

- retrieval signal does not automatically translate into benchmark prediction
- we would need a different representation or a more target-aware method later

## Outputs

Cycle 1 should produce:

- one joined prediction dataset summary
- one metrics table by target and method
- one out-of-fold prediction table
- one short report interpreting whether prediction is promising enough to continue

## Decision After Cycle 1

If the pilot works:

- continue to prediction cycle 2 with one or two better-targeted methods

If the pilot is weak:

- keep the result as an honest negative or mixed pilot
- do not force a broad prediction story into the paper

## Summary

This cycle is intentionally small.

It tests the moonshot direction without pretending the problem is solved:

`Can model-zoo neighborhoods derived from spectral structure help estimate benchmark performance on a broader scored subset of models?`
