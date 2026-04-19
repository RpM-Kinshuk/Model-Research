# Performance Prediction Cycle 2 Design

## Goal

Run one small follow-up prediction cycle that stays tied to the model-zoo goal:

`can a simple heterogeneity-aware neighborhood strategy improve prediction where the current plain trace methods weaken, without hurting the strong slices too much?`

This is not a broad benchmark sweep.
It is a focused check on whether mixed-zoo prediction can be improved in a principled, simple way.

## Why This Is The Right Next Step

Cycle 1 established three useful facts:

- spectral neighborhoods carry real performance signal on several slices
- `full_module_trace` is strongest on many clean and broad slices
- `role_aggregated_trace` becomes relatively more useful outside the easiest settings, but the heterogeneous slice is still weak overall

So the next question is not:

`can we build a bigger prediction system?`

It is:

`can we bridge homogeneous and heterogeneous prediction with one small heterogeneity-aware prediction rule?`

That remains aligned with the north star:

- organize and work with a large mixed zoo of models
- support both homogeneous and heterogeneous mixtures
- move toward useful performance estimation without overclaiming

## Scope

Included:

- the same joined scored dataset
- the same three targets
- the same grouped-CV evaluation
- the current method ladder
- one small new routed prediction variant

Excluded:

- wide benchmark sweeps
- many new predictor families
- deep supervised models
- claim inflation around general benchmark completion

## Dataset

Use the same joined scored dataset from Cycle 1.

This keeps the comparison honest and avoids mixing dataset effects with method effects.

The analysis should still report bucket breakdowns for:

- `supported_candidate`
- `heterogeneous`
- `unsupported_special`
- `unsupported_unknown`

## Targets

Keep the same three targets:

- `Average ⬆️`
- `MMLU-PRO`
- `MATH Lvl 5`

But treat them differently:

- `Average ⬆️` is a supporting sanity target only
- `MMLU-PRO` is the main broad-capability target
- `MATH Lvl 5` is the main specialized stress target

This keeps the study broad enough to matter, but prevents the aggregate score from dominating the interpretation.

## Methods

### 1. Structural Control

Keep:

- `structural_basics_only`

This remains the non-spectral control.

### 2. Existing Spectral Methods

Keep:

- `pl_alpha_scalar`
- `full_module_trace`
- `role_aggregated_trace`

These are already implemented and should remain the reference methods.

### 3. One Small Routed Variant

Add exactly one new prediction method:

- `schema_routed_trace`

Idea:

- prefer `full_module_trace` when it has enough valid neighborhood support for the query
- otherwise fall back to `role_aggregated_trace`

The rule should stay simple and local.
For example, it may route based on:

- whether enough finite `full_module_trace` neighbor scores exist
- or whether exact module/schema agreement is present for enough top candidates

The important part is:

- no learned router
- no big hybrid stack
- no many-parameter combination

This is a practical zoo rule:

`use the sharper exact trace when the schema supports it, and use the broader role-aggregated trace when it does not`

### 4. Structural-Control Variant

As in Cycle 1, evaluate the spectral methods:

- without structural control
- with structural control

But this remains secondary.
Cycle 1 already suggested that raw neighborhood prediction is usually the stronger path.

## Evaluation

Use the same grouped cross-validation setup as Cycle 1.

Primary metrics:

- `R²`
- `MAE`
- `Spearman`

Primary reporting:

- overall by target
- bucket breakdowns by target

Main interpretation emphasis:

- `MMLU-PRO`
- `MATH Lvl 5`

`Average ⬆️` should be reported, but not allowed to carry the whole conclusion by itself.

## Main Questions

Cycle 2 is successful if it answers these:

1. Can a simple routed trace method improve heterogeneous prediction over both `full_module_trace` and `role_aggregated_trace`?
2. Can it do that without clearly damaging the strong non-heterogeneous slices?
3. Do spectral methods still beat the structural baseline on at least two of the three targets?
4. Is the strongest predictive signal still coming from raw neighborhood structure rather than heavier controls?

## Keep / Kill Rules

Keep:

- any method that beats the structural baseline on at least two of the three targets
- `schema_routed_trace` only if it clearly improves heterogeneous prediction on at least one main target (`MMLU-PRO` or `MATH Lvl 5`) without a large regression elsewhere

Kill:

- any routed or combined method that only gives tiny gains on `Average ⬆️`
- any routed method that mostly duplicates the better of its components without real improvement
- any expansion beyond this one routed method

## Expected Outcomes

Best useful outcome:

- `schema_routed_trace` improves the heterogeneous slice modestly
- `full_module_trace` remains best on cleaner slices
- this gives us a more believable mixed-zoo prediction story

Useful negative outcome:

- the routed method does not help

That is still informative.
It would mean:

- current retrieval-style representations have predictive value
- but heterogeneity-aware prediction needs a more meaningful representation change, not just routing

## Outputs

Cycle 2 should produce:

- one metrics table for the Cycle 2 method set
- one short comparison note against Cycle 1
- one short report focused on whether heterogeneity-aware prediction improved at all

## Decision After Cycle 2

If Cycle 2 works:

- continue to one more small mixed-zoo prediction cycle
- likely with one carefully chosen representation improvement

If Cycle 2 does not work:

- stop growing the prediction stack
- keep prediction as a promising but limited side result
- return focus to the stronger retrieval/organization story
