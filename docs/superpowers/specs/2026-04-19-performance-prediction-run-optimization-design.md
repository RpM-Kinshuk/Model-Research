# Performance Prediction Run Optimization Design

## Goal

Speed up the current performance-prediction pilot enough to rerun it reliably on the full target set without changing the research question or prediction logic.

The optimization should target the real bottleneck:

- repeated pairwise similarity scoring inside nested cross-validation loops

It should not turn into a full rewrite.

## Scope

Included:

- reuse the current prediction methods and outputs
- precompute and cache pairwise similarities once per method
- use GPU acceleration where it is clean and actually useful
- pin the run to GPU `0` by default
- allow up to `3` GPUs only for the pairwise precompute stage if that cleanly shortens the run

Excluded:

- changing the benchmark or targets
- changing the scoring rules or evaluation logic
- large architectural rewrites
- new prediction methods

## Current Bottleneck

The current runner is slow because it recomputes pairwise model-to-model scores many times.

That repeated work happens across:

- targets
- CV folds
- `k`
- weighted vs unweighted prediction
- raw vs structurally controlled prediction

This means the expensive part is not the grouped regression baseline.
The expensive part is repeated neighborhood scoring.

## Design

### 1. Keep The Research Logic Fixed

The following must stay logically unchanged:

- joined prediction dataset
- targets
- grouped CV behavior
- method list:
  - `structural_basics_only`
  - `pl_alpha_scalar`
  - `full_module_trace`
  - `role_aggregated_trace`
- prediction outputs and report structure

This is a run optimization, not a new experiment.

### 2. Precompute Method-Specific Pairwise Similarities Once

For each supported method, build a dense pairwise similarity matrix once over the joined scored dataset.

Use that matrix for all later:

- folds
- targets
- `k`
- weighted vs unweighted runs
- raw vs residualized runs

This should remove the repeated nested pair-scoring loop from the hot path.

### 3. Use GPU Only For Vectorizable Pairwise Work

Use GPU only where it cleanly helps:

- dense pairwise scoring for:
  - `pl_alpha_scalar`
  - `full_module_trace`
  - `role_aggregated_trace`

Use `torch` tensors for this precompute.

Default device policy:

- use GPU `0` when running the optimized pilot

Allowed multi-GPU policy:

- use at most `3` GPUs
- only for independent method-level or block-level precompute work
- do not introduce distributed training or complex synchronization

If a method or code path does not benefit cleanly from GPU batching, keep it on CPU but still use the cached-matrix path.

### 4. Keep Structural Control Simple

Do not force the structural baseline onto GPU.

Keep:

- ridge baseline on CPU
- structural feature construction on CPU

Reason:

- that is not the dominant cost
- moving it to GPU adds complexity without clear gain

### 5. Cache Reusable Artifacts In Memory Per Run

Within one run, cache:

- model order / index mapping
- structural baseline predictions per target
- pairwise similarity matrices per method

No persistent on-disk cache is required for this cycle.

## Execution Policy

Before the optimized rerun:

- terminate stale long-running prediction processes from earlier runs

Then:

- rerun the full pilot with `CUDA_VISIBLE_DEVICES=0`
- optionally widen visible devices to at most `3` GPUs only if the precompute implementation actually uses them

The runner should still work correctly on CPU-only fallback if GPU precompute is unavailable.

## Success Criteria

This optimization cycle succeeds if:

1. the full prediction pilot completes reliably in a materially shorter time than the current long-running CPU path
2. the output files remain the same shape and meaning as before
3. the numerical results stay consistent with the earlier committed pilot, up to normal floating-point noise
4. the implementation remains small and understandable

## Keep / Kill Rules

Keep:

- cached pairwise-matrix optimization
- GPU precompute only if it is simple and clearly useful

Kill:

- any optimization that changes the experiment definition
- any multi-GPU orchestration beyond a small independent precompute split
- any rewrite that makes the runner much harder to reason about

## Expected Result

The likely outcome is:

- most of the speedup comes from caching and vectorized pairwise precompute
- GPU helps during spectral matrix construction
- CPU still handles the lighter baseline and reporting steps

That is the right tradeoff for this stage:

- real speedup
- minimal research drift
- minimal code complexity
