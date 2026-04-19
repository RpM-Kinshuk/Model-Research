# Performance Prediction Run Optimization Design

## Goal

Speed up the current performance-prediction pilot enough to rerun it reliably on the full target set without changing the research question or prediction logic.

The purpose is only to make the prediction experiments cheaper and faster to iterate on.

The optimization should target the real bottleneck:

- repeated pairwise similarity scoring inside nested cross-validation loops

It should not turn into a full rewrite or a side project.

## Scope

Included:

- reuse the current prediction methods and outputs
- run a very small optimization bakeoff
- precompute and cache pairwise similarities once per method
- use GPU acceleration where it is clean and actually useful
- pin the run to GPU `0` by default
- allow up to `3` GPUs only for the pairwise precompute stage if that cleanly shortens the run without extra orchestration

Excluded:

- changing the benchmark or targets
- changing the scoring rules or evaluation logic
- large architectural rewrites
- new prediction methods
- long-lived infrastructure for persistent caches or distributed execution

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

### 3. Test A Small Optimization Ladder

Only test these three approaches:

1. `cpu_cache_only`
- precompute each method matrix once on CPU
- reuse it everywhere else

2. `cache_plus_gpu0`
- same cached design
- use GPU `0` only for spectral pairwise precompute

3. `up_to_3_gpus_if_trivial`
- only split independent precompute work if it is genuinely simple
- if it needs nontrivial orchestration, drop it immediately

The goal is not to optimize everything.
The goal is to find the simplest promising path quickly.

### 4. Use GPU Only For Vectorizable Pairwise Work

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

### 5. Keep Structural Control Simple

Do not force the structural baseline onto GPU.

Keep:

- ridge baseline on CPU
- structural feature construction on CPU

Reason:

- that is not the dominant cost
- moving it to GPU adds complexity without clear gain

### 6. Cache Reusable Artifacts In Memory Per Run

Within one run, cache:

- model order / index mapping
- structural baseline predictions per target
- pairwise similarity matrices per method

No persistent on-disk cache is required for this cycle.

## Execution Policy

Before the optimized rerun:

- terminate stale long-running prediction processes from earlier runs

Then:

- run the small optimization bakeoff
- compare timing and numerical consistency
- keep the simplest promising approach
- rerun the full pilot once with that approach

GPU policy:

- start with `CUDA_VISIBLE_DEVICES=0`
- widen visible devices to at most `3` GPUs only if the precompute implementation actually uses them cleanly and the simpler run is still too slow

The runner should still work correctly on CPU-only fallback if GPU precompute is unavailable.

## Success Criteria

This optimization cycle succeeds if:

1. at least one optimized path runs materially faster than the current repeated-score CPU path
2. the output files remain the same shape and meaning as before
3. the numerical results stay consistent with the earlier committed pilot, up to normal floating-point noise
4. the implementation remains small and understandable
5. we can quickly return to evaluating prediction quality instead of extending the optimization work

## Keep / Kill Rules

Keep:

- cached pairwise-matrix optimization
- GPU precompute only if it is simple and clearly useful
- short timing bakeoff followed by one full rerun

Kill:

- any optimization that changes the experiment definition
- any multi-GPU orchestration beyond a small independent precompute split
- any rewrite that makes the runner much harder to reason about
- any optimization work that starts becoming a separate engineering objective

## Expected Result

The likely outcome is:

- most of the speedup comes from caching and vectorized pairwise precompute
- GPU may help during spectral matrix construction
- CPU still handles the lighter baseline and reporting steps
- the final chosen path is probably `cache_plus_gpu0` or even `cpu_cache_only` if that is already fast enough

That is the right tradeoff for this stage:

- real speedup
- minimal research drift
- minimal code complexity
