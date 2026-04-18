# Lean Mixed-Zoo Benchmark Design

Date: 2026-04-18

## Decision

Add a second, broader benchmark alongside the current strict mixed-zoo cycle.

The current strict benchmark stays in place as the cleaner control benchmark.
The new benchmark is a **lean mixed-zoo benchmark** meant to test broader zoo usefulness when richer catalog fields are missing and heterogeneous models need more support.

This lean benchmark is intentionally noisier than the strict benchmark.
It is a parallel benchmark, not a replacement.

## Why This Benchmark

The current mixed-zoo cycle now measures the right slices, but it exposed a practical limitation:

- the strict control benchmark still has no supported heterogeneous test queries

That means the current cycle is good for honest control results, but not yet good for testing whether the method helps on a larger and more heterogeneous zoo.

We need a broader benchmark that:

- does not depend on `Architecture` or `#Params (B)` as method inputs
- allows a larger supported heterogeneous slice
- still stays grounded in model-intrinsic or directly recoverable structure

## Goal

Test whether spectral methods help organize a broader mixed zoo when:

- architecture and parameter-count fields are not used as method inputs
- heterogeneous supported models may rely on `provisional` labels as well as `confirmed` labels

## Core Principles

1. This is a **parallel** benchmark, not the new headline by default.
2. The strict benchmark remains the cleaner control.
3. The lean benchmark is allowed to be noisier if it materially improves mixed-zoo coverage.
4. Method inputs should remain model-intrinsic or directly derivable from the model record.
5. Do not add new complex methods unless the current ladder clearly fails.

## Label Policy

### Strict benchmark

- supported slices use `confirmed` labels only

### Lean benchmark

- supported slices may use `confirmed` and `provisional` labels
- `unknown` remains unsupported
- special buckets like merge / adapter / quantized remain unsupported

Interpretation:

- `confirmed` means manually trusted
- `provisional` means plausible but not fully trusted

So the lean benchmark trades some label cleanliness for broader coverage.

## Allowed Method Inputs In Lean Mode

Allowed:

- `num_layers`
- `raw_module_count`
- `common_schema`
- `schema_name`
- module-role or schema-presence pattern if easy to derive from the spectral record
- spectral fingerprints from weights

Not allowed as main method inputs:

- `Architecture`
- `#Params (B)`
- leaderboard `Type`
- externally curated catalog annotations not guaranteed to exist in a real zoo

## Benchmark Slices

### Slice A: Homogeneous Supported

- common-schema models
- supported labels may be `confirmed` or `provisional`

Purpose:

- broader homogeneous control without architecture/params shortcuts

### Slice B: Heterogeneous Supported

- non-common-schema models
- supported labels may be `confirmed` or `provisional`

Purpose:

- main utility slice for this benchmark

### Slice C: Unsupported / Special

- `unknown`
- merge
- adapter
- quantized
- other special-case buckets

Purpose:

- stress and rejection support only

## Method Ladder

### Baseline: `structural_basics_only`

Use only lean structural facts:

- `num_layers`
- `raw_module_count`
- `common_schema`
- `schema_name`

If role-presence pattern is already easy to derive from the record, it may be included.

This baseline replaces `intrinsic_basics_only` in lean mode.

### Spectral Methods

Keep the same spectral ladder:

- `pl_alpha_scalar`
- `full_module_trace`
- `shared_role_trace`
- `role_aggregated_trace`

Do not add a new spectral method in this benchmark unless this ladder fails clearly.

## Similarity Logic

Keep it simple and reuse the current retrieval logic where possible.

- `structural_basics_only`: nearest-family retrieval in structural feature space
- `pl_alpha_scalar`: scalar spectral retrieval
- `full_module_trace`: current trace method
- `shared_role_trace`: overlap-aware role matching
- `role_aggregated_trace`: grouped role matching

## Evaluation

Primary metrics:

- top-1 accuracy
- MRR
- Recall@k

Secondary metrics:

- accepted-set precision under simple rejection
- supported-slice counts

The most important comparison is:

`Do spectral methods beat structural_basics_only on the broader lean benchmark, especially on the heterogeneous supported slice?`

## Success Conditions

This lean benchmark is useful if it gives us all of the following:

1. a nontrivial supported heterogeneous slice
2. a broader test than the strict benchmark
3. a fair comparison that does not rely on architecture/params shortcuts
4. evidence about whether the current spectral ladder remains useful in a broader zoo

## Keep / Kill Rules

Keep:

- the strict benchmark as control
- the lean benchmark as the broader parallel benchmark
- the simplest spectral method that performs well on the broader heterogeneous slice

Kill:

- any method input that reintroduces architecture/params shortcuts
- any new complexity added only for the lean benchmark without clear payoff
- any claim that treats the lean benchmark as equally clean as the strict benchmark

## Expected Outcome

At the end of this benchmark, we should know:

1. whether the broader mixed-zoo benchmark yields supported heterogeneous evaluation at all
2. whether spectral methods still add value when architecture/params are removed
3. whether `provisional` support is enough to make heterogeneous testing practical
4. whether we should continue scaling this lean benchmark before moving toward prediction work
