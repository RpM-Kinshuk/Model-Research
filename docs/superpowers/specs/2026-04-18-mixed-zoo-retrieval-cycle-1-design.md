# Mixed-Zoo Retrieval Cycle 1 Design

Date: 2026-04-18

## Decision

The next cycle should move from a narrow clean-family benchmark toward a **mixed-zoo retrieval study**.

The homogeneous strict benchmark remains useful, but only as a control.
The main target now is a method that still makes sense when the zoo contains a mixture of:

- common-schema models
- heterogeneous-schema models
- weak-label and special-case models

The method direction should remain **model-intrinsic** and **zoo-scalable**.

## Why This Cycle

The preliminary study answered one important question:

`Do spectral fingerprints contain real family-retrieval signal at all?`

The answer was yes.

But the same study also exposed two limits:

1. The clean strict benchmark is too narrow to stand in for a real model zoo.
2. External or prefilled metadata can become too strong on that narrow benchmark, which drifts away from the intended use case.

So the next cycle should not optimize the narrow benchmark further.
It should test whether the method still helps on a more realistic mixed zoo.

## Goal

Test whether a model-intrinsic spectral representation helps organize a mixed zoo of models better than simple model-intrinsic structural facts.

This cycle is still retrieval-first.
It is **not** yet a full atlas paper and **not** yet a full prediction paper.

## Core Principles

1. The north star remains **managing a zoo of models**.
2. Heterogeneity is a **design target**, not an appendix.
3. Method inputs should come from the model itself whenever possible.
4. Do not add complexity unless it clearly improves mixed-zoo usefulness.
5. Keep the prediction direction alive, but only as a small bridge if the mixed-zoo representation is already meaningful.

## What Counts As A Fair Input

### Allowed as method inputs

- architecture class recoverable from the model
- parameter count recoverable from the model
- layer count
- module/schema pattern
- spectral fingerprints computed from weights

### Not allowed as main method inputs

- externally curated leaderboard `Type`
- prefilled release-era annotations
- catalog metadata that may be absent in a real zoo setting

These can still be used for:

- audit
- benchmark construction
- analysis

But they should not define the main zoo-management method.

## Main Question

Can a model-intrinsic spectral representation support useful retrieval and organization in a mixed zoo containing both homogeneous and heterogeneous models?

## Benchmark Design

Use three slices.

### Slice A: Homogeneous Control

Purpose:

- preserve continuity with the preliminary study
- verify that the spectral signal is still present under the same clean conditions

Composition:

- common-schema supported models
- same broad family-retrieval setup as before

This slice is a control, not the destination.

### Slice B: Heterogeneous Supported Slice

Purpose:

- primary utility test

Composition:

- models with non-common schemas or partial schema mismatch
- only cases with supportable family labels

This is the main new benchmark slice.

### Slice C: Unsupported / Special Slice

Purpose:

- stress test and rejection support

Composition:

- weak-label models
- merge models
- quantized models
- adapter models
- other special cases

This slice is not the headline metric, but it helps test whether the method remains usable in a real zoo.

## Method Ladder

Do not commit to one heterogeneity method before testing.
Run a small bakeoff and keep the simplest winner.

### Baseline 1: `intrinsic_basics_only`

Features recoverable from the model itself:

- architecture class
- parameter count
- layer count
- schema/module pattern

This replaces the broader metadata-heavy baseline.

### Baseline 2: `pl_alpha_scalar`

- one simple scalar spectral summary

This tests whether trace structure matters.

### Method 1: `full_module_trace`

- current module-aware trace method
- tested unchanged on both homogeneous and heterogeneous slices

This gets the first chance to generalize.

### Method 2: `shared_role_trace`

- compare only canonical roles present in both models
- do not force exact full-schema alignment

This is the simplest heterogeneity-aware variant.

### Method 3: `role_aggregated_trace`

- group modules into broader role categories when exact role matching is weak
- example: broader attention-role and MLP-role groupings

This should only survive if it clearly helps.

## Similarity Logic

Keep the similarity simple.

### For homogeneous models

- use the current trace comparison

### For heterogeneous models

- compare only shared canonical roles
- track overlap explicitly
- penalize low overlap instead of discarding the model immediately

Do not add learned alignment, pairwise neuron matching, or large hybrid systems in this cycle.

## Evaluation

### Primary metrics

- top-1 accuracy
- MRR
- Recall@k

### Secondary metrics

- accepted-set precision under a simple rejection rule
- neighborhood sanity checks

### Key comparison question

Does spectral information add value beyond `intrinsic_basics_only` on:

- the homogeneous control slice
- the heterogeneous supported slice

## Hypotheses

### H1

`full_module_trace` should beat `intrinsic_basics_only` on the homogeneous control slice.

### H2

At least one spectral method should remain useful on the heterogeneous slice.
If heterogeneity-aware variants help, they must beat the current method cleanly enough to justify the added complexity.

### H3

A useful mixed-zoo neighborhood structure should exist, not just easy family recovery on the clean slice.

### H4

Only if H1-H3 hold, neighborhood-based performance prediction becomes plausible enough for a small pilot.

## Keep / Kill Rules

Keep:

- the homogeneous slice as a control
- the simplest spectral method that works on the heterogeneous slice

Kill:

- any method that depends on external prefilled catalog metadata
- any method that only works on homogeneous models
- any heterogeneity-aware variant that does not beat the current method clearly enough
- any complex combination method that adds moving parts without mixed-zoo benefit

## Prediction Bridge

Do not do a full prediction project yet.

Only if the mixed-zoo retrieval neighborhoods look meaningful, run one tiny pilot:

- neighborhood-based prediction of one or two benchmark aggregates on the scored subset

The point is not to claim prediction is solved.
The point is to test whether the zoo representation carries enough structure to make prediction plausible.

## Outcome We Want From This Cycle

At the end of the cycle, we should know:

1. whether the current spectral method survives contact with heterogeneous models
2. whether a simple heterogeneity-aware variant is needed
3. whether the spectral representation adds value beyond intrinsic basics in a real mixed zoo
4. whether the zoo neighborhoods are strong enough to justify a small prediction pilot next
