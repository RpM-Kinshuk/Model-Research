# Spectral Family Retrieval Preliminary Findings

## What This Note Is

This note consolidates the current preliminary work on spectral family retrieval.

The goal of this stage was not to prove every part of the final paper idea.
The goal was to answer one simpler question first:

`Can spectral fingerprints help us identify the base family of a model from its weights alone?`

That is now the main thing we have tested.

## Short Answer

Yes, on the current data, the answer is **yes**.

The strongest result is:

- module-aware spectral traces are clearly better than plain metadata for family retrieval
- the best current method is `full_module_trace`
- `attention_trace` is almost as good and simpler
- rejection improved a little with a better confidence rule, but it is still not strong enough to be a main selling point

## Why We Narrowed The Problem

The original idea was broader: a large model zoo, clustering, lineage hints, and database organization.

That was too broad for a first result.

So we narrowed to:

- `base-family retrieval`
- on current spectral records
- with a clean audited benchmark
- using only model weights and metadata already in the repo

This was the right choice, because it gave us one question that can actually be tested cleanly.

## What We Tested

We compared several simple ways to represent a model.

- `metadata_only`
  - uses simple metadata like architecture and size
- `pl_alpha_scalar`
  - uses a small scalar summary of the spectral signal
- `summary_module`
  - compresses the spectral trace into summary numbers
- `attention_trace`
  - keeps a layer-by-layer spectral trace for attention-related modules
- `full_module_trace`
  - keeps a layer-by-layer spectral trace across more modules

Plain-language interpretation:

- `metadata_only` is the weak baseline
- `pl_alpha_scalar` is the simple spectral baseline
- `attention_trace` and `full_module_trace` are the main methods

## How The Benchmark Was Tightened

We did not trust the first broad benchmark as the final answer.

We tightened it in three stages.

### 1. Broad Curated Slice

This was the first useful filtered benchmark.

- gallery: `355`
- supported queries: `165`
- unsupported queries: `1566`
- heterogeneous queries: `97`
- supported families: `23`

This slice was useful for checking whether the idea had signal at all.

### 2. Root/Base-Only Slice

This was stricter.

Only root or base families were kept as targets.

- gallery: `159`
- supported queries: `76`
- unsupported queries: `1851`
- heterogeneous queries: `97`
- supported families: `12`

This removed a lot of ambiguity from derived or repackaged families.

### 3. Strict Headline Slice

This is the best current benchmark and the one we should treat as the main preliminary result.

It added hard vetoes for:

- derived-family targets like `Instruct`, `Chat`, `sft`, `Preview`
- lexical red flags like `merge`, `prun`, `quant`, `bnb`, `4bit`, `gptq`, `awq`, `gguf`, `lora`, `adapter`, `distill`
- plausible size mismatches
- weak lineage relations

Final strict counts:

- confirmed rows: `225`
- confirmed families: `12`
- gallery: `153`
- supported queries: `72`
- unsupported queries: `1861`
- heterogeneous queries: `97`

This is the cleanest current version of Tier A.

## Main Retrieval Results

The clearest results come from the strict headline slice.

From [retrieval_metrics.csv](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/spectral_family_retrieval_strict_headline_v2/retrieval_metrics.csv):

- `full_module_trace`
  - top-1: `0.918`
  - MRR: `0.950`
- `attention_trace`
  - top-1: `0.918`
  - MRR: `0.946`
- `pl_alpha_scalar`
  - top-1: `0.857`
  - MRR: `0.903`
- `summary_module`
  - top-1: `0.796`
  - MRR: `0.873`
- `metadata_only`
  - top-1: `0.184`
  - MRR: `0.323`

What that means:

- the spectral signal is real
- richer traces beat simple metadata by a large margin
- richer traces also beat simpler compressed summaries

## What Changed As The Benchmark Got Stricter

This is important.

The benchmark became smaller, but the best retrieval results became cleaner and slightly stronger.

Broad curated slice, `all`:

- `full_module_trace`: top-1 `0.720`, MRR `0.833`
- `attention_trace`: top-1 `0.692`, MRR `0.822`

Root/base-only slice, `all`:

- `attention_trace`: top-1 `0.885`, MRR `0.923`
- `full_module_trace`: top-1 `0.885`, MRR `0.922`

Strict headline slice, `all`:

- `full_module_trace`: top-1 `0.918`, MRR `0.950`
- `attention_trace`: top-1 `0.918`, MRR `0.946`

Interpretation:

- the method did not depend on noisy labels
- removing ambiguous families helped
- the strongest story is on the clean root/base benchmark, not the broader noisy one

## What We Learned About The Methods

### `full_module_trace`

This is the current best method.

It is the strongest choice for the paper’s main retrieval result.

### `attention_trace`

This is nearly as good.

It is useful as:

- a simpler backup method
- a strong ablation

### `pl_alpha_scalar`

This is a very useful baseline.

It shows that even a small scalar spectral summary carries real signal.

But it is not the best method.

### `summary_module`

This is useful as a compressed comparison.

It is clearly weaker than the full trace methods.

### `metadata_only`

This is much weaker on the main retrieval task.

That is good news, because it means the spectral methods are doing real work.

## Rejection Results Before The Mini-Study

Rejection means:

- when a query does not really belong to the supported family set
- the system should know that it is not trustworthy

Before the mini-study, rejection was the weak part.

On the strict headline slice, from [rejection_metrics.csv](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/spectral_family_retrieval_strict_headline_v2/rejection_metrics.csv):

- `pl_alpha_scalar` AUROC: `0.549`
- `attention_trace` AUROC: `0.538`
- `full_module_trace` AUROC: `0.527`
- `metadata_only` AUROC: `0.475`

This is not strong enough for a headline claim.

## Rejection Mini-Study

We then ran one small follow-up study.

We did **not** add new fingerprints.
We only changed the confidence rule used to decide whether a guess looks trustworthy.

We compared:

- `top_score`
  - trust the guess if the best score is high
- `margin`
  - trust the guess if the best score is clearly better than the second-best
- `score_plus_margin`
  - a simple combination of both

Study outputs:

- [confidence_rule_metrics.csv](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/rejection_confidence_study_strict_v1/confidence_rule_metrics.csv)
- [best_rule_bucket_breakdown.csv](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/rejection_confidence_study_strict_v1/best_rule_bucket_breakdown.csv)
- [report.md](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/rejection_confidence_study_strict_v1/report.md)

### Main Mini-Study Result

For the two main trace methods, `score_plus_margin` was best.

For `full_module_trace`, `all` slice:

- `top_score` AUROC: `0.559`
- `margin` AUROC: `0.583`
- `score_plus_margin` AUROC: `0.598`

For `attention_trace`, `all` slice:

- `top_score` AUROC: `0.561`
- `margin` AUROC: `0.565`
- `score_plus_margin` AUROC: `0.583`

Interpretation:

- ambiguity between the top two families matters
- the confidence rule improved a bit
- but the improvement is still small

### Bucket-Level Result

The confidence rule works much better on heterogeneous models than on the realistic unknown bucket.

For example:

- `full_module_trace + score_plus_margin`
  - `unsupported_unknown` AUROC: `0.585`
  - `unsupported_special` AUROC: `0.574`
  - `heterogeneous` AUROC: `0.831`

This means:

- the system can notice some more obvious out-of-distribution cases
- but it still struggles on the messy, ambiguous unknowns

## Benchmark Expansion Cycle 1

We then tested one minimal benchmark-expansion pass without changing the retrieval methods.

First, we reran the strict benchmark with `min_family_size=5` but kept the same strict freeze.
That changed nothing.
So the family cutoff by itself was **not** the real bottleneck.

Next, we created an experimental derived freeze that promoted only three near-miss official base families:

- `mistralai/Mistral-7B-v0.3`
- `mistralai/Mistral-Small-24B-Base-2501`
- `Qwen/Qwen2.5-1.5B`

That expansion changed the benchmark from:

- `12` to `15` confirmed families
- `225` to `241` confirmed rows
- `153` to `165` gallery rows
- `72` to `76` supported queries

But the retrieval quality dropped enough that this is **not** a better headline benchmark.

Main `all`-slice deltas:

- `full_module_trace`: top-1 `0.918 -> 0.881`, MRR `0.950 -> 0.914`
- `attention_trace`: top-1 `0.918 -> 0.881`, MRR `0.946 -> 0.914`
- `pl_alpha_scalar`: top-1 `0.857 -> 0.786`, MRR `0.903 -> 0.852`

Interpretation:

- the expansion is real and worth keeping as an exploratory artifact
- it does **not** replace the current strict headline benchmark
- the next higher-value benchmark task is metadata cleanup on the confirmed strict slice, not lowering the support bar further

## Metadata Cleanup Cycle 1

We then backfilled the missing metadata on the confirmed strict slice.

The backfill was fully local:

- we kept the same strict freeze
- we kept the same family set
- we filled missing `Architecture`, `#Params (B)`, `Generation`, and `Type`
- the values came from the official base-family rows already present in `sampled_leaderboard.csv`

This changed metadata coverage from:

- `41 / 225` confirmed rows with architecture metadata
- to `225 / 225`

The mechanical result was good:

- the matched slices now cover the full supported query set instead of only `17` queries
- the main spectral retrieval results stayed stable

For example:

- `full_module_trace`, all slice: unchanged at top-1 `0.918`, MRR `0.950`
- `attention_trace`, all slice: unchanged at top-1 `0.918`, MRR `0.946`

But the cleanup also exposed a benchmark limitation:

- `metadata_only` became extremely strong after backfill
- it jumped from top-1 `0.184`, MRR `0.323`
- to top-1 `0.918`, MRR `0.959`

Interpretation:

- the cleanup itself was correct and useful
- on this narrow strict root/base benchmark, canonical architecture plus parameter size already identify the family very well
- so the metadata-complete benchmark is better documented, but it no longer supports a strong claim that spectral fingerprints clearly beat metadata

This changes the next priority:

- not more metadata filling
- not more family expansion
- instead, stronger metadata-controlled evaluation on the same retrieval task

## Mixed-Zoo Lean Benchmark

We then added a broader mixed-zoo benchmark to stop overfitting the story to the narrow strict root/base slice.

This lean benchmark changes two things:

- it removes `Architecture` and `#Params (B)` from the baseline inputs
- it allows `confirmed + provisional` supported rows, especially for heterogeneous models

The point was not to replace the strict benchmark.
The point was to add a broader zoo-oriented probe while keeping the strict benchmark as a clean control.

The lean benchmark uses:

- `structural_basics_only`
  - model-intrinsic structure only: `num_layers`, `raw_module_count`, `common_schema`, `schema_name`
- `pl_alpha_scalar`
- `full_module_trace`
- `shared_role_trace`
- `role_aggregated_trace`

The crucial curation fix was simple:

- homogeneous supported rows come from `supported_candidate`
- heterogeneous supported rows come from the `heterogeneous` bucket itself

That produced a real wider heterogeneous slice instead of a fake zero:

- gallery supported: `184`
- supported queries: `86`
- homogeneous test queries: `58`
- heterogeneous test queries: `14`
- heterogeneous supported pool before split: `45`
- unsupported special: `312`
- unsupported unknown: `1549`

The heterogeneous-supported results are the most important part:

- `structural_basics_only`: top-1 `0.714`, MRR `0.845`
- `pl_alpha_scalar`: top-1 `0.786`, MRR `0.893`
- `full_module_trace`: top-1 `0.857`, MRR `0.907`
- `shared_role_trace`: top-1 `0.857`, MRR `0.907`
- `role_aggregated_trace`: top-1 `0.929`, MRR `0.964`

Interpretation:

- we now have a real broader heterogeneous test slice
- spectral methods still add value beyond simple structural basics
- the current best heterogeneity-capable method is `role_aggregated_trace`
- the strict benchmark should now be treated as the clean control, while the lean benchmark is the broader mixed-zoo probe

## Performance Prediction Cycle 1

We then ran one small performance-prediction pilot on the current joined scored subset.

Pilot outputs:

- [prediction_dataset.csv](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_prediction_cycle1/prediction_dataset.csv)
- [prediction_results.csv](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_prediction_cycle1/prediction_results.csv)
- [oof_predictions.csv](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_prediction_cycle1/oof_predictions.csv)
- [report.md](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_prediction_cycle1/report.md)

The prediction dataset had:

- `374` scored models
- `257` grouped CV units
- query buckets: `41` supported-candidate, `97` unsupported-special, `215` unsupported-unknown, `21` heterogeneous
- targets: `Average ⬆️`, `MMLU-PRO`, `MATH Lvl 5`

The strongest rows all came from raw neighborhood prediction on the richer spectral traces.

Best observed rows:

- `Average ⬆️`, `unsupported_special`
  - `full_module_trace`, raw, `k=5`, weighted
  - R² `0.703`, MAE `2.933`, Spearman `0.820`
  - structural baseline on the same slice: R² `-0.024`, Spearman `0.074`
- `MMLU-PRO`, `supported_candidate`
  - `full_module_trace`, raw, `k=3`, weighted
  - R² `0.831`, MAE `3.209`, Spearman `0.894`
  - structural baseline on the same slice: R² `0.354`, Spearman `0.351`
- `MMLU-PRO`, `unsupported_special`
  - `full_module_trace`, raw, `k=5`, weighted
  - R² `0.806`, MAE `2.590`, Spearman `0.872`
  - structural baseline on the same slice: R² `0.172`, Spearman `0.158`
- `MATH Lvl 5`, `unsupported_special`
  - `full_module_trace`, raw, `k=5`, weighted
  - R² `0.779`, MAE `4.544`, Spearman `0.848`
  - structural baseline on the same slice: R² `-0.100`, Spearman `-0.013`
- `MATH Lvl 5`, `unsupported_unknown`
  - `role_aggregated_trace`, raw, `k=5`, unweighted
  - R² `0.480`, MAE `5.279`, Spearman `0.760`
  - structural baseline on the same slice: R² `-0.241`, Spearman `-0.255`

Overall pattern:

- `full_module_trace` produced the best row in `7 / 12` target-bucket slices
- `role_aggregated_trace` produced the best row in the other `5 / 12`
- raw spectral prediction produced the best row in `10 / 12` slices
- structurally controlled spectral prediction produced the best row in only `2 / 12` slices
- `pl_alpha_scalar` never produced the best row

### Prediction Run Optimization Note

This pilot was then rerun once as a pure run-cost optimization, not as a new experiment.

- tested `repeated_cpu` (`7:33.59`), `cpu_cache_only` (`0:45.37`), and `cache_plus_gpu0` on GPU 0 (`0:36.39`) on the single-target `Average ⬆️` bakeoff
- kept `cpu_cache_only` because it was already materially faster than `repeated_cpu` and the extra GPU gain was modest relative to the added complexity
- `cpu_cache_only` matched the earlier pilot exactly on summary metrics and OOF predictions; `cache_plus_gpu0` also matched up to tiny floating-point noise
- audit trail for the timing decision: [repeated bakeoff provenance](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_prediction_bakeoff_repeated/provenance.md), [cpu-cache bakeoff provenance](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_prediction_bakeoff_cpu_cache/provenance.md), [gpu0 bakeoff provenance](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_prediction_bakeoff_gpu0/provenance.md), [optimized final-run provenance](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_prediction_cycle1_optimized/provenance.md)

Interpretation:

- the current representation does carry real predictive signal
- that signal is strong enough to beat the structural baseline by a large margin on several slices
- the strongest current method family is still the richer trace representation, not the scalar summary
- but this is **not** a solved or uniform result
- the heterogeneous slice is still weak: best R² stayed negative on all three targets (`Average ⬆️`: `-0.664`, `MMLU-PRO`: `-0.649`, `MATH Lvl 5`: `-0.640`)

So prediction should stay as an active side direction, but not as a headline claim yet.

## Performance Prediction Cycle 2

We then ran one Cycle 2 follow-up with the kept optimized mode `cpu_cache_only` and added `schema_routed_trace`.

Cycle 2 outputs:

- [prediction_dataset.csv](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_prediction_cycle2/prediction_dataset.csv)
- [prediction_results.csv](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_prediction_cycle2/prediction_results.csv)
- [oof_predictions.csv](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_prediction_cycle2/oof_predictions.csv)
- [report.md](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_prediction_cycle2/report.md)

Interpretation stayed centered on `MMLU-PRO` and `MATH Lvl 5`.
`Average ⬆️` stayed only a sanity target.

Compared with Cycle 1, the heterogeneous slice changed only partially:

- `MATH Lvl 5`, heterogeneous: the best row improved from `role_aggregated_trace` raw `k=3` unweighted, R² `-0.640`, Spearman `0.464`, to `schema_routed_trace` raw `k=5` unweighted, R² `-0.255`, Spearman `0.488`
- `MMLU-PRO`, heterogeneous: `schema_routed_trace` did **not** beat the Cycle 1 `full_module_trace` best row; the best routed row reached R² `-0.796`, Spearman `0.121`, versus Cycle 1 `full_module_trace` at R² `-0.649`, Spearman `0.630`
- on the stronger `supported_candidate` and `unsupported_special` slices for `MMLU-PRO` and `MATH Lvl 5`, `full_module_trace` stayed best in Cycle 2, so the routed method did not produce a clear regression there
- outside the main emphasis, `schema_routed_trace` did become the best `MMLU-PRO` row on `unsupported_unknown`, improving the best R² from `0.584` in Cycle 1 to `0.602` in Cycle 2

## Performance Trend Diagnostics

Outputs: [report.md](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_trend_diagnostics/report.md), [trend_results.csv](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_trend_diagnostics/trend_results.csv), [absolute_trend_dataset.csv](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_trend_diagnostics/absolute_trend_dataset.csv), [exact_parent_delta_dataset.csv](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_trend_diagnostics/exact_parent_delta_dataset.csv), [family_relative_delta_dataset.csv](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_trend_diagnostics/family_relative_delta_dataset.csv).

`structural_adjusted` means adjusted/partial Spearman against the structural controls.

The managed `supported_candidate` slice should be read first. It now shows the clearest absolute-trend associations: structural-adjusted `trend_global_mean` vs `Average ⬆️` (`0.604`, `n=41`), plus `trend_global_start` vs `MMLU-PRO` (`0.646`, `n=41`) and `MATH Lvl 5` (`0.649`, `n=41`).

Exact-parent support exists on that managed slice but remains small-sample: the best `supported_candidate` row is `MATH Lvl 5_delta` on `trend_global_start_delta` (`0.575`, `n=13`).

Family-relative support on the managed slice is weak/noisy: the best `supported_candidate` rows are only `0.179` for `Average ⬆️_delta`, `0.099` for `MATH Lvl 5_delta`, and `-0.163` for `MMLU-PRO_delta` (`n=41`).

`heterogeneous` remains useful as broader context and still produces larger-magnitude small-sample rows, but `all_scored` is a mixed-zoo aggregate (`41` supported-candidate, `21` heterogeneous, `97` unsupported-special, `215` unsupported-unknown) and is dominated by unsupported buckets, so it should stay secondary.

Trend-style signals should stay only as a secondary feature family. The rerun is coherent enough to keep them in the background, but not strong enough to move them ahead of the zoo-management-first focus.

## What We Can Claim Now

We can claim this:

`Module-aware spectral traces are effective for base-family retrieval on a clean audited benchmark.`

We can also say this:

`A simple confidence rule that uses both best score and top-two margin improves rejection slightly for the trace methods.`

And after metadata cleanup, we can say this:

`On the metadata-complete version of this narrow strict benchmark, metadata itself becomes a very strong control, so the next discriminating evaluation must control metadata more tightly.`

And after the lean mixed-zoo run, we can also say this:

`A broader heterogeneous test slice is now viable, and a simple heterogeneity-aware spectral method can perform strongly on it.`

## What We Should Not Claim Yet

We should **not** claim this yet:

- that rejection is solved
- that open-world identification is solved
- that the whole model-zoo organization problem is solved
- that one simple scalar predicts everything

## Best Current Paper Framing

The strongest framing right now is:

`model-intrinsic spectral retrieval for organizing a mixed zoo of models`

Operationally, that means:

- strict benchmark = clean homogeneous control
- lean benchmark = broader mixed-zoo probe
- current best heterogeneous method = `role_aggregated_trace`

The rejection result should be treated as:

- a short support result
- one table or one small subsection
- not the center of the story

## Most Important Artifacts

Strict benchmark:

- [spectral_family_retrieval_label_freeze.csv](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/data/manual_audits/spectral_family_retrieval_label_freeze.csv)
- [report.md](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/spectral_family_retrieval_strict_headline_v2/report.md)

Rejection mini-study:

- [report.md](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/rejection_confidence_study_strict_v1/report.md)

Lean mixed-zoo benchmark:

- [report.md](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/lean_mixed_zoo_benchmark/report.md)
- [retrieval_metrics.csv](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/lean_mixed_zoo_benchmark/retrieval_metrics.csv)

Implementation history:

- `abd1f5c` `feat: run audited spectral family retrieval preliminaries`
- `5dd4731` `feat: add rejection confidence helpers`
- `f54bfd2` `feat: add rejection confidence study script`
- `1478bd8` `run rejection confidence mini study`

## Bottom Line

The preliminary stage succeeded.

The main retrieval story is now believable.
The benchmark is much cleaner than it was at the start.
The rejection story improved a little, but it is still a side result.
And we now have a real heterogeneous probe instead of only a homogeneous control.

So the next move should be:

- consolidate this into the paper outline
- keep the center on model-zoo retrieval and organization
- use the strict benchmark as the control and the lean benchmark as the broader probe
- avoid expanding into too many new side experiments before the writing is stable
