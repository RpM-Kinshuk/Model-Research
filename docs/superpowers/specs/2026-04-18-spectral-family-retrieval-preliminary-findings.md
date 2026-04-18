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

## What We Can Claim Now

We can claim this:

`Module-aware spectral traces are effective for base-family retrieval on a clean audited benchmark, and they clearly outperform plain metadata baselines.`

We can also say this:

`A simple confidence rule that uses both best score and top-two margin improves rejection slightly for the trace methods.`

## What We Should Not Claim Yet

We should **not** claim this yet:

- that rejection is solved
- that open-world identification is solved
- that the whole model-zoo organization problem is solved
- that one simple scalar predicts everything

## Best Current Paper Framing

The strongest framing right now is:

`module-aware spectral candidate generation for base-family retrieval`

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

So the next move should be:

- consolidate this into the paper outline
- keep the center on retrieval
- avoid expanding into too many new side experiments before the writing is stable
