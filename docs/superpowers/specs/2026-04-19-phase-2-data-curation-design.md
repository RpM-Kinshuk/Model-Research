# Phase 2 Data Curation Design

Date: 2026-04-19

## Goal

Build a larger, cleaner, and more reusable model-zoo dataset for the scale-up phase, without changing the core research direction.

Phase 2 should produce:

- a `15,000`-model curated candidate zoo
- one canonical unified model table
- better availability and recovery tracking
- cleaner enrichment for evaluation and downstream experiments
- a data layer that plugs directly into the existing experiment family

This phase is about **data curation and data shape**, not about inventing new retrieval or prediction methods.
It also does **not** include the later batch spectral-analysis runner.

## Why This Phase Exists

The current preliminary study proved that the research direction is worth scaling, but the data layer still has the same weaknesses as the original one-off collection workflow:

- metadata, leaderboard matches, and analysis status are split across different files
- later experiments depend on joins that are easy to get wrong
- the sample is useful, but still imbalanced across model types
- many models that were curated never became analysis-ready
- availability and failure reasons are not tracked in a reusable way

The next phase should fix those issues while preserving the current sample and the work already done on it.

## Principles

1. Reuse the current dataset first.
   - Do not throw away the existing `5,000`-model sample, current leaderboard matches, local analysis outputs, or manual audit work.

2. Keep the study zoo-oriented.
   - The goal is still a diverse model zoo with enough lineage signal to test retrieval and enough eval coverage to support prediction work.

3. Keep the data shape simple.
   - One canonical model table should become the default input for experiments.

4. Balance diversity without letting noise take over.
   - Adjacent categories like `MoE`, `multimodal`, `adapter`, and `quantized` should be represented, but not allowed to dominate the zoo.

5. Track candidate vs available vs analysis-ready explicitly.
   - A curated model is not the same thing as a fetchable model, and a fetchable model is not the same thing as an analysis-ready model.

6. Prefer recovery over unnecessary resampling.
   - If a model is still public on the Hub, phase 2 should try to recover it before replacing it.

7. Keep curation lightweight on disk.
   - Phase 2 curation should not persistently download large model weights just to test availability.

## Current Baseline

The current local dataset already gives us a useful seed:

- `data/sampled_metadata.csv`: `5,000` curated rows
- `data/sampled_leaderboard.csv`: `936` eval-enriched rows
- local spectral analysis coverage in `svd_results/`: `3,817 / 5,000 = 76.3%`
- known lineage/source signal: `4,135 / 5,000 = 82.7%`
- unknown or weak-lineage rows: `865 / 5,000 = 17.3%`

The current sample is not broken. It is simply too ad hoc and somewhat imbalanced.

Important observations from the current sample:

- lineage coverage is already strong enough for retrieval experiments
- the zoo is heavily weighted toward `finetune` and still has meaningful `merge` coverage
- `adapter`, `quantized`, `MoE`, and `multimodal` are still thin
- eval coverage is useful but limited, especially for prediction work
- many missing analysis rows appear to be recoverable public models, not permanently lost ones

## Scope

Phase 2 includes:

- ingesting the current sample into a unified canonical table
- enlarging the candidate zoo to about `15,000` models
- improving diversity by type without losing lineage utility
- tracking availability and recovery status explicitly
- enriching from the Open LLM Leaderboard and Hugging Face Hub
- preparing the data layer for the scaled-up rerun of the same experiment family

Phase 2 does not include:

- redesigning the main experiment family
- introducing a new benchmark contract for methods
- changing the meaning of the current preliminary findings
- building a large new supervised prediction stack
- implementing or revisiting the batch analysis runner

## Target Dataset Shape

### Two Levels of Dataset

Phase 2 should distinguish two levels clearly:

1. `candidate zoo`
   - all models we intentionally curate into the large zoo
   - target size: about `15,000`

2. `analysis-ready zoo`
   - the subset of the candidate zoo that has already been processed in a later phase and is ready for experiments

Between those two, phase 2 should also track a lighter state:

- `fetchable`
  - models that appear accessible and analyzable based on lightweight Hub checks, without a full persistent download

This split is important because the current failure pattern is not just sampling error. A large share of missing analysis rows appear to still exist publicly and should be recoverable, but curation cannot keep all of them downloaded locally and does not handle the later analysis pass itself.

### Lineage Balance

The candidate zoo should keep a majority of models with some lineage signal, but not become overly narrow.

Recommended overall target:

- `70-80%` known-lineage or weakly-resolved lineage
- `20-30%` unknown or unresolved lineage

This is slightly broader than the current sample while still supporting lineage-oriented evaluation.

### Type-Aware Lineage Balance

The known vs unknown balance should also be considered **inside major type buckets**, but only as a soft target.

For larger buckets such as:

- `base/source`
- `finetune/chat/instruct`
- `merge`
- `quantized`

phase 2 should avoid one-sided buckets that are almost entirely known or almost entirely unknown.

For smaller buckets such as:

- `MoE`
- `multimodal`
- some `adapter` or niche architecture categories

phase 2 should try to include both known and unknown examples when available, but should not enforce rigid ratios.

The key rule is:

- balance by type when practical
- do not let ratio enforcement make the sample brittle

## Type Budgeting Strategy

The scaled-up zoo should use a **banded mixed-zoo budget**.

Recommended top-level budget:

- `65-75%` standard open-weight text-generation models
- `25-35%` adjacent or noisier categories combined

The adjacent/noisier portion should reserve meaningful room for:

- `merge`
- `adapter`
- `quantized`
- `MoE`
- `multimodal`
- `continued pretraining / domain-adapted base`
- a small controlled bucket of architectural outliers

This should be done with **floors and caps**, not exact quotas.

Examples of the intended effect:

- do not let `finetune` absorb almost the entire zoo
- do not let `merge` become the only form of diversity
- do not keep `adapter`, `MoE`, or `multimodal` so rare that they cannot be studied meaningfully
- do not overinflate `quantized` rows if they remain unstable for analysis

## Canonical Model Table

Phase 2 should end in one canonical unified table with one row per model.

Suggested location:

- `data/curated/model_zoo_phase2.csv`
- optionally a parallel Parquet file for faster downstream use

Suggested column groups:

### Identity

- `model_id`
- normalized repo identifier
- display or full name
- revision if applicable

### Lineage

- `source_model`
- `base_model_relation`
- lineage provenance source
- lineage confidence band
- known vs unknown lineage flag

### Type Taxonomy

- primary type bucket
- secondary type tags
- architecture family
- `is_merge`
- `is_adapter`
- `is_quantized`
- `is_moe`
- `is_multimodal`
- continued-pretraining or domain-adaptation flags if available

### Availability and Recovery

- hub availability status
- gated/private/deleted/public flags
- lightweight fetchability probe status
- weight-manifest presence status
- last checked timestamp
- failure class
- retry eligibility

### Hub Metadata

- downloads
- likes
- pipeline tag
- config model type
- parameter count if recoverable

### Analysis Status

- analysis artifact already present
- spectral analysis complete
- local analysis artifact path
- schema name
- common-schema flag
- raw module count
- layer count if available

### Evaluation Enrichment

- leaderboard match status
- leaderboard repo id or eval id
- key benchmark targets
- aggregate score columns

Experiments should eventually read from this canonical table or from small derived views produced from it.

## Data Sources

Phase 2 should use four source layers.

1. `existing local sample`
   - `data/sampled_metadata.csv`
   - current seed sample and lineage metadata

2. `existing local leaderboard enrichment`
   - `data/sampled_leaderboard.csv`
   - current seed eval enrichment

3. `existing local spectral analysis outputs`
   - `svd_results/`
   - current already-processed subset and artifact paths

4. `live Hugging Face and leaderboard enrichment`
   - Hugging Face Hub API
   - Open LLM Leaderboard dataset

The existing data should be ingested first. Live collection should then extend and repair it.

## Lightweight Fetchability Checks

Phase 2 curation should not perform full persistent downloads of model weights.

Instead, it should use lightweight Hub checks to estimate whether a model is likely fetchable and structurally analyzable later.

Recommended lightweight checks:

- `repo_exists` or equivalent repo visibility check
- `model_info` for access state and core metadata
- `list_repo_files` or equivalent file listing
- existence of key files such as:
  - `config.json`
  - safetensor or weight index files when present
- optional dry-run file probe for small files only

The curation outcome should be:

- `public and likely fetchable`
- `gated`
- `private or deleted`
- `public but structurally incomplete`
- `public but probe failed`

This is enough for curation. The full download-and-analysis process belongs to a later phase and is intentionally out of scope here.

## Collection Pipeline

Phase 2 should replace the old three-script workflow with one clearer staged pipeline.

### Stage 1: Seed Ingestion

Ingest the current local sample and analysis state into the canonical table.

This creates a seed zoo with:

- current curated sample rows
- current eval matches
- current local analysis coverage
- current known/unknown lineage pattern

### Stage 2: Candidate Expansion

Expand the zoo toward `15,000` using:

- the atlas-style source metadata already available locally
- leaderboard-derived model references where useful for underrepresented categories
- direct Hugging Face Hub metadata lookups when needed to backfill missing categories

Sampling should remain zoo-oriented rather than benchmark-oriented.

### Stage 3: Type and Lineage Balancing

Apply the soft floors/caps and type-aware lineage balance rules.

The sampler should preserve:

- lineage utility
- type diversity
- architectural diversity
- a meaningful unknown minority

### Stage 4: Enrichment

Enrich rows from:

- Hugging Face Hub API
- Open LLM Leaderboard dataset

This stage should not decide sampling by itself. It should add information to already selected candidates.

### Stage 5: Recovery Preflight

Attempt to improve future analysis coverage using explicit status tracking.

This stage should use:

- lightweight Hub/API probes only
- no persistent full-weight download
- no direct execution of the later analysis runner

Track failures as classes such as:

- `public_fetch_failed`
- `gated`
- `private_or_deleted`
- `config_missing`
- `weight_manifest_missing`
- `download_timeout`
- `analysis_failed`

Public models that fail should stay in the canonical table and remain eligible for retry.
The purpose here is to improve future analysis readiness, not to perform the analysis phase now.

### Stage 6: Derived Views

Materialize small derived views for downstream work, such as:

- `analysis_ready`
- `prediction_ready`
- `known_lineage_only`
- `heterogeneous_focus`

These should be derived outputs, not separate manually curated datasets.

## Recovery Strategy

The current uncovered-model audit suggests that many missing analysis rows are still public on the Hub.

That means phase 2 should prioritize:

- better availability checks
- explicit failure recording
- retryable public failures
- better separation of unavailable models from merely unprocessed ones

Phase 2 should not silently collapse all failures into one bucket.

## Evaluation Enrichment Strategy

The Open LLM Leaderboard should be treated as an **enrichment layer**, not the main zoo sampler.

Reason:

- the zoo should stay broader than the evaluated subset
- performance prediction should use enriched rows where possible
- the main zoo should not become artificially biased toward only leaderboard-covered models

At the same time, prediction will benefit from broader eval coverage than the current sample provides.

So phase 2 should:

- continue joining leaderboard data into the canonical table
- expand that coverage where possible
- expose a `prediction-ready` view for models with usable eval targets

## Repository Phase 2 Data Layout

This phase should define the next data layer of the repo, but keep it simple.

Recommended structure:

- `data/curated/`
  - canonical unified zoo table
  - derived views such as `analysis_ready` and `prediction_ready`
  - taxonomy or mapping files if needed

- `data/manual_audits/`
  - manual overrides and freezes remain here

- `fetch_data/`
  - active collection and enrichment code only
  - preferably one orchestrated pipeline plus a small number of helpers

- `research_runs/phase2/`
  - scaled-up reruns of the same experiment family already used in phase 1

## Success Criteria

Phase 2 is successful if it produces:

1. one canonical model table that replaces the current multi-file join pattern
2. a curated zoo of about `15,000` models
3. explicit availability, fetchability, and recovery status per model
4. better representation of adjacent categories without letting them dominate
5. better eval enrichment for downstream prediction work
6. a clean handoff into the same experiment family used in phase 1 once later phases run

## Non-Goals

Phase 2 should not try to solve everything at once.

It is not meant to:

- make every curated model analysis-ready immediately
- maximize one type such as `merge` or `quantized`
- convert the zoo into a leaderboard-only dataset
- redesign the retrieval or prediction methods before scale-up
- turn data curation into a separate research project

## Recommendation

Use phase 2 to build a unified, balanced, and recovery-aware data layer around the zoo we already have.

The simplest good strategy is:

- ingest the current `5,000`-model sample first
- preserve current analysis results and eval matches
- expand to `15,000` with soft type floors and caps
- maintain a majority of known-lineage rows with a meaningful unknown minority
- track availability, lightweight fetchability, and failure states explicitly
- use leaderboard data as enrichment for prediction, not as the main curator

That gives us a larger zoo without abandoning the preliminary study structure that already worked.
