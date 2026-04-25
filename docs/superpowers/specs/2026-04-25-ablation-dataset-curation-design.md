# Ablation Dataset Curation Design

Date: 2026-04-25

## Goal

Curate a second dataset for ablation studies that complements the phase-2 model zoo rather than duplicating it. The dataset should support both performance-oriented ablations and lineage-oriented experiments using Wasserstein distance, raw eigenvalue statistics, and ESD-derived metrics.

The target size is 5,000 models. The hard quality constraint is eval coverage: no more than 15% of selected models should lack usable evaluation data.

## Principles

- Prefer eval-labelled models over exact phase-2 type proportions.
- Preserve broad model diversity, but do not force rare types if doing so substantially increases missing eval labels or analysis noise.
- Avoid phase-2 overlap by default.
- Keep raw ESD comparability explicit; do not mix model-format effects with lineage effects without labels.
- Reuse the phase-2 schema, loader knowledge, and preflight conventions where possible.
- Do not download model weights during curation.

## Source Priority

1. Open LLM Leaderboard `contents`.
   This is the cleanest source for per-model eval summaries and should be the first seed source. It has roughly 4.58k aggregate rows.

2. Open LLM Leaderboard `results` and details datasets.
   These can supplement the aggregate table when we need more eval-labelled rows, task-level detail, or stronger evidence for performance labels.

3. Hugging Face Hub metadata.
   Use this to enrich model type, architecture, tags, quantization, source/base model hints, availability, files, revision, and sha.

4. Existing phase-2 data.
   Use this for de-duplication and schema alignment, not as a seed source unless an overlap exception is intentionally recorded.

## Selection Targets

- Target rows: 5,000.
- Eval-labelled rows: at least 4,250.
- Eval-missing rows: at most 750.
- Phase-2 duplicates: ideally 0.
- Diversity: include bases, finetunes, merges, adapters with resolvable bases, quantized models, MoE models, and non-causal architectures when they are analytically usable.

Exact proportions can differ from phase 2. The ablation dataset should be eval-first, diversity-second, and phase-2-proportion matching third.

## De-Duplication

Create normalized identifiers before selection:

- `model_id_norm`: lowercased repo id with whitespace removed.
- `repo_key`: normalized `owner/model`.
- `model_sha`: leaderboard or Hub sha when available.
- `revision_norm`: explicit revision or empty string.
- `canonical_model_key`: `repo_key@revision_or_sha` when possible, otherwise `repo_key`.

Remove rows that match phase 2 by:

- exact `model_id_norm`;
- exact `canonical_model_key`;
- exact `model_sha` when present;
- leaderboard `fullname` / HF `repo_id` equivalence.

If a model appears multiple times in eval sources, keep the row with the richest eval coverage and newest valid metadata, while preserving alternate eval rows in an audit file.

## Model-Type Suitability

Add `esd_suitability`:

- `good`: dense Transformers-loadable base, finetune, instruction tune, or merge with clear checkpoint weights.
- `usable_with_caution`: MoE, seq2seq, sequence classification, moderate quantized Transformers-native models, and adapters with resolvable/mergeable bases.
- `separate_or_exclude`: GGUF-only, EXL2, adapter-only without base, inaccessible/gated repos, removed repos, extreme quantized formats, or models that only expose non-comparable artifacts.

For lineage experiments, `usable_with_caution` rows can still be valuable, but analyses should stratify by model type and artifact format. For Wasserstein/raw-eigenvalue methods, `good` rows should be the main comparison set.

## Lineage Fields

Preserve or add:

- `source_model`
- `base_model_relation`
- `known_lineage_flag`
- `lineage_status`
- `lineage_source`
- `lineage_confidence`
- `lineage_eval_group`

`lineage_eval_group` should group models into comparable lineage scenarios:

- same-base finetunes;
- known merge families;
- adapter-versus-merged-base pairs;
- base-model family variants;
- unrelated negative pairs.

## Eval Fields

Preserve or add:

- `eval_source`
- `eval_name`
- `eval_status`
- `average_numeric`
- benchmark component columns when present, such as `IFEval`, `BBH`, `MATH Lvl 5`, `GPQA`, `MUSR`, and `MMLU-PRO`
- `eval_coverage_flag`: `complete`, `partial`, or `missing`
- `eval_task_count`
- `eval_score_version`

Rows with `eval_coverage_flag = missing` should be capped below 15% and mainly used for lineage-only diversity, not performance-prediction ablations.

## Output Files

Use a simple output layout:

- `data/ablation_5000/ablation_5000_candidates.csv`
- `data/ablation_5000/ablation_5000_selected.csv`
- `data/ablation_5000/ablation_5000_rejected.csv`
- `data/ablation_5000/ablation_5000_overlap_audit.csv`
- `data/ablation_5000/ablation_5000_summary.json`

The selected CSV should be directly usable by the ESD runner after column normalization.

## Summary Metrics

The curation script should report:

- total candidate rows;
- selected rows;
- eval-labelled count and missing-eval percentage;
- phase-2 overlap count;
- counts by `primary_type_bucket`;
- counts by `esd_suitability`;
- counts by `lineage_status`;
- counts by `eval_source`;
- top rejection reasons.

## Rejection Reasons

Use explicit rejection labels:

- `phase2_duplicate`
- `missing_eval_over_cap`
- `repo_inaccessible`
- `unsupported_artifact_format`
- `adapter_base_unresolved`
- `duplicate_lower_priority`
- `insufficient_metadata`
- `type_quota_pressure`
- `unsafe_for_raw_esd`

## Implementation Shape

Keep this as a small curation layer, not a large pipeline:

1. Load phase-2 selected data and build duplicate keys.
2. Load eval sources.
3. Normalize model ids and eval fields.
4. Enrich with HF metadata using the existing phase-2 conventions.
5. Classify type, lineage, eval coverage, and ESD suitability.
6. Filter phase-2 duplicates.
7. Select up to 5,000 rows under the eval-missing cap and diversity constraints.
8. Write selected/rejected/audit/summary outputs.

No model downloads should happen during curation. Availability checks should stay metadata-only.

## References

- Open LLM Leaderboard contents: https://huggingface.co/datasets/open-llm-leaderboard/contents
- Open LLM Leaderboard collection: https://huggingface.co/collections/open-llm-leaderboard/open-llm-leaderboard-2

## Implementation Defaults

- Start with Open LLM Leaderboard sources. Add additional domain-specific leaderboard sources only if the non-overlap eval-labelled count falls below 4,250.
- Include both `good` and `usable_with_caution` rows in the selected dataset, but preserve `esd_suitability` so the first Wasserstein/raw-eigenvalue ablation can run on `good` rows first and then repeat on caution strata.
