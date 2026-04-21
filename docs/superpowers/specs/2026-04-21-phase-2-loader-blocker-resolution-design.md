# Phase 2 Loader Blocker Resolution Design

## Goal

Unblock the scaled phase-2 ESD run by fixing the main failure layer exposed by the `100`-model mixed-zoo smoke run.

This step should improve:

- model eligibility screening before dispatch
- loader routing for different model families
- terminal failure accounting

It should **not**:

- redesign the ESD analysis itself
- introduce a new orchestration framework
- start the larger run
- widen scope into phase-3 methods

## Why This Is The Right Target

The smoke run did not fail because the zoo was simply too broad.

The dominant failures were concentrated in a few operational buckets:

- quantized backend dependency/runtime gaps
- loader/model-type mismatches
- adapter rows whose effective backend was inherited from the base model
- incomplete or inaccessible repos
- incomplete run accounting

So the blocker is mostly in:

- preflight classification
- effective loader selection
- failure recording

not in the spectral analysis core.

## External Guidance

The latest Hugging Face docs and large-scale evaluation systems point to the same pattern:

- use explicit backend/model routing, not one generic loader
- classify unsupported or unavailable models before expensive execution
- preserve partial and failed outcomes explicitly
- treat metadata and accessibility as first-class operational state

Relevant references:

- HF Hub API / dry-run downloads
  - https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api
  - https://huggingface.co/docs/huggingface_hub/package_reference/file_download
- HF Transformers GPTQ / AWQ / GGUF
  - https://huggingface.co/docs/transformers/main/en/quantization/gptq
  - https://huggingface.co/docs/transformers/quantization/awq
  - https://huggingface.co/docs/transformers/gguf
- HF PEFT config / merge
  - https://huggingface.co/docs/peft/en/package_reference/config
  - https://huggingface.co/docs/peft/en/package_reference/peft_model
- HF model classes
  - https://huggingface.co/docs/transformers/model_doc/auto
  - https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
- Large eval frameworks
  - https://github.com/EleutherAI/lm-evaluation-harness
  - https://github.com/huggingface/lighteval
  - https://github.com/open-compass/opencompass
  - https://github.com/bigcode-project/bigcode-evaluation-harness

## Recommendation

Keep the current Model-Zoo runner architecture, but add one thin operational layer:

1. `preflight eligibility`
2. `effective loader resolution`
3. `atomic terminal status recording`

That is the smallest change that matches the actual blocker.

## Support Matrix

Treat loader support as an explicit matrix, not a guess from `primary_type_bucket`.

### `standard_causal`

Use:

- `AutoModelForCausalLM`

Applies to:

- normal text-generation Transformer checkpoints
- standard finetunes
- many merges

### `seq2seq`

Use:

- `AutoModelForSeq2SeqLM`

Applies to:

- `T5`-style and other encoder-decoder checkpoints

This needs to be explicit because current smoke failures show some such models are being misrouted through causal loading.

### `multimodal`

Use:

- `AutoModelForImageTextToText`

Applies to:

- actual image-text-to-text / VLM checkpoints

Do not choose this path only from a coarse type label. It should be driven by config / model class compatibility.

### `adapter_requires_base`

Use:

- `adapter_config.json`
- `base_model_name_or_path`
- `revision` if present

Then resolve the **base model's** effective loader/backend before loading.

This is critical because many current adapter failures are really base-model quantization/backend failures.

### `gptq`

Use only if the required GPTQ runtime is present.

Current HF guidance points to:

- `gptqmodel`
- compatible `accelerate`
- compatible `optimum`
- compatible `transformers`

If not present, mark:

- `unsupported_backend`

not:

- generic `model_load_error`

### `awq`

Use only if the required AWQ backend is present.

Detect from:

- quantization config
- repo structure / metadata

If not present, mark:

- `unsupported_backend`

### `gguf`

GGUF is a separate explicit loading path using `gguf_file`.

For this blocker pass, keep it as:

- `unsupported_loader_scenario`

unless we deliberately add GGUF support.

That keeps scope under control.

## Effective Loader Resolution

The current coarse `loader_scenario` is not enough by itself.

We need one small resolution step that produces an `effective_loader`.

Examples:

- adapter row -> inspect adapter config -> base model is GPTQ -> `effective_loader = gptq`
- model tagged multimodal but config is actually text-only -> route to `standard_causal` or `seq2seq`
- seq2seq MoE checkpoint -> route to `seq2seq`, not `standard_causal`

This resolution should be deterministic and logged.

## Preflight Eligibility

Before dispatching a model to `worker.py`, perform lightweight Hub checks.

Use:

- `HfApi.model_info`
- `list_repo_files`
- `snapshot_download(..., dry_run=True)` where useful

Classify rows into:

- `eligible`
- `repo_inaccessible`
- `unsupported_backend`
- `unsupported_loader_scenario`
- `adapter_base_unresolved`
- `missing_required_artifact`
- `download_or_cache_risk`

Examples of preflight catches:

- gated/private/invalid repo
- missing `adapter_config.json`
- unresolved adapter base
- GGUF-only repo
- missing shard files
- quantized backend required but not installed

Preflight should stop clearly ineligible models before they consume GPU time.

## Terminal Outcome Contract

Every scheduled model must end in exactly one terminal state:

- `success`
- `unsupported_backend`
- `unsupported_loader_scenario`
- `adapter_base_unresolved`
- `repo_inaccessible`
- `missing_required_artifact`
- `download_or_cache_failure`
- `load_failure`
- `analysis_failure`

This must be recorded atomically.

The current smoke run showed:

- malformed `failure_records.jsonl`
- dispatcher summary claiming `Failed: 0`
- models with no success artifact and no clean terminal failure record

That is unacceptable for scale-up.

## Accounting Rule

Final run status must be derived from artifacts and terminal records, not from process-level completion logs.

For a model to count as success:

- matching `stats/*.csv`
- matching `metrics/*.h5`

For a model to count as failure:

- one valid terminal failure record

No model should be left unclassified at the end of a run.

## Keep / Kill Rules

Keep:

- current runner architecture
- current worker-based execution
- current ESD analysis path
- existing failure artifacts, but hardened

Kill:

- generic catch-all loading
- deep worker failures for clearly unsupported models
- misleading success/failure summaries
- malformed concurrent JSONL writes

## Success Criteria

This blocker step is successful if:

- the next `100`-model smoke run has no missing terminal outcomes
- failure accounting is internally consistent
- the largest current failure bucket is no longer a generic backend miss
- adapters are routed through their base-model requirements correctly
- unsupported categories are rejected early and cleanly

## Next Step After This

Once this blocker is resolved:

1. rerun the same `100`-model mixed-zoo smoke sample
2. inspect success/failure by effective loader
3. only then decide whether the pipeline is ready for a larger phase-2 run
