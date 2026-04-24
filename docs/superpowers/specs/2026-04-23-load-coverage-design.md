# 2026-04-23 Load Coverage Design

## Goal

Improve `Model-Zoo` phase-2 **model loading coverage** for the curated `15,000`-model zoo while keeping changes small and avoiding disruption to:

- dispatcher / parallelization
- GPU batching
- core ESD execution flow

The main target is broader success on models that are still realistically **Transformers-loadable** in the current workflow.

## Current State

On the curated `15,000`-row table, the current runtime can already load a large portion of the zoo, but a few backend/format gaps dominate the remaining blocked population.

Current environment:

- present: `gptqmodel`, `bitsandbytes`
- missing: `gguf`, `autoawq`, `compressed_tensors`

Current preflight-blocked slice:

- `1,663` `gguf`
- `140` `awq`
- `294` `exl2` / `quantized_alt_format`
- `85` inaccessible repos

This means the highest-leverage loader work is not general refactoring. It is:

1. support a few missing backends cleanly
2. keep routing explicit and lightweight
3. avoid over-gatekeeping models that could still load

## Principles

- Keep `Model-Zoo` as the operational source of truth.
- Do not rewrite dispatch, threading, batching, or GPU scheduling.
- Keep preflight mostly as `routing + obvious hard blocks`, not a heavy compatibility oracle.
- Let native backend/model loaders do most of the real validation.
- Prefer explicit support lanes over ad hoc special cases.
- Optimize first for **load coverage**, then handle the smaller post-load ESD/export gaps.

## Approaches

### 1. Single-env narrow expansion

Expand the current main environment just enough to support more HF-native paths:

- add `gguf`
- add `compressed-tensors`
- keep current GPTQ path
- keep AWQ and EXL2 unsupported
- add a spectral-only `AutoModel` fallback after task-head loaders fail

Pros:

- smallest operational risk
- biggest immediate gain comes from `gguf`

Cons:

- leaves `awq` blocked

### 2. Two-lane support matrix (recommended)

Keep one main `Transformers-first` lane and treat fragile formats separately.

Main lane:

- `standard_causal`
- `seq2seq`
- `sequence_classification`
- `multimodal`
- `adapter_requires_base`
- `gptq`
- `bitsandbytes`
- `gguf`
- `compressed_tensors`

Separate lane later:

- `awq`

Explicitly unsupported for now:

- `exl2`
- inaccessible repos

Pros:

- matches current HF ecosystem reality
- avoids polluting the main env with fragile AWQ dependencies
- keeps the loader surface understandable

Cons:

- `awq` stays out of the first pass

### 3. Max-coverage multi-backend expansion

Add support lanes for `awq`, `exl2`, and more specialized runtimes immediately.

Pros:

- highest theoretical coverage

Cons:

- too much complexity
- too much risk for the current phase

## Recommended Design

Use **Approach 2**.

### Main Support Matrix

Support in the main `Model-Zoo` loader:

- standard HF text-generation checkpoints
- seq2seq checkpoints
- sequence-classification checkpoints
- multimodal checkpoints that fit the existing HF task classes
- PEFT adapters with task-aware base loading
- GPTQ through `gptqmodel`
- bitsandbytes-style 4/8-bit checkpoints
- GGUF through the Transformers `gguf_file=...` path
- compressed-tensors checkpoints

Keep out of the first pass:

- AWQ in the main env

Keep explicitly unsupported:

- EXL2
- removed / private / gated repos we cannot access

### Loader Routing

Routing should remain explicit and minimal:

1. For adapters:
   - use `adapter_config.json` first
   - resolve base model and revision from PEFT config when available
   - keep current dense-upstream remap behavior for quantized mirror bases

2. For standard models:
   - use `AutoConfig` and explicit metadata to route to:
     - `AutoModelForCausalLM`
     - `AutoModelForSeq2SeqLM`
     - `AutoModelForSequenceClassification`
     - `AutoModelForImageTextToText`

3. If task-head routing fails but the checkpoint still looks Transformers-loadable:
   - fall back to plain `AutoModel`

That fallback is acceptable because ESD only needs analyzable weights/modules, not generation behavior.

### Preflight Policy

Preflight should stay light.

It should:

- block inaccessible repos
- block explicitly unsupported formats like `exl2`
- block resolved loaders whose required backend is definitely absent

It should not:

- aggressively prune models just because they look unusual
- try to predict all downstream loader failures

### Expected Coverage Impact

The biggest single gain is `gguf`.

Adding `gguf` support in the main environment should materially reduce the blocked set, because `gguf` is the largest missing family in the curated zoo.

After that, the next meaningful gain comes from:

- `compressed-tensors`
- broader `AutoModel` fallback

AWQ can be reconsidered later as a separate lane if the first pass still leaves too much value on the table.

## Out of Scope For This Pass

- dispatcher redesign
- GPU scheduler changes
- broad retry-policy redesign
- EXL2 support
- full AWQ integration in the main environment
- the deeper ESD/export fixes, except where a tiny adjustment is absolutely required to avoid masking load coverage

## Follow-on Work

Once load coverage improves, the next likely bottlenecks are:

- non-fatal handling when H5 export parsing yields zero valid `(layer, module)` pairs
- broader eligible-layer detection for quantized/custom linear layers
- a few parser/export edge cases around longname normalization

Those are real, but they should follow the load-coverage pass rather than being mixed into it.
