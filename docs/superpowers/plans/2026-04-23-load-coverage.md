# Load Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Broaden `Model-Zoo` phase-2 model loading coverage for the curated `15,000`-model zoo with minimal code changes, focusing on `gguf`, `compressed-tensors`, and a spectral-only fallback path while leaving dispatch, batching, and core ESD flow intact.

**Architecture:** Keep the current `Transformers-first` loader and preflight structure. Extend it with a small support-matrix expansion: add `compressed-tensors` backend awareness, keep the existing `gguf` path but make it runnable in the main env, and add a final `AutoModel` fallback after task-head loaders fail on otherwise valid Transformers checkpoints. Preflight remains light and only blocks clearly inaccessible or explicitly unsupported cases.

**Tech Stack:** Python, `transformers`, `peft`, `huggingface_hub`, `pytest`, optional backends `gptqmodel`, `gguf`, `compressed_tensors`

---

### Task 1: Add Focused Failing Tests For The New Support Matrix

**Files:**
- Modify: `Model-Zoo/esd_experiment/tests/test_model_loader.py`
- Modify: `Model-Zoo/esd_experiment/tests/test_model_preflight.py`
- Modify: `Model-Zoo/esd_experiment/tests/test_run_experiment.py`

- [ ] **Step 1: Add failing loader tests for `AutoModel` fallback and compressed-tensors detection**

Add focused tests to `Model-Zoo/esd_experiment/tests/test_model_loader.py` for:

```python
def test_load_model_falls_back_to_automodel_for_supported_non_head_config(monkeypatch):
    ...

def test_resolve_effective_loader_for_repo_detects_compressed_tensors_quant_method(monkeypatch):
    ...
```

The first test should simulate:
- `AutoModelForCausalLM.from_pretrained(...)` raising an unrecognized-config style error
- `AutoModel.from_pretrained(...)` succeeding
- `load_model(...)` returning the fallback model instead of failing

The second test should simulate:
- `AutoConfig.from_pretrained(...)` returning a config whose `quantization_config.quant_method == "compressed-tensors"`
- `resolve_effective_loader_for_repo(...)` returning `"compressed_tensors"`

- [ ] **Step 2: Add failing preflight tests for compressed-tensors backend gating**

Add focused tests to `Model-Zoo/esd_experiment/tests/test_model_preflight.py` for:

```python
def test_classify_row_preflight_marks_compressed_tensors_missing_backend():
    ...

def test_classify_row_preflight_allows_compressed_tensors_with_backend():
    ...
```

These should verify:
- resolved loader `"compressed_tensors"` is blocked as `unsupported_backend` when backend is missing
- the same row becomes eligible when backend status is `available`

- [ ] **Step 3: Add failing run-level tests for backend discovery**

Add focused tests to `Model-Zoo/esd_experiment/tests/test_run_experiment.py` for:

```python
def test_available_backends_includes_compressed_tensors(monkeypatch):
    ...

def test_row_backend_status_marks_compressed_tensors_from_tags(monkeypatch):
    ...
```

These should verify that run-level preflight can discover and propagate `compressed_tensors` availability the same way it already does for `gptq`, `awq`, and `gguf`.

- [ ] **Step 4: Run focused tests and verify they fail for the intended reasons**

Run:

```bash
pytest /scratch/kinshuk/Model-Research/Model-Zoo/esd_experiment/tests/test_model_loader.py \
  /scratch/kinshuk/Model-Research/Model-Zoo/esd_experiment/tests/test_model_preflight.py \
  /scratch/kinshuk/Model-Research/Model-Zoo/esd_experiment/tests/test_run_experiment.py -q
```

Expected:
- FAIL on the new tests
- existing tests may still pass

- [ ] **Step 5: Commit the test-only change**

```bash
git -C /scratch/kinshuk/Model-Research/Model-Zoo add \
  esd_experiment/tests/test_model_loader.py \
  esd_experiment/tests/test_model_preflight.py \
  esd_experiment/tests/test_run_experiment.py
git -C /scratch/kinshuk/Model-Research/Model-Zoo commit -m "test: cover load coverage expansion"
```

### Task 2: Extend Loader Routing With A Spectral-Only `AutoModel` Fallback

**Files:**
- Modify: `Model-Zoo/esd_experiment/src/model_loader.py`
- Test: `Model-Zoo/esd_experiment/tests/test_model_loader.py`

- [ ] **Step 1: Add the fallback import and selection helper**

Update `Model-Zoo/esd_experiment/src/model_loader.py` to import `AutoModel` and add a tiny helper like:

```python
from transformers import AutoModel

def _fallback_auto_model_cls(current_loader: str, error: Exception):
    message = str(error).lower()
    if "unrecognized configuration class" in message:
        return AutoModel
    return None
```

Keep this intentionally narrow. Do not create a large registry.

- [ ] **Step 2: Apply the fallback only after task-head loading fails**

In `load_model(...)`, keep the current task-head-first behavior:

```python
auto_model_cls = _select_auto_model_cls(effective_loader)
```

Then, inside the `except Exception as exc:` block for standard models:

```python
fallback_cls = _fallback_auto_model_cls(effective_loader, exc)
if fallback_cls is not None:
    model = hf_from_pretrained(
        fallback_cls,
        repo_id,
        **load_kwargs,
    )
```

Only use this fallback for otherwise normal Transformers checkpoints. Keep:
- GPTQ special handling
- existing multimodal / seq2seq fallback logic
- explicit unsupported alt-format behavior unchanged

- [ ] **Step 3: Add compressed-tensors loader resolution**

In `resolve_effective_loader_for_repo(...)`, extend quantization detection:

```python
if quant_method in {"gptq", "awq", "compressed-tensors"}:
    return quant_method.replace("-", "_")
```

Also allow the normalized value in `classify_loader_scenario_support(...)`:

```python
"compressed_tensors",
```

Do not add a new large backend lane. This should behave like another explicit Transformers-native quantized path.

- [ ] **Step 4: Run focused tests and verify they pass**

Run:

```bash
pytest /scratch/kinshuk/Model-Research/Model-Zoo/esd_experiment/tests/test_model_loader.py -q
```

Expected:
- PASS for the new fallback and compressed-tensors tests

- [ ] **Step 5: Commit the loader-routing change**

```bash
git -C /scratch/kinshuk/Model-Research/Model-Zoo add \
  esd_experiment/src/model_loader.py \
  esd_experiment/tests/test_model_loader.py
git -C /scratch/kinshuk/Model-Research/Model-Zoo commit -m "feat: broaden transformers loader coverage"
```

### Task 3: Extend Preflight And Backend Discovery Without Turning It Into A Gatekeeper

**Files:**
- Modify: `Model-Zoo/esd_experiment/src/model_preflight.py`
- Modify: `Model-Zoo/esd_experiment/src/run_experiment.py`
- Test: `Model-Zoo/esd_experiment/tests/test_model_preflight.py`
- Test: `Model-Zoo/esd_experiment/tests/test_run_experiment.py`

- [ ] **Step 1: Add compressed-tensors awareness to preflight**

Update `resolve_effective_loader(...)` in `model_preflight.py` to recognize compressed-tensors from lightweight row metadata:

```python
if "compressed-tensors" in loader_hints or "compressed_tensors" in loader_hints:
    return "compressed_tensors"
```

Keep this hint-based. Do not add Hub probing or expensive preflight.

- [ ] **Step 2: Add backend gating for compressed-tensors**

Extend `classify_row_preflight(...)`:

```python
if effective_loader == "compressed_tensors":
    if not backend_status or backend_status in {"missing", "absent", "none"}:
        return PreflightDecision(
            eligible=False,
            reason="unsupported_backend",
            effective_loader="compressed_tensors",
        )
    return PreflightDecision(
        eligible=True,
        reason="eligible",
        effective_loader="compressed_tensors",
    )
```

Do not broaden blocking elsewhere.

- [ ] **Step 3: Extend run-level backend discovery**

In `_available_backends()` inside `run_experiment.py`, add:

```python
if importlib.util.find_spec("compressed_tensors") is not None:
    backends.add("compressed_tensors")
```

In `_row_backend_status(...)`, add simple propagation from resolved loader / tags:

```python
if resolved_loader == "compressed_tensors" or "compressed-tensors" in tags_blob:
    return "available" if "compressed_tensors" in available_backends else "missing"
```

- [ ] **Step 4: Run focused tests and verify they pass**

Run:

```bash
pytest /scratch/kinshuk/Model-Research/Model-Zoo/esd_experiment/tests/test_model_preflight.py \
  /scratch/kinshuk/Model-Research/Model-Zoo/esd_experiment/tests/test_run_experiment.py -q
```

Expected:
- PASS for the new compressed-tensors backend tests

- [ ] **Step 5: Commit the preflight/backend changes**

```bash
git -C /scratch/kinshuk/Model-Research/Model-Zoo add \
  esd_experiment/src/model_preflight.py \
  esd_experiment/src/run_experiment.py \
  esd_experiment/tests/test_model_preflight.py \
  esd_experiment/tests/test_run_experiment.py
git -C /scratch/kinshuk/Model-Research/Model-Zoo commit -m "feat: extend load coverage preflight"
```

### Task 4: Document The Main Support Matrix And Runtime Requirements

**Files:**
- Modify: `Model-Zoo/README.md`
- Modify: `Model-Zoo/esd_experiment/README.md`

- [ ] **Step 1: Update the short support matrix**

Add one concise section to both READMEs describing the current main lane:

```md
Supported in the main env:
- standard causal / seq2seq / sequence classification
- multimodal HF task classes
- PEFT adapters
- GPTQ
- bitsandbytes
- GGUF
- compressed-tensors

Not in the main lane:
- AWQ
- EXL2
```

Also note that:
- `gguf` requires the `gguf` package
- `compressed-tensors` requires the `compressed_tensors` package
- `AWQ` is intentionally deferred

- [ ] **Step 2: Add the spectral-only fallback note**

Document that the loader now prefers task-head classes first and only falls back to `AutoModel` for otherwise valid Transformers checkpoints when spectral analysis does not require a task head.

- [ ] **Step 3: Commit the docs update**

```bash
git -C /scratch/kinshuk/Model-Research/Model-Zoo add README.md esd_experiment/README.md
git -C /scratch/kinshuk/Model-Research/Model-Zoo commit -m "docs: clarify load coverage support matrix"
```

### Task 5: Verify In The Real Runtime Without Launching A Large Run

**Files:**
- Modify: none
- Verify: `Model-Zoo/esd_experiment/tests`

- [ ] **Step 1: Install the two missing main-lane dependencies into `esd_ind`**

Run:

```bash
conda run -n esd_ind pip install gguf compressed-tensors
```

Expected:
- both packages install successfully

Do **not** install `autoawq` in this pass.

- [ ] **Step 2: Verify imports in the runtime**

Run:

```bash
conda run -n esd_ind python -c "import gguf, compressed_tensors; print('ok')"
```

Expected:

```text
ok
```

- [ ] **Step 3: Run the full ESD test suite**

Run:

```bash
pytest /scratch/kinshuk/Model-Research/Model-Zoo/esd_experiment/tests -q
```

Expected:
- all tests pass
- only pre-existing warnings may remain

- [ ] **Step 4: Recompute the preflight block counts only**

Run a small local summary command or helper script to recompute blocked counts on:

```text
/scratch/kinshuk/Model-Research/Model-Zoo/data/curated/model_zoo_phase2.csv
```

Expected:
- blocked `gguf` rows should disappear once `gguf` is importable
- blocked `compressed_tensors` rows, if any, should be classified correctly
- `exl2` and inaccessible repos should remain blocked

Do **not** launch a smoke run or larger analysis run in this task.

- [ ] **Step 5: Commit only if verification is clean**

```bash
git -C /scratch/kinshuk/Model-Research/Model-Zoo status --short
```

If only the intended files are changed and all checks pass:

```bash
git -C /scratch/kinshuk/Model-Research/Model-Zoo commit --allow-empty -m "chore: verify load coverage runtime"
```

If verification reveals a new issue, stop here and open a follow-up plan rather than widening this one.

---

## Spec Coverage Check

- `GGUF` first-pass support: covered by Task 5 runtime dependency step and existing loader path verification.
- `compressed-tensors` first-pass support: covered by Tasks 1–3 and Task 5.
- `AWQ` deferred: preserved explicitly in Tasks 4–5.
- `EXL2` unsupported: preserved explicitly in Tasks 3–5.
- light preflight + explicit routing: covered by Tasks 2–3.
- no dispatcher/batching/ESD core rewrite: preserved throughout.

## Placeholder Scan

- No `TODO` / `TBD` markers remain.
- No “write tests” placeholder without named tests remains.
- Commands and files are explicit.

## Type / Name Consistency Check

- Normalized loader name used throughout: `compressed_tensors`
- Existing loader names preserved: `standard_causal`, `seq2seq`, `sequence_classification`, `multimodal`, `gptq`, `awq`, `gguf`
- Backend discovery name matches preflight/backend gating: `compressed_tensors`
