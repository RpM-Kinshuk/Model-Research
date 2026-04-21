# Phase 2 Loader Blocker Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the minimum preflight, loader-routing, and terminal-accounting changes needed to make the next `100`-model mixed-zoo smoke rerun trustworthy and materially more successful.

**Architecture:** Keep the current `Model-Zoo` runner architecture. Add one small preflight module that classifies each model into an effective loader/support state before dispatch; extend the loader to support explicit `seq2seq` and backend-aware quantized paths; and harden worker/run accounting so every scheduled model ends in one atomic terminal state. Do not redesign the ESD analysis code or add a new orchestration layer.

**Tech Stack:** Python, pandas, pathlib, dataclasses, Hugging Face Hub API, Transformers, PEFT, pytest, JSONL

---

## File Structure Map

### New `Model-Zoo` code

- Create: `Model-Zoo/esd_experiment/src/model_preflight.py`
  - lightweight Hub checks, effective loader resolution, backend/support classification, and per-row preflight decision

### Existing `Model-Zoo` code to modify

- Modify: `Model-Zoo/esd_experiment/src/model_loader.py`
  - explicit support matrix for `standard_causal`, `seq2seq`, `multimodal`, `adapter_requires_base`, `gptq`, `awq`, `gguf`
  - structured backend-missing failures
- Modify: `Model-Zoo/esd_experiment/src/run_experiment.py`
  - run preflight before command generation
  - skip ineligible rows with structured terminal records
  - derive final run summary from per-model terminal status rather than dispatcher text
- Modify: `Model-Zoo/esd_experiment/src/worker.py`
  - atomic per-model terminal status file
  - atomic structured failure recording
  - no missing terminal outcomes
- Modify: `Model-Zoo/README.md`
  - short note that phase-2 run eligibility now depends on preflight/backend support
- Modify: `Model-Zoo/esd_experiment/README.md`
  - short support-matrix summary and smoke-run interpretation note

### New or updated `Model-Zoo` tests

- Create: `Model-Zoo/esd_experiment/tests/test_model_preflight.py`
  - effective loader resolution and ineligible/runnable classification
- Modify: `Model-Zoo/esd_experiment/tests/test_model_loader.py`
  - seq2seq routing, GPTQ/AWQ backend classification, GGUF rejection, adapter-base effective loader inheritance
- Modify: `Model-Zoo/esd_experiment/tests/test_run_experiment.py`
  - preflight filtering and final summary derivation
- Modify: `Model-Zoo/esd_experiment/tests/test_worker.py`
  - atomic terminal status files and aggregate failure log hardening

### Validation targets

- `pytest Model-Zoo/esd_experiment/tests/test_model_preflight.py -q`
- `pytest Model-Zoo/esd_experiment/tests/test_model_loader.py -q`
- `pytest Model-Zoo/esd_experiment/tests/test_run_experiment.py -q`
- `pytest Model-Zoo/esd_experiment/tests/test_worker.py -q`
- `pytest Model-Zoo/esd_experiment/tests -q`

---

### Task 1: Add Preflight Classification And Effective Loader Resolution

**Files:**
- Create: `Model-Zoo/esd_experiment/src/model_preflight.py`
- Create: `Model-Zoo/esd_experiment/tests/test_model_preflight.py`

- [ ] **Step 1: Write the failing preflight tests**

Create `Model-Zoo/esd_experiment/tests/test_model_preflight.py`:

```python
from dataclasses import asdict

import pandas as pd

from model_preflight import (
    PreflightDecision,
    classify_row_preflight,
    resolve_effective_loader,
)


def test_resolve_effective_loader_prefers_seq2seq_for_t5_configs():
    row = pd.Series(
        {
            "model_id": "org/t5-model",
            "loader_scenario": "standard_transformers",
            "primary_type_bucket": "moe",
            "config_model_type": "t5",
            "config_architectures": "T5ForConditionalGeneration",
        }
    )

    assert resolve_effective_loader(row) == "seq2seq"


def test_resolve_effective_loader_prefers_multimodal_for_image_text_configs():
    row = pd.Series(
        {
            "model_id": "org/vlm",
            "loader_scenario": "multimodal_transformers",
            "config_model_type": "llava",
            "config_architectures": "LlavaForConditionalGeneration",
        }
    )

    assert resolve_effective_loader(row) == "multimodal"


def test_classify_row_preflight_marks_missing_adapter_config_ineligible():
    row = pd.Series(
        {
            "model_id": "org/adapter",
            "loader_scenario": "adapter_requires_base",
            "repo_private": False,
            "repo_gated": False,
            "repo_exists": True,
            "repo_files": "adapter_model.safetensors,tokenizer.json",
        }
    )

    decision = classify_row_preflight(row, available_backends={"standard_causal"})

    assert isinstance(decision, PreflightDecision)
    assert decision.eligible is False
    assert decision.reason == "missing_required_artifact"
    assert decision.effective_loader == "adapter_requires_base"


def test_classify_row_preflight_marks_gptq_backend_missing():
    row = pd.Series(
        {
            "model_id": "org/gptq-model",
            "loader_scenario": "quantized_transformers_native",
            "repo_private": False,
            "repo_gated": False,
            "repo_exists": True,
            "quant_method": "gptq",
            "repo_files": "config.json,model.safetensors",
        }
    )

    decision = classify_row_preflight(row, available_backends={"standard_causal"})

    assert decision.eligible is False
    assert decision.reason == "unsupported_backend"
    assert decision.effective_loader == "gptq"


def test_classify_row_preflight_allows_standard_transformers_text_model():
    row = pd.Series(
        {
            "model_id": "org/model",
            "loader_scenario": "standard_transformers",
            "repo_private": False,
            "repo_gated": False,
            "repo_exists": True,
            "config_model_type": "llama",
            "config_architectures": "LlamaForCausalLM",
            "repo_files": "config.json,model.safetensors",
        }
    )

    decision = classify_row_preflight(
        row,
        available_backends={"standard_causal", "seq2seq", "multimodal"},
    )

    assert decision.eligible is True
    assert decision.reason == "eligible"
    assert decision.effective_loader == "standard_causal"
```

- [ ] **Step 2: Run the preflight tests to verify they fail**

Run:

```bash
pytest Model-Zoo/esd_experiment/tests/test_model_preflight.py -q
```

Expected: import failure because `model_preflight.py` does not exist yet.

- [ ] **Step 3: Write the minimal preflight module**

Create `Model-Zoo/esd_experiment/src/model_preflight.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class PreflightDecision:
    model_id: str
    eligible: bool
    effective_loader: str
    reason: str
    message: str


def _text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _file_set(row: pd.Series) -> set[str]:
    raw = _text(row.get("repo_files", ""))
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def resolve_effective_loader(row: pd.Series) -> str:
    scenario = _text(row.get("loader_scenario", "")).lower()
    model_type = _text(row.get("config_model_type", "")).lower()
    architectures = _text(row.get("config_architectures", "")).lower()
    quant_method = _text(row.get("quant_method", "")).lower()

    if scenario == "adapter_requires_base":
        return "adapter_requires_base"
    if quant_method == "gptq":
        return "gptq"
    if quant_method == "awq":
        return "awq"
    if scenario == "quantized_alt_format":
        return "gguf"
    if "gguf" in architectures or model_type == "gguf":
        return "gguf"
    if "image" in scenario or "llava" in model_type or "vision" in architectures:
        return "multimodal"
    if model_type in {"t5", "mt5", "bart"} or "conditionalgeneration" in architectures and "causallm" not in architectures:
        return "seq2seq"
    return "standard_causal"


def classify_row_preflight(row: pd.Series, available_backends: set[str]) -> PreflightDecision:
    model_id = _text(row.get("model_id", ""))
    effective_loader = resolve_effective_loader(row)
    repo_private = bool(row.get("repo_private", False))
    repo_gated = bool(row.get("repo_gated", False))
    repo_exists = bool(row.get("repo_exists", True))
    files = _file_set(row)

    if not repo_exists or repo_private or repo_gated:
        return PreflightDecision(model_id, False, effective_loader, "repo_inaccessible", "Repo is missing, private, or gated")

    if effective_loader == "adapter_requires_base" and "adapter_config.json" not in files:
        return PreflightDecision(model_id, False, effective_loader, "missing_required_artifact", "adapter_config.json is required for adapter loading")

    if effective_loader == "gptq" and "gptq" not in available_backends:
        return PreflightDecision(model_id, False, effective_loader, "unsupported_backend", "GPTQ backend is unavailable in the current runtime")

    if effective_loader == "awq" and "awq" not in available_backends:
        return PreflightDecision(model_id, False, effective_loader, "unsupported_backend", "AWQ backend is unavailable in the current runtime")

    if effective_loader == "gguf":
        return PreflightDecision(model_id, False, effective_loader, "unsupported_loader_scenario", "GGUF repos are out of scope for this blocker pass")

    return PreflightDecision(model_id, True, effective_loader, "eligible", "Eligible for dispatch")
```

- [ ] **Step 4: Run the preflight tests to verify they pass**

Run:

```bash
pytest Model-Zoo/esd_experiment/tests/test_model_preflight.py -q
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/Model-Zoo add \
  esd_experiment/src/model_preflight.py \
  esd_experiment/tests/test_model_preflight.py
git -C /scratch/kinshuk/Model-Research/Model-Zoo commit -m "feat: add phase 2 preflight classification"
```

---

### Task 2: Extend Loader Support Matrix Without Rewriting The Loader

**Files:**
- Modify: `Model-Zoo/esd_experiment/src/model_loader.py`
- Modify: `Model-Zoo/esd_experiment/tests/test_model_loader.py`

- [ ] **Step 1: Write the failing loader tests**

Append these tests to `Model-Zoo/esd_experiment/tests/test_model_loader.py`:

```python
@patch("model_loader_under_test.hf_from_pretrained")
def test_load_model_uses_seq2seq_auto_class(mock_from_pretrained):
    mock_model = Mock()
    mock_from_pretrained.return_value = mock_model

    model, is_adapter = load_model(
        "org/t5-model",
        loader_scenario="seq2seq",
    )

    assert model is mock_model
    assert is_adapter is False
    assert mock_from_pretrained.call_args.args[0] is model_loader.AutoModelForSeq2SeqLM


def test_classify_loader_scenario_support_rejects_gguf():
    failure = classify_loader_scenario_support("gguf")

    assert failure is not None
    assert failure.reason == "unsupported_loader_scenario"


@patch("model_loader_under_test.resolve_base_model", return_value="base/model")
@patch("model_loader_under_test.resolve_effective_loader_for_repo", return_value="gptq")
def test_adapter_base_effective_loader_is_backend_sensitive(mock_effective, mock_base):
    assert model_loader.resolve_adapter_effective_loader("org/adapter", None) == "gptq"
```

- [ ] **Step 2: Run the loader tests to verify they fail**

Run:

```bash
pytest Model-Zoo/esd_experiment/tests/test_model_loader.py -q
```

Expected: failure because `seq2seq` and adapter-base effective loader support are not implemented.

- [ ] **Step 3: Extend `model_loader.py` minimally**

Update imports and support functions in `Model-Zoo/esd_experiment/src/model_loader.py`:

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM
```

Add or extend helper functions:

```python
def resolve_effective_loader_for_repo(repo_id: str, loader_scenario: Optional[str] = None) -> str:
    scenario = (loader_scenario or "").strip().lower()
    if scenario in {"seq2seq", "multimodal", "gptq", "awq", "gguf"}:
        return scenario
    cfg = AutoConfig.from_pretrained(repo_id, token=get_hf_token(), trust_remote_code=True)
    model_type = getattr(cfg, "model_type", "") or ""
    architectures = " ".join(getattr(cfg, "architectures", []) or [])
    quantization_config = getattr(cfg, "quantization_config", None) or {}
    quant_method = str(quantization_config.get("quant_method", "")).lower()
    if quant_method == "gptq":
        return "gptq"
    if quant_method == "awq":
        return "awq"
    if model_type in {"t5", "mt5", "bart"}:
        return "seq2seq"
    if "llava" in model_type.lower() or "conditionalgeneration" in architectures.lower() and "causallm" not in architectures.lower():
        return "multimodal" if "vision" in architectures.lower() or "llava" in architectures.lower() else "seq2seq"
    return "standard_transformers"


def resolve_adapter_effective_loader(adapter_repo: str, source_model: Optional[str]) -> str:
    base_repo = resolve_base_model(adapter_repo, source_model)
    return resolve_effective_loader_for_repo(base_repo)
```

Update `classify_loader_scenario_support`:

```python
if scenario in {"gguf", "quantized_alt_format"}:
    return LoaderFailure("load", "unsupported_loader_scenario", "GGUF-style repos are not supported by the current loader")
```

Update `load_model` routing:

```python
if loader_scenario == "seq2seq":
    model = hf_from_pretrained(
        AutoModelForSeq2SeqLM,
        repo_id,
        device_map=device_map,
        torch_dtype=torch_dtype,
        revision=revision,
    )
    model.eval()
    return model, False
```

And when adapter loading is chosen:

```python
effective_loader = resolve_adapter_effective_loader(repo_id, source_model)
if effective_loader == "gptq":
    raise LoaderFailure("load", "unsupported_backend", "Adapter base model requires GPTQ backend support in the current runtime")
if effective_loader == "awq":
    raise LoaderFailure("load", "unsupported_backend", "Adapter base model requires AWQ backend support in the current runtime")
```

- [ ] **Step 4: Run the loader tests to verify they pass**

Run:

```bash
pytest Model-Zoo/esd_experiment/tests/test_model_loader.py -q
```

Expected: all loader tests pass.

- [ ] **Step 5: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/Model-Zoo add \
  esd_experiment/src/model_loader.py \
  esd_experiment/tests/test_model_loader.py
git -C /scratch/kinshuk/Model-Research/Model-Zoo commit -m "feat: add explicit phase 2 loader support matrix"
```

---

### Task 3: Integrate Preflight Into Dispatch And Make Run Summaries Trustworthy

**Files:**
- Modify: `Model-Zoo/esd_experiment/src/run_experiment.py`
- Modify: `Model-Zoo/esd_experiment/tests/test_run_experiment.py`

- [ ] **Step 1: Write the failing dispatch/accounting tests**

Append these tests to `Model-Zoo/esd_experiment/tests/test_run_experiment.py`:

```python
def test_filter_models_to_run_skips_rows_marked_ineligible_by_preflight(tmp_path: Path, monkeypatch):
    df = pd.DataFrame(
        [
            {"model_id": "org/model-a", "loader_scenario": "standard_transformers"},
            {"model_id": "org/model-b", "loader_scenario": "adapter_requires_base"},
        ]
    )

    monkeypatch.setattr(
        run_experiment,
        "preflight_model_list",
        lambda frame: (
            frame.iloc[[0]].copy(),
            pd.DataFrame([{"model_id": "org/model-b", "reason": "missing_required_artifact"}]),
        ),
    )

    runnable, blocked = run_experiment.apply_preflight(df)

    assert list(runnable["model_id"]) == ["org/model-a"]
    assert list(blocked["model_id"]) == ["org/model-b"]


def test_collect_run_outcomes_uses_artifacts_and_terminal_failures(tmp_path: Path):
    stats_dir = tmp_path / "stats"
    metrics_dir = tmp_path / "metrics"
    logs_dir = tmp_path / "logs" / "terminal_status"
    stats_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)

    (stats_dir / "org--success.csv").write_text("alpha\\n1.0\\n")
    (metrics_dir / "org--success.h5").write_text("ok")
    (logs_dir / "org--failed.json").write_text('{\"model_id\":\"org/failed\",\"terminal_status\":\"load_failure\"}\\n')

    summary = run_experiment.collect_run_outcomes(tmp_path)

    assert summary["success"] == 1
    assert summary["failure"] == 1
```

- [ ] **Step 2: Run the run-experiment tests to verify they fail**

Run:

```bash
pytest Model-Zoo/esd_experiment/tests/test_run_experiment.py -q
```

Expected: failures because preflight integration and artifact-based outcome collection do not exist.

- [ ] **Step 3: Add minimal preflight integration**

In `Model-Zoo/esd_experiment/src/run_experiment.py`, add imports:

```python
from model_preflight import classify_row_preflight
```

Add helper functions:

```python
def _available_backends() -> set[str]:
    backends = {"standard_causal", "seq2seq", "multimodal"}
    try:
        import gptqmodel  # noqa: F401
        backends.add("gptq")
    except Exception:
        pass
    try:
        import awq  # noqa: F401
        backends.add("awq")
    except Exception:
        pass
    return backends


def apply_preflight(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = [classify_row_preflight(row, _available_backends()) for _, row in model_df.iterrows()]
    decision_df = pd.DataFrame([decision.__dict__ for decision in decisions])
    runnable = model_df[decision_df["eligible"]].reset_index(drop=True)
    blocked = decision_df[~decision_df["eligible"]].reset_index(drop=True)
    return runnable, blocked


def collect_run_outcomes(output_dir: Path) -> dict[str, int]:
    stats_dir = output_dir / "stats"
    metrics_dir = output_dir / "metrics"
    terminal_dir = output_dir / "logs" / "terminal_status"
    success_ids = {p.stem.replace("--", "/", 1) for p in stats_dir.glob("*.csv")} & {
        p.stem.replace("--", "/", 1) for p in metrics_dir.glob("*.h5")
    }
    failure_ids = {p.stem.replace("--", "/", 1) for p in terminal_dir.glob("*.json")}
    return {"success": len(success_ids), "failure": len(failure_ids)}
```

Then apply preflight after `load_model_list(...)` and before command generation:

```python
model_df, blocked_df = apply_preflight(model_df)
logger.info(f"Runnable after preflight: {len(model_df)}")
logger.info(f"Blocked by preflight: {len(blocked_df)}")
```

And replace the final summary counts with `collect_run_outcomes(output_dir)`.

- [ ] **Step 4: Run the run-experiment tests to verify they pass**

Run:

```bash
pytest Model-Zoo/esd_experiment/tests/test_run_experiment.py -q
```

Expected: all run-experiment tests pass.

- [ ] **Step 5: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/Model-Zoo add \
  esd_experiment/src/run_experiment.py \
  esd_experiment/tests/test_run_experiment.py
git -C /scratch/kinshuk/Model-Research/Model-Zoo commit -m "feat: add phase 2 preflight dispatch gating"
```

---

### Task 4: Make Worker Failures Atomic And Ensure One Terminal Outcome Per Model

**Files:**
- Modify: `Model-Zoo/esd_experiment/src/worker.py`
- Modify: `Model-Zoo/esd_experiment/tests/test_worker.py`

- [ ] **Step 1: Write the failing worker tests**

Append these tests to `Model-Zoo/esd_experiment/tests/test_worker.py`:

```python
def test_record_terminal_status_writes_one_json_file_per_model(tmp_path: Path):
    worker = load_worker_module()

    worker.record_terminal_status(
        output_dir=tmp_path,
        model_id="org/model",
        terminal_status="unsupported_backend",
        stage="load",
        reason="unsupported_backend",
        message="GPTQ backend unavailable",
    )

    status_file = tmp_path / "logs" / "terminal_status" / "org--model.json"
    record = json.loads(status_file.read_text())
    assert record["model_id"] == "org/model"
    assert record["terminal_status"] == "unsupported_backend"


def test_record_failure_appends_valid_json_line_per_call(tmp_path: Path):
    worker = load_worker_module()
    output_dir = tmp_path / "results"

    worker.record_failure(output_dir, "org/model-a", "load", "load_failure", "a", 1)
    worker.record_failure(output_dir, "org/model-b", "load", "load_failure", "b", 1)

    lines = (output_dir / "logs" / "failure_records.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["model_id"] == "org/model-a"
    assert json.loads(lines[1])["model_id"] == "org/model-b"
```

- [ ] **Step 2: Run the worker tests to verify they fail**

Run:

```bash
pytest Model-Zoo/esd_experiment/tests/test_worker.py -q
```

Expected: failure because `record_terminal_status` does not exist and failure recording is not yet hardened.

- [ ] **Step 3: Implement atomic terminal status recording**

In `Model-Zoo/esd_experiment/src/worker.py`, add:

```python
def _terminal_status_path(output_dir: Path, model_id: str) -> Path:
    return output_dir / "logs" / "terminal_status" / f"{safe_filename(model_id)}.json"


def record_terminal_status(
    output_dir: Path,
    model_id: str,
    terminal_status: str,
    stage: str,
    reason: str,
    message: str,
) -> None:
    path = _terminal_status_path(output_dir, model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    payload = {
        "model_id": model_id,
        "terminal_status": terminal_status,
        "stage": stage,
        "reason": reason,
        "message": message,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    temp_path.write_text(json.dumps(payload) + "\n")
    temp_path.replace(path)
```

Update `record_failure(...)` to write one JSON string per line using a single append call:

```python
with open(jsonl_path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record) + "\n")
```

Then, in each terminal success/failure branch of `main()`, call `record_terminal_status(...)`.

- [ ] **Step 4: Run the worker tests to verify they pass**

Run:

```bash
pytest Model-Zoo/esd_experiment/tests/test_worker.py -q
```

Expected: all worker tests pass.

- [ ] **Step 5: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/Model-Zoo add \
  esd_experiment/src/worker.py \
  esd_experiment/tests/test_worker.py
git -C /scratch/kinshuk/Model-Research/Model-Zoo commit -m "fix: harden phase 2 terminal failure accounting"
```

---

### Task 5: Update Short Docs And Run The Focused Verification Suite

**Files:**
- Modify: `Model-Zoo/README.md`
- Modify: `Model-Zoo/esd_experiment/README.md`

- [ ] **Step 1: Add one short support-matrix note to the docs**

Update `Model-Zoo/esd_experiment/README.md` with a short section like:

```md
## Phase 2 Loader Support

Phase 2 now uses a preflight step before GPU dispatch.

Current explicit loader paths:

- `standard_causal`
- `seq2seq`
- `multimodal`
- `adapter_requires_base`
- `gptq`
- `awq`
- `gguf` (currently rejected as unsupported in this blocker pass)

The run summary is derived from per-model success artifacts and per-model terminal failure records, not only from dispatcher completion logs.
```

Update `Model-Zoo/README.md` with one short pointer:

```md
Phase-2 ESD runs use `data/curated/model_zoo_phase2.csv`, a preflight eligibility step, and output-root accounting under `analysis_runs/phase2/<run_name>/`.
```

- [ ] **Step 2: Run the focused suite**

Run:

```bash
pytest Model-Zoo/esd_experiment/tests/test_model_preflight.py -q
pytest Model-Zoo/esd_experiment/tests/test_model_loader.py -q
pytest Model-Zoo/esd_experiment/tests/test_run_experiment.py -q
pytest Model-Zoo/esd_experiment/tests/test_worker.py -q
pytest Model-Zoo/esd_experiment/tests -q
```

Expected:

- all four targeted files pass
- the full `esd_experiment/tests` suite passes

- [ ] **Step 3: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/Model-Zoo add \
  README.md \
  esd_experiment/README.md
git -C /scratch/kinshuk/Model-Research/Model-Zoo commit -m "docs: clarify phase 2 loader support matrix"
```

---

## Self-Review

### Spec coverage

Covered:

- support matrix
- effective loader resolution
- lightweight preflight eligibility
- atomic terminal accounting
- no ESD redesign
- no larger run kickoff

Not included:

- GGUF support implementation
- full backend installation or environment changes
- smoke rerun execution itself

Those are intentionally out of scope for this blocker plan.

### Placeholder scan

No `TODO`, `TBD`, or deferred implementation placeholders remain.

### Type consistency

Plan uses these stable names consistently:

- `PreflightDecision`
- `resolve_effective_loader`
- `classify_row_preflight`
- `apply_preflight`
- `collect_run_outcomes`
- `record_terminal_status`

