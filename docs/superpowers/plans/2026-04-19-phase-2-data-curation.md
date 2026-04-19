# Phase 2 Data Curation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the phase-2 data curation pipeline that ingests the current sampled zoo, expands it toward a balanced 15k candidate set, applies duplicate collapse and type-aware lineage balancing, enriches it with lightweight Hugging Face and leaderboard metadata, and materializes one canonical model table plus a few derived views.

**Architecture:** Keep phase 2 narrowly focused on curation. Add one focused schema/utilities module for normalization, duplicate handling, taxonomy, and seed ingestion; one focused Hub preflight module for lightweight fetchability checks; and one orchestration CLI that reads current local sources, expands from `fetch_data/large_nlp_data.json`, applies soft balancing, and writes canonical outputs under `data/curated/`. Do not redesign the later analysis runner; only preserve compatibility with its current assumptions.

**Tech Stack:** Python, pandas, pathlib, dataclasses, Hugging Face Hub API, datasets, pytest, Markdown

---

## File Structure Map

### New active phase-2 data-curation code

- Create: `fetch_data/__init__.py`
  - make `fetch_data` an explicit importable package for the new phase-2 helpers and tests
- Create: `fetch_data/phase2_schema.py`
  - repo normalization, revision parsing, canonical row key, duplicate collapse, type taxonomy, loader scenario classification, seed ingestion helpers
- Create: `fetch_data/hub_preflight.py`
  - lightweight Hub/API fetchability probes and file-structure checks
- Create: `fetch_data/build_phase2_zoo.py`
  - main phase-2 orchestration CLI that builds the canonical table and derived outputs
- Create: `fetch_data/README.md`
  - explain which fetch-data entrypoints are current for phase 2 and which older scripts remain preserved helpers

### New data layer

- Create: `data/curated/README.md`
  - explain generated outputs and which files are canonical
- Generate via CLI:
  - `data/curated/model_zoo_phase2.csv`
  - `data/curated/model_zoo_phase2.parquet`
  - `data/curated/model_zoo_phase2_prediction_ready.csv`
  - `data/curated/model_zoo_phase2_analysis_ready.csv`
  - `data/curated/model_zoo_phase2_leaderboard_raw.csv`
  - `data/curated/model_zoo_phase2_probe_failures.csv`

### Tests

- Create: `tests/test_phase2_schema.py`
  - normalization, duplicate handling, taxonomy, loader-scenario classification, seed ingestion
- Create: `tests/test_hub_preflight.py`
  - lightweight probe logic using mocks, no live full downloads
- Create: `tests/test_build_phase2_zoo.py`
  - end-to-end curation build with temp fixtures and `--skip-hub-probes`

### Existing source inputs that remain in use

- `fetch_data/large_nlp_data.json`
  - atlas-style source graph for candidate expansion
- `data/sampled_metadata.csv`
  - current 5k seed sample
- `data/sampled_leaderboard.csv`
  - current leaderboard enrichment
- `data/sampled_model_subgraph.json`
  - current sampled graph
- `data/manual_audits/spectral_family_retrieval_label_freeze.csv`
  - keep intact, no changes in this phase
- `svd_results/`
  - existing already-processed local outputs used only to mark current analysis-ready rows

### Validation targets

- `pytest tests/test_phase2_schema.py -q`
- `pytest tests/test_hub_preflight.py -q`
- `pytest tests/test_build_phase2_zoo.py -q`
- `pytest -q`
- smoke build:
  - `python fetch_data/build_phase2_zoo.py --target-size 200 --skip-hub-probes --output-dir data/curated/smoke_phase2`

---

### Task 1: Add Canonical Schema Utilities And Duplicate Handling

**Files:**
- Create: `fetch_data/__init__.py`
- Create: `fetch_data/phase2_schema.py`
- Test: `tests/test_phase2_schema.py`

- [ ] **Step 1: Write the failing schema tests**

Create `tests/test_phase2_schema.py` with these tests:

```python
from pathlib import Path

import pandas as pd

from fetch_data.phase2_schema import (
    canonical_model_key,
    classify_loader_scenario_from_row,
    classify_primary_type,
    collapse_leaderboard_duplicates,
    normalize_repo_id,
    parse_repo_and_revision,
    seed_analysis_status,
)


def test_parse_repo_and_revision_splits_explicit_revision():
    repo_id, revision = parse_repo_and_revision("org/model@refs/pr/7")
    assert repo_id == "org/model"
    assert revision == "refs/pr/7"


def test_normalize_repo_id_collapses_hf_urls_and_owner_model_escape():
    assert normalize_repo_id("https://huggingface.co/owner/model") == "owner/model"
    assert normalize_repo_id("owner--model") == "owner/model"


def test_canonical_model_key_uses_revision_when_present():
    assert canonical_model_key("owner/model", None) == "owner/model"
    assert canonical_model_key("owner/model", "rev-a") == "owner/model@rev-a"


def test_collapse_leaderboard_duplicates_prefers_more_complete_row():
    leaderboard = pd.DataFrame(
        [
            {
                "model_id": "owner/model",
                "Precision": "float16",
                "Architecture": "LlamaForCausalLM",
                "#Params (B)": 8.0,
                "Average ⬆️": 30.0,
                "Type": "chat",
            },
            {
                "model_id": "owner/model",
                "Precision": "bfloat16",
                "Architecture": "Unknown",
                "#Params (B)": 0.0,
                "Average ⬆️": 28.0,
                "Type": "other",
            },
        ]
    )
    canonical, raw = collapse_leaderboard_duplicates(leaderboard)
    assert len(raw) == 2
    assert len(canonical) == 1
    row = canonical.iloc[0]
    assert row["Architecture"] == "LlamaForCausalLM"
    assert row["canonical_model_key"] == "owner/model"


def test_seed_analysis_status_marks_existing_svd_outputs(tmp_path: Path):
    (tmp_path / "owner--model.csv").write_text("alpha\\n1.2\\n")
    frame = pd.DataFrame({"model_id": ["owner/model", "other/model"]})
    status = seed_analysis_status(frame, tmp_path)
    assert status.loc[status["model_id"] == "owner/model", "analysis_artifact_present"].item() is True
    assert status.loc[status["model_id"] == "other/model", "analysis_artifact_present"].item() is False


def test_classify_primary_type_and_loader_scenario_cover_core_cases():
    adapter_row = pd.Series(
        {
            "base_model_relation": "adapter",
            "pipeline_tag": "text-generation",
            "files": ["adapter_config.json", "adapter_model.safetensors"],
        }
    )
    gguf_row = pd.Series(
        {
            "base_model_relation": "quantized",
            "pipeline_tag": "text-generation",
            "files": ["model.Q4_K_M.gguf", "tokenizer.json"],
        }
    )
    multimodal_row = pd.Series(
        {
            "base_model_relation": "finetune",
            "pipeline_tag": "image-text-to-text",
            "files": ["config.json", "preprocessor_config.json"],
            "Architecture": "MllamaForConditionalGeneration",
        }
    )
    assert classify_primary_type(adapter_row) == "adapter"
    assert classify_loader_scenario_from_row(adapter_row) == "adapter_requires_base"
    assert classify_loader_scenario_from_row(gguf_row) == "quantized_alt_format"
    assert classify_loader_scenario_from_row(multimodal_row) == "multimodal_transformers"
```

- [ ] **Step 2: Run the schema tests to verify they fail**

Run:

```bash
pytest tests/test_phase2_schema.py -q
```

Expected: `ModuleNotFoundError` or import failure for `fetch_data.phase2_schema`

- [ ] **Step 3: Create the importable package marker**

Create `fetch_data/__init__.py`:

```python
"""Data curation helpers for phase 2 model-zoo construction."""
```

- [ ] **Step 4: Implement the core schema utilities**

Create `fetch_data/phase2_schema.py` with this initial implementation:

```python
from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import pandas as pd


KNOWN_QUANT_METHODS = {"awq", "gptq", "bitsandbytes"}


def parse_repo_and_revision(model_id: object) -> tuple[str | None, str | None]:
    if pd.isna(model_id):
        return None, None
    value = str(model_id).strip()
    if not value:
        return None, None
    if "@" in value and not value.startswith("@"):
        repo_id, revision = value.split("@", 1)
        return normalize_repo_id(repo_id), revision.strip() or None
    return normalize_repo_id(value), None


def normalize_repo_id(raw_value: object) -> str | None:
    if pd.isna(raw_value):
        return None
    value = str(raw_value).strip()
    value = re.sub(r"^https?://(?:www\\.)?huggingface\\.co/", "", value, flags=re.IGNORECASE)
    value = value.split("?", 1)[0].split("#", 1)[0].strip().strip("/")
    if not value:
        return None
    if "/" not in value and "--" in value:
        owner, name = value.split("--", 1)
        if owner and name:
            value = f"{owner}/{name}"
    if "/" not in value:
        return None
    return value


def canonical_model_key(repo_id: str | None, revision: str | None) -> str | None:
    if not repo_id:
        return None
    return f"{repo_id}@{revision}" if revision else repo_id


def _completeness_score(frame: pd.DataFrame) -> pd.Series:
    architecture_known = frame.get("Architecture", pd.Series(index=frame.index, dtype="object")).fillna("").ne("").astype(int)
    params_known = pd.to_numeric(frame.get("#Params (B)"), errors="coerce").notna().astype(int)
    type_known = frame.get("Type", pd.Series(index=frame.index, dtype="object")).fillna("").ne("").astype(int)
    average_known = pd.to_numeric(frame.get("Average ⬆️"), errors="coerce").notna().astype(int)
    return architecture_known + params_known + type_known + average_known


def collapse_leaderboard_duplicates(leaderboard: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = leaderboard.copy()
    repo_revision = raw["model_id"].map(parse_repo_and_revision)
    raw["repo_id_norm"] = [item[0] for item in repo_revision]
    raw["revision_norm"] = [item[1] for item in repo_revision]
    raw["canonical_model_key"] = [
        canonical_model_key(repo_id, revision)
        for repo_id, revision in zip(raw["repo_id_norm"], raw["revision_norm"], strict=False)
    ]
    raw["completeness_score"] = _completeness_score(raw)
    raw["average_numeric"] = pd.to_numeric(raw.get("Average ⬆️"), errors="coerce").fillna(-1.0)
    canonical = (
        raw.sort_values(
            ["canonical_model_key", "completeness_score", "average_numeric"],
            ascending=[True, False, False],
        )
        .drop_duplicates(subset=["canonical_model_key"], keep="first")
        .reset_index(drop=True)
    )
    return canonical, raw


def seed_analysis_status(frame: pd.DataFrame, svd_dir: Path) -> pd.DataFrame:
    result = frame.copy()
    existing = {path.stem for path in svd_dir.glob("*.csv")}
    safe_names = result["model_id"].astype(str).str.replace("/", "--", regex=False).str.replace("@", "__", regex=False)
    result["analysis_artifact_present"] = safe_names.isin(existing)
    result["analysis_artifact_path"] = [
        str(svd_dir / f"{safe_name}.csv") if present else None
        for safe_name, present in zip(safe_names, result["analysis_artifact_present"], strict=False)
    ]
    return result


def classify_primary_type(row: pd.Series) -> str:
    relation = str(row.get("base_model_relation", "")).strip().lower()
    if relation == "source":
        return "base_source"
    if relation in {"adapter", "lora", "peft"}:
        return "adapter"
    if relation == "merge":
        return "merge"
    if relation == "quantized":
        return "quantized"
    if relation == "finetune":
        return "finetune"
    if row.get("pipeline_tag") == "image-text-to-text":
        return "multimodal"
    return "unknown"


def classify_loader_scenario_from_row(row: pd.Series) -> str:
    files = {str(item).lower() for item in row.get("files", [])}
    architecture = str(row.get("Architecture", "")).lower()
    relation = str(row.get("base_model_relation", "")).lower()
    if "adapter_config.json" in files or relation in {"adapter", "lora", "peft"}:
        return "adapter_requires_base"
    if any(name.endswith(".gguf") for name in files):
        return "quantized_alt_format"
    if row.get("pipeline_tag") == "image-text-to-text" or "conditionalgeneration" in architecture:
        return "multimodal_transformers"
    if any(method in str(row.get("quant_method", "")).lower() for method in KNOWN_QUANT_METHODS):
        return "quantized_transformers_native"
    if relation == "quantized":
        return "quantized_transformers_native"
    return "standard_transformers"
```

- [ ] **Step 5: Run the schema tests to verify they pass**

Run:

```bash
pytest tests/test_phase2_schema.py -q
```

Expected: all tests in `tests/test_phase2_schema.py` pass

- [ ] **Step 6: Commit**

```bash
git add fetch_data/__init__.py fetch_data/phase2_schema.py tests/test_phase2_schema.py
git commit -m "feat: add phase 2 schema utilities"
```

---

### Task 2: Add Lightweight Hub Preflight Probes

**Files:**
- Create: `fetch_data/hub_preflight.py`
- Test: `tests/test_hub_preflight.py`

- [ ] **Step 1: Write the failing Hub preflight tests**

Create `tests/test_hub_preflight.py`:

```python
from unittest.mock import Mock

from fetch_data.hub_preflight import (
    ProbeResult,
    classify_fetchability_from_files,
    probe_repo_fetchability,
)


def test_classify_fetchability_from_files_flags_adapter_without_base_resolution():
    result = classify_fetchability_from_files(
        repo_id="owner/adapter-model",
        revision=None,
        files=["adapter_config.json", "adapter_model.safetensors"],
        loader_scenario="adapter_requires_base",
        base_model_name_or_path=None,
    )
    assert result.fetchability_status == "public_but_structurally_incomplete"
    assert result.failure_class == "adapter_base_unresolved"


def test_classify_fetchability_from_files_accepts_standard_transformers_repo():
    result = classify_fetchability_from_files(
        repo_id="owner/base-model",
        revision=None,
        files=["config.json", "model.safetensors", "tokenizer.json"],
        loader_scenario="standard_transformers",
        base_model_name_or_path=None,
    )
    assert result.fetchability_status == "public_and_likely_fetchable"
    assert result.failure_class is None


def test_probe_repo_fetchability_uses_api_without_downloading_weights():
    api = Mock()
    api.model_info.return_value = Mock(
        private=False,
        gated=False,
        config={"model_type": "llama"},
    )
    api.list_repo_files.return_value = ["config.json", "model.safetensors"]
    result = probe_repo_fetchability(
        api=api,
        repo_id="owner/model",
        revision=None,
        loader_scenario="standard_transformers",
        base_model_name_or_path=None,
        dry_run_probe=None,
    )
    assert isinstance(result, ProbeResult)
    assert result.availability_status == "public"
    assert result.fetchability_status == "public_and_likely_fetchable"
    api.model_info.assert_called_once()
    api.list_repo_files.assert_called_once()
```

- [ ] **Step 2: Run the Hub preflight tests to verify they fail**

Run:

```bash
pytest tests/test_hub_preflight.py -q
```

Expected: import failure for `fetch_data.hub_preflight`

- [ ] **Step 3: Implement the lightweight probe module**

Create `fetch_data/hub_preflight.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from huggingface_hub.errors import HfHubHTTPError


@dataclass
class ProbeResult:
    repo_id: str
    revision: str | None
    availability_status: str
    fetchability_status: str
    failure_class: str | None
    loader_scenario: str
    config_model_type: str | None
    base_model_name_or_path: str | None
    file_count: int


def classify_fetchability_from_files(
    *,
    repo_id: str,
    revision: str | None,
    files: list[str],
    loader_scenario: str,
    base_model_name_or_path: str | None,
) -> ProbeResult:
    files_lower = {file_name.lower() for file_name in files}
    if loader_scenario == "adapter_requires_base" and not base_model_name_or_path:
        return ProbeResult(
            repo_id=repo_id,
            revision=revision,
            availability_status="public",
            fetchability_status="public_but_structurally_incomplete",
            failure_class="adapter_base_unresolved",
            loader_scenario=loader_scenario,
            config_model_type=None,
            base_model_name_or_path=base_model_name_or_path,
            file_count=len(files_lower),
        )
    if loader_scenario == "standard_transformers" and "config.json" not in files_lower:
        return ProbeResult(
            repo_id=repo_id,
            revision=revision,
            availability_status="public",
            fetchability_status="public_but_structurally_incomplete",
            failure_class="config_missing",
            loader_scenario=loader_scenario,
            config_model_type=None,
            base_model_name_or_path=base_model_name_or_path,
            file_count=len(files_lower),
        )
    return ProbeResult(
        repo_id=repo_id,
        revision=revision,
        availability_status="public",
        fetchability_status="public_and_likely_fetchable",
        failure_class=None,
        loader_scenario=loader_scenario,
        config_model_type=None,
        base_model_name_or_path=base_model_name_or_path,
        file_count=len(files_lower),
    )


def probe_repo_fetchability(
    *,
    api,
    repo_id: str,
    revision: str | None,
    loader_scenario: str,
    base_model_name_or_path: str | None,
    dry_run_probe: Callable[[str, str | None], object] | None = None,
) -> ProbeResult:
    try:
        info = api.model_info(repo_id=repo_id, revision=revision, files_metadata=False)
        files = api.list_repo_files(repo_id=repo_id, revision=revision, repo_type="model")
    except HfHubHTTPError as error:
        status_code = getattr(getattr(error, "response", None), "status_code", None)
        if status_code in {401, 403}:
            return ProbeResult(repo_id, revision, "gated", "gated", "gated", loader_scenario, None, base_model_name_or_path, 0)
        if status_code == 404:
            return ProbeResult(repo_id, revision, "private_or_deleted", "private_or_deleted", "private_or_deleted", loader_scenario, None, base_model_name_or_path, 0)
        return ProbeResult(repo_id, revision, "probe_failed", "public_but_probe_failed", f"http_{status_code}", loader_scenario, None, base_model_name_or_path, 0)

    result = classify_fetchability_from_files(
        repo_id=repo_id,
        revision=revision,
        files=files,
        loader_scenario=loader_scenario,
        base_model_name_or_path=base_model_name_or_path,
    )
    result.config_model_type = None if not isinstance(getattr(info, "config", None), dict) else info.config.get("model_type")
    if dry_run_probe is not None and result.fetchability_status == "public_and_likely_fetchable":
        dry_run_probe(repo_id, revision)
    return result
```

- [ ] **Step 4: Run the Hub preflight tests to verify they pass**

Run:

```bash
pytest tests/test_hub_preflight.py -q
```

Expected: all tests in `tests/test_hub_preflight.py` pass

- [ ] **Step 5: Commit**

```bash
git add fetch_data/hub_preflight.py tests/test_hub_preflight.py
git commit -m "feat: add phase 2 hub preflight probes"
```

---

### Task 3: Build The Phase-2 Zoo Orchestration CLI

**Files:**
- Create: `fetch_data/build_phase2_zoo.py`
- Modify: `fetch_data/phase2_schema.py`
- Test: `tests/test_build_phase2_zoo.py`

- [ ] **Step 1: Write the failing orchestration tests**

Create `tests/test_build_phase2_zoo.py`:

```python
import json
from pathlib import Path

import pandas as pd

from fetch_data.build_phase2_zoo import build_phase2_outputs


def test_build_phase2_outputs_ingests_seed_and_writes_views(tmp_path: Path):
    metadata_path = tmp_path / "sampled_metadata.csv"
    leaderboard_path = tmp_path / "sampled_leaderboard.csv"
    atlas_path = tmp_path / "large_nlp_data.json"
    svd_dir = tmp_path / "svd_results"
    output_dir = tmp_path / "curated"
    svd_dir.mkdir()
    (svd_dir / "org--seed-model.csv").write_text("alpha\\n1.0\\n")

    pd.DataFrame(
        [
            {"model_id": "org/seed-model", "source_model": "base/model", "base_model_relation": "finetune"},
            {"model_id": "org/adapter-model", "source_model": "base/model", "base_model_relation": "adapter"},
        ]
    ).to_csv(metadata_path, index=False)

    pd.DataFrame(
        [
            {
                "model_id": "org/seed-model",
                "Architecture": "LlamaForCausalLM",
                "#Params (B)": 8.0,
                "Type": "chat",
                "Average ⬆️": 30.0,
                "pipeline_tag": "text-generation",
            },
            {
                "model_id": "org/seed-model",
                "Architecture": "Unknown",
                "#Params (B)": 0.0,
                "Type": "other",
                "Average ⬆️": 10.0,
                "pipeline_tag": "text-generation",
            },
        ]
    ).to_csv(leaderboard_path, index=False)

    atlas_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "org/seed-model", "attributes": {"base_model_clean": "base/model", "base_model_relation": "finetune"}},
                    {"id": "org/new-merge", "attributes": {"base_model_clean": "base/model", "base_model_relation": "merge"}},
                    {"id": "org/new-unknown", "attributes": {"base_model_clean": "unknown", "base_model_relation": "unknown"}},
                ],
                "edges": [],
            }
        )
    )

    outputs = build_phase2_outputs(
        metadata_path=metadata_path,
        leaderboard_path=leaderboard_path,
        atlas_json_path=atlas_path,
        svd_dir=svd_dir,
        output_dir=output_dir,
        target_size=3,
        skip_hub_probes=True,
    )
    assert (output_dir / "model_zoo_phase2.csv").exists()
    assert (output_dir / "model_zoo_phase2_prediction_ready.csv").exists()
    assert outputs["canonical"]["canonical_model_key"].nunique() == len(outputs["canonical"])
    assert set(outputs["canonical"]["model_id"]) == {"org/seed-model", "org/adapter-model", "org/new-merge"}
    assert outputs["canonical"].loc[outputs["canonical"]["model_id"] == "org/seed-model", "analysis_artifact_present"].item() is True


def test_build_phase2_outputs_keeps_known_and_unknown_when_available(tmp_path: Path):
    metadata_path = tmp_path / "sampled_metadata.csv"
    leaderboard_path = tmp_path / "sampled_leaderboard.csv"
    atlas_path = tmp_path / "large_nlp_data.json"
    svd_dir = tmp_path / "svd_results"
    output_dir = tmp_path / "curated"
    svd_dir.mkdir()

    pd.DataFrame(
        [{"model_id": "org/base", "source_model": "org/base", "base_model_relation": "source"}]
    ).to_csv(metadata_path, index=False)
    pd.DataFrame([], columns=["model_id", "Architecture", "#Params (B)", "Type", "Average ⬆️", "pipeline_tag"]).to_csv(leaderboard_path, index=False)
    atlas_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "org/base", "attributes": {"base_model_clean": "org/base", "base_model_relation": "source"}},
                    {"id": "org/known-ft", "attributes": {"base_model_clean": "org/base", "base_model_relation": "finetune"}},
                    {"id": "org/unknown-ft", "attributes": {"base_model_clean": "unknown", "base_model_relation": "finetune"}},
                ],
                "edges": [],
            }
        )
    )
    outputs = build_phase2_outputs(
        metadata_path=metadata_path,
        leaderboard_path=leaderboard_path,
        atlas_json_path=atlas_path,
        svd_dir=svd_dir,
        output_dir=output_dir,
        target_size=3,
        skip_hub_probes=True,
    )
    finetunes = outputs["canonical"][outputs["canonical"]["primary_type_bucket"] == "finetune"]
    assert finetunes["known_lineage_flag"].nunique() == 2
```

- [ ] **Step 2: Run the orchestration tests to verify they fail**

Run:

```bash
pytest tests/test_build_phase2_zoo.py -q
```

Expected: import failure for `fetch_data.build_phase2_zoo`

- [ ] **Step 3: Extend the schema module with atlas ingestion and bucket helpers**

Append these helpers to `fetch_data/phase2_schema.py`:

```python
import json


def load_atlas_candidates(atlas_json_path: Path) -> pd.DataFrame:
    data = json.loads(atlas_json_path.read_text())
    rows = []
    for node in data.get("nodes", []):
        attrs = node.get("attributes", {})
        rows.append(
            {
                "model_id": node["id"],
                "source_model": attrs.get("base_model_clean"),
                "base_model_relation": attrs.get("base_model_relation", "unknown"),
            }
        )
    frame = pd.DataFrame(rows)
    repo_revision = frame["model_id"].map(parse_repo_and_revision)
    frame["repo_id_norm"] = [item[0] for item in repo_revision]
    frame["revision_norm"] = [item[1] for item in repo_revision]
    frame["canonical_model_key"] = [
        canonical_model_key(repo_id, revision)
        for repo_id, revision in zip(frame["repo_id_norm"], frame["revision_norm"], strict=False)
    ]
    return frame


def add_lineage_flags(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    source = result.get("source_model", pd.Series(index=result.index, dtype="object")).fillna("")
    result["known_lineage_flag"] = source.ne("") & source.ne("unknown") & source.ne("N/A")
    return result


def apply_type_taxonomy(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["primary_type_bucket"] = result.apply(classify_primary_type, axis=1)
    result["loader_scenario"] = result.apply(classify_loader_scenario_from_row, axis=1)
    result["is_adapter"] = result["primary_type_bucket"].eq("adapter")
    result["is_merge"] = result["primary_type_bucket"].eq("merge")
    result["is_quantized"] = result["primary_type_bucket"].eq("quantized")
    result["is_multimodal"] = result["loader_scenario"].eq("multimodal_transformers")
    return result


def select_to_target(frame: pd.DataFrame, target_size: int) -> pd.DataFrame:
    result = frame.copy()
    result["selection_priority"] = (
        result["analysis_artifact_present"].fillna(False).astype(int) * 4
        + result["known_lineage_flag"].fillna(False).astype(int) * 2
        + result["primary_type_bucket"].isin({"adapter", "quantized", "merge", "multimodal"}).astype(int)
    )
    return (
        result.sort_values(["selection_priority", "canonical_model_key"], ascending=[False, True])
        .drop_duplicates(subset=["canonical_model_key"], keep="first")
        .head(target_size)
        .reset_index(drop=True)
    )
```

- [ ] **Step 4: Implement the orchestration CLI and programmatic builder**

Create `fetch_data/build_phase2_zoo.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fetch_data.phase2_schema import (
    add_lineage_flags,
    apply_type_taxonomy,
    collapse_leaderboard_duplicates,
    load_atlas_candidates,
    normalize_repo_id,
    parse_repo_and_revision,
    seed_analysis_status,
    select_to_target,
)


def build_phase2_outputs(
    *,
    metadata_path: Path,
    leaderboard_path: Path,
    atlas_json_path: Path,
    svd_dir: Path,
    output_dir: Path,
    target_size: int,
    skip_hub_probes: bool,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path)
    leaderboard = pd.read_csv(leaderboard_path)
    leaderboard_canonical, leaderboard_raw = collapse_leaderboard_duplicates(leaderboard)
    atlas = load_atlas_candidates(atlas_json_path)

    metadata["repo_id_norm"] = metadata["model_id"].map(normalize_repo_id)
    metadata["revision_norm"] = metadata["model_id"].map(lambda value: parse_repo_and_revision(value)[1])
    metadata["canonical_model_key"] = [
        f"{repo}@{rev}" if rev else repo
        for repo, rev in zip(metadata["repo_id_norm"], metadata["revision_norm"], strict=False)
    ]

    combined = pd.concat([metadata, atlas], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["canonical_model_key"], keep="first")
    combined = combined.merge(
        leaderboard_canonical.drop(columns=["model_id"], errors="ignore"),
        on="canonical_model_key",
        how="left",
        suffixes=("", "_lb"),
    )
    combined = seed_analysis_status(combined, svd_dir)
    combined = add_lineage_flags(combined)
    combined = apply_type_taxonomy(combined)
    canonical = select_to_target(combined, target_size=target_size)

    prediction_ready = canonical[canonical["Average ⬆️"].notna()].copy() if "Average ⬆️" in canonical.columns else canonical.iloc[0:0].copy()
    analysis_ready = canonical[canonical["analysis_artifact_present"]].copy()
    probe_failures = canonical.iloc[0:0].copy()

    canonical.to_csv(output_dir / "model_zoo_phase2.csv", index=False)
    canonical.to_parquet(output_dir / "model_zoo_phase2.parquet", index=False)
    prediction_ready.to_csv(output_dir / "model_zoo_phase2_prediction_ready.csv", index=False)
    analysis_ready.to_csv(output_dir / "model_zoo_phase2_analysis_ready.csv", index=False)
    leaderboard_raw.to_csv(output_dir / "model_zoo_phase2_leaderboard_raw.csv", index=False)
    probe_failures.to_csv(output_dir / "model_zoo_phase2_probe_failures.csv", index=False)

    return {
        "canonical": canonical,
        "prediction_ready": prediction_ready,
        "analysis_ready": analysis_ready,
        "leaderboard_raw": leaderboard_raw,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the phase 2 curated model zoo.")
    parser.add_argument("--metadata-path", type=Path, default=Path("data/sampled_metadata.csv"))
    parser.add_argument("--leaderboard-path", type=Path, default=Path("data/sampled_leaderboard.csv"))
    parser.add_argument("--atlas-json-path", type=Path, default=Path("fetch_data/large_nlp_data.json"))
    parser.add_argument("--svd-dir", type=Path, default=Path("svd_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/curated"))
    parser.add_argument("--target-size", type=int, default=15000)
    parser.add_argument("--skip-hub-probes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_phase2_outputs(
        metadata_path=args.metadata_path,
        leaderboard_path=args.leaderboard_path,
        atlas_json_path=args.atlas_json_path,
        svd_dir=args.svd_dir,
        output_dir=args.output_dir,
        target_size=args.target_size,
        skip_hub_probes=args.skip_hub_probes,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the orchestration tests to verify they pass**

Run:

```bash
pytest tests/test_build_phase2_zoo.py -q
```

Expected: all tests in `tests/test_build_phase2_zoo.py` pass

- [ ] **Step 6: Commit**

```bash
git add fetch_data/phase2_schema.py fetch_data/build_phase2_zoo.py tests/test_build_phase2_zoo.py
git commit -m "feat: add phase 2 zoo build pipeline"
```

---

### Task 4: Wire In Live Hub Probes Without Full Downloads

**Files:**
- Modify: `fetch_data/build_phase2_zoo.py`
- Modify: `fetch_data/hub_preflight.py`
- Test: `tests/test_hub_preflight.py`
- Test: `tests/test_build_phase2_zoo.py`

- [ ] **Step 1: Add failing tests for probe integration**

Append to `tests/test_build_phase2_zoo.py`:

```python
from unittest.mock import Mock


def test_build_phase2_outputs_records_probe_failures(tmp_path: Path):
    metadata_path = tmp_path / "sampled_metadata.csv"
    leaderboard_path = tmp_path / "sampled_leaderboard.csv"
    atlas_path = tmp_path / "large_nlp_data.json"
    svd_dir = tmp_path / "svd_results"
    output_dir = tmp_path / "curated"
    svd_dir.mkdir()

    pd.DataFrame(
        [{"model_id": "org/model", "source_model": "base/model", "base_model_relation": "adapter"}]
    ).to_csv(metadata_path, index=False)
    pd.DataFrame([], columns=["model_id", "Architecture", "#Params (B)", "Type", "Average ⬆️", "pipeline_tag"]).to_csv(leaderboard_path, index=False)
    atlas_path.write_text(json.dumps({"nodes": [], "edges": []}))

    fake_probe = Mock(
        return_value={
            "availability_status": "public",
            "fetchability_status": "public_but_structurally_incomplete",
            "failure_class": "adapter_base_unresolved",
        }
    )

    outputs = build_phase2_outputs(
        metadata_path=metadata_path,
        leaderboard_path=leaderboard_path,
        atlas_json_path=atlas_path,
        svd_dir=svd_dir,
        output_dir=output_dir,
        target_size=1,
        skip_hub_probes=False,
        probe_fn=fake_probe,
    )
    assert fake_probe.called
    assert (output_dir / "model_zoo_phase2_probe_failures.csv").exists()
    assert outputs["probe_failures"]["failure_class"].iloc[0] == "adapter_base_unresolved"
```

- [ ] **Step 2: Run the affected tests to verify they fail**

Run:

```bash
pytest tests/test_hub_preflight.py tests/test_build_phase2_zoo.py -q
```

Expected: failure because `build_phase2_outputs` does not yet accept `probe_fn` or emit populated probe outputs

- [ ] **Step 3: Add a dry-run file probe and integrate the live probe path**

Update `fetch_data/hub_preflight.py` to add:

```python
from huggingface_hub import hf_hub_download


def dry_run_small_file(repo_id: str, revision: str | None) -> object:
    return hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
        revision=revision,
        dry_run=True,
    )
```

Update `fetch_data/build_phase2_zoo.py` so `build_phase2_outputs` accepts:

```python
    probe_fn=None,
```

and after canonical selection adds:

```python
    if not skip_hub_probes:
        from huggingface_hub import HfApi
        from fetch_data.hub_preflight import dry_run_small_file, probe_repo_fetchability

        api = HfApi()
        probe_callable = probe_fn or (
            lambda row: probe_repo_fetchability(
                api=api,
                repo_id=row["repo_id_norm"],
                revision=row.get("revision_norm"),
                loader_scenario=row["loader_scenario"],
                base_model_name_or_path=row.get("source_model"),
                dry_run_probe=dry_run_small_file,
            )
        )
        probe_rows = []
        for _, row in canonical.iterrows():
            probe = probe_callable(row)
            if isinstance(probe, dict):
                probe_rows.append(probe)
            else:
                probe_rows.append(probe.__dict__)
        probe_frame = pd.DataFrame(probe_rows)
        canonical = canonical.merge(
            probe_frame[["repo_id", "revision", "availability_status", "fetchability_status", "failure_class"]],
            left_on=["repo_id_norm", "revision_norm"],
            right_on=["repo_id", "revision"],
            how="left",
        )
        probe_failures = canonical[canonical["failure_class"].notna()].copy()
```

- [ ] **Step 4: Run the affected tests to verify they pass**

Run:

```bash
pytest tests/test_hub_preflight.py tests/test_build_phase2_zoo.py -q
```

Expected: both files pass

- [ ] **Step 5: Commit**

```bash
git add fetch_data/hub_preflight.py fetch_data/build_phase2_zoo.py tests/test_hub_preflight.py tests/test_build_phase2_zoo.py
git commit -m "feat: add live hub preflight integration"
```

---

### Task 5: Add Data-Layer Docs And Active Fetch-Data Entry Guidance

**Files:**
- Create: `fetch_data/README.md`
- Create: `data/curated/README.md`

- [ ] **Step 1: Create the curated-data directory**

Run:

```bash
mkdir -p data/curated
```

Expected: `data/curated/` exists

- [ ] **Step 2: Add the fetch-data README**

Create `fetch_data/README.md`:

```md
# Fetch Data

This directory now has one active phase-2 entrypoint:

- `build_phase2_zoo.py`

It builds the unified candidate zoo for scale-up:

- ingests the current 5k sample
- collapses duplicate leaderboard rows
- expands from `large_nlp_data.json`
- applies soft balancing
- runs lightweight Hugging Face preflight checks
- writes canonical outputs under `data/curated/`

Preserved helper scripts:

- `atlas_subsample.py`
- `check_diversity.py`
- `fetch_eval_data.py`
- `load_model.py`

These remain useful references, but they are not the main phase-2 pipeline.
```

- [ ] **Step 3: Add the curated-data README**

Create `data/curated/README.md`:

```md
# Curated Model Zoo Outputs

This directory is the canonical output root for phase-2 data curation.

Main outputs:

- `model_zoo_phase2.csv`
- `model_zoo_phase2.parquet`
- `model_zoo_phase2_prediction_ready.csv`
- `model_zoo_phase2_analysis_ready.csv`
- `model_zoo_phase2_leaderboard_raw.csv`
- `model_zoo_phase2_probe_failures.csv`

The canonical table is `model_zoo_phase2.csv`.

Derived views should be regenerated from the canonical table rather than hand-edited.
```

- [ ] **Step 4: Verify the docs exist**

Run:

```bash
find fetch_data data/curated -maxdepth 2 -type f | sort | sed -n '1,80p'
```

Expected output contains:

- `fetch_data/README.md`
- `data/curated/README.md`

- [ ] **Step 5: Commit**

```bash
git add fetch_data/README.md data/curated/README.md
git commit -m "docs: add phase 2 data curation guidance"
```

---

### Task 6: Run Verification And A Smoke Build

**Files:**
- Verify:
  - `fetch_data/phase2_schema.py`
  - `fetch_data/hub_preflight.py`
  - `fetch_data/build_phase2_zoo.py`
  - `tests/test_phase2_schema.py`
  - `tests/test_hub_preflight.py`
  - `tests/test_build_phase2_zoo.py`

- [ ] **Step 1: Run the focused phase-2 tests**

Run:

```bash
pytest tests/test_phase2_schema.py tests/test_hub_preflight.py tests/test_build_phase2_zoo.py -q
```

Expected: all three test files pass

- [ ] **Step 2: Run the full test suite**

Run:

```bash
pytest -q
```

Expected: existing suite still passes with the new phase-2 modules added

- [ ] **Step 3: Run a smoke build without live probes**

Run:

```bash
python fetch_data/build_phase2_zoo.py \
  --target-size 200 \
  --skip-hub-probes \
  --output-dir data/curated/smoke_phase2
```

Expected: these files exist:

- `data/curated/smoke_phase2/model_zoo_phase2.csv`
- `data/curated/smoke_phase2/model_zoo_phase2_prediction_ready.csv`
- `data/curated/smoke_phase2/model_zoo_phase2_analysis_ready.csv`
- `data/curated/smoke_phase2/model_zoo_phase2_leaderboard_raw.csv`

- [ ] **Step 4: Inspect the smoke build for key invariants**

Run:

```bash
python - <<'PY'
import pandas as pd
frame = pd.read_csv("data/curated/smoke_phase2/model_zoo_phase2.csv")
print("rows", len(frame))
print("unique canonical keys", frame["canonical_model_key"].nunique())
print("has duplicates", frame["canonical_model_key"].duplicated().any())
print("primary buckets", frame["primary_type_bucket"].value_counts().head(10).to_dict())
print("known lineage ratio", float(frame["known_lineage_flag"].mean()))
PY
```

Expected:

- `rows` is less than or equal to `200`
- `unique canonical keys` equals `rows`
- `has duplicates` is `False`

- [ ] **Step 5: Commit**

```bash
git add fetch_data data/curated tests
git commit -m "test: verify phase 2 curation pipeline"
```

---

## Self-Review Checklist

### Spec coverage

- unified canonical table: covered in Tasks 1, 3, and 5
- duplicate handling: covered in Task 1
- soft type-aware lineage balance: covered in Task 3
- reuse of current 5k sample and current `svd_results`: covered in Task 3
- lightweight fetchability only, no full downloads: covered in Tasks 2 and 4
- loader compatibility without redesigning the loader: covered in Tasks 1 and 2
- leaderboard as enrichment rather than sampler: covered in Task 3
- data-layer outputs under `data/curated/`: covered in Tasks 3 and 5

### Placeholder scan

- no `TODO`
- no `TBD`
- no undefined file paths
- no unspecified test commands

### Type consistency

- canonical identity uses `repo_id_norm`, `revision_norm`, and `canonical_model_key` consistently
- duplicate collapse returns `(canonical, raw)` consistently
- probe integration uses `availability_status`, `fetchability_status`, and `failure_class` consistently
- derived outputs use stable names matching the spec

