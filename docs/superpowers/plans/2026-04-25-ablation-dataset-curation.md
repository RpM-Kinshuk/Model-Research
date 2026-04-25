# Ablation Dataset Curation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small ablation-dataset curation layer that selects up to 5,000 mostly eval-labelled, phase-2-disjoint Hugging Face models under `data/ablation/`.

**Architecture:** Reuse the existing phase-2 normalization/type helpers, add ablation-specific eval coverage, ESD suitability, overlap, and selection helpers in one focused schema module, then expose one builder script that writes candidates, selected rows, rejected rows, overlap audit, and summary JSON. Keep curation metadata-only: no model weight downloads, no ESD execution, and no changes to Model-Zoo.

**Tech Stack:** Python, pandas, pytest, optional `datasets` only when fetching Open LLM Leaderboard through an explicit CLI flag.

---

## File Structure

- Create `fetch_data/ablation_schema.py`
  - Ablation-only dataframe helpers: eval normalization, phase-2 duplicate keys, overlap marking, ESD suitability, lineage/eval grouping, selection, and summary metrics.
- Create `fetch_data/build_ablation_dataset.py`
  - CLI and callable builder that loads eval sources, optional local Hub metadata, phase-2 rows for de-duplication, and writes `data/ablation/*`.
- Create `tests/test_ablation_schema.py`
  - Unit tests for normalization, overlap detection, ESD suitability, lineage/eval groups, missing-eval cap, and rejection labels.
- Create `tests/test_build_ablation_dataset.py`
  - End-to-end tests on tiny synthetic CSV inputs to verify output files and summary metrics.
- Modify `fetch_data/README.md`
  - Add the new ablation entrypoint, expected inputs, outputs, and safe run command.

## Task 1: Ablation Schema Normalization

**Files:**
- Create: `fetch_data/ablation_schema.py`
- Create: `tests/test_ablation_schema.py`

- [ ] **Step 1: Write failing normalization and overlap tests**

Create `tests/test_ablation_schema.py` with this initial content:

```python
import pandas as pd

from fetch_data.ablation_schema import (
    build_phase2_duplicate_keys,
    classify_esd_suitability,
    mark_phase2_overlaps,
    normalize_ablation_candidates,
)


def test_normalize_ablation_candidates_adds_eval_and_identity_fields():
    frame = pd.DataFrame(
        [
            {
                "model_id": "Owner/Model@main",
                "Model sha": "abc123",
                "Average ⬆️": 42.5,
                "IFEval": 50.0,
                "BBH": 30.0,
                "MATH Lvl 5": 10.0,
                "GPQA": 20.0,
                "MUSR": 25.0,
                "MMLU-PRO": 60.0,
            }
        ]
    )

    normalized = normalize_ablation_candidates(frame, eval_source="oll_contents")
    row = normalized.iloc[0]

    assert row["repo_id_norm"] == "Owner/Model"
    assert row["revision_norm"] == "main"
    assert row["model_id_norm"] == "owner/model"
    assert row["canonical_model_key"] == "Owner/Model@main"
    assert row["model_sha"] == "abc123"
    assert row["average_numeric"] == 42.5
    assert row["eval_task_count"] == 6
    assert row["eval_coverage_flag"] == "complete"
    assert row["eval_status"] == "eval_present"
    assert row["eval_source"] == "oll_contents"


def test_normalize_ablation_candidates_marks_partial_and_missing_eval():
    frame = pd.DataFrame(
        [
            {"model_id": "org/partial", "Average ⬆️": 10.0, "IFEval": 50.0},
            {"model_id": "org/missing", "Average ⬆️": None},
        ]
    )

    normalized = normalize_ablation_candidates(frame, eval_source="oll_contents")

    assert normalized.loc[normalized["model_id"] == "org/partial", "eval_coverage_flag"].item() == "partial"
    assert normalized.loc[normalized["model_id"] == "org/missing", "eval_coverage_flag"].item() == "missing"
    assert normalized.loc[normalized["model_id"] == "org/missing", "eval_status"].item() == "eval_missing"


def test_mark_phase2_overlaps_matches_case_insensitive_repo_key_and_sha():
    phase2 = pd.DataFrame(
        [
            {"model_id": "Org/Seen", "canonical_model_key": "Org/Seen", "Model sha": "sha-seen"},
        ]
    )
    candidates = normalize_ablation_candidates(
        pd.DataFrame(
            [
                {"model_id": "org/seen", "Average ⬆️": 1.0},
                {"model_id": "org/sha-match", "Model sha": "sha-seen", "Average ⬆️": 2.0},
                {"model_id": "org/new", "Average ⬆️": 3.0},
            ]
        ),
        eval_source="oll_contents",
    )

    marked = mark_phase2_overlaps(candidates, build_phase2_duplicate_keys(phase2))

    assert marked.loc[marked["model_id"] == "org/seen", "phase2_overlap"].item() is True
    assert marked.loc[marked["model_id"] == "org/sha-match", "phase2_overlap"].item() is True
    assert marked.loc[marked["model_id"] == "org/new", "phase2_overlap"].item() is False


def test_classify_esd_suitability_uses_loader_and_lineage_signals():
    dense = pd.Series({"primary_type_bucket": "finetune", "loader_scenario": "standard_transformers"})
    moe = pd.Series({"primary_type_bucket": "moe", "loader_scenario": "standard_transformers"})
    gguf = pd.Series({"primary_type_bucket": "quantized", "loader_scenario": "quantized_alt_format", "files": ["model.gguf"]})
    adapter_missing_base = pd.Series(
        {"primary_type_bucket": "adapter", "loader_scenario": "adapter_requires_base", "source_model": "unknown"}
    )

    assert classify_esd_suitability(dense) == "good"
    assert classify_esd_suitability(moe) == "usable_with_caution"
    assert classify_esd_suitability(gguf) == "separate_or_exclude"
    assert classify_esd_suitability(adapter_missing_base) == "separate_or_exclude"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ablation_schema.py -q`

Expected: FAIL during import with `ModuleNotFoundError: No module named 'fetch_data.ablation_schema'`.

- [ ] **Step 3: Add normalization, overlap, and suitability helpers**

Create `fetch_data/ablation_schema.py` with:

```python
from __future__ import annotations

import math

import pandas as pd

from fetch_data.phase2_schema import (
    UNKNOWN_LINEAGE_VALUES,
    add_lineage_flags,
    apply_type_taxonomy,
    canonical_model_key,
    normalize_repo_id,
    parse_repo_and_revision,
)


BENCHMARK_COLUMNS = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
MODEL_ID_COLUMNS = ["model_id", "fullname", "leaderboard_repo_id", "atlas_repo_id"]
SHA_COLUMNS = ["model_sha", "Model sha", "sha", "commit_hash"]
SEPARATE_FORMAT_SUFFIXES = (".gguf", ".exl2")


def _text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _first_text(row: pd.Series, columns: list[str]) -> str:
    for column in columns:
        if column in row:
            value = _text(row[column])
            if value:
                return value
    return ""


def _lower_key(value: object) -> str:
    return _text(value).lower()


def _task_count(frame: pd.DataFrame) -> pd.Series:
    present = pd.Series(0, index=frame.index, dtype="int64")
    for column in BENCHMARK_COLUMNS:
        if column in frame.columns:
            present = present + pd.to_numeric(frame[column], errors="coerce").notna().astype("int64")
    return present


def _coverage_flags(average: pd.Series, task_count: pd.Series) -> pd.Series:
    complete = task_count.eq(len(BENCHMARK_COLUMNS))
    partial = average.notna() | task_count.gt(0)
    return pd.Series("missing", index=average.index).where(~partial, "partial").where(~complete, "complete")


def normalize_ablation_candidates(frame: pd.DataFrame, *, eval_source: str) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        result["eval_source"] = []
        return result

    model_ids = result.apply(lambda row: _first_text(row, MODEL_ID_COLUMNS), axis=1)
    existing_model_ids = result["model_id"] if "model_id" in result.columns else pd.Series("", index=result.index)
    result["model_id"] = existing_model_ids.where(
        existing_model_ids.fillna("").astype(str).str.strip().ne(""),
        model_ids,
    )
    repo_revision = result["model_id"].map(parse_repo_and_revision)
    result["repo_id_norm"] = [repo_id for repo_id, _ in repo_revision]
    result["revision_norm"] = [revision for _, revision in repo_revision]
    result["model_id_norm"] = result["repo_id_norm"].map(_lower_key)
    result["canonical_model_key"] = [
        canonical_model_key(repo_id, revision)
        for repo_id, revision in zip(result["repo_id_norm"], result["revision_norm"], strict=False)
    ]
    result["model_sha"] = result.apply(lambda row: _first_text(row, SHA_COLUMNS), axis=1)
    result["eval_source"] = result.get("eval_source", eval_source)
    result["eval_name"] = result.get("eval_name", result["model_id"])
    result["eval_score_version"] = result.get("eval_score_version", "")
    result["average_numeric"] = pd.to_numeric(result.get("Average ⬆️"), errors="coerce")
    result["eval_task_count"] = _task_count(result)
    result["eval_coverage_flag"] = _coverage_flags(result["average_numeric"], result["eval_task_count"])
    result["eval_status"] = result["eval_coverage_flag"].eq("missing").map(
        {True: "eval_missing", False: "eval_present"}
    )
    return result


def build_phase2_duplicate_keys(phase2: pd.DataFrame) -> set[str]:
    keys: set[str] = set()
    if phase2.empty:
        return keys

    for _, row in phase2.iterrows():
        for column in ["model_id", "repo_id_norm", "model_id_norm", "canonical_model_key", "fullname", *SHA_COLUMNS]:
            if column in row:
                value = _lower_key(row[column])
                if value:
                    keys.add(value)
        repo_id, revision = parse_repo_and_revision(row.get("model_id"))
        repo_id = normalize_repo_id(repo_id)
        key = canonical_model_key(repo_id, revision)
        if key:
            keys.add(key.lower())
    return keys


def _candidate_duplicate_keys(row: pd.Series) -> set[str]:
    keys: set[str] = set()
    for column in ["model_id", "repo_id_norm", "model_id_norm", "canonical_model_key", "fullname", *SHA_COLUMNS]:
        value = _lower_key(row.get(column, ""))
        if value:
            keys.add(value)
    return keys


def mark_phase2_overlaps(candidates: pd.DataFrame, duplicate_keys: set[str]) -> pd.DataFrame:
    result = candidates.copy()
    result["phase2_overlap"] = [
        bool(_candidate_duplicate_keys(row) & duplicate_keys)
        for _, row in result.iterrows()
    ]
    return result


def _file_names(row: pd.Series) -> set[str]:
    files = row.get("files", [])
    if isinstance(files, str):
        return {item.strip().lower() for item in files.split(",") if item.strip()}
    if isinstance(files, (list, tuple, set)):
        return {str(item).strip().lower() for item in files if str(item).strip()}
    return set()


def _source_model_known(row: pd.Series) -> bool:
    source_model = _lower_key(row.get("source_model", ""))
    return source_model not in UNKNOWN_LINEAGE_VALUES


def classify_esd_suitability(row: pd.Series) -> str:
    loader_scenario = _lower_key(row.get("loader_scenario", ""))
    primary_type = _lower_key(row.get("primary_type_bucket", ""))
    availability = _lower_key(row.get("availability_status", row.get("Available on the hub", "")))
    files = _file_names(row)

    if availability in {"false", "removed", "private", "gated_denied", "inaccessible"}:
        return "separate_or_exclude"
    if loader_scenario == "quantized_alt_format" or any(name.endswith(SEPARATE_FORMAT_SUFFIXES) for name in files):
        return "separate_or_exclude"
    if loader_scenario == "adapter_requires_base" and not _source_model_known(row):
        return "separate_or_exclude"
    if primary_type in {"moe", "adapter", "quantized", "multimodal", "other_text"}:
        return "usable_with_caution"
    if primary_type in {"base_source", "finetune", "merge"}:
        return "good"
    return "usable_with_caution"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ablation_schema.py -q`

Expected: PASS for 4 tests.

- [ ] **Step 5: Commit schema normalization**

```bash
git add fetch_data/ablation_schema.py tests/test_ablation_schema.py
git commit -m "Add ablation schema normalization helpers"
```

## Task 2: Ablation Selection and Summary Helpers

**Files:**
- Modify: `fetch_data/ablation_schema.py`
- Modify: `tests/test_ablation_schema.py`

- [ ] **Step 1: Add failing selection and summary tests**

Append this to `tests/test_ablation_schema.py`:

```python
from fetch_data.ablation_schema import (
    assign_lineage_eval_group,
    prepare_ablation_candidates,
    select_ablation_rows,
    summarize_ablation_outputs,
)


def test_assign_lineage_eval_group_names_comparable_scenarios():
    assert assign_lineage_eval_group(pd.Series({"base_model_relation": "finetune", "source_model": "org/base"})) == "same_base_finetune"
    assert assign_lineage_eval_group(pd.Series({"base_model_relation": "merge", "source_model": "org/base"})) == "known_merge_family"
    assert assign_lineage_eval_group(pd.Series({"base_model_relation": "adapter", "source_model": "org/base"})) == "adapter_versus_merged_base_pair"
    assert assign_lineage_eval_group(pd.Series({"base_model_relation": "source", "source_model": "org/base"})) == "base_model_family_variant"
    assert assign_lineage_eval_group(pd.Series({"base_model_relation": "unknown", "source_model": "unknown"})) == "unrelated_negative_pair"


def test_prepare_ablation_candidates_adds_type_lineage_suitability_and_overlap():
    raw = pd.DataFrame(
        [
            {
                "model_id": "org/new-chat",
                "source_model": "org/base",
                "base_model_relation": "finetune",
                "Type": "chat",
                "Average ⬆️": 12.0,
                "pipeline_tag": "text-generation",
            },
            {
                "model_id": "org/old",
                "source_model": "unknown",
                "base_model_relation": "unknown",
                "Average ⬆️": 11.0,
                "pipeline_tag": "text-generation",
            },
        ]
    )
    phase2 = pd.DataFrame([{"model_id": "org/old"}])

    prepared = prepare_ablation_candidates(raw, phase2_frame=phase2, eval_source="oll_contents")

    assert prepared.loc[prepared["model_id"] == "org/new-chat", "known_lineage_flag"].item() is True
    assert prepared.loc[prepared["model_id"] == "org/new-chat", "primary_type_bucket"].item() == "finetune"
    assert prepared.loc[prepared["model_id"] == "org/new-chat", "esd_suitability"].item() == "good"
    assert prepared.loc[prepared["model_id"] == "org/old", "phase2_overlap"].item() is True


def test_select_ablation_rows_rejects_overlaps_and_caps_missing_eval():
    raw = pd.DataFrame(
        [
            {"model_id": "org/eval-a", "source_model": "org/base", "base_model_relation": "finetune", "Average ⬆️": 1.0},
            {"model_id": "org/eval-b", "source_model": "org/base", "base_model_relation": "merge", "Average ⬆️": 2.0},
            {"model_id": "org/missing-a", "source_model": "unknown", "base_model_relation": "unknown", "Average ⬆️": None},
            {"model_id": "org/missing-b", "source_model": "unknown", "base_model_relation": "unknown", "Average ⬆️": None},
            {"model_id": "org/phase2", "source_model": "org/base", "base_model_relation": "finetune", "Average ⬆️": 3.0},
            {"model_id": "org/gguf", "source_model": "org/base", "base_model_relation": "quantized", "Average ⬆️": 4.0, "files": ["model.gguf"]},
        ]
    )
    prepared = prepare_ablation_candidates(
        raw,
        phase2_frame=pd.DataFrame([{"model_id": "org/phase2"}]),
        eval_source="oll_contents",
    )

    selected, rejected = select_ablation_rows(prepared, target_size=4, max_eval_missing_fraction=0.25)

    assert set(selected["model_id"]) == {"org/eval-a", "org/eval-b", "org/missing-a"}
    assert selected["eval_status"].eq("eval_missing").sum() == 1
    assert rejected.loc[rejected["model_id"] == "org/phase2", "rejection_reason"].item() == "phase2_duplicate"
    assert rejected.loc[rejected["model_id"] == "org/gguf", "rejection_reason"].item() == "unsupported_artifact_format"
    assert rejected.loc[rejected["model_id"] == "org/missing-b", "rejection_reason"].item() == "missing_eval_over_cap"


def test_summarize_ablation_outputs_reports_core_metrics():
    candidates = prepare_ablation_candidates(
        pd.DataFrame(
            [
                {"model_id": "org/a", "source_model": "org/base", "base_model_relation": "finetune", "Average ⬆️": 1.0},
                {"model_id": "org/b", "source_model": "unknown", "base_model_relation": "unknown", "Average ⬆️": None},
            ]
        ),
        phase2_frame=pd.DataFrame(),
        eval_source="oll_contents",
    )
    selected, rejected = select_ablation_rows(candidates, target_size=2, max_eval_missing_fraction=0.5)

    summary = summarize_ablation_outputs(candidates, selected, rejected)

    assert summary["total_candidate_rows"] == 2
    assert summary["selected_rows"] == 2
    assert summary["eval_labelled_rows"] == 1
    assert summary["missing_eval_percentage"] == 50.0
    assert summary["phase2_overlap_rows"] == 0
    assert summary["counts_by_eval_source"] == {"oll_contents": 2}
```

- [ ] **Step 2: Run selection tests to verify they fail**

Run: `pytest tests/test_ablation_schema.py -q`

Expected: FAIL with import errors for `assign_lineage_eval_group`, `prepare_ablation_candidates`, `select_ablation_rows`, and `summarize_ablation_outputs`.

- [ ] **Step 3: Add selection and summary implementation**

Append this to `fetch_data/ablation_schema.py`:

```python

def assign_lineage_eval_group(row: pd.Series) -> str:
    relation = _lower_key(row.get("base_model_relation", ""))
    if relation == "finetune" and _source_model_known(row):
        return "same_base_finetune"
    if relation == "merge" and _source_model_known(row):
        return "known_merge_family"
    if relation in {"adapter", "lora", "peft"} and _source_model_known(row):
        return "adapter_versus_merged_base_pair"
    if relation == "source":
        return "base_model_family_variant"
    return "unrelated_negative_pair"


def prepare_ablation_candidates(
    frame: pd.DataFrame,
    *,
    phase2_frame: pd.DataFrame,
    eval_source: str,
) -> pd.DataFrame:
    normalized = normalize_ablation_candidates(frame, eval_source=eval_source)
    if normalized.empty:
        return normalized
    with_lineage = add_lineage_flags(normalized)
    typed = apply_type_taxonomy(with_lineage)
    typed["esd_suitability"] = typed.apply(classify_esd_suitability, axis=1)
    typed["lineage_eval_group"] = typed.apply(assign_lineage_eval_group, axis=1)
    typed["lineage_source"] = typed.get("lineage_source", "metadata")
    typed["lineage_confidence"] = typed["known_lineage_flag"].map({True: "medium", False: "unknown"})
    return mark_phase2_overlaps(typed, build_phase2_duplicate_keys(phase2_frame))


def _unsupported_reason(row: pd.Series) -> str:
    loader_scenario = _lower_key(row.get("loader_scenario", ""))
    files = _file_names(row)
    if loader_scenario == "quantized_alt_format" or any(name.endswith(SEPARATE_FORMAT_SUFFIXES) for name in files):
        return "unsupported_artifact_format"
    if loader_scenario == "adapter_requires_base" and not _source_model_known(row):
        return "adapter_base_unresolved"
    availability = _lower_key(row.get("availability_status", row.get("Available on the hub", "")))
    if availability in {"false", "removed", "private", "gated_denied", "inaccessible"}:
        return "repo_inaccessible"
    return "unsafe_for_raw_esd"


def _priority_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["selection_priority"] = (
        result["eval_status"].eq("eval_present").astype("int64") * 16
        + result["eval_coverage_flag"].eq("complete").astype("int64") * 8
        + result["known_lineage_flag"].fillna(False).astype("int64") * 4
        + result["esd_suitability"].eq("good").astype("int64") * 2
        + result["primary_type_bucket"].isin({"merge", "adapter", "quantized", "moe"}).astype("int64")
    )
    return result.sort_values(
        ["selection_priority", "primary_type_bucket", "canonical_model_key"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def _round_robin_by_type(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    cells = {
        bucket: group.reset_index(drop=True)
        for bucket, group in frame.groupby("primary_type_bucket", sort=True)
    }
    rows: list[dict[str, object]] = []
    index = 0
    while len(rows) < limit:
        progressed = False
        for bucket in sorted(cells):
            group = cells[bucket]
            if index < len(group):
                rows.append(group.iloc[index].to_dict())
                progressed = True
                if len(rows) == limit:
                    break
        if not progressed:
            break
        index += 1
    return pd.DataFrame(rows)


def select_ablation_rows(
    candidates: pd.DataFrame,
    *,
    target_size: int = 5000,
    max_eval_missing_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        empty = candidates.copy()
        empty["rejection_reason"] = []
        return empty, empty

    working = candidates.copy()
    working["_row_id"] = range(len(working))
    working["rejection_reason"] = ""

    hard_reject = working["phase2_overlap"].fillna(False)
    working.loc[hard_reject, "rejection_reason"] = "phase2_duplicate"

    unsupported = working["rejection_reason"].eq("") & working["esd_suitability"].eq("separate_or_exclude")
    working.loc[unsupported, "rejection_reason"] = [
        _unsupported_reason(row)
        for _, row in working.loc[unsupported].iterrows()
    ]

    eligible = _priority_frame(working[working["rejection_reason"].eq("")])
    duplicate_mask = eligible.duplicated(subset=["canonical_model_key"], keep="first")
    duplicate_row_ids = set(eligible.loc[duplicate_mask, "_row_id"])
    if duplicate_row_ids:
        working.loc[
            working["_row_id"].isin(duplicate_row_ids) & working["rejection_reason"].eq(""),
            "rejection_reason",
        ] = "duplicate_lower_priority"
    eligible = eligible[~duplicate_mask].reset_index(drop=True)

    labelled = eligible[eligible["eval_status"].eq("eval_present")]
    missing = eligible[eligible["eval_status"].eq("eval_missing")]
    labelled_selected = _round_robin_by_type(labelled, min(target_size, len(labelled)))
    remaining_slots = target_size - len(labelled_selected)
    missing_cap = math.floor(target_size * max_eval_missing_fraction)
    missing_limit = max(0, min(remaining_slots, missing_cap))
    missing_selected = _round_robin_by_type(missing, missing_limit)

    selected = pd.concat([labelled_selected, missing_selected], ignore_index=True, sort=False)
    selected_keys = set(selected["canonical_model_key"].dropna())

    unselected_missing = (
        working["rejection_reason"].eq("")
        & working["eval_status"].eq("eval_missing")
        & ~working["canonical_model_key"].isin(selected_keys)
    )
    working.loc[unselected_missing, "rejection_reason"] = "missing_eval_over_cap"

    unselected_other = working["rejection_reason"].eq("") & ~working["canonical_model_key"].isin(selected_keys)
    working.loc[unselected_other, "rejection_reason"] = "type_quota_pressure"

    rejected = working[working["rejection_reason"].ne("")].reset_index(drop=True)
    return selected.drop(columns=["_row_id"], errors="ignore").reset_index(drop=True), rejected.drop(columns=["_row_id"], errors="ignore")


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns or frame.empty:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].fillna("missing").value_counts().sort_index().items()
    }


def summarize_ablation_outputs(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    rejected: pd.DataFrame,
) -> dict[str, object]:
    selected_count = len(selected)
    missing_count = int(selected["eval_status"].eq("eval_missing").sum()) if selected_count else 0
    labelled_count = selected_count - missing_count
    missing_percentage = round((missing_count / selected_count * 100.0), 4) if selected_count else 0.0
    return {
        "total_candidate_rows": int(len(candidates)),
        "selected_rows": int(selected_count),
        "eval_labelled_rows": int(labelled_count),
        "missing_eval_rows": int(missing_count),
        "missing_eval_percentage": missing_percentage,
        "phase2_overlap_rows": int(candidates.get("phase2_overlap", pd.Series(False, index=candidates.index)).fillna(False).sum()),
        "counts_by_primary_type_bucket": _value_counts(selected, "primary_type_bucket"),
        "counts_by_esd_suitability": _value_counts(selected, "esd_suitability"),
        "counts_by_lineage_status": _value_counts(selected, "lineage_status"),
        "counts_by_eval_source": _value_counts(selected, "eval_source"),
        "top_rejection_reasons": _value_counts(rejected, "rejection_reason"),
    }
```

- [ ] **Step 4: Run schema tests to verify they pass**

Run: `pytest tests/test_ablation_schema.py -q`

Expected: PASS for 8 tests.

- [ ] **Step 5: Commit selection helpers**

```bash
git add fetch_data/ablation_schema.py tests/test_ablation_schema.py
git commit -m "Add ablation dataset selection logic"
```

## Task 3: Ablation Builder and CLI

**Files:**
- Create: `fetch_data/build_ablation_dataset.py`
- Create: `tests/test_build_ablation_dataset.py`

- [ ] **Step 1: Write failing builder tests**

Create `tests/test_build_ablation_dataset.py` with:

```python
import json
from pathlib import Path

import pandas as pd

from fetch_data.build_ablation_dataset import build_ablation_outputs


def test_build_ablation_outputs_writes_expected_files(tmp_path: Path):
    contents_path = tmp_path / "contents.csv"
    phase2_path = tmp_path / "phase2.csv"
    output_dir = tmp_path / "ablation"

    pd.DataFrame(
        [
            {"model_id": "org/eval-a", "source_model": "org/base", "base_model_relation": "finetune", "Average ⬆️": 1.0, "IFEval": 2.0},
            {"model_id": "org/eval-b", "source_model": "org/base", "base_model_relation": "merge", "Average ⬆️": 2.0, "BBH": 3.0},
            {"model_id": "org/missing", "source_model": "unknown", "base_model_relation": "unknown", "Average ⬆️": None},
            {"model_id": "org/phase2", "source_model": "org/base", "base_model_relation": "finetune", "Average ⬆️": 4.0},
        ]
    ).to_csv(contents_path, index=False)
    pd.DataFrame([{"model_id": "org/phase2"}]).to_csv(phase2_path, index=False)

    outputs = build_ablation_outputs(
        contents_path=contents_path,
        phase2_path=phase2_path,
        output_dir=output_dir,
        target_size=3,
        max_eval_missing_fraction=0.34,
    )

    assert (output_dir / "ablation_candidates.csv").exists()
    assert (output_dir / "ablation_selected.csv").exists()
    assert (output_dir / "ablation_rejected.csv").exists()
    assert (output_dir / "ablation_overlap_audit.csv").exists()
    assert (output_dir / "ablation_summary.json").exists()
    assert set(outputs) == {"candidates", "selected", "rejected", "overlap_audit", "summary"}
    assert set(outputs["selected"]["model_id"]) == {"org/eval-a", "org/eval-b", "org/missing"}
    assert outputs["rejected"].loc[outputs["rejected"]["model_id"] == "org/phase2", "rejection_reason"].item() == "phase2_duplicate"

    summary = json.loads((output_dir / "ablation_summary.json").read_text())
    assert summary["selected_rows"] == 3
    assert summary["eval_labelled_rows"] == 2
    assert summary["phase2_overlap_rows"] == 1


def test_build_ablation_outputs_combines_optional_results_source(tmp_path: Path):
    contents_path = tmp_path / "contents.csv"
    results_path = tmp_path / "results.csv"
    phase2_path = tmp_path / "phase2.csv"
    output_dir = tmp_path / "ablation"

    pd.DataFrame([{"model_id": "org/content", "Average ⬆️": 1.0}]).to_csv(contents_path, index=False)
    pd.DataFrame([{"model_id": "org/result", "Average ⬆️": 2.0, "MMLU-PRO": 5.0}]).to_csv(results_path, index=False)
    pd.DataFrame([], columns=["model_id"]).to_csv(phase2_path, index=False)

    outputs = build_ablation_outputs(
        contents_path=contents_path,
        results_path=results_path,
        phase2_path=phase2_path,
        output_dir=output_dir,
        target_size=5,
    )

    assert set(outputs["candidates"]["model_id"]) == {"org/content", "org/result"}
    assert set(outputs["selected"]["eval_source"]) == {"open_llm_leaderboard_contents", "open_llm_leaderboard_results"}
```

- [ ] **Step 2: Run builder tests to verify they fail**

Run: `pytest tests/test_build_ablation_dataset.py -q`

Expected: FAIL during import with `ModuleNotFoundError: No module named 'fetch_data.build_ablation_dataset'`.

- [ ] **Step 3: Add builder implementation**

Create `fetch_data/build_ablation_dataset.py` with:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fetch_data.ablation_schema import (
    prepare_ablation_candidates,
    select_ablation_rows,
    summarize_ablation_outputs,
)


DEFAULT_OUTPUT_DIR = Path("data/ablation")
DEFAULT_PHASE2_PATH = Path("data/curated/model_zoo_phase2.csv")


def _load_csv(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    return pd.read_csv(path)


def _source_frame(path: Path, *, eval_source: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["eval_source"] = eval_source
    return frame


def _load_sources(contents_path: Path, results_path: Path | None) -> pd.DataFrame:
    frames = [_source_frame(contents_path, eval_source="open_llm_leaderboard_contents")]
    if results_path is not None:
        frames.append(_source_frame(results_path, eval_source="open_llm_leaderboard_results"))
    return pd.concat(frames, ignore_index=True, sort=False)


def build_ablation_outputs(
    *,
    contents_path: Path,
    phase2_path: Path = DEFAULT_PHASE2_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    results_path: Path | None = None,
    target_size: int = 5000,
    max_eval_missing_fraction: float = 0.15,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    source = _load_sources(contents_path, results_path)
    phase2 = _load_csv(phase2_path) if phase2_path.exists() else pd.DataFrame()
    candidates = prepare_ablation_candidates(
        source,
        phase2_frame=phase2,
        eval_source="open_llm_leaderboard_contents",
    )
    selected, rejected = select_ablation_rows(
        candidates,
        target_size=target_size,
        max_eval_missing_fraction=max_eval_missing_fraction,
    )
    overlap_audit = candidates[candidates["phase2_overlap"].fillna(False)].reset_index(drop=True)
    summary = summarize_ablation_outputs(candidates, selected, rejected)

    candidates.to_csv(output_dir / "ablation_candidates.csv", index=False)
    selected.to_csv(output_dir / "ablation_selected.csv", index=False)
    rejected.to_csv(output_dir / "ablation_rejected.csv", index=False)
    overlap_audit.to_csv(output_dir / "ablation_overlap_audit.csv", index=False)
    (output_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    return {
        "candidates": candidates,
        "selected": selected,
        "rejected": rejected,
        "overlap_audit": overlap_audit,
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the ablation curation dataset.")
    parser.add_argument("--contents-path", type=Path, required=True)
    parser.add_argument("--results-path", type=Path)
    parser.add_argument("--phase2-path", type=Path, default=DEFAULT_PHASE2_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-size", type=int, default=5000)
    parser.add_argument("--max-eval-missing-fraction", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_ablation_outputs(
        contents_path=args.contents_path,
        results_path=args.results_path,
        phase2_path=args.phase2_path,
        output_dir=args.output_dir,
        target_size=args.target_size,
        max_eval_missing_fraction=args.max_eval_missing_fraction,
    )
    summary = outputs["summary"]
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run builder tests to verify they pass**

Run: `pytest tests/test_build_ablation_dataset.py -q`

Expected: PASS for 2 tests.

- [ ] **Step 5: Run schema and builder tests together**

Run: `pytest tests/test_ablation_schema.py tests/test_build_ablation_dataset.py -q`

Expected: PASS for 10 tests.

- [ ] **Step 6: Commit builder**

```bash
git add fetch_data/build_ablation_dataset.py tests/test_build_ablation_dataset.py
git commit -m "Add ablation dataset builder"
```

## Task 4: Optional Hugging Face Source Fetching and Metadata Merge

**Files:**
- Modify: `fetch_data/build_ablation_dataset.py`
- Modify: `tests/test_build_ablation_dataset.py`

- [ ] **Step 1: Add failing tests for explicit HF dataset fetch and local Hub metadata**

Append this to `tests/test_build_ablation_dataset.py`:

```python
from fetch_data.build_ablation_dataset import load_open_llm_leaderboard_contents


def test_load_open_llm_leaderboard_contents_uses_injected_loader():
    calls = []

    class FakeDataset:
        def to_pandas(self):
            return pd.DataFrame([{"model_id": "org/fetched", "Average ⬆️": 9.0}])

    def fake_loader(dataset_id, split):
        calls.append((dataset_id, split))
        return FakeDataset()

    frame = load_open_llm_leaderboard_contents(load_dataset_fn=fake_loader)

    assert calls == [("open-llm-leaderboard/contents", "train")]
    assert frame.loc[0, "model_id"] == "org/fetched"
    assert frame.loc[0, "eval_source"] == "open_llm_leaderboard_contents"


def test_build_ablation_outputs_merges_metadata_path_before_suitability(tmp_path: Path):
    contents_path = tmp_path / "contents.csv"
    metadata_path = tmp_path / "metadata.csv"
    phase2_path = tmp_path / "phase2.csv"
    output_dir = tmp_path / "ablation"

    pd.DataFrame([{"model_id": "org/gguf", "Average ⬆️": 5.0}]).to_csv(contents_path, index=False)
    pd.DataFrame(
        [
            {
                "model_id": "org/gguf",
                "source_model": "org/base",
                "base_model_relation": "quantized",
                "files": "model.gguf,tokenizer.json",
            }
        ]
    ).to_csv(metadata_path, index=False)
    pd.DataFrame([], columns=["model_id"]).to_csv(phase2_path, index=False)

    outputs = build_ablation_outputs(
        contents_path=contents_path,
        metadata_path=metadata_path,
        phase2_path=phase2_path,
        output_dir=output_dir,
        target_size=1,
    )

    assert outputs["selected"].empty
    assert outputs["rejected"].loc[0, "rejection_reason"] == "unsupported_artifact_format"
```

- [ ] **Step 2: Run targeted tests to verify they fail**

Run: `pytest tests/test_build_ablation_dataset.py::test_load_open_llm_leaderboard_contents_uses_injected_loader tests/test_build_ablation_dataset.py::test_build_ablation_outputs_merges_metadata_path_before_suitability -q`

Expected: FAIL during collection with `ImportError` for `load_open_llm_leaderboard_contents`.

- [ ] **Step 3: Add explicit fetch helper, metadata merge, and CLI flags**

Add this import in `fetch_data/build_ablation_dataset.py`:

```python
from fetch_data.phase2_schema import canonical_model_key, parse_repo_and_revision
```

Insert this in `fetch_data/build_ablation_dataset.py` before `_load_csv`:

```python
def load_open_llm_leaderboard_contents(
    *,
    dataset_id: str = "open-llm-leaderboard/contents",
    load_dataset_fn=None,
) -> pd.DataFrame:
    if load_dataset_fn is None:
        from datasets import load_dataset

        load_dataset_fn = load_dataset
    dataset = load_dataset_fn(dataset_id, split="train")
    frame = dataset.to_pandas()
    frame["eval_source"] = "open_llm_leaderboard_contents"
    return frame


def _with_join_key(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "repo_id_norm" not in result.columns:
        model_ids = result["model_id"] if "model_id" in result.columns else pd.Series("", index=result.index)
        repo_revision = model_ids.map(parse_repo_and_revision)
        result["repo_id_norm"] = [repo_id for repo_id, _ in repo_revision]
        result["revision_norm"] = [revision for _, revision in repo_revision]
        result["canonical_model_key"] = [
            canonical_model_key(repo_id, revision)
            for repo_id, revision in zip(result["repo_id_norm"], result["revision_norm"], strict=False)
        ]
    result["_metadata_join_key"] = result["repo_id_norm"].fillna("").astype(str).str.lower()
    return result


def _merge_metadata(source: pd.DataFrame, metadata_path: Path | None) -> pd.DataFrame:
    if metadata_path is None:
        return source
    source_with_key = _with_join_key(source)
    metadata = _with_join_key(pd.read_csv(metadata_path))
    metadata_columns = [
        "source_model",
        "base_model_relation",
        "pipeline_tag",
        "Architecture",
        "Type",
        "#Params (B)",
        "quant_method",
        "files",
        "tags",
        "availability_status",
        "Available on the hub",
    ]
    available_columns = [column for column in metadata_columns if column in metadata.columns]
    metadata = metadata[["_metadata_join_key", *available_columns]].drop_duplicates(
        subset=["_metadata_join_key"],
        keep="first",
    )
    merged = source_with_key.merge(metadata, on="_metadata_join_key", how="left", suffixes=("", "_metadata"))
    for column in available_columns:
        metadata_column = f"{column}_metadata"
        if metadata_column not in merged.columns:
            continue
        if column not in merged.columns:
            merged[column] = merged[metadata_column]
        else:
            current = merged[column].fillna("").astype(str).str.strip()
            merged[column] = merged[column].where(current.ne(""), merged[metadata_column])
        merged = merged.drop(columns=[metadata_column])
    return merged.drop(columns=["_metadata_join_key"], errors="ignore")
```

Change `_load_sources` to accept fetched contents:

```python
def _load_sources(
    contents_path: Path | None,
    results_path: Path | None,
    *,
    fetch_contents: bool,
) -> pd.DataFrame:
    frames = []
    if fetch_contents:
        frames.append(load_open_llm_leaderboard_contents())
    if contents_path is not None:
        frames.append(_source_frame(contents_path, eval_source="open_llm_leaderboard_contents"))
    if results_path is not None:
        frames.append(_source_frame(results_path, eval_source="open_llm_leaderboard_results"))
    if not frames:
        raise ValueError("Provide --contents-path or --fetch-contents.")
    return pd.concat(frames, ignore_index=True, sort=False)
```

Change the `build_ablation_outputs` signature and source loading lines:

```python
def build_ablation_outputs(
    *,
    contents_path: Path | None = None,
    phase2_path: Path = DEFAULT_PHASE2_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    results_path: Path | None = None,
    metadata_path: Path | None = None,
    target_size: int = 5000,
    max_eval_missing_fraction: float = 0.15,
    fetch_contents: bool = False,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    source = _load_sources(contents_path, results_path, fetch_contents=fetch_contents)
    source = _merge_metadata(source, metadata_path)
```

Change CLI args:

```python
parser.add_argument("--contents-path", type=Path)
parser.add_argument("--fetch-contents", action="store_true")
parser.add_argument("--metadata-path", type=Path)
```

Pass `fetch_contents=args.fetch_contents` and `metadata_path=args.metadata_path` from `main()`.

- [ ] **Step 4: Run builder tests to verify they pass**

Run: `pytest tests/test_build_ablation_dataset.py -q`

Expected: PASS for 4 tests.

- [ ] **Step 5: Commit explicit HF source fetching and metadata merge**

```bash
git add fetch_data/build_ablation_dataset.py tests/test_build_ablation_dataset.py
git commit -m "Add optional ablation source fetch and metadata merge"
```

## Task 5: Documentation and Verification

**Files:**
- Modify: `fetch_data/README.md`

- [ ] **Step 1: Add README coverage for ablation curation**

Append this section to `fetch_data/README.md`:

```markdown

## Ablation Dataset Curation

The ablation entrypoint is:

- `build_ablation_dataset.py`

It builds a second, phase-2-disjoint candidate set for Wasserstein/raw-eigenvalue and ESD ablations:

- prioritizes Open LLM Leaderboard eval-labelled models
- caps missing eval rows with `--max-eval-missing-fraction`
- rejects phase-2 duplicates using normalized repo ids, canonical keys, and model sha
- can merge local Hub metadata with `--metadata-path` before type and suitability labeling
- labels `esd_suitability` as `good`, `usable_with_caution`, or `separate_or_exclude`
- writes outputs under `data/ablation/`

Safe local-file run:

```bash
python fetch_data/build_ablation_dataset.py \
  --contents-path data/sampled_leaderboard.csv \
  --metadata-path data/sampled_metadata.csv \
  --phase2-path data/curated/model_zoo_phase2.csv \
  --output-dir data/ablation \
  --target-size 5000 \
  --max-eval-missing-fraction 0.15
```

Network-backed Open LLM Leaderboard run:

```bash
python fetch_data/build_ablation_dataset.py \
  --fetch-contents \
  --phase2-path data/curated/model_zoo_phase2.csv \
  --output-dir data/ablation \
  --target-size 5000 \
  --max-eval-missing-fraction 0.15
```

Expected outputs:

- `data/ablation/ablation_candidates.csv`
- `data/ablation/ablation_selected.csv`
- `data/ablation/ablation_rejected.csv`
- `data/ablation/ablation_overlap_audit.csv`
- `data/ablation/ablation_summary.json`
```

- [ ] **Step 2: Run focused tests**

Run: `pytest tests/test_ablation_schema.py tests/test_build_ablation_dataset.py -q`

Expected: PASS for 12 tests.

- [ ] **Step 3: Run related curation regression tests**

Run: `pytest tests/test_phase2_schema.py tests/test_build_phase2_zoo.py tests/test_ablation_schema.py tests/test_build_ablation_dataset.py -q`

Expected: PASS.

- [ ] **Step 4: Run compile check**

Run: `python3 -m py_compile fetch_data/ablation_schema.py fetch_data/build_ablation_dataset.py`

Expected: command exits with status 0.

- [ ] **Step 5: Inspect changed files**

Run: `git diff --stat`

Expected changed paths:

```text
fetch_data/README.md
fetch_data/ablation_schema.py
fetch_data/build_ablation_dataset.py
tests/test_ablation_schema.py
tests/test_build_ablation_dataset.py
```

- [ ] **Step 6: Commit documentation**

```bash
git add fetch_data/README.md
git commit -m "Document ablation dataset curation"
```

## Final Verification

- [ ] Run: `pytest tests/test_phase2_schema.py tests/test_build_phase2_zoo.py tests/test_ablation_schema.py tests/test_build_ablation_dataset.py -q`
- [ ] Run: `python3 -m py_compile fetch_data/ablation_schema.py fetch_data/build_ablation_dataset.py`
- [ ] Run: `git status --short`
- [ ] Confirm no generated `data/ablation/*.csv` or `data/ablation/*.json` files were committed unless the user explicitly requested dataset materialization.

## Spec Coverage Review

- Eval-first target and 15% missing-eval cap: Task 2 selection logic and Task 3 builder defaults.
- Phase-2 de-duplication: Task 1 duplicate keys and overlap marking; Task 2 rejection reason `phase2_duplicate`.
- Diverse model zoo without over-weighting noisy formats: Task 1 `esd_suitability`; Task 2 round-robin selection by `primary_type_bucket`.
- Hugging Face source and metadata use: Task 4 explicit Open LLM Leaderboard fetch and optional local Hub metadata merge.
- Preserve methods/results reproducibility: Task 3 writes candidates, selected, rejected, overlap audit, and summary JSON.
- No model downloads during curation: all tasks use CSV/HF dataset metadata only; no loader or ESD runner is invoked.
- Generic path naming: Task 3 writes to `data/ablation/`, not a target-size-specific directory.
