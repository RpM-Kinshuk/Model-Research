from __future__ import annotations

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
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set, dict)):
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _first_text(row: pd.Series, columns: list[str]) -> str:
    for column in columns:
        value = _text(row.get(column, ""))
        if value:
            return value
    return ""


def _lower_key(value: object) -> str:
    return _text(value).lower()


def _task_count(row: pd.Series) -> int:
    count = 0
    for column in BENCHMARK_COLUMNS:
        value = _text(row.get(column, ""))
        if value and pd.notna(pd.to_numeric(value, errors="coerce")):
            count += 1
    return count


def _coverage_flags(task_count: int, average_value: object = None) -> str:
    if task_count == len(BENCHMARK_COLUMNS):
        return "complete"
    if task_count > 0 or pd.notna(pd.to_numeric(average_value, errors="coerce")):
        return "partial"
    return "missing"


def _file_names(row: pd.Series) -> list[str]:
    files = row.get("files", [])
    if isinstance(files, str):
        return [files.strip().lower()] if files.strip() else []
    if isinstance(files, (list, tuple, set)):
        return [str(item).strip().lower() for item in files if str(item).strip()]
    return []


def _source_model_known(row: pd.Series) -> bool:
    return _lower_key(row.get("source_model", "")) not in UNKNOWN_LINEAGE_VALUES


def normalize_ablation_candidates(frame: pd.DataFrame, *, eval_source: str) -> pd.DataFrame:
    result = frame.copy()

    model_values = result.apply(lambda row: _first_text(row, MODEL_ID_COLUMNS), axis=1) if not result.empty else []
    repo_revision = [parse_repo_and_revision(value) for value in model_values]
    result["repo_id_norm"] = [item[0] for item in repo_revision]
    result["revision_norm"] = [item[1] for item in repo_revision]
    result["model_id_norm"] = [_lower_key(repo_id) for repo_id, _ in repo_revision]
    result["canonical_model_key"] = [
        canonical_model_key(repo_id, revision) for repo_id, revision in repo_revision
    ]

    result["model_sha"] = (
        result.apply(lambda row: _first_text(row, SHA_COLUMNS), axis=1) if not result.empty else []
    )
    result["average_numeric"] = pd.to_numeric(result.get("Average ⬆️"), errors="coerce")
    result["eval_task_count"] = result.apply(_task_count, axis=1) if not result.empty else []
    result["eval_coverage_flag"] = [
        _coverage_flags(task_count, average)
        for task_count, average in zip(result["eval_task_count"], result["average_numeric"], strict=False)
    ]
    result["eval_source"] = eval_source

    result = add_lineage_flags(result)
    if result.empty:
        result["primary_type_bucket"] = pd.Series(dtype="object")
        result["loader_scenario"] = pd.Series(dtype="object")
        result["is_adapter"] = pd.Series(dtype="bool")
        result["is_merge"] = pd.Series(dtype="bool")
        result["is_quantized"] = pd.Series(dtype="bool")
        result["is_moe"] = pd.Series(dtype="bool")
        result["is_multimodal"] = pd.Series(dtype="bool")
    else:
        result = apply_type_taxonomy(result)
    result.loc[result["eval_coverage_flag"].eq("missing"), "eval_status"] = "eval_missing"
    return result


def build_phase2_duplicate_keys(phase2: pd.DataFrame) -> set[str]:
    duplicate_keys: set[str] = set()
    for _, row in phase2.iterrows():
        canonical_key = _text(row.get("canonical_model_key", ""))
        if canonical_key:
            duplicate_keys.add(f"model:{_lower_key(canonical_key)}")

        model_id = _first_text(row, MODEL_ID_COLUMNS)
        repo_id, revision = parse_repo_and_revision(model_id)
        model_key = canonical_model_key(repo_id, revision)
        if model_key:
            duplicate_keys.add(f"model:{_lower_key(model_key)}")
        if repo_id:
            duplicate_keys.add(f"repo:{_lower_key(normalize_repo_id(repo_id))}")

        sha = _first_text(row, SHA_COLUMNS)
        if sha:
            duplicate_keys.add(f"sha:{_lower_key(sha)}")
    return duplicate_keys


def mark_phase2_overlaps(candidates: pd.DataFrame, duplicate_keys: set[str]) -> pd.DataFrame:
    result = candidates.copy()

    def overlaps(row: pd.Series) -> bool:
        row_keys: set[str] = set()
        canonical_key = _text(row.get("canonical_model_key", ""))
        if canonical_key:
            row_keys.add(f"model:{_lower_key(canonical_key)}")
        repo_id = _text(row.get("repo_id_norm", ""))
        if repo_id:
            row_keys.add(f"repo:{_lower_key(repo_id)}")
        sha = _first_text(row, SHA_COLUMNS)
        if sha:
            row_keys.add(f"sha:{_lower_key(sha)}")
        return bool(row_keys & duplicate_keys)

    result["phase2_overlap"] = result.apply(overlaps, axis=1) if not result.empty else []
    return result


def classify_esd_suitability(row: pd.Series) -> str:
    primary_type = _lower_key(row.get("primary_type_bucket", ""))
    loader_scenario = _lower_key(row.get("loader_scenario", ""))
    files = _file_names(row)

    if loader_scenario == "quantized_alt_format":
        return "separate_or_exclude"
    if any(name.endswith(SEPARATE_FORMAT_SUFFIXES) for name in files):
        return "separate_or_exclude"
    if loader_scenario == "adapter_requires_base" and not _source_model_known(row):
        return "separate_or_exclude"

    if primary_type in {"moe", "multimodal"}:
        return "usable_with_caution"
    if loader_scenario in {"adapter_requires_base", "multimodal_transformers", "quantized_transformers_native"}:
        return "usable_with_caution"
    return "good"
