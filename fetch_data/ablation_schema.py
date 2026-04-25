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
        return [item.strip().lower() for item in files.split(",") if item.strip()]
    if isinstance(files, (list, tuple, set)):
        return [str(item).strip().lower() for item in files if str(item).strip()]
    return []


def _source_model_known(row: pd.Series) -> bool:
    return _lower_key(row.get("source_model", "")) not in UNKNOWN_LINEAGE_VALUES


def normalize_ablation_candidates(frame: pd.DataFrame, *, eval_source: str) -> pd.DataFrame:
    result = frame.copy()

    model_values = result.apply(lambda row: _first_text(row, MODEL_ID_COLUMNS), axis=1) if not result.empty else []
    if "model_id" not in result.columns:
        result["model_id"] = model_values
    else:
        result["model_id"] = [
            _text(existing) or fallback
            for existing, fallback in zip(result["model_id"], model_values, strict=False)
        ]
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
    if "files" in result.columns:
        result["files"] = result.apply(_file_names, axis=1)
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
    result["eval_status"] = result["eval_coverage_flag"].eq("missing").map({True: "eval_missing", False: "eval_present"})
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
    result = normalize_ablation_candidates(frame, eval_source=eval_source)
    result["esd_suitability"] = result.apply(classify_esd_suitability, axis=1) if not result.empty else []
    result["lineage_eval_group"] = result.apply(assign_lineage_eval_group, axis=1) if not result.empty else []
    result["lineage_source"] = result.get("lineage_source", "metadata")
    result["lineage_confidence"] = result["known_lineage_flag"].map({True: "medium", False: "unknown"})
    return mark_phase2_overlaps(result, build_phase2_duplicate_keys(phase2_frame))


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

    phase2_duplicate = working["phase2_overlap"].fillna(False)
    working.loc[phase2_duplicate, "rejection_reason"] = "phase2_duplicate"

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

    eval_present = eligible[eligible["eval_status"].eq("eval_present")]
    eval_missing = eligible[eligible["eval_status"].eq("eval_missing")]
    present_selected = _round_robin_by_type(eval_present, min(target_size, len(eval_present)))
    remaining_slots = target_size - len(present_selected)
    missing_cap = math.floor(target_size * max_eval_missing_fraction)
    missing_limit = max(0, min(remaining_slots, missing_cap))
    missing_selected = _round_robin_by_type(eval_missing, missing_limit)

    selected = pd.concat([present_selected, missing_selected], ignore_index=True, sort=False)
    selected_row_ids = set(selected["_row_id"]) if "_row_id" in selected.columns else set()

    unselected_missing = (
        working["rejection_reason"].eq("")
        & working["eval_status"].eq("eval_missing")
        & ~working["_row_id"].isin(selected_row_ids)
    )
    working.loc[unselected_missing, "rejection_reason"] = "missing_eval_over_cap"

    unselected_other = working["rejection_reason"].eq("") & ~working["_row_id"].isin(selected_row_ids)
    working.loc[unselected_other, "rejection_reason"] = "type_quota_pressure"

    rejected = working[working["rejection_reason"].ne("")].reset_index(drop=True)
    return (
        selected.drop(columns=["_row_id"], errors="ignore").reset_index(drop=True),
        rejected.drop(columns=["_row_id"], errors="ignore").reset_index(drop=True),
    )


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
    phase2_overlap = candidates.get("phase2_overlap", pd.Series(False, index=candidates.index))
    return {
        "total_candidate_rows": int(len(candidates)),
        "selected_rows": int(selected_count),
        "eval_labelled_rows": int(labelled_count),
        "missing_eval_rows": int(missing_count),
        "missing_eval_percentage": missing_percentage,
        "phase2_overlap_rows": int(phase2_overlap.fillna(False).sum()),
        "counts_by_primary_type_bucket": _value_counts(selected, "primary_type_bucket"),
        "counts_by_esd_suitability": _value_counts(selected, "esd_suitability"),
        "counts_by_lineage_status": _value_counts(selected, "lineage_status"),
        "counts_by_eval_source": _value_counts(selected, "eval_source"),
        "top_rejection_reasons": _value_counts(rejected, "rejection_reason"),
    }
