from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_cluster_dashboard.data import AlphaRecord

PERFORMANCE_TARGETS = ("Average ⬆️", "MMLU-PRO", "MATH Lvl 5")
STRUCTURAL_CONTROL_COLUMNS = ("num_layers", "raw_module_count", "common_schema", "schema_name")
TREND_FEATURE_COLUMNS = (
    "trend_global_mean",
    "trend_global_start",
    "trend_global_end",
    "trend_global_end_minus_start",
)


def _clean_key(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text


def _absolute_trend_columns(dataset: pd.DataFrame) -> list[str]:
    columns = list(dataset.columns)
    for column in TREND_FEATURE_COLUMNS:
        if column not in columns:
            columns.append(column)
    return columns


def _exact_parent_delta_columns(feature_frame: pd.DataFrame) -> list[str]:
    columns = list(feature_frame.columns)
    if "parent_model_id" not in columns:
        columns.append("parent_model_id")
    for column in TREND_FEATURE_COLUMNS:
        delta_column = f"{column}_delta"
        if delta_column not in columns:
            columns.append(delta_column)
    for target in PERFORMANCE_TARGETS:
        delta_column = f"{target}_delta"
        if delta_column not in columns:
            columns.append(delta_column)
    return columns


def _family_relative_delta_columns(feature_frame: pd.DataFrame) -> list[str]:
    columns = list(feature_frame.columns)
    if "family_anchor" not in columns:
        columns.append("family_anchor")
    for column in TREND_FEATURE_COLUMNS:
        delta_column = f"{column}_delta"
        if delta_column not in columns:
            columns.append(delta_column)
    for target in PERFORMANCE_TARGETS:
        delta_column = f"{target}_delta"
        if delta_column not in columns:
            columns.append(delta_column)
    return columns


def _safe_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _record_trend_features(record: AlphaRecord) -> dict[str, float]:
    alpha = np.asarray(record.alpha, dtype=float)
    if alpha.size == 0:
        return {column: float("nan") for column in TREND_FEATURE_COLUMNS}

    start_level = _safe_mean(alpha[0])
    end_level = _safe_mean(alpha[-1])
    global_mean = _safe_mean(alpha.reshape(-1))
    delta = end_level - start_level if np.isfinite(start_level) and np.isfinite(end_level) else float("nan")
    return {
        "trend_global_mean": global_mean,
        "trend_global_start": start_level,
        "trend_global_end": end_level,
        "trend_global_end_minus_start": delta,
    }


def build_absolute_trend_frame(
    dataset: pd.DataFrame,
    records_by_model_id: dict[str, AlphaRecord],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in dataset.to_dict(orient="records"):
        model_id = str(row["model_id"])
        record = records_by_model_id.get(model_id)
        if record is None:
            continue
        rows.append({**row, **_record_trend_features(record)})
    return pd.DataFrame(rows).reindex(columns=_absolute_trend_columns(dataset))


def build_exact_parent_delta_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    by_model_id = feature_frame.set_index("model_id", drop=False)
    rows: list[dict[str, object]] = []
    for row in feature_frame.to_dict(orient="records"):
        source_model = _clean_key(row.get("source_model", ""))
        if not source_model or source_model not in by_model_id.index:
            continue
        parent = by_model_id.loc[source_model]
        delta_row = {**row, "parent_model_id": source_model}
        for column in TREND_FEATURE_COLUMNS:
            delta_row[f"{column}_delta"] = float(row[column]) - float(parent[column])
        for target in PERFORMANCE_TARGETS:
            delta_row[f"{target}_delta"] = float(row[target]) - float(parent[target])
        rows.append(delta_row)
    return pd.DataFrame(rows).reindex(columns=_exact_parent_delta_columns(feature_frame))


def _family_anchor_key(row: pd.Series) -> str:
    audited_family = _clean_key(row.get("audited_base_family", ""))
    if audited_family:
        return audited_family
    return _clean_key(row.get("candidate_family", ""))


def build_family_relative_delta_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    family_keys = feature_frame.apply(_family_anchor_key, axis=1)
    rows: list[dict[str, object]] = []
    for family, family_frame in feature_frame.groupby(family_keys, dropna=False):
        family_key = _clean_key(family)
        if not family_key or len(family_frame) < 2:
            continue
        trend_anchor = family_frame.loc[:, list(TREND_FEATURE_COLUMNS)].median(numeric_only=True)
        target_anchor = family_frame.loc[:, list(PERFORMANCE_TARGETS)].median(numeric_only=True)
        for row in family_frame.to_dict(orient="records"):
            delta_row = {**row, "family_anchor": family_key}
            for column in TREND_FEATURE_COLUMNS:
                delta_row[f"{column}_delta"] = float(row[column]) - float(trend_anchor[column])
            for target in PERFORMANCE_TARGETS:
                delta_row[f"{target}_delta"] = float(row[target]) - float(target_anchor[target])
            rows.append(delta_row)
    return pd.DataFrame(rows).reindex(columns=_family_relative_delta_columns(feature_frame))
