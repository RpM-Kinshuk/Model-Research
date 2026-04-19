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
    return pd.DataFrame(rows)


def build_exact_parent_delta_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    by_model_id = feature_frame.set_index("model_id", drop=False)
    rows: list[dict[str, object]] = []
    for row in feature_frame.to_dict(orient="records"):
        source_model = str(row.get("source_model", "") or "").strip()
        if not source_model or source_model not in by_model_id.index:
            continue
        parent = by_model_id.loc[source_model]
        delta_row = {**row, "parent_model_id": source_model}
        for column in TREND_FEATURE_COLUMNS:
            delta_row[f"{column}_delta"] = float(row[column]) - float(parent[column])
        for target in PERFORMANCE_TARGETS:
            delta_row[f"{target}_delta"] = float(row[target]) - float(parent[target])
        rows.append(delta_row)
    return pd.DataFrame(rows)


def build_family_relative_delta_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    anchor_key = feature_frame["audited_base_family"].fillna("").replace("", pd.NA)
    anchored = feature_frame.assign(_family_anchor=anchor_key.fillna(feature_frame["candidate_family"]))
    rows: list[dict[str, object]] = []
    for family, family_frame in anchored.groupby("_family_anchor", dropna=False):
        if pd.isna(family) or len(family_frame) < 2:
            continue
        trend_anchor = family_frame.loc[:, list(TREND_FEATURE_COLUMNS)].median(numeric_only=True)
        target_anchor = family_frame.loc[:, list(PERFORMANCE_TARGETS)].median(numeric_only=True)
        for row in family_frame.to_dict(orient="records"):
            delta_row = {**row, "family_anchor": family}
            for column in TREND_FEATURE_COLUMNS:
                delta_row[f"{column}_delta"] = float(row[column]) - float(trend_anchor[column])
            for target in PERFORMANCE_TARGETS:
                delta_row[f"{target}_delta"] = float(row[target]) - float(target_anchor[target])
            rows.append(delta_row)
    return pd.DataFrame(rows)
