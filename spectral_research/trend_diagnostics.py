from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from alpha_cluster_dashboard.data import AlphaRecord

PERFORMANCE_TARGETS = ("Average ⬆️", "MMLU-PRO", "MATH Lvl 5")
STRUCTURAL_CONTROL_COLUMNS = ("num_layers", "raw_module_count", "common_schema", "schema_name")
TREND_FEATURE_COLUMNS = (
    "trend_global_mean",
    "trend_global_start",
    "trend_global_end",
    "trend_global_end_minus_start",
)
ANALYSIS_COLUMNS = (
    "analysis",
    "bucket",
    "target",
    "feature",
    "control_mode",
    "n",
    "spearman",
    "r2",
    "mae",
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


def _bucket_mask(frame: pd.DataFrame, bucket_name: str) -> pd.Series:
    if bucket_name == "all_scored":
        return pd.Series(True, index=frame.index)
    if bucket_name == "heterogeneous":
        return frame["query_bucket"].eq("heterogeneous")
    return frame["query_bucket"].eq(bucket_name)


def _structural_design(frame: pd.DataFrame) -> pd.DataFrame:
    design = pd.DataFrame(index=frame.index)
    design["num_layers"] = pd.to_numeric(frame["num_layers"], errors="coerce")
    design["raw_module_count"] = pd.to_numeric(frame["raw_module_count"], errors="coerce")
    design["common_schema"] = frame["common_schema"].astype(float)
    schema_dummies = pd.get_dummies(frame["schema_name"].fillna("unknown"), prefix="schema")
    return pd.concat([design, schema_dummies], axis=1).fillna(0.0)


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


def _linear_fit_metrics(frame: pd.DataFrame, feature_columns: list[str], target_column: str) -> tuple[float, float]:
    fit_frame = frame.loc[:, [*feature_columns, target_column]].dropna()
    if len(fit_frame) < 4:
        return float("nan"), float("nan")

    x = fit_frame.loc[:, feature_columns].to_numpy(dtype=float)
    y = fit_frame[target_column].to_numpy(dtype=float)

    model = LinearRegression()
    model.fit(x, y)
    predictions = model.predict(x)
    try:
        r2 = float(r2_score(y, predictions))
    except ValueError:
        r2 = float("nan")
    try:
        mae = float(mean_absolute_error(y, predictions))
    except ValueError:
        mae = float("nan")
    return r2, mae


def _spearman(feature: pd.Series, target: pd.Series) -> float:
    corr = spearmanr(feature.to_numpy(dtype=float), target.to_numpy(dtype=float), nan_policy="omit")
    statistic = getattr(corr, "statistic", corr[0])
    return float(statistic)


def analyze_trend_rows(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_columns: list[str],
    analysis_name: str,
    bucket_name: str,
    use_structural_controls: bool,
) -> pd.DataFrame:
    bucket_frame = frame.loc[_bucket_mask(frame, bucket_name)].copy()
    if bucket_frame.empty:
        return pd.DataFrame(columns=ANALYSIS_COLUMNS)

    structural = _structural_design(bucket_frame) if use_structural_controls else None
    rows: list[dict[str, object]] = []

    for feature in feature_columns:
        if feature not in bucket_frame.columns:
            continue
        for target in target_columns:
            if target not in bucket_frame.columns:
                continue

            subset = bucket_frame.loc[:, [feature, target]].copy()
            if structural is not None:
                subset = subset.join(structural)
            subset = subset.dropna()
            if len(subset) < 4:
                continue

            fit_columns = [feature]
            control_mode = "raw"
            if use_structural_controls:
                fit_columns = [feature, *[column for column in structural.columns if column in subset.columns]]
                control_mode = "structural_adjusted"

            r2, mae = _linear_fit_metrics(subset, fit_columns, target)
            rows.append(
                {
                    "analysis": analysis_name,
                    "bucket": bucket_name,
                    "target": target,
                    "feature": feature,
                    "control_mode": control_mode,
                    "n": int(len(subset)),
                    "spearman": _spearman(subset[feature], subset[target]),
                    "r2": r2,
                    "mae": mae,
                }
            )

    if not rows:
        return pd.DataFrame(columns=ANALYSIS_COLUMNS)
    return pd.DataFrame(rows).reindex(columns=ANALYSIS_COLUMNS)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"

    def cell(value: object) -> str:
        if pd.isna(value):
            return "n/a"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.loc[:, columns].itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def render_trend_report(results: pd.DataFrame) -> str:
    if results.empty or "analysis" not in results.columns:
        return "\n".join(
            [
                "# Performance Trend Diagnostics",
                "",
                "## Best Absolute-Trend Rows",
                "_No rows._",
                "",
                "## Best Exact-Parent Rows",
                "_No rows._",
                "",
                "## Best Family-Relative Rows",
                "_No rows._",
                "",
                "## Notes",
                "- Exact-parent and family-relative evidence are reported separately on purpose.",
                "- Raw and lightweight structural-adjusted variants are both included when enough rows are available.",
            ]
        ) + "\n"

    def best_rows(analysis: str) -> pd.DataFrame:
        subset = results.loc[results["analysis"].eq(analysis)].copy()
        if subset.empty:
            return subset
        return (
            subset.sort_values(["target", "r2", "spearman"], ascending=[True, False, False])
            .groupby("target", as_index=False)
            .head(3)
            .reset_index(drop=True)
        )

    return "\n".join(
        [
            "# Performance Trend Diagnostics",
            "",
            "## Best Absolute-Trend Rows",
            _markdown_table(
                best_rows("absolute"),
                ["bucket", "target", "feature", "control_mode", "n", "spearman", "r2", "mae"],
            ),
            "",
            "## Best Exact-Parent Rows",
            _markdown_table(
                best_rows("exact_parent"),
                ["bucket", "target", "feature", "control_mode", "n", "spearman", "r2", "mae"],
            ),
            "",
            "## Best Family-Relative Rows",
            _markdown_table(
                best_rows("family_relative"),
                ["bucket", "target", "feature", "control_mode", "n", "spearman", "r2", "mae"],
            ),
            "",
            "## Notes",
            "- Exact-parent and family-relative evidence are reported separately on purpose.",
            "- Raw and lightweight structural-adjusted variants are both included when enough rows are available.",
        ]
    ) + "\n"
