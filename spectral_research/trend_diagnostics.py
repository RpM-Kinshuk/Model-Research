from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from alpha_cluster_dashboard.data import AlphaRecord

PERFORMANCE_TARGETS = ("Average ⬆️", "MMLU-PRO", "MATH Lvl 5")
STRUCTURAL_CONTROL_COLUMNS = ("num_layers", "raw_module_count", "common_schema", "schema_name")
TREND_FEATURE_COLUMNS = (
    "trend_global_mean",
    "trend_global_start",
    "trend_global_end",
    "trend_global_end_minus_start",
)
BUCKET_ORDER = ("supported_candidate", "heterogeneous", "all_scored")
EXACT_PARENT_RELATIONS = {"finetune"}
ANALYSIS_COLUMNS = (
    "analysis",
    "bucket",
    "target",
    "feature",
    "control_mode",
    "n",
    "spearman",
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


def _model_identity_keys(value: object) -> tuple[str, ...]:
    text = _clean_key(value)
    if not text:
        return ()
    keys = [text, text.replace("/", "--"), text.replace("--", "/")]
    return tuple(dict.fromkeys(key for key in keys if key))


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
    by_display_name = (
        feature_frame.set_index("display_name", drop=False)
        if "display_name" in feature_frame.columns
        else pd.DataFrame()
    )
    rows: list[dict[str, object]] = []
    for row in feature_frame.to_dict(orient="records"):
        source_model = _clean_key(row.get("source_model", ""))
        relation = _clean_key(row.get("base_model_relation", "")).lower()
        if relation not in EXACT_PARENT_RELATIONS:
            continue
        parent = None
        for key in _model_identity_keys(source_model):
            if key in by_display_name.index:
                parent = by_display_name.loc[key]
                break
            if key in by_model_id.index:
                parent = by_model_id.loc[key]
                break
        if parent is None:
            continue
        if isinstance(parent, pd.DataFrame):
            parent = parent.iloc[0]
        parent_id = _clean_key(parent.get("model_id", ""))
        delta_row = {**row, "parent_model_id": parent_id}
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


def _spearman(feature: pd.Series, target: pd.Series) -> float:
    corr = spearmanr(feature.to_numpy(dtype=float), target.to_numpy(dtype=float), nan_policy="omit")
    statistic = getattr(corr, "statistic", corr[0])
    return float(statistic)


def _rank_array(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def _residualize(response: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(response.shape[0], dtype=float), covariates])
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    return response - design @ coefficients


def _partial_spearman(feature: pd.Series, target: pd.Series, controls: pd.DataFrame) -> float:
    matrix = np.column_stack(
        [
            feature.to_numpy(dtype=float),
            target.to_numpy(dtype=float),
            controls.to_numpy(dtype=float),
        ]
    )
    finite_mask = np.all(np.isfinite(matrix), axis=1)
    pair_count = int(np.count_nonzero(finite_mask))
    if pair_count < 4:
        return float("nan")

    x_values = feature.to_numpy(dtype=float)[finite_mask]
    y_values = target.to_numpy(dtype=float)[finite_mask]
    control_values = controls.to_numpy(dtype=float)[finite_mask]
    if np.nanstd(x_values) < 1e-12 or np.nanstd(y_values) < 1e-12:
        return float("nan")

    ranked_x = _rank_array(x_values)
    ranked_y = _rank_array(y_values)
    ranked_controls = np.column_stack(
        [_rank_array(control_values[:, index]) for index in range(control_values.shape[1])]
    )
    x_residual = _residualize(ranked_x, ranked_controls)
    y_residual = _residualize(ranked_y, ranked_controls)
    if np.nanstd(x_residual) < 1e-12 or np.nanstd(y_residual) < 1e-12:
        return float("nan")

    corr = pearsonr(x_residual, y_residual)
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

            control_mode = "raw"
            if use_structural_controls:
                control_mode = "structural_adjusted"
                control_columns = [column for column in structural.columns if column in subset.columns]
                spearman = _partial_spearman(subset[feature], subset[target], subset.loc[:, control_columns])
            else:
                spearman = _spearman(subset[feature], subset[target])
            rows.append(
                {
                    "analysis": analysis_name,
                    "bucket": bucket_name,
                    "target": target,
                    "feature": feature,
                    "control_mode": control_mode,
                    "n": int(len(subset)),
                    "spearman": spearman,
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


def _bucket_composition_note(bucket_counts: dict[str, int] | None) -> str:
    if not bucket_counts:
        return "- `all_scored` is a mixed-zoo aggregate across every scored row, so it should be read as context rather than the primary managed slice."

    parts = [f"`{bucket}`={count}" for bucket, count in bucket_counts.items()]
    return (
        "- `all_scored` is a mixed-zoo aggregate across every scored row, so it should be read as context rather than the primary managed slice"
        f" (current composition: {', '.join(parts)})."
    )


def render_trend_report(
    results: pd.DataFrame,
    *,
    bucket_counts: dict[str, int] | None = None,
) -> str:
    if results.empty or "analysis" not in results.columns:
        return "\n".join(
            [
                "# Performance Trend Diagnostics",
                "",
                "## Top Absolute-Trend Rows By Target And Bucket",
                "_No rows._",
                "",
                "## Top Exact-Parent Rows By Target And Bucket",
                "_No rows._",
                "",
                "## Top Family-Relative Rows By Target And Bucket",
                "_No rows._",
                "",
                "## Notes",
                "- `supported_candidate` is the primary managed slice and should be read first.",
                "- `heterogeneous` is the broader cross-zoo context slice.",
                _bucket_composition_note(bucket_counts),
                "- Exact-parent and family-relative evidence are reported separately on purpose.",
                "- Raw and lightweight structural-adjusted variants are both included when enough rows are available.",
            ]
        ) + "\n"

    def best_rows(analysis: str) -> pd.DataFrame:
        subset = results.loc[results["analysis"].eq(analysis)].copy()
        if subset.empty:
            return subset
        subset = subset.loc[subset["bucket"].isin(BUCKET_ORDER)].copy()
        if subset.empty:
            return subset
        subset["abs_spearman"] = subset["spearman"].abs()
        subset["bucket_order"] = subset["bucket"].map({bucket: index for index, bucket in enumerate(BUCKET_ORDER)})
        return (
            subset.sort_values(
                ["bucket_order", "target", "abs_spearman", "spearman"],
                ascending=[True, True, False, False],
            )
            .groupby(["bucket", "target"], as_index=False, sort=False)
            .head(1)
            .sort_values(["bucket_order", "target"], ascending=[True, True])
            .drop(columns=["abs_spearman", "bucket_order"])
            .reset_index(drop=True)
        )

    return "\n".join(
        [
            "# Performance Trend Diagnostics",
            "",
            "## Top Absolute-Trend Rows By Target And Bucket",
            _markdown_table(
                best_rows("absolute"),
                ["bucket", "target", "feature", "control_mode", "n", "spearman"],
            ),
            "",
            "## Top Exact-Parent Rows By Target And Bucket",
            _markdown_table(
                best_rows("exact_parent"),
                ["bucket", "target", "feature", "control_mode", "n", "spearman"],
            ),
            "",
            "## Top Family-Relative Rows By Target And Bucket",
            _markdown_table(
                best_rows("family_relative"),
                ["bucket", "target", "feature", "control_mode", "n", "spearman"],
            ),
            "",
            "## Notes",
            "- `supported_candidate` is the primary managed slice and should be read first.",
            "- `heterogeneous` is the broader cross-zoo context slice.",
            _bucket_composition_note(bucket_counts),
            "- Exact-parent and family-relative evidence are reported separately on purpose.",
            "- Raw and lightweight structural-adjusted variants are both included when enough rows are available.",
        ]
    ) + "\n"
