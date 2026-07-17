from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from spectral_research.study.artifacts import (
    ARTIFACT_MANIFEST_FILE,
    ArtifactValidationError,
    validate_artifact_manifest,
)


REQUIRED_STUDY_FILES = (
    "manifest.csv",
    "feature_table.csv",
    "cluster_assignments.csv",
    "lineage_predictions.csv",
    "lineage_representative_predictions.csv",
    "lineage_metrics.json",
    "performance_predictions.csv",
    "performance_metrics.json",
    "exclusions.csv",
    "provenance.json",
    "report.md",
    ARTIFACT_MANIFEST_FILE,
)

REQUIRED_TABLE_COLUMNS = {
    "manifest.csv": ("model_id",),
    "feature_table.csv": ("model_id", "M__layer_count"),
    "cluster_assignments.csv": ("model_id", "cluster"),
    "lineage_predictions.csv": (
        "query_model_id",
        "method",
        "true_family",
        "predicted_family",
        "top_score",
        "correct",
        "abstained",
    ),
    "lineage_representative_predictions.csv": (
        "endpoint",
        "query_model_id",
        "method",
        "true_family",
        "predicted_family",
        "top_score",
        "correct",
        "abstained",
    ),
    "performance_predictions.csv": (
        "evaluation_scope",
        "model_id",
        "ancestry_component",
        "target",
        "feature_block",
        "actual",
        "prediction",
        "external_slice",
    ),
    "exclusions.csv": ("stage", "model_id", "reason", "detail"),
}

OBSERVED_FAMILY_ENDPOINT = "cross_fitted_observed_family_representative"


@dataclass(frozen=True)
class StudyOutputs:
    run_dir: Path
    provenance: Mapping[str, Any]
    manifest: pd.DataFrame
    features: pd.DataFrame
    atlas: pd.DataFrame
    cluster_assignments: pd.DataFrame
    lineage_predictions: pd.DataFrame
    representative_lineage_predictions: pd.DataFrame
    performance_predictions: pd.DataFrame
    exclusions: pd.DataFrame
    lineage_metrics: Mapping[str, Any]
    performance_metrics: Mapping[str, Any]
    lineage_summary: Mapping[str, Any]
    performance_summary: Mapping[str, Any]
    summary: Mapping[str, int]
    external_status: str


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid frozen study JSON {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Frozen study JSON must be an object: {path.name}")
    return value


def _require_unique_ids(frame: pd.DataFrame, column: str, table: str) -> None:
    if column not in frame.columns:
        raise ValueError(f"{table} is missing {column}")
    identifiers = frame[column].astype("string").str.strip()
    if identifiers.isna().any() or identifiers.eq("").any():
        raise ValueError(f"{table} contains a missing {column}")
    if identifiers.duplicated().any():
        raise ValueError(f"{table} contains duplicate {column} values")


def _require_columns(
    columns: list[str] | pd.Index,
    required: tuple[str, ...],
    table: str,
) -> None:
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"{table} is missing required columns: {missing}")


def _require_unique_key(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    table: str,
) -> None:
    if frame.duplicated(list(columns), keep=False).any():
        raise ValueError(
            f"{table} contains duplicate composite key values for {list(columns)}"
        )


def _validate_cluster_assignments(frame: pd.DataFrame) -> None:
    clusters = pd.to_numeric(frame["cluster"], errors="coerce")
    valid = (
        clusters.notna()
        & np.isfinite(clusters)
        & clusters.ge(0)
        & clusters.eq(np.floor(clusters))
    )
    if not valid.all():
        raise ValueError(
            "cluster_assignments.csv contains invalid cluster values; "
            "expected nonnegative integers"
        )
    frame["cluster"] = clusters.astype(np.int64)


def _require_close(recorded: object, observed: float, label: str) -> None:
    expected = _as_finite_float(recorded)
    if expected is None:
        raise ValueError(f"{label} is missing or non-finite")
    if not np.isclose(expected, observed, rtol=1e-10, atol=1e-10):
        raise ValueError(f"{label} mismatch: metrics={expected}, predictions={observed}")


def _root_macro_top1_from_predictions(frame: pd.DataFrame, table: str) -> float:
    tokens = frame["correct"].astype("string").str.strip().str.casefold()
    allowed = {"true", "false", "1", "0"}
    if tokens.isna().any() or not tokens.isin(allowed).all():
        raise ValueError(f"{table} contains invalid correct values")
    correct = tokens.isin({"true", "1"}).astype(float)
    families = frame["true_family"].astype("string").str.strip()
    if families.isna().any() or families.eq("").any():
        raise ValueError(f"{table} contains invalid true_family values")
    return float(
        pd.DataFrame({"family": families, "correct": correct})
        .groupby("family", sort=False)["correct"]
        .mean()
        .mean()
    )


def _component_weighted_mae(frame: pd.DataFrame, table: str) -> float:
    actual = pd.to_numeric(frame["actual"], errors="coerce")
    prediction = pd.to_numeric(frame["prediction"], errors="coerce")
    valid = np.isfinite(actual) & np.isfinite(prediction)
    if not valid.all() or frame.empty:
        raise ValueError(f"{table} contains invalid primary prediction values")
    errors = (prediction - actual).abs()
    return float(
        pd.DataFrame(
            {
                "component": frame["ancestry_component"].astype(str),
                "absolute_error": errors,
            }
        )
        .groupby("component", sort=False)["absolute_error"]
        .mean()
        .mean()
    )


def _validate_metric_table_agreement(
    lineage: pd.DataFrame,
    representative_lineage: pd.DataFrame,
    performance: pd.DataFrame,
    lineage_metrics: Mapping[str, Any],
    performance_metrics: Mapping[str, Any],
) -> None:
    closed = lineage_metrics.get(
        "closed_set", lineage_metrics.get("closed_set_metrics", {})
    )
    closed_record = _record_from_records(
        closed, method="combined", metric="root_macro_top1"
    )
    closed_expected = _as_int(closed_record.get("n_queries"))
    closed_rows = lineage.loc[lineage["method"].astype(str).eq("combined")]
    closed_observed = closed_rows["query_model_id"].astype(str).nunique()
    if closed_expected is not None and closed_expected != closed_observed:
        raise ValueError(
            "closed-set lineage n_queries mismatch: "
            f"metrics={closed_expected}, predictions={closed_observed}"
        )
    if closed_observed:
        _require_close(
            closed_record.get("estimate", closed_record.get("value")),
            _root_macro_top1_from_predictions(
                closed_rows, "lineage_predictions.csv"
            ),
            "closed-set lineage top-1",
        )

    observed = lineage_metrics.get("observed_family", {})
    observed_record = _record_from_records(
        observed, method="combined", metric="root_macro_top1"
    )
    observed_expected = _as_int(observed_record.get("n_queries"))
    observed_rows = representative_lineage.loc[
        representative_lineage["method"].astype(str).eq("combined")
    ]
    observed_count = observed_rows["query_model_id"].astype(str).nunique()
    if observed_expected is not None and observed_expected != observed_count:
        raise ValueError(
            "observed-family lineage n_queries mismatch: "
            f"metrics={observed_expected}, predictions={observed_count}"
        )
    if observed_count:
        _require_close(
            observed_record.get("estimate", observed_record.get("value")),
            _root_macro_top1_from_predictions(
                observed_rows, "lineage_representative_predictions.csv"
            ),
            "observed-family lineage top-1",
        )

    primary = performance_metrics.get("primary", {})
    if not isinstance(primary, Mapping) or not primary:
        return
    target = str(primary.get("target", "modern_average"))
    primary_rows = performance.loc[
        performance["evaluation_scope"].astype(str).eq("development")
        & performance["target"].astype(str).eq(target)
        & performance["feature_block"].astype(str).eq("M+S+H")
    ]
    expected_models = _as_int(primary.get("n_models"))
    observed_models = primary_rows["model_id"].astype(str).nunique()
    if expected_models is not None and expected_models != observed_models:
        raise ValueError(
            "primary performance n_models mismatch: "
            f"metrics={expected_models}, predictions={observed_models}"
        )
    expected_components = _as_int(primary.get("n_components"))
    observed_components = primary_rows["ancestry_component"].astype(str).nunique()
    if expected_components is not None and expected_components != observed_components:
        raise ValueError(
            "primary performance n_components mismatch: "
            f"metrics={expected_components}, predictions={observed_components}"
        )
    if observed_models:
        primary_mae = _component_weighted_mae(
            primary_rows, "performance_predictions.csv"
        )
        _require_close(
            primary.get("group_weighted_mae"),
            primary_mae,
            "primary performance MAE",
        )
        baseline_rows = performance.loc[
            performance["evaluation_scope"].astype(str).eq("development")
            & performance["target"].astype(str).eq(target)
            & performance["feature_block"].astype(str).eq("M+S")
        ]
        if not baseline_rows.empty:
            baseline_mae = _component_weighted_mae(
                baseline_rows, "performance_predictions.csv"
            )
            _require_close(
                primary.get("baseline_group_weighted_mae"),
                baseline_mae,
                "baseline performance MAE",
            )
            _require_close(
                primary.get("htsr_delta_mae"),
                primary_mae - baseline_mae,
                "primary HTSR delta MAE",
            )

    external = performance_metrics.get("external", {})
    if not isinstance(external, Mapping):
        return
    if str(external.get("status", "unavailable")) == "unavailable":
        return
    external_metrics = external.get("metrics", [])
    if not isinstance(external_metrics, list):
        raise ValueError("external performance metrics must be a list")
    for row in external_metrics:
        if not isinstance(row, Mapping):
            raise ValueError("external performance metrics contain a non-object row")
        if str(row.get("external_slice", "")) != "component_disjoint":
            continue
        block = str(row.get("feature_block", ""))
        external_target = str(row.get("target", target))
        metric_rows = performance.loc[
            performance["evaluation_scope"].astype(str).eq("external")
            & performance["target"].astype(str).eq(external_target)
            & performance["feature_block"].astype(str).eq(block)
            & performance["external_slice"].astype(str).eq("component_disjoint")
        ]
        expected_models = _as_int(row.get("n_models"))
        expected_components = _as_int(row.get("n_components"))
        observed_models = metric_rows["model_id"].astype(str).nunique()
        observed_components = metric_rows["ancestry_component"].astype(str).nunique()
        if expected_models is not None and expected_models != observed_models:
            raise ValueError(
                f"external {block} n_models mismatch: "
                f"metrics={expected_models}, predictions={observed_models}"
            )
        if expected_components is not None and expected_components != observed_components:
            raise ValueError(
                f"external {block} n_components mismatch: "
                f"metrics={expected_components}, predictions={observed_components}"
            )
        if metric_rows.empty:
            raise ValueError(f"external {block} predictions are missing")
        _require_close(
            row.get("group_weighted_mae"),
            _component_weighted_mae(metric_rows, "performance_predictions.csv"),
            f"external {block} MAE",
        )


def _as_finite_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _metric_from_records(
    section: object,
    *,
    method: str,
    metric: str,
) -> float | None:
    if isinstance(section, Mapping):
        by_method = section.get(method)
        if isinstance(by_method, Mapping):
            return _as_finite_float(by_method.get(metric))
        direct = section.get(metric)
        if direct is not None:
            return _as_finite_float(direct)
    if isinstance(section, list):
        for row in section:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("method", method)) != method:
                continue
            if str(row.get("metric", "")) != metric:
                continue
            return _as_finite_float(row.get("estimate", row.get("value")))
    return None


def _record_from_records(
    section: object,
    *,
    method: str,
    metric: str,
) -> Mapping[str, Any]:
    if isinstance(section, Mapping):
        by_method = section.get(method)
        if isinstance(by_method, Mapping):
            return by_method
        if metric in section:
            return section
    if isinstance(section, list):
        for row in section:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("method", method)) != method:
                continue
            if str(row.get("metric", "")) == metric:
                return row
    return {}


def _as_int(value: object) -> int | None:
    numeric = _as_finite_float(value)
    return int(numeric) if numeric is not None else None


def _lineage_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    closed = metrics.get("closed_set", metrics.get("closed_set_metrics", {}))
    opened = metrics.get("open_set", metrics.get("open_set_metrics", {}))
    observed = metrics.get("observed_family", {})
    closed_scope = _record_from_records(
        closed, method="combined", metric="root_macro_top1"
    )
    open_scope = _record_from_records(opened, method="combined", metric="auroc")
    observed_scope = _record_from_records(
        observed, method="combined", metric="root_macro_top1"
    )
    observed_family_top1 = _metric_from_records(
        observed, method="combined", metric="root_macro_top1"
    )
    observed_spectral_shape_top1 = _metric_from_records(
        observed, method="spectral_shape", metric="root_macro_top1"
    )
    observed_htsr_lift = (
        observed_family_top1 - observed_spectral_shape_top1
        if observed_family_top1 is not None
        and observed_spectral_shape_top1 is not None
        else None
    )
    inferential = closed_scope.get("inferential")
    return {
        "root_macro_top1": _metric_from_records(
            closed, method="combined", metric="root_macro_top1"
        ),
        "mrr": _metric_from_records(closed, method="combined", metric="mrr"),
        "unknown_auroc": _metric_from_records(
            opened, method="combined", metric="auroc"
        ),
        "oscr": _metric_from_records(opened, method="combined", metric="oscr"),
        "n_queries": _as_int(closed_scope.get("n_queries")),
        "n_families": _as_int(closed_scope.get("n_families")),
        "n_open_folds": _as_int(open_scope.get("n_folds")),
        "minimum_families": _as_int(closed_scope.get("minimum_families")),
        "inferential": inferential if isinstance(inferential, bool) else None,
        "observed_family_top1": observed_family_top1,
        "observed_spectral_shape_top1": observed_spectral_shape_top1,
        "observed_htsr_lift": observed_htsr_lift,
        "observed_family_queries": _as_int(observed_scope.get("n_queries")),
        "observed_family_families": _as_int(observed_scope.get("n_families")),
    }


def _performance_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    primary = metrics.get("primary", {})
    if not isinstance(primary, Mapping):
        primary = {}
    decision = metrics.get("primary_decision", {})
    if not isinstance(decision, Mapping):
        decision = {}
    external = metrics.get("external", {})
    if not isinstance(external, Mapping):
        external = {}
    external_rows: dict[str, Mapping[str, Any]] = {}
    if isinstance(external.get("metrics"), list):
        external_rows = {
            str(row.get("feature_block")): row
            for row in external["metrics"]
            if isinstance(row, Mapping)
            and row.get("external_slice") == "component_disjoint"
        }
    external_delta = None
    if {"M+S", "M+S+H"}.issubset(external_rows):
        baseline = _as_finite_float(external_rows["M+S"].get("group_weighted_mae"))
        htsr = _as_finite_float(external_rows["M+S+H"].get("group_weighted_mae"))
        if baseline is not None and htsr is not None:
            external_delta = htsr - baseline
    external_headline = external_rows.get("M+S+H", {})
    primary_passed = decision.get("passed")
    primary_reasons = decision.get("reasons", [])
    return {
        "target": (
            str(primary["target"]) if primary.get("target") is not None else None
        ),
        "group_weighted_mae": _as_finite_float(primary.get("group_weighted_mae")),
        "spearman": _as_finite_float(primary.get("spearman")),
        "htsr_delta_mae": _as_finite_float(primary.get("htsr_delta_mae")),
        "htsr_ci_low": _as_finite_float(primary.get("htsr_ci_low")),
        "htsr_ci_high": _as_finite_float(primary.get("htsr_ci_high")),
        "n_models": _as_int(primary.get("n_models")),
        "n_components": _as_int(primary.get("n_components")),
        "primary_passed": (
            primary_passed if isinstance(primary_passed, bool) else None
        ),
        "primary_reasons": (
            [str(reason) for reason in primary_reasons]
            if isinstance(primary_reasons, list)
            else []
        ),
        "relative_mae_improvement": _as_finite_float(
            decision.get("relative_mae_improvement")
        ),
        "external_htsr_delta_mae": external_delta,
        "external_n_models": _as_int(external_headline.get("n_models")),
        "external_n_components": _as_int(external_headline.get("n_components")),
        "external_target": (
            str(external_headline["target"])
            if external_headline.get("target") is not None
            else None
        ),
        "external_status": str(external.get("status", "unavailable")),
        "external_evidence_role": (
            str(external["evidence_role"])
            if external.get("evidence_role") is not None
            else None
        ),
    }


def _preferred_lineage_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "query_model_id" not in frame.columns:
        return pd.DataFrame(columns=["model_id"])
    selected = frame.copy()
    if "method" in selected.columns and selected["method"].eq("combined").any():
        selected = selected.loc[selected["method"].eq("combined")].copy()
    selected = selected.drop_duplicates("query_model_id", keep="first")
    keep = [
        column
        for column in (
            "query_model_id",
            "predicted_family",
            "true_family",
            "top_score",
            "correct",
            "abstained",
        )
        if column in selected.columns
    ]
    return selected.loc[:, keep].rename(columns={"query_model_id": "model_id"})


def _preferred_performance_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "model_id" not in frame.columns:
        return pd.DataFrame(columns=["model_id"])
    selected = frame.copy()
    if "feature_block" in selected.columns and selected["feature_block"].eq("M+S+H").any():
        selected = selected.loc[selected["feature_block"].eq("M+S+H")].copy()
    if "target" in selected.columns and selected["target"].eq("modern_average").any():
        selected = selected.loc[selected["target"].eq("modern_average")].copy()
    if "external_slice" in selected.columns:
        development = selected["external_slice"].isna() | selected["external_slice"].astype(str).eq("")
        if development.any():
            selected = selected.loc[development].copy()
    selected = selected.drop_duplicates("model_id", keep="first")
    keep = [column for column in ("model_id", "actual", "prediction") if column in selected.columns]
    return selected.loc[:, keep]


def _build_atlas(
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    assignments: pd.DataFrame,
    lineage: pd.DataFrame,
    representative_lineage: pd.DataFrame,
    performance: pd.DataFrame,
) -> pd.DataFrame:
    atlas = manifest.copy()
    atlas["model_id"] = atlas["model_id"].astype(str)
    atlas["display_name"] = atlas["model_id"]
    if not features.empty:
        atlas = atlas.merge(features, on="model_id", how="left", validate="one_to_one")
    if not assignments.empty:
        atlas = atlas.merge(assignments, on="model_id", how="left", validate="one_to_one")
    lineage_view = _preferred_lineage_predictions(lineage)
    canonical_model_ids = set(
        lineage_view.get("model_id", pd.Series(dtype=str)).astype(str)
    )
    if len(lineage_view.columns) > 1:
        atlas = atlas.merge(lineage_view, on="model_id", how="left", validate="one_to_one")
    representative_view = _preferred_lineage_predictions(representative_lineage)
    observed_model_ids = set(
        representative_view.get("model_id", pd.Series(dtype=str)).astype(str)
    )
    if len(representative_view.columns) > 1:
        representative_view = representative_view.rename(
            columns={
                column: f"observed_family_{column}"
                for column in representative_view.columns
                if column != "model_id"
            }
        )
        atlas = atlas.merge(
            representative_view,
            on="model_id",
            how="left",
            validate="one_to_one",
        )
        canonical_endpoint = atlas["model_id"].isin(canonical_model_ids)
        observed_endpoint = atlas["model_id"].isin(observed_model_ids)
        observed_fallback = ~canonical_endpoint & observed_endpoint
        for column in (
            "predicted_family",
            "true_family",
            "top_score",
            "correct",
            "abstained",
        ):
            observed_column = f"observed_family_{column}"
            base = atlas.get(column, pd.Series(pd.NA, index=atlas.index))
            atlas[column] = base.mask(observed_fallback, atlas[observed_column])

    canonical_endpoint = atlas["model_id"].isin(canonical_model_ids)
    observed_endpoint = atlas["model_id"].isin(observed_model_ids)
    atlas["lineage_endpoint"] = np.select(
        [canonical_endpoint, observed_endpoint],
        ["canonical_root", "observed_family"],
        default="unavailable",
    )
    performance_view = _preferred_performance_predictions(performance)
    if len(performance_view.columns) > 1:
        performance_view = performance_view.rename(
            columns={
                "actual": "modern_average",
                "prediction": "modern_average prediction",
            }
        )
        performance_view["modern_average absolute_error"] = (
            pd.to_numeric(performance_view["modern_average"], errors="coerce")
            - pd.to_numeric(performance_view["modern_average prediction"], errors="coerce")
        ).abs()
        atlas = atlas.merge(performance_view, on="model_id", how="left", validate="one_to_one")

    if "predicted_family" not in atlas.columns:
        atlas["predicted_family"] = pd.NA
    audited = atlas.get("audited_base_family", pd.Series(pd.NA, index=atlas.index))
    atlas["coarse_family"] = atlas["predicted_family"].fillna(audited).fillna("unknown")
    alpha_columns = [
        column
        for column in atlas.columns
        if column.startswith("H__") and column.endswith("__alpha__median")
    ]
    if alpha_columns:
        alpha = atlas.loc[:, alpha_columns].apply(
            pd.to_numeric, errors="coerce"
        ).where(lambda values: values.gt(0.0))
        atlas["overall_log_alpha"] = np.log1p(alpha).mean(axis=1)
    else:
        atlas["overall_log_alpha"] = np.nan
    return atlas.sort_values("model_id", kind="stable", ignore_index=True)


def load_study_outputs(run_dir: str | Path) -> StudyOutputs:
    """Load one explicit frozen run without fitting or selecting an artifact by time."""

    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Frozen study run directory does not exist: {root}")
    missing = [name for name in REQUIRED_STUDY_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"Frozen study run is incomplete; missing: {missing}")
    try:
        artifact_manifest = validate_artifact_manifest(
            root,
            expected_run_id=root.name,
            required_files=tuple(
                name
                for name in REQUIRED_STUDY_FILES
                if name != ARTIFACT_MANIFEST_FILE
            ),
        )
    except ArtifactValidationError as error:
        raise ValueError(f"Invalid frozen study artifacts: {error}") from error

    manifest = pd.read_csv(root / "manifest.csv", low_memory=False)
    feature_path = root / "feature_table.csv"
    feature_columns = pd.read_csv(feature_path, nrows=0).columns.tolist()
    dashboard_feature_columns = [
        column
        for column in feature_columns
        if column in {"model_id", "M__layer_count"}
        or (column.startswith("H__") and column.endswith("__alpha__median"))
    ]
    features = pd.read_csv(
        feature_path,
        usecols=dashboard_feature_columns,
        low_memory=False,
    )
    assignments = pd.read_csv(root / "cluster_assignments.csv", low_memory=False)
    lineage = pd.read_csv(root / "lineage_predictions.csv", low_memory=False)
    representative_lineage = pd.read_csv(
        root / "lineage_representative_predictions.csv", low_memory=False
    )
    performance = pd.read_csv(root / "performance_predictions.csv", low_memory=False)
    exclusions = pd.read_csv(root / "exclusions.csv", low_memory=False)
    provenance = _read_json(root / "provenance.json")
    lineage_metrics = _read_json(root / "lineage_metrics.json")
    performance_metrics = _read_json(root / "performance_metrics.json")

    for table_name, columns in (
        ("manifest.csv", manifest.columns),
        ("feature_table.csv", feature_columns),
        ("cluster_assignments.csv", assignments.columns),
        ("lineage_predictions.csv", lineage.columns),
        (
            "lineage_representative_predictions.csv",
            representative_lineage.columns,
        ),
        ("performance_predictions.csv", performance.columns),
        ("exclusions.csv", exclusions.columns),
    ):
        _require_columns(columns, REQUIRED_TABLE_COLUMNS[table_name], table_name)

    artifact_records = artifact_manifest["artifacts"]
    for table_name, frame in (
        ("manifest.csv", manifest),
        ("feature_table.csv", features),
        ("cluster_assignments.csv", assignments),
        ("lineage_predictions.csv", lineage),
        ("lineage_representative_predictions.csv", representative_lineage),
        ("performance_predictions.csv", performance),
        ("exclusions.csv", exclusions),
    ):
        record = artifact_records[table_name]
        observed_columns = (
            feature_columns if table_name == "feature_table.csv" else list(frame.columns)
        )
        if record.get("rows") != len(frame) or record.get("columns") != observed_columns:
            raise ValueError(
                f"{table_name} shape does not match artifact_manifest.json"
            )

    _require_unique_ids(manifest, "model_id", "manifest.csv")
    _require_unique_ids(features, "model_id", "feature_table.csv")
    _require_unique_ids(assignments, "model_id", "cluster_assignments.csv")
    _validate_cluster_assignments(assignments)
    _require_unique_key(
        lineage,
        ("query_model_id", "method"),
        "lineage_predictions.csv",
    )
    _require_unique_key(
        representative_lineage,
        ("query_model_id", "method"),
        "lineage_representative_predictions.csv",
    )
    endpoint_values = representative_lineage["endpoint"].astype("string")
    invalid_endpoints = endpoint_values.isna() | endpoint_values.ne(
        OBSERVED_FAMILY_ENDPOINT
    )
    if invalid_endpoints.any():
        unexpected_endpoints = sorted(
            endpoint_values.loc[invalid_endpoints].dropna().astype(str).unique()
        )
        raise ValueError(
            "lineage_representative_predictions.csv contains unexpected endpoint "
            f"values: {unexpected_endpoints or ['<missing>']}"
        )
    _require_unique_key(
        performance,
        (
            "evaluation_scope",
            "model_id",
            "target",
            "feature_block",
            "external_slice",
        ),
        "performance_predictions.csv",
    )
    manifest_ids = set(manifest["model_id"].astype(str))
    for table_name, frame, column in (
        ("feature_table.csv", features, "model_id"),
        ("cluster_assignments.csv", assignments, "model_id"),
        ("lineage_predictions.csv", lineage, "query_model_id"),
        (
            "lineage_representative_predictions.csv",
            representative_lineage,
            "query_model_id",
        ),
        ("performance_predictions.csv", performance, "model_id"),
    ):
        if column not in frame.columns:
            raise ValueError(f"{table_name} is missing {column}")
        outside = sorted(set(frame[column].astype(str)) - manifest_ids)
        if outside:
            raise ValueError(f"{table_name} contains models outside manifest.csv: {outside[:3]}")
    _validate_metric_table_agreement(
        lineage,
        representative_lineage,
        performance,
        lineage_metrics,
        performance_metrics,
    )
    if provenance.get("run_id") != root.name:
        raise ValueError("provenance.json run_id does not match the run directory")

    atlas = _build_atlas(
        manifest,
        features,
        assignments,
        lineage,
        representative_lineage,
        performance,
    )
    lineage_model_ids: set[str] = set()
    for frame in (lineage, representative_lineage):
        if "query_model_id" in frame.columns:
            lineage_model_ids.update(frame["query_model_id"].astype(str))
    lineage_models = len(lineage_model_ids)
    performance_models = (
        performance["model_id"].astype(str).nunique()
        if "model_id" in performance.columns
        else 0
    )
    summary = {
        "n_models": len(manifest),
        "n_feature_models": len(features),
        "n_clustered_models": len(assignments),
        "n_lineage_models": int(lineage_models),
        "n_performance_models": int(performance_models),
        "n_exclusions": len(exclusions),
    }
    external = performance_metrics.get("external", provenance.get("external", {}))
    external_status = (
        str(external.get("status", "unavailable"))
        if isinstance(external, Mapping)
        else "unavailable"
    )
    return StudyOutputs(
        run_dir=root,
        provenance=provenance,
        manifest=manifest,
        features=features,
        atlas=atlas,
        cluster_assignments=assignments,
        lineage_predictions=lineage,
        representative_lineage_predictions=representative_lineage,
        performance_predictions=performance,
        exclusions=exclusions,
        lineage_metrics=lineage_metrics,
        performance_metrics=performance_metrics,
        lineage_summary=_lineage_summary(lineage_metrics),
        performance_summary=_performance_summary(performance_metrics),
        summary=summary,
        external_status=external_status,
    )
