from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from spectral_research.study.artifacts import validate_artifact_manifest


LEADING_PERFORMANCE_METHOD = "alpha_profile_knn3"
BASELINE_PERFORMANCE_METHOD = "training_mean"
DASHBOARD_ARTIFACT_FILES = (
    "metadata.json",
    "population_ledger.csv",
    "lineage_cluster_assignments.csv",
    "lineage_internal_metrics.csv",
    "lineage_cluster_selection.csv",
    "lineage_validation_metrics.csv",
    "lineage_gallery_metrics.csv",
    "performance_predictions.csv",
    "performance_metrics.csv",
    "performance_paper_summary.csv",
    "performance_leakage_audit.csv",
    "performance_cluster_holdout_metrics.csv",
    "performance_target_audit.csv",
    "report.md",
)
REQUIRED_SCALEUP_FILES = (*DASHBOARD_ARTIFACT_FILES, "artifact_manifest.json")


@dataclass(frozen=True)
class ScaleupOutputs:
    run_dir: Path
    metadata: Mapping[str, Any]
    counts: Mapping[str, int]
    population: pd.DataFrame
    predictions: Mapping[str, pd.DataFrame]
    performance: Mapping[str, Mapping[str, Any]]
    lineage_assignments: pd.DataFrame
    lineage_internal_metrics: pd.DataFrame
    lineage_cluster_selection: pd.DataFrame
    lineage_validation_metrics: pd.DataFrame
    lineage_gallery_metrics: pd.DataFrame
    performance_metrics: pd.DataFrame
    performance_paper_summary: pd.DataFrame
    performance_leakage_audit: pd.DataFrame
    performance_cluster_holdout_metrics: pd.DataFrame
    performance_target_audit: pd.DataFrame
    report: str
    role_masking_patterns: pd.DataFrame = field(default_factory=pd.DataFrame)
    role_coverage_selection: pd.DataFrame = field(default_factory=pd.DataFrame)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid scale-up metadata: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("Scale-up metadata must be an object")
    return value


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _require_false(frame: pd.DataFrame, column: str, name: str) -> None:
    values = frame[column].astype("string").str.casefold()
    if not values.eq("false").all():
        raise ValueError(f"{name} must record {column}=False for every row")


def _prediction_tables(
    frame: pd.DataFrame,
    *,
    leading_method: str = LEADING_PERFORMANCE_METHOD,
) -> dict[str, pd.DataFrame]:
    _require_columns(
        frame,
        (
            "generation",
            "model_id",
            "target",
            "feature_block",
            "actual",
            "prediction",
            "nearest_neighbors",
        ),
        "performance_predictions.csv",
    )
    selected = frame.loc[
        frame["feature_block"].astype(str).eq(leading_method)
    ].copy()
    selected["actual"] = pd.to_numeric(selected["actual"], errors="coerce")
    selected["prediction"] = pd.to_numeric(selected["prediction"], errors="coerce")
    finite = np.isfinite(selected["actual"]) & np.isfinite(selected["prediction"])
    if not finite.all():
        raise ValueError("Scale-up leading predictions contain non-finite values")
    if selected.duplicated(["target", "model_id"]).any():
        raise ValueError("Scale-up leading predictions contain duplicate model targets")

    tables: dict[str, pd.DataFrame] = {}
    for target, group in selected.groupby("target", sort=False):
        table = group.copy()
        table["display_name"] = table["model_id"].astype(str)
        table["absolute_error"] = (table["actual"] - table["prediction"]).abs()
        tables[str(target)] = table.reset_index(drop=True)
    if not tables:
        raise ValueError(
            f"Scale-up run contains no predictions for {leading_method!r}"
        )
    return tables


def _performance_records(
    metrics: pd.DataFrame,
    cluster_holdout_metrics: pd.DataFrame,
    *,
    leading_method: str = LEADING_PERFORMANCE_METHOD,
) -> dict[str, dict[str, Any]]:
    required = (
        "evaluation_scope",
        "generation",
        "target",
        "feature_block",
        "n_models",
        "n_profile_groups",
        "model_weighted_mae",
        "spearman",
        "prediction_inputs",
        "lineage_metadata_used",
    )
    _require_columns(metrics, required, "performance_metrics.csv")
    _require_columns(
        cluster_holdout_metrics,
        required,
        "performance_cluster_holdout_metrics.csv",
    )
    _require_false(metrics, "lineage_metadata_used", "performance_metrics.csv")
    _require_false(
        cluster_holdout_metrics,
        "lineage_metadata_used",
        "performance_cluster_holdout_metrics.csv",
    )

    leading = metrics.loc[
        metrics["feature_block"].astype(str).eq(leading_method)
    ]
    baseline = metrics.loc[
        metrics["feature_block"].astype(str).eq(BASELINE_PERFORMANCE_METHOD)
    ]
    stress = cluster_holdout_metrics.loc[
        cluster_holdout_metrics["feature_block"].astype(str).eq(
            leading_method
        )
    ]

    records: dict[str, dict[str, Any]] = {}
    for _, row in leading.iterrows():
        target = str(row["target"])
        generation = str(row["generation"])
        if target in records:
            raise ValueError(f"Duplicate scale-up metric target: {target}")
        baseline_row = baseline.loc[
            baseline["target"].astype(str).eq(target)
            & baseline["generation"].astype(str).eq(generation)
        ]
        stress_row = stress.loc[
            stress["target"].astype(str).eq(target)
            & stress["generation"].astype(str).eq(generation)
        ]
        baseline_mae = (
            baseline_row.iloc[0]["model_weighted_mae"]
            if len(baseline_row) == 1
            else np.nan
        )
        cluster_holdout = stress_row.iloc[0] if len(stress_row) == 1 else None
        records[target] = {
            "generation": generation,
            "target": target,
            "feature_block": leading_method,
            "model_weighted_mae": row["model_weighted_mae"],
            "baseline_mae": baseline_mae,
            "delta_mae": row["model_weighted_mae"] - baseline_mae,
            "spearman": row["spearman"],
            "kendall_tau": row.get("kendall_tau"),
            "r2": row.get("r2"),
            "n_models": row["n_models"],
            "n_profile_groups": row["n_profile_groups"],
            "cluster_holdout_mae": (
                cluster_holdout["model_weighted_mae"]
                if cluster_holdout is not None
                else np.nan
            ),
            "cluster_holdout_spearman": (
                cluster_holdout["spearman"]
                if cluster_holdout is not None
                else np.nan
            ),
        }
    return records


def _validate_population(population: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        population,
        (
            "model_id",
            "predicted_cluster",
            "spectral_medoid_model",
            "nearest_neighbors",
            "htsr_profile_available",
        ),
        "population_ledger.csv",
    )
    model_ids = population["model_id"].astype("string").str.strip()
    if model_ids.isna().any() or model_ids.eq("").any() or model_ids.duplicated().any():
        raise ValueError("population_ledger.csv contains invalid model IDs")
    clusters = pd.to_numeric(population["predicted_cluster"], errors="coerce")
    valid = clusters.isna() | (clusters.ge(0) & clusters.eq(np.floor(clusters)))
    if not valid.all():
        raise ValueError("population_ledger.csv contains invalid predicted_cluster values")
    result = population.copy()
    result["predicted_cluster"] = clusters.astype("Int64")
    return result


def load_scaleup_outputs(run_dir: str | Path) -> ScaleupOutputs:
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Scale-up run directory does not exist: {root}")
    missing = [name for name in REQUIRED_SCALEUP_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"Scale-up run is incomplete; missing: {missing}")

    metadata = _read_json(root / "metadata.json")
    if str(metadata.get("run_id", "")) != root.name:
        raise ValueError(
            f"Scale-up run_id does not match directory: {metadata.get('run_id')!r} != {root.name!r}"
        )
    validate_artifact_manifest(
        root,
        required_files=DASHBOARD_ARTIFACT_FILES,
        expected_run_id=root.name,
    )
    method_boundary = metadata.get("method_boundary", {})
    if not isinstance(method_boundary, dict) or method_boundary.get("metadata_role") != "post_freeze_validation_only":
        raise ValueError("Scale-up run does not declare metadata as post-freeze validation only")
    raw_counts = metadata.get("counts", {})
    counts = {
        str(key): int(value)
        for key, value in raw_counts.items()
        if isinstance(value, (int, float)) and np.isfinite(value)
    }

    population = _validate_population(
        pd.read_csv(root / "population_ledger.csv", low_memory=False)
    )
    assignments = pd.read_csv(
        root / "lineage_cluster_assignments.csv", low_memory=False
    )
    _require_columns(
        assignments,
        (
            "model_id",
            "predicted_cluster",
            "spectral_medoid_model",
            "nearest_neighbors",
        ),
        "lineage_cluster_assignments.csv",
    )
    if assignments["model_id"].astype(str).duplicated().any():
        raise ValueError("lineage_cluster_assignments.csv contains duplicate model IDs")

    lineage_internal = pd.read_csv(
        root / "lineage_internal_metrics.csv", low_memory=False
    )
    _require_columns(
        lineage_internal,
        (
            "n_models",
            "n_clusters",
            "silhouette_cosine",
            "mean_seed_stability_ari",
            "metadata_loaded_before_freeze",
        ),
        "lineage_internal_metrics.csv",
    )
    _require_false(
        lineage_internal,
        "metadata_loaded_before_freeze",
        "lineage_internal_metrics.csv",
    )
    cluster_selection = pd.read_csv(
        root / "lineage_cluster_selection.csv", low_memory=False
    )
    _require_columns(
        cluster_selection,
        ("n_clusters", "silhouette_cosine", "selection_uses_metadata", "selected"),
        "lineage_cluster_selection.csv",
    )
    _require_false(
        cluster_selection,
        "selection_uses_metadata",
        "lineage_cluster_selection.csv",
    )
    validation_metrics = pd.read_csv(
        root / "lineage_validation_metrics.csv", low_memory=False
    )
    _require_columns(
        validation_metrics,
        (
            "label_source",
            "n_models",
            "n_families",
            "adjusted_rand",
            "normalized_mutual_information",
            "evidence_role",
        ),
        "lineage_validation_metrics.csv",
    )
    if not validation_metrics["evidence_role"].astype(str).eq(
        "post_freeze_metadata_validation"
    ).all():
        raise ValueError("Lineage validation labels must be marked post-freeze")
    gallery_metrics = pd.read_csv(
        root / "lineage_gallery_metrics.csv", low_memory=False
    )
    _require_columns(
        gallery_metrics,
        (
            "n_scored_queries",
            "n_supplied_roots",
            "n_profiled_roots",
            "root_macro_top1",
            "root_macro_top3",
            "evidence_role",
        ),
        "lineage_gallery_metrics.csv",
    )
    if not gallery_metrics["evidence_role"].astype(str).eq(
        "post_freeze_metadata_validation"
    ).all():
        raise ValueError("Gallery validation labels must be marked post-freeze")

    performance_metrics = pd.read_csv(
        root / "performance_metrics.csv", low_memory=False
    )
    cluster_holdout_metrics = pd.read_csv(
        root / "performance_cluster_holdout_metrics.csv", low_memory=False
    )
    leading_method = str(
        metadata.get("performance_method", LEADING_PERFORMANCE_METHOD)
    ).strip()
    if not leading_method:
        raise ValueError("Scale-up performance_method must be non-empty")
    predictions = _prediction_tables(
        pd.read_csv(root / "performance_predictions.csv", low_memory=False),
        leading_method=leading_method,
    )
    performance = _performance_records(
        performance_metrics,
        cluster_holdout_metrics,
        leading_method=leading_method,
    )
    missing_metrics = sorted(set(predictions).difference(performance))
    if missing_metrics:
        raise ValueError(f"Scale-up predictions are missing metrics for: {missing_metrics}")

    paper_summary = pd.read_csv(
        root / "performance_paper_summary.csv", low_memory=False
    )
    _require_columns(
        paper_summary,
        ("generation", "target", "repeats", "n_models", "mean_mae", "mean_spearman"),
        "performance_paper_summary.csv",
    )
    leakage_audit = pd.read_csv(
        root / "performance_leakage_audit.csv", low_memory=False
    )
    _require_columns(
        leakage_audit,
        ("generation", "n_models", "exact_profiles_crossing_folds"),
        "performance_leakage_audit.csv",
    )
    target_audit = pd.read_csv(root / "performance_target_audit.csv", low_memory=False)
    _require_columns(
        target_audit,
        ("benchmark_generation", "target_resolution", "included", "n_models"),
        "performance_target_audit.csv",
    )
    role_masking_patterns = (
        pd.read_csv(root / "role_masking_patterns.csv", low_memory=False)
        if (root / "role_masking_patterns.csv").is_file()
        else pd.DataFrame()
    )
    role_coverage_selection = (
        pd.read_csv(root / "role_coverage_selection.csv", low_memory=False)
        if (root / "role_coverage_selection.csv").is_file()
        else pd.DataFrame()
    )
    report = (root / "report.md").read_text(encoding="utf-8")
    return ScaleupOutputs(
        run_dir=root,
        metadata=metadata,
        counts=counts,
        population=population,
        predictions=predictions,
        performance=performance,
        lineage_assignments=assignments,
        lineage_internal_metrics=lineage_internal,
        lineage_cluster_selection=cluster_selection,
        lineage_validation_metrics=validation_metrics,
        lineage_gallery_metrics=gallery_metrics,
        performance_metrics=performance_metrics,
        performance_paper_summary=paper_summary,
        performance_leakage_audit=leakage_audit,
        performance_cluster_holdout_metrics=cluster_holdout_metrics,
        performance_target_audit=target_audit,
        role_masking_patterns=role_masking_patterns,
        role_coverage_selection=role_coverage_selection,
        report=report,
    )
