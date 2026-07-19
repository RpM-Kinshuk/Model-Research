from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


LEADING_PERFORMANCE_METHOD = "alpha_profile_k5_component_distinct"
REQUIRED_SCALEUP_FILES = (
    "metadata.json",
    "population_ledger.csv",
    "modern_pooled_predictions.csv",
    "modern_pooled_metrics.csv",
    "modern_pooled_increments.csv",
    "legacy_paired_predictions.csv",
    "legacy_paired_metrics.csv",
    "legacy_increments.csv",
    "lineage_broad_cluster_metrics.csv",
    "report.md",
)


@dataclass(frozen=True)
class ScaleupOutputs:
    run_dir: Path
    metadata: Mapping[str, Any]
    counts: Mapping[str, int]
    population: pd.DataFrame
    predictions: Mapping[str, pd.DataFrame]
    performance: Mapping[str, Mapping[str, Any]]
    broad_cluster_metrics: pd.DataFrame
    report: str


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


def _prediction_tables(frames: list[pd.DataFrame]) -> dict[str, pd.DataFrame]:
    combined = pd.concat(frames, ignore_index=True)
    _require_columns(
        combined,
        ("model_id", "target", "feature_block", "actual", "prediction"),
        "scale-up predictions",
    )
    selected = combined.loc[
        combined["feature_block"].astype(str).eq(LEADING_PERFORMANCE_METHOD)
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
        table["nearest_neighbors"] = ""
        tables[str(target)] = table.reset_index(drop=True)
    if not tables:
        raise ValueError("Scale-up run contains no leading alpha-profile predictions")
    return tables


def _performance_records(
    metric_frames: list[pd.DataFrame],
    increment_frames: list[pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    metrics = pd.concat(metric_frames, ignore_index=True)
    increments = pd.concat(increment_frames, ignore_index=True)
    _require_columns(
        metrics,
        (
            "target",
            "feature_block",
            "n_models",
            "n_components",
            "component_weighted_mae",
            "spearman",
        ),
        "scale-up metrics",
    )
    _require_columns(
        increments,
        ("target", "candidate", "delta_mae", "ci_low", "ci_high"),
        "scale-up increments",
    )
    metrics = metrics.loc[
        metrics["feature_block"].astype(str).eq(LEADING_PERFORMANCE_METHOD)
    ]
    increments = increments.loc[
        increments["candidate"].astype(str).eq(LEADING_PERFORMANCE_METHOD)
    ]

    records: dict[str, dict[str, Any]] = {}
    for _, row in metrics.iterrows():
        target = str(row["target"])
        if target in records:
            raise ValueError(f"Duplicate scale-up metric target: {target}")
        record = {
            "target": target,
            "feature_block": LEADING_PERFORMANCE_METHOD,
            "group_weighted_mae": row["component_weighted_mae"],
            "model_weighted_mae": row.get("model_weighted_mae"),
            "spearman": row["spearman"],
            "kendall_tau": row.get("kendall_tau"),
            "n_models": row["n_models"],
            "n_components": row["n_components"],
        }
        increment = increments.loc[increments["target"].astype(str).eq(target)]
        if len(increment) == 1:
            delta = increment.iloc[0]
            record.update(
                {
                    "baseline": delta.get("baseline"),
                    "delta_mae": delta["delta_mae"],
                    "ci_low": delta["ci_low"],
                    "ci_high": delta["ci_high"],
                    "p_value": delta.get("p_value_raw"),
                }
            )
        records[target] = record
    return records


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
    raw_counts = metadata.get("counts", {})
    counts = {
        str(key): int(value)
        for key, value in raw_counts.items()
        if isinstance(value, (int, float)) and np.isfinite(value)
    }

    population = pd.read_csv(root / "population_ledger.csv", low_memory=False)
    _require_columns(
        population,
        ("model_id", "htsr_artifact_available", "strict_cluster", "broad_cluster"),
        "population_ledger.csv",
    )
    model_ids = population["model_id"].astype("string").str.strip()
    if model_ids.isna().any() or model_ids.eq("").any() or model_ids.duplicated().any():
        raise ValueError("population_ledger.csv contains invalid model IDs")
    for column in ("strict_cluster", "broad_cluster"):
        values = pd.to_numeric(population[column], errors="coerce")
        valid = values.isna() | (values.ge(0) & values.eq(np.floor(values)))
        if not valid.all():
            raise ValueError(f"population_ledger.csv contains invalid {column} values")
        population[column] = values.astype("Int64")

    modern_predictions = pd.read_csv(root / "modern_pooled_predictions.csv", low_memory=False)
    legacy_predictions = pd.read_csv(root / "legacy_paired_predictions.csv", low_memory=False)
    predictions = _prediction_tables([modern_predictions, legacy_predictions])
    performance = _performance_records(
        [
            pd.read_csv(root / "modern_pooled_metrics.csv", low_memory=False),
            pd.read_csv(root / "legacy_paired_metrics.csv", low_memory=False),
        ],
        [
            pd.read_csv(root / "modern_pooled_increments.csv", low_memory=False),
            pd.read_csv(root / "legacy_increments.csv", low_memory=False),
        ],
    )
    missing_metrics = sorted(set(predictions).difference(performance))
    if missing_metrics:
        raise ValueError(f"Scale-up predictions are missing metrics for: {missing_metrics}")

    broad_metrics = pd.read_csv(
        root / "lineage_broad_cluster_metrics.csv", low_memory=False
    )
    report = (root / "report.md").read_text(encoding="utf-8")
    return ScaleupOutputs(
        run_dir=root,
        metadata=metadata,
        counts=counts,
        population=population,
        predictions=predictions,
        performance=performance,
        broad_cluster_metrics=broad_metrics,
        report=report,
    )
