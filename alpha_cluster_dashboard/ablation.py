from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alpha_cluster_dashboard.data import AlphaRecord
from alpha_cluster_dashboard.dashboard_cache import infer_prediction_targets
from alpha_cluster_dashboard.paths import PROJECT_ROOT, resolve_model_zoo_artifact
from spectral_research.ablation_method_study import (
    DEFAULT_METHODS,
    build_similarity_matrix,
    build_structural_similarity_matrix,
)


DEFAULT_ABLATION_METRICS_DIR = resolve_model_zoo_artifact("output/results/metrics")


def _latest_ablation_study_dir() -> Path:
    root = PROJECT_ROOT / "research_runs" / "preliminary" / "ablation"
    if not root.exists():
        return root
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and (
            (path / "method_decision_report.md").exists()
            or (path / "method_family_metrics.csv").exists()
            or (path / "method_prediction_metrics.csv").exists()
        )
    ]
    if not candidates:
        return root
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


DEFAULT_ABLATION_STUDY_DIR = _latest_ablation_study_dir()
DEFAULT_ABLATION_STUDY_DATASET = DEFAULT_ABLATION_STUDY_DIR / "method_study_dataset.csv"


def load_ablation_study_dataset(path: str | Path) -> pd.DataFrame:
    study_path = Path(path)
    if not study_path.exists():
        return pd.DataFrame()
    return pd.read_csv(study_path)


def load_method_study_tables(study_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(study_dir)
    family_path = root / "method_family_metrics.csv"
    prediction_path = root / "method_prediction_metrics.csv"
    family = pd.read_csv(family_path) if family_path.exists() else pd.DataFrame()
    prediction = pd.read_csv(prediction_path) if prediction_path.exists() else pd.DataFrame()
    return family, prediction


def merge_ablation_metadata(metadata: pd.DataFrame, ablation: pd.DataFrame | None) -> pd.DataFrame:
    enriched = metadata.copy()
    if ablation is None or ablation.empty or "model_id" not in ablation.columns:
        return enriched

    join = ablation.drop_duplicates(subset=["model_id"]).copy()
    join["_ablation_display_name"] = join["model_id"].astype(str)
    join = join.rename(columns={"model_id": "ablation_model_id"})
    overlapping_columns = (set(join.columns) & set(enriched.columns)) - {"_ablation_display_name"}
    if overlapping_columns:
        join = join.drop(columns=sorted(overlapping_columns))
    return enriched.merge(
        join,
        left_on="display_name",
        right_on="_ablation_display_name",
        how="left",
    ).drop(columns=["_ablation_display_name"], errors="ignore")


def available_prediction_targets(frame: pd.DataFrame) -> list[str]:
    return infer_prediction_targets(frame)


def build_ablation_similarity_matrix(
    records: list[AlphaRecord],
    method: str,
    metadata: pd.DataFrame | None = None,
) -> np.ndarray:
    if method == "structural_basics":
        if metadata is None:
            raise ValueError("metadata is required for structural_basics")
        return build_structural_similarity_matrix(metadata, records)
    return build_similarity_matrix(records, method=method)


def similarity_to_distance(similarity: np.ndarray) -> np.ndarray:
    scores = np.asarray(similarity, dtype=float)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("similarity must be a square matrix")

    finite = np.isfinite(scores)
    if not finite.any():
        return np.zeros_like(scores, dtype=float)

    finite_scores = scores[finite]
    max_score = float(np.max(finite_scores))
    min_score = float(np.min(finite_scores))
    missing_distance = max_score - min_score
    if missing_distance <= 0:
        missing_distance = 1.0

    distances = max_score - scores
    distances[~np.isfinite(distances)] = missing_distance
    distances = np.maximum(distances, 0.0)
    np.fill_diagonal(distances, 0.0)
    return distances


def build_knn_prediction_frame(
    frame: pd.DataFrame,
    similarity: np.ndarray,
    *,
    target: str,
    k: int,
) -> pd.DataFrame:
    values = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
    names = frame["display_name"].astype(str).to_numpy() if "display_name" in frame.columns else np.arange(len(frame)).astype(str)
    rows: list[dict[str, object]] = []

    for index, actual in enumerate(values):
        if not np.isfinite(actual):
            continue
        scores = np.asarray(similarity[index], dtype=float).copy()
        scores[index] = np.nan
        valid = np.isfinite(scores) & np.isfinite(values)
        if not valid.any():
            continue

        candidate_indices = np.where(valid)[0]
        order = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
        chosen = order[:k]
        if chosen.size == 0:
            continue

        chosen_scores = scores[chosen]
        weights = np.exp(chosen_scores - float(np.nanmax(chosen_scores)))
        prediction = float(np.average(values[chosen], weights=weights))
        rows.append(
            {
                "display_name": names[index],
                "actual": float(actual),
                "prediction": prediction,
                "absolute_error": abs(float(actual) - prediction),
                "nearest_neighbors": ", ".join(names[chosen].tolist()),
            }
        )

    return pd.DataFrame(rows)


def ablation_method_options() -> list[str]:
    return list(DEFAULT_METHODS)
