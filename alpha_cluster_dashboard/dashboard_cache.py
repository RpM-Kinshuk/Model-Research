from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Callable
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from alpha_cluster_dashboard.data import AlphaRecord, CANONICAL_MODULE_ORDER, load_alpha_records
from alpha_cluster_dashboard.features import resample_trace
from alpha_cluster_dashboard.paths import resolve_model_zoo_artifact
from spectral_research.ablation_analysis import infer_coarse_family
from spectral_research.ablation_method_study import role_traces


CACHE_VERSION = "ablation-dashboard-cache-v3"
DEFAULT_CACHE_DIR = (
    Path(__file__).resolve().parents[1]
    / "research_runs"
    / "preliminary"
    / "results"
    / "dashboard_cache"
)
DEFAULT_ABLATION_METADATA = resolve_model_zoo_artifact("data/curated/model_zoo_phase2.csv")

_TARGET_DENY_EXACT = {
    "modelid",
    "displayname",
    "id",
    "t",
    "precision",
    "cluster",
    "predictedcluster",
    "actual",
    "prediction",
    "absoluteerror",
    "nearestneighbors",
    "numlayers",
    "rawmodulecount",
    "commonschema",
    "generation",
}
_TARGET_DENY_PARTS = (
    "param",
    "download",
    "like",
    "layer",
    "module",
    "cluster",
    "embedding",
    "galaxy",
    "lineage",
    "radius",
    "angle",
    "neighbor",
    "rank",
    "index",
    "count",
    "size",
    "seed",
    "path",
    "sha",
    "hub",
    "co2",
    "cost",
    "license",
    "completeness",
    "selection",
    "priority",
    "numeric",
)
_TARGET_ALLOW_PARTS = (
    "eval",
    "benchmark",
    "score",
    "accuracy",
    "average",
    "avg",
    "mmlu",
    "math",
    "bbh",
    "gpqa",
    "musr",
    "truthful",
    "gsm",
    "arena",
    "elo",
    "arc",
    "hellaswag",
    "humaneval",
    "winogrande",
    "leaderboard",
)


@dataclass(frozen=True)
class DashboardCacheConfig:
    n_bins: int = 24
    n_clusters: int | None = None
    n_neighbors: int = 3
    random_state: int = 7
    max_records: int | None = None


@dataclass
class DashboardCache:
    cache_dir: Path
    metadata: pd.DataFrame
    features: np.ndarray
    feature_names: list[str]
    predictions: dict[str, pd.DataFrame]
    cluster_profiles: pd.DataFrame
    cluster_summary: pd.DataFrame
    manifest: dict[str, object]


ProgressCallback = Callable[[float, str], None]


def _progress(callback: ProgressCallback | None, fraction: float, message: str) -> None:
    if callback is not None:
        callback(float(np.clip(fraction, 0.0, 1.0)), message)


def _normalized_column_name(column: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(column).lower())


def _safe_target_slug(target: str, used: set[str] | None = None) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "_", target).strip("_") or "target"
    if used is None:
        return base
    slug = base
    suffix = 2
    while slug in used:
        slug = f"{base}_{suffix}"
        suffix += 1
    used.add(slug)
    return slug


def recommended_cluster_count(model_count: int | None) -> int:
    if not model_count or model_count <= 1:
        return 1
    return int(np.clip(round(np.sqrt(model_count) / 2.0), 2, min(model_count, 96)))


def _effective_cluster_count(config: DashboardCacheConfig, n_samples: int) -> int:
    if n_samples <= 1:
        return n_samples
    requested = config.n_clusters if config.n_clusters is not None else recommended_cluster_count(n_samples)
    return min(max(2, int(requested)), n_samples)


def infer_prediction_targets(metadata: pd.DataFrame, *, min_finite: int = 2) -> list[str]:
    targets: list[str] = []
    if metadata.empty:
        return targets

    for column in metadata.columns:
        name = str(column)
        normalized = _normalized_column_name(name)
        if (
            not normalized
            or normalized in _TARGET_DENY_EXACT
            or normalized.endswith("lb")
            or normalized.endswith("prediction")
            or normalized.endswith("absoluteerror")
            or any(part in normalized for part in _TARGET_DENY_PARTS)
            or not any(part in normalized for part in _TARGET_ALLOW_PARTS)
        ):
            continue
        values = pd.to_numeric(metadata[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if len(finite) < min_finite or finite.nunique(dropna=True) <= 1:
            continue
        unique_values = set(finite.dropna().unique().tolist())
        if unique_values.issubset({0, 1, 0.0, 1.0}):
            continue
        targets.append(name)
    return targets


def _clean_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    frame = metadata.copy()
    if "model_id" not in frame.columns:
        return frame
    frame["model_id"] = frame["model_id"].astype(str)
    return frame.drop_duplicates(subset=["model_id"], keep="first")


def source_fingerprint(
    metrics_dir: str | Path,
    metadata_path: str | Path | None,
    *,
    settings: dict[str, object],
) -> dict[str, object]:
    root = Path(metrics_dir)
    digest = hashlib.sha256()
    metric_files = sorted(list(root.glob("*.h5")) + list(root.glob("*.csv")))
    for path in metric_files:
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))

    metadata_info = None
    if metadata_path:
        path = Path(metadata_path)
        if path.exists():
            stat = path.stat()
            metadata_info = {
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
            digest.update(str(path).encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))

    return {
        "version": CACHE_VERSION,
        "metrics_dir": str(root),
        "metric_file_count": len(metric_files),
        "metadata": metadata_info,
        "settings": settings,
        "digest": digest.hexdigest(),
    }


def cache_is_valid(cache_dir: str | Path, fingerprint: dict[str, object]) -> bool:
    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return manifest.get("fingerprint") == fingerprint


def _role_feature_row(record: AlphaRecord, module_names: list[str], n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    traces = _safe_role_traces(record)
    feature_values: list[float] = []
    profile_values: list[np.ndarray] = []
    for module in module_names:
        trace = traces.get(module)
        if trace is None:
            rank_profile = np.full(n_bins, np.nan, dtype=float)
            level_profile = np.full(n_bins, np.nan, dtype=float)
        else:
            cleaned = np.asarray(trace, dtype=float).copy()
            cleaned[~np.isfinite(cleaned)] = np.nan
            cleaned[cleaned <= 0] = np.nan
            sampled = resample_trace(cleaned, n_bins)
            valid = np.isfinite(sampled)
            rank_profile = np.full(n_bins, np.nan, dtype=float)
            if np.count_nonzero(valid) >= 2:
                scale = max(int(np.count_nonzero(valid)) - 1, 1)
                rank_profile[valid] = (rankdata(sampled[valid], method="average") - 1.0) / scale
            elif np.count_nonzero(valid) == 1:
                rank_profile[valid] = 0.5
            level_profile = np.full(n_bins, np.nan, dtype=float)
            level_profile[valid] = np.log1p(sampled[valid])
        profile_values.append(level_profile)
        feature_values.extend(rank_profile.tolist())
        feature_values.extend((0.35 * level_profile).tolist())
        feature_values.append(float(np.isfinite(level_profile).any()))
    return np.asarray(feature_values, dtype=float), np.vstack(profile_values)


def _build_feature_matrix(records: list[AlphaRecord], config: DashboardCacheConfig) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    observed_roles = set()
    for record in records:
        observed_roles.update(_safe_role_traces(record))
    module_names = [module for module in CANONICAL_MODULE_ORDER if module in observed_roles]
    if not module_names:
        raise ValueError("No canonical alpha roles are available for the selected records.")

    feature_names: list[str] = []
    for module in module_names:
        feature_names.extend(f"{module}__rank_{index:02d}" for index in range(config.n_bins))
        feature_names.extend(f"{module}__log_alpha_{index:02d}" for index in range(config.n_bins))
        feature_names.append(f"{module}__present")

    rows: list[np.ndarray] = []
    profiles: list[np.ndarray] = []
    for record in records:
        row, profile = _role_feature_row(record, module_names, config.n_bins)
        rows.append(row)
        profiles.append(profile)
    return np.vstack(rows), feature_names, np.stack(profiles, axis=0), module_names


def _fit_features(features: np.ndarray) -> np.ndarray:
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return pipeline.fit_transform(features)


def _embedding(fitted_features: np.ndarray, random_state: int) -> np.ndarray:
    n_components = min(3, fitted_features.shape[0], fitted_features.shape[1])
    if n_components == 0:
        return np.zeros((fitted_features.shape[0], 3), dtype=float)
    coords = PCA(n_components=n_components, random_state=random_state).fit_transform(fitted_features)
    if coords.shape[1] < 3:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 3 - coords.shape[1]))])
    return coords[:, :3]


def _cluster_labels(fitted_features: np.ndarray, config: DashboardCacheConfig) -> np.ndarray:
    n_samples = fitted_features.shape[0]
    if n_samples <= 1:
        return np.zeros(n_samples, dtype=int)
    n_clusters = _effective_cluster_count(config, n_samples)
    estimator = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=config.random_state,
        n_init=10,
        batch_size=min(512, max(32, n_samples)),
    )
    return estimator.fit_predict(fitted_features)


def _prediction_table(
    metadata: pd.DataFrame,
    fitted_features: np.ndarray,
    target: str,
    config: DashboardCacheConfig,
) -> pd.DataFrame:
    values = pd.to_numeric(metadata[target], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values)
    if int(valid.sum()) < 2:
        return pd.DataFrame()

    k = min(config.n_neighbors + 1, int(valid.sum()))
    valid_indices = np.flatnonzero(valid)
    target_features = fitted_features[valid]
    target_values = values[valid]
    target_names = metadata.loc[valid, "display_name"].astype(str).to_numpy()
    neighbors = NearestNeighbors(n_neighbors=k, metric="euclidean")
    neighbors.fit(target_features)
    distances, indices = neighbors.kneighbors(fitted_features)

    rows: list[dict[str, object]] = []
    display_names = metadata["display_name"].astype(str).to_numpy()
    for row_index, actual in enumerate(values):
        chosen = indices[row_index]
        chosen_distances = distances[row_index]
        chosen_record_indices = valid_indices[chosen]
        keep = chosen_record_indices != row_index
        chosen = chosen[keep][: config.n_neighbors]
        chosen_distances = chosen_distances[keep][: config.n_neighbors]
        if chosen.size == 0:
            continue
        weights = stable_distance_weights(chosen_distances)
        prediction = float(np.average(target_values[chosen], weights=weights))
        has_actual = bool(np.isfinite(actual))
        rows.append(
            {
                "display_name": display_names[row_index],
                "target": target,
                "actual": float(actual) if has_actual else np.nan,
                "prediction": prediction,
                "absolute_error": abs(float(actual) - prediction) if has_actual else np.nan,
                "nearest_neighbors": ", ".join(target_names[chosen].tolist()),
            }
        )
    return pd.DataFrame(rows)


def stable_distance_weights(distances: np.ndarray) -> np.ndarray:
    values = np.asarray(distances, dtype=float)
    weights = np.zeros_like(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.ones_like(values, dtype=float)

    finite_values = np.maximum(values[finite], 0.0)
    shifted = finite_values - float(np.min(finite_values))
    positive_shift = shifted[shifted > 1e-12]
    scale = float(np.median(positive_shift)) if positive_shift.size else 1.0
    scale = max(scale, 1e-8)
    weights[finite] = np.exp(-shifted / scale)
    if float(np.sum(weights)) <= 0.0:
        nearest = int(np.nanargmin(values))
        weights[nearest] = 1.0
    return weights


def _cluster_profile_table(
    profiles: np.ndarray,
    labels: np.ndarray,
    module_names: list[str],
    n_bins: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    depths = np.linspace(0.0, 1.0, n_bins)
    for cluster in sorted(set(labels.tolist())):
        mask = labels == cluster
        if not mask.any():
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            mean_profiles = np.nanmean(profiles[mask], axis=0)
        for module_index, module in enumerate(module_names):
            for depth, alpha in zip(depths, mean_profiles[module_index], strict=False):
                rows.append(
                    {
                        "cluster": int(cluster),
                        "role": module,
                        "depth": float(depth),
                        "log_alpha": float(alpha) if np.isfinite(alpha) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _cluster_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if "cluster" not in metadata.columns:
        return pd.DataFrame()
    for cluster, group in metadata.groupby("cluster", sort=True):
        family = "unknown"
        family_share = np.nan
        if "coarse_family" in group.columns and group["coarse_family"].notna().any():
            counts = group["coarse_family"].fillna("unknown").astype(str).value_counts()
            family = str(counts.index[0])
            family_share = float(counts.iloc[0] / len(group))
        rows.append(
            {
                "cluster": int(cluster),
                "size": int(len(group)),
                "dominant_family": family,
                "dominant_family_share": family_share,
                "mean_layers": float(pd.to_numeric(group["num_layers"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def _merge_metadata(records: list[AlphaRecord], metadata: pd.DataFrame) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "display_name": [record.display_name for record in records],
            "model_id": [record.display_name for record in records],
            "num_layers": [record.num_layers for record in records],
            "raw_module_count": [record.raw_module_count for record in records],
            "common_schema": [record.common_schema for record in records],
            "path": [record.path for record in records],
        }
    )
    if metadata.empty or "model_id" not in metadata.columns:
        return base
    join = _clean_metadata(metadata).rename(columns={"model_id": "_metadata_model_id"})
    overlapping = (set(join.columns) & set(base.columns)) - {"_metadata_model_id"}
    join = join.drop(columns=sorted(overlapping), errors="ignore")
    enriched = base.merge(join, left_on="display_name", right_on="_metadata_model_id", how="left").drop(
        columns=["_metadata_model_id"], errors="ignore"
    )
    if "coarse_family" not in enriched.columns:
        enriched["coarse_family"] = enriched.apply(infer_coarse_family, axis=1)
    return enriched


def _safe_role_traces(record: AlphaRecord) -> dict[str, np.ndarray]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        return role_traces(record)


def _overall_profile_mean(profiles: np.ndarray) -> np.ndarray:
    values = np.full(profiles.shape[0], np.nan, dtype=float)
    for index in range(profiles.shape[0]):
        finite = np.isfinite(profiles[index])
        if finite.any():
            values[index] = float(np.mean(profiles[index][finite]))
    return values


def build_cache_from_records(
    records: list[AlphaRecord],
    metadata: pd.DataFrame,
    cache_dir: str | Path,
    *,
    config: DashboardCacheConfig,
    fingerprint: dict[str, object],
    progress: ProgressCallback | None = None,
) -> DashboardCache:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    for stale_prediction in root.glob("predictions_*.csv"):
        stale_prediction.unlink()
    if config.max_records is not None and len(records) > config.max_records:
        records = records[: config.max_records]

    _progress(progress, 0.05, "Building rank-shape alpha features")
    raw_features, feature_names, profiles, module_names = _build_feature_matrix(records, config)
    fitted_features = _fit_features(raw_features)

    _progress(progress, 0.30, "Embedding models into 2D/3D atlas")
    coords = _embedding(fitted_features, config.random_state)

    _progress(progress, 0.45, "Clustering feature atlas")
    labels = _cluster_labels(fitted_features, config)

    enriched = _merge_metadata(records, metadata)
    enriched["embedding_x"] = coords[:, 0]
    enriched["embedding_y"] = coords[:, 1]
    enriched["embedding_z"] = coords[:, 2]
    enriched["cluster"] = labels.astype(int)
    enriched["overall_log_alpha"] = _overall_profile_mean(profiles)

    _progress(progress, 0.65, "Building performance-neighborhood predictions")
    predictions: dict[str, pd.DataFrame] = {}
    prediction_files: dict[str, str] = {}
    used_slugs: set[str] = set()
    for target in infer_prediction_targets(enriched):
        table = _prediction_table(enriched, fitted_features, target, config)
        if table.empty:
            continue
        predictions[target] = table
        file_name = f"predictions_{_safe_target_slug(target, used_slugs)}.csv"
        prediction_files[target] = file_name
        table.to_csv(root / file_name, index=False)
        enriched = enriched.merge(
            table[["display_name", "prediction", "absolute_error"]].rename(
                columns={
                    "prediction": f"{target} prediction",
                    "absolute_error": f"{target} absolute_error",
                }
            ),
            on="display_name",
            how="left",
        )

    _progress(progress, 0.80, "Summarizing cluster alpha profiles")
    cluster_profiles = _cluster_profile_table(profiles, labels, module_names, config.n_bins)
    cluster_summary = _cluster_summary(enriched)

    manifest = {
        "fingerprint": fingerprint,
        "config": asdict(config),
        "feature_names": feature_names,
        "module_names": module_names,
        "prediction_files": prediction_files,
        "prediction_targets": list(predictions),
        "model_count": len(records),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    enriched.to_csv(root / "model_atlas.csv", index=False)
    cluster_profiles.to_csv(root / "cluster_profiles.csv", index=False)
    cluster_summary.to_csv(root / "cluster_summary.csv", index=False)
    np.savez_compressed(root / "features.npz", features=fitted_features, raw_features=raw_features)

    _progress(progress, 1.0, "Dashboard cache ready")
    return DashboardCache(
        cache_dir=root,
        metadata=enriched,
        features=fitted_features,
        feature_names=feature_names,
        predictions=predictions,
        cluster_profiles=cluster_profiles,
        cluster_summary=cluster_summary,
        manifest=manifest,
    )


def load_dashboard_cache(cache_dir: str | Path) -> DashboardCache:
    root = Path(cache_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    metadata = pd.read_csv(root / "model_atlas.csv")
    feature_data = np.load(root / "features.npz")
    prediction_files = dict(manifest.get("prediction_files", {}))
    ordered_targets = list(manifest.get("prediction_targets", [])) or list(prediction_files)
    predictions = {}
    for target in ordered_targets:
        file_name = prediction_files.get(target)
        if file_name and (root / file_name).exists():
            predictions[target] = pd.read_csv(root / file_name)
    cluster_profiles = pd.read_csv(root / "cluster_profiles.csv") if (root / "cluster_profiles.csv").exists() else pd.DataFrame()
    cluster_summary = pd.read_csv(root / "cluster_summary.csv") if (root / "cluster_summary.csv").exists() else pd.DataFrame()
    return DashboardCache(
        cache_dir=root,
        metadata=metadata,
        features=np.asarray(feature_data["features"], dtype=float),
        feature_names=list(manifest.get("feature_names", [])),
        predictions=predictions,
        cluster_profiles=cluster_profiles,
        cluster_summary=cluster_summary,
        manifest=manifest,
    )


def load_or_build_dashboard_cache(
    metrics_dir: str | Path,
    metadata_path: str | Path | None,
    cache_dir: str | Path,
    *,
    config: DashboardCacheConfig,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> DashboardCache:
    fingerprint = source_fingerprint(metrics_dir, metadata_path, settings=asdict(config))
    if not force and cache_is_valid(cache_dir, fingerprint):
        _progress(progress, 1.0, "Loaded existing dashboard cache")
        return load_dashboard_cache(cache_dir)

    _progress(progress, 0.02, "Loading alpha records")
    records = load_alpha_records(metrics_dir)
    metadata = pd.read_csv(metadata_path) if metadata_path and Path(metadata_path).exists() else pd.DataFrame()
    return build_cache_from_records(records, metadata, cache_dir, config=config, fingerprint=fingerprint, progress=progress)


def pairwise_subset_distances(features: np.ndarray, indices: np.ndarray) -> np.ndarray:
    subset = np.asarray(features, dtype=float)[np.asarray(indices, dtype=int)]
    return pairwise_distances(subset, metric="euclidean")
