from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from spectral_research.study.artifacts import (
    ARTIFACT_MANIFEST_FILE,
    validate_artifact_manifest,
    write_artifact_manifest,
)
from spectral_research.study.contracts import (
    RunSettings,
    StudyInputs,
    build_model_manifest,
    build_provenance,
    derive_analysis_metadata,
    load_graph_contract,
    path_digest,
    study_run_id,
)
from spectral_research.study.features import (
    ALPHA_ESTIMATOR,
    ALPHA_ESTIMATOR_VERSION,
    FEATURE_SCHEMA_VERSION,
    confirmatory_schema_population,
    extract_study_tables,
    index_stats_directory,
)
from spectral_research.study.lineage import (
    cluster_lineage_diagnostics,
    evaluate_closed_set_lineage,
    evaluate_cross_fitted_family_representatives,
    evaluate_exact_parent_diagnostic,
    evaluate_unknown_family_rejection,
    label_independent_cluster_count,
)
from spectral_research.study.performance import (
    TargetSemanticsError,
    assess_htsr_increment,
    build_modern_targets,
    confirmatory_group_population,
    evaluate_external_transfer,
    evaluate_grouped_ridge,
    evaluate_repeated_group_holdout,
    evaluate_secondary_endpoints,
    grouped_outer_folds,
    paired_component_bootstrap,
    paired_component_signflip_pvalue,
)


REQUIRED_ARTIFACTS = (
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

STUDY_IMPLEMENTATION_VERSION = "htsr-combined-implementation-v7"

_FEATURE_CACHE_FILES = (
    "feature_table.csv",
    "alpha_trajectories.csv",
    "exclusions.csv",
    "stats_index.csv",
    "cache_metadata.json",
    ARTIFACT_MANIFEST_FILE,
)

_PERFORMANCE_PREDICTION_COLUMNS = (
    "evaluation_scope",
    "model_id",
    "ancestry_component",
    "target",
    "feature_block",
    "outer_fold",
    "chosen_alpha",
    "actual",
    "prediction",
    "external_slice",
)

_PERFORMANCE_METRIC_COLUMNS = (
    "target",
    "feature_block",
    "n_models",
    "n_components",
    "group_weighted_mae",
    "kendall_tau",
    "spearman",
    "r2",
    "external_slice",
)

_PERFORMANCE_AUDIT_COLUMNS = (
    "target",
    "outer_fold",
    "train_groups",
    "test_groups",
    "n_train",
    "n_test",
)

_CLUSTER_ASSIGNMENT_COLUMNS = ("model_id", "cluster")
_CLUSTER_METRIC_COLUMNS = (
    "method",
    "n_clusters",
    "n_clustered",
    "n_labeled",
    "ari",
    "nmi",
    "root_macro_purity",
    "silhouette",
)
_LINEAGE_EXCLUSION_COLUMNS = (
    "stage",
    "root_family",
    "model_id",
    "reason",
    "detail",
)


def _study_implementation_digest() -> str:
    """Hash the executable study sources that determine persisted results."""

    repository_root = Path(__file__).resolve().parents[2]
    sources = sorted((repository_root / "spectral_research" / "study").glob("*.py"))
    sources.extend(
        [
            repository_root / "spectral_research" / "io_contract.py",
            repository_root / "scripts" / "htsr_combined_study.py",
        ]
    )
    digest = hashlib.sha256()
    for path in sorted(sources):
        if not path.is_file():
            raise FileNotFoundError(f"implementation source is missing: {path}")
        relative = path.relative_to(repository_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _feature_extractor_digest() -> str:
    """Hash local sources that determine cached feature values and identities."""

    repository_root = Path(__file__).resolve().parents[2]
    sources = (
        repository_root / "spectral_research" / "study" / "contracts.py",
        repository_root / "spectral_research" / "study" / "features.py",
    )
    digest = hashlib.sha256()
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(f"feature extractor source is missing: {path}")
        relative = path.relative_to(repository_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


@dataclass(frozen=True)
class StudyRunResult:
    run_id: str
    run_dir: Path | None
    dry_run: bool
    cache_reused: bool
    summary: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.run_dir is not None:
            object.__setattr__(self, "run_dir", Path(self.run_dir).resolve())
        object.__setattr__(self, "summary", MappingProxyType(dict(self.summary)))


@dataclass(frozen=True)
class _LoadedContracts:
    manifest: pd.DataFrame
    manifest_exclusions: pd.DataFrame
    graph_edges: pd.DataFrame
    graph_nodes: tuple[str, ...]
    graph_fingerprint: str
    development_index: pd.DataFrame
    development_index_exclusions: pd.DataFrame
    development_targets: pd.DataFrame
    development_target_exclusions: pd.DataFrame
    development_target_source_rows: int
    external_index: pd.DataFrame | None
    external_index_exclusions: pd.DataFrame | None
    external_targets: pd.DataFrame | None
    external_target_exclusions: pd.DataFrame | None
    external_target_source_rows: int
    metadata_rows: int
    label_rows: int

    def summary(self) -> dict[str, int]:
        return {
            "manifest_rows": len(self.manifest),
            "metadata_rows": self.metadata_rows,
            "label_rows": self.label_rows,
            "graph_node_rows": len(self.graph_nodes),
            "graph_edge_rows": len(self.graph_edges),
            "development_stats_artifacts": len(self.development_index),
            "development_target_source_rows": self.development_target_source_rows,
            "development_target_rows": len(self.development_targets),
            "external_stats_artifacts": (
                len(self.external_index) if self.external_index is not None else 0
            ),
            "external_target_source_rows": self.external_target_source_rows,
            "external_target_rows": (
                len(self.external_targets) if self.external_targets is not None else 0
            ),
        }


def _validate_sources(inputs: StudyInputs) -> None:
    directory_sources = {"development_stats", "external_stats"}
    for logical_name, path in inputs.source_paths().items():
        if not path.exists():
            raise FileNotFoundError(
                f"{logical_name}: source path does not exist: {path}"
            )
        if logical_name in directory_sources and not path.is_dir():
            raise NotADirectoryError(
                f"{logical_name}: expected a stats directory: {path}"
            )
        if logical_name not in directory_sources and not path.is_file():
            raise FileNotFoundError(
                f"{logical_name}: expected a source file: {path}"
            )


def _read_csv(path: Path, logical_name: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ValueError(f"{logical_name}: invalid CSV source: {error}") from error


def _load_contracts(
    inputs: StudyInputs, *, graph_digest: str | None = None
) -> _LoadedContracts:
    metadata = _read_csv(inputs.metadata, "metadata")
    labels = _read_csv(inputs.labels, "labels")
    raw_development_targets = _read_csv(
        inputs.development_targets, "development_targets"
    )
    try:
        graph_edges, graph_nodes, graph_metadata = load_graph_contract(inputs.graph)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"graph: invalid graph source: {error}") from error

    try:
        development_index, development_index_exclusions = index_stats_directory(
            inputs.development_stats
        )
    except (OSError, ValueError, NotADirectoryError) as error:
        raise ValueError(f"development_stats: invalid stats source: {error}") from error
    try:
        development_targets, development_target_exclusions = build_modern_targets(
            raw_development_targets
        )
    except TargetSemanticsError as error:
        raise TargetSemanticsError(f"development_targets: {error}") from error

    external_index: pd.DataFrame | None = None
    external_index_exclusions: pd.DataFrame | None = None
    external_targets: pd.DataFrame | None = None
    external_target_exclusions: pd.DataFrame | None = None
    external_source_rows = 0
    if inputs.external_stats is not None and inputs.external_targets is not None:
        raw_external_targets = _read_csv(inputs.external_targets, "external_targets")
        external_source_rows = len(raw_external_targets)
        try:
            external_index, external_index_exclusions = index_stats_directory(
                inputs.external_stats
            )
        except (OSError, ValueError, NotADirectoryError) as error:
            raise ValueError(f"external_stats: invalid stats source: {error}") from error
        try:
            external_targets, external_target_exclusions = build_modern_targets(
                raw_external_targets
            )
        except TargetSemanticsError as error:
            raise TargetSemanticsError(f"external_targets: {error}") from error

    population_tables = {
        "development_stats": development_index,
        "development_targets": development_targets,
        "labels": labels,
    }
    if external_index is not None and external_targets is not None:
        population_tables["external_stats"] = external_index
        population_tables["external_targets"] = external_targets
    analysis_metadata, population_exclusions = derive_analysis_metadata(
        metadata,
        graph_metadata,
        population_tables,
    )
    grouping_universe = set(development_index["model_id"].astype(str))
    if external_index is not None:
        grouping_universe.update(external_index["model_id"].astype(str))
    manifest, label_exclusions = build_model_manifest(
        analysis_metadata,
        labels,
        graph_edges,
        graph_nodes=graph_nodes,
        grouping_universe=sorted(grouping_universe),
        graph_metadata=graph_metadata,
    )
    manifest_exclusions = pd.concat(
        [population_exclusions, label_exclusions],
        ignore_index=True,
    )
    graph_fingerprint = graph_digest or path_digest(inputs.graph)
    manifest["component_graph_fingerprint"] = graph_fingerprint

    development_ids = set(development_index["model_id"].astype(str))
    manifest["development_artifact_available"] = manifest["model_id"].isin(
        development_ids
    )
    manifest["artifact_available"] = manifest[
        "development_artifact_available"
    ]
    manifest = manifest.merge(
        development_targets.loc[
            :,
            [
                "model_id",
                "target_model_sha",
                "target_checkpoint_status",
                "target_eval_name",
                "target_architecture",
            ],
        ],
        on="model_id",
        how="left",
        validate="one_to_one",
    )
    if external_index is not None:
        external_ids = set(external_index["model_id"].astype(str))
        manifest["external_artifact_available"] = manifest["model_id"].isin(
            external_ids
        )
    else:
        manifest["external_artifact_available"] = False

    return _LoadedContracts(
        manifest=manifest,
        manifest_exclusions=manifest_exclusions,
        graph_edges=graph_edges,
        graph_nodes=graph_nodes,
        graph_fingerprint=graph_fingerprint,
        development_index=development_index,
        development_index_exclusions=development_index_exclusions,
        development_targets=development_targets,
        development_target_exclusions=development_target_exclusions,
        development_target_source_rows=len(raw_development_targets),
        external_index=external_index,
        external_index_exclusions=external_index_exclusions,
        external_targets=external_targets,
        external_target_exclusions=external_target_exclusions,
        external_target_source_rows=external_source_rows,
        metadata_rows=len(metadata),
        label_rows=len(labels),
    )


def _complete_directory(path: Path, filenames: Sequence[str]) -> bool:
    return path.is_dir() and all((path / filename).is_file() for filename in filenames)


def _read_artifact_csv(path: Path, artifact: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise RuntimeError(f"{artifact} is not a readable CSV: {error}") from error


def _read_artifact_json(path: Path, artifact: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{artifact} is not readable JSON: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{artifact} must contain a JSON object")
    return payload


def _require_artifact_columns(
    frame: pd.DataFrame, required: Sequence[str], artifact: str
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{artifact} is missing columns: {missing}")


def _load_feature_cache(
    cache_dir: Path,
    *,
    key: str,
    stats_digest: str,
    extractor_source_digest: str,
    model_ids: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    if not cache_dir.exists():
        return None
    if not _complete_directory(cache_dir, _FEATURE_CACHE_FILES):
        raise RuntimeError(f"feature cache is incomplete: {cache_dir}")

    metadata = _read_artifact_json(
        cache_dir / "cache_metadata.json", "feature cache metadata"
    )
    expected = {
        "cache_key": key,
        "alpha_estimator": ALPHA_ESTIMATOR,
        "alpha_estimator_version": ALPHA_ESTIMATOR_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "extractor_source_digest": extractor_source_digest,
        "stats_sha256": stats_digest,
        "requested_model_ids": list(model_ids),
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise RuntimeError(
                f"feature cache metadata mismatch for {field}: {cache_dir}"
            )

    features = _read_artifact_csv(
        cache_dir / "feature_table.csv", "feature cache feature_table.csv"
    )
    trajectories = _read_artifact_csv(
        cache_dir / "alpha_trajectories.csv",
        "feature cache alpha_trajectories.csv",
    )
    exclusions = _read_artifact_csv(
        cache_dir / "exclusions.csv", "feature cache exclusions.csv"
    )
    stats_index = _read_artifact_csv(
        cache_dir / "stats_index.csv", "feature cache stats_index.csv"
    )
    for frame, required, artifact in (
        (features, ("model_id",), "feature cache feature_table.csv"),
        (trajectories, ("model_id",), "feature cache alpha_trajectories.csv"),
        (exclusions, ("stage", "reason"), "feature cache exclusions.csv"),
        (stats_index, ("model_id",), "feature cache stats_index.csv"),
    ):
        _require_artifact_columns(frame, required, artifact)

    row_counts = {
        "feature_rows": len(features),
        "trajectory_rows": len(trajectories),
        "stats_index_rows": len(stats_index),
    }
    for field, value in row_counts.items():
        if metadata.get(field) != value:
            raise RuntimeError(
                f"feature cache metadata mismatch for {field}: "
                f"recorded={metadata.get(field)!r}, observed={value}"
            )
    if features["model_id"].astype(str).duplicated().any():
        raise RuntimeError("feature cache feature_table.csv has duplicate model_id values")
    requested = set(model_ids)
    for frame, artifact in (
        (features, "feature_table.csv"),
        (trajectories, "alpha_trajectories.csv"),
    ):
        outside = set(frame["model_id"].astype(str)) - requested
        if outside:
            raise RuntimeError(
                f"feature cache {artifact} contains unrequested model identifiers"
            )
    validate_artifact_manifest(
        cache_dir,
        expected_run_id=key,
        required_files=tuple(
            filename
            for filename in _FEATURE_CACHE_FILES
            if filename != ARTIFACT_MANIFEST_FILE
        ),
        validate_all=True,
    )
    return features, trajectories, exclusions


def _validated_run_summary(run_dir: Path, run_id: str) -> dict[str, Any]:
    if not run_dir.is_dir():
        raise RuntimeError(f"published run is not a directory: {run_dir}")
    missing = [
        filename
        for filename in REQUIRED_ARTIFACTS
        if not (run_dir / filename).is_file()
    ]
    if missing:
        raise RuntimeError(f"published run is incomplete; missing: {missing}")

    validate_artifact_manifest(
        run_dir,
        expected_run_id=run_id,
        required_files=tuple(
            filename
            for filename in REQUIRED_ARTIFACTS
            if filename != ARTIFACT_MANIFEST_FILE
        ),
        validate_all=True,
    )

    provenance = _read_artifact_json(
        run_dir / "provenance.json", "published run provenance.json"
    )
    if provenance.get("run_id") != run_id:
        raise RuntimeError(
            "published run provenance run_id does not match its content address"
        )
    _read_artifact_json(
        run_dir / "lineage_metrics.json", "published run lineage_metrics.json"
    )
    _read_artifact_json(
        run_dir / "performance_metrics.json", "published run performance_metrics.json"
    )
    frames = {
        "manifest_rows": _read_artifact_csv(
            run_dir / "manifest.csv", "published run manifest.csv"
        ),
        "development_feature_rows": _read_artifact_csv(
            run_dir / "feature_table.csv", "published run feature_table.csv"
        ),
        "cluster_assignment_rows": _read_artifact_csv(
            run_dir / "cluster_assignments.csv",
            "published run cluster_assignments.csv",
        ),
        "lineage_prediction_rows": _read_artifact_csv(
            run_dir / "lineage_predictions.csv",
            "published run lineage_predictions.csv",
        ),
        "performance_prediction_rows": _read_artifact_csv(
            run_dir / "performance_predictions.csv",
            "published run performance_predictions.csv",
        ),
        "exclusion_rows": _read_artifact_csv(
            run_dir / "exclusions.csv", "published run exclusions.csv"
        ),
    }
    schemas = {
        "manifest_rows": ("model_id",),
        "development_feature_rows": ("model_id",),
        "cluster_assignment_rows": ("model_id", "cluster"),
        "lineage_prediction_rows": ("method", "query_model_id"),
        "performance_prediction_rows": (
            "model_id",
            "target",
            "feature_block",
            "actual",
            "prediction",
        ),
        "exclusion_rows": ("stage", "reason"),
    }
    for name, frame in frames.items():
        _require_artifact_columns(frame, schemas[name], f"published run {name}")
    for name in (
        "manifest_rows",
        "development_feature_rows",
        "cluster_assignment_rows",
    ):
        if frames[name]["model_id"].astype(str).duplicated().any():
            raise RuntimeError(f"published run {name} has duplicate model_id values")

    counts = provenance.get("counts")
    if not isinstance(counts, dict):
        raise RuntimeError("published run provenance counts must be an object")
    for field, frame in frames.items():
        if counts.get(field) != len(frame):
            raise RuntimeError(
                f"published run count mismatch for {field}: "
                f"recorded={counts.get(field)!r}, observed={len(frame)}"
            )
    return dict(counts)


def _cache_key(
    stats_digest: str,
    model_ids: Sequence[str],
    *,
    extractor_source_digest: str,
) -> str:
    payload = {
        "alpha_estimator": ALPHA_ESTIMATOR,
        "alpha_estimator_version": ALPHA_ESTIMATOR_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "extractor_source_digest": extractor_source_digest,
        "stats_sha256": stats_digest,
        "model_ids": list(model_ids),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _atomic_directory(destination: Path, writer: Any) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid4().hex}"
    temporary.mkdir()
    try:
        writer(temporary)
        if destination.exists():
            return False
        try:
            os.rename(temporary, destination)
        except OSError as error:
            if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            return False
        return True
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _feature_cache(
    cache_root: Path,
    *,
    stats_digest: str,
    stats_index: pd.DataFrame,
    model_ids: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool, Path]:
    extractor_source_digest = _feature_extractor_digest()
    key = _cache_key(
        stats_digest,
        model_ids,
        extractor_source_digest=extractor_source_digest,
    )
    cache_dir = cache_root / "study_features" / key
    cached = _load_feature_cache(
        cache_dir,
        key=key,
        stats_digest=stats_digest,
        extractor_source_digest=extractor_source_digest,
        model_ids=model_ids,
    )
    if cached is not None:
        return (*cached, True, cache_dir)

    (
        features,
        trajectories,
        feature_exclusions,
        trajectory_exclusions,
    ) = extract_study_tables(
        stats_index,
        model_ids,
    )
    exclusions = _combine_exclusions(
        (feature_exclusions, "feature_extraction"),
        (trajectory_exclusions, "alpha_trajectory_extraction"),
    )

    def write_cache(temporary: Path) -> None:
        features.to_csv(temporary / "feature_table.csv", index=False)
        trajectories.to_csv(temporary / "alpha_trajectories.csv", index=False)
        exclusions.to_csv(temporary / "exclusions.csv", index=False)
        stats_index.to_csv(temporary / "stats_index.csv", index=False)
        _write_json(
            temporary / "cache_metadata.json",
            {
                "cache_key": key,
                "alpha_estimator": ALPHA_ESTIMATOR,
                "alpha_estimator_version": ALPHA_ESTIMATOR_VERSION,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "extractor_source_digest": extractor_source_digest,
                "stats_sha256": stats_digest,
                "requested_model_ids": list(model_ids),
                "feature_rows": len(features),
                "trajectory_rows": len(trajectories),
                "stats_index_rows": len(stats_index),
            },
        )
        write_artifact_manifest(temporary, run_id=key)

    published = _atomic_directory(cache_dir, write_cache)
    if not published:
        concurrent = _load_feature_cache(
            cache_dir,
            key=key,
            stats_digest=stats_digest,
            extractor_source_digest=extractor_source_digest,
            model_ids=model_ids,
        )
        if concurrent is None:
            raise RuntimeError("feature cache publication lost without a valid cache")
        return (*concurrent, True, cache_dir)
    return features, trajectories, exclusions, False, cache_dir


def _combine_exclusions(
    *items: tuple[pd.DataFrame | None, str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for frame, stage in items:
        if frame is None or frame.empty:
            continue
        normalized = frame.copy()
        if "stage" not in normalized.columns:
            normalized.insert(0, "stage", stage)
        else:
            normalized["stage"] = normalized["stage"].fillna(stage)
        if "model_id" not in normalized.columns:
            normalized["model_id"] = ""
        if "reason" not in normalized.columns:
            normalized["reason"] = "unspecified"
        if "detail" not in normalized.columns:
            normalized["detail"] = ""
        frames.append(normalized)
    if not frames:
        return pd.DataFrame(columns=["stage", "model_id", "reason", "detail"])
    return pd.concat(frames, ignore_index=True, sort=False)


def _exclusion(stage: str, reason: str, detail: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{"stage": stage, "model_id": "", "reason": reason, "detail": detail}]
    )


def _model_ids_for_features(
    manifest: pd.DataFrame, stats_index: pd.DataFrame
) -> tuple[list[str], pd.DataFrame]:
    manifest_ids = set(manifest["model_id"].astype(str))
    indexed_ids = set(stats_index["model_id"].astype(str))
    requested = sorted(manifest_ids & indexed_ids)
    outside = sorted(indexed_ids - manifest_ids)
    exclusions = pd.DataFrame(
        [
            {
                "stage": "feature_population",
                "model_id": model_id,
                "reason": "stats_without_manifest",
                "detail": "indexed stats model has no metadata/graph manifest row",
            }
            for model_id in outside
        ]
    )
    return requested, exclusions


def _analysis_population(
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    stage: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_ids = set(manifest["model_id"].astype(str))
    feature_ids = set(features["model_id"].astype(str))
    target_ids = set(targets["model_id"].astype(str))
    rows: list[dict[str, str]] = []
    for model_id in sorted(target_ids - manifest_ids):
        rows.append(
            {
                "stage": stage,
                "model_id": model_id,
                "reason": "target_without_manifest",
                "detail": "target row has no graph-derived manifest row",
            }
        )
    for model_id in sorted((target_ids & manifest_ids) - feature_ids):
        rows.append(
            {
                "stage": stage,
                "model_id": model_id,
                "reason": "target_without_features",
                "detail": "target row has no extracted feature record",
            }
        )
    population = (
        manifest.merge(features, on="model_id", how="inner", validate="one_to_one")
        .merge(targets, on="model_id", how="inner", validate="one_to_one")
        .sort_values("model_id", kind="stable", ignore_index=True)
    )
    return population, pd.DataFrame(rows)


def _performance_feasibility(
    population: pd.DataFrame,
    *,
    target: str,
    settings: RunSettings,
) -> tuple[pd.DataFrame | None, str | None]:
    eligible, _ = confirmatory_group_population(population)
    values = pd.to_numeric(eligible[target], errors="coerce")
    eligible = eligible.loc[np.isfinite(values)].reset_index(drop=True)
    group_count = eligible["ancestry_component"].nunique()
    if group_count < settings.outer_folds:
        return None, (
            f"target {target!r} has {group_count} eligible ancestry components; "
            f"outer_folds={settings.outer_folds}"
        )
    folds = grouped_outer_folds(
        eligible,
        n_splits=settings.outer_folds,
        random_state=settings.random_seed,
    )
    minimum_training_groups = min(
        eligible.iloc[fold.train_indices]["ancestry_component"].nunique()
        for fold in folds
    )
    if minimum_training_groups < settings.inner_folds:
        return None, (
            f"target {target!r} leaves only {minimum_training_groups} training "
            f"components in an outer fold; inner_folds={settings.inner_folds}"
        )
    return eligible, None


def _label_independent_cluster_count(
    model_count: int, *, rule: str = "sqrt_half"
) -> int:
    return label_independent_cluster_count(model_count, rule=rule)


def _empty_cluster_result() -> dict[str, pd.DataFrame]:
    return {
        "assignments": pd.DataFrame(columns=_CLUSTER_ASSIGNMENT_COLUMNS),
        "metrics": pd.DataFrame(columns=_CLUSTER_METRIC_COLUMNS),
        "exclusions": pd.DataFrame(columns=_LINEAGE_EXCLUSION_COLUMNS),
    }


def _annotate_lineage_inference(
    frame: pd.DataFrame,
    *,
    family_count_column: str,
    minimum_families: int,
    suppress_interval: bool = False,
) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        return result
    family_counts = pd.to_numeric(
        result[family_count_column], errors="coerce"
    ).fillna(0).astype(int)
    result["minimum_families"] = int(minimum_families)
    result["inferential"] = family_counts.ge(int(minimum_families))
    result["inference_reason"] = np.where(
        result["inferential"],
        "sufficient_independent_families",
        "insufficient_independent_families",
    )
    if suppress_interval:
        for column in ("ci_low", "ci_high"):
            if column in result.columns:
                result.loc[~result["inferential"], column] = np.nan
    return result


def _mark_posthoc_descriptive(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        return result
    result["inferential"] = False
    result["inference_reason"] = "post_hoc_descriptive_endpoint"
    return result


def _evaluate_lineage(
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    trajectories: pd.DataFrame,
    graph_edges: pd.DataFrame,
    settings: RunSettings,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], pd.DataFrame]:
    architecture_column = (
        "target_architecture"
        if "target_architecture" in manifest.columns
        and manifest["target_architecture"].astype("string").str.strip().ne("").any()
        else None
    )
    closed = evaluate_closed_set_lineage(
        manifest,
        features,
        trajectories,
        architecture_column=architecture_column,
        bootstrap_replicates=settings.bootstrap_replicates,
        random_state=settings.random_seed,
        min_feature_coverage=settings.min_feature_coverage,
        min_pair_coverage=settings.min_pair_coverage,
    )
    open_set = evaluate_unknown_family_rejection(
        manifest,
        features,
        trajectories,
        architecture_column=architecture_column,
        min_feature_coverage=settings.min_feature_coverage,
        min_pair_coverage=settings.min_pair_coverage,
    )
    observed_family = evaluate_cross_fitted_family_representatives(
        manifest,
        features,
        trajectories,
        architecture_column=architecture_column,
        bootstrap_replicates=settings.bootstrap_replicates,
        random_state=settings.random_seed,
        min_feature_coverage=settings.min_feature_coverage,
        min_pair_coverage=settings.min_pair_coverage,
    )
    closed["metrics"] = _annotate_lineage_inference(
        closed["metrics"],
        family_count_column="n_families",
        minimum_families=settings.minimum_lineage_families,
    )
    closed["confidence_intervals"] = _annotate_lineage_inference(
        closed["confidence_intervals"],
        family_count_column="n_families",
        minimum_families=settings.minimum_lineage_families,
        suppress_interval=True,
    )
    closed["contrasts"] = _annotate_lineage_inference(
        closed["contrasts"],
        family_count_column="n_families",
        minimum_families=settings.minimum_lineage_families,
        suppress_interval=True,
    )
    open_set["metrics"] = _annotate_lineage_inference(
        open_set["metrics"],
        family_count_column="n_folds",
        minimum_families=settings.minimum_lineage_families,
    )
    for key in ("metrics", "confidence_intervals", "contrasts"):
        observed_family[key] = _mark_posthoc_descriptive(observed_family[key])

    cluster_exclusions = pd.DataFrame()
    if len(features) >= 3:
        try:
            cluster = cluster_lineage_diagnostics(
                features,
                manifest,
                n_clusters=None,
                random_state=settings.random_seed,
                min_feature_coverage=settings.min_feature_coverage,
                cluster_count_rule=settings.cluster_count_rule,
            )
        except ValueError as error:
            cluster = _empty_cluster_result()
            cluster_exclusions = _exclusion(
                "clustering", "infeasible_cluster_diagnostic", str(error)
            )
    else:
        cluster = _empty_cluster_result()
        cluster_exclusions = _exclusion(
            "clustering",
            "infeasible_cluster_diagnostic",
            f"requires at least three eligible models; feature_rows={len(features)}",
        )

    parent = evaluate_exact_parent_diagnostic(
        graph_edges,
        manifest,
        features,
        min_feature_coverage=settings.min_feature_coverage,
        min_pair_coverage=settings.min_pair_coverage,
    )
    metrics = {
        "closed_set": _records(closed["metrics"]),
        "closed_set_confidence_intervals": _records(
            closed["confidence_intervals"]
        ),
        "closed_set_contrasts": _records(closed["contrasts"]),
        "open_set": _records(open_set["metrics"]),
        "observed_family": _records(observed_family["metrics"]),
        "observed_family_confidence_intervals": _records(
            observed_family["confidence_intervals"]
        ),
        "observed_family_contrasts": _records(observed_family["contrasts"]),
        "cluster": _records(cluster["metrics"]),
        "exact_parent": _records(parent["metrics"]),
    }
    tables = {
        "lineage_predictions.csv": closed["predictions"],
        "lineage_candidates.csv": closed["candidates"],
        "lineage_closed_metrics.csv": closed["metrics"],
        "lineage_closed_confidence_intervals.csv": closed[
            "confidence_intervals"
        ],
        "lineage_closed_contrasts.csv": closed["contrasts"],
        "lineage_open_scores.csv": open_set["scores"],
        "lineage_open_fold_gallery.csv": open_set["fold_gallery"],
        "lineage_open_fold_metrics.csv": open_set.get(
            "fold_metrics", pd.DataFrame()
        ),
        "lineage_open_metrics.csv": open_set["metrics"],
        "lineage_representative_candidates.csv": observed_family["candidates"],
        "lineage_representative_predictions.csv": observed_family["predictions"],
        "lineage_representative_metrics.csv": observed_family["metrics"],
        "lineage_representative_confidence_intervals.csv": observed_family[
            "confidence_intervals"
        ],
        "lineage_representative_contrasts.csv": observed_family["contrasts"],
        "lineage_risk_coverage.csv": open_set["risk_coverage"],
        "lineage_cluster_assignments.csv": cluster["assignments"],
        "cluster_assignments.csv": cluster["assignments"],
        "lineage_cluster_metrics.csv": cluster["metrics"],
        "lineage_parent_candidates.csv": parent["candidates"],
        "lineage_parent_predictions.csv": parent["predictions"],
        "lineage_parent_metrics.csv": parent["metrics"],
    }
    exclusions = _combine_exclusions(
        (closed["exclusions"], "lineage_closed"),
        (open_set["exclusions"], "lineage_open"),
        (observed_family["exclusions"], "lineage_representative"),
        (cluster["exclusions"], "clustering"),
        (cluster_exclusions, "clustering"),
        (parent["exclusions"], "exact_parent"),
    )
    return metrics, tables, exclusions


def _evaluate_performance(
    development: pd.DataFrame,
    *,
    settings: RunSettings,
    manifest: pd.DataFrame,
    external: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    development_metric_frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []
    primary_increment: dict[str, Any] | None = None
    secondary_increments = pd.DataFrame()
    holdout_summary: dict[str, Any] = {"status": "disabled"}
    exclusions: list[pd.DataFrame] = []

    _, primary_infeasible = _performance_feasibility(
        development, target="modern_average", settings=settings
    )
    if primary_infeasible is not None:
        exclusions.append(
            _exclusion(
                "performance", "infeasible_grouped_cv", primary_infeasible
            ).assign(target="modern_average")
        )
    else:
        primary_predictions, primary_metrics, primary_audit = evaluate_grouped_ridge(
            development,
            target="modern_average",
            outer_splits=settings.outer_folds,
            inner_splits=settings.inner_folds,
            ridge_alphas=settings.ridge_alphas,
            random_state=settings.random_seed,
        )
        primary_predictions.insert(0, "evaluation_scope", "development")
        prediction_frames.append(primary_predictions)
        development_metric_frames.append(primary_metrics)
        audit_frame = pd.DataFrame(primary_audit)
        audit_frame.insert(0, "target", "modern_average")
        audit_frames.append(audit_frame)
        interval = paired_component_bootstrap(
            primary_predictions,
            n_bootstrap=settings.bootstrap_replicates,
            random_state=settings.random_seed,
            minimum_components=settings.minimum_components,
        )
        primary_increment = {
            "target": "modern_average",
            **interval,
            "p_value_raw": paired_component_signflip_pvalue(
                primary_predictions,
                n_replicates=settings.signflip_replicates,
                random_state=settings.random_seed,
                minimum_components=settings.minimum_components,
            ),
        }
        if settings.holdout_repeats > 0:
            try:
                holdout_predictions, holdout_summary, holdout_audit = (
                    evaluate_repeated_group_holdout(
                        development,
                        target="modern_average",
                        repeats=settings.holdout_repeats,
                        test_fraction=settings.holdout_test_fraction,
                        inner_splits=settings.inner_folds,
                        ridge_alphas=settings.ridge_alphas,
                        bootstrap_replicates=settings.bootstrap_replicates,
                        random_state=settings.random_seed,
                        minimum_components=settings.minimum_components,
                    )
                )
                holdout_predictions.insert(
                    0,
                    "evaluation_scope",
                    holdout_predictions["repeat"].map(
                        lambda repeat: f"development_holdout_{int(repeat):03d}"
                    ),
                )
                holdout_predictions["external_slice"] = ""
                prediction_frames.append(holdout_predictions)
                holdout_audit_frame = pd.DataFrame(holdout_audit)
                holdout_audit_frame.insert(0, "target", "modern_average")
                holdout_audit_frame.insert(
                    0, "evaluation_scope", "development_repeated_holdout"
                )
                audit_frames.append(holdout_audit_frame)
            except ValueError as error:
                holdout_summary = {
                    "status": "unavailable",
                    "reason": "infeasible_repeated_group_holdout",
                    "detail": str(error),
                }
                exclusions.append(
                    _exclusion(
                        "performance_holdout",
                        "repeated_group_holdout_unavailable",
                        str(error),
                    )
                )
        try:
            secondary = evaluate_secondary_endpoints(
                development,
                outer_splits=settings.outer_folds,
                inner_splits=settings.inner_folds,
                ridge_alphas=settings.ridge_alphas,
                bootstrap_replicates=settings.bootstrap_replicates,
                signflip_replicates=settings.signflip_replicates,
                random_state=settings.random_seed,
                minimum_components=settings.minimum_components,
            )
            secondary_predictions = secondary["predictions"]
            secondary_predictions.insert(0, "evaluation_scope", "development")
            prediction_frames.append(secondary_predictions)
            development_metric_frames.append(secondary["metrics"])
            secondary_increments = secondary["increments"]
            secondary_audit = pd.DataFrame(secondary["audit"])
            if not secondary_audit.empty:
                audit_frames.append(secondary_audit)
        except (TargetSemanticsError, ValueError) as error:
            exclusions.append(
                _exclusion(
                    "performance_secondary",
                    "secondary_endpoints_unavailable",
                    str(error),
                )
            )

    secondary_adjusted = (
        secondary_increments.set_index("target")["p_value_holm"].to_dict()
        if not secondary_increments.empty
        else {}
    )

    external_status: dict[str, Any] = {
        "status": "unavailable",
        "reason": "not_supplied",
    }
    external_metrics = pd.DataFrame()
    external_audit = pd.DataFrame()
    if external is not None:
        eligible_development, _ = confirmatory_group_population(development)
        if "development_artifact_available" not in manifest.columns:
            raise ValueError(
                "manifest requires development_artifact_available for external transfer"
            )
        development_component_universe = manifest.loc[
            manifest["development_artifact_available"].fillna(False).astype(bool)
        ].copy()
        development_groups = eligible_development["ancestry_component"].nunique()
        if development_groups < settings.inner_folds:
            external_status = {
                "status": "unavailable",
                "reason": "infeasible_development_inner_cv",
                "development_components": int(development_groups),
            }
        else:
            try:
                external_predictions, external_metrics, external_audit = (
                    evaluate_external_transfer(
                        development,
                        external,
                        target="modern_average",
                        development_component_universe=development_component_universe,
                        inner_splits=settings.inner_folds,
                        ridge_alphas=settings.ridge_alphas,
                        random_state=settings.random_seed,
                    )
                )
                external_predictions.insert(0, "evaluation_scope", "external")
                prediction_frames.append(external_predictions)
                strict = external_predictions.loc[
                    external_predictions["external_slice"].eq("component_disjoint")
                ]
                if strict.empty:
                    external_status = {
                        "status": "unavailable",
                        "reason": "no_component_disjoint_rows",
                    }
                elif strict["ancestry_component"].nunique() < settings.minimum_components:
                    external_status = {
                        "status": "descriptive",
                        "reason": "insufficient_component_disjoint_components",
                        "headline_slice": "component_disjoint",
                        "component_disjoint_models": int(strict["model_id"].nunique()),
                        "component_disjoint_components": int(
                            strict["ancestry_component"].nunique()
                        ),
                    }
                else:
                    external_status = {
                        "status": "available",
                        "headline_slice": "component_disjoint",
                        "component_disjoint_models": int(
                            strict["model_id"].nunique()
                        ),
                        "component_disjoint_components": int(
                            strict["ancestry_component"].nunique()
                        ),
                    }
            except (TargetSemanticsError, ValueError) as error:
                external_status = {
                    "status": "unavailable",
                    "reason": "incompatible_or_infeasible",
                    "detail": str(error),
                }
                exclusions.append(
                    _exclusion(
                        "external_transfer",
                        "external_unavailable",
                        str(error),
                    )
                )

    external_status = _apply_external_evidence_role(
        external_status,
        role=settings.external_evidence_role,
    )

    predictions = (
        pd.concat(prediction_frames, ignore_index=True, sort=False)
        if prediction_frames
        else pd.DataFrame(columns=_PERFORMANCE_PREDICTION_COLUMNS)
    )
    development_metrics = (
        pd.concat(development_metric_frames, ignore_index=True)
        if development_metric_frames
        else pd.DataFrame(columns=_PERFORMANCE_METRIC_COLUMNS)
    )
    audit = (
        pd.concat(audit_frames, ignore_index=True, sort=False)
        if audit_frames
        else pd.DataFrame(columns=_PERFORMANCE_AUDIT_COLUMNS)
    )

    primary_metrics = development_metrics.loc[
        development_metrics["target"].eq("modern_average")
    ]
    decision: dict[str, Any]
    primary_summary: dict[str, Any] = {"target": "modern_average"}
    if primary_metrics.empty or primary_increment is None:
        decision = {
            "passed": False,
            "reasons": ["primary_analysis_infeasible"],
        }
    else:
        maes = primary_metrics.set_index("feature_block")["group_weighted_mae"]
        htsr_row = primary_metrics.loc[
            primary_metrics["feature_block"].eq("M+S+H")
        ].iloc[0]
        primary_summary.update(
            {
                "n_models": int(htsr_row["n_models"]),
                "n_components": int(htsr_row["n_components"]),
                "baseline_group_weighted_mae": float(maes["M+S"]),
                "group_weighted_mae": float(maes["M+S+H"]),
                "spearman": float(htsr_row["spearman"]),
                "htsr_delta_mae": float(primary_increment["delta_mae"]),
                "htsr_ci_low": float(primary_increment["ci_low"]),
                "htsr_ci_high": float(primary_increment["ci_high"]),
            }
        )
        external_delta = float("nan")
        external_available = (
            external_status.get(
                "component_contract_status", external_status.get("status")
            )
            == "available"
        )
        if external_available:
            strict_metrics = external_metrics.loc[
                external_metrics["external_slice"].eq("component_disjoint")
            ].set_index("feature_block")
            if {"M+S", "M+S+H"}.issubset(strict_metrics.index):
                external_delta = float(
                    strict_metrics.at["M+S+H", "group_weighted_mae"]
                    - strict_metrics.at["M+S", "group_weighted_mae"]
                )
            else:
                external_available = False
        decision = assess_htsr_increment(
            baseline_mae=float(maes["M+S"]),
            htsr_mae=float(maes["M+S+H"]),
            ci_low=float(primary_increment["ci_low"]),
            ci_high=float(primary_increment["ci_high"]),
            external_delta=external_delta,
            external_available=external_available,
            external_confirmatory=(
                settings.external_evidence_role == "confirmatory_locked"
            ),
            external_slice=str(external_status.get("headline_slice", "")),
            n_components=int(primary_increment["n_components"]),
            minimum_components=settings.minimum_components,
        )

    metrics_payload = {
        "primary": primary_summary,
        "development": _records(development_metrics),
        "primary_increment": primary_increment or {},
        "secondary_increments": _records(secondary_increments),
        "secondary_holm_adjusted_p": secondary_adjusted,
        "repeated_group_holdout": holdout_summary,
        "external": {
            **external_status,
            "metrics": _records(external_metrics),
            "fit_audit": _records(external_audit),
        },
        "primary_decision": decision,
    }
    exclusion_frame = _combine_exclusions(
        *((frame, "performance") for frame in exclusions)
    )
    return predictions, metrics_payload, exclusion_frame, audit


def _apply_external_evidence_role(
    status: Mapping[str, Any], *, role: str
) -> dict[str, Any]:
    result = dict(status)
    result["evidence_role"] = role
    if role == "confirmatory_locked" or result.get("status") == "unavailable":
        return result
    if role != "reused_descriptive":
        raise ValueError(f"unknown external evidence role: {role}")
    result["component_contract_status"] = result.get("status")
    if "reason" in result:
        result["component_contract_reason"] = result["reason"]
    result["status"] = "descriptive_reuse"
    result["reason"] = "previous_method_development_reuse"
    return result


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.to_dict(orient="records") if not frame.empty else []


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if value is pd.NA or value is None:
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _performance_effect_interpretation(
    delta_mae: object,
    decision: Mapping[str, Any],
) -> str:
    try:
        delta = float(delta_mae)
    except (TypeError, ValueError):
        delta = float("nan")
    if not math.isfinite(delta):
        return "The HTSR increment was not estimable."
    if delta < 0 and bool(decision.get("passed", False)):
        return "Adding H satisfied the primary improvement rule."
    if delta < 0:
        return (
            "The point estimate favored H, but the primary evidence was insufficient "
            "to claim improvement."
        )
    if delta > 0:
        return "The point estimate worsened with H."
    return "The point estimate was unchanged with H."


def _report(
    *,
    run_id: str,
    counts: Mapping[str, int],
    lineage_metrics: Mapping[str, Any],
    performance_metrics: Mapping[str, Any],
) -> str:
    external = performance_metrics["external"]
    holdout = performance_metrics.get("repeated_group_holdout", {})
    decision = performance_metrics["primary_decision"]
    primary = performance_metrics.get("primary", {})
    development = performance_metrics.get("development", [])
    primary_rows = {
        str(row.get("feature_block")): row
        for row in development
        if row.get("target") == "modern_average"
    }

    def number(value: object, digits: int = 3) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "n/a"
        return f"{numeric:.{digits}f}" if math.isfinite(numeric) else "n/a"

    def lineage_metric(section: str, method: str, metric: str) -> str:
        for row in lineage_metrics.get(section, []):
            if row.get("method") == method and row.get("metric") == metric:
                return number(row.get("estimate"))
        return "n/a"

    def lineage_row(section: str, method: str, metric: str) -> Mapping[str, Any]:
        for row in lineage_metrics.get(section, []):
            if row.get("method") == method and row.get("metric") == metric:
                return row
        return {}

    closed_scope = lineage_row(
        "closed_set", "combined", "root_macro_top1"
    )
    open_scope = lineage_row("open_set", "combined", "auroc")
    observed_scope = lineage_row(
        "observed_family", "combined", "root_macro_top1"
    )
    closed_status = (
        "inferential" if closed_scope.get("inferential", False) else "descriptive"
    )
    open_status = (
        "inferential" if open_scope.get("inferential", False) else "descriptive"
    )
    minimum_families = int(closed_scope.get("minimum_families", 0) or 0)
    primary_models = int(primary.get("n_models", 0) or 0)
    primary_components = int(primary.get("n_components", 0) or 0)
    candidate_models = int(counts.get("development_candidate_rows", 0) or 0)
    eligible_models = int(counts.get("development_group_eligible_rows", 0) or 0)
    eligible_components = int(
        counts.get("development_group_eligible_components", 0) or 0
    )
    closure_models = int(counts.get("development_closure_eligible_rows", 0) or 0)
    closure_components = int(
        counts.get("development_closure_components", 0) or 0
    )

    cluster = next(iter(lineage_metrics.get("cluster", [])), {})
    parent = next(iter(lineage_metrics.get("exact_parent", [])), {})
    delta = primary.get("htsr_delta_mae")
    effect = _performance_effect_interpretation(delta, decision)
    reasons = ", ".join(str(reason) for reason in decision.get("reasons", [])) or "none"
    external_rows = {
        str(row.get("feature_block")): row
        for row in external.get("metrics", [])
        if row.get("external_slice") == "component_disjoint"
    }
    external_delta = float("nan")
    if {"M+S", "M+S+H"}.issubset(external_rows):
        external_delta = float(
            external_rows["M+S+H"]["group_weighted_mae"]
            - external_rows["M+S"]["group_weighted_mae"]
        )
    external_status = str(external.get("status", "unavailable"))
    if external_status == "available":
        external_interpretation = (
            "External component-disjoint transfer met the minimum-component contract; "
            "it supports nonreversal but cannot replace the primary development rule."
        )
    elif external_status == "descriptive":
        external_interpretation = (
            "External transfer remained descriptive because the component-disjoint "
            "minimum was not met."
        )
    elif external_status == "descriptive_reuse":
        external_interpretation = (
            "External component-disjoint transfer is a reused descriptive sensitivity "
            "because this source informed earlier method development; it cannot confirm "
            "nonreversal."
        )
    else:
        external_interpretation = (
            "External transfer was unavailable for confirmatory nonreversal."
        )
    return "\n".join(
        [
            "# HTSR Combined Study",
            "",
            f"Run ID: `{run_id}`",
            "",
            "## Population",
            "",
            f"- Manifest rows: {counts['manifest_rows']}",
            f"- Development feature rows: {counts['development_feature_rows']}",
            f"- Confirmatory feature rows: {counts.get('development_confirmatory_feature_rows', 0)}",
            f"- Development target rows: {counts['development_target_rows']}",
            f"- Primary candidate population: {candidate_models} joined models",
            f"- Primary graph-eligible population: {eligible_models} models / {eligible_components} graph-supported ancestry components",
            f"- Primary evaluated OOF population: {primary_models} models / {primary_components} graph-supported ancestry components",
            f"- Merge-retaining grouping diagnostic: {closure_models} edge-supported models / {closure_components} ancestry-closure components (not fitted)",
            "- Manifest membership is derived from analysis sources; supplied metadata is enrichment only.",
            "",
            "## Lineage",
            "",
            f"- Closed-set scope: {int(closed_scope.get('n_queries', 0) or 0)} queries / {int(closed_scope.get('n_families', 0) or 0)} root families",
            f"- Closed-set inference status: {closed_status} (minimum {minimum_families} independent families)",
            f"- Combined root-macro top-1: {lineage_metric('closed_set', 'combined', 'root_macro_top1')}",
            f"- Paper-style stored-DKS root-macro top-1: {lineage_metric('closed_set', 'paper_style_alpha_dks', 'root_macro_top1')}",
            f"- Open-set scope: {int(open_scope.get('n_folds', 0) or 0)} held-out root families",
            f"- Open-set inference status: {open_status} (minimum {int(open_scope.get('minimum_families', minimum_families) or 0)} independent families)",
            f"- Combined LOFO pseudo-unknown AUROC: {lineage_metric('open_set', 'combined', 'auroc')}",
            f"- Cross-fitted observed-family scope: {int(observed_scope.get('n_queries', 0) or 0)} queries / {int(observed_scope.get('n_families', 0) or 0)} audited families (descriptive)",
            f"- Cross-fitted observed-family combined top-1: {lineage_metric('observed_family', 'combined', 'root_macro_top1')}",
            f"- Cross-fitted observed-family spectral-shape top-1: {lineage_metric('observed_family', 'spectral_shape', 'root_macro_top1')}",
            f"- Cross-fitted observed-family config top-1: {lineage_metric('observed_family', 'config', 'root_macro_top1')}",
            f"- Clustered models / labeled models: {cluster.get('n_clustered', 0)} / {cluster.get('n_labeled', 0)}",
            f"- Unsupervised ARI / NMI / matched family recall: {number(cluster.get('ari'))} / {number(cluster.get('nmi'))} / {number(cluster.get('root_macro_purity'))}",
            f"- Exact-parent sensitivity top-1: {number(parent.get('top1_accuracy'))}",
            "",
            "## Performance",
            "",
            f"- M MAE: {number(primary_rows.get('M', {}).get('group_weighted_mae'))}",
            f"- M+S MAE: {number(primary_rows.get('M+S', {}).get('group_weighted_mae'))}",
            f"- M+S+H MAE: {number(primary_rows.get('M+S+H', {}).get('group_weighted_mae'))}",
            f"- Paired H increment (positive is worse): {number(delta)} [{number(primary.get('htsr_ci_low'))}, {number(primary.get('htsr_ci_high'))}]",
            f"- Repeated grouped holdout: {int(holdout.get('n_repeats', 0) or 0)} repeats at {number(holdout.get('test_fraction'), 2)} test fraction",
            f"- Repeated-holdout H increment (positive is worse): {number(holdout.get('htsr_delta_mae'))} [{number(holdout.get('ci_low'))}, {number(holdout.get('ci_high'))}]",
            f"- Primary HTSR decision passed: {bool(decision.get('passed', False))}",
            f"- Decision reasons: {reasons}",
            f"- External transfer status: {external['status']}",
            f"- External transfer reason: {external.get('reason', 'available')}",
            f"- External component-disjoint population: {int(external.get('component_disjoint_models', 0) or 0)} models / {int(external.get('component_disjoint_components', 0) or 0)} components",
            f"- External H increment (positive is worse): {number(external_delta)}",
            "",
            "## Interpretation",
            "",
            f"- {effect}",
            "- Closed-set lineage comparisons are limited to root-eligible audited families; clustering metrics use their separately reported labeled subset.",
            "- LOFO pseudo-unknown scores measure discrimination under held-out audited families; they are not a calibrated deployment rejection threshold.",
            "- Cross-fitted observed-family retrieval is known reference-library attribution, not canonical-root recovery or unseen-family generalization.",
            "- Performance intervals are conditional on the frozen grouped folds and their out-of-fold predictions.",
            "- The repeated Phase-2 holdout is an internal development sensitivity, not an untouched external test.",
            "- Spearman, Kendall, and R2 are model-level descriptive statistics; primary MAE is component weighted.",
            f"- {external_interpretation}",
            "- Alpha uses the stored DKS estimator; this is not a fixed-Hill replication of the paper.",
            "- Targets and stats are joined by canonical model ID because checkpoint SHA is not available in stats artifacts.",
            "",
        ]
    )


def run_study(
    inputs: StudyInputs,
    settings: RunSettings,
    dry_run: bool = False,
) -> StudyRunResult:
    """Validate, fingerprint, evaluate, cache, and atomically persist one study."""

    if not isinstance(inputs, StudyInputs):
        raise TypeError("inputs must be a StudyInputs instance")
    if not isinstance(settings, RunSettings):
        raise TypeError("settings must be a RunSettings instance")
    if not isinstance(dry_run, bool):
        raise TypeError("dry_run must be boolean")

    _validate_sources(inputs)
    source_paths = inputs.source_paths()
    source_digests = {
        name: path_digest(path) for name, path in sorted(source_paths.items())
    }
    implementation_identity = {
        "implementation_version": STUDY_IMPLEMENTATION_VERSION,
        "implementation_digest": _study_implementation_digest(),
        "feature_extractor_digest": _feature_extractor_digest(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "alpha_estimator": ALPHA_ESTIMATOR,
        "alpha_estimator_version": ALPHA_ESTIMATOR_VERSION,
    }
    run_id = study_run_id(
        source_paths,
        settings,
        source_digests=source_digests,
        identity=implementation_identity,
    )
    run_dir = inputs.output_root / run_id
    if not dry_run and run_dir.exists():
        return StudyRunResult(
            run_id=run_id,
            run_dir=run_dir,
            dry_run=False,
            cache_reused=True,
            summary=_validated_run_summary(run_dir, run_id),
        )

    loaded = _load_contracts(inputs, graph_digest=source_digests["graph"])
    contract_summary = loaded.summary()
    if dry_run:
        return StudyRunResult(
            run_id=run_id,
            run_dir=None,
            dry_run=True,
            cache_reused=False,
            summary=contract_summary,
        )

    development_model_ids, development_population_exclusions = (
        _model_ids_for_features(loaded.manifest, loaded.development_index)
    )
    development_features, trajectories, development_feature_exclusions, reused, development_cache = (
        _feature_cache(
            inputs.cache_root,
            stats_digest=source_digests["development_stats"],
            stats_index=loaded.development_index,
            model_ids=development_model_ids,
        )
    )
    development_analysis_features, development_schema_exclusions = (
        confirmatory_schema_population(
            development_features,
            min_role_coverage=settings.min_role_coverage,
        )
    )
    development_analysis_ids = set(
        development_analysis_features["model_id"].astype(str)
    )
    analysis_trajectories = trajectories.loc[
        trajectories["model_id"].astype(str).isin(development_analysis_ids)
    ].reset_index(drop=True)

    external_features: pd.DataFrame | None = None
    external_analysis_features: pd.DataFrame | None = None
    external_feature_exclusions = pd.DataFrame()
    external_schema_exclusions = pd.DataFrame()
    external_population_exclusions = pd.DataFrame()
    external_cache: Path | None = None
    if loaded.external_index is not None and inputs.external_stats is not None:
        external_model_ids, external_population_exclusions = _model_ids_for_features(
            loaded.manifest, loaded.external_index
        )
        (
            external_features,
            _,
            external_feature_exclusions,
            external_reused,
            external_cache,
        ) = _feature_cache(
            inputs.cache_root,
            stats_digest=source_digests["external_stats"],
            stats_index=loaded.external_index,
            model_ids=external_model_ids,
        )
        reused = reused or external_reused
        external_analysis_features, external_schema_exclusions = (
            confirmatory_schema_population(
                external_features,
                min_role_coverage=settings.min_role_coverage,
            )
        )

    lineage_metrics, lineage_tables, lineage_exclusions = _evaluate_lineage(
        loaded.manifest,
        development_analysis_features,
        analysis_trajectories,
        loaded.graph_edges,
        settings,
    )

    development_population, development_join_exclusions = _analysis_population(
        loaded.manifest,
        development_analysis_features,
        loaded.development_targets,
        stage="development_population",
    )
    development_group_population, development_group_exclusions = confirmatory_group_population(
        development_population
    )
    closure_groups = development_population["ancestor_closure_component"].astype(
        "string"
    )
    closure_provenance = development_population[
        "ancestor_closure_provenance"
    ].astype("string")
    closure_edges = pd.to_numeric(
        development_population["ancestor_closure_edge_count"], errors="coerce"
    ).fillna(0)
    closure_population = development_population.loc[
        closure_groups.notna()
        & closure_groups.str.strip().ne("")
        & closure_provenance.eq("graph")
        & closure_edges.ge(1)
    ].copy()
    external_population: pd.DataFrame | None = None
    external_join_exclusions = pd.DataFrame()
    external_group_exclusions = pd.DataFrame()
    if external_analysis_features is not None and loaded.external_targets is not None:
        external_population, external_join_exclusions = _analysis_population(
            loaded.manifest,
            external_analysis_features,
            loaded.external_targets,
            stage="external_population",
        )
        _, external_group_exclusions = confirmatory_group_population(
            external_population
        )

    (
        performance_predictions,
        performance_metrics,
        performance_exclusions,
        performance_audit,
    ) = _evaluate_performance(
        development_population,
        settings=settings,
        manifest=loaded.manifest,
        external=external_population,
    )

    exclusions = _combine_exclusions(
        (loaded.manifest_exclusions, "manifest"),
        (loaded.development_index_exclusions, "development_stats_index"),
        (loaded.development_target_exclusions, "development_targets"),
        (development_population_exclusions, "development_feature_population"),
        (development_feature_exclusions, "development_features"),
        (development_schema_exclusions, "development_confirmatory_schema"),
        (development_join_exclusions, "development_population"),
        (development_group_exclusions, "development_grouping"),
        (loaded.external_index_exclusions, "external_stats_index"),
        (loaded.external_target_exclusions, "external_targets"),
        (external_population_exclusions, "external_feature_population"),
        (external_feature_exclusions, "external_features"),
        (external_schema_exclusions, "external_confirmatory_schema"),
        (external_join_exclusions, "external_population"),
        (external_group_exclusions, "external_grouping"),
        (lineage_exclusions, "lineage"),
        (performance_exclusions, "performance"),
    )

    development_components = development_group_population[
        "ancestry_component"
    ].nunique()
    counts = {
        **contract_summary,
        "development_feature_rows": len(development_features),
        "development_confirmatory_feature_rows": len(
            development_analysis_features
        ),
        "development_trajectory_rows": len(trajectories),
        "external_feature_rows": (
            len(external_features) if external_features is not None else 0
        ),
        "external_confirmatory_feature_rows": (
            len(external_analysis_features)
            if external_analysis_features is not None
            else 0
        ),
        "development_components": int(development_components),
        "development_candidate_rows": len(development_population),
        "development_group_eligible_rows": len(development_group_population),
        "development_group_eligible_components": int(development_components),
        "development_closure_eligible_rows": len(closure_population),
        "development_closure_components": int(
            closure_population["ancestor_closure_component"].nunique()
        ),
        "cluster_assignment_rows": len(
            lineage_tables["cluster_assignments.csv"]
        ),
        "lineage_prediction_rows": len(
            lineage_tables["lineage_predictions.csv"]
        ),
        "performance_prediction_rows": len(performance_predictions),
        "exclusion_rows": len(exclusions),
    }
    row_counts = {
        "metadata": loaded.metadata_rows,
        "labels": loaded.label_rows,
        "graph": len(loaded.graph_nodes) + len(loaded.graph_edges),
        "development_stats": len(loaded.development_index),
        "development_targets": loaded.development_target_source_rows,
    }
    if loaded.external_index is not None:
        row_counts["external_stats"] = len(loaded.external_index)
        row_counts["external_targets"] = loaded.external_target_source_rows
    provenance = build_provenance(
        source_paths,
        row_counts=row_counts,
        settings=settings,
        manifest_rows=len(loaded.manifest),
        exclusions=exclusions,
        source_digests=source_digests,
    )
    provenance.update(
        {
            "run_id": run_id,
            "implementation_identity": implementation_identity,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "graph_fingerprint": loaded.graph_fingerprint,
            "counts": counts,
            "cache": {
                "development": str(development_cache),
                "external": str(external_cache) if external_cache is not None else None,
            },
            "external": {
                key: value
                for key, value in performance_metrics["external"].items()
                if key not in {"metrics", "fit_audit"}
            },
            "methodological_scope": {
                "alpha_estimator_limitation": (
                    "stored DKS power-law estimator; not fixed-Hill replication"
                ),
                "open_set_claim": (
                    "LOFO pseudo-unknown discrimination; no deployment threshold"
                ),
                "performance_inference": (
                    "conditional on frozen grouped folds and OOF predictions"
                ),
                "performance_grouping": (
                    "full-graph single-source roots; declared merge, multi-parent, "
                    "and downstream ambiguous lineages excluded"
                ),
                "performance_grouping_sensitivity": (
                    "merge-retaining weak components of the outcome-independent "
                    "development/external stats ancestor closure; not fitted"
                ),
                "checkpoint_matching": (
                    "canonical_model_id; checkpoint SHA is not available in stats artifacts"
                ),
            },
        }
    )

    def write_run(temporary: Path) -> None:
        loaded.manifest.to_csv(temporary / "manifest.csv", index=False)
        loaded.graph_edges.to_csv(temporary / "graph_edges.csv", index=False)
        loaded.development_index.to_csv(
            temporary / "development_stats_index.csv", index=False
        )
        loaded.development_targets.to_csv(
            temporary / "development_targets.csv", index=False
        )
        development_features.to_csv(temporary / "feature_table.csv", index=False)
        trajectories.to_csv(temporary / "alpha_trajectories.csv", index=False)
        if external_features is not None:
            external_features.to_csv(
                temporary / "external_feature_table.csv", index=False
            )
        if loaded.external_index is not None:
            loaded.external_index.to_csv(
                temporary / "external_stats_index.csv", index=False
            )
        if loaded.external_targets is not None:
            loaded.external_targets.to_csv(
                temporary / "external_targets.csv", index=False
            )
        for filename, frame in lineage_tables.items():
            frame.to_csv(temporary / filename, index=False)
        performance_predictions.to_csv(
            temporary / "performance_predictions.csv", index=False
        )
        performance_audit.to_csv(temporary / "performance_fit_audit.csv", index=False)
        exclusions.to_csv(temporary / "exclusions.csv", index=False)
        _write_json(temporary / "lineage_metrics.json", lineage_metrics)
        _write_json(temporary / "performance_metrics.json", performance_metrics)
        _write_json(temporary / "provenance.json", provenance)
        (temporary / "report.md").write_text(
            _report(
                run_id=run_id,
                counts=counts,
                lineage_metrics=lineage_metrics,
                performance_metrics=performance_metrics,
            ),
            encoding="utf-8",
        )
        write_artifact_manifest(temporary, run_id=run_id)
        missing = [
            filename
            for filename in REQUIRED_ARTIFACTS
            if not (temporary / filename).is_file()
        ]
        if missing:
            raise RuntimeError(f"atomic run is missing required artifacts: {missing}")
        _validated_run_summary(temporary, run_id)

    published = _atomic_directory(run_dir, write_run)
    if not published:
        counts = _validated_run_summary(run_dir, run_id)
        return StudyRunResult(
            run_id=run_id,
            run_dir=run_dir,
            dry_run=False,
            cache_reused=True,
            summary=counts,
        )
    return StudyRunResult(
        run_id=run_id,
        run_dir=run_dir,
        dry_run=False,
        cache_reused=reused,
        summary=counts,
    )


__all__ = [
    "REQUIRED_ARTIFACTS",
    "STUDY_IMPLEMENTATION_VERSION",
    "StudyRunResult",
    "run_study",
]
