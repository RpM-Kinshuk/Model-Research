from __future__ import annotations

"""Run the metadata-isolated HTSR lineage and performance study."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spectral_research.study.artifacts import write_artifact_manifest
from spectral_research.study.contracts import canonical_model_id
from spectral_research.study.performance import MODERN_TASKS
from spectral_research.study.partial_roles import (
    build_partial_role_cache,
    combine_partial_role_caches,
    cross_validated_partial_performance_freeze,
    extend_strict_clusters,
    masking_stability,
    partial_profile_split_table,
    select_reliable_role_patterns,
)
from spectral_research.study.scaleup import sha256_file
from spectral_research.study.weight_only import (
    build_rank_profile_cache,
    cluster_holdout_split_table,
    combine_profile_caches,
    cross_validated_performance_freeze,
    discover_clusters,
    paper_style_repeated_performance,
    performance_metrics,
    profile_split_table,
    summarize_leakage,
    supplied_gallery_predictions,
    validate_clusters,
    validate_gallery_predictions,
)


LEGACY_TASKS = ("ARC", "HellaSwag", "MMLU", "TruthfulQA", "Winogrande", "GSM8K")
MODERN_TARGETS = ("modern_average", *MODERN_TASKS)
LEGACY_TARGETS = ("legacy_average", *LEGACY_TASKS)
STRICT_MODERN_RESOLUTIONS = {
    "single_or_identical_score_vector",
    "selected_source_row",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run weight-only clustering, gallery routing, and score prediction."
    )
    parser.add_argument("--study-run", type=Path, required=True)
    parser.add_argument("--validation-run", type=Path, required=True)
    parser.add_argument(
        "--gallery",
        type=Path,
        default=Path("research_inputs/lineage_candidate_gallery.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("research_runs/htsr_weight_only"),
    )
    parser.add_argument(
        "--minimum-neighbor-retention",
        type=float,
        default=0.90,
        help=(
            "Choose the smallest role count whose masked summaries retain this "
            "share of full-summary nearest neighbors in the top 10."
        ),
    )
    return parser.parse_args()


def canonical(value: object) -> str:
    return canonical_model_id(value) or ""


def dataframe_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def _strict_targets(validation_run: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    modern = pd.read_csv(validation_run / "performance_modern_targets.csv", low_memory=False)
    legacy = pd.read_csv(validation_run / "performance_legacy_targets.csv", low_memory=False)
    modern["model_id"] = modern["model_id"].map(canonical)
    legacy["model_id"] = legacy["model_id"].map(canonical)
    modern_resolution = modern.get(
        "target_resolution", pd.Series("", index=modern.index)
    ).astype(str)
    keep = modern_resolution.isin(STRICT_MODERN_RESOLUTIONS)
    audit = (
        pd.DataFrame(
            {
                "benchmark_generation": "modern",
                "target_resolution": modern_resolution,
                "included": keep,
            }
        )
        .groupby(["benchmark_generation", "target_resolution", "included"], dropna=False)
        .size()
        .rename("n_models")
        .reset_index()
    )
    modern = modern.loc[keep].copy()
    if modern["model_id"].duplicated().any() or legacy["model_id"].duplicated().any():
        raise ValueError("Strict target tables contain duplicate model IDs")
    return modern, legacy, audit


def _metadata_validation(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    lineage_validation: pd.DataFrame,
) -> pd.DataFrame:
    selected = predictions.loc[
        predictions["feature_block"].isin(
            ("alpha_profile_knn3", "coverage_aware_alpha_knn3")
        )
        & predictions["target"].isin(("modern_average", "legacy_average"))
    ].copy()
    target_metadata = pd.concat(
        [
            frame.loc[:, [
                column
                for column in ("model_id", "source_population", "target_architecture")
                if column in frame.columns
            ]]
            for frame in targets
        ],
        ignore_index=True,
        sort=False,
    ).drop_duplicates("model_id")
    validation_columns = [
        column
        for column in ("model_id", "component_root_model", "moderate_metadata_family")
        if column in lineage_validation.columns
    ]
    selected = selected.merge(target_metadata, on="model_id", how="left").merge(
        lineage_validation.loc[:, validation_columns], on="model_id", how="left"
    )
    rows: list[dict[str, object]] = []
    for dimension in (
        "source_population",
        "target_architecture",
        "component_root_model",
        "moderate_metadata_family",
    ):
        if dimension not in selected:
            continue
        for (generation, target, value), frame in selected.dropna(subset=[dimension]).groupby(
            ["generation", "target", dimension], sort=False
        ):
            rows.append(
                {
                    "generation": generation,
                    "target": target,
                    "validation_dimension": dimension,
                    "validation_value": value,
                    "n_models": len(frame),
                    "mae": float(np.mean(np.abs(frame["actual"] - frame["prediction"]))),
                    "evidence_role": "post_freeze_metadata_validation",
                }
            )
    return pd.DataFrame(rows)


def _population_ledger(
    development_source: Path,
    external_source: Path,
    profile_table: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    parts = []
    for name, path in (("phase2", development_source), ("ablation", external_source)):
        frame = pd.read_csv(path, usecols=["model_id"], low_memory=False)
        frame["model_id"] = frame["model_id"].map(canonical)
        frame = frame.loc[frame["model_id"].ne("")].drop_duplicates("model_id")
        frame["source_collection"] = name
        parts.append(frame)
    universe = pd.concat(parts, ignore_index=True)
    universe = (
        universe.groupby("model_id", as_index=False)["source_collection"]
        .agg(lambda values: "|".join(sorted(set(values))))
        .sort_values("model_id", ignore_index=True)
    )
    assignment_columns = [
        column
        for column in (
            "model_id",
            "predicted_cluster",
            "spectral_medoid_model",
            "similarity_to_cluster_mean",
            "nearest_neighbors",
            "nearest_similarity",
            "nearest_shared_roles",
            "assignment_status",
        )
        if column in assignments.columns
    ]
    ledger = universe.merge(profile_table, on="model_id", how="left").merge(
        assignments.loc[:, assignment_columns],
        on="model_id",
        how="left",
        validate="one_to_one",
    )
    ledger["performance_profile_available"] = ledger["profile_hash"].notna()
    ledger["lineage_profile_available"] = ledger["predicted_cluster"].notna()
    ledger["htsr_profile_available"] = ledger["lineage_profile_available"]
    strict_available = ledger["strict_seven_role_available"].astype("boolean")
    ledger["strict_seven_role_available"] = strict_available.fillna(False).astype(bool)
    return ledger


def main() -> None:
    args = parse_args()
    study_run = args.study_run.resolve()
    validation_run = args.validation_run.resolve()
    gallery_path = args.gallery.resolve()
    output_root = args.output_root.resolve()

    provenance = json.loads((study_run / "provenance.json").read_text(encoding="utf-8"))
    settings = provenance["settings"]
    seed = int(settings["random_seed"])
    folds = int(settings["outer_folds"])
    development_features = pd.read_csv(study_run / "feature_table.csv", usecols=["model_id"])
    external_features = pd.read_csv(study_run / "external_feature_table.csv", usecols=["model_id"])
    development_features["model_id"] = development_features["model_id"].map(canonical)
    external_features["model_id"] = external_features["model_id"].map(canonical)
    external_cache = Path(str(provenance["cache"]["external"]))

    strict_development_ids, strict_development_matrix, development_cache = build_rank_profile_cache(
        study_run / "alpha_trajectories.csv",
        development_features["model_id"],
        cache_root=output_root / "cache",
        source_digest=str(provenance["sources"]["development_stats"]["sha256"]),
        source_name="phase2_all_weight_only",
    )
    strict_external_ids, strict_external_matrix, external_profile_cache = build_rank_profile_cache(
        external_cache / "alpha_trajectories.csv",
        external_features["model_id"],
        cache_root=output_root / "cache",
        source_digest=str(provenance["sources"]["external_stats"]["sha256"]),
        source_name="ablation_all_weight_only",
    )
    strict_corpus = combine_profile_caches(
        (
            (strict_development_ids, strict_development_matrix, development_cache),
            (strict_external_ids, strict_external_matrix, external_profile_cache),
        )
    )

    partial_development = build_partial_role_cache(
        study_run / "alpha_trajectories.csv",
        development_features["model_id"],
        cache_root=output_root / "cache",
        source_digest=str(provenance["sources"]["development_stats"]["sha256"]),
        source_name="phase2_partial_roles",
    )
    partial_external = build_partial_role_cache(
        external_cache / "alpha_trajectories.csv",
        external_features["model_id"],
        cache_root=output_root / "cache",
        source_digest=str(provenance["sources"]["external_stats"]["sha256"]),
        source_name="ablation_partial_roles",
    )
    all_partial_roles = combine_partial_role_caches(
        (partial_development, partial_external)
    )
    masking_patterns, masking_thresholds, minimum_roles = masking_stability(
        all_partial_roles,
        seed=seed,
        minimum_retention=float(args.minimum_neighbor_retention),
    )
    performance_corpus = all_partial_roles.select(minimum_roles)
    lineage_corpus = select_reliable_role_patterns(
        all_partial_roles,
        masking_patterns,
        minimum_retention=float(args.minimum_neighbor_retention),
    )

    # Prediction phase: no lineage, family, architecture, or source annotations are loaded.
    split_table = partial_profile_split_table(
        performance_corpus, seed=seed, n_folds=folds
    )
    strict_split_table = profile_split_table(strict_corpus, seed=seed, n_folds=folds)
    strict_assignments, cluster_selection, cluster_internal = discover_clusters(
        strict_corpus, seed=seed
    )
    cluster_assignments = extend_strict_clusters(
        strict_corpus.model_ids, strict_assignments, lineage_corpus
    )
    performance_cluster_assignments = extend_strict_clusters(
        strict_corpus.model_ids, strict_assignments, performance_corpus
    )
    cluster_internal = cluster_internal.assign(
        method="strict_core_kmeans_with_mask_calibrated_extensions",
        n_models=len(lineage_corpus.model_ids),
        n_core_models=len(strict_corpus.model_ids),
        n_extension_models=len(lineage_corpus.model_ids) - len(strict_corpus.model_ids),
        metric_scope="strict_seven_role_core",
        extension_rule=(
            "observed_role_pattern retains full-profile nearest neighbor in top10"
        ),
        required_neighbor_retention=float(args.minimum_neighbor_retention),
    )
    cluster_selection = cluster_selection.assign(
        metric_scope="strict_seven_role_core"
    )
    cluster_split_table = cluster_holdout_split_table(
        split_table, performance_cluster_assignments, n_folds=folds
    )
    gallery = pd.read_csv(gallery_path)
    if list(gallery.columns) != ["model_id"]:
        raise ValueError("Gallery must contain exactly one model_id column")
    gallery["model_id"] = gallery["model_id"].map(canonical)
    gallery_predictions, gallery_audit = supplied_gallery_predictions(
        strict_corpus, gallery["model_id"].tolist()
    )
    lineage_freeze_sha = dataframe_sha256(cluster_assignments)
    gallery_freeze_sha = dataframe_sha256(gallery_predictions)

    # Score-supervised phase: splits were already frozen from weight profiles alone.
    modern_targets, legacy_targets, target_audit = _strict_targets(validation_run)
    modern_freeze, modern_predictions, modern_leakage = cross_validated_partial_performance_freeze(
        performance_corpus,
        split_table,
        modern_targets,
        MODERN_TARGETS,
        generation="modern",
        minimum_shared_roles=minimum_roles,
    )
    legacy_freeze, legacy_predictions, legacy_leakage = cross_validated_partial_performance_freeze(
        performance_corpus,
        split_table,
        legacy_targets,
        LEGACY_TARGETS,
        generation="legacy",
        minimum_shared_roles=minimum_roles,
    )
    performance_freeze = pd.concat([modern_freeze, legacy_freeze], ignore_index=True)
    performance_predictions = pd.concat(
        [modern_predictions, legacy_predictions], ignore_index=True
    )
    performance_metric_table = performance_metrics(
        performance_predictions, evaluation_scope="blind_profile_group_5fold"
    )
    strict_modern_predictions = cross_validated_performance_freeze(
        strict_corpus,
        strict_split_table,
        modern_targets,
        MODERN_TARGETS,
        generation="modern",
    )[1]
    strict_legacy_predictions = cross_validated_performance_freeze(
        strict_corpus,
        strict_split_table,
        legacy_targets,
        LEGACY_TARGETS,
        generation="legacy",
    )[1]
    strict_performance_metrics = performance_metrics(
        pd.concat(
            [strict_modern_predictions, strict_legacy_predictions],
            ignore_index=True,
        ),
        evaluation_scope="strict_seven_role_profile_group_5fold",
    )
    performance_freeze_sha = dataframe_sha256(performance_freeze)

    # Harder extrapolation check: every spectral cluster is absent from its training fold.
    modern_cluster_freeze, modern_cluster_predictions, modern_cluster_leakage = (
        cross_validated_partial_performance_freeze(
            performance_corpus,
            cluster_split_table,
            modern_targets,
            MODERN_TARGETS,
            generation="modern",
            minimum_shared_roles=minimum_roles,
        )
    )
    legacy_cluster_freeze, legacy_cluster_predictions, legacy_cluster_leakage = (
        cross_validated_partial_performance_freeze(
            performance_corpus,
            cluster_split_table,
            legacy_targets,
            LEGACY_TARGETS,
            generation="legacy",
            minimum_shared_roles=minimum_roles,
        )
    )
    cluster_performance_freeze = pd.concat(
        [modern_cluster_freeze, legacy_cluster_freeze], ignore_index=True
    )
    cluster_performance_predictions = pd.concat(
        [modern_cluster_predictions, legacy_cluster_predictions], ignore_index=True
    )
    cluster_performance_metrics = performance_metrics(
        cluster_performance_predictions,
        evaluation_scope="blind_spectral_cluster_holdout",
    )
    cluster_performance_freeze_sha = dataframe_sha256(cluster_performance_freeze)
    cluster_performance_leakage = summarize_leakage(
        pd.concat(
            [modern_cluster_leakage, legacy_cluster_leakage], ignore_index=True
        )
    )
    paper_performance = pd.concat(
        [
            paper_style_repeated_performance(
                strict_corpus,
                strict_split_table,
                modern_targets,
                MODERN_TARGETS,
                generation="modern",
                seed=seed,
                repeats=100,
            ),
            paper_style_repeated_performance(
                strict_corpus,
                strict_split_table,
                legacy_targets,
                LEGACY_TARGETS,
                generation="legacy",
                seed=seed,
                repeats=100,
            ),
        ],
        ignore_index=True,
    )
    leakage_rows = pd.concat([modern_leakage, legacy_leakage], ignore_index=True)
    leakage_summary = summarize_leakage(leakage_rows)

    # Validation phase: provenance and descriptive metadata are revealed only now.
    lineage_validation = pd.read_csv(validation_run / "population_ledger.csv", low_memory=False)
    lineage_validation["model_id"] = lineage_validation["model_id"].map(canonical)
    confirmed_mask = (
        lineage_validation.get("label_status", pd.Series("", index=lineage_validation.index))
        .astype(str)
        .eq("confirmed")
        & lineage_validation["audited_base_family"].notna()
    )
    cluster_validation = validate_clusters(
        cluster_assignments,
        lineage_validation,
        (
            ("manual_confirmed", "audited_base_family", confirmed_mask),
            ("metadata_graph_root", "component_root_model", None),
            ("metadata_root_consensus", "moderate_metadata_family", None),
        ),
    )
    strict_cluster_validation = validate_clusters(
        strict_assignments,
        lineage_validation,
        (
            ("manual_confirmed", "audited_base_family", confirmed_mask),
            ("metadata_graph_root", "component_root_model", None),
            ("metadata_root_consensus", "moderate_metadata_family", None),
        ),
    )
    lineage_validation_comparison = pd.concat(
        [
            strict_cluster_validation.assign(assignment_scope="strict_core"),
            cluster_validation.assign(
                assignment_scope="core_plus_mask_calibrated_extensions"
            ),
        ],
        ignore_index=True,
    )
    cluster_validation_assignments = cluster_assignments.merge(
        lineage_validation.loc[
            :,
            [
                "model_id",
                "audited_base_family",
                "label_status",
                "component_root_model",
                "moderate_metadata_family",
            ],
        ],
        on="model_id",
        how="left",
    )
    gallery_validation_predictions, gallery_metrics = validate_gallery_predictions(
        gallery_predictions,
        lineage_validation,
        gallery_audit,
        truth_column="component_root_model",
    )
    metadata_validation = _metadata_validation(
        performance_predictions,
        (modern_targets, legacy_targets),
        lineage_validation,
    )

    development_partial_ids = set(map(str, partial_development[0]))
    strict_profile_ids = set(map(str, strict_corpus.model_ids))
    profile_table = split_table.merge(
        pd.DataFrame(
            {
                "model_id": performance_corpus.model_ids,
                "profile_source": [
                    "phase2" if model_id in development_partial_ids else "ablation"
                    for model_id in performance_corpus.model_ids
                ],
                "strict_seven_role_available": [
                    model_id in strict_profile_ids
                    for model_id in performance_corpus.model_ids
                ],
            }
        ),
        on="model_id",
        validate="one_to_one",
    )
    population = _population_ledger(
        Path(str(provenance["sources"]["development_targets"]["path"])),
        Path(str(provenance["sources"]["external_targets"]["path"])),
        profile_table,
        cluster_assignments,
    )

    run_payload = {
        "method": "htsr_weight_only_generalized_v3",
        "seed": seed,
        "folds": folds,
        "minimum_neighbor_retention": float(args.minimum_neighbor_retention),
        "minimum_roles": int(minimum_roles),
        "profile": "independent_available_role_normalized_depth_rank32_stored_dks",
        "lineage": "strict_core_kmeans_with_mask_calibrated_extensions",
        "gallery": "operator_supplied_interpolated_spearman_dks",
        "performance": "partial_role_hash_grouped_5fold_knn3",
        "performance_stress_test": "coverage_aware_cluster_held_out_knn3",
        "study_run": study_run.name,
        "validation_run": validation_run.name,
        "gallery_sha256": sha256_file(gallery_path),
        "development_stats_sha256": provenance["sources"]["development_stats"]["sha256"],
        "external_stats_sha256": provenance["sources"]["external_stats"]["sha256"],
        "lineage_prediction_freeze_sha256": lineage_freeze_sha,
        "gallery_prediction_freeze_sha256": gallery_freeze_sha,
        "performance_prediction_freeze_sha256": performance_freeze_sha,
        "performance_cluster_holdout_freeze_sha256": cluster_performance_freeze_sha,
        "script_sha256": sha256_file(__file__),
        "implementation_sha256": {
            name: sha256_file(Path(__file__).resolve().parents[1] / relative)
            for name, relative in {
                "features": "spectral_research/study/features.py",
                "partial_roles": "spectral_research/study/partial_roles.py",
                "weight_only": "spectral_research/study/weight_only.py",
            }.items()
        },
    }
    run_id = "weight_only_" + hashlib.sha256(
        json.dumps(run_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    destination = output_root / "runs" / run_id
    if destination.exists():
        raise RuntimeError(f"Run already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=destination.parent))

    tables = {
        "population_ledger.csv": population,
        "lineage_cluster_assignments.csv": cluster_assignments,
        "lineage_strict_core_assignments.csv": strict_assignments,
        "lineage_cluster_selection.csv": cluster_selection,
        "lineage_internal_metrics.csv": cluster_internal,
        "lineage_validation_metrics.csv": cluster_validation,
        "lineage_validation_comparison.csv": lineage_validation_comparison,
        "lineage_validation_assignments.csv": cluster_validation_assignments,
        "lineage_supplied_gallery.csv": gallery_audit,
        "lineage_gallery_prediction_freeze.csv": gallery_predictions,
        "lineage_gallery_validation_predictions.csv": gallery_validation_predictions,
        "lineage_gallery_metrics.csv": gallery_metrics,
        "performance_spectral_groups.csv": profile_table,
        "performance_cluster_assignments.csv": performance_cluster_assignments,
        "performance_frozen_splits.csv": split_table,
        "performance_cluster_holdout_splits.csv": cluster_split_table,
        "performance_prediction_freeze.csv": performance_freeze,
        "performance_predictions.csv": performance_predictions,
        "performance_metrics.csv": performance_metric_table,
        "performance_strict_comparison_metrics.csv": strict_performance_metrics,
        "performance_cluster_holdout_prediction_freeze.csv": cluster_performance_freeze,
        "performance_cluster_holdout_predictions.csv": cluster_performance_predictions,
        "performance_cluster_holdout_metrics.csv": cluster_performance_metrics,
        "performance_cluster_holdout_leakage_audit.csv": cluster_performance_leakage,
        "performance_paper_summary.csv": paper_performance,
        "performance_leakage_rows.csv": leakage_rows,
        "performance_leakage_audit.csv": leakage_summary,
        "performance_metadata_validation.csv": metadata_validation,
        "performance_target_audit.csv": target_audit,
        "role_masking_patterns.csv": masking_patterns,
        "role_coverage_selection.csv": masking_thresholds,
    }
    for filename, frame in tables.items():
        frame.to_csv(temporary / filename, index=False)

    counts = {
        "source_models": len(population),
        "profiled_models": len(lineage_corpus.model_ids),
        "lineage_models": len(lineage_corpus.model_ids),
        "lineage_extension_models": len(lineage_corpus.model_ids)
        - len(strict_corpus.model_ids),
        "performance_profiled_models": len(performance_corpus.model_ids),
        "strict_seven_role_models": len(strict_corpus.model_ids),
        "minimum_usable_roles": int(minimum_roles),
        "unlabelled_clusters": int(cluster_internal.iloc[0]["n_clusters"]),
        "gallery_candidates_supplied": len(gallery_audit),
        "gallery_candidates_profiled": int(gallery_audit["profile_available"].sum()),
        "gallery_predictions_frozen": len(gallery_predictions),
        "gallery_validation_models": int(gallery_metrics.iloc[0]["n_scored_queries"]),
        "modern_performance_models": int(
            performance_metric_table.loc[
                performance_metric_table["generation"].eq("modern")
                & performance_metric_table["target"].eq("modern_average")
                & performance_metric_table["feature_block"].eq(
                    "coverage_aware_alpha_knn3"
                ),
                "n_models",
            ].iloc[0]
        ),
        "legacy_performance_models": int(
            performance_metric_table.loc[
                performance_metric_table["generation"].eq("legacy")
                & performance_metric_table["target"].eq("legacy_average")
                & performance_metric_table["feature_block"].eq(
                    "coverage_aware_alpha_knn3"
                ),
                "n_models",
            ].iloc[0]
        ),
    }
    metadata = {
        "run_id": run_id,
        "config": run_payload,
        "counts": counts,
        "method_boundary": {
            "unlabelled_lineage_discovery_inputs": [
                "available_role_alpha_summaries",
                "label_free_masking_calibration",
            ],
            "supplied_gallery_routing_inputs": [
                "seven_role_alpha_profiles",
                "operator_supplied_candidate_ids",
            ],
            "performance_prediction_inputs": [
                "available_role_alpha_summaries",
                "weight_only_role_coverage",
                "training_benchmark_scores",
            ],
            "excluded_before_prediction_freeze": [
                "lineage_labels",
                "family_names",
                "architecture_labels",
                "model_type",
                "parameter_count",
                "source_collection",
                "ancestry_graph",
            ],
            "metadata_role": "post_freeze_validation_only",
        },
        "performance_method": "coverage_aware_alpha_knn3",
        "caches": {
            "strict_seven_role": [str(path) for path in strict_corpus.cache_paths],
            "coverage_aware": [str(path) for path in performance_corpus.cache_paths],
        },
        "sources": {
            "study_run": str(study_run),
            "validation_run": str(validation_run),
            "candidate_gallery": str(gallery_path),
        },
    }
    (temporary / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    internal = cluster_internal.iloc[0]
    graph_validation = cluster_validation.loc[
        cluster_validation["label_source"].eq("metadata_graph_root")
    ].iloc[0]
    consensus_validation = cluster_validation.loc[
        cluster_validation["label_source"].eq("metadata_root_consensus")
    ].iloc[0]
    manual_validation = cluster_validation.loc[
        cluster_validation["label_source"].eq("manual_confirmed")
    ].iloc[0]
    gallery_result = gallery_metrics.iloc[0]
    coverage_selection = masking_thresholds.loc[masking_thresholds["selected"]].iloc[0]
    source_counts = profile_table.groupby("profile_source").size().to_dict()
    lineage_source_counts = {
        "phase2": int(
            np.count_nonzero(
                np.isin(lineage_corpus.model_ids, list(development_partial_ids))
            )
        )
    }
    lineage_source_counts["ablation"] = (
        len(lineage_corpus.model_ids) - lineage_source_counts["phase2"]
    )
    modern_result = performance_metric_table.loc[
        performance_metric_table["generation"].eq("modern")
        & performance_metric_table["target"].eq("modern_average")
        & performance_metric_table["feature_block"].eq("coverage_aware_alpha_knn3")
    ].iloc[0]
    legacy_result = performance_metric_table.loc[
        performance_metric_table["generation"].eq("legacy")
        & performance_metric_table["target"].eq("legacy_average")
        & performance_metric_table["feature_block"].eq("coverage_aware_alpha_knn3")
    ].iloc[0]
    strict_modern_result = strict_performance_metrics.loc[
        strict_performance_metrics["generation"].eq("modern")
        & strict_performance_metrics["target"].eq("modern_average")
        & strict_performance_metrics["feature_block"].eq("alpha_profile_knn3")
    ].iloc[0]
    strict_legacy_result = strict_performance_metrics.loc[
        strict_performance_metrics["generation"].eq("legacy")
        & strict_performance_metrics["target"].eq("legacy_average")
        & strict_performance_metrics["feature_block"].eq("alpha_profile_knn3")
    ].iloc[0]
    modern_stress = cluster_performance_metrics.loc[
        cluster_performance_metrics["generation"].eq("modern")
        & cluster_performance_metrics["target"].eq("modern_average")
        & cluster_performance_metrics["feature_block"].eq("coverage_aware_alpha_knn3")
    ].iloc[0]
    legacy_stress = cluster_performance_metrics.loc[
        cluster_performance_metrics["generation"].eq("legacy")
        & cluster_performance_metrics["target"].eq("legacy_average")
        & cluster_performance_metrics["feature_block"].eq("coverage_aware_alpha_knn3")
    ].iloc[0]
    modern_stress_baseline = cluster_performance_metrics.loc[
        cluster_performance_metrics["generation"].eq("modern")
        & cluster_performance_metrics["target"].eq("modern_average")
        & cluster_performance_metrics["feature_block"].eq("training_mean")
    ].iloc[0]
    legacy_stress_baseline = cluster_performance_metrics.loc[
        cluster_performance_metrics["generation"].eq("legacy")
        & cluster_performance_metrics["target"].eq("legacy_average")
        & cluster_performance_metrics["feature_block"].eq("training_mean")
    ].iloc[0]
    report = f"""# Weight-only HTSR study

Run: `{run_id}`

## Method boundary

- The paper-comparison test remains strict: it uses all seven role-wise alpha summaries and a fixed candidate-base list.
- Outside the paper comparison, each available role is summarized independently. Missing roles are not filled in.
- Unlabelled clustering is fitted once on the complete seven-role core. An incomplete pattern is admitted only if masking that same pattern on complete models keeps the original nearest model within its top ten at least {args.minimum_neighbor_retention:.0%} of the time. Admitted models are attached to the frozen core groups using only their observed roles.
- Performance uses the broader set with at least {minimum_roles} usable roles. That cutoff was selected without scores or metadata: the observed mix of missing-role patterns retains the full-summary nearest model at {coverage_selection['weighted_full_nearest_retained_in_top10']:.1%}, above the required {coverage_selection['required_retention']:.0%}.
- A performance neighbor must share at least {minimum_roles} roles with the held-out model. Exact duplicate summaries stay in one test part. Lineage, architecture, model type, scale, and source do not affect eligibility, parts, neighbors, or predictions.
- Metadata is attached only after predictions are hashed and frozen.

## Results

- The strict core has {len(strict_corpus.model_ids):,} models. Reliable masking adds {len(lineage_corpus.model_ids) - len(strict_corpus.model_ids):,}, giving {len(lineage_corpus.model_ids):,} lineage assignments: {lineage_source_counts['phase2']:,} from Phase 2 and {lineage_source_counts['ablation']:,} from ablation.
- The frozen core forms {int(internal['n_clusters'])} unlabelled clusters. On that core, cosine silhouette is {internal['silhouette_cosine']:.3f}; mean seed stability adjusted Rand index is {internal['mean_seed_stability_ari']:.3f}. These internal metrics do not score the later attachments.
- Performance covers {len(performance_corpus.model_ids):,} models with at least {minimum_roles} usable roles: {int(source_counts.get('phase2', 0)):,} from Phase 2 and {int(source_counts.get('ablation', 0)):,} from ablation.
- After labels are revealed, adjusted Rand index is {manual_validation['adjusted_rand']:.3f} on {int(manual_validation['n_models']):,} manually confirmed models and {consensus_validation['adjusted_rand']:.3f} on {int(consensus_validation['n_models']):,} models with one of 11 consensus family labels. Agreement with {int(graph_validation['n_families']):,} exact graph roots is {graph_validation['adjusted_rand']:.3f}; that label set is much finer than the {int(internal['n_clusters'])} discovered groups.
- The fixed candidate-base test contains {int(gallery_result['n_profiled_roots'])} usable seven-role summaries from {int(gallery_result['n_supplied_roots'])} supplied roots. It scores {int(gallery_result['n_scored_queries']):,} labelled queries at {gallery_result['micro_top1']:.3f} overall top-1 and {gallery_result['root_macro_top1']:.3f} root-balanced top-1.
- Five-part summary-grouped performance prediction reaches MAE {modern_result['model_weighted_mae']:.3f} and Spearman {modern_result['spearman']:.3f} on {int(modern_result['n_models']):,} modern-score models.
- The legacy-score result is MAE {legacy_result['model_weighted_mae']:.3f} and Spearman {legacy_result['spearman']:.3f} on {int(legacy_result['n_models']):,} models.
- Under the same five-part design, the strict seven-role subset has modern MAE {strict_modern_result['model_weighted_mae']:.3f} on {int(strict_modern_result['n_models']):,} models and legacy MAE {strict_legacy_result['model_weighted_mae']:.3f} on {int(strict_legacy_result['n_models']):,}. The broader method trades a small amount of interpolation accuracy for wider coverage.
- When complete weight-derived spectral clusters are held out, modern MAE is {modern_stress['model_weighted_mae']:.3f} versus {modern_stress_baseline['model_weighted_mae']:.3f} for the training-mean baseline. Legacy MAE is {legacy_stress['model_weighted_mae']:.3f} versus {legacy_stress_baseline['model_weighted_mae']:.3f} for the baseline. This is the harder extrapolation check.

## Interpretation

- Unlabelled clusters recover weight-derived groups. They do not identify historical parents or edge direction.
- The named cluster center is the observed core model nearest its cluster mean, not necessarily the historical base checkpoint.
- Fixed-candidate matching answers which supplied base has the closest seven-role summary. It cannot discover a missing base.
- Stored alpha values use the WeightWatcher DKS fit. The paper uses fixed-tail Hill alpha, so the candidate-base result follows its evaluation design but is not an exact replication.
- The summary-grouped performance result measures interpolation among models with known training scores. The complete-cluster holdout reports how much accuracy remains outside represented regions.
- In the harder cluster holdout, modern Spearman is {modern_stress['spearman']:.3f} and legacy Spearman is {legacy_stress['spearman']:.3f}. These values, together with the mean-baseline errors above, show how much signal remains outside represented weight groups. Predictions should carry neighbor similarity and abstain when no training model shares {minimum_roles} roles.
- Benchmark scores are supervised targets, not model metadata. Architecture, lineage, parameter count, model type, and source collection remain unavailable to the predictor.
"""
    (temporary / "report.md").write_text(report, encoding="utf-8")

    os.rename(temporary, destination)
    write_artifact_manifest(destination, run_id=run_id)
    print(json.dumps({"run": str(destination), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
