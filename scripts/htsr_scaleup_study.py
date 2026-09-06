from __future__ import annotations

"""Scale a retained HTSR study without mutating its source artifacts."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spectral_research.study.artifacts import write_artifact_manifest
from spectral_research.study.contracts import (
    build_single_source_components,
    load_graph_contract,
)
from spectral_research.study.contracts import canonical_model_id
from spectral_research.study.features import CANONICAL_ROLES, confirmatory_schema_population
from spectral_research.study.lineage import (
    _signature_matrix,
    label_independent_cluster_count,
    paper_style_alpha_similarity,
)
from spectral_research.study.performance import (
    MODERN_TASKS,
    build_modern_targets,
    evaluate_external_transfer,
    evaluate_grouped_ridge,
    group_weighted_mae,
    holm_adjust,
    paired_component_bootstrap,
    paired_component_signflip_pvalue,
)
from spectral_research.study.scaleup import (
    balanced_branch_folds,
    consensus_roots,
    corroborated_edges,
    first_root_branch,
    map_overlap_to_phase2,
    matched_family_recall,
    metadata_edges,
    name_consensus_families,
    propagated_component_labels,
    reconcile_phase2_legacy_targets,
    reconcile_phase2_modern_targets,
    sha256_file,
)


RUN = Path()
SCALE_ROOT = Path()
SEED = 17
GRID_SIZE = 32
GROUP_K = 5
PAPER_K = 3
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
BROAD_COVERAGE = 0.8
EXTERNAL_EVIDENCE_ROLE = "reused_descriptive"
OUTER_FOLDS = 5
INNER_FOLDS = 4
BOOTSTRAP_REPLICATES = 2_000
SIGNFLIP_REPLICATES = 10_000
MINIMUM_COMPONENTS = 10
PAPER_REPEATS = 100
KMEANS_N_INIT = 20
SILHOUETTE_SAMPLE = 2_000
LEGACY_TASKS = ("ARC", "HellaSwag", "MMLU", "TruthfulQA", "Winogrande", "GSM8K")
MODERN_TARGETS = ("modern_average", *MODERN_TASKS)
LEGACY_TARGETS = ("legacy_average", *LEGACY_TASKS)


def canonical(value: object) -> str:
    return canonical_model_id(value) or ""


def load_provenance() -> dict[str, object]:
    return json.loads((RUN / "provenance.json").read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scale a retained HTSR study across phase-2 and ablation labels."
    )
    parser.add_argument("--study-run", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("research_runs/htsr_scaleup")
    )
    return parser.parse_args()


def alpha_embedding_cache(
    trajectory_path: Path,
    requested_ids: set[str],
    *,
    source_digest: str,
    source_name: str,
) -> tuple[list[str], np.ndarray, Path]:
    model_ids = sorted(requested_ids)
    payload = {
        "method": "rolewise_interp32_rank_cosine_v1",
        "grid_size": GRID_SIZE,
        "source_digest": source_digest,
        "model_ids": model_ids,
    }
    key = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    cache = SCALE_ROOT / "cache" / f"{source_name}_{key}.npz"
    if cache.is_file():
        loaded = np.load(cache, allow_pickle=False)
        cached_ids = loaded["model_ids"].astype(str).tolist()
        matrix = loaded["embeddings"].astype(np.float32, copy=False)
        assert cached_ids == sorted(cached_ids)
        assert matrix.shape == (len(cached_ids), len(CANONICAL_ROLES) * GRID_SIZE)
        return cached_ids, matrix, cache

    selected: list[pd.DataFrame] = []
    for chunk in pd.read_csv(trajectory_path, chunksize=300_000, low_memory=False):
        kept = chunk.loc[chunk["model_id"].astype(str).isin(requested_ids)]
        if not kept.empty:
            selected.append(kept)
    trajectories = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    grid = np.linspace(0.0, 1.0, GRID_SIZE)
    kept_ids: list[str] = []
    embeddings: list[np.ndarray] = []
    for model_id, frame in trajectories.groupby("model_id", sort=True):
        signature = _signature_matrix(frame, name=str(model_id))
        if signature is None:
            continue
        depths, values = signature
        span = float(depths[-1] - depths[0])
        if not np.isfinite(span) or span <= 0.0:
            continue
        normalized_depth = (depths - depths[0]) / span
        role_vectors: list[np.ndarray] = []
        valid = True
        for role_index in range(len(CANONICAL_ROLES)):
            interpolated = np.interp(grid, normalized_depth, values[:, role_index])
            ranks = rankdata(interpolated, method="average").astype(float)
            ranks -= ranks.mean()
            norm = float(np.linalg.norm(ranks))
            if not np.isfinite(norm) or norm <= np.finfo(float).eps:
                valid = False
                break
            role_vectors.append(ranks / norm)
        if valid:
            kept_ids.append(str(model_id))
            embeddings.append(np.concatenate(role_vectors).astype(np.float32))
    matrix = np.vstack(embeddings) if embeddings else np.empty((0, 224), dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, model_ids=np.asarray(kept_ids), embeddings=matrix)
    return kept_ids, matrix, cache


def paper_root_similarity_cache(
    trajectory_paths: tuple[Path, ...],
    query_ids: set[str],
    root_ids: set[str],
    *,
    source_digests: tuple[str, ...],
) -> tuple[list[str], list[str], np.ndarray, Path]:
    ordered_queries = sorted(query_ids)
    ordered_roots = sorted(root_ids)
    payload = {
        "method": "paper_eq2_spearman_dp_stored_dks_v1",
        "source_digests": source_digests,
        "query_ids": ordered_queries,
        "root_ids": ordered_roots,
    }
    key = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    cache = SCALE_ROOT / "cache" / f"lineage_closed_root_paper_{key}.npz"
    if cache.is_file():
        loaded = np.load(cache, allow_pickle=False)
        cached_queries = loaded["query_ids"].astype(str).tolist()
        cached_roots = loaded["root_ids"].astype(str).tolist()
        similarity = loaded["similarity"].astype(np.float32, copy=False)
        assert cached_queries == ordered_queries
        assert cached_roots == ordered_roots
        assert similarity.shape == (len(cached_queries), len(cached_roots))
        return cached_queries, cached_roots, similarity, cache

    wanted = query_ids | root_ids
    selected: list[pd.DataFrame] = []
    for trajectory_path in trajectory_paths:
        for chunk in pd.read_csv(trajectory_path, chunksize=300_000, low_memory=False):
            kept = chunk.loc[chunk["model_id"].astype(str).isin(wanted)]
            if not kept.empty:
                selected.append(kept)
    trajectories = pd.concat(selected, ignore_index=True)
    if trajectories.duplicated(["model_id", "role", "depth"]).any():
        raise ValueError("closed-root trajectory sources overlap")
    groups = {
        str(model_id): frame
        for model_id, frame in trajectories.groupby("model_id", sort=False)
    }
    missing = wanted.difference(groups)
    if missing:
        raise ValueError(f"closed-root trajectories missing requested models: {sorted(missing)[:3]}")

    similarity = np.full(
        (len(ordered_queries), len(ordered_roots)), np.nan, dtype=np.float32
    )
    for query_index, query_id in enumerate(ordered_queries):
        for root_index, root_id in enumerate(ordered_roots):
            similarity[query_index, root_index] = paper_style_alpha_similarity(
                groups[query_id], groups[root_id]
            )
        if (query_index + 1) % 250 == 0:
            print(
                f"closed-root paper similarity: {query_index + 1}/{len(ordered_queries)}",
                flush=True,
            )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        query_ids=np.asarray(ordered_queries),
        root_ids=np.asarray(ordered_roots),
        similarity=similarity,
    )
    return ordered_queries, ordered_roots, similarity, cache


def embedding_lookup(
    dev_ids: list[str], dev_matrix: np.ndarray, ext_ids: list[str], ext_matrix: np.ndarray
) -> dict[str, np.ndarray]:
    overlap = set(dev_ids) & set(ext_ids)
    assert not overlap, f"development/external trajectory overlap: {sorted(overlap)[:3]}"
    return {
        **{model_id: dev_matrix[index] for index, model_id in enumerate(dev_ids)},
        **{model_id: ext_matrix[index] for index, model_id in enumerate(ext_ids)},
    }


def knn_predict(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    train_values: np.ndarray,
    *,
    k: int,
    train_groups: np.ndarray | None = None,
    distinct_groups: bool,
) -> tuple[np.ndarray, np.ndarray]:
    similarity = np.clip(
        (test_embeddings @ train_embeddings.T) / float(len(CANONICAL_ROLES)),
        -1.0,
        1.0,
    )
    neighbor_positions = np.empty((len(test_embeddings), k), dtype=int)
    if distinct_groups:
        assert train_groups is not None
        order = np.argsort(-similarity, axis=1, kind="stable")
        for row_index, candidates in enumerate(order):
            chosen: list[int] = []
            observed: set[str] = set()
            for candidate in candidates:
                group = str(train_groups[candidate])
                if group in observed:
                    continue
                observed.add(group)
                chosen.append(int(candidate))
                if len(chosen) == k:
                    break
            if len(chosen) < k:
                raise ValueError("fewer distinct training ancestry components than k")
            neighbor_positions[row_index] = chosen
    else:
        partial = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
        partial_scores = np.take_along_axis(similarity, partial, axis=1)
        local_order = np.argsort(-partial_scores, axis=1, kind="stable")
        neighbor_positions = np.take_along_axis(partial, local_order, axis=1)
    neighbor_scores = np.take_along_axis(similarity, neighbor_positions, axis=1)
    weights = 1.0 / (np.maximum(1.0 - neighbor_scores, 0.0) + 1e-6)
    values = np.asarray(train_values, dtype=float)[neighbor_positions]
    if values.ndim == 3:
        prediction = (weights[:, :, None] * values).sum(axis=1) / weights.sum(
            axis=1
        )[:, None]
    else:
        prediction = (weights * values).sum(axis=1) / weights.sum(axis=1)
    return prediction, neighbor_positions


def group_ready(frame: pd.DataFrame) -> pd.DataFrame:
    edge_count = pd.to_numeric(frame["component_edge_count"], errors="coerce").fillna(0)
    eligible = (
        frame["ancestry_component"].notna()
        & frame["component_provenance"].eq("graph")
        & edge_count.ge(1)
    )
    return frame.loc[eligible].copy().reset_index(drop=True)


def alpha_on_ridge_folds(
    population: pd.DataFrame,
    ridge_predictions: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    *,
    target: str,
    method: str = "alpha_profile_k5_component_distinct",
) -> pd.DataFrame:
    baseline = ridge_predictions.loc[
        ridge_predictions["target"].eq(target)
        & ridge_predictions["feature_block"].eq("M+S+H")
    ].copy()
    ids = set(baseline["model_id"].astype(str))
    frame = population.loc[population["model_id"].astype(str).isin(ids)].copy()
    frame = frame.sort_values("model_id", kind="stable").reset_index(drop=True)
    assert set(frame["model_id"].astype(str)) == ids
    matrix = np.vstack([embeddings[model_id] for model_id in frame["model_id"]])
    values = pd.to_numeric(frame[target], errors="coerce").to_numpy(float)
    groups = frame["ancestry_component"].astype(str).to_numpy()
    id_to_index = {model_id: index for index, model_id in enumerate(frame["model_id"].astype(str))}
    rows: list[dict[str, object]] = []
    for fold_id in sorted(baseline["outer_fold"].unique()):
        test_ids = baseline.loc[baseline["outer_fold"].eq(fold_id), "model_id"].astype(str).tolist()
        test_indices = np.asarray([id_to_index[model_id] for model_id in test_ids], dtype=int)
        test_set = set(test_indices)
        train_indices = np.asarray([index for index in range(len(frame)) if index not in test_set], dtype=int)
        predicted, _ = knn_predict(
            matrix[train_indices], matrix[test_indices], values[train_indices],
            k=GROUP_K, train_groups=groups[train_indices], distinct_groups=True,
        )
        assert set(groups[train_indices]).isdisjoint(set(groups[test_indices]))
        for local_index, global_index in enumerate(test_indices):
            rows.append(
                {
                    "model_id": str(frame.at[global_index, "model_id"]),
                    "ancestry_component": groups[global_index],
                    "target": target,
                    "feature_block": method,
                    "outer_fold": int(fold_id),
                    "actual": float(values[global_index]),
                    "prediction": float(predicted[local_index]),
                }
            )
    result = pd.DataFrame(rows)
    assert len(result) == len(frame)
    return result


def paper_random_splits(
    frame: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    targets: tuple[str, ...],
    *,
    repeats: int = PAPER_REPEATS,
) -> pd.DataFrame:
    population = frame.loc[frame["model_id"].isin(embeddings)].copy()
    population = population.loc[
        population.loc[:, targets].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    ].sort_values("model_id", kind="stable").reset_index(drop=True)
    matrix = np.vstack([embeddings[model_id] for model_id in population["model_id"]])
    values = population.loc[:, targets].to_numpy(float)
    groups = population["ancestry_component"].fillna("").astype(str).to_numpy()
    groups = np.asarray(
        [group if group else f"unknown::{model_id}" for group, model_id in zip(groups, population["model_id"])],
        dtype=object,
    )
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    train_size = len(population) // 2
    for repeat in range(repeats):
        order = rng.permutation(len(population))
        train, test = order[:train_size], order[train_size:]
        predicted, neighbors = knn_predict(
            matrix[train], matrix[test], values[train], k=PAPER_K,
            train_groups=None, distinct_groups=False,
        )
        same_component = groups[train][neighbors] == groups[test, None]
        for target_index, target in enumerate(targets):
            tau = kendalltau(values[test, target_index], predicted[:, target_index])
            rows.append(
                {
                    "repeat": repeat,
                    "target": target,
                    "n_models": len(population),
                    "mae": float(np.mean(np.abs(values[test, target_index] - predicted[:, target_index]))),
                    "kendall_tau": float(tau.statistic),
                    "kendall_p": float(tau.pvalue),
                    "same_component_neighbor_fraction": float(np.mean(same_component.any(axis=1))),
                }
            )
    raw = pd.DataFrame(rows)
    return (
        raw.groupby("target", sort=False)
        .agg(
            n_models=("n_models", "first"),
            mean_mae=("mae", "mean"),
            sd_mae=("mae", "std"),
            mean_kendall=("kendall_tau", "mean"),
            significant_trials=("kendall_p", lambda values: int((values < 0.05).sum())),
            test_models_with_same_component_neighbor=("same_component_neighbor_fraction", "mean"),
        )
        .reset_index()
    )


def metric_rows(predictions: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    group_columns = ["target", "feature_block"]
    if "external_slice" in predictions.columns:
        group_columns.insert(0, "external_slice")
    rows: list[dict[str, object]] = []
    for key, frame in predictions.groupby(group_columns, sort=False, dropna=False):
        keys = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(group_columns, keys))
        actual = frame["actual"].to_numpy(float)
        predicted = frame["prediction"].to_numpy(float)
        groups = frame["ancestry_component"].astype(str).to_numpy()
        rows.append(
            {
                "evaluation_scope": scope,
                **identity,
                "n_models": len(frame),
                "n_components": frame["ancestry_component"].nunique(),
                "component_weighted_mae": group_weighted_mae(actual, predicted, groups),
                "model_weighted_mae": float(np.mean(np.abs(actual - predicted))),
                "spearman": float(spearmanr(actual, predicted).statistic),
                "kendall_tau": float(kendalltau(actual, predicted).statistic),
            }
        )
    return pd.DataFrame(rows)


def increment_rows(
    predictions: pd.DataFrame,
    *,
    baseline: str,
    candidates: tuple[str, ...],
    secondary_targets: tuple[str, ...],
    scope: str,
) -> pd.DataFrame:
    slice_values = (
        predictions["external_slice"].dropna().astype(str).unique().tolist()
        if "external_slice" in predictions.columns
        else [""]
    )
    rows: list[dict[str, object]] = []
    for slice_value in slice_values:
        sliced = (
            predictions.loc[predictions["external_slice"].astype(str).eq(slice_value)]
            if slice_value
            else predictions
        )
        for candidate in candidates:
            raw_p: dict[str, float] = {}
            candidate_rows: list[dict[str, object]] = []
            for target in sliced["target"].astype(str).unique():
                frame = sliced.loc[
                    sliced["target"].eq(target)
                    & sliced["feature_block"].isin((baseline, candidate))
                ]
                if set(frame["feature_block"]) != {baseline, candidate}:
                    continue
                candidate_ids = set(
                    frame.loc[frame["feature_block"].eq(candidate), "model_id"].astype(str)
                )
                frame = frame.loc[frame["model_id"].astype(str).isin(candidate_ids)]
                interval = paired_component_bootstrap(
                    frame, baseline_block=baseline, htsr_block=candidate,
                    n_bootstrap=BOOTSTRAP_REPLICATES,
                    random_state=SEED,
                    minimum_components=MINIMUM_COMPONENTS,
                )
                p_value = paired_component_signflip_pvalue(
                    frame, baseline_block=baseline, htsr_block=candidate,
                    n_replicates=SIGNFLIP_REPLICATES,
                    random_state=SEED,
                    minimum_components=MINIMUM_COMPONENTS,
                )
                raw_p[target] = p_value
                candidate_rows.append(
                    {
                        "evaluation_scope": scope,
                        "external_slice": slice_value,
                        "target": target,
                        "baseline": baseline,
                        "candidate": candidate,
                        **interval,
                        "p_value_raw": p_value,
                    }
                )
            finite_secondary = {
                target: raw_p[target]
                for target in secondary_targets
                if target in raw_p and np.isfinite(raw_p[target])
            }
            adjusted = holm_adjust(finite_secondary) if finite_secondary else {}
            for row in candidate_rows:
                row["p_value_holm"] = adjusted.get(str(row["target"]), np.nan)
            rows.extend(candidate_rows)
    return pd.DataFrame(rows)


def broad_lineage(
    phase2: pd.DataFrame, features: pd.DataFrame, manifest: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = [column for column in features if column.startswith(("M__", "S__", "H__"))]
    numeric = features.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    coverage = np.mean(np.isfinite(numeric.to_numpy(float)), axis=1)
    eligible_mask = coverage >= BROAD_COVERAGE
    eligible = features.loc[eligible_mask].copy().reset_index(drop=True)
    numeric = numeric.loc[eligible_mask].reset_index(drop=True)
    active = numeric.columns[np.isfinite(numeric.to_numpy(float)).any(axis=0)]
    matrix = numeric.loc[:, active].to_numpy(float)
    medians = np.nanmedian(matrix, axis=0)
    missing_rows, missing_columns = np.where(~np.isfinite(matrix))
    matrix[missing_rows, missing_columns] = medians[missing_columns]
    standardized = StandardScaler().fit_transform(matrix)
    n_clusters = label_independent_cluster_count(len(eligible), rule="sqrt_half")
    clusters = KMeans(
        n_clusters=n_clusters,
        random_state=SEED,
        n_init=KMEANS_N_INIT,
    ).fit_predict(standardized)
    broad = pd.DataFrame(
        {
            "model_id": eligible["model_id"].astype(str),
            "broad_cluster": clusters.astype(int),
            "feature_coverage": coverage[eligible_mask],
        }
    )

    labels = manifest.loc[
        manifest["label_status"].eq("confirmed") & manifest["audited_base_family"].notna(),
        ["model_id", "audited_base_family"],
    ]
    scored = broad.merge(labels, on="model_id", how="inner", validate="one_to_one")
    metric = pd.DataFrame(
        [
            {
                "method": "broad_kmeans_population_sensitivity",
                "n_clusters": n_clusters,
                "n_clustered": len(broad),
                "n_labeled": len(scored),
                "ari": float(adjusted_rand_score(scored["audited_base_family"], scored["broad_cluster"])),
                "nmi": float(normalized_mutual_info_score(scored["audited_base_family"], scored["broad_cluster"])),
                "silhouette_sample_2000": float(
                    silhouette_score(
                        standardized,
                        clusters,
                        sample_size=min(SILHOUETTE_SAMPLE, len(standardized)),
                        random_state=SEED,
                    )
                ),
            }
        ]
    )

    routes = route_spectral_families(features, manifest, broad)
    strict = pd.read_csv(RUN / "lineage_cluster_assignments.csv").rename(
        columns={"cluster": "strict_cluster"}
    )
    shared = strict.merge(broad, on="model_id", how="inner", validate="one_to_one")
    strict_scored = scored.loc[scored["model_id"].isin(set(strict["model_id"].astype(str)))]
    metric["shared_strict_models"] = len(shared)
    metric["strict_broad_partition_ari"] = float(
        adjusted_rand_score(shared["strict_cluster"], shared["broad_cluster"])
    )
    metric["broad_ari_on_strict_labeled"] = float(
        adjusted_rand_score(
            strict_scored["audited_base_family"], strict_scored["broad_cluster"]
        )
    )
    metric["family_routes"] = int((~routes["family_abstained"]).sum())
    metric["family_coverage_abstentions"] = int(routes["family_abstained"].sum())
    graph_columns = [
        "model_id", "ancestry_component", "component_provenance", "component_edge_count",
        "component_root_model", "audited_base_family", "label_status",
    ]
    census = phase2.loc[:, ["model_id"]].copy()
    census = census.merge(
        pd.DataFrame({"model_id": features["model_id"].astype(str), "htsr_artifact_available": True}),
        on="model_id", how="left", validate="one_to_one",
    )
    census = census.merge(broad, on="model_id", how="left", validate="one_to_one")
    census = census.merge(strict, on="model_id", how="left", validate="one_to_one")
    census = census.merge(routes, on="model_id", how="left", validate="one_to_one")
    census = census.merge(manifest.loc[:, graph_columns], on="model_id", how="left", validate="one_to_one")
    census["htsr_artifact_available"] = census["htsr_artifact_available"].eq(True)
    census["lineage_evidence_status"] = np.select(
        [
            census["strict_cluster"].notna(),
            census["broad_cluster"].notna(),
            census["htsr_artifact_available"],
        ],
        ["strict_qc_clustered", "nonconfirmatory_broad_clustered", "insufficient_coverage"],
        default="missing_htsr_artifact",
    )
    assert len(census) == len(phase2) and census["model_id"].is_unique
    return census, broad, metric


def route_spectral_families(
    features: pd.DataFrame, manifest: pd.DataFrame, broad: pd.DataFrame
) -> pd.DataFrame:
    feature_index = features.set_index("model_id", verify_integrity=True)
    query_ids = broad["model_id"].astype(str).tolist()
    labels = manifest.loc[
        manifest["label_status"].eq("confirmed") & manifest["audited_base_family"].notna(),
        ["model_id", "audited_base_family"],
    ].copy()
    labels = labels.loc[labels["model_id"].isin(feature_index.index)]
    columns = [column for column in features if column.startswith("S__")]
    gallery = feature_index.loc[labels["model_id"], columns].apply(pd.to_numeric, errors="coerce")
    active = gallery.columns[np.isfinite(gallery.to_numpy(float)).any(axis=0)]
    gallery_matrix = gallery.loc[:, active].to_numpy(float)
    center = np.nanmedian(gallery_matrix, axis=0)
    q25, q75 = np.nanquantile(gallery_matrix, [0.25, 0.75], axis=0)
    scale = q75 - q25
    scale = np.where(np.isfinite(scale) & (scale > np.finfo(float).eps), scale, 1.0)
    gallery_z = (gallery_matrix - center) / scale
    label_values = labels.set_index("model_id").loc[gallery.index, "audited_base_family"].astype(str)
    families = sorted(label_values.unique())
    family_centers = np.full((len(families), gallery_z.shape[1]), np.nan, dtype=float)
    for family_index, family in enumerate(families):
        family_values = gallery_z[label_values.to_numpy() == family]
        for column_index in range(family_values.shape[1]):
            finite = family_values[:, column_index]
            finite = finite[np.isfinite(finite)]
            if finite.size:
                family_centers[family_index, column_index] = float(np.median(finite))
    query = feature_index.loc[query_ids, active].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    query_z = (query - center) / scale
    scores = np.full((len(query), len(families)), np.nan, dtype=float)
    coverages = np.zeros_like(scores)
    for family_index, family_center in enumerate(family_centers):
        shared = np.isfinite(query_z) & np.isfinite(family_center)[None, :]
        coverage = shared.mean(axis=1)
        squared = np.where(shared, np.square(query_z - family_center), 0.0)
        distance = np.sqrt(squared.sum(axis=1) / np.maximum(shared.sum(axis=1), 1))
        valid = coverage >= BROAD_COVERAGE
        scores[valid, family_index] = 1.0 / (1.0 + distance[valid])
        coverages[:, family_index] = coverage
    safe = np.where(np.isfinite(scores), scores, -np.inf)
    best = np.argmax(safe, axis=1)
    sorted_scores = np.sort(safe, axis=1)
    top = safe[np.arange(len(safe)), best]
    second = sorted_scores[:, -2] if len(families) >= 2 else np.full(len(safe), -np.inf)
    abstained = ~np.isfinite(top) | np.isneginf(top)
    margin = np.full(len(query_ids), np.nan, dtype=float)
    margin[~abstained] = top[~abstained] - second[~abstained]
    return pd.DataFrame(
        {
            "model_id": query_ids,
            "predicted_family": ["" if abstained[i] else families[best[i]] for i in range(len(query_ids))],
            "family_score": np.where(abstained, np.nan, top),
            "family_margin": margin,
            "family_feature_coverage": np.where(abstained, np.nan, coverages[np.arange(len(query_ids)), best]),
            "family_abstained": abstained,
            "family_prediction_scope": "descriptive_full_gallery_spectral_shape",
        }
    )


def _graph_metadata_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_id": frame["model_id"].map(canonical),
            "source_model": frame.get("source_model", pd.Series("", index=frame.index)).map(canonical),
            "base_model_relation": frame.get(
                "base_model_relation", pd.Series("unknown", index=frame.index)
            ).fillna("unknown"),
            "graph_architecture": frame.get(
                "Architecture", pd.Series("unknown", index=frame.index)
            ).fillna("unknown"),
        }
    )


def build_scaled_graph(
    phase2: pd.DataFrame,
    ablation: pd.DataFrame,
    labels: pd.DataFrame,
    graph_path: Path,
) -> dict[str, object]:
    base_edges, base_nodes, graph_metadata = load_graph_contract(graph_path)
    augmented, accepted_edges, source_audit = metadata_edges(
        (("phase2", phase2), ("ablation", ablation)),
        base_edges,
        base_nodes,
    )
    augmented_nodes = set(base_nodes) | set(accepted_edges.get("parent_id", [])) | set(
        accepted_edges.get("child_id", [])
    )
    combined_metadata = pd.concat(
        [graph_metadata, _graph_metadata_frame(phase2), _graph_metadata_frame(ablation)],
        ignore_index=True,
    )
    selected_ids = list(
        dict.fromkeys([*phase2["model_id"].astype(str), *ablation["model_id"].astype(str)])
    )
    components = build_single_source_components(
        selected_ids,
        augmented,
        graph_nodes=augmented_nodes,
        graph_metadata=combined_metadata,
    )
    components["component_graph_fingerprint"] = hashlib.sha256(
        "\n".join(
            f"{parent}\0{child}"
            for parent, child in augmented.itertuples(index=False, name=None)
        ).encode("utf-8")
    ).hexdigest()

    confirmed = labels.loc[
        labels["label_status"].eq("confirmed")
        & labels["audited_base_family"].notna()
    ].copy()
    conservative = propagated_component_labels(
        phase2["model_id"].astype(str).tolist(),
        corroborated_edges(base_edges, graph_metadata),
        confirmed,
    ).rename(
        columns={
            "metadata_family": "conservative_metadata_family",
            "metadata_family_conflict": "conservative_family_conflict",
        }
    )
    atlas_clean = propagated_component_labels(
        phase2["model_id"].astype(str).tolist(),
        base_edges,
        confirmed,
        graph_nodes=base_nodes,
        graph_metadata=graph_metadata,
    ).rename(
        columns={
            "metadata_family": "atlas_clean_metadata_family",
            "metadata_family_conflict": "atlas_clean_family_conflict",
        }
    )
    roots = consensus_roots(augmented, augmented_nodes)
    moderate = name_consensus_families(
        roots,
        confirmed,
        phase2["model_id"].astype(str).tolist(),
    ).rename(
        columns={
            "metadata_family": "moderate_metadata_family",
            "metadata_family_conflict": "moderate_family_conflict",
        }
    )
    phase2_components = components.loc[
        components["model_id"].isin(set(phase2["model_id"].astype(str)))
    ].copy()
    branches = first_root_branch(phase2_components, augmented)
    return {
        "base_edges": base_edges,
        "base_nodes": base_nodes,
        "augmented_edges": augmented,
        "accepted_edges": accepted_edges,
        "source_audit": source_audit,
        "components": components,
        "branches": branches,
        "conservative": conservative,
        "atlas_clean": atlas_clean,
        "moderate": moderate,
        "graph_metadata": graph_metadata,
    }


def _modern_table(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = raw.rename(
        columns={
            "target_model_sha": "model_sha",
            "target_eval_name": "eval_name",
            "target_architecture": "architecture",
        }
    )
    targets, exclusions = build_modern_targets(prepared)
    provenance_columns = [
        "model_id",
        "target_source",
        "target_source_row",
        "target_resolution",
        "source_population",
    ]
    available = [column for column in provenance_columns if column in raw.columns]
    targets = targets.merge(
        raw.loc[:, available].drop_duplicates("model_id"),
        on="model_id",
        how="left",
        validate="one_to_one",
    )
    return targets, exclusions


def _legacy_table(
    raw: pd.DataFrame,
    *,
    source_population: str,
    target_source: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = raw.copy()
    working["model_id"] = working["model_id"].map(canonical)
    values = working.loc[:, LEGACY_TASKS].apply(pd.to_numeric, errors="coerce")
    valid = values.notna().all(axis=1) & values.ge(0.0).all(axis=1) & values.le(100.0).all(axis=1)
    rejected = working.loc[~valid, ["model_id"]].copy()
    rejected["reason"] = "incomplete_or_out_of_range_legacy_target"
    working = working.loc[valid].copy()
    working.loc[:, LEGACY_TASKS] = values.loc[valid]
    if working["model_id"].duplicated().any():
        raise ValueError(f"duplicate complete legacy targets in {target_source}")
    working["legacy_average"] = working.loc[:, LEGACY_TASKS].mean(axis=1)
    if "target_model_sha" not in working:
        working["target_model_sha"] = working.get(
            "model_sha", working.get("Model sha", pd.Series("", index=working.index))
        ).fillna("")
    if "target_eval_name" not in working:
        working["target_eval_name"] = working.get(
            "eval_name", pd.Series("", index=working.index)
        ).fillna("")
    if "target_architecture" not in working:
        working["target_architecture"] = working.get(
            "Architecture", pd.Series("", index=working.index)
        ).fillna("")
    working["target_source"] = working.get(
        "target_source", pd.Series(target_source, index=working.index)
    ).fillna(target_source)
    working["source_population"] = source_population
    working["benchmark_generation"] = "legacy"
    columns = [
        "model_id",
        *LEGACY_TASKS,
        "legacy_average",
        "target_model_sha",
        "target_eval_name",
        "target_architecture",
        "target_source",
        "source_population",
        "benchmark_generation",
    ]
    return working.loc[:, columns].sort_values("model_id", ignore_index=True), rejected


def reconcile_targets(
    phase2: pd.DataFrame,
    ablation: pd.DataFrame,
    overlap: pd.DataFrame,
    retained_modern: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    mapped = map_overlap_to_phase2(phase2, overlap)
    recovered_modern_raw, modern_audit = reconcile_phase2_modern_targets(
        phase2,
        mapped,
        retained_modern,
        tasks=MODERN_TASKS,
    )
    phase2_modern, modern_exclusions = _modern_table(recovered_modern_raw)
    ablation_modern_raw = ablation.copy()
    ablation_modern_raw["target_source"] = "ablation_selected"
    ablation_modern_raw["target_source_row"] = np.arange(len(ablation_modern_raw))
    ablation_modern_raw["target_resolution"] = "selected_source_row"
    ablation_modern_raw["source_population"] = "ablation"
    ablation_modern, ablation_modern_exclusions = _modern_table(ablation_modern_raw)
    modern = pd.concat([phase2_modern, ablation_modern], ignore_index=True, sort=False)
    if modern["model_id"].duplicated().any():
        raise ValueError("reconciled modern targets overlap between populations")

    recovered_legacy_raw, legacy_audit = reconcile_phase2_legacy_targets(
        mapped, tasks=LEGACY_TASKS
    )
    phase2_legacy, legacy_phase_exclusions = _legacy_table(
        recovered_legacy_raw,
        source_population="phase2",
        target_source="ablation_overlap_audit",
    )
    ablation_legacy, ablation_legacy_exclusions = _legacy_table(
        ablation,
        source_population="ablation",
        target_source="ablation_selected",
    )
    legacy = pd.concat([phase2_legacy, ablation_legacy], ignore_index=True, sort=False)
    if legacy["model_id"].duplicated().any():
        raise ValueError("reconciled legacy targets overlap between populations")

    audit = pd.concat([modern_audit, legacy_audit], ignore_index=True, sort=False)
    exclusions = pd.concat(
        [
            modern_exclusions.assign(source="phase2_modern"),
            ablation_modern_exclusions.assign(source="ablation_modern"),
            legacy_phase_exclusions.assign(source="phase2_legacy"),
            ablation_legacy_exclusions.assign(source="ablation_legacy"),
        ],
        ignore_index=True,
        sort=False,
    )
    return {
        "mapped_overlap": mapped,
        "modern": modern.sort_values("model_id", ignore_index=True),
        "legacy": legacy.sort_values("model_id", ignore_index=True),
        "audit": audit,
        "exclusions": exclusions,
    }


def _evaluation_groups(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    components = sorted(result["ancestry_component"].astype(str).unique())
    parent = {component: component for component in components}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    identity_groups: list[pd.Series] = []
    sha = result.get("target_model_sha", pd.Series("", index=result.index)).fillna("").astype(str).str.strip().str.casefold()
    identity_groups.extend(group for _, group in result.loc[sha.ne("")].groupby(sha[sha.ne("")]))
    model_key = result["model_id"].astype(str).str.casefold()
    identity_groups.extend(group for _, group in result.groupby(model_key))
    for group in identity_groups:
        values = sorted(set(group["ancestry_component"].astype(str)))
        for value in values[1:]:
            union(values[0], value)
    members: dict[str, list[str]] = defaultdict(list)
    for component in components:
        members[find(component)].append(component)
    identifiers = {
        member: "split_" + hashlib.sha256("\0".join(sorted(values)).encode("utf-8")).hexdigest()[:16]
        for values in members.values()
        for member in values
    }
    result["graph_component"] = result["ancestry_component"].astype(str)
    result["ancestry_component"] = result["graph_component"].map(identifiers)
    return result


def performance_population(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    components: pd.DataFrame,
    *,
    min_role_coverage: float,
) -> pd.DataFrame:
    strict, _ = confirmatory_schema_population(features, min_role_coverage)
    graph_columns = [
        "model_id",
        "ancestry_component",
        "component_provenance",
        "component_edge_count",
        "component_root_model",
        "component_graph_fingerprint",
    ]
    population = (
        strict.merge(targets, on="model_id", how="inner", validate="one_to_one")
        .merge(
            components.loc[:, graph_columns],
            on="model_id",
            how="left",
            validate="one_to_one",
        )
        .sort_values("model_id", kind="stable", ignore_index=True)
    )
    population = group_ready(population)
    return _evaluation_groups(population)


def grouped_performance(
    population: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    targets: tuple[str, ...],
    *,
    scope: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paired = population.loc[population["model_id"].isin(embeddings)].copy().reset_index(drop=True)
    prediction_frames: list[pd.DataFrame] = []
    fit_audit: list[dict[str, object]] = []
    for target in targets:
        ridge, _, audit = evaluate_grouped_ridge(
            paired,
            target=target,
            outer_splits=OUTER_FOLDS,
            inner_splits=INNER_FOLDS,
            ridge_alphas=RIDGE_ALPHAS,
            random_state=SEED,
        )
        alpha = alpha_on_ridge_folds(paired, ridge, embeddings, target=target)
        prediction_frames.extend([ridge, alpha])
        fit_audit.extend({"target": target, **row} for row in audit)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = metric_rows(predictions, scope=scope)
    increments = increment_rows(
        predictions,
        baseline="M+S+H",
        candidates=("alpha_profile_k5_component_distinct",),
        secondary_targets=targets[1:],
        scope=scope,
    )
    return predictions, metrics, increments, pd.DataFrame(fit_audit)


def source_heldout_modern_full(
    population: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    development = population.loc[population["source_population"].eq("phase2")].copy()
    external = population.loc[population["source_population"].eq("ablation")].copy()
    prediction_frames: list[pd.DataFrame] = []
    for target in MODERN_TARGETS:
        ridge, _, _ = evaluate_external_transfer(
            development,
            external,
            target=target,
            development_component_universe=development,
            inner_splits=INNER_FOLDS,
            ridge_alphas=RIDGE_ALPHAS,
            random_state=SEED,
        )
        prediction_frames.append(ridge)
        baseline = ridge.loc[ridge["feature_block"].eq("M+S+H")].set_index("model_id")
        train_matrix = np.vstack([embeddings[model_id] for model_id in development["model_id"]])
        test_matrix = np.vstack([embeddings[model_id] for model_id in external["model_id"]])
        predicted, _ = knn_predict(
            train_matrix,
            test_matrix,
            development[target].to_numpy(float),
            k=GROUP_K,
            train_groups=development["ancestry_component"].astype(str).to_numpy(),
            distinct_groups=True,
        )
        rows = []
        for position, row in external.reset_index(drop=True).iterrows():
            model_id = str(row["model_id"])
            rows.append(
                {
                    "model_id": model_id,
                    "ancestry_component": str(row["ancestry_component"]),
                    "target": target,
                    "feature_block": "alpha_profile_k5_strict_train",
                    "actual": float(row[target]),
                    "prediction": float(predicted[position]),
                    "external_slice": str(baseline.at[model_id, "external_slice"]),
                }
            )
        prediction_frames.append(pd.DataFrame(rows))
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = metric_rows(predictions, scope="source_heldout_modern_full")
    increments = increment_rows(
        predictions,
        baseline="M+S+H",
        candidates=("alpha_profile_k5_strict_train",),
        secondary_targets=MODERN_TASKS,
        scope="source_heldout_modern_full",
    )
    increments["inferential"] = False
    increments["inference_reason"] = EXTERNAL_EVIDENCE_ROLE
    return predictions, metrics, increments


def _centroid_predictions(
    population: pd.DataFrame,
    columns: list[str],
    *,
    method: str,
) -> pd.DataFrame:
    matrix = population.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    actual = population["component_root_model"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    for fold in sorted(population["branch_fold"].unique()):
        train = population["branch_fold"].to_numpy() != fold
        test = ~train
        active = np.isfinite(matrix[train]).any(axis=0)
        train_matrix = matrix[train][:, active]
        query = matrix[test][:, active]
        center = np.nanmedian(train_matrix, axis=0)
        q25, q75 = np.nanquantile(train_matrix, [0.25, 0.75], axis=0)
        scale = q75 - q25
        scale = np.where(np.isfinite(scale) & (scale > np.finfo(float).eps), scale, 1.0)
        train_z = (train_matrix - center) / scale
        query_z = (query - center) / scale
        train_labels = actual[train]
        classes = sorted(set(train_labels))
        centroids = np.vstack(
            [np.nanmedian(train_z[train_labels == label], axis=0) for label in classes]
        )
        distances = np.full((len(query_z), len(classes)), np.inf)
        for index, centroid in enumerate(centroids):
            shared = np.isfinite(query_z) & np.isfinite(centroid)[None, :]
            coverage = shared.mean(axis=1)
            squared = np.where(shared, np.square(query_z - centroid), 0.0)
            distance = np.sqrt(squared.sum(axis=1) / np.maximum(shared.sum(axis=1), 1))
            distance[coverage < BROAD_COVERAGE] = np.inf
            distances[:, index] = distance
        order = np.argsort(distances, axis=1, kind="stable")
        global_indices = np.flatnonzero(test)
        class_values = np.asarray(classes, dtype=object)
        for local_index, global_index in enumerate(global_indices):
            if not np.isfinite(distances[local_index]).any():
                rows.append(
                    {
                        "model_id": str(population.iloc[global_index]["model_id"]),
                        "actual_root": actual[global_index],
                        "predicted_root": "",
                        "root_branch_model": str(population.iloc[global_index]["root_branch_model"]),
                        "branch_fold": int(fold),
                        "method": method,
                        "scorable": False,
                        "correct": False,
                        "actual_rank": pd.NA,
                        "top3_correct": False,
                        "score": np.nan,
                        "margin": np.nan,
                    }
                )
                continue
            ranked = class_values[order[local_index]]
            rank = int(np.flatnonzero(ranked == actual[global_index])[0]) + 1
            best = int(order[local_index, 0])
            second = int(order[local_index, 1]) if len(classes) > 1 else best
            rows.append(
                {
                    "model_id": str(population.iloc[global_index]["model_id"]),
                    "actual_root": actual[global_index],
                    "predicted_root": str(classes[best]),
                    "root_branch_model": str(population.iloc[global_index]["root_branch_model"]),
                    "branch_fold": int(fold),
                    "method": method,
                    "scorable": True,
                    "correct": bool(rank == 1),
                    "actual_rank": rank,
                    "top3_correct": bool(rank <= 3),
                    "score": float(-distances[local_index, best]),
                    "margin": float(distances[local_index, second] - distances[local_index, best]),
                }
            )
    return pd.DataFrame(rows)


def _profile_centroid_predictions(
    population: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
) -> pd.DataFrame:
    population = population.reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for fold in sorted(population["branch_fold"].unique()):
        train = population.loc[
            population["branch_fold"].ne(fold)
            & population["model_id"].isin(embeddings)
        ]
        test = population.loc[population["branch_fold"].eq(fold)]
        train_labels = train["component_root_model"].astype(str).to_numpy()
        classes = sorted(set(train_labels))
        train_matrix = np.vstack([embeddings[model_id] for model_id in train["model_id"]])
        centers = []
        for label in classes:
            center = train_matrix[train_labels == label].mean(axis=0).reshape(
                len(CANONICAL_ROLES), -1
            )
            norms = np.linalg.norm(center, axis=1, keepdims=True)
            centers.append((center / np.maximum(norms, np.finfo(float).eps)).reshape(-1))
        centers_array = np.vstack(centers)
        class_values = np.asarray(classes, dtype=object)
        for row in test.itertuples(index=False):
            model_id = str(row.model_id)
            actual_root = str(row.component_root_model)
            reason = ""
            if model_id not in embeddings:
                reason = "query_profile_unavailable"
            elif actual_root not in classes:
                reason = "root_training_profile_unavailable"
            if reason:
                rows.append(
                    {
                        "model_id": model_id,
                        "actual_root": actual_root,
                        "predicted_root": "",
                        "root_branch_model": str(row.root_branch_model),
                        "branch_fold": int(fold),
                        "method": "alpha_profile",
                        "scorable": False,
                        "unscorable_reason": reason,
                        "correct": False,
                        "actual_rank": pd.NA,
                        "top3_correct": False,
                        "score": np.nan,
                        "margin": np.nan,
                    }
                )
                continue
            similarity = embeddings[model_id] @ centers_array.T / float(
                len(CANONICAL_ROLES)
            )
            order = np.argsort(-similarity, kind="stable")
            ranked = class_values[order]
            rank = int(np.flatnonzero(ranked == actual_root)[0]) + 1
            best = int(order[0])
            second = int(order[1]) if len(classes) > 1 else best
            rows.append(
                {
                    "model_id": model_id,
                    "actual_root": actual_root,
                    "predicted_root": str(classes[best]),
                    "root_branch_model": str(row.root_branch_model),
                    "branch_fold": int(fold),
                    "method": "alpha_profile",
                    "scorable": True,
                    "unscorable_reason": "",
                    "correct": bool(rank == 1),
                    "actual_rank": rank,
                    "top3_correct": bool(rank <= 3),
                    "score": float(similarity[best]),
                    "margin": float(similarity[best] - similarity[second]),
                }
            )
    return pd.DataFrame(rows)


def branch_metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for method, eligible in predictions.groupby("method", sort=False):
        frame = eligible.loc[eligible["scorable"].eq(True)].copy()
        by_root = frame.assign(reciprocal_rank=1.0 / frame["actual_rank"]).groupby("actual_root").agg(
            n=("model_id", "size"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
            top3=("top3_correct", "mean"),
            mrr=("reciprocal_rank", "mean"),
        )
        roots = np.arange(len(by_root))
        macro_samples = np.empty(BOOTSTRAP_REPLICATES)
        micro_samples = np.empty(BOOTSTRAP_REPLICATES)
        for replicate in range(BOOTSTRAP_REPLICATES):
            sample = rng.choice(roots, size=len(roots), replace=True)
            chosen = by_root.iloc[sample]
            macro_samples[replicate] = chosen["accuracy"].mean()
            micro_samples[replicate] = chosen["correct"].sum() / chosen["n"].sum()
        rows.append(
            {
                "method": method,
                "n_eligible": int(len(eligible)),
                "n_models": int(len(frame)),
                "coverage": float(len(frame) / len(eligible)),
                "n_roots": int(frame["actual_root"].nunique()),
                "n_branches": int(frame["root_branch_model"].nunique()),
                "micro_top1": float(frame["correct"].mean()),
                "micro_top1_ci_low": float(np.quantile(micro_samples, 0.025)),
                "micro_top1_ci_high": float(np.quantile(micro_samples, 0.975)),
                "root_macro_top1": float(by_root["accuracy"].mean()),
                "root_macro_top1_ci_low": float(np.quantile(macro_samples, 0.025)),
                "root_macro_top1_ci_high": float(np.quantile(macro_samples, 0.975)),
                "top3_accuracy": float(frame["top3_correct"].mean()),
                "root_macro_top3": float(by_root["top3"].mean()),
                "mean_reciprocal_rank": float((1.0 / frame["actual_rank"]).mean()),
                "root_macro_mrr": float(by_root["mrr"].mean()),
                "label_source": "atlas_plus_represented_strong_local_sources",
                "evidence_role": "metadata_graph_agreement",
            }
        )
    return pd.DataFrame(rows)


def branch_lineage_population(
    branches: pd.DataFrame,
    features: pd.DataFrame,
    strict_assignments: pd.DataFrame,
) -> pd.DataFrame:
    population = branches.loc[
        branches["model_id"].isin(set(strict_assignments["model_id"].astype(str)))
        & branches["ancestry_component"].notna()
        & branches["root_branch_model"].notna()
    ].copy()
    root_counts = population.groupby("component_root_model").agg(
        n_models=("model_id", "size"),
        n_branches=("root_branch_model", "nunique"),
    )
    eligible_roots = set(
        root_counts.loc[
            root_counts["n_models"].ge(5) & root_counts["n_branches"].ge(2)
        ].index
    )
    population = population.loc[
        population["component_root_model"].isin(eligible_roots)
    ].merge(features, on="model_id", how="inner", validate="one_to_one")
    population["branch_fold"] = balanced_branch_folds(population, n_splits=OUTER_FOLDS)
    return population


def evaluate_branch_lineage(
    population: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    m_columns = [column for column in population if column.startswith("M__")]
    s_columns = [column for column in population if column.startswith("S__")]
    h_columns = [column for column in population if column.startswith("H__")]
    methods = {
        "M": m_columns,
        "S": s_columns,
        "H": h_columns,
        "alpha_summary": [column for column in h_columns if "__alpha__" in column],
        "S+H": [*s_columns, *h_columns],
        "M+S+H": [*m_columns, *s_columns, *h_columns],
    }
    predictions = pd.concat(
        [
            *[
                _centroid_predictions(population, columns, method=method)
                for method, columns in methods.items()
            ],
            _profile_centroid_predictions(population, embeddings),
        ],
        ignore_index=True,
    )
    fold_audit = (
        population.groupby("branch_fold")
        .agg(
            n_models=("model_id", "size"),
            n_roots=("component_root_model", "nunique"),
            n_branches=("root_branch_model", "nunique"),
        )
        .reset_index()
    )
    return predictions, branch_metric_rows(predictions), fold_audit


def evaluate_exact_root_lineage(
    population: pd.DataFrame,
    development_features: pd.DataFrame,
    external_features: pd.DataFrame,
    development_profile_ids: set[str],
    external_profile_ids: set[str],
    embeddings: dict[str, np.ndarray],
    paper_query_ids: list[str],
    paper_root_ids: list[str],
    paper_similarity: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    roots = sorted(population["component_root_model"].astype(str).unique())
    development_feature_ids = set(development_features["model_id"].astype(str))
    external_feature_ids = set(external_features["model_id"].astype(str))
    candidate_roots = [root for root in roots if root in embeddings]
    candidate_values = np.asarray(candidate_roots, dtype=object)
    root_matrix = (
        np.vstack([embeddings[root] for root in candidate_roots])
        if candidate_roots
        else np.empty((0, len(CANONICAL_ROLES) * GRID_SIZE), dtype=np.float32)
    )

    rows: list[dict[str, object]] = []
    for row in population.itertuples(index=False):
        model_id = str(row.model_id)
        actual_root = str(row.component_root_model)
        reason = ""
        if actual_root not in candidate_roots:
            reason = "exact_root_profile_unavailable"
        elif model_id not in embeddings:
            reason = "query_profile_unavailable"

        result: dict[str, object] = {
            "model_id": model_id,
            "actual_root": actual_root,
            "predicted_root": "",
            "root_branch_model": str(row.root_branch_model),
            "branch_fold": int(row.branch_fold),
            "method": "exact_root_alpha_profile",
            "endpoint": "closed_set_exact_root_gallery",
            "candidate_count": len(candidate_roots),
            "scorable": not reason,
            "unscorable_reason": reason,
            "correct": False,
            "actual_rank": pd.NA,
            "top3_correct": False,
            "score": np.nan,
            "margin": np.nan,
            "estimator_provenance": (
                "rolewise_interp32_rank_cosine_stored_dks_not_fixed_hill"
            ),
        }
        if not reason:
            similarity = embeddings[model_id] @ root_matrix.T / float(len(CANONICAL_ROLES))
            order = np.argsort(-similarity, kind="stable")
            ranked = candidate_values[order]
            rank = int(np.flatnonzero(ranked == actual_root)[0]) + 1
            best = int(order[0])
            second = int(order[1]) if len(order) > 1 else best
            result.update(
                {
                    "predicted_root": str(candidate_values[best]),
                    "correct": bool(rank == 1),
                    "actual_rank": rank,
                    "top3_correct": bool(rank <= 3),
                    "score": float(similarity[best]),
                    "margin": float(similarity[best] - similarity[second]),
                }
            )
        rows.append(result)

    paper_query_index = {model_id: index for index, model_id in enumerate(paper_query_ids)}
    paper_root_index = {model_id: index for index, model_id in enumerate(paper_root_ids)}
    for row in population.itertuples(index=False):
        model_id = str(row.model_id)
        actual_root = str(row.component_root_model)
        reason = ""
        if actual_root not in paper_root_index:
            reason = "exact_root_profile_unavailable"
        elif model_id not in paper_query_index:
            reason = "query_profile_unavailable"
        scores = (
            paper_similarity[paper_query_index[model_id]]
            if not reason
            else np.asarray([], dtype=float)
        )
        if not reason and not np.isfinite(scores[paper_root_index[actual_root]]):
            reason = "true_root_similarity_unavailable"
        result = {
            "model_id": model_id,
            "actual_root": actual_root,
            "predicted_root": "",
            "root_branch_model": str(row.root_branch_model),
            "branch_fold": int(row.branch_fold),
            "method": "exact_root_paper_spearman_dks",
            "endpoint": "closed_set_exact_root_gallery",
            "candidate_count": len(paper_root_ids),
            "scorable": not reason,
            "unscorable_reason": reason,
            "correct": False,
            "actual_rank": pd.NA,
            "top3_correct": False,
            "score": np.nan,
            "margin": np.nan,
            "estimator_provenance": (
                "paper_eq2_spearman_dp_depth_alignment_stored_dks_not_fixed_hill"
            ),
        }
        if not reason:
            finite_order = np.asarray(
                sorted(
                    np.flatnonzero(np.isfinite(scores)),
                    key=lambda index: (-float(scores[index]), paper_root_ids[index]),
                ),
                dtype=int,
            )
            ranked = np.asarray(paper_root_ids, dtype=object)[finite_order]
            rank = int(np.flatnonzero(ranked == actual_root)[0]) + 1
            best = int(finite_order[0])
            second = int(finite_order[1]) if len(finite_order) > 1 else best
            result.update(
                {
                    "predicted_root": paper_root_ids[best],
                    "correct": bool(rank == 1),
                    "actual_rank": rank,
                    "top3_correct": bool(rank <= 3),
                    "score": float(scores[best]),
                    "margin": float(scores[best] - scores[second]),
                }
            )
        rows.append(result)

    predictions = pd.DataFrame(rows)
    metrics = branch_metric_rows(predictions)
    metrics["endpoint"] = "closed_set_exact_root_gallery"
    metrics["n_population_roots"] = len(roots)
    metrics["n_candidate_roots"] = len(candidate_roots)
    metrics["root_profile_coverage"] = len(candidate_roots) / len(roots)
    metrics["estimator_provenance"] = metrics["method"].map(
        {
            "exact_root_alpha_profile": (
                "rolewise_interp32_rank_cosine_stored_dks_not_fixed_hill"
            ),
            "exact_root_paper_spearman_dks": (
                "paper_eq2_spearman_dp_depth_alignment_stored_dks_not_fixed_hill"
            ),
        }
    )

    root_audit = []
    for root, frame in population.groupby("component_root_model", sort=True):
        root = str(root)
        if root in development_profile_ids:
            profile_source = "phase2"
        elif root in external_profile_ids:
            profile_source = "ablation"
        else:
            profile_source = ""
        feature_available = (
            root in development_feature_ids or root in external_feature_ids
        )
        root_audit.append(
            {
                "root_model": root,
                "n_query_models": int(len(frame)),
                "n_query_branches": int(frame["root_branch_model"].nunique()),
                "phase2_feature_row_available": root in development_feature_ids,
                "ablation_feature_row_available": root in external_feature_ids,
                "root_alpha_profile_valid": root in embeddings,
                "profile_source": profile_source,
                "status": (
                    "valid_exact_root_profile"
                    if root in embeddings
                    else "invalid_alpha_profile"
                    if feature_available
                    else "missing_htsr_artifact"
                ),
            }
        )
    return predictions, metrics, pd.DataFrame(root_audit)


def evaluate_matched_branch_lineage(
    population: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    candidate_roots: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matched = population.loc[
        population["component_root_model"].astype(str).isin(candidate_roots)
        & population["model_id"].isin(embeddings)
    ].copy()
    predictions = _profile_centroid_predictions(matched, embeddings)
    predictions["method"] = "matched_branch_alpha_profile"
    predictions["endpoint"] = "branch_held_out_matched_closed_set_population"
    metrics = branch_metric_rows(predictions)
    metrics["endpoint"] = "branch_held_out_matched_closed_set_population"
    metrics["n_population_roots"] = len(candidate_roots)
    metrics["n_candidate_roots"] = len(candidate_roots)
    metrics["root_profile_coverage"] = 1.0
    metrics["estimator_provenance"] = (
        "rolewise_interp32_rank_cosine_stored_dks_not_fixed_hill"
    )
    return predictions, metrics


def branch_profile_duplicate_audit(
    population: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
) -> pd.DataFrame:
    profiled = population.loc[population["model_id"].isin(embeddings)].copy()
    profiled["profile_hash"] = [
        hashlib.sha256(embeddings[model_id].tobytes()).hexdigest()
        for model_id in profiled["model_id"]
    ]
    groups = profiled.groupby("profile_hash").agg(
        n_models=("model_id", "size"),
        n_branches=("root_branch_model", "nunique"),
        n_folds=("branch_fold", "nunique"),
    )
    duplicates = groups.loc[groups["n_models"].gt(1)]
    cross_branch = set(duplicates.loc[duplicates["n_branches"].gt(1)].index)
    cross_fold = set(duplicates.loc[duplicates["n_folds"].gt(1)].index)
    return pd.DataFrame(
        [
            {
                "n_profiled_models": int(len(profiled)),
                "n_duplicate_profile_groups": int(len(duplicates)),
                "n_models_in_duplicate_profile_groups": int(
                    duplicates["n_models"].sum()
                ),
                "n_cross_branch_duplicate_groups": int(len(cross_branch)),
                "n_models_in_cross_branch_duplicate_groups": int(
                    profiled["profile_hash"].isin(cross_branch).sum()
                ),
                "n_cross_fold_duplicate_groups": int(len(cross_fold)),
                "n_models_in_cross_fold_duplicate_groups": int(
                    profiled["profile_hash"].isin(cross_fold).sum()
                ),
                "interpretation": (
                    "identical_derived_profiles_not_verified_identical_checkpoints"
                ),
            }
        ]
    )


def lineage_metric_table(
    census: pd.DataFrame,
    strict_assignments: pd.DataFrame,
    broad_assignments: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    label_sources = (
        ("manual_confirmed", "audited_base_family"),
        ("conservative_corroborated", "conservative_metadata_family"),
        ("atlas_clean_single_root", "atlas_clean_metadata_family"),
        ("moderate_root_consensus", "moderate_metadata_family"),
    )
    for population_name, assignments, cluster_column in (
        ("strict", strict_assignments, "cluster"),
        ("broad", broad_assignments.rename(columns={"broad_cluster": "cluster"}), "cluster"),
    ):
        for label_source, label_column in label_sources:
            scored = assignments.loc[:, ["model_id", cluster_column]].merge(
                census.loc[:, ["model_id", label_column]],
                on="model_id",
                how="inner",
                validate="one_to_one",
            ).dropna(subset=[label_column])
            if scored.empty:
                continue
            rows.append(
                {
                    "population": population_name,
                    "label_source": label_source,
                    "n_models": int(len(scored)),
                    "n_families": int(scored[label_column].nunique()),
                    "n_clusters": int(scored[cluster_column].nunique()),
                    "ari": float(adjusted_rand_score(scored[label_column], scored[cluster_column])),
                    "nmi": float(normalized_mutual_info_score(scored[label_column], scored[cluster_column])),
                    "matched_family_recall": matched_family_recall(
                        scored[label_column], scored[cluster_column]
                    ),
                    "evidence_role": (
                        "manual_audit" if label_source == "manual_confirmed" else "metadata_derived"
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    global RUN, SCALE_ROOT, SEED, RIDGE_ALPHAS, BROAD_COVERAGE
    global EXTERNAL_EVIDENCE_ROLE, OUTER_FOLDS, INNER_FOLDS
    global BOOTSTRAP_REPLICATES, SIGNFLIP_REPLICATES, MINIMUM_COMPONENTS

    args = parse_args()
    RUN = args.study_run.resolve()
    SCALE_ROOT = args.output_root.resolve()
    provenance = load_provenance()
    settings = dict(provenance["settings"])
    SEED = int(settings["random_seed"])
    RIDGE_ALPHAS = tuple(float(value) for value in settings["ridge_alphas"])
    BROAD_COVERAGE = float(settings["min_feature_coverage"])
    EXTERNAL_EVIDENCE_ROLE = str(settings["external_evidence_role"])
    OUTER_FOLDS = int(settings["outer_folds"])
    INNER_FOLDS = int(settings["inner_folds"])
    BOOTSTRAP_REPLICATES = int(settings["bootstrap_replicates"])
    SIGNFLIP_REPLICATES = int(settings["signflip_replicates"])
    MINIMUM_COMPONENTS = int(settings["minimum_components"])

    phase2_path = Path(str(provenance["sources"]["development_targets"]["path"]))
    ablation_path = Path(str(provenance["sources"]["external_targets"]["path"]))
    overlap_path = ablation_path.parent / "ablation_overlap_audit.csv"
    graph_path = Path(str(provenance["sources"]["graph"]["path"]))
    labels_path = Path(str(provenance["sources"]["labels"]["path"]))

    manifest = pd.read_csv(RUN / "manifest.csv", low_memory=False)
    development_features = pd.read_csv(RUN / "feature_table.csv", low_memory=False)
    external_features = pd.read_csv(RUN / "external_feature_table.csv", low_memory=False)
    retained_modern = pd.read_csv(RUN / "development_targets.csv", low_memory=False)
    strict_assignments = pd.read_csv(RUN / "lineage_cluster_assignments.csv")
    phase2 = pd.read_csv(phase2_path, low_memory=False)
    ablation = pd.read_csv(ablation_path, low_memory=False)
    overlap = pd.read_csv(overlap_path, low_memory=False)
    labels = pd.read_csv(labels_path, low_memory=False)
    for frame in (phase2, ablation, development_features, external_features):
        frame["model_id"] = frame["model_id"].map(canonical)
    if phase2["model_id"].duplicated().any() or ablation["model_id"].duplicated().any():
        raise ValueError("source populations require unique model IDs")

    targets = reconcile_targets(phase2, ablation, overlap, retained_modern)
    graph = build_scaled_graph(phase2, ablation, labels, graph_path)
    components = graph["components"]
    branch_population = branch_lineage_population(
        graph["branches"], development_features, strict_assignments
    )

    requested_development = set(
        pd.concat(
            [targets["modern"], targets["legacy"]], ignore_index=True, sort=False
        ).loc[lambda frame: frame["source_population"].eq("phase2"), "model_id"]
    )
    lineage_root_ids = set(branch_population["component_root_model"].astype(str))
    requested_development.update(branch_population["model_id"].astype(str))
    requested_development.update(lineage_root_ids)
    requested_external = set(
        pd.concat(
            [targets["modern"], targets["legacy"]], ignore_index=True, sort=False
        ).loc[lambda frame: frame["source_population"].eq("ablation"), "model_id"]
    )
    development_ids, development_embedding, development_cache = alpha_embedding_cache(
        RUN / "alpha_trajectories.csv",
        requested_development,
        source_digest=str(provenance["sources"]["development_stats"]["sha256"]),
        source_name="phase2_full_targets_and_lineage_roots",
    )
    requested_external.update(lineage_root_ids.difference(development_ids))
    external_cache = Path(str(provenance["cache"]["external"]))
    external_ids, external_embedding, external_cache_path = alpha_embedding_cache(
        external_cache / "alpha_trajectories.csv",
        requested_external,
        source_digest=str(provenance["sources"]["external_stats"]["sha256"]),
        source_name="ablation_full_targets_and_missing_lineage_roots",
    )
    embeddings = embedding_lookup(
        development_ids, development_embedding, external_ids, external_embedding
    )
    closed_root_ids = lineage_root_ids.intersection(embeddings)
    closed_query_ids = set(
        branch_population.loc[
            branch_population["component_root_model"].astype(str).isin(closed_root_ids)
            & branch_population["model_id"].isin(embeddings),
            "model_id",
        ].astype(str)
    )
    (
        paper_query_ids,
        paper_root_ids,
        paper_similarity,
        paper_similarity_cache,
    ) = paper_root_similarity_cache(
        (RUN / "alpha_trajectories.csv", external_cache / "alpha_trajectories.csv"),
        closed_query_ids,
        closed_root_ids,
        source_digests=(
            str(provenance["sources"]["development_stats"]["sha256"]),
            str(provenance["sources"]["external_stats"]["sha256"]),
        ),
    )

    census, broad_clusters, broad_cluster_metrics = broad_lineage(
        phase2, development_features, manifest
    )
    branch_columns = [
        "model_id",
        "ancestry_component",
        "component_root_model",
        "component_exclusion_reason",
        "root_branch_model",
        "path_depth",
    ]
    census = census.drop(
        columns=[
            "ancestry_component",
            "component_root_model",
            "component_exclusion_reason",
        ],
        errors="ignore",
    ).merge(
        graph["branches"].loc[:, branch_columns],
        on="model_id",
        how="left",
        validate="one_to_one",
    )
    for key, columns in (
        (
            "conservative",
            ["model_id", "conservative_metadata_family", "conservative_family_conflict"],
        ),
        (
            "atlas_clean",
            ["model_id", "atlas_clean_metadata_family", "atlas_clean_family_conflict"],
        ),
        (
            "moderate",
            [
                "model_id",
                "consensus_root_model",
                "moderate_metadata_family",
                "moderate_family_conflict",
            ],
        ),
    ):
        census = census.merge(
            graph[key].loc[:, columns],
            on="model_id",
            how="left",
            validate="one_to_one",
        )
    source_audit = graph["source_audit"].loc[
        graph["source_audit"]["source"].eq("phase2")
    ].rename(
        columns={
            "status": "direct_source_status",
            "reason": "direct_source_reason",
            "already_in_atlas": "direct_source_in_atlas",
            "source_model": "direct_source_model",
            "relation": "direct_source_relation",
        }
    )
    census = census.merge(
        source_audit.loc[
            :,
            [
                "model_id",
                "direct_source_status",
                "direct_source_reason",
                "direct_source_in_atlas",
                "direct_source_model",
                "direct_source_relation",
            ],
        ],
        on="model_id",
        how="left",
        validate="one_to_one",
    )
    raw_flags = phase2.set_index("model_id")["known_lineage_flag"]
    census["raw_source_metadata_flag"] = census["model_id"].map(raw_flags).fillna(False)
    census["metadata_tier"] = np.select(
        [
            census["label_status"].eq("confirmed"),
            census["conservative_metadata_family"].notna(),
            census["moderate_metadata_family"].notna(),
        ],
        ["manual_confirmed", "conservative_corroborated", "moderate_root_consensus"],
        default="unresolved",
    )

    lineage_cluster_metrics = lineage_metric_table(
        census, strict_assignments, broad_clusters
    )
    branch_predictions, branch_metrics, branch_folds = evaluate_branch_lineage(
        branch_population, embeddings
    )
    closed_root_predictions, closed_root_metrics, closed_root_audit = (
        evaluate_exact_root_lineage(
            branch_population,
            development_features,
            external_features,
            set(development_ids),
            set(external_ids),
            embeddings,
            paper_query_ids,
            paper_root_ids,
            paper_similarity,
        )
    )
    matched_branch_predictions, matched_branch_metrics = evaluate_matched_branch_lineage(
        branch_population, embeddings, set(paper_root_ids)
    )
    duplicate_profile_audit = branch_profile_duplicate_audit(
        branch_population, embeddings
    )

    all_features = pd.concat(
        [development_features, external_features], ignore_index=True, sort=False
    )
    if all_features["model_id"].duplicated().any():
        raise ValueError("development and ablation feature populations overlap")
    modern_population = performance_population(
        all_features,
        targets["modern"],
        components,
        min_role_coverage=float(settings["min_role_coverage"]),
    )
    legacy_population = performance_population(
        all_features,
        targets["legacy"],
        components,
        min_role_coverage=float(settings["min_role_coverage"]),
    )
    modern_paired = modern_population.loc[
        modern_population["model_id"].isin(embeddings)
    ].reset_index(drop=True)
    legacy_paired = legacy_population.loc[
        legacy_population["model_id"].isin(embeddings)
    ].reset_index(drop=True)
    (
        modern_predictions,
        modern_metrics,
        modern_increments,
        modern_fit_audit,
    ) = grouped_performance(
        modern_paired,
        embeddings,
        MODERN_TARGETS,
        scope="modern_full_reconciled_grouped_paired",
    )
    (
        legacy_predictions,
        legacy_metrics,
        legacy_increments,
        legacy_fit_audit,
    ) = grouped_performance(
        legacy_paired,
        embeddings,
        LEGACY_TARGETS,
        scope="legacy_full_reconciled_grouped_paired",
    )
    modern_full_ridge, modern_full_ridge_metrics, modern_full_ridge_audit = evaluate_grouped_ridge(
        modern_population,
        target="modern_average",
        outer_splits=OUTER_FOLDS,
        inner_splits=INNER_FOLDS,
        ridge_alphas=RIDGE_ALPHAS,
        random_state=SEED,
    )
    legacy_full_ridge, legacy_full_ridge_metrics, legacy_full_ridge_audit = evaluate_grouped_ridge(
        legacy_population,
        target="legacy_average",
        outer_splits=OUTER_FOLDS,
        inner_splits=INNER_FOLDS,
        ridge_alphas=RIDGE_ALPHAS,
        random_state=SEED,
    )

    modern_random_frame = targets["modern"].merge(
        components.loc[:, ["model_id", "ancestry_component"]],
        on="model_id",
        how="left",
        validate="one_to_one",
    )
    legacy_random_frame = targets["legacy"].merge(
        components.loc[:, ["model_id", "ancestry_component"]],
        on="model_id",
        how="left",
        validate="one_to_one",
    )
    modern_paper = paper_random_splits(
        modern_random_frame, embeddings, MODERN_TARGETS
    )
    legacy_paper = paper_random_splits(
        legacy_random_frame, embeddings, LEGACY_TARGETS
    )
    source_predictions, source_metrics, source_increments = source_heldout_modern_full(
        modern_paired, embeddings
    )

    confirmed = labels.loc[
        labels["label_status"].eq("confirmed")
        & labels["audited_base_family"].notna()
    ].copy()
    confirmed["model_id"] = confirmed["model_id"].map(canonical)
    feature_ids = set(development_features["model_id"].astype(str))
    strict_ids = set(strict_assignments["model_id"].astype(str))
    broad_ids = set(broad_clusters["model_id"].astype(str))

    def lineage_count(column: str, eligible: set[str] | None = None) -> int:
        mask = census[column].notna()
        if eligible is not None:
            mask &= census["model_id"].isin(eligible)
        return int(mask.sum())

    feature_model_ids = set(all_features["model_id"].astype(str))
    phase2_edge_audit = graph["source_audit"].loc[
        graph["source_audit"]["source"].eq("phase2")
    ]
    accepted_phase2_edges = phase2_edge_audit.loc[
        phase2_edge_audit["status"].eq("accepted")
    ]

    def edge_reason_count(reason: str) -> int:
        return int(phase2_edge_audit["reason"].eq(reason).sum())

    counts = {
        "phase2_census": int(len(census)),
        "phase2_htsr_artifacts": int(census["htsr_artifact_available"].sum()),
        "phase2_strict_lineage": int(census["strict_cluster"].notna().sum()),
        "phase2_broad_lineage": int(census["broad_cluster"].notna().sum()),
        "lineage_manual_reviewed": int(len(labels)),
        "lineage_manual_confirmed": int(len(confirmed)),
        "lineage_manual_phase2": int(confirmed["model_id"].isin(set(phase2["model_id"])).sum()),
        "lineage_manual_with_htsr": int(confirmed["model_id"].isin(feature_ids).sum()),
        "lineage_manual_strict": int(confirmed["model_id"].isin(strict_ids).sum()),
        "lineage_raw_source_metadata_flag": int(census["raw_source_metadata_flag"].sum()),
        "lineage_direct_edges_accepted": int(len(accepted_phase2_edges)),
        "lineage_direct_edges_in_atlas": int(accepted_phase2_edges["already_in_atlas"].sum()),
        "lineage_direct_edges_added": int((~accepted_phase2_edges["already_in_atlas"]).sum()),
        "lineage_direct_self_references": edge_reason_count("self_reference"),
        "lineage_direct_displayed_self_merges": edge_reason_count("displayed_self_merge"),
        "lineage_direct_weak_relations": edge_reason_count("relation_not_strong"),
        "lineage_direct_invalid_parents": edge_reason_count("invalid_parent_id"),
        "lineage_direct_unrepresented_parents": edge_reason_count("parent_not_represented"),
        "lineage_conservative_all": lineage_count("conservative_metadata_family"),
        "lineage_conservative_with_htsr": lineage_count(
            "conservative_metadata_family", feature_ids
        ),
        "lineage_conservative_broad": lineage_count(
            "conservative_metadata_family", broad_ids
        ),
        "lineage_conservative_strict": lineage_count(
            "conservative_metadata_family", strict_ids
        ),
        "lineage_atlas_clean_all": lineage_count("atlas_clean_metadata_family"),
        "lineage_atlas_clean_strict": lineage_count(
            "atlas_clean_metadata_family", strict_ids
        ),
        "lineage_moderate_all": lineage_count("moderate_metadata_family"),
        "lineage_moderate_with_htsr": lineage_count("moderate_metadata_family", feature_ids),
        "lineage_moderate_broad": lineage_count("moderate_metadata_family", broad_ids),
        "lineage_moderate_strict": lineage_count("moderate_metadata_family", strict_ids),
        "lineage_branch_models": int(branch_predictions["model_id"].nunique()),
        "lineage_branch_alpha_models": int(
            branch_predictions.loc[
                branch_predictions["method"].eq("alpha_profile")
                & branch_predictions["scorable"].eq(True),
                "model_id",
            ].nunique()
        ),
        "lineage_branch_roots": int(branch_predictions["actual_root"].nunique()),
        "lineage_branch_groups": int(branch_predictions["root_branch_model"].nunique()),
        "lineage_closed_root_models": int(
            closed_root_predictions.loc[
                closed_root_predictions["scorable"].eq(True), "model_id"
            ].nunique()
        ),
        "lineage_closed_root_candidates": int(
            closed_root_audit["root_alpha_profile_valid"].sum()
        ),
        "lineage_closed_root_population_roots": int(len(closed_root_audit)),
        "lineage_matched_branch_models": int(
            matched_branch_predictions.loc[
                matched_branch_predictions["scorable"].eq(True), "model_id"
            ].nunique()
        ),
        "lineage_cross_fold_duplicate_profile_models": int(
            duplicate_profile_audit.at[
                0, "n_models_in_cross_fold_duplicate_groups"
            ]
        ),
        "lineage_family_suggestions": int(census["family_abstained"].eq(False).sum()),
        "modern_complete_targets": int(len(targets["modern"])),
        "modern_phase2_targets": int(targets["modern"]["source_population"].eq("phase2").sum()),
        "modern_ablation_targets": int(targets["modern"]["source_population"].eq("ablation").sum()),
        "modern_targets_with_htsr": int(targets["modern"]["model_id"].isin(feature_model_ids).sum()),
        "modern_alpha_profiles": int(targets["modern"]["model_id"].isin(embeddings).sum()),
        "modern_grouped_models": int(len(modern_population)),
        "modern_paired_models": int(len(modern_paired)),
        "modern_paired_components": int(modern_paired["ancestry_component"].nunique()),
        "legacy_complete_targets": int(len(targets["legacy"])),
        "legacy_phase2_targets": int(targets["legacy"]["source_population"].eq("phase2").sum()),
        "legacy_ablation_targets": int(targets["legacy"]["source_population"].eq("ablation").sum()),
        "legacy_targets_with_htsr": int(targets["legacy"]["model_id"].isin(feature_model_ids).sum()),
        "legacy_alpha_profiles": int(targets["legacy"]["model_id"].isin(embeddings).sum()),
        "legacy_grouped_models": int(len(legacy_population)),
        "legacy_paired_models": int(len(legacy_paired)),
        "legacy_paired_components": int(legacy_paired["ancestry_component"].nunique()),
    }

    run_payload = {
        "method": "htsr_full_scale_reconciled_v6",
        "seed": SEED,
        "grid_size": GRID_SIZE,
        "group_knn_k": GROUP_K,
        "paper_knn_k": PAPER_K,
        "paper_repeats": PAPER_REPEATS,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "signflip_replicates": SIGNFLIP_REPLICATES,
        "minimum_components": MINIMUM_COMPONENTS,
        "lineage_feature_coverage": BROAD_COVERAGE,
        "lineage_clusterer": f"kmeans_n_init{KMEANS_N_INIT}_sqrt_half",
        "lineage_branch_rule": "first_child_below_clean_root_balanced_5fold",
        "lineage_closed_root_rule": "direct_exact_root_gallery_no_fitting",
        "lineage_root_identity": "canonical_repository_id_not_verified_revision_hash",
        "lineage_closed_root_similarity": (
            "paper_eq2_spearman_dp_stored_dks_primary_plus_interp32_secondary"
        ),
        "lineage_graph_rule": "atlas_plus_represented_strong_local_sources",
        "target_reconciliation": "exact_id_key_or_sha_then_identity_or_aggregate_v1",
        "script_sha256": sha256_file(__file__),
        "development_stats_sha256": provenance["sources"]["development_stats"]["sha256"],
        "external_stats_sha256": provenance["sources"]["external_stats"]["sha256"],
        "development_target_sha256": provenance["sources"]["development_targets"]["sha256"],
        "external_target_sha256": provenance["sources"]["external_targets"]["sha256"],
        "overlap_audit_sha256": sha256_file(overlap_path),
        "graph_sha256": provenance["sources"]["graph"]["sha256"],
        "labels_sha256": provenance["sources"]["labels"]["sha256"],
        "augmented_graph_sha256": str(components["component_graph_fingerprint"].iloc[0]),
    }
    run_id = "scaleup_" + hashlib.sha256(
        json.dumps(run_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    destination = SCALE_ROOT / "runs" / run_id
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError(f"scale-up run already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=destination.parent))

    tables = {
        "population_ledger.csv": census,
        "lineage_source_audit.csv": graph["source_audit"],
        "lineage_broad_cluster_assignments.csv": broad_clusters,
        "lineage_broad_cluster_metrics.csv": broad_cluster_metrics,
        "lineage_cluster_label_metrics.csv": lineage_cluster_metrics,
        "lineage_branch_predictions.csv": branch_predictions,
        "lineage_branch_metrics.csv": branch_metrics,
        "lineage_branch_fold_audit.csv": branch_folds,
        "lineage_closed_root_predictions.csv": closed_root_predictions,
        "lineage_closed_root_metrics.csv": closed_root_metrics,
        "lineage_closed_root_audit.csv": closed_root_audit,
        "lineage_matched_branch_predictions.csv": matched_branch_predictions,
        "lineage_matched_branch_metrics.csv": matched_branch_metrics,
        "lineage_branch_profile_duplicate_audit.csv": duplicate_profile_audit,
        "performance_modern_targets.csv": targets["modern"],
        "performance_legacy_targets.csv": targets["legacy"],
        "performance_target_reconciliation_audit.csv": targets["audit"],
        "performance_target_exclusions.csv": targets["exclusions"],
        "modern_paper_like_summary.csv": modern_paper,
        "modern_grouped_alpha_predictions.csv": modern_predictions.loc[
            modern_predictions["feature_block"].eq("alpha_profile_k5_component_distinct")
        ],
        "modern_grouped_alpha_metrics.csv": modern_metrics.loc[
            modern_metrics["feature_block"].eq("alpha_profile_k5_component_distinct")
        ],
        "modern_source_heldout_predictions.csv": source_predictions,
        "modern_source_heldout_metrics.csv": source_metrics,
        "modern_source_heldout_increments.csv": source_increments,
        "modern_pooled_predictions.csv": modern_predictions,
        "modern_pooled_metrics.csv": modern_metrics,
        "modern_pooled_increments.csv": modern_increments,
        "modern_full_ridge_predictions.csv": modern_full_ridge,
        "modern_full_ridge_metrics.csv": modern_full_ridge_metrics,
        "legacy_paper_like_summary.csv": legacy_paper,
        "legacy_grouped_alpha_predictions.csv": legacy_predictions.loc[
            legacy_predictions["feature_block"].eq("alpha_profile_k5_component_distinct")
        ],
        "legacy_grouped_alpha_metrics.csv": legacy_metrics.loc[
            legacy_metrics["feature_block"].eq("alpha_profile_k5_component_distinct")
        ],
        "legacy_paired_predictions.csv": legacy_predictions,
        "legacy_paired_metrics.csv": legacy_metrics,
        "legacy_increments.csv": legacy_increments,
        "legacy_full_ridge_predictions.csv": legacy_full_ridge,
        "legacy_full_ridge_metrics.csv": legacy_full_ridge_metrics,
        "performance_fit_audit.csv": pd.concat(
            [
                modern_fit_audit.assign(generation="modern", population="paired"),
                legacy_fit_audit.assign(generation="legacy", population="paired"),
                pd.DataFrame(modern_full_ridge_audit).assign(
                    generation="modern", population="all_grouped"
                ),
                pd.DataFrame(legacy_full_ridge_audit).assign(
                    generation="legacy", population="all_grouped"
                ),
            ],
            ignore_index=True,
            sort=False,
        ),
    }
    for filename, frame in tables.items():
        frame.to_csv(temporary / filename, index=False)

    metadata = {
        "run_id": run_id,
        "config": run_payload,
        "counts": counts,
        "caches": {
            "phase2_alpha": str(development_cache),
            "ablation_alpha": str(external_cache_path),
            "closed_root_paper_similarity": str(paper_similarity_cache),
        },
        "sources": {
            "phase2": {"path": str(phase2_path), "sha256": run_payload["development_target_sha256"]},
            "ablation": {"path": str(ablation_path), "sha256": run_payload["external_target_sha256"]},
            "overlap_audit": {"path": str(overlap_path), "sha256": run_payload["overlap_audit_sha256"]},
            "graph": {"path": str(graph_path), "sha256": run_payload["graph_sha256"]},
            "labels": {"path": str(labels_path), "sha256": run_payload["labels_sha256"]},
        },
        "evidence_roles": {
            "manual_lineage": "manual_audit",
            "propagated_lineage": "metadata_derived",
            "branch_lineage": "metadata_graph_agreement",
            "closed_root_lineage": "metadata_graph_agreement",
            "modern_pooled": "internal_ancestry_grouped",
            "modern_source_heldout": EXTERNAL_EVIDENCE_ROLE,
            "paper_like_random_split": "interpolation_diagnostic",
        },
    }
    (temporary / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    best_branch = branch_metrics.loc[
        branch_metrics["method"].eq("alpha_profile")
    ].iloc[0]
    exact_root = closed_root_metrics.loc[
        closed_root_metrics["method"].eq("exact_root_paper_spearman_dks")
    ].iloc[0]
    exact_root_fast = closed_root_metrics.loc[
        closed_root_metrics["method"].eq("exact_root_alpha_profile")
    ].iloc[0]
    matched_branch = matched_branch_metrics.iloc[0]
    modern_average = modern_metrics.loc[
        modern_metrics["target"].eq("modern_average")
    ].set_index("feature_block")
    legacy_average = legacy_metrics.loc[
        legacy_metrics["target"].eq("legacy_average")
    ].set_index("feature_block")
    modern_difference = modern_increments.loc[
        modern_increments["target"].eq("modern_average")
        & modern_increments["baseline"].eq("M+S+H")
        & modern_increments["candidate"].eq(
            "alpha_profile_k5_component_distinct"
        )
    ].iloc[0]
    legacy_difference = legacy_increments.loc[
        legacy_increments["target"].eq("legacy_average")
        & legacy_increments["baseline"].eq("M+S+H")
        & legacy_increments["candidate"].eq(
            "alpha_profile_k5_component_distinct"
        )
    ].iloc[0]
    report = [
        "# HTSR full scale study",
        "",
        f"Run: `{run_id}`",
        "",
        "## Main findings",
        "",
        f"- The 174-model result is the last step of a manual funnel: {counts['lineage_manual_reviewed']:,} reviewed, {counts['lineage_manual_confirmed']:,} confirmed, {counts['lineage_manual_phase2']:,} present in Phase 2, {counts['lineage_manual_with_htsr']:,} with HTSR, and {counts['lineage_manual_strict']:,} passing strict feature rules.",
        f"- Phase 2 contains {counts['lineage_raw_source_metadata_flag']:,} raw lineage flags. The audit accepts {counts['lineage_direct_edges_accepted']:,} strong represented parent links: {counts['lineage_direct_edges_in_atlas']:,} already match Model Atlas and {counts['lineage_direct_edges_added']:,} are represented local additions.",
        f"- Directed label propagation gives {counts['lineage_conservative_strict']:,} corroborated strict family labels and {counts['lineage_moderate_strict']:,} clean root-consensus strict labels. These are metadata-derived, not independent truth.",
        f"- Closed-set exact-root attribution compares each descendant directly with every valid candidate root profile; it does not fit descendant centers. Valid profiles exist for {int(exact_root['n_candidate_roots']):,} of {int(exact_root['n_population_roots']):,} roots, covering {int(exact_root['n_models']):,} of {int(exact_root['n_eligible']):,} eligible descendants. Paper-style DKS Spearman reaches {exact_root['root_macro_top1']:.3f} root-balanced top-1 ({exact_root['micro_top1']:.3f} overall top-1); the scalable interpolated profile reaches {exact_root_fast['root_macro_top1']:.3f}.",
        f"- On the same {int(matched_branch['n_models']):,} models and {int(matched_branch['n_roots']):,} roots as the closed-set gallery, holding complete branches out gives {matched_branch['root_macro_top1']:.3f} root-balanced top-1 with the scalable alpha profile. This matched comparison isolates evaluation design better than comparing different root populations.",
        f"- Branch-held-out root prediction has {counts['lineage_branch_models']:,} eligible strict models in {int(best_branch['n_roots']):,} roots. Alpha profiles score {int(best_branch['n_models']):,} of them and reach {best_branch['root_macro_top1']:.3f} root-balanced top-1 and {best_branch['top3_accuracy']:.3f} top-3 agreement.",
        f"- Modern target funnel: {counts['modern_phase2_targets']:,} Phase 2 + {counts['modern_ablation_targets']:,} ablation = {counts['modern_complete_targets']:,} complete targets; {counts['modern_targets_with_htsr']:,} have HTSR, {counts['modern_alpha_profiles']:,} have valid alpha profiles, and {counts['modern_paired_models']:,} in {counts['modern_paired_components']:,} ancestry/checkpoint groups enter the paired comparison.",
        f"- Legacy target funnel: {counts['legacy_phase2_targets']:,} Phase 2 + {counts['legacy_ablation_targets']:,} ablation = {counts['legacy_complete_targets']:,} complete targets; {counts['legacy_targets_with_htsr']:,} have HTSR, {counts['legacy_alpha_profiles']:,} have valid alpha profiles, and {counts['legacy_paired_models']:,} in {counts['legacy_paired_components']:,} groups enter the paired comparison.",
        f"- Modern average: alpha-profile MAE {modern_average.at['alpha_profile_k5_component_distinct', 'component_weighted_mae']:.3f} versus Ridge M+S+H {modern_average.at['M+S+H', 'component_weighted_mae']:.3f}; difference {modern_difference['delta_mae']:.3f}, 95% interval [{modern_difference['ci_low']:.3f}, {modern_difference['ci_high']:.3f}], p={modern_difference['p_value_raw']:.3f}.",
        f"- Legacy average: alpha-profile MAE {legacy_average.at['alpha_profile_k5_component_distinct', 'component_weighted_mae']:.3f} versus Ridge M+S+H {legacy_average.at['M+S+H', 'component_weighted_mae']:.3f}; difference {legacy_difference['delta_mae']:.3f}, 95% interval [{legacy_difference['ci_low']:.3f}, {legacy_difference['ci_high']:.3f}], p={legacy_difference['p_value_raw']:.4f}.",
        "",
        "## Interpretation limits",
        "",
        "- Manual labels, graph-derived labels, and spectral predictions remain separate columns.",
        "- A graph result measures agreement with repository metadata. It does not independently prove training ancestry.",
        "- The exact-root endpoint is closed-set and covers only roots with valid analyzed profiles. Its score is not an estimate for missing roots or for unseen families.",
        "- An exact graph root means an exact canonical repository ID in the graph. A revision hash was not available for every root, so this is not checkpoint-level identity verification.",
        f"- {int(duplicate_profile_audit.at[0, 'n_models_in_cross_fold_duplicate_groups']):,} branch-test models share an identical derived alpha profile with a model in another fold. This is not proof of identical weights, but it can make branch prediction easier and remains a sensitivity-analysis target.",
        "- Merges, conflicting roots, cycles, malformed parent IDs, and unrepresented parents are not forced into one family.",
        "- Modern and legacy leaderboard generations are never pooled into one target.",
        "- Main performance estimates hold complete graph groups and shared checkpoint hashes out together.",
        "- Random 50/50 results are retained only to compare with the paper's easier interpolation setting.",
        f"- Source-held-out ablation results are `{EXTERNAL_EVIDENCE_ROLE}` because that collection was previously used during method development.",
    ]
    (temporary / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    os.rename(temporary, destination)
    write_artifact_manifest(destination, run_id=run_id)
    print(json.dumps({"run": str(destination), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
