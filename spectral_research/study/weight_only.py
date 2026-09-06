from __future__ import annotations

"""Weight-only lineage discovery and score prediction helpers."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau, rankdata, spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    r2_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.neighbors import NearestNeighbors

from .features import CANONICAL_ROLES
from .lineage import _signature_matrix


@dataclass(frozen=True)
class ProfileCorpus:
    model_ids: np.ndarray
    embeddings: np.ndarray
    cache_paths: tuple[Path, ...]


def _content_key(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def build_rank_profile_cache(
    trajectory_path: Path,
    model_ids: Iterable[str],
    *,
    cache_root: Path,
    source_digest: str,
    source_name: str,
    grid_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, Path]:
    """Build normalized role-wise rank profiles from stored alpha trajectories."""

    requested = sorted(set(map(str, model_ids)))
    payload = {
        "method": "rolewise_normalized_depth_rank_profile_v1",
        "grid_size": int(grid_size),
        "source_digest": source_digest,
        "model_ids": requested,
    }
    cache_path = cache_root / f"{source_name}_{_content_key(payload)}.npz"
    if cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        cached_ids = cached["model_ids"].astype(str)
        embeddings = cached["embeddings"].astype(np.float32, copy=False)
        if cached_ids.tolist() != sorted(cached_ids.tolist()):
            raise ValueError(f"Profile cache identifiers are not sorted: {cache_path}")
        expected = (len(cached_ids), len(CANONICAL_ROLES) * grid_size)
        if embeddings.shape != expected:
            raise ValueError(f"Profile cache has shape {embeddings.shape}, expected {expected}")
        return cached_ids, embeddings, cache_path

    requested_set = set(requested)
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(trajectory_path, chunksize=300_000, low_memory=False):
        selected = chunk.loc[chunk["model_id"].astype(str).isin(requested_set)]
        if not selected.empty:
            chunks.append(selected)
    trajectories = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    grid = np.linspace(0.0, 1.0, grid_size)
    kept_ids: list[str] = []
    rows: list[np.ndarray] = []
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
        for role_index in range(len(CANONICAL_ROLES)):
            interpolated = np.interp(grid, normalized_depth, values[:, role_index])
            ranks = rankdata(interpolated, method="average").astype(float)
            ranks -= ranks.mean()
            norm = float(np.linalg.norm(ranks))
            if not np.isfinite(norm) or norm <= np.finfo(float).eps:
                role_vectors = []
                break
            role_vectors.append(ranks / norm)
        if role_vectors:
            kept_ids.append(str(model_id))
            rows.append(np.concatenate(role_vectors).astype(np.float32))

    embeddings = (
        np.vstack(rows)
        if rows
        else np.empty((0, len(CANONICAL_ROLES) * grid_size), dtype=np.float32)
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        model_ids=np.asarray(kept_ids, dtype=str),
        embeddings=embeddings,
    )
    return np.asarray(kept_ids, dtype=str), embeddings, cache_path


def combine_profile_caches(
    parts: Sequence[tuple[np.ndarray, np.ndarray, Path]],
) -> ProfileCorpus:
    model_ids = np.concatenate([part[0].astype(str) for part in parts])
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("Profile sources contain overlapping model IDs")
    embeddings = np.vstack([part[1].astype(np.float32, copy=False) for part in parts])
    order = np.argsort(model_ids, kind="stable")
    return ProfileCorpus(
        model_ids=model_ids[order],
        embeddings=embeddings[order],
        cache_paths=tuple(part[2] for part in parts),
    )


def profile_split_table(
    corpus: ProfileCorpus,
    *,
    seed: int,
    n_folds: int,
) -> pd.DataFrame:
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    profile_hashes = [
        hashlib.sha256(row.astype(np.float32, copy=False).tobytes()).hexdigest()
        for row in corpus.embeddings
    ]
    folds = [
        int(hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()[:16], 16)
        % n_folds
        for value in profile_hashes
    ]
    result = pd.DataFrame(
        {
            "model_id": corpus.model_ids,
            "profile_hash": profile_hashes,
            "blind_fold": folds,
        }
    )
    if result.groupby("profile_hash")["blind_fold"].nunique().max() != 1:
        raise RuntimeError("An exact profile group was split across folds")
    return result


def cluster_holdout_split_table(
    profile_splits: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    *,
    n_folds: int,
) -> pd.DataFrame:
    """Keep each weight-derived spectral cluster wholly inside one fold."""

    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    sizes = (
        cluster_assignments.groupby("predicted_cluster")
        .size()
        .sort_values(ascending=False, kind="stable")
    )
    loads = np.zeros(n_folds, dtype=int)
    fold_by_cluster: dict[int, int] = {}
    for cluster, size in sizes.items():
        fold = int(np.argmin(loads))
        fold_by_cluster[int(cluster)] = fold
        loads[fold] += int(size)
    result = profile_splits.loc[:, ["model_id", "profile_hash"]].merge(
        cluster_assignments.loc[:, ["model_id", "predicted_cluster"]],
        on="model_id",
        validate="one_to_one",
    )
    result["blind_fold"] = result["predicted_cluster"].map(fold_by_cluster).astype(int)
    if result.groupby("predicted_cluster")["blind_fold"].nunique().max() != 1:
        raise RuntimeError("A spectral cluster was split across folds")
    return result


def _cluster_candidates(n_models: int) -> list[int]:
    if n_models < 4:
        return [2]
    upper = max(2, int(np.ceil(np.sqrt(n_models))))
    count = max(4, int(np.ceil(np.log2(n_models))))
    return sorted(set(np.geomspace(2, upper, count).round().astype(int).tolist()))


def _batch_size(n_models: int) -> int:
    return min(n_models, max(256, int(np.ceil(np.sqrt(n_models))) * 16))


def _fit_clusters(matrix: np.ndarray, n_clusters: int, seed: int) -> MiniBatchKMeans:
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=10,
        batch_size=_batch_size(len(matrix)),
        reassignment_ratio=0.01,
    ).fit(matrix)


def discover_clusters(
    corpus: ProfileCorpus,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select and fit clusters using spectral profiles and internal metrics only."""

    matrix = corpus.embeddings
    if len(matrix) < 4:
        raise ValueError("At least four valid profiles are required")
    sample_size = min(len(matrix), max(500, int(np.ceil(np.sqrt(len(matrix)))) * 20))
    selection_rows: list[dict[str, object]] = []
    fitted: dict[int, MiniBatchKMeans] = {}
    for n_clusters in _cluster_candidates(len(matrix)):
        estimator = _fit_clusters(matrix, n_clusters, seed)
        fitted[n_clusters] = estimator
        labels = estimator.labels_
        selection_rows.append(
            {
                "n_clusters": n_clusters,
                "silhouette_cosine": float(
                    silhouette_score(
                        matrix,
                        labels,
                        metric="cosine",
                        sample_size=sample_size,
                        random_state=seed,
                    )
                ),
                "largest_cluster_fraction": float(
                    np.bincount(labels).max() / len(labels)
                ),
                "selection_uses_metadata": False,
            }
        )
    selection = pd.DataFrame(selection_rows)
    selected_k = int(
        selection.sort_values(
            ["silhouette_cosine", "n_clusters"],
            ascending=[False, True],
            kind="stable",
        ).iloc[0]["n_clusters"]
    )
    estimator = fitted[selected_k]
    labels = estimator.labels_.astype(int)

    medoid_by_cluster: dict[int, str] = {}
    similarity_by_model = np.empty(len(matrix), dtype=float)
    for cluster in range(selected_k):
        positions = np.flatnonzero(labels == cluster)
        values = matrix[positions]
        mean = values.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        scores = values @ mean / max(norm * np.sqrt(len(CANONICAL_ROLES)), 1e-12)
        local_medoid = int(np.argmax(scores))
        medoid_by_cluster[cluster] = str(corpus.model_ids[positions[local_medoid]])
        similarity_by_model[positions] = scores

    ordered_clusters = sorted(medoid_by_cluster, key=lambda value: medoid_by_cluster[value])
    remap = {old: new for new, old in enumerate(ordered_clusters)}
    canonical_labels = np.asarray([remap[value] for value in labels], dtype=int)
    canonical_medoids = {
        remap[cluster]: model_id for cluster, model_id in medoid_by_cluster.items()
    }

    nearest_names = np.repeat("", len(matrix)).astype(object)
    nearest_scores = np.full(len(matrix), np.nan, dtype=float)
    for cluster in range(selected_k):
        positions = np.flatnonzero(canonical_labels == cluster)
        values = matrix[positions]
        neighbors = NearestNeighbors(
            n_neighbors=min(3, len(values)), metric="cosine", algorithm="brute", n_jobs=-1
        ).fit(values)
        distances, indices = neighbors.kneighbors(values)
        for local, global_position in enumerate(positions):
            candidates = [
                int(value) for value in indices[local] if int(value) != local
            ][:2]
            nearest_names[global_position] = ",".join(
                str(corpus.model_ids[positions[value]]) for value in candidates
            )
            if candidates:
                candidate_position = int(np.flatnonzero(indices[local] == candidates[0])[0])
                nearest_scores[global_position] = 1.0 - float(distances[local, candidate_position])

    assignments = pd.DataFrame(
        {
            "model_id": corpus.model_ids,
            "predicted_cluster": canonical_labels,
            "spectral_medoid_model": [canonical_medoids[value] for value in canonical_labels],
            "similarity_to_cluster_mean": similarity_by_model,
            "nearest_neighbors": nearest_names,
            "nearest_similarity": nearest_scores,
        }
    ).sort_values("model_id", kind="stable", ignore_index=True)

    stability = []
    for offset in (1, 2, 3):
        repeated = _fit_clusters(matrix, selected_k, seed + offset).labels_
        stability.append(float(adjusted_rand_score(labels, repeated)))
    internal = pd.DataFrame(
        [
            {
                "method": "minibatch_kmeans_rank_profile",
                "n_models": len(matrix),
                "n_clusters": selected_k,
                "selection_rule": "maximum_cosine_silhouette",
                "silhouette_cosine": float(
                    selection.loc[
                        selection["n_clusters"].eq(selected_k), "silhouette_cosine"
                    ].iloc[0]
                ),
                "davies_bouldin_euclidean": float(davies_bouldin_score(matrix, labels)),
                "mean_seed_stability_ari": float(np.mean(stability)),
                "minimum_seed_stability_ari": float(np.min(stability)),
                "metadata_loaded_before_freeze": False,
            }
        ]
    )
    selection["selected"] = selection["n_clusters"].eq(selected_k)
    return assignments, selection, internal


def _matched_recall(truth: pd.Series, predicted: pd.Series) -> float:
    table = pd.crosstab(truth.astype(str), predicted.astype(str))
    if table.empty:
        return float("nan")
    rows, columns = linear_sum_assignment(-table.to_numpy())
    return float(table.to_numpy()[rows, columns].sum() / table.to_numpy().sum())


def validate_clusters(
    assignments: pd.DataFrame,
    validation: pd.DataFrame,
    label_specs: Sequence[tuple[str, str, pd.Series | None]],
) -> pd.DataFrame:
    """Reveal labels only after cluster assignments have been fixed."""

    rows: list[dict[str, object]] = []
    for label_source, label_column, mask in label_specs:
        columns = ["model_id", label_column]
        selected = validation.loc[mask, columns] if mask is not None else validation.loc[:, columns]
        scored = assignments.merge(selected, on="model_id", how="inner").dropna(
            subset=[label_column]
        )
        if scored.empty:
            continue
        truth = scored[label_column].astype(str)
        predicted = scored["predicted_cluster"].astype(str)
        counts = scored.groupby("predicted_cluster")[label_column].value_counts()
        purity = float(counts.groupby(level=0).max().sum() / len(scored))
        rows.append(
            {
                "label_source": label_source,
                "label_column": label_column,
                "n_models": len(scored),
                "n_families": truth.nunique(),
                "n_clusters": predicted.nunique(),
                "adjusted_rand": float(adjusted_rand_score(truth, predicted)),
                "normalized_mutual_information": float(
                    normalized_mutual_info_score(truth, predicted)
                ),
                "homogeneity": float(homogeneity_score(truth, predicted)),
                "completeness": float(completeness_score(truth, predicted)),
                "v_measure": float(v_measure_score(truth, predicted)),
                "cluster_purity": purity,
                "matched_family_recall": _matched_recall(truth, predicted),
                "evidence_role": "post_freeze_metadata_validation",
            }
        )
    return pd.DataFrame(rows)


def supplied_gallery_predictions(
    corpus: ProfileCorpus,
    gallery_ids: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Route every non-gallery profile using only a supplied candidate list."""

    positions = {model_id: index for index, model_id in enumerate(corpus.model_ids)}
    supplied = list(dict.fromkeys(map(str, gallery_ids)))
    valid = [model_id for model_id in supplied if model_id in positions]
    if not valid:
        raise ValueError("No supplied gallery checkpoint has a valid profile")
    root_matrix = np.vstack([corpus.embeddings[positions[model_id]] for model_id in valid])
    query_mask = ~np.isin(corpus.model_ids, supplied)
    query_ids = corpus.model_ids[query_mask]
    scores = corpus.embeddings[query_mask] @ root_matrix.T / float(len(CANONICAL_ROLES))
    order = np.argsort(-scores, axis=1, kind="stable")
    ranked = np.asarray(valid, dtype=object)[order]
    best = order[:, 0]
    second = order[:, 1] if len(valid) > 1 else best
    rows = pd.DataFrame(
        {
            "model_id": query_ids,
            "predicted_root": ranked[:, 0],
            "top3_roots": ["|".join(map(str, values[:3])) for values in ranked],
            "root_ranking": ["|".join(map(str, values)) for values in ranked],
            "score": scores[np.arange(len(scores)), best],
            "margin": scores[np.arange(len(scores)), best]
            - scores[np.arange(len(scores)), second],
            "candidate_count": len(valid),
            "method": "supplied_gallery_interpolated_spearman_dks",
            "prediction_uses_metadata": False,
        }
    )
    audit = pd.DataFrame(
        {
            "gallery_model_id": supplied,
            "profile_available": [model_id in positions for model_id in supplied],
            "gallery_role": "operator_supplied_candidate",
        }
    )
    return rows, audit


def validate_gallery_predictions(
    predictions: pd.DataFrame,
    validation: pd.DataFrame,
    gallery_audit: pd.DataFrame,
    *,
    truth_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_roots = set(
        gallery_audit.loc[gallery_audit["profile_available"], "gallery_model_id"].astype(str)
    )
    labelled = predictions.merge(
        validation.loc[:, ["model_id", truth_column]], on="model_id", how="inner"
    ).dropna(subset=[truth_column])
    labelled["actual_root"] = labelled[truth_column].astype(str)
    labelled["truth_in_gallery"] = labelled["actual_root"].isin(valid_roots)
    scored = labelled.loc[labelled["truth_in_gallery"]].copy()
    scored["correct"] = scored["predicted_root"].eq(scored["actual_root"])
    scored["actual_rank"] = [
        roots.split("|").index(actual) + 1
        for roots, actual in zip(scored["root_ranking"], scored["actual_root"], strict=False)
    ]
    scored["top3_correct"] = scored["actual_rank"].le(3)
    by_root = scored.groupby("actual_root").agg(
        accuracy=("correct", "mean"),
        top3=("top3_correct", "mean"),
        reciprocal_rank=("actual_rank", lambda values: float(np.mean(1.0 / values))),
    )
    metrics = pd.DataFrame(
        [
            {
                "method": "supplied_gallery_interpolated_spearman_dks",
                "n_predictions_frozen": len(predictions),
                "n_labelled_queries": len(labelled),
                "n_scored_queries": len(scored),
                "n_supplied_roots": len(gallery_audit),
                "n_profiled_roots": len(valid_roots),
                "micro_top1": float(scored["correct"].mean()),
                "root_macro_top1": float(by_root["accuracy"].mean()),
                "micro_top3": float(scored["top3_correct"].mean()),
                "root_macro_top3": float(by_root["top3"].mean()),
                "root_macro_mrr": float(by_root["reciprocal_rank"].mean()),
                "label_source": truth_column,
                "evidence_role": "post_freeze_metadata_validation",
                "estimator_note": "stored_dks_alpha_not_fixed_hill",
                "depth_note": "normalized_depth_interpolation_not_paper_exact_alignment",
            }
        ]
    )
    return scored, metrics


def _rank_similarity(test: np.ndarray, train: np.ndarray) -> np.ndarray:
    return np.clip(
        test @ train.T / float(len(CANONICAL_ROLES)), -1.0, 1.0
    )


def _weighted_knn(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    similarities = _rank_similarity(test_x, train_x)
    count = min(k, len(train_x))
    neighbor_positions = np.argpartition(-similarities, count - 1, axis=1)[:, :count]
    local_scores = np.take_along_axis(similarities, neighbor_positions, axis=1)
    local_order = np.argsort(-local_scores, axis=1, kind="stable")
    neighbor_positions = np.take_along_axis(neighbor_positions, local_order, axis=1)
    local_scores = np.take_along_axis(similarities, neighbor_positions, axis=1)
    weights = 1.0 / (np.maximum(1.0 - local_scores, 0.0) + 1e-6)
    values = train_y[neighbor_positions]
    prediction = (weights[:, :, None] * values).sum(axis=1) / weights.sum(axis=1)[:, None]
    return prediction, neighbor_positions, local_scores


def cross_validated_performance_freeze(
    corpus: ProfileCorpus,
    splits: pd.DataFrame,
    targets: pd.DataFrame,
    target_columns: Sequence[str],
    *,
    generation: str,
    k: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Predict held-out scores; test targets never enter the predictor."""

    positions = {model_id: index for index, model_id in enumerate(corpus.model_ids)}
    population = targets.loc[
        targets["model_id"].astype(str).isin(positions), ["model_id", *target_columns]
    ].copy()
    population = population.merge(
        splits.loc[:, ["model_id", "profile_hash", "blind_fold"]],
        on="model_id",
        how="inner",
        validate="one_to_one",
    )
    numeric = population.loc[:, target_columns].apply(pd.to_numeric, errors="coerce")
    finite = numeric.notna().all(axis=1) & np.isfinite(numeric).all(axis=1)
    population = population.loc[finite].reset_index(drop=True)
    values = numeric.loc[finite].to_numpy(float)
    matrix = np.vstack([corpus.embeddings[positions[value]] for value in population["model_id"]])
    profile_hash_by_model = population.set_index("model_id")["profile_hash"].astype(str)
    freeze_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []
    folds = sorted(population["blind_fold"].unique())
    for fold in folds:
        test_mask = population["blind_fold"].eq(fold).to_numpy()
        train_mask = ~test_mask
        train_ids = population.loc[train_mask, "model_id"].astype(str).to_numpy()
        test_ids = population.loc[test_mask, "model_id"].astype(str).to_numpy()
        prediction, neighbors, similarities = _weighted_knn(
            matrix[train_mask], matrix[test_mask], values[train_mask], k=k
        )
        baseline = np.repeat(values[train_mask].mean(axis=0)[None, :], len(test_ids), axis=0)
        for local, model_id in enumerate(test_ids):
            neighbor_ids = train_ids[neighbors[local]]
            leakage_rows.append(
                {
                    "generation": generation,
                    "model_id": model_id,
                    "blind_fold": int(fold),
                    "nearest_train_similarity": float(similarities[local, 0]),
                    "exact_profile_cross_fold": False,
                }
            )
            for target_index, target in enumerate(target_columns):
                freeze_rows.extend(
                    [
                        {
                            "generation": generation,
                            "model_id": model_id,
                            "profile_hash": profile_hash_by_model.at[model_id],
                            "blind_fold": int(fold),
                            "target": target,
                            "feature_block": "alpha_profile_knn3",
                            "prediction": float(prediction[local, target_index]),
                            "nearest_neighbors": ",".join(map(str, neighbor_ids)),
                            "nearest_similarity": float(similarities[local, 0]),
                        },
                        {
                            "generation": generation,
                            "model_id": model_id,
                            "profile_hash": profile_hash_by_model.at[model_id],
                            "blind_fold": int(fold),
                            "target": target,
                            "feature_block": "training_mean",
                            "prediction": float(baseline[local, target_index]),
                            "nearest_neighbors": "",
                            "nearest_similarity": np.nan,
                        },
                    ]
                )
    freeze = pd.DataFrame(freeze_rows).sort_values(
        ["generation", "target", "feature_block", "model_id"], ignore_index=True
    )
    actual = population.loc[:, ["model_id", *target_columns]].melt(
        id_vars="model_id", var_name="target", value_name="actual"
    )
    evaluated = freeze.merge(actual, on=["model_id", "target"], validate="many_to_one")
    return freeze, evaluated, pd.DataFrame(leakage_rows).sort_values("model_id", ignore_index=True)


def performance_metrics(
    evaluated: pd.DataFrame,
    *,
    evaluation_scope: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (generation, target, method), frame in evaluated.groupby(
        ["generation", "target", "feature_block"], sort=False
    ):
        actual = frame["actual"].to_numpy(float)
        predicted = frame["prediction"].to_numpy(float)
        finite = np.isfinite(actual) & np.isfinite(predicted)
        n_evaluated = int(finite.sum())
        n_abstained = int(len(frame) - n_evaluated)
        actual = actual[finite]
        predicted = predicted[finite]
        rows.append(
            {
                "evaluation_scope": evaluation_scope,
                "generation": generation,
                "target": target,
                "feature_block": method,
                "n_models": n_evaluated,
                "n_abstained": n_abstained,
                "n_profile_groups": frame["profile_hash"].nunique(),
                "model_weighted_mae": float(np.mean(np.abs(actual - predicted))),
                "rmse": float(np.sqrt(np.mean(np.square(actual - predicted)))),
                "spearman": float(spearmanr(actual, predicted).statistic),
                "kendall_tau": float(kendalltau(actual, predicted).statistic),
                "r2": float(r2_score(actual, predicted)),
                "prediction_inputs": (
                    "available_role_alpha_and_training_scores"
                    if method == "coverage_aware_alpha_knn3"
                    else "alpha_profile_and_training_scores"
                    if method == "alpha_profile_knn3"
                    else "training_scores_only"
                ),
                "lineage_metadata_used": False,
            }
        )
    return pd.DataFrame(rows)


def paper_style_repeated_performance(
    corpus: ProfileCorpus,
    splits: pd.DataFrame,
    targets: pd.DataFrame,
    target_columns: Sequence[str],
    *,
    generation: str,
    seed: int,
    repeats: int,
    k: int = 3,
) -> pd.DataFrame:
    positions = {model_id: index for index, model_id in enumerate(corpus.model_ids)}
    population = targets.loc[
        targets["model_id"].astype(str).isin(positions), ["model_id", *target_columns]
    ].copy()
    population = population.merge(
        splits.loc[:, ["model_id", "profile_hash"]],
        on="model_id",
        how="inner",
        validate="one_to_one",
    )
    numeric = population.loc[:, target_columns].apply(pd.to_numeric, errors="coerce")
    finite = numeric.notna().all(axis=1) & np.isfinite(numeric).all(axis=1)
    population = population.loc[finite].reset_index(drop=True)
    values = numeric.loc[finite].to_numpy(float)
    matrix = np.vstack([corpus.embeddings[positions[value]] for value in population["model_id"]])
    similarity = _rank_similarity(matrix, matrix).astype(np.float32)
    groups = population["profile_hash"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    for repeat in range(repeats):
        train_mask = np.asarray(
            [
                int(hashlib.sha256(f"{seed + repeat}\0{value}".encode()).hexdigest()[:16], 16)
                % 2
                == 0
                for value in groups
            ],
            dtype=bool,
        )
        if train_mask.sum() < k or (~train_mask).sum() == 0:
            continue
        train_indices = np.flatnonzero(train_mask)
        test_indices = np.flatnonzero(~train_mask)
        local = similarity[np.ix_(test_indices, train_indices)]
        count = min(k, len(train_indices))
        neighbors = np.argpartition(-local, count - 1, axis=1)[:, :count]
        scores = np.take_along_axis(local, neighbors, axis=1)
        order = np.argsort(-scores, axis=1, kind="stable")
        neighbors = np.take_along_axis(neighbors, order, axis=1)
        scores = np.take_along_axis(local, neighbors, axis=1)
        weights = 1.0 / (np.maximum(1.0 - scores, 0.0) + 1e-6)
        predicted = (
            weights[:, :, None] * values[train_indices][neighbors]
        ).sum(axis=1) / weights.sum(axis=1)[:, None]
        for target_index, target in enumerate(target_columns):
            actual = values[test_indices, target_index]
            estimate = predicted[:, target_index]
            tau = kendalltau(actual, estimate)
            rows.append(
                {
                    "generation": generation,
                    "repeat": repeat,
                    "target": target,
                    "n_models": len(population),
                    "n_train": int(train_mask.sum()),
                    "n_test": int((~train_mask).sum()),
                    "mae": float(np.mean(np.abs(actual - estimate))),
                    "spearman": float(spearmanr(actual, estimate).statistic),
                    "kendall_tau": float(tau.statistic),
                    "kendall_p": float(tau.pvalue),
                    "lineage_metadata_used": False,
                }
            )
    raw = pd.DataFrame(rows)
    return (
        raw.groupby(["generation", "target"], sort=False)
        .agg(
            repeats=("repeat", "nunique"),
            n_models=("n_models", "first"),
            mean_train=("n_train", "mean"),
            mean_test=("n_test", "mean"),
            mean_mae=("mae", "mean"),
            sd_mae=("mae", "std"),
            mean_spearman=("spearman", "mean"),
            mean_kendall=("kendall_tau", "mean"),
            significant_trials=("kendall_p", lambda values: int((values < 0.05).sum())),
        )
        .reset_index()
    )


def summarize_leakage(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for generation, frame in rows.groupby("generation", sort=False):
        values = frame["nearest_train_similarity"].to_numpy(float)
        records.append(
            {
                "generation": generation,
                "n_models": len(frame),
                "exact_profiles_crossing_folds": int(frame["exact_profile_cross_fold"].sum()),
                "nearest_similarity_median": float(np.quantile(values, 0.50)),
                "nearest_similarity_p95": float(np.quantile(values, 0.95)),
                "nearest_similarity_p99": float(np.quantile(values, 0.99)),
                "nearest_similarity_max": float(np.max(values)),
            }
        )
    return pd.DataFrame(records)
