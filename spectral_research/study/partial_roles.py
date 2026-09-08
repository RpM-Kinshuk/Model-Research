from __future__ import annotations

"""Coverage-aware role summaries for metadata-free HTSR comparisons."""

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .features import CANONICAL_ROLES
from .contracts import file_digest
from .lineage import _role_trajectory, _validate_alpha_provenance


@dataclass(frozen=True)
class PartialRoleCorpus:
    model_ids: np.ndarray
    role_embeddings: np.ndarray
    role_mask: np.ndarray
    cache_paths: tuple[Path, ...]

    @property
    def role_counts(self) -> np.ndarray:
        return self.role_mask.sum(axis=1).astype(int)

    @property
    def grid_size(self) -> int:
        return int(self.role_embeddings.shape[2])

    def select(self, minimum_roles: int) -> "PartialRoleCorpus":
        if not 1 <= minimum_roles <= len(CANONICAL_ROLES):
            raise ValueError("minimum_roles must be between 1 and 7")
        keep = self.role_counts >= minimum_roles
        return PartialRoleCorpus(
            model_ids=self.model_ids[keep],
            role_embeddings=self.role_embeddings[keep],
            role_mask=self.role_mask[keep],
            cache_paths=self.cache_paths,
        )


def _content_key(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def build_partial_role_cache(
    trajectory_path: Path,
    model_ids: Iterable[str],
    *,
    cache_root: Path,
    source_digest: str,
    source_name: str,
    grid_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    """Build one normalized rank vector per usable role without imputation."""

    requested = sorted(set(map(str, model_ids)))
    payload = {
        "method": "independent_role_normalized_depth_rank_v2",
        "grid_size": int(grid_size),
        "source_digest": source_digest,
        "model_ids": requested,
        "role_vocabulary": list(CANONICAL_ROLES),
        "trajectory_sha256": file_digest(trajectory_path),
        "implementation_sha256": {
            name: file_digest(Path(__file__).with_name(name))
            for name in ("partial_roles.py", "lineage.py", "features.py")
        },
    }
    cache_path = cache_root / f"{source_name}_{_content_key(payload)}.npz"
    if cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        cached_ids = cached["model_ids"].astype(str)
        role_embeddings = cached["role_embeddings"].astype(np.float32, copy=False)
        role_mask = cached["role_mask"].astype(bool, copy=False)
        expected = (len(cached_ids), len(CANONICAL_ROLES), grid_size)
        if role_embeddings.shape != expected or role_mask.shape != expected[:2]:
            raise ValueError(f"Invalid partial-role cache shape: {cache_path}")
        return cached_ids, role_embeddings, role_mask, cache_path

    requested_set = set(requested)
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(trajectory_path, chunksize=300_000, low_memory=False):
        selected = chunk.loc[chunk["model_id"].astype(str).isin(requested_set)]
        if not selected.empty:
            chunks.append(selected)
    trajectories = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if not trajectories.empty:
        _validate_alpha_provenance(trajectories, name=str(trajectory_path))

    grid = np.linspace(0.0, 1.0, grid_size)
    kept_ids: list[str] = []
    embeddings: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for model_id, frame in trajectories.groupby("model_id", sort=True):
        role_values = np.zeros(
            (len(CANONICAL_ROLES), grid_size), dtype=np.float32
        )
        role_mask = np.zeros(len(CANONICAL_ROLES), dtype=bool)
        for role_index, role in enumerate(CANONICAL_ROLES):
            trajectory = _role_trajectory(frame, role)
            if trajectory is None:
                continue
            depths, alpha = trajectory
            span = float(depths[-1] - depths[0])
            if not np.isfinite(span) or span <= 0.0:
                continue
            normalized_depth = (depths - depths[0]) / span
            interpolated = np.interp(grid, normalized_depth, alpha)
            ranks = rankdata(interpolated, method="average").astype(float)
            ranks -= ranks.mean()
            norm = float(np.linalg.norm(ranks))
            if not np.isfinite(norm) or norm <= np.finfo(float).eps:
                continue
            role_values[role_index] = (ranks / norm).astype(np.float32)
            role_mask[role_index] = True
        if role_mask.any():
            kept_ids.append(str(model_id))
            embeddings.append(role_values)
            masks.append(role_mask)

    role_embeddings = (
        np.stack(embeddings)
        if embeddings
        else np.empty((0, len(CANONICAL_ROLES), grid_size), dtype=np.float32)
    )
    role_mask = (
        np.stack(masks)
        if masks
        else np.empty((0, len(CANONICAL_ROLES)), dtype=bool)
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        model_ids=np.asarray(kept_ids, dtype=str),
        role_embeddings=role_embeddings,
        role_mask=role_mask,
    )
    return np.asarray(kept_ids, dtype=str), role_embeddings, role_mask, cache_path


def combine_partial_role_caches(
    parts: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, Path]],
) -> PartialRoleCorpus:
    model_ids = np.concatenate([part[0].astype(str) for part in parts])
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("Partial-role sources contain overlapping model IDs")
    role_embeddings = np.concatenate([part[1] for part in parts], axis=0)
    role_mask = np.concatenate([part[2] for part in parts], axis=0)
    order = np.argsort(model_ids, kind="stable")
    return PartialRoleCorpus(
        model_ids=model_ids[order],
        role_embeddings=role_embeddings[order],
        role_mask=role_mask[order],
        cache_paths=tuple(part[3] for part in parts),
    )


def _role_mask_text(mask: np.ndarray) -> str:
    return "|".join(
        role for role, present in zip(CANONICAL_ROLES, mask, strict=True) if present
    )


def select_reliable_role_patterns(
    corpus: PartialRoleCorpus,
    pattern_metrics: pd.DataFrame,
    *,
    minimum_retention: float,
) -> PartialRoleCorpus:
    """Keep complete summaries and incomplete masks that pass masking calibration."""

    required = {
        "role_mask",
        "full_nearest_retained_in_top10",
    }
    missing = sorted(required.difference(pattern_metrics.columns))
    if missing:
        raise ValueError(f"pattern metrics are missing columns: {missing}")
    reliable = set(
        pattern_metrics.loc[
            pd.to_numeric(
                pattern_metrics["full_nearest_retained_in_top10"], errors="coerce"
            ).ge(minimum_retention),
            "role_mask",
        ].astype(str)
    )
    keep = np.asarray(
        [
            int(mask.sum()) == len(CANONICAL_ROLES)
            or _role_mask_text(mask) in reliable
            for mask in corpus.role_mask
        ],
        dtype=bool,
    )
    return PartialRoleCorpus(
        model_ids=corpus.model_ids[keep],
        role_embeddings=corpus.role_embeddings[keep],
        role_mask=corpus.role_mask[keep],
        cache_paths=corpus.cache_paths,
    )


def masking_stability(
    corpus: PartialRoleCorpus,
    *,
    seed: int,
    sample_size: int = 1_200,
    neighbor_count: int = 10,
    minimum_retention: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Choose role coverage using only neighbor stability after masking roles."""

    complete = np.flatnonzero(corpus.role_counts == len(CANONICAL_ROLES))
    if len(complete) <= neighbor_count:
        raise ValueError("Masking stability requires complete seven-role models")
    rng = np.random.default_rng(seed)
    sampled = np.sort(
        rng.choice(complete, size=min(sample_size, len(complete)), replace=False)
    )
    values = corpus.role_embeddings[sampled]
    full_similarity = np.einsum(
        "ard,brd->ab", values, values, optimize=True
    ) / float(len(CANONICAL_ROLES))
    np.fill_diagonal(full_similarity, -np.inf)
    k = min(neighbor_count, len(sampled) - 1)
    full_neighbors = np.argpartition(-full_similarity, k - 1, axis=1)[:, :k]
    full_nearest = np.argmax(full_similarity, axis=1)

    unique_masks, support = np.unique(corpus.role_mask, axis=0, return_counts=True)
    pattern_rows: list[dict[str, object]] = []
    for mask, count in zip(unique_masks, support, strict=True):
        role_count = int(mask.sum())
        if role_count == len(CANONICAL_ROLES) or role_count == 0:
            continue
        selected = np.flatnonzero(mask)
        similarity = np.einsum(
            "ard,brd->ab",
            values[:, selected],
            values[:, selected],
            optimize=True,
        ) / float(role_count)
        np.fill_diagonal(similarity, -np.inf)
        neighbors = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
        overlap = np.mean(
            [
                len(set(left).intersection(right)) / k
                for left, right in zip(full_neighbors, neighbors, strict=True)
            ]
        )
        nearest_retained = np.mean(
            [
                int(full_nearest[index]) in set(neighbors[index])
                for index in range(len(sampled))
            ]
        )
        pattern_rows.append(
            {
                "role_mask": _role_mask_text(mask),
                "role_count": role_count,
                "observed_models": int(count),
                "top10_neighbor_overlap": float(overlap),
                "full_nearest_retained_in_top10": float(nearest_retained),
                "metadata_used": False,
            }
        )
    patterns = pd.DataFrame(pattern_rows)

    threshold_rows: list[dict[str, object]] = []
    for minimum_roles in range(2, len(CANONICAL_ROLES) + 1):
        selected = patterns.loc[patterns["role_count"].ge(minimum_roles)]
        partial_support = int(selected["observed_models"].sum())
        if partial_support:
            weights = selected["observed_models"].to_numpy(float)
            retention = float(
                np.average(selected["full_nearest_retained_in_top10"], weights=weights)
            )
            overlap = float(
                np.average(selected["top10_neighbor_overlap"], weights=weights)
            )
        else:
            retention = 1.0
            overlap = 1.0
        threshold_rows.append(
            {
                "minimum_roles": minimum_roles,
                "eligible_models": int(np.count_nonzero(corpus.role_counts >= minimum_roles)),
                "eligible_partial_models": partial_support,
                "weighted_full_nearest_retained_in_top10": retention,
                "weighted_top10_neighbor_overlap": overlap,
                "required_retention": float(minimum_retention),
                "selection_uses_metadata": False,
            }
        )
    thresholds = pd.DataFrame(threshold_rows)
    eligible = thresholds.loc[
        thresholds["eligible_partial_models"].gt(0)
        & thresholds["weighted_full_nearest_retained_in_top10"].ge(minimum_retention)
    ]
    selected_minimum = (
        int(eligible["minimum_roles"].min())
        if not eligible.empty
        else len(CANONICAL_ROLES)
    )
    thresholds["selected"] = thresholds["minimum_roles"].eq(selected_minimum)
    return patterns, thresholds, selected_minimum


def partial_profile_split_table(
    corpus: PartialRoleCorpus,
    *,
    seed: int,
    n_folds: int,
) -> pd.DataFrame:
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    hashes = [
        hashlib.sha256(
            mask.tobytes() + values.astype(np.float32, copy=False).tobytes()
        ).hexdigest()
        for values, mask in zip(
            corpus.role_embeddings, corpus.role_mask, strict=True
        )
    ]
    folds = [
        int(hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()[:16], 16)
        % n_folds
        for value in hashes
    ]
    result = pd.DataFrame(
        {
            "model_id": corpus.model_ids,
            "profile_hash": hashes,
            "blind_fold": folds,
            "usable_role_count": corpus.role_counts,
            "usable_roles": [_role_mask_text(mask) for mask in corpus.role_mask],
        }
    )
    if result.groupby("profile_hash")["blind_fold"].nunique().max() != 1:
        raise RuntimeError("An exact partial-role summary was split across folds")
    return result


def pairwise_role_similarity(
    left: PartialRoleCorpus,
    right: PartialRoleCorpus,
    *,
    average_shared: bool = False,
    minimum_shared_roles: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    if minimum_shared_roles < 1:
        raise ValueError("minimum_shared_roles must be positive")
    numerator = np.zeros((len(left.model_ids), len(right.model_ids)), dtype=np.float32)
    for role_index in range(len(CANONICAL_ROLES)):
        numerator += (
            left.role_embeddings[:, role_index]
            @ right.role_embeddings[:, role_index].T
        )
    shared = left.role_mask.astype(np.int8) @ right.role_mask.astype(np.int8).T
    denominator = (
        np.maximum(shared, 1)
        if average_shared
        else np.full_like(shared, len(CANONICAL_ROLES))
    )
    similarity = numerator / denominator.astype(np.float32)
    similarity = np.clip(similarity, -1.0, 1.0)
    similarity[shared < minimum_shared_roles] = -np.inf
    return similarity, shared


def _cluster_candidates(n_models: int) -> list[int]:
    if n_models < 4:
        return [2]
    upper = max(2, int(np.ceil(np.sqrt(n_models))))
    count = max(4, int(np.ceil(np.log2(n_models))))
    return sorted(set(np.geomspace(2, upper, count).round().astype(int).tolist()))


def _search_matrix(corpus: PartialRoleCorpus) -> np.ndarray:
    matrix = corpus.role_embeddings.reshape(len(corpus.model_ids), -1).copy()
    scale = np.sqrt(
        len(CANONICAL_ROLES) / np.maximum(corpus.role_counts, 1)
    ).astype(np.float32)
    matrix *= scale[:, None]
    return matrix


def _role_centers(
    corpus: PartialRoleCorpus,
    labels: np.ndarray,
    n_clusters: int,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    centers = np.zeros(
        (n_clusters, len(CANONICAL_ROLES), corpus.grid_size), dtype=np.float32
    )
    global_centers = np.zeros((len(CANONICAL_ROLES), corpus.grid_size), dtype=np.float32)
    for role_index in range(len(CANONICAL_ROLES)):
        available = corpus.role_mask[:, role_index]
        mean = corpus.role_embeddings[available, role_index].mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm > np.finfo(float).eps:
            global_centers[role_index] = mean / norm
    for cluster in range(n_clusters):
        members = labels == cluster
        for role_index in range(len(CANONICAL_ROLES)):
            available = members & corpus.role_mask[:, role_index]
            if available.any():
                mean = corpus.role_embeddings[available, role_index].mean(axis=0)
                norm = float(np.linalg.norm(mean))
                if norm > np.finfo(float).eps:
                    centers[cluster, role_index] = mean / norm
                    continue
            if fallback is not None and np.linalg.norm(fallback[cluster, role_index]) > 0:
                centers[cluster, role_index] = fallback[cluster, role_index]
            else:
                centers[cluster, role_index] = global_centers[role_index]
    return centers


def extend_strict_clusters(
    strict_corpus_ids: Sequence[str],
    strict_assignments: pd.DataFrame,
    target: PartialRoleCorpus,
) -> pd.DataFrame:
    """Preserve strict clusters and attach partial summaries by observed-role similarity."""

    required = {
        "model_id",
        "predicted_cluster",
        "spectral_medoid_model",
    }
    missing = sorted(required.difference(strict_assignments.columns))
    if missing:
        raise ValueError(f"strict assignments are missing columns: {missing}")
    strict = strict_assignments.copy()
    strict["model_id"] = strict["model_id"].astype(str)
    if strict["model_id"].duplicated().any():
        raise ValueError("strict assignments contain duplicate model IDs")
    if set(map(str, strict_corpus_ids)) != set(strict["model_id"]):
        raise ValueError("strict corpus and strict assignments contain different models")

    positions = {model_id: index for index, model_id in enumerate(target.model_ids)}
    missing_core = sorted(set(strict["model_id"]).difference(positions))
    if missing_core:
        raise ValueError("target corpus does not contain every strict model")
    core_positions = np.asarray([positions[value] for value in strict["model_id"]], dtype=int)
    core_labels = pd.to_numeric(strict["predicted_cluster"], errors="raise").to_numpy(int)
    n_clusters = int(core_labels.max()) + 1
    core = _take(target, core_positions)
    centers = _role_centers(core, core_labels, n_clusters)

    scores = np.zeros((len(target.model_ids), n_clusters), dtype=np.float32)
    for role_index in range(len(CANONICAL_ROLES)):
        scores += target.role_embeddings[:, role_index] @ centers[:, role_index].T
    labels = np.argmax(scores, axis=1).astype(int)
    labels[core_positions] = core_labels
    center_similarity = scores[np.arange(len(labels)), labels] / np.maximum(
        target.role_counts, 1
    )

    medoids = (
        strict.groupby("predicted_cluster", sort=False)["spectral_medoid_model"]
        .first()
        .astype(str)
        .to_dict()
    )
    nearest_names = np.repeat("", len(target.model_ids)).astype(object)
    nearest_scores = np.full(len(target.model_ids), np.nan, dtype=float)
    nearest_shared = np.zeros(len(target.model_ids), dtype=int)
    core_ids = set(strict["model_id"])
    strict_by_id = strict.set_index("model_id")
    for position, model_id in enumerate(target.model_ids):
        if model_id not in strict_by_id.index:
            continue
        row = strict_by_id.loc[model_id]
        nearest_names[position] = str(row.get("nearest_neighbors", ""))
        nearest_scores[position] = pd.to_numeric(
            row.get("nearest_similarity", np.nan), errors="coerce"
        )
        nearest_shared[position] = len(CANONICAL_ROLES)

    for cluster in range(n_clusters):
        extension_positions = np.asarray(
            [
                position
                for position in np.flatnonzero(labels == cluster)
                if str(target.model_ids[position]) not in core_ids
            ],
            dtype=int,
        )
        if not len(extension_positions):
            continue
        reference_positions = np.flatnonzero(core_labels == cluster)
        query = _take(target, extension_positions)
        reference = _take(core, reference_positions)
        similarity, shared = pairwise_role_similarity(
            query,
            reference,
            average_shared=True,
            minimum_shared_roles=1,
        )
        count = min(2, len(reference_positions))
        candidates = np.argpartition(-similarity, count - 1, axis=1)[:, :count]
        candidate_scores = np.take_along_axis(similarity, candidates, axis=1)
        order = np.argsort(-candidate_scores, axis=1, kind="stable")
        candidates = np.take_along_axis(candidates, order, axis=1)
        for local_position, global_position in enumerate(extension_positions):
            selected = candidates[local_position]
            nearest_names[global_position] = ",".join(
                map(str, reference.model_ids[selected])
            )
            nearest_scores[global_position] = float(
                similarity[local_position, selected[0]]
            )
            nearest_shared[global_position] = int(
                shared[local_position, selected[0]]
            )

    return pd.DataFrame(
        {
            "model_id": target.model_ids,
            "predicted_cluster": labels,
            "spectral_medoid_model": [medoids[value] for value in labels],
            "similarity_to_cluster_mean": center_similarity,
            "nearest_neighbors": nearest_names,
            "nearest_similarity": nearest_scores,
            "nearest_shared_roles": nearest_shared,
            "usable_role_count": target.role_counts,
            "usable_roles": [_role_mask_text(mask) for mask in target.role_mask],
            "assignment_status": [
                "strict_core" if model_id in core_ids else "coverage_extension"
                for model_id in target.model_ids
            ],
        }
    ).sort_values("model_id", kind="stable", ignore_index=True)


def _fit_partial_clusters(
    corpus: PartialRoleCorpus,
    n_clusters: int,
    seed: int,
    *,
    max_iter: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    initializer = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=5,
        batch_size=min(
            len(corpus.model_ids),
            max(256, int(np.ceil(np.sqrt(len(corpus.model_ids)))) * 16),
        ),
        reassignment_ratio=0.01,
    ).fit(_search_matrix(corpus))
    labels = initializer.labels_.astype(int)
    centers = _role_centers(corpus, labels, n_clusters)
    for _ in range(max_iter):
        scores = np.zeros((len(corpus.model_ids), n_clusters), dtype=np.float32)
        for role_index in range(len(CANONICAL_ROLES)):
            scores += corpus.role_embeddings[:, role_index] @ centers[:, role_index].T
        updated = np.argmax(scores, axis=1).astype(int)
        missing = sorted(set(range(n_clusters)).difference(updated.tolist()))
        if missing:
            confidence = scores[np.arange(len(updated)), updated]
            for cluster in missing:
                donor = int(np.argmin(confidence))
                updated[donor] = cluster
                confidence[donor] = np.inf
        if np.array_equal(updated, labels):
            break
        previous = centers
        labels = updated
        centers = _role_centers(corpus, labels, n_clusters, fallback=previous)
    return labels, centers


def _sample_masked_distance(corpus: PartialRoleCorpus, indices: np.ndarray) -> np.ndarray:
    sample = PartialRoleCorpus(
        model_ids=corpus.model_ids[indices],
        role_embeddings=corpus.role_embeddings[indices],
        role_mask=corpus.role_mask[indices],
        cache_paths=corpus.cache_paths,
    )
    similarity, _ = pairwise_role_similarity(sample, sample, average_shared=True)
    distance = np.clip(1.0 - similarity, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    return distance


def discover_partial_clusters(
    corpus: PartialRoleCorpus,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cluster partial-role summaries using only observed role vectors."""

    if len(corpus.model_ids) < 4:
        raise ValueError("At least four coverage-aware summaries are required")
    sample_size = min(
        len(corpus.model_ids),
        max(500, int(np.ceil(np.sqrt(len(corpus.model_ids)))) * 20),
    )
    rng = np.random.default_rng(seed)
    sample_indices = np.sort(
        rng.choice(len(corpus.model_ids), size=sample_size, replace=False)
    )
    sample_distance = _sample_masked_distance(corpus, sample_indices)
    selection_rows: list[dict[str, object]] = []
    fitted: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for n_clusters in _cluster_candidates(len(corpus.model_ids)):
        labels, centers = _fit_partial_clusters(corpus, n_clusters, seed)
        fitted[n_clusters] = (labels, centers)
        selection_rows.append(
            {
                "n_clusters": n_clusters,
                "silhouette_cosine": float(
                    silhouette_score(
                        sample_distance,
                        labels[sample_indices],
                        metric="precomputed",
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
    labels, centers = fitted[selected_k]

    medoid_by_cluster: dict[int, str] = {}
    similarity_to_center = np.empty(len(corpus.model_ids), dtype=float)
    for cluster in range(selected_k):
        positions = np.flatnonzero(labels == cluster)
        values = corpus.role_embeddings[positions]
        scores = np.zeros(len(positions), dtype=np.float32)
        for role_index in range(len(CANONICAL_ROLES)):
            scores += values[:, role_index] @ centers[cluster, role_index]
        conservative = scores / float(len(CANONICAL_ROLES))
        similarity_to_center[positions] = conservative
        medoid_by_cluster[cluster] = str(
            corpus.model_ids[positions[int(np.argmax(conservative))]]
        )

    ordered_clusters = sorted(medoid_by_cluster, key=medoid_by_cluster.get)
    remap = {old: new for new, old in enumerate(ordered_clusters)}
    canonical_labels = np.asarray([remap[value] for value in labels], dtype=int)
    canonical_medoids = {
        remap[cluster]: model_id for cluster, model_id in medoid_by_cluster.items()
    }

    nearest_names = np.repeat("", len(corpus.model_ids)).astype(object)
    nearest_scores = np.full(len(corpus.model_ids), np.nan, dtype=float)
    nearest_shared = np.zeros(len(corpus.model_ids), dtype=int)
    for cluster in range(selected_k):
        positions = np.flatnonzero(canonical_labels == cluster)
        local = PartialRoleCorpus(
            model_ids=corpus.model_ids[positions],
            role_embeddings=corpus.role_embeddings[positions],
            role_mask=corpus.role_mask[positions],
            cache_paths=corpus.cache_paths,
        )
        similarity, shared = pairwise_role_similarity(local, local)
        np.fill_diagonal(similarity, -np.inf)
        count = min(2, max(len(positions) - 1, 0))
        if count == 0:
            continue
        candidates = np.argpartition(-similarity, count - 1, axis=1)[:, :count]
        candidate_scores = np.take_along_axis(similarity, candidates, axis=1)
        order = np.argsort(-candidate_scores, axis=1, kind="stable")
        candidates = np.take_along_axis(candidates, order, axis=1)
        for local_position, global_position in enumerate(positions):
            selected = candidates[local_position]
            nearest_names[global_position] = ",".join(
                map(str, corpus.model_ids[positions[selected]])
            )
            nearest_scores[global_position] = float(
                similarity[local_position, selected[0]]
            )
            nearest_shared[global_position] = int(
                shared[local_position, selected[0]]
            )

    assignments = pd.DataFrame(
        {
            "model_id": corpus.model_ids,
            "predicted_cluster": canonical_labels,
            "spectral_medoid_model": [
                canonical_medoids[value] for value in canonical_labels
            ],
            "similarity_to_cluster_mean": similarity_to_center,
            "nearest_neighbors": nearest_names,
            "nearest_similarity": nearest_scores,
            "nearest_shared_roles": nearest_shared,
            "usable_role_count": corpus.role_counts,
            "usable_roles": [_role_mask_text(mask) for mask in corpus.role_mask],
        }
    ).sort_values("model_id", kind="stable", ignore_index=True)

    stability: list[float] = []
    for offset in (1, 2, 3):
        repeated, _ = _fit_partial_clusters(corpus, selected_k, seed + offset)
        stability.append(float(adjusted_rand_score(labels, repeated)))
    selected_silhouette = float(
        selection.loc[
            selection["n_clusters"].eq(selected_k), "silhouette_cosine"
        ].iloc[0]
    )
    internal = pd.DataFrame(
        [
            {
                "method": "masked_role_kmeans_rank_summary",
                "n_models": len(corpus.model_ids),
                "n_clusters": selected_k,
                "selection_rule": "maximum_masked_cosine_silhouette",
                "silhouette_cosine": selected_silhouette,
                "davies_bouldin_euclidean": np.nan,
                "mean_seed_stability_ari": float(np.mean(stability)),
                "minimum_seed_stability_ari": float(np.min(stability)),
                "metadata_loaded_before_freeze": False,
            }
        ]
    )
    selection["selected"] = selection["n_clusters"].eq(selected_k)
    return assignments, selection, internal


def _weighted_partial_knn(
    train: PartialRoleCorpus,
    test: PartialRoleCorpus,
    train_y: np.ndarray,
    *,
    k: int,
    minimum_shared_roles: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    similarity, shared = pairwise_role_similarity(
        test,
        train,
        average_shared=True,
        minimum_shared_roles=minimum_shared_roles,
    )
    count = min(k, len(train.model_ids))
    neighbors = np.argpartition(-similarity, count - 1, axis=1)[:, :count]
    scores = np.take_along_axis(similarity, neighbors, axis=1)
    order = np.argsort(-scores, axis=1, kind="stable")
    neighbors = np.take_along_axis(neighbors, order, axis=1)
    scores = np.take_along_axis(similarity, neighbors, axis=1)
    shared_roles = np.take_along_axis(shared, neighbors, axis=1)
    valid = np.isfinite(scores)
    weights = np.where(
        valid,
        1.0 / (np.maximum(1.0 - np.where(valid, scores, 0.0), 0.0) + 1e-6),
        0.0,
    )
    values = train_y[neighbors]
    weight_sum = weights.sum(axis=1)
    prediction = np.full((len(test.model_ids), train_y.shape[1]), np.nan, dtype=float)
    has_neighbor = weight_sum > 0
    prediction[has_neighbor] = (
        (weights[has_neighbor, :, None] * values[has_neighbor]).sum(axis=1)
        / weight_sum[has_neighbor, None]
    )
    return prediction, neighbors, scores, shared_roles


def _take(corpus: PartialRoleCorpus, indices: np.ndarray) -> PartialRoleCorpus:
    return PartialRoleCorpus(
        model_ids=corpus.model_ids[indices],
        role_embeddings=corpus.role_embeddings[indices],
        role_mask=corpus.role_mask[indices],
        cache_paths=corpus.cache_paths,
    )


def cross_validated_partial_performance_freeze(
    corpus: PartialRoleCorpus,
    splits: pd.DataFrame,
    targets: pd.DataFrame,
    target_columns: Sequence[str],
    *,
    generation: str,
    k: int = 3,
    minimum_shared_roles: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Predict held-out scores from coverage-aware neighbors."""

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
    corpus_indices = np.asarray(
        [positions[value] for value in population["model_id"]], dtype=int
    )
    local_corpus = _take(corpus, corpus_indices)
    hash_by_model = population.set_index("model_id")["profile_hash"].astype(str)
    freeze_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []
    for fold in sorted(population["blind_fold"].unique()):
        test_mask = population["blind_fold"].eq(fold).to_numpy()
        train_mask = ~test_mask
        train_indices = np.flatnonzero(train_mask)
        test_indices = np.flatnonzero(test_mask)
        train_ids = population.loc[train_mask, "model_id"].astype(str).to_numpy()
        test_ids = population.loc[test_mask, "model_id"].astype(str).to_numpy()
        train_hashes = set(population.loc[train_mask, "profile_hash"].astype(str))
        prediction, neighbors, similarities, shared_roles = _weighted_partial_knn(
            _take(local_corpus, train_indices),
            _take(local_corpus, test_indices),
            values[train_indices],
            k=k,
            minimum_shared_roles=minimum_shared_roles,
        )
        baseline = np.repeat(
            values[train_indices].mean(axis=0)[None, :], len(test_ids), axis=0
        )
        for local_position, model_id in enumerate(test_ids):
            valid_neighbors = np.isfinite(similarities[local_position])
            neighbor_ids = train_ids[neighbors[local_position][valid_neighbors]]
            first_similarity = (
                float(similarities[local_position, 0])
                if valid_neighbors.any()
                else np.nan
            )
            first_shared = (
                int(shared_roles[local_position, 0]) if valid_neighbors.any() else 0
            )
            leakage_rows.append(
                {
                    "generation": generation,
                    "model_id": model_id,
                    "blind_fold": int(fold),
                    "nearest_train_similarity": first_similarity,
                    "nearest_shared_roles": first_shared,
                    "prediction_available": bool(valid_neighbors.any()),
                    "usable_role_count": int(
                        local_corpus.role_counts[test_indices[local_position]]
                    ),
                    "exact_profile_cross_fold": str(hash_by_model.at[model_id]) in train_hashes,
                    "shared_role_exact_match_cross_fold": bool(
                        np.isfinite(first_similarity) and first_similarity >= 1.0 - 1e-6
                    ),
                }
            )
            for target_index, target in enumerate(target_columns):
                freeze_rows.extend(
                    [
                        {
                            "generation": generation,
                            "model_id": model_id,
                            "profile_hash": hash_by_model.at[model_id],
                            "blind_fold": int(fold),
                            "target": target,
                            "feature_block": "coverage_aware_alpha_knn3",
                            "prediction": float(
                                prediction[local_position, target_index]
                            ),
                            "nearest_neighbors": ",".join(map(str, neighbor_ids)),
                            "nearest_similarity": first_similarity,
                            "nearest_shared_roles": first_shared,
                            "minimum_shared_roles": int(minimum_shared_roles),
                            "usable_role_count": int(
                                local_corpus.role_counts[test_indices[local_position]]
                            ),
                        },
                        {
                            "generation": generation,
                            "model_id": model_id,
                            "profile_hash": hash_by_model.at[model_id],
                            "blind_fold": int(fold),
                            "target": target,
                            "feature_block": "training_mean",
                            "prediction": float(baseline[local_position, target_index]),
                            "nearest_neighbors": "",
                            "nearest_similarity": np.nan,
                            "nearest_shared_roles": np.nan,
                            "minimum_shared_roles": np.nan,
                            "usable_role_count": int(
                                local_corpus.role_counts[test_indices[local_position]]
                            ),
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
    leakage = pd.DataFrame(leakage_rows).sort_values("model_id", ignore_index=True)
    return freeze, evaluated, leakage


__all__ = [
    "PartialRoleCorpus",
    "build_partial_role_cache",
    "combine_partial_role_caches",
    "cross_validated_partial_performance_freeze",
    "discover_partial_clusters",
    "extend_strict_clusters",
    "masking_stability",
    "pairwise_role_similarity",
    "partial_profile_split_table",
    "select_reliable_role_patterns",
]
