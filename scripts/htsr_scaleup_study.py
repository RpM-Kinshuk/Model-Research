from __future__ import annotations

"""Scale a retained HTSR study without mutating its source artifacts."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from spectral_research.study.contracts import canonical_model_id
from spectral_research.study.features import CANONICAL_ROLES, confirmatory_schema_population
from spectral_research.study.lineage import _signature_matrix, label_independent_cluster_count
from spectral_research.study.performance import (
    MODERN_TASKS,
    evaluate_external_transfer,
    evaluate_grouped_ridge,
    group_weighted_mae,
    grouped_outer_folds,
    holm_adjust,
    paired_component_bootstrap,
    paired_component_signflip_pvalue,
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


def alpha_oof(
    frame: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    targets: tuple[str, ...],
    *,
    method: str,
) -> pd.DataFrame:
    population = frame.loc[frame["model_id"].isin(embeddings)].copy()
    population = group_ready(population).sort_values("model_id", kind="stable").reset_index(drop=True)
    assert population["model_id"].is_unique
    values = population.loc[:, targets].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    assert np.isfinite(values).all()
    matrix = np.vstack([embeddings[model_id] for model_id in population["model_id"]])
    groups = population["ancestry_component"].astype(str).to_numpy()
    predictions = np.full_like(values, np.nan)
    fold_ids = np.full(len(population), -1, dtype=int)
    for fold in grouped_outer_folds(
        population,
        n_splits=OUTER_FOLDS,
        random_state=SEED,
        group_column="ancestry_component",
    ):
        predicted, _ = knn_predict(
            matrix[fold.train_indices],
            matrix[fold.test_indices],
            values[fold.train_indices],
            k=GROUP_K,
            train_groups=groups[fold.train_indices],
            distinct_groups=True,
        )
        predictions[fold.test_indices] = predicted
        fold_ids[fold.test_indices] = fold.fold_id
        assert set(groups[fold.train_indices]).isdisjoint(set(groups[fold.test_indices]))
    assert np.isfinite(predictions).all() and np.all(fold_ids >= 0)
    rows: list[dict[str, object]] = []
    for target_index, target in enumerate(targets):
        for row_index, row in population.iterrows():
            rows.append(
                {
                    "model_id": str(row["model_id"]),
                    "ancestry_component": str(row["ancestry_component"]),
                    "source": str(row.get("source", "")),
                    "target": target,
                    "feature_block": method,
                    "outer_fold": int(fold_ids[row_index]),
                    "actual": float(values[row_index, target_index]),
                    "prediction": float(predictions[row_index, target_index]),
                }
            )
    return pd.DataFrame(rows)


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


def residual_hybrid_on_ridge_folds(
    population: pd.DataFrame,
    ridge_predictions: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    *,
    target: str,
    method: str = "M+S+H+alpha_residual_k5_nested",
) -> pd.DataFrame:
    baseline = ridge_predictions.loc[
        ridge_predictions["target"].eq(target)
        & ridge_predictions["feature_block"].eq("M+S+H")
    ].copy()
    ids = set(baseline["model_id"].astype(str))
    frame = population.loc[
        population["model_id"].astype(str).isin(ids)
        & population["model_id"].isin(embeddings)
    ].copy()
    frame = frame.sort_values("model_id", kind="stable").reset_index(drop=True)
    assert set(frame["model_id"].astype(str)) == ids
    baseline_by_model = baseline.set_index("model_id")
    rows: list[dict[str, object]] = []
    for fold_id in sorted(baseline["outer_fold"].unique()):
        test_ids = set(
            baseline.loc[baseline["outer_fold"].eq(fold_id), "model_id"].astype(str)
        )
        train = frame.loc[~frame["model_id"].astype(str).isin(test_ids)].reset_index(drop=True)
        test = frame.loc[frame["model_id"].astype(str).isin(test_ids)].reset_index(drop=True)
        assert set(train["ancestry_component"].astype(str)).isdisjoint(
            set(test["ancestry_component"].astype(str))
        )
        inner_predictions, _, _ = evaluate_grouped_ridge(
            train,
            target=target,
            outer_splits=INNER_FOLDS,
            inner_splits=INNER_FOLDS,
            ridge_alphas=RIDGE_ALPHAS,
            random_state=SEED + int(fold_id),
        )
        inner = inner_predictions.loc[
            inner_predictions["feature_block"].eq("M+S+H")
        ].set_index("model_id")
        assert set(inner.index.astype(str)) == set(train["model_id"].astype(str))
        residuals = np.asarray(
            [float(inner.at[model_id, "actual"] - inner.at[model_id, "prediction"]) for model_id in train["model_id"]],
            dtype=float,
        )
        train_embeddings = np.vstack([embeddings[model_id] for model_id in train["model_id"]])
        test_embeddings = np.vstack([embeddings[model_id] for model_id in test["model_id"]])
        correction, _ = knn_predict(
            train_embeddings,
            test_embeddings,
            residuals,
            k=GROUP_K,
            train_groups=train["ancestry_component"].astype(str).to_numpy(),
            distinct_groups=True,
        )
        for index, row in test.iterrows():
            model_id = str(row["model_id"])
            rows.append(
                {
                    "model_id": model_id,
                    "ancestry_component": str(row["ancestry_component"]),
                    "target": target,
                    "feature_block": method,
                    "outer_fold": int(fold_id),
                    "actual": float(row[target]),
                    "prediction": float(baseline_by_model.at[model_id, "prediction"] + correction[index]),
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
    census["htsr_artifact_available"] = census["htsr_artifact_available"].fillna(False)
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


def build_exact_modern_populations(
    manifest: pd.DataFrame,
    dev_features: pd.DataFrame,
    ext_features: pd.DataFrame,
    dev_targets: pd.DataFrame,
    ext_targets: pd.DataFrame,
    frozen: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    graph_columns = [
        "model_id", "ancestry_component", "component_provenance", "component_edge_count",
        "component_graph_fingerprint", "development_artifact_available",
    ]
    populations: list[pd.DataFrame] = []
    for scope, features, targets in (
        ("development", dev_features, dev_targets),
        ("external", ext_features, ext_targets),
    ):
        ids = set(frozen.loc[frozen["evaluation_scope"].eq(scope), "model_id"].astype(str))
        population = (
            features.loc[features["model_id"].astype(str).isin(ids)]
            .merge(targets, on="model_id", how="inner", validate="one_to_one")
            .merge(manifest.loc[:, graph_columns], on="model_id", how="inner", validate="one_to_one")
            .sort_values("model_id", kind="stable", ignore_index=True)
        )
        assert set(population["model_id"].astype(str)) == ids
        population["source"] = "phase2" if scope == "development" else "ablation"
        populations.append(population)
    return populations[0], populations[1]


def source_heldout_modern(
    dev: pd.DataFrame,
    ext: pd.DataFrame,
    manifest: pd.DataFrame,
    frozen: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    broad_dev_targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = manifest.loc[manifest["development_artifact_available"].fillna(False).astype(bool)]
    prediction_frames: list[pd.DataFrame] = []
    dev_alpha = dev.loc[dev["model_id"].isin(embeddings)].reset_index(drop=True)
    ext_alpha = ext.loc[ext["model_id"].isin(embeddings)].reset_index(drop=True)
    dev_matrix = np.vstack([embeddings[model_id] for model_id in dev_alpha["model_id"]])
    ext_matrix = np.vstack([embeddings[model_id] for model_id in ext_alpha["model_id"]])
    dev_groups = dev_alpha["ancestry_component"].astype(str).to_numpy()
    broad_train = broad_dev_targets.loc[broad_dev_targets["model_id"].isin(embeddings)].copy()
    broad_train = broad_train.sort_values("model_id", kind="stable").reset_index(drop=True)
    broad_matrix = np.vstack([embeddings[model_id] for model_id in broad_train["model_id"]])
    broad_groups = broad_train["ancestry_component"].fillna("").astype(str).to_numpy()
    broad_groups = np.asarray(
        [group if group else f"unknown::{model_id}" for group, model_id in zip(broad_groups, broad_train["model_id"])],
        dtype=object,
    )

    for target in MODERN_TARGETS:
        ridge_predictions, _, _ = evaluate_external_transfer(
            dev,
            ext,
            target=target,
            development_component_universe=universe,
            inner_splits=INNER_FOLDS,
            ridge_alphas=RIDGE_ALPHAS,
            random_state=SEED,
        )
        prediction_frames.append(ridge_predictions)
        slice_by_model = ridge_predictions.loc[
            ridge_predictions["feature_block"].eq("M+S+H"),
            ["model_id", "external_slice"],
        ].set_index("model_id")["external_slice"]
        dev_values = dev_alpha[target].to_numpy(float)
        strict_alpha, _ = knn_predict(
            dev_matrix, ext_matrix, dev_values, k=GROUP_K,
            train_groups=dev_groups, distinct_groups=True,
        )
        broad_values = broad_train[target].to_numpy(float)
        broad_alpha, _ = knn_predict(
            broad_matrix, ext_matrix, broad_values, k=GROUP_K,
            train_groups=broad_groups, distinct_groups=True,
        )
        frozen_dev = frozen.loc[
            frozen["evaluation_scope"].eq("development")
            & frozen["target"].eq(target)
            & frozen["feature_block"].eq("M+S+H")
        ].set_index("model_id")
        residuals = np.asarray(
            [float(frozen_dev.at[model_id, "actual"] - frozen_dev.at[model_id, "prediction"]) for model_id in dev_alpha["model_id"]],
            dtype=float,
        )
        residual_prediction, _ = knn_predict(
            dev_matrix, ext_matrix, residuals, k=GROUP_K,
            train_groups=dev_groups, distinct_groups=True,
        )
        baseline = ridge_predictions.loc[ridge_predictions["feature_block"].eq("M+S+H")].set_index("model_id")
        for row_index, row in ext_alpha.iterrows():
            model_id = str(row["model_id"])
            common = {
                "model_id": model_id,
                "ancestry_component": str(row["ancestry_component"]),
                "target": target,
                "actual": float(row[target]),
                "external_slice": str(slice_by_model.at[model_id]),
            }
            for method, prediction in (
                ("alpha_profile_k5_strict_train", strict_alpha[row_index]),
                ("alpha_profile_k5_all_labeled_train", broad_alpha[row_index]),
                (
                    "M+S+H+alpha_residual_k5",
                    float(baseline.at[model_id, "prediction"]) + residual_prediction[row_index],
                ),
            ):
                prediction_frames.append(pd.DataFrame([{**common, "feature_block": method, "prediction": float(prediction)}]))
    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    metrics = metric_rows(predictions, scope="source_heldout_modern")
    increments = increment_rows(
        predictions,
        baseline="M+S+H",
        candidates=(
            "alpha_profile_k5_strict_train",
            "alpha_profile_k5_all_labeled_train",
            "M+S+H+alpha_residual_k5",
        ),
        secondary_targets=MODERN_TASKS,
        scope="source_heldout_modern",
    )
    if EXTERNAL_EVIDENCE_ROLE != "prospective_confirmatory":
        increments["inferential"] = False
        increments["inference_reason"] = EXTERNAL_EVIDENCE_ROLE
    return predictions, metrics, increments


def build_legacy_targets(raw: pd.DataFrame) -> pd.DataFrame:
    working = raw.loc[:, ["model_id", *LEGACY_TASKS]].copy()
    working["model_id"] = working["model_id"].map(canonical)
    for target in LEGACY_TASKS:
        working[target] = pd.to_numeric(working[target], errors="coerce")
        working.loc[~working[target].between(0.0, 100.0), target] = np.nan
    working["legacy_average"] = working.loc[:, LEGACY_TASKS].mean(axis=1, skipna=False)
    working = working.loc[working["model_id"].ne("")].drop_duplicates("model_id", keep=False)
    return working.sort_values("model_id", kind="stable", ignore_index=True)


def legacy_study(
    legacy_targets: pd.DataFrame,
    ext_features: pd.DataFrame,
    manifest: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    graph_columns = [
        "model_id", "ancestry_component", "component_provenance", "component_edge_count",
    ]
    broad = legacy_targets.merge(manifest.loc[:, graph_columns], on="model_id", how="left", validate="one_to_one")
    broad["source"] = "ablation"
    complete = broad.loc[broad.loc[:, LEGACY_TARGETS].notna().all(axis=1)].copy()
    paper = paper_random_splits(complete, embeddings, LEGACY_TARGETS)
    grouped = alpha_oof(complete, embeddings, LEGACY_TARGETS, method="alpha_profile_k5_component_distinct")

    strict_features, _ = confirmatory_schema_population(
        ext_features, min_role_coverage=BROAD_COVERAGE
    )
    strict = (
        strict_features.merge(legacy_targets, on="model_id", how="inner", validate="one_to_one")
        .merge(manifest.loc[:, graph_columns], on="model_id", how="left", validate="one_to_one")
    )
    ridge_frames: list[pd.DataFrame] = []
    for target in LEGACY_TARGETS:
        target_population = strict.loc[
            np.isfinite(pd.to_numeric(strict[target], errors="coerce"))
            & strict["model_id"].isin(embeddings)
        ].copy()
        ridge_predictions, _, _ = evaluate_grouped_ridge(
            target_population,
            target=target,
            outer_splits=OUTER_FOLDS,
            inner_splits=INNER_FOLDS,
            ridge_alphas=RIDGE_ALPHAS,
            random_state=SEED,
        )
        alpha_predictions = alpha_on_ridge_folds(
            target_population, ridge_predictions, embeddings, target=target
        )
        ridge_frames.extend((ridge_predictions, alpha_predictions))
        if target == "legacy_average":
            ridge_frames.append(
                residual_hybrid_on_ridge_folds(
                    target_population,
                    ridge_predictions,
                    embeddings,
                    target=target,
                )
            )
    paired = pd.concat(ridge_frames, ignore_index=True, sort=False)
    paired_metrics = metric_rows(paired, scope="legacy_grouped_paired")
    increments = increment_rows(
        paired,
        baseline="M+S+H",
        candidates=(
            "alpha_profile_k5_component_distinct",
            "M+S+H+alpha_residual_k5_nested",
        ),
        secondary_targets=LEGACY_TASKS,
        scope="legacy_grouped_paired",
    )
    grouped_metrics = metric_rows(grouped, scope="legacy_grouped_alpha_broad")
    return paper, grouped, grouped_metrics, paired, paired_metrics, increments


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
    manifest = pd.read_csv(RUN / "manifest.csv", low_memory=False)
    dev_features = pd.read_csv(RUN / "feature_table.csv", low_memory=False)
    ext_features = pd.read_csv(RUN / "external_feature_table.csv", low_memory=False)
    dev_targets = pd.read_csv(RUN / "development_targets.csv", low_memory=False)
    ext_targets = pd.read_csv(RUN / "external_targets.csv", low_memory=False)
    frozen = pd.read_csv(RUN / "performance_predictions.csv", low_memory=False)
    phase2 = pd.read_csv(phase2_path, low_memory=False)
    phase2["model_id"] = phase2["model_id"].map(canonical)
    phase2 = phase2.loc[phase2["model_id"].ne("")].drop_duplicates("model_id", keep=False)
    assert phase2["model_id"].is_unique
    ablation = pd.read_csv(ablation_path, low_memory=False)
    ablation["model_id"] = ablation["model_id"].map(canonical)
    legacy_targets = build_legacy_targets(ablation)

    external_cache = Path(str(provenance["cache"]["external"]))
    requested_dev = set(dev_targets["model_id"].astype(str))
    requested_ext = set(ext_targets["model_id"].astype(str)) | set(legacy_targets["model_id"].astype(str))
    dev_ids, dev_embedding, dev_cache = alpha_embedding_cache(
        RUN / "alpha_trajectories.csv",
        requested_dev,
        source_digest=str(provenance["sources"]["development_stats"]["sha256"]),
        source_name="phase2",
    )
    ext_ids, ext_embedding, ext_cache = alpha_embedding_cache(
        external_cache / "alpha_trajectories.csv",
        requested_ext,
        source_digest=str(provenance["sources"]["external_stats"]["sha256"]),
        source_name="ablation",
    )
    embeddings = embedding_lookup(dev_ids, dev_embedding, ext_ids, ext_embedding)

    census, broad_clusters, broad_cluster_metrics = broad_lineage(phase2, dev_features, manifest)

    dev, ext = build_exact_modern_populations(
        manifest, dev_features, ext_features, dev_targets, ext_targets, frozen
    )
    broad_graph_columns = [
        "model_id", "ancestry_component", "component_provenance", "component_edge_count"
    ]
    broad_dev_targets = dev_targets.merge(
        manifest.loc[:, broad_graph_columns], on="model_id", how="left", validate="one_to_one"
    )
    broad_dev_targets["source"] = "phase2"
    broad_ext_targets = ext_targets.merge(
        manifest.loc[:, broad_graph_columns], on="model_id", how="left", validate="one_to_one"
    )
    broad_ext_targets["source"] = "ablation"
    modern_broad = pd.concat([broad_dev_targets, broad_ext_targets], ignore_index=True)
    modern_grouped_alpha = alpha_oof(
        modern_broad, embeddings, MODERN_TARGETS,
        method="alpha_profile_k5_component_distinct",
    )
    modern_grouped_metrics = metric_rows(
        modern_grouped_alpha, scope="modern_grouped_alpha_broad"
    )
    modern_paper = paper_random_splits(modern_broad, embeddings, MODERN_TARGETS)

    source_predictions, source_metrics, source_increments = source_heldout_modern(
        dev, ext, manifest, frozen, embeddings, broad_dev_targets
    )

    pooled = pd.concat([dev, ext], ignore_index=True, sort=False)
    pooled = pooled.loc[pooled["model_id"].isin(embeddings)].reset_index(drop=True)
    pooled_ridge, _, _ = evaluate_grouped_ridge(
        pooled,
        target="modern_average",
        outer_splits=OUTER_FOLDS,
        inner_splits=INNER_FOLDS,
        ridge_alphas=RIDGE_ALPHAS,
        random_state=SEED,
    )
    pooled_alpha = alpha_on_ridge_folds(
        pooled, pooled_ridge, embeddings, target="modern_average"
    )
    pooled_hybrid = residual_hybrid_on_ridge_folds(
        pooled, pooled_ridge, embeddings, target="modern_average"
    )
    pooled_predictions = pd.concat(
        [pooled_ridge, pooled_alpha, pooled_hybrid], ignore_index=True
    )
    pooled_metrics = metric_rows(pooled_predictions, scope="modern_pooled_grouped_paired")
    pooled_increments = increment_rows(
        pooled_predictions,
        baseline="M+S+H",
        candidates=(
            "alpha_profile_k5_component_distinct",
            "M+S+H+alpha_residual_k5_nested",
        ),
        secondary_targets=(),
        scope="modern_pooled_grouped_paired",
    )

    (
        legacy_paper,
        legacy_grouped,
        legacy_grouped_metrics,
        legacy_paired,
        legacy_paired_metrics,
        legacy_increments,
    ) = legacy_study(legacy_targets, ext_features, manifest, embeddings)

    run_payload = {
        "method": "htsr_scaleup_rolewise_interp32_rank_cosine_k5_v2",
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
        "silhouette_sample": SILHOUETTE_SAMPLE,
        "family_router": "spectral_shape_full_gallery_manual_triage_only",
        "development_stats_sha256": provenance["sources"]["development_stats"]["sha256"],
        "external_stats_sha256": provenance["sources"]["external_stats"]["sha256"],
        "development_target_sha256": provenance["sources"]["development_targets"]["sha256"],
        "external_target_sha256": provenance["sources"]["external_targets"]["sha256"],
    }
    run_id = "scaleup_" + hashlib.sha256(
        json.dumps(run_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    destination = SCALE_ROOT / "runs" / run_id
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError(f"scale-up run already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=destination.parent))
    tables = {
        "population_ledger.csv": census,
        "lineage_broad_cluster_assignments.csv": broad_clusters,
        "lineage_broad_cluster_metrics.csv": broad_cluster_metrics,
        "modern_paper_like_summary.csv": modern_paper,
        "modern_grouped_alpha_predictions.csv": modern_grouped_alpha,
        "modern_grouped_alpha_metrics.csv": modern_grouped_metrics,
        "modern_source_heldout_predictions.csv": source_predictions,
        "modern_source_heldout_metrics.csv": source_metrics,
        "modern_source_heldout_increments.csv": source_increments,
        "modern_pooled_predictions.csv": pooled_predictions,
        "modern_pooled_metrics.csv": pooled_metrics,
        "modern_pooled_increments.csv": pooled_increments,
        "legacy_paper_like_summary.csv": legacy_paper,
        "legacy_grouped_alpha_predictions.csv": legacy_grouped,
        "legacy_grouped_alpha_metrics.csv": legacy_grouped_metrics,
        "legacy_paired_predictions.csv": legacy_paired,
        "legacy_paired_metrics.csv": legacy_paired_metrics,
        "legacy_increments.csv": legacy_increments,
    }
    for filename, frame in tables.items():
        frame.to_csv(temporary / filename, index=False)
    counts = {
        "phase2_census": len(census),
        "phase2_htsr_artifacts": int(census["htsr_artifact_available"].sum()),
        "phase2_strict_lineage": int(census["strict_cluster"].notna().sum()),
        "phase2_broad_lineage": int(census["broad_cluster"].notna().sum()),
        "modern_alpha_profiles": len(set(modern_broad["model_id"]) & set(embeddings)),
        "modern_grouped_models": modern_grouped_alpha["model_id"].nunique(),
        "modern_paired_development": len(dev),
        "modern_external_ridge_models": len(ext),
        "modern_external_alpha_paired_models": int(ext["model_id"].isin(embeddings).sum()),
        "legacy_alpha_complete": int(
            legacy_targets.loc[legacy_targets.loc[:, LEGACY_TARGETS].notna().all(axis=1), "model_id"].isin(embeddings).sum()
        ),
        "legacy_grouped_models": legacy_grouped["model_id"].nunique(),
        "legacy_paired_models_min": int(
            legacy_paired.groupby("target")["model_id"].nunique().min()
        ),
        "legacy_paired_models_max": int(
            legacy_paired.groupby("target")["model_id"].nunique().max()
        ),
    }
    metadata = {
        "run_id": run_id,
        "config": run_payload,
        "counts": counts,
        "caches": {"phase2_alpha": str(dev_cache), "ablation_alpha": str(ext_cache)},
        "evidence_roles": {
            "modern_source_heldout": EXTERNAL_EVIDENCE_ROLE,
            "modern_pooled": "internal_grouped",
            "legacy": "internal_grouped_and_interpolation_diagnostic",
        },
    }
    (temporary / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = [
        "# HTSR Scale-Up Study",
        "",
        f"Run: `{run_id}`",
        "",
        "## Population",
        "",
        *[f"- {key}: {value}" for key, value in counts.items()],
        "",
        "## Contracts",
        "",
        "- Modern and legacy benchmark generations are analyzed separately.",
        "- Grouped results hold graph-derived ancestry components out intact.",
        "- Paper-like random splits are interpolation diagnostics only.",
        "- Strict lineage is the quality-controlled descriptive reference.",
        "- Broad clusters are population sensitivity only; family routes are manual-triage candidates.",
        f"- Source-heldout evidence role: {EXTERNAL_EVIDENCE_ROLE}.",
    ]
    (temporary / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    os.rename(temporary, destination)
    print(json.dumps({"run": str(destination), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
