"""Reassess saved HTSR measurements without downloading or changing checkpoints."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_research.study.artifacts import (
    validate_artifact_manifest,
    write_artifact_manifest,
)
from spectral_research.study.contracts import canonical_model_id
from spectral_research.study.partial_roles import PartialRoleCorpus, pairwise_role_similarity
from spectral_research.study.weight_only import validate_clusters


QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
SEED = 17
FOLDS = 5
METHODS = (
    "training_mean", "training_median", "rank_knn3",
    "role_alpha_ridge", "role_spectral_ridge", "role_spectral_trees",
    "all_matrix_alpha_trees", "all_matrix_spectral_trees",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def summarize_file(path: Path) -> dict:
    raw = path.read_bytes()
    frame = pd.read_csv(BytesIO(raw), low_memory=False)
    ids = frame["model_id"].dropna().map(canonical_model_id).dropna().unique()
    if len(ids) != 1:
        raise ValueError(f"Expected one internal model ID: {path}")
    numeric = frame.apply(lambda s: pd.to_numeric(s, errors="coerce"))
    # The final CSV row can be a model aggregate rather than a weight matrix.
    mask = frame["longname"].notna() & numeric["M"].gt(0) & numeric["N"].gt(0)
    numeric = numeric.loc[mask]
    alpha = numeric["alpha"].where(numeric["alpha"].gt(1))
    norm = numeric["norm"].where(numeric["norm"].gt(0))
    spectral = numeric["spectral_norm"].where(numeric["spectral_norm"].gt(0))
    log_spectral = np.log10(spectral)
    quantities = {
        "alpha": alpha,
        "alpha_weighted": alpha * log_spectral,
        "log_spectral": log_spectral,
        "log_rms": 0.5 * np.log10(norm / (numeric["M"] * numeric["N"])),
        "stable_rank_fraction": (norm / spectral) / numeric[["M", "N"]].min(axis=1),
        "entropy": numeric["entropy"].where(numeric["entropy"].between(0, 1)),
    }
    result = {
        "model_id": ids[0], "artifact_path": str(path), "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "matrix_count": len(numeric), "alpha_count": int(np.isfinite(alpha).sum()),
        "alpha_failure_fraction": float((~np.isfinite(alpha)).mean()),
        "poor_fit_fraction": float(numeric["D"].gt(0.1).mean()),
    }
    for name, values in quantities.items():
        finite = np.asarray(values)[np.isfinite(values)]
        for q in QUANTILES:
            result[f"all__{name}__q{int(q * 100)}"] = float(np.quantile(finite, q)) if len(finite) else np.nan
    return result


def all_matrix_features(provenance: dict, cache_dir: Path, workers: int) -> pd.DataFrame:
    paths = sorted(p for key in ("development_stats", "external_stats")
                   for p in Path(provenance["sources"][key]["path"]).glob("*.csv"))
    stamp = hashlib.sha256(json.dumps([
        (str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths
    ]).encode() + inspect.getsource(summarize_file).encode()
        + repr(QUANTILES).encode()).hexdigest()[:20]
    cache = cache_dir / f"all_matrices_{stamp}.csv"
    if cache.exists():
        return pd.read_csv(cache, low_memory=False)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, row in enumerate(executor.map(summarize_file, paths), 1):
            rows.append(row)
            if index % 1500 == 0:
                print(f"matrix summaries: {index}/{len(paths)}", flush=True)
    frame = pd.DataFrame(rows).sort_values("model_id").reset_index(drop=True)
    if frame["model_id"].duplicated().any():
        raise ValueError("An internal model ID appears in more than one stats file")
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)
    return frame


def load_partial(metadata: dict) -> PartialRoleCorpus:
    parts = [np.load(p, allow_pickle=False) for p in metadata["caches"]["coverage_aware"]]
    ids = np.concatenate([p["model_ids"].astype(str) for p in parts])
    order = np.argsort(ids)
    return PartialRoleCorpus(ids[order], np.concatenate([p["role_embeddings"] for p in parts])[order],
                             np.concatenate([p["role_mask"] for p in parts])[order].astype(bool), ())


def balanced_group_folds(groups: np.ndarray) -> np.ndarray:
    labels, sizes = np.unique(groups, return_counts=True)
    tie = [hashlib.sha256(f"{SEED}:{v}".encode()).hexdigest() for v in labels]
    order = np.lexsort((tie, -sizes))
    loads = np.zeros(FOLDS, dtype=int)
    assignments = {}
    for pos in order:
        fold = int(np.argmin(loads))
        assignments[labels[pos]] = fold
        loads[fold] += sizes[pos]
    return np.asarray([assignments[v] for v in groups])


def signal_columns(frame: pd.DataFrame) -> dict[str, list[str]]:
    alpha = [c for c in frame if c.startswith("H__") and "__alpha__" in c and not c.endswith("missing_fraction")]
    spectral = [c for c in frame if c.startswith(("H__", "S__")) and not c.endswith("missing_fraction")]
    return {
        "role_alpha_ridge": alpha,
        "role_spectral_ridge": spectral,
        "role_spectral_trees": spectral,
        "all_matrix_alpha_trees": [c for c in frame if c.startswith("all__alpha__")],
        "all_matrix_spectral_trees": [c for c in frame if c.startswith("all__")],
        "matrix_shape_trees": [c for c in frame if c.startswith("M__") and not c.endswith("missing_fraction")],
        "nonalpha_spectral_trees": [c for c in spectral if c.startswith("S__")],
        "nonalpha_spectral_ridge": [c for c in spectral if c.startswith("S__")],
    }


def fit_predict(method: str, train: np.ndarray, test: np.ndarray, y: np.ndarray) -> np.ndarray:
    if method.endswith("ridge"):
        model = make_pipeline(SimpleImputer(add_indicator=True), StandardScaler(), Ridge(alpha=100.0))
    else:
        model = HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15,
            min_samples_leaf=20, learning_rate=0.05, l2_regularization=10.0,
            early_stopping=False, random_state=SEED)
    model.fit(train, y)
    return model.predict(test)


def knn(similarity: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = min(3, len(y))
    positions = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
    scores = np.take_along_axis(similarity, positions, axis=1)
    valid = np.isfinite(scores)
    weights = np.where(valid, 1.0 / (np.maximum(1.0 - scores, 0) + 1e-6), 0)
    total = weights.sum(axis=1)
    prediction = np.divide((weights * y[positions]).sum(axis=1), total,
                           out=np.full(len(similarity), np.nan), where=total > 0)
    return prediction, scores[:, 0]


def metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, all_rows in predictions.groupby(["population", "generation", "split", "method"], sort=False):
        frame = all_rows.dropna(subset=["prediction"])
        errors = np.abs(frame["actual"] - frame["prediction"])
        rows.append(dict(zip(("population", "generation", "split", "method"), keys),
            n_models=len(frame), n_eligible=len(all_rows), coverage=len(frame)/len(all_rows),
            mae=float(errors.mean()), median_error=float(errors.median()),
            spearman=float(spearmanr(frame["actual"], frame["prediction"]).statistic),
            rmse=float(np.sqrt(np.mean((frame["actual"]-frame["prediction"])**2)))))
    return pd.DataFrame(rows)


def evaluate_generation(generation: str, targets: pd.DataFrame, features: pd.DataFrame,
                        corpus: PartialRoleCorpus, run: Path, columns: dict,
                        ) -> tuple[list[pd.DataFrame], list[dict], list[pd.DataFrame]]:
    target = generation + "_average"
    targets = targets.copy()
    targets["model_id"] = targets["model_id"].map(canonical_model_id)
    if generation == "modern":
        targets = targets.loc[targets["target_resolution"].isin(
            ("single_or_identical_score_vector", "selected_source_row"))]
    population = features.merge(targets[["model_id", target, "source_population"]], on="model_id", validate="one_to_one")
    population = population.loc[np.isfinite(population[target]) & population["alpha_count"].ge(2)].reset_index(drop=True)
    splits = pd.read_csv(run/"performance_frozen_splits.csv")
    cluster_splits = pd.read_csv(run/"performance_cluster_holdout_splits.csv")
    population = population.merge(splits[["model_id", "profile_hash", "blind_fold"]], on="model_id", how="left")
    rows, audits, split_tables = [], [], []
    for scope in ("matched_role_population", "all_matrix_population"):
        table = population.loc[population["blind_fold"].notna()].copy() if scope.startswith("matched") else population.copy()
        table = table.sort_values("model_id").reset_index(drop=True)
        y = table[target].to_numpy()
        ids = table["model_id"].to_numpy()
        if scope.startswith("matched"):
            positions = pd.Index(corpus.model_ids).get_indexer(ids)
            subset = PartialRoleCorpus(ids, corpus.role_embeddings[positions], corpus.role_mask[positions], ())
            similarity, shared = pairwise_role_similarity(subset, subset, average_shared=True, minimum_shared_roles=4)
            # Keep observationally indistinguishable pairs together even when their masks differ.
            adjacency = np.isfinite(similarity) & (similarity >= 1.0 - 1e-6)
            n_components, exact_groups = connected_components(csr_matrix(adjacency), directed=False)
            near_adjacency = np.isfinite(similarity) & (similarity >= 0.995)
            _, near_groups = connected_components(csr_matrix(near_adjacency), directed=False)
            original = table["blind_fold"].to_numpy(int)
            cross = adjacency & (original[:, None] != original[None, :])
            audits.append({"generation": generation, "n_models": len(ids),
                "shared_identical_pairs_crossing_old_folds": int(np.triu(cross, 1).sum()),
                "models_with_shared_identical_cross_fold_neighbor": int(cross.any(axis=1).sum()),
                "exact_similarity_components": int(n_components),
                "near_similarity_components": int(len(np.unique(near_groups))),
                "largest_near_component": int(np.bincount(near_groups).max())})
            scheme = {
                "original_hash": (original, table["profile_hash"].to_numpy()),
                "shared_exact_group": (balanced_group_folds(exact_groups), exact_groups),
                "near_group_0995": (balanced_group_folds(near_groups), near_groups),
                "whole_weight_cluster": (cluster_splits.set_index("model_id").loc[ids,"blind_fold"].to_numpy(),
                                         cluster_splits.set_index("model_id").loc[ids,"predicted_cluster"].to_numpy()),
            }
            np.fill_diagonal(similarity, -np.inf)
            active_methods = METHODS
        else:
            hashes = table.filter(regex=r"^all__").round(6).apply(
                lambda r: hashlib.sha256(r.to_numpy(np.float64).tobytes()).hexdigest(), axis=1).to_numpy()
            scheme = {"all_summary_group": (balanced_group_folds(hashes), hashes)}
            similarity = None
            active_methods = ("training_mean", "training_median", "all_matrix_alpha_trees", "all_matrix_spectral_trees")
        source = table["source_population"].astype(str).to_numpy()
        source_names = np.unique(source)
        if len(source_names) == 2:
            scheme["collection_holdout"] = (np.searchsorted(source_names, source), source)
        for split_name, (folds, groups) in scheme.items():
            split_tables.append(pd.DataFrame({"population": scope, "generation": generation, "split": split_name,
                "model_id": ids, "group": groups, "fold": folds}))
            print(f"evaluate {generation} {scope} {split_name}: {len(ids)} models", flush=True)
            for fold in np.unique(folds):
                test = folds == fold
                train = ~test
                for method in active_methods:
                    nearest = np.full(test.sum(), np.nan)
                    if not train.any():
                        prediction = np.full(test.sum(), np.nan)
                    elif method == "training_mean":
                        prediction = np.full(test.sum(), y[train].mean())
                    elif method == "training_median":
                        prediction = np.full(test.sum(), np.median(y[train]))
                    elif method == "rank_knn3":
                        prediction, nearest = knn(similarity[np.ix_(test, train)], y[train])
                    else:
                        x = table[columns[method]].to_numpy(float)
                        prediction = fit_predict(method, x[train], x[test], y[train])
                    rows.append(pd.DataFrame({"population": scope, "generation": generation,
                        "split": split_name, "fold": int(fold), "group": groups[test], "model_id": ids[test],
                        "method": method, "actual": y[test], "prediction": prediction,
                        "nearest_similarity": nearest, "source_population": source[test], "n_train": int(train.sum())}))
    return rows, audits, split_tables


def all_matrix_clustering(raw: pd.DataFrame, parent: Path, out: Path) -> None:
    """Choose groups using weights only, then open the parent's validation labels."""
    frame = raw.loc[raw["alpha_count"].ge(2)].sort_values("model_id")
    x = make_pipeline(SimpleImputer(), StandardScaler()).fit_transform(
        frame.filter(regex=r"^all__").to_numpy(float))
    candidates, fits = [], {}
    for k in (2, 4, 8, 16, 32, 64):
        if k >= len(x):
            continue
        model = MiniBatchKMeans(n_clusters=k, batch_size=1024, n_init=5, random_state=SEED)
        labels = model.fit_predict(x)
        score = silhouette_score(x, labels, sample_size=min(2000, len(x)), random_state=SEED)
        candidates.append({"n_clusters": k, "silhouette_euclidean": float(score)})
        fits[k] = (model, labels)
    selection = pd.DataFrame(candidates)
    k = int(selection.loc[selection.silhouette_euclidean.idxmax(), "n_clusters"])
    model, labels = fits[k]
    assignments = pd.DataFrame({"model_id": frame.model_id.to_numpy(), "predicted_cluster": labels})
    assignments["representative_model"] = ""
    for cluster in np.unique(labels):
        members = np.flatnonzero(labels == cluster)
        representative = members[np.argmin(np.linalg.norm(x[members] - model.cluster_centers_[cluster], axis=1))]
        assignments.loc[labels == cluster, "representative_model"] = frame.iloc[representative].model_id
    selection.to_csv(out/"all_matrix_cluster_selection.csv", index=False)
    assignments.to_csv(out/"all_matrix_cluster_assignments.csv", index=False)
    freeze = {"assignment_sha256": digest(out/"all_matrix_cluster_assignments.csv"),
              "inputs": "30 saved matrix quantiles; all unlabeled models used to standardize and cluster",
              "candidate_counts": candidates, "selected_count": k,
              "status": "transductive descriptive grouping, not prediction of unseen families"}
    (out/"all_matrix_cluster_freeze.json").write_text(json.dumps(freeze, indent=2)+"\n")
    labels = pd.read_csv(parent/"lineage_validation_assignments.csv")
    confirmed = labels["label_status"].eq("confirmed") & labels["audited_base_family"].notna()
    validation = validate_clusters(assignments, labels, (
        ("manual_confirmed", "audited_base_family", confirmed),
        ("metadata_root_consensus", "moderate_metadata_family", None),
        ("metadata_graph_root", "component_root_model", None),
    ))
    validation.to_csv(out/"all_matrix_cluster_validation.csv", index=False)


def task_checks(predictions: pd.DataFrame, features: pd.DataFrame, corpus: PartialRoleCorpus,
                validation: Path, out: Path) -> None:
    """Check individual benchmarks on the same eligible rows and fixed parts."""
    rows = []
    columns = signal_columns(features)
    indexed = features.set_index("model_id")
    for generation in ("modern", "legacy"):
        targets = pd.read_csv(validation/f"performance_{generation}_targets.csv").set_index("model_id")
        tasks = [c for c in targets if not c.startswith(("target_", generation+"_"))
                 and pd.api.types.is_numeric_dtype(targets[c])]
        for split in ("near_group_0995", "whole_weight_cluster"):
            base = predictions.loc[(predictions.population == "matched_role_population") &
                (predictions.generation == generation) & (predictions.split == split) &
                (predictions.method == "rank_knn3")].sort_values("model_id")
            ids = base.model_id.to_numpy()
            positions = pd.Index(corpus.model_ids).get_indexer(ids)
            subset = PartialRoleCorpus(ids, corpus.role_embeddings[positions], corpus.role_mask[positions], ())
            similarity, _ = pairwise_role_similarity(subset, subset, average_shared=True, minimum_shared_roles=4)
            for task in tasks:
                y = targets.loc[ids, task].to_numpy(float)
                for fold in base.fold.unique():
                    train = (base.fold.to_numpy() != fold) & np.isfinite(y)
                    test = (base.fold.to_numpy() == fold) & np.isfinite(y)
                    if not test.any():
                        continue
                    for method in ("training_median", "rank_knn3", "role_spectral_ridge", "role_spectral_trees"):
                        if not train.any():
                            predicted = np.full(test.sum(), np.nan)
                        elif method == "training_median":
                            predicted = np.full(test.sum(), np.median(y[train]))
                        elif method == "rank_knn3":
                            predicted, _ = knn(similarity[np.ix_(test, train)], y[train])
                        else:
                            x = indexed.loc[ids, columns[method]].to_numpy(float)
                            predicted = fit_predict(method, x[train], x[test], y[train])
                        rows.append(pd.DataFrame({"population": "matched_role_population", "generation": generation,
                            "split": split, "method": method, "target": task, "model_id": ids[test],
                            "fold": fold, "actual": y[test], "prediction": predicted}))
    frame = pd.concat(rows, ignore_index=True)
    frame.to_csv(out/"performance_task_predictions.csv", index=False)
    tables = [metrics(part).assign(target=target) for target, part in frame.groupby("target")]
    pd.concat(tables, ignore_index=True).to_csv(out/"performance_task_metrics.csv", index=False)


def control_checks(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = signal_columns(features)
    indexed = features.set_index("model_id")
    for (generation, split), base in predictions.loc[
        (predictions.population == "matched_role_population") & (predictions.method == "rank_knn3") &
        predictions.split.isin(("near_group_0995", "whole_weight_cluster"))
    ].groupby(["generation", "split"]):
        base = base.sort_values("model_id")
        y = base.actual.to_numpy()
        for method in ("matrix_shape_trees", "nonalpha_spectral_trees", "nonalpha_spectral_ridge"):
            x = indexed.loc[base.model_id, columns[method]].to_numpy(float)
            for fold in base.fold.unique():
                test = base.fold.to_numpy() == fold
                train = ~test
                frame = base.loc[test].copy()
                frame["method"] = method
                frame["prediction"] = fit_predict(method, x[train], x[test], y[train]) if train.any() else np.nan
                frame["nearest_similarity"] = np.nan
                rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT/"research_runs/htsr_reassessment")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reuse-run", type=Path, help="Reuse validated aggregate predictions and matrix summaries; run added diagnostics")
    parser.add_argument("--lineage-check", type=Path, help="Attach separately frozen gallery reassessment outputs")
    args = parser.parse_args()
    run = args.run.resolve()
    validate_artifact_manifest(run, required_files=("metadata.json", "performance_predictions.csv", "performance_frozen_splits.csv"))
    metadata = read_json(run/"metadata.json")
    study = Path(metadata["sources"]["study_run"])
    validation = Path(metadata["sources"]["validation_run"])
    provenance = read_json(study/"provenance.json")
    if args.reuse_run:
        validate_artifact_manifest(args.reuse_run, required_files=("metadata.json", "all_matrix_features.csv", "performance_predictions.csv", "splits.csv", "duplicate_audit.csv"))
    raw = (pd.read_csv(args.reuse_run/"all_matrix_features.csv") if args.reuse_run
           else all_matrix_features(provenance, args.output_root/"cache", args.workers))
    inputs = {str(p): digest(p) for p in (study/"feature_table.csv", study/"external_feature_table.csv")}
    inputs["raw_matrix_artifacts"] = hashlib.sha256(raw[["model_id", "artifact_sha256"]].to_csv(index=False).encode()).hexdigest()
    for path in metadata["caches"]["coverage_aware"]:
        inputs[path] = digest(Path(path))
    dependency_paths = [Path(__file__), ROOT/"spectral_research/study/partial_roles.py",
                        ROOT/"spectral_research/study/weight_only.py"]
    dependencies = {str(p.relative_to(ROOT)): digest(p) for p in dependency_paths}
    key = hashlib.sha256((json.dumps(dependencies, sort_keys=True) + json.dumps(inputs, sort_keys=True) + digest(run/"artifact_manifest.json")
        + (digest(args.reuse_run/"artifact_manifest.json") if args.reuse_run else "")
        + (digest(args.lineage_check/"summary.json") if args.lineage_check else "")).encode()).hexdigest()[:20]
    out = args.output_root/"runs"/f"reassessment_{key}"
    out.mkdir(parents=True, exist_ok=True)
    protocol = {"parent_run": str(run), "method": "htsr_reassessment_v1", "seed": SEED,
        "folds": FOLDS, "methods": METHODS, "near_similarity_threshold": 0.995,
        "primary_targets": ["modern_average", "legacy_average"],
        "predictor_inputs": "saved matrix measurements only; training benchmark scores",
        "metadata_use": "post-prediction validation; collection ID defines collection-holdout evaluation only",
        "status": "exploratory reanalysis of previously studied data; not an untouched confirmation set",
        "trees": "200 boosting iterations, 15 leaves, learning_rate=.05, L2=10, min_leaf=20",
        "ridge": "alpha=100; imputation and scaling fitted on training rows only",
        "diagnostics": "same-part individual benchmark checks; unlabeled all-matrix clustering; no tuned settings",
        "controls": ["matrix_shape_trees", "nonalpha_spectral_trees", "nonalpha_spectral_ridge"],
        "tie_policy": "stable model-ID order for equal neighbor scores; differs from prior argpartition ties",
        "sources": {"parent_manifest": digest(run/"artifact_manifest.json"), "implementations": dependencies, "inputs": inputs}}
    if args.reuse_run:
        validate_artifact_manifest(args.reuse_run, required_files=("metadata.json", "all_matrix_features.csv", "performance_predictions.csv", "splits.csv", "duplicate_audit.csv"))
        if Path(read_json(args.reuse_run/"metadata.json")["parent_run"]).resolve() != run:
            raise ValueError("Reused predictions must have the same parent")
        protocol["reused_aggregate_run"] = str(args.reuse_run.resolve())
        protocol["sources"]["reused_manifest"] = digest(args.reuse_run/"artifact_manifest.json")
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    print(f"RUN {out}", flush=True)
    raw.to_csv(out/"all_matrix_features.csv", index=False)
    role_features = pd.concat([pd.read_csv(study/"feature_table.csv"), pd.read_csv(study/"external_feature_table.csv")], ignore_index=True)
    features = raw.merge(role_features, on="model_id", how="left", validate="one_to_one")
    corpus = load_partial(metadata)
    columns = signal_columns(features)
    predictions, audits, splits = [], [], []
    with threadpool_limits(limits=4):
        if args.reuse_run:
            predictions = [pd.read_csv(args.reuse_run/"performance_predictions.csv", low_memory=False)]
            audits = pd.read_csv(args.reuse_run/"duplicate_audit.csv").to_dict("records")
            splits = [pd.read_csv(args.reuse_run/"splits.csv")]
        else:
            for generation in ("modern", "legacy"):
                targets = pd.read_csv(validation/f"performance_{generation}_targets.csv")
                p, a, s = evaluate_generation(generation, targets, features, corpus, run, columns)
                predictions.extend(p)
                audits.extend(a)
                splits.extend(s)
        combined = pd.concat(predictions, ignore_index=True)
        combined = combined.loc[combined["method"].isin(METHODS)].copy()
        combined = pd.concat([combined, control_checks(combined, features)], ignore_index=True)
        predictions = [combined]
        combined.to_csv(out/"performance_predictions.csv", index=False)
        task_checks(combined, features, corpus, validation, out)
        all_matrix_clustering(raw, run, out)
    pd.DataFrame(audits).to_csv(out/"duplicate_audit.csv", index=False)
    pd.concat(splits, ignore_index=True).to_csv(out/"splits.csv", index=False)
    results = metrics(pd.concat(predictions, ignore_index=True))
    results.to_csv(out/"performance_metrics.csv", index=False)
    if args.lineage_check:
        summary = read_json(args.lineage_check/"summary.json")
        if summary["status"] != "complete" or Path(summary["parent_run"]).resolve() != run:
            raise ValueError("Lineage evidence is incomplete or has a different parent")
        frozen = read_json(args.lineage_check/"prediction_freeze_manifest.json")
        for name, record in frozen["frozen_artifacts"].items():
            if digest(args.lineage_check/name) != record["sha256"]:
                raise ValueError(f"Lineage prediction digest mismatch: {name}")
        protocol["lineage_check"] = str(args.lineage_check.resolve())
        for path in args.lineage_check.iterdir():
            if path.suffix in (".csv", ".json"):
                (out/("lineage_"+path.name)).write_bytes(path.read_bytes())
    (out/"metadata.json").write_text(json.dumps({**protocol, "run_id": out.name,
        "counts": {"saved_models": len(raw), "all_matrix_alpha_models": int(raw["alpha_count"].ge(2).sum())}}, indent=2)+"\n")
    write_artifact_manifest(out, run_id=out.name)
    print(results[["population", "generation", "split", "method", "n_models", "mae", "spearman"]].to_string(index=False))


if __name__ == "__main__":
    main()
