from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from alpha_cluster_dashboard.data import AlphaRecord, load_alpha_records, schema_name
from spectral_research.benchmark import build_audit_manifest
from spectral_research.features import alpha_scalar_features
from spectral_research.heterogeneity import role_aggregated_trace_score, structural_basics_feature_map

PERFORMANCE_TARGETS = ("Average ⬆️", "MMLU-PRO", "MATH Lvl 5")


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _group_key(row: pd.Series) -> str:
    for column in ("audited_base_family", "candidate_family", "model_id"):
        text = _clean_text(row.get(column))
        if text:
            return text
    return "unknown"


def _read_csv(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns or [])
    if columns is None:
        return frame
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, columns].copy()


def _lookup_targets(leaderboard: pd.DataFrame, key: str) -> pd.DataFrame:
    columns = [key, *PERFORMANCE_TARGETS]
    available = [column for column in columns if column in leaderboard.columns]
    if key not in available:
        return pd.DataFrame(columns=PERFORMANCE_TARGETS)

    frame = leaderboard.loc[:, available].copy()
    frame[key] = frame[key].map(_clean_text)
    frame = frame[frame[key].ne("")]
    if frame.empty:
        return pd.DataFrame(columns=PERFORMANCE_TARGETS)

    for column in PERFORMANCE_TARGETS:
        if column not in frame.columns:
            frame[column] = pd.NA

    row_scores = frame.loc[:, list(PERFORMANCE_TARGETS)].notna().sum(axis=1)
    ordered = frame.assign(_row_score=row_scores, _row_order=range(len(frame))).sort_values(
        [key, "_row_score", "_row_order"], ascending=[True, False, True], kind="mergesort"
    )
    best_rows = ordered.drop_duplicates(subset=[key], keep="first").set_index(key)
    return best_rows.loc[:, list(PERFORMANCE_TARGETS)]


def join_prediction_targets(manifest: pd.DataFrame, leaderboard: pd.DataFrame) -> pd.DataFrame:
    joined = manifest.copy()
    display_names = (
        joined["display_name"].map(_clean_text)
        if "display_name" in joined.columns
        else joined["model_id"].map(_clean_text)
    )

    atlas_lookup = _lookup_targets(leaderboard, "atlas_repo_id")
    fullname_lookup = _lookup_targets(leaderboard, "fullname")

    for target in PERFORMANCE_TARGETS:
        if target in atlas_lookup.columns:
            values = display_names.map(atlas_lookup[target])
        else:
            values = pd.Series(index=joined.index, dtype="object")
        if target in fullname_lookup.columns:
            values = values.combine_first(display_names.map(fullname_lookup[target]))
        joined[target] = pd.to_numeric(values, errors="coerce")

    joined["cv_group"] = joined.apply(_group_key, axis=1)
    return joined


def _negative_l2(left: np.ndarray, right: np.ndarray) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    mask = np.isfinite(left_array) & np.isfinite(right_array)
    if not mask.any():
        return float("nan")
    delta = left_array[mask] - right_array[mask]
    return float(-np.mean(delta * delta))


def _cv_groups(frame: pd.DataFrame) -> np.ndarray:
    if "cv_group" in frame.columns:
        groups = frame["cv_group"]
    else:
        groups = frame.apply(_group_key, axis=1)
    return groups.fillna("unknown").astype(str).to_numpy()


def _group_kfold(frame: pd.DataFrame, *, cv_splits: int) -> GroupKFold | None:
    group_count = pd.Index(_cv_groups(frame)).nunique()
    if group_count < 2:
        return None
    return GroupKFold(n_splits=min(cv_splits, group_count))


def _pair_score(
    *,
    method: str,
    left_row: pd.Series,
    right_row: pd.Series,
    left_record: AlphaRecord | None,
    right_record: AlphaRecord | None,
    structural_vectors: dict[str, np.ndarray],
) -> float:
    left_model_id = str(left_row["model_id"])
    right_model_id = str(right_row["model_id"])

    if method == "structural_basics_only":
        left_vector = structural_vectors.get(left_model_id)
        right_vector = structural_vectors.get(right_model_id)
        if left_vector is None or right_vector is None:
            return float("nan")
        return _negative_l2(left_vector, right_vector)

    if left_record is None or right_record is None:
        return float("nan")

    if method == "pl_alpha_scalar":
        return _negative_l2(
            alpha_scalar_features(left_record.alpha).reshape(-1),
            alpha_scalar_features(right_record.alpha).reshape(-1),
        )

    if method == "full_module_trace":
        if tuple(left_record.raw_modules) != tuple(right_record.raw_modules):
            return float("nan")
        depth = min(left_record.alpha.shape[0], right_record.alpha.shape[0])
        if depth <= 0:
            return float("nan")
        return _negative_l2(left_record.alpha[:depth], right_record.alpha[:depth])

    if method == "role_aggregated_trace":
        return role_aggregated_trace_score(
            query_alpha=np.asarray(left_record.alpha, dtype=float),
            query_modules=tuple(left_record.raw_modules),
            proto_alpha=np.asarray(right_record.alpha, dtype=float),
            proto_modules=tuple(right_record.raw_modules),
            overlap_penalty=0.2,
        )

    raise KeyError(f"Unsupported method: {method}")


def _weighted_mean(scored_neighbors: list[tuple[float, float]]) -> float:
    scores = np.asarray([score for score, _ in scored_neighbors], dtype=float)
    values = np.asarray([value for _, value in scored_neighbors], dtype=float)
    weights = np.exp(scores - scores.max())
    return float(np.average(values, weights=weights))


def build_pairwise_score_matrix(
    frame: pd.DataFrame,
    records_by_model_id: dict[str, AlphaRecord],
    *,
    method: str = "pl_alpha_scalar",
    backend: str = "cpu",
) -> tuple[np.ndarray, dict[str, int]]:
    if backend != "cpu":
        raise NotImplementedError("Only the CPU pairwise score matrix backend is supported")

    model_ids = [str(model_id) for model_id in frame["model_id"]]
    model_index = {model_id: index for index, model_id in enumerate(model_ids)}
    matrix = np.full((len(model_ids), len(model_ids)), np.nan, dtype=float)
    structural_vectors = structural_basics_feature_map(frame)

    for left_index, left_model_id in enumerate(model_ids):
        left_row = frame.iloc[left_index]
        left_record = records_by_model_id.get(left_model_id)
        for right_index in range(left_index, len(model_ids)):
            right_model_id = model_ids[right_index]
            score = _pair_score(
                method=method,
                left_row=left_row,
                right_row=frame.iloc[right_index],
                left_record=left_record,
                right_record=records_by_model_id.get(right_model_id),
                structural_vectors=structural_vectors,
            )
            matrix[left_index, right_index] = score
            matrix[right_index, left_index] = score

    return matrix, model_index


def structural_baseline_oof(
    frame: pd.DataFrame, *, target: str, cv_splits: int = 5
) -> pd.DataFrame:
    out = frame[["model_id", "query_bucket", target]].copy()
    out["baseline_prediction"] = np.nan
    out["mode"] = "structural_baseline"

    if out.empty:
        return out

    vectors = structural_basics_feature_map(frame)
    y = frame[target].to_numpy(dtype=float)
    groups = _cv_groups(frame)
    splitter = _group_kfold(frame, cv_splits=cv_splits)

    if splitter is None:
        out["baseline_prediction"] = float(np.nanmean(y))
        return out

    prediction_column = out.columns.get_loc("baseline_prediction")
    for train_idx, test_idx in splitter.split(frame, y, groups):
        x_train = np.vstack([vectors[str(frame.iloc[index]["model_id"])] for index in train_idx])
        x_test = np.vstack([vectors[str(frame.iloc[index]["model_id"])] for index in test_idx])
        model = Ridge(alpha=1.0)
        model.fit(x_train, y[train_idx])
        out.iloc[test_idx, prediction_column] = model.predict(x_test)

    return out


def neighborhood_predict_oof(
    frame: pd.DataFrame,
    records_by_model_id: dict[str, AlphaRecord],
    *,
    target: str,
    method: str,
    k: int,
    weighted: bool,
    residualize: bool,
    pairwise_scores: np.ndarray | None = None,
    model_index: dict[str, int] | None = None,
    baseline_by_model_id: dict[str, float] | None = None,
    cv_splits: int = 5,
) -> pd.DataFrame:
    out = frame[["model_id", "query_bucket", target]].copy()
    out["prediction"] = np.nan
    out["mode"] = "spectral_plus_structural" if residualize else "raw_spectral"

    if out.empty:
        return out

    structural_vectors = structural_basics_feature_map(frame)
    if baseline_by_model_id is None:
        baseline = structural_baseline_oof(frame, target=target, cv_splits=cv_splits)
        baseline_by_model_id = {
            str(model_id): float(prediction)
            for model_id, prediction in zip(
                baseline["model_id"],
                baseline["baseline_prediction"],
                strict=False,
            )
        }
    target_by_model_id = {
        str(model_id): float(value)
        for model_id, value in zip(frame["model_id"], frame[target], strict=False)
    }
    working_target_by_model_id = {
        model_id: target_by_model_id[model_id] - baseline_by_model_id[model_id]
        if residualize
        else target_by_model_id[model_id]
        for model_id in target_by_model_id
    }

    groups = _cv_groups(frame)
    splitter = _group_kfold(frame, cv_splits=cv_splits)
    splits = (
        list(splitter.split(frame, frame[target].to_numpy(dtype=float), groups))
        if splitter is not None
        else [(np.arange(len(frame), dtype=int), np.arange(len(frame), dtype=int))]
    )

    prediction_column = out.columns.get_loc("prediction")
    for train_idx, test_idx in splits:
        train_frame = frame.iloc[train_idx]
        for test_position in test_idx:
            test_row = frame.iloc[test_position]
            test_model_id = str(test_row["model_id"])
            test_record = records_by_model_id.get(test_model_id)
            scored_neighbors: list[tuple[float, float]] = []
            for train_row in train_frame.itertuples(index=False):
                train_model_id = str(train_row.model_id)
                if train_model_id == test_model_id:
                    continue
                score = np.nan
                if pairwise_scores is not None and model_index is not None:
                    left_index = model_index.get(test_model_id)
                    right_index = model_index.get(train_model_id)
                    if left_index is not None and right_index is not None:
                        score = float(pairwise_scores[left_index, right_index])
                if not np.isfinite(score):
                    score = _pair_score(
                        method=method,
                        left_row=test_row,
                        right_row=pd.Series(train_row._asdict()),
                        left_record=test_record,
                        right_record=records_by_model_id.get(train_model_id),
                        structural_vectors=structural_vectors,
                    )
                if not np.isfinite(score):
                    continue
                scored_neighbors.append((float(score), working_target_by_model_id[train_model_id]))

            scored_neighbors.sort(key=lambda item: item[0], reverse=True)
            top_neighbors = scored_neighbors[:k]
            if not top_neighbors:
                continue

            prediction = (
                _weighted_mean(top_neighbors)
                if weighted
                else float(np.mean([value for _, value in top_neighbors]))
            )
            if residualize:
                prediction += baseline_by_model_id[test_model_id]
            out.iloc[test_position, prediction_column] = prediction

    return out


def summarize_predictions(
    predictions: pd.DataFrame, *, target: str, method: str, k: int, weighted: bool
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if "prediction" in predictions.columns:
        prediction_column = "prediction"
    elif "baseline_prediction" in predictions.columns:
        prediction_column = "baseline_prediction"
    else:
        raise KeyError("predictions must include `prediction` or `baseline_prediction`")

    mode = (
        str(predictions["mode"].iloc[0])
        if "mode" in predictions.columns and not predictions.empty
        else ("structural_baseline" if method == "structural_basics_only" else "raw_spectral")
    )

    for bucket, bucket_frame in predictions.groupby("query_bucket", dropna=False):
        target_values = pd.to_numeric(bucket_frame[target], errors="coerce").to_numpy(dtype=float)
        prediction_values = pd.to_numeric(bucket_frame[prediction_column], errors="coerce").to_numpy(
            dtype=float
        )
        valid_mask = np.isfinite(target_values) & np.isfinite(prediction_values)
        if not valid_mask.any():
            continue

        y_true = target_values[valid_mask]
        y_pred = prediction_values[valid_mask]
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
        rho = (
            float("nan")
            if len(y_true) < 2 or np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0])
            else float(spearmanr(y_true, y_pred).statistic)
        )
        rows.append(
            {
                "target": target,
                "method": method,
                "mode": mode,
                "bucket": bucket,
                "k": int(k),
                "weighted": bool(weighted),
                "n": int(valid_mask.sum()),
                "r2": r2,
                "mae": mae,
                "spearman": rho,
            }
        )

    return pd.DataFrame(rows)


def _records_frame(records: list[AlphaRecord]) -> pd.DataFrame:
    columns = [
        "model_id",
        "display_name",
        "path",
        "num_layers",
        "raw_module_count",
        "raw_modules",
        "common_schema",
        "schema_name",
    ]
    frame = pd.DataFrame(
        [
            {
                "model_id": record.model_id,
                "display_name": record.display_name,
                "path": record.path,
                "num_layers": record.num_layers,
                "raw_module_count": record.raw_module_count,
                "raw_modules": ", ".join(record.raw_modules),
                "common_schema": record.common_schema,
                "schema_name": schema_name(record),
            }
            for record in records
        ]
    )
    return frame.reindex(columns=columns)


def build_prediction_dataset(
    metrics_dir: str | Path,
    leaderboard_csv: str | Path,
    lineage_csv: str | Path,
    freeze_csv: str | Path,
) -> tuple[pd.DataFrame, dict[str, AlphaRecord]]:
    records = load_alpha_records(metrics_dir)
    records_frame = _records_frame(records)
    metadata = _read_csv(Path(lineage_csv))
    leaderboard = _read_csv(Path(leaderboard_csv))
    overrides = _read_csv(
        Path(freeze_csv),
        columns=["model_id", "audited_base_family", "label_status", "bucket_override", "audit_notes"],
    )

    manifest = build_audit_manifest(records_frame, metadata, leaderboard, overrides)
    joined = join_prediction_targets(manifest, leaderboard)
    joined = joined.dropna(subset=list(PERFORMANCE_TARGETS)).reset_index(drop=True)
    records_by_model_id = {record.model_id: record for record in records}
    return joined, records_by_model_id
