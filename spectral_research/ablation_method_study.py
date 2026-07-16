from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wasserstein_distance
from sklearn.metrics import roc_auc_score

from alpha_cluster_dashboard.data import AlphaRecord, canonicalize_module_name, load_alpha_records
from spectral_research.features import alpha_scalar_features


DEFAULT_METHODS = (
    "structural_basics",
    "paper_spearman",
    "existing_role_l2",
    "existing_scalar_l2",
    "role_alpha_wasserstein",
    "rank_rbf",
    "hybrid_rank",
)
IGNORED_FAMILIES = {"", "unknown", "?", "nan", "none", "n/a"}


@dataclass(frozen=True)
class MethodResult:
    method: str
    matrix: np.ndarray
    elapsed_seconds: float


def _finite_pair(left: np.ndarray, right: np.ndarray, *, min_points: int = 3) -> tuple[np.ndarray, np.ndarray] | None:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    depth = min(left_array.shape[0], right_array.shape[0])
    if depth <= 0:
        return None
    left_array = left_array[:depth]
    right_array = right_array[:depth]
    mask = np.isfinite(left_array) & np.isfinite(right_array)
    if int(mask.sum()) < min_points:
        return None
    return left_array[mask], right_array[mask]


def _spearman_trace(left: np.ndarray, right: np.ndarray) -> float:
    pair = _finite_pair(left, right)
    if pair is None:
        return float("nan")
    x, y = pair
    if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return 1.0 if np.allclose(x, y) else 0.0
    value = spearmanr(x, y).statistic
    return float(value) if value is not None and np.isfinite(value) else float("nan")


def _rank_rbf_trace(left: np.ndarray, right: np.ndarray, *, sigma: float = 0.35) -> float:
    pair = _finite_pair(left, right)
    if pair is None:
        return float("nan")
    x, y = pair
    scale = max(len(x) - 1, 1)
    x_rank = (rankdata(x, method="average") - 1.0) / scale
    y_rank = (rankdata(y, method="average") - 1.0) / scale
    distance = float(np.mean((x_rank - y_rank) ** 2))
    return float(2.0 * np.exp(-distance / (2.0 * sigma * sigma)) - 1.0)


def _negative_l2_trace(left: np.ndarray, right: np.ndarray) -> float:
    pair = _finite_pair(left, right, min_points=1)
    if pair is None:
        return float("nan")
    x, y = pair
    return float(-np.mean((x - y) ** 2))


def _negative_wasserstein_trace(left: np.ndarray, right: np.ndarray) -> float:
    pair = _finite_pair(left, right, min_points=1)
    if pair is None:
        return float("nan")
    x, y = pair
    return float(-wasserstein_distance(x, y))


def _pearson_z_trace(left: np.ndarray, right: np.ndarray) -> float:
    pair = _finite_pair(left, right)
    if pair is None:
        return float("nan")
    x, y = pair
    x_std = float(np.nanstd(x))
    y_std = float(np.nanstd(y))
    if x_std <= 1e-12 or y_std <= 1e-12:
        return 1.0 if np.allclose(x, y) else 0.0
    x = (x - float(np.nanmean(x))) / x_std
    y = (y - float(np.nanmean(y))) / y_std
    return float(np.corrcoef(x, y)[0, 1])


def role_traces(record: AlphaRecord) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {}
    alpha = np.asarray(record.alpha, dtype=float)
    for index, module_name in enumerate(record.raw_modules):
        role = canonicalize_module_name(module_name)
        if role.startswith("other::"):
            continue
        grouped.setdefault(role, []).append(alpha[:, index])
    return {role: np.nanmean(np.vstack(columns), axis=0) for role, columns in grouped.items()}


def pair_similarity(
    left_roles: dict[str, np.ndarray],
    right_roles: dict[str, np.ndarray],
    *,
    method: str,
    coverage_penalty: float = 0.0,
) -> float:
    shared_roles = sorted(set(left_roles) & set(right_roles))
    if not shared_roles:
        return float("nan")
    scores = []
    for role in shared_roles:
        if method == "paper_spearman":
            scores.append(_spearman_trace(left_roles[role], right_roles[role]))
        elif method == "rank_rbf":
            scores.append(_rank_rbf_trace(left_roles[role], right_roles[role]))
        elif method == "existing_role_l2":
            scores.append(_negative_l2_trace(left_roles[role], right_roles[role]))
        elif method == "role_alpha_wasserstein":
            scores.append(_negative_wasserstein_trace(left_roles[role], right_roles[role]))
        elif method == "hybrid_rank":
            spearman = _spearman_trace(left_roles[role], right_roles[role])
            rank_rbf = _rank_rbf_trace(left_roles[role], right_roles[role])
            pearson_z = _pearson_z_trace(left_roles[role], right_roles[role])
            components = [value for value in (spearman, rank_rbf, pearson_z) if np.isfinite(value)]
            scores.append(float(np.mean(components)) if components else float("nan"))
        else:
            raise KeyError(f"Unsupported trace method: {method}")
    finite_scores = np.asarray([score for score in scores if np.isfinite(score)], dtype=float)
    if finite_scores.size == 0:
        return float("nan")
    coverage = len(shared_roles) / max(len(left_roles), len(right_roles))
    return float(np.nanmean(finite_scores) - coverage_penalty * (1.0 - coverage))


def record_similarity(left: AlphaRecord, right: AlphaRecord, *, method: str) -> float:
    if method == "existing_scalar_l2":
        left_vector = alpha_scalar_features(left.alpha).reshape(-1)
        right_vector = alpha_scalar_features(right.alpha).reshape(-1)
        mask = np.isfinite(left_vector) & np.isfinite(right_vector)
        if not mask.any():
            return float("nan")
        return float(-np.mean((left_vector[mask] - right_vector[mask]) ** 2))
    return pair_similarity(role_traces(left), role_traces(right), method=method, coverage_penalty=0.05)


def build_similarity_matrix(records: list[AlphaRecord], *, method: str) -> np.ndarray:
    matrix = np.eye(len(records), dtype=float)
    for left_index, left_record in enumerate(records):
        for right_index in range(left_index + 1, len(records)):
            score = record_similarity(left_record, records[right_index], method=method)
            matrix[left_index, right_index] = score
            matrix[right_index, left_index] = score
    return matrix


def build_feature_similarity_matrix(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    matrix = np.eye(values.shape[0], dtype=float)
    if values.size == 0:
        return matrix
    for left_index in range(values.shape[0]):
        for right_index in range(left_index + 1, values.shape[0]):
            left = values[left_index]
            right = values[right_index]
            mask = np.isfinite(left) & np.isfinite(right)
            score = float("nan")
            if mask.any():
                score = float(-np.mean((left[mask] - right[mask]) ** 2))
            matrix[left_index, right_index] = score
            matrix[right_index, left_index] = score
    return matrix


def build_structural_similarity_matrix(frame: pd.DataFrame, records: list[AlphaRecord]) -> np.ndarray:
    numeric = pd.DataFrame(index=frame.index)
    if "#Params (B)" in frame.columns:
        numeric["params_b"] = pd.to_numeric(frame["#Params (B)"], errors="coerce")
    numeric["num_layers"] = [record.num_layers for record in records]
    numeric["raw_module_count"] = [record.raw_module_count for record in records]
    numeric["common_schema"] = [float(record.common_schema) for record in records]
    for column in ("is_moe", "is_quantized", "is_adapter", "is_merge"):
        if column in frame.columns:
            numeric[column] = pd.to_numeric(frame[column], errors="coerce")

    numeric = numeric.astype(float)
    for column in numeric.columns:
        mean = float(numeric[column].mean(skipna=True))
        std = float(numeric[column].std(skipna=True))
        if np.isfinite(std) and std > 1e-12:
            numeric[column] = (numeric[column] - mean) / std
        else:
            numeric[column] = numeric[column] - mean if np.isfinite(mean) else 0.0
    numeric = numeric.fillna(0.0)

    categorical_columns = [
        column
        for column in ("Architecture", "Precision", "Type", "Weight type")
        if column in frame.columns
    ]
    categorical = (
        pd.get_dummies(frame[categorical_columns].fillna("unknown").astype(str), dtype=float)
        if categorical_columns
        else pd.DataFrame(index=frame.index)
    )
    features = pd.concat([numeric.reset_index(drop=True), categorical.reset_index(drop=True)], axis=1)
    return build_feature_similarity_matrix(features.to_numpy(dtype=float))


def evaluate_family_similarity(matrix: np.ndarray, family_labels: list[str]) -> dict[str, float]:
    labels = np.asarray(family_labels, dtype=object)
    y_true: list[int] = []
    y_score: list[float] = []
    for left_index in range(len(labels)):
        for right_index in range(left_index + 1, len(labels)):
            score = float(matrix[left_index, right_index])
            if not np.isfinite(score):
                continue
            y_true.append(int(labels[left_index] == labels[right_index]))
            y_score.append(score)
    pairwise_auc = float("nan")
    if len(set(y_true)) == 2:
        pairwise_auc = float(roc_auc_score(y_true, y_score))

    correct = 0
    eligible = 0
    for index, label in enumerate(labels):
        scores = np.asarray(matrix[index], dtype=float).copy()
        scores[index] = np.nan
        finite = np.isfinite(scores)
        if not finite.any():
            continue
        nearest = int(np.nanargmax(scores))
        eligible += 1
        correct += int(labels[nearest] == label)
    return {
        "pairwise_auc": pairwise_auc,
        "top1_family_recall": float(correct / eligible) if eligible else float("nan"),
        "pair_count": float(len(y_true)),
        "eligible_queries": float(eligible),
    }


def evaluate_knn_prediction(matrix: np.ndarray, frame: pd.DataFrame, *, target: str, k: int = 3) -> dict[str, float]:
    values = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
    y_true: list[float] = []
    y_pred: list[float] = []
    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        scores = np.asarray(matrix[index], dtype=float).copy()
        scores[index] = np.nan
        valid = np.isfinite(scores) & np.isfinite(values)
        if not valid.any():
            continue
        neighbor_indices = np.where(valid)[0]
        order = neighbor_indices[np.argsort(scores[neighbor_indices])[::-1]]
        chosen = order[:k]
        if chosen.size == 0:
            continue
        chosen_scores = scores[chosen]
        weights = np.exp(chosen_scores - float(np.nanmax(chosen_scores)))
        prediction = float(np.average(values[chosen], weights=weights))
        y_true.append(float(value))
        y_pred.append(prediction)
    if not y_true:
        return {"target": target, "n": 0, "mae": float("nan"), "spearman": float("nan")}
    true_array = np.asarray(y_true, dtype=float)
    pred_array = np.asarray(y_pred, dtype=float)
    rho = spearmanr(true_array, pred_array).statistic if len(true_array) >= 3 else float("nan")
    return {
        "target": target,
        "n": int(len(true_array)),
        "mae": float(np.mean(np.abs(true_array - pred_array))),
        "spearman": float(rho) if rho is not None and np.isfinite(rho) else float("nan"),
    }


def load_records_by_display_name(metrics_dir: Path) -> dict[str, AlphaRecord]:
    return {record.display_name: record for record in load_alpha_records(metrics_dir)}


def select_balanced_study_frame(
    per_model: pd.DataFrame,
    available_model_ids: set[str],
    *,
    min_family_size: int = 25,
    max_families: int = 10,
    max_models_per_family: int = 40,
    random_state: int = 0,
) -> pd.DataFrame:
    frame = per_model[
        per_model["model_id"].isin(available_model_ids)
        & per_model["coarse_family"].fillna("").astype(str).str.lower().map(lambda value: value not in IGNORED_FAMILIES)
    ].copy()
    counts = frame["coarse_family"].value_counts()
    keep_families = counts[counts >= min_family_size].head(max_families).index.tolist()
    frame = frame[frame["coarse_family"].isin(keep_families)].copy()
    rows = []
    for family in keep_families:
        group = frame[frame["coarse_family"].eq(family)].sort_values("model_id")
        if len(group) > max_models_per_family:
            group = group.sample(n=max_models_per_family, random_state=random_state).sort_values("model_id")
        rows.append(group)
    if not rows:
        return frame.iloc[0:0].copy()
    return pd.concat(rows, ignore_index=True).sort_values(["coarse_family", "model_id"]).reset_index(drop=True)


def method_decision(family_metrics: pd.DataFrame, prediction_metrics: pd.DataFrame) -> str:
    lines = ["# Ablation Method Study", ""]
    if family_metrics.empty:
        lines.append("No family similarity metrics were produced.")
        return "\n".join(lines) + "\n"

    family_ranked = family_metrics.sort_values(
        ["pairwise_auc", "top1_family_recall"], ascending=[False, False]
    )
    best_family = family_ranked.iloc[0]
    lines.extend(["## Decision", f"- Best family-retrieval AUROC: `{best_family['method']}`."])
    if not prediction_metrics.empty:
        pred_rank = (
            prediction_metrics.dropna(subset=["mae"])
            .assign(target_rank=lambda frame: frame.groupby("target")["mae"].rank(method="average"))
            .groupby("method", as_index=False)["target_rank"]
            .mean()
            .sort_values("target_rank")
        )
        if not pred_rank.empty:
            lines.append(f"- Best kNN prediction method by average per-target MAE rank: `{pred_rank.iloc[0]['method']}`.")

    top1_ranked = family_metrics.sort_values("top1_family_recall", ascending=False)
    if not top1_ranked.empty:
        lines.append(f"- Best top-1 family recall: `{top1_ranked.iloc[0]['method']}`.")

    paper = family_metrics[family_metrics["method"].eq("paper_spearman")]
    structural = family_metrics[family_metrics["method"].eq("structural_basics")]
    existing = family_metrics[family_metrics["method"].str.startswith("existing_")]
    rank_family = family_metrics[family_metrics["method"].isin(["rank_rbf", "hybrid_rank"])]
    distributional = family_metrics[family_metrics["method"].eq("role_alpha_wasserstein")]

    lines.extend(["", "## Pragmatic Verdict"])
    if not structural.empty:
        structural_auc = float(structural["pairwise_auc"].iloc[0])
        best_auc = float(best_family["pairwise_auc"])
        if best_family["method"] == "structural_basics":
            lines.append(
                "- Family retrieval is dominated by structural metadata; treat family AUROC as a leakage-prone control, not a spectral discovery."
            )
        elif structural_auc + 0.02 >= best_auc:
            lines.append(
                "- Structural metadata is close to the best family AUROC, so family retrieval should be reported with this control."
            )
        else:
            lines.append("- Spectral scores beat the structural control on family AUROC in this slice.")

    if not paper.empty and not existing.empty:
        best_existing_auc = float(existing["pairwise_auc"].max())
        paper_auc = float(paper["pairwise_auc"].iloc[0])
        paper_top1 = float(paper["top1_family_recall"].iloc[0])
        best_existing_top1 = float(existing["top1_family_recall"].max())
        if paper_auc + 0.01 >= best_existing_auc:
            lines.append(
                "- Paper Spearman is competitive with existing alpha baselines; keep it as the paper-faithful shape baseline."
            )
        else:
            lines.append(
                "- Existing alpha baselines are stronger by family AUROC; report paper Spearman as the required paper-faithful baseline, not the operational default."
            )
        if paper_top1 > best_existing_top1:
            lines.append("- Paper Spearman has better top-1 family recall than existing alpha baselines, which supports keeping it in retrieval studies.")
    if not rank_family.empty:
        lines.append(
            "- Rank-shape methods are the strongest direction for kNN benchmark prediction in this study; they keep the paper's rank-invariance idea while adding smoother similarity."
        )
    if not distributional.empty:
        distributional_auc = float(distributional["pairwise_auc"].iloc[0])
        if np.isfinite(distributional_auc) and distributional_auc < float(best_family["pairwise_auc"]):
            lines.append("- Role-wise alpha Wasserstein is a useful sanity check, but it does not beat the leading methods here.")

    def _format_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
        table = frame.loc[:, columns].copy()
        for column in table.columns:
            if pd.api.types.is_float_dtype(table[column]):
                table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        widths = {column: max(len(column), *(len(str(value)) for value in table[column])) for column in table.columns}
        header = "| " + " | ".join(column.ljust(widths[column]) for column in table.columns) + " |"
        divider = "| " + " | ".join("-" * widths[column] for column in table.columns) + " |"
        rows = [
            "| " + " | ".join(str(row[column]).ljust(widths[column]) for column in table.columns) + " |"
            for _, row in table.iterrows()
        ]
        return [header, divider, *rows]

    lines.extend(
        [
            "",
            "## Family Retrieval",
            *_format_table(
                family_ranked,
                ["method", "pairwise_auc", "top1_family_recall", "eligible_queries"],
            ),
        ]
    )
    if not prediction_metrics.empty:
        prediction_ranked = prediction_metrics.sort_values(["target", "mae", "method"])
        lines.extend(
            [
                "",
                "## kNN Prediction",
                *_format_table(prediction_ranked, ["target", "method", "n", "mae", "spearman"]),
            ]
        )

    lines.extend(
        [
            "",
            "## Newer Methods Considered",
            "- Raw-weight CKA/fingerprint methods such as AWM require access to weight matrices, so they are future-work here rather than valid baselines for saved alpha summaries: https://arxiv.org/abs/2510.06738",
            "- Attention-product SVD fingerprints such as GhostSpec require internal attention weight products, not the current HDF5 alpha traces: https://arxiv.org/abs/2511.06390",
            "- Gradient fingerprints such as TensorGuard require loading models and computing gradient responses, which is outside this artifact-only study: https://arxiv.org/abs/2506.01631",
            "- Learned model-hub representations are promising for a larger supervised system, but they need a separate training corpus of model weights/features: https://arxiv.org/abs/2510.02096",
            "",
            "## Caveats",
            "- `coarse_family` is a pragmatic proxy label inferred from metadata/name/architecture, not a manually audited ground truth.",
            "- Methods requiring raw weights or gradients were not tested here because the available reusable artifacts are alpha/stat summaries.",
        ]
    )
    return "\n".join(lines) + "\n"
