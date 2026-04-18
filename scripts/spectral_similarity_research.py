from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_research.pipeline import main as pipeline_main, parse_args


def main(argv: list[str] | None = None) -> int:
    return pipeline_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

'''

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau, pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_research.datasets import select_balanced_family_subset
from spectral_research.evaluation import (
    evaluate_clustering,
    evaluate_open_set_classification,
    evaluate_pairwise_similarity,
)
from spectral_research.similarity import (
    module_reliability_weights,
    pearson_z_similarity,
    rank_rbf_similarity,
    spearman_similarity,
    similarity_to_distance,
)
from alpha_cluster_dashboard.clustering import cluster_distance_matrix
from alpha_cluster_dashboard.data import records_to_frame
from alpha_cluster_dashboard.lineage import build_lineage_quality_metrics


DEFAULT_SUMMARY_STATS = ["mean", "std", "median", "q25", "q75", "start", "end", "delta", "slope"]
DEFAULT_METHODS = ["paper_spearman", "pearson_z", "rank_rbf", "hybrid"]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    kind: str
    spearman_weight: float = 0.45
    pearson_weight: float = 0.25
    dtw_weight: float = 0.30
    rank_rbf_sigma: float = 6.0
    dtw_gamma: float = 0.25
    dtw_tau: float = 0.75
    coverage_penalty: float = 0.20


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    max_families: int
    max_models_per_family: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run spectral similarity, open-set, and clustering-stability experiments."
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=REPO_ROOT / "svd_results",
        help="Directory containing per-model alpha metric files (.h5 or .csv).",
    )
    parser.add_argument(
        "--lineage-csv",
        type=Path,
        default=REPO_ROOT / "sampled_metadata.csv",
        help="Path to sampled_metadata.csv with lineage annotations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "research_runs" / "spectral_similarity_research",
        help="Directory where CSV and Markdown outputs will be written.",
    )
    parser.add_argument(
        "--profile-bins",
        type=int,
        default=32,
        help="Normalized depth bins for profile representations.",
    )
    parser.add_argument(
        "--summary-stats",
        type=str,
        default=",".join(DEFAULT_SUMMARY_STATS),
        help="Comma-separated summary statistics to keep for fallback tables.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for sampling and bootstrap evaluation.",
    )
    parser.add_argument(
        "--min-family-size",
        type=int,
        default=8,
        help="Minimum lineage family size to keep in balanced subsets.",
    )
    parser.add_argument(
        "--balanced-max-families",
        type=int,
        default=8,
        help="Number of families to keep in the balanced subset.",
    )
    parser.add_argument(
        "--balanced-max-models-per-family",
        type=int,
        default=12,
        help="Maximum models per family in the balanced subset.",
    )
    parser.add_argument(
        "--include-common-schema",
        action="store_true",
        help="Also run a larger common-schema lineage subset.",
    )
    parser.add_argument(
        "--common-max-families",
        type=int,
        default=20,
        help="Number of families to keep in the larger common-schema subset.",
    )
    parser.add_argument(
        "--common-max-models-per-family",
        type=int,
        default=20,
        help="Maximum models per family in the larger common-schema subset.",
    )
    parser.add_argument(
        "--open-set-repeats",
        type=int,
        default=8,
        help="Number of random open-set trials per dataset and method.",
    )
    parser.add_argument(
        "--open-set-unknown-fraction",
        type=float,
        default=0.25,
        help="Fraction of families treated as unknown in each open-set trial.",
    )
    parser.add_argument(
        "--bootstrap-rounds",
        type=int,
        default=12,
        help="Bootstrap rounds for clustering stability.",
    )
    parser.add_argument(
        "--subsample-fraction",
        type=float,
        default=0.80,
        help="Fraction of models kept in each clustering bootstrap subsample.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated similarity methods to evaluate.",
    )
    return parser.parse_args()


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _safe_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else np.nan


def _format_float(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    numeric = float(value)
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}"


def _format_pct(value: object, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    numeric = float(value)
    if not np.isfinite(numeric):
        return "n/a"
    return f"{100.0 * numeric:.{digits}f}%"


def _markdown_table(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    table = frame.loc[:, columns].copy()
    if limit is not None:
        table = table.head(limit)

    def cell(value: object) -> str:
        if pd.isna(value):
            return "n/a"
        if isinstance(value, (float, np.floating)):
            return _format_float(value)
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for row in table.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join([header, separator, *rows])


def _compact_trace(trace: np.ndarray) -> np.ndarray:
    array = np.asarray(trace, dtype=float)
    return array[np.isfinite(array)]


def _zscore_trace(trace: np.ndarray) -> np.ndarray:
    values = _compact_trace(trace)
    if values.size == 0:
        return values
    std = float(np.std(values))
    if std <= 1e-12:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std


def _softmin(values: np.ndarray, gamma: float) -> float:
    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    if finite.size == 0:
        return np.inf
    minimum = float(np.min(finite))
    return float(minimum - gamma * np.log(np.sum(np.exp(-(finite - minimum) / gamma))))


def _soft_dtw_distance(left: np.ndarray, right: np.ndarray, gamma: float) -> float:
    a = _zscore_trace(left)
    b = _zscore_trace(right)
    if a.size == 0 or b.size == 0:
        return np.nan

    dp = np.full((a.size + 1, b.size + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0
    for i in range(1, a.size + 1):
        for j in range(1, b.size + 1):
            cost = abs(float(a[i - 1]) - float(b[j - 1]))
            dp[i, j] = cost + _softmin(np.array([dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]]), gamma)
    return float(dp[a.size, b.size] / (a.size + b.size))


def _soft_dtw_similarity(left: np.ndarray, right: np.ndarray, gamma: float, tau: float) -> float:
    distance = _soft_dtw_distance(left, right, gamma=gamma)
    if not np.isfinite(distance):
        return np.nan
    return float(2.0 * np.exp(-distance / tau) - 1.0)


def _trace_similarity(left: np.ndarray, right: np.ndarray, spec: MethodSpec) -> float:
    if spec.kind == "paper_spearman":
        return spearman_similarity(left, right)
    if spec.kind == "pearson_z":
        return pearson_z_similarity(left, right)
    if spec.kind == "rank_rbf":
        return rank_rbf_similarity(left, right, sigma=spec.rank_rbf_sigma)
    if spec.kind == "kendall":
        x = np.asarray(left, dtype=float)
        y = np.asarray(right, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(mask) < 3:
            return np.nan
        value = kendalltau(x[mask], y[mask]).statistic
        return np.nan if value is None else float(value)
    if spec.kind == "hybrid":
        components = [
            (spec.spearman_weight, spearman_similarity(left, right)),
            (spec.pearson_weight, pearson_z_similarity(left, right)),
            (spec.dtw_weight, _soft_dtw_similarity(left, right, gamma=spec.dtw_gamma, tau=spec.dtw_tau)),
        ]
        weight_sum = 0.0
        score_sum = 0.0
        for weight, value in components:
            if not np.isfinite(value) or weight <= 0:
                continue
            weight_sum += weight
            score_sum += weight * value
        return float(np.clip(score_sum / weight_sum, -1.0, 1.0)) if weight_sum > 0 else np.nan
    raise ValueError(f"Unsupported similarity kind: {spec.kind}")


def _trace_pair_similarity(
    left_profiles: np.ndarray,
    right_profiles: np.ndarray,
    module_names: list[str],
    weights: np.ndarray,
    spec: MethodSpec,
) -> tuple[float, float]:
    score_sum = 0.0
    weight_sum = 0.0
    intersection = 0
    union = 0
    for module_index, _ in enumerate(module_names):
        left_trace = left_profiles[module_index]
        right_trace = right_profiles[module_index]
        left_valid = np.any(np.isfinite(left_trace))
        right_valid = np.any(np.isfinite(right_trace))
        if left_valid or right_valid:
            union += 1
        if not left_valid or not right_valid:
            continue
        similarity = _trace_similarity(left_trace, right_trace, spec)
        if not np.isfinite(similarity):
            continue
        intersection += 1
        weight = float(weights[module_index])
        score_sum += weight * similarity
        weight_sum += weight
    if weight_sum <= 0:
        return np.nan, 0.0
    coverage = float(intersection / union) if union else 0.0
    score = float(score_sum / weight_sum) - spec.coverage_penalty * (1.0 - coverage)
    return float(np.clip(score, -1.0, 1.0)), coverage


def build_similarity_matrix(
    profiles: np.ndarray,
    module_names: list[str],
    spec: MethodSpec,
) -> tuple[np.ndarray, np.ndarray]:
    num_models = profiles.shape[0]
    weights = module_reliability_weights(profiles)
    similarities = np.eye(num_models, dtype=float)
    coverages = np.ones((num_models, num_models), dtype=float)
    for left_index in range(num_models):
        for right_index in range(left_index + 1, num_models):
            score, coverage = _trace_pair_similarity(
                profiles[left_index],
                profiles[right_index],
                module_names,
                weights,
                spec,
            )
            similarities[left_index, right_index] = score
            similarities[right_index, left_index] = score
            coverages[left_index, right_index] = coverage
            coverages[right_index, left_index] = coverage
    return similarities, coverages


def _family_labels(metadata: pd.DataFrame) -> np.ndarray:
    if "base_model" not in metadata.columns:
        return metadata["display_name"].astype(str).to_numpy()
    labels = metadata["base_model"].fillna(metadata["display_name"]).astype(str)
    return labels.to_numpy()


def _quality_metrics_to_dict(quality_metrics: pd.DataFrame) -> dict[str, float]:
    if quality_metrics.empty:
        return {}
    rows = {}
    for _, row in quality_metrics.iterrows():
        value = row["value"]
        if pd.isna(value):
            continue
        rows[str(row["metric_key"])] = float(value)
    return rows


def _balanced_records_for_dataset(
    records: list,
    lineage_csv: Path,
    schema_mode: str,
    min_family_size: int,
    max_families: int,
    max_models_per_family: int,
    random_state: int,
) -> tuple[list, pd.DataFrame]:
    from alpha_cluster_dashboard.lineage import load_lineage_metadata, merge_lineage_metadata

    metadata = records_to_frame(records)
    lineage = load_lineage_metadata(str(lineage_csv))
    metadata = merge_lineage_metadata(metadata, lineage)
    selected = select_balanced_family_subset(
        metadata=metadata,
        family_col="base_model",
        min_family_size=min_family_size,
        max_families=max_families,
        max_models_per_family=max_models_per_family,
        random_state=random_state,
    )
    selected_names = set(selected["display_name"].astype(str).tolist())
    selected_records = [record for record in records if record.display_name in selected_names]
    selected_records.sort(key=lambda record: record.display_name)
    return selected_records, selected


def _prepare_dataset(
    lineage_csv: Path,
    schema_mode: str,
    records: list,
    profile_bins: int,
) -> tuple[object, pd.DataFrame]:
    from alpha_cluster_dashboard.data import available_modules
    from alpha_cluster_dashboard.features import build_prepared_dataset
    from alpha_cluster_dashboard.lineage import load_lineage_metadata, merge_lineage_metadata

    module_names = available_modules(records, schema_mode)
    prepared = build_prepared_dataset(
        records=records,
        schema_mode=schema_mode,
        selected_modules=module_names,
        representation="profile",
        num_bins=profile_bins,
        summary_stats=DEFAULT_SUMMARY_STATS,
        include_presence_features=False,
        sanitize_non_positive=True,
        log_transform=False,
        zscore_within_trace=False,
    )
    lineage = load_lineage_metadata(str(lineage_csv))
    metadata = merge_lineage_metadata(prepared.metadata, lineage)
    return prepared, metadata


def _dataset_summary_frame(dataset_name: str, prepared, metadata: pd.DataFrame, profile_bins: int) -> pd.DataFrame:
    family_labels = _family_labels(metadata)
    families = pd.Series(family_labels).value_counts()
    profile_coverage = float(np.mean(np.isfinite(prepared.profiles))) if prepared.profiles.size else np.nan
    rows = [
        {
            "dataset": dataset_name,
            "n_models": int(len(metadata)),
            "n_families": int(families.size),
            "min_family_size": int(families.min()) if not families.empty else np.nan,
            "median_family_size": float(families.median()) if not families.empty else np.nan,
            "max_family_size": int(families.max()) if not families.empty else np.nan,
            "n_modules": int(len(prepared.module_names)),
            "profile_bins": int(profile_bins),
            "mean_num_layers": float(metadata["num_layers"].mean()) if "num_layers" in metadata.columns else np.nan,
            "min_num_layers": int(metadata["num_layers"].min()) if "num_layers" in metadata.columns else np.nan,
            "max_num_layers": int(metadata["num_layers"].max()) if "num_layers" in metadata.columns else np.nan,
            "mean_profile_coverage": profile_coverage,
        }
    ]
    return pd.DataFrame(rows)


def _run_method_on_dataset(
    dataset_name: str,
    prepared,
    metadata: pd.DataFrame,
    method: MethodSpec,
    open_set_repeats: int,
    open_set_unknown_fraction: float,
    bootstrap_rounds: int,
    subsample_fraction: float,
    random_state: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    family_labels = _family_labels(metadata)
    similarities, coverages = build_similarity_matrix(prepared.profiles, prepared.module_names, method)
    pairwise = evaluate_pairwise_similarity(similarities, family_labels.tolist())
    pairwise.update(
        {
            "dataset": dataset_name,
            "method": method.name,
            "similarity_kind": method.kind,
            "mean_pair_coverage": _safe_mean(coverages[np.triu_indices_from(coverages, k=1)]),
        }
    )

    open_set = evaluate_open_set_classification(
        similarities=similarities,
        family_labels=family_labels.tolist(),
        n_repeats=open_set_repeats,
        unknown_fraction=open_set_unknown_fraction,
        random_state=random_state,
    )
    open_set.update(
        {
            "dataset": dataset_name,
            "method": method.name,
            "similarity_kind": method.kind,
        }
    )

    distances = similarity_to_distance(similarities)
    family_count = int(pd.Series(family_labels).nunique())
    k = max(2, min(family_count, len(metadata) - 1))
    clustering_eval = evaluate_clustering(
        distances=distances,
        family_labels=family_labels.tolist(),
        n_clusters=k,
        random_state=random_state,
        bootstrap_rounds=bootstrap_rounds,
        subsample_fraction=subsample_fraction,
    )
    cluster_result = cluster_distance_matrix(
        distances=distances,
        algorithm="agglomerative",
        n_clusters=k,
        dbscan_eps=1.0,
        dbscan_min_samples=4,
    )
    lineage_metrics = build_lineage_quality_metrics(metadata=metadata, labels=cluster_result.labels, quality_metrics=cluster_result.quality_metrics)
    lineage_values = {
        row["metric_key"]: float(row["value"])
        for _, row in lineage_metrics.iterrows()
        if pd.notna(row["value"])
    }
    quality_values = {
        f"cluster_{key}": value for key, value in _quality_metrics_to_dict(cluster_result.quality_metrics).items()
    }

    clustering = {
        "dataset": dataset_name,
        "method": method.name,
        "similarity_kind": method.kind,
        "n_clusters": int(k),
        **quality_values,
        **clustering_eval,
        **lineage_values,
    }
    return pairwise, open_set, clustering


def _best_row(frame: pd.DataFrame, group_cols: list[str], score_col: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=frame.columns)
    rows = []
    for _, group in frame.groupby(group_cols, dropna=False):
        valid = group[group[score_col].notna()].copy()
        if valid.empty:
            continue
        rows.append(valid.sort_values(score_col, ascending=False).iloc[0])
    if not rows:
        return pd.DataFrame(columns=frame.columns)
    return pd.DataFrame(rows).reset_index(drop=True)


def _render_report(
    output_path: Path,
    dataset_summary: pd.DataFrame,
    selected_models: pd.DataFrame,
    pairwise_results: pd.DataFrame,
    open_set_results: pd.DataFrame,
    clustering_results: pd.DataFrame,
) -> None:
    lines = [
        "# Spectral Similarity Research",
        "",
        "## Dataset summary",
        _markdown_table(
            dataset_summary,
            [
                "dataset",
                "n_models",
                "n_families",
                "min_family_size",
                "median_family_size",
                "max_family_size",
                "n_modules",
                "profile_bins",
                "mean_profile_coverage",
            ],
        ),
        "",
        "## Selected models",
        _markdown_table(
            selected_models,
            ["dataset", "display_name", "base_model", "base_model_relation", "num_layers"],
            limit=18,
        ),
        "",
        "## Best pairwise similarity per dataset",
    ]
    best_pairwise = _best_row(pairwise_results, ["dataset"], "pairwise_auroc")
    if best_pairwise.empty:
        lines.append("_No valid pairwise results._")
    else:
        lines.append(
            _markdown_table(
                best_pairwise,
                ["dataset", "method", "similarity_kind", "pairwise_auroc", "pairwise_auprc", "top1_family_recall", "mean_pair_coverage"],
            )
        )

    lines.extend(["", "## Best open-set similarity per dataset"])
    best_open = _best_row(open_set_results, ["dataset"], "open_macro_f1")
    if best_open.empty:
        lines.append("_No valid open-set results._")
    else:
        lines.append(
            _markdown_table(
                best_open,
                ["dataset", "method", "similarity_kind", "open_macro_f1", "open_known_accuracy", "open_unknown_recall", "open_known_coverage", "open_threshold"],
            )
        )

    lines.extend(["", "## Best clustering stability per dataset"])
    best_cluster = _best_row(clustering_results, ["dataset"], "lineage_run_score")
    if best_cluster.empty:
        lines.append("_No valid clustering results._")
    else:
        lines.append(
            _markdown_table(
                best_cluster,
                [
                    "dataset",
                    "method",
                    "similarity_kind",
                    "n_clusters",
                    "lineage_run_score",
                    "lineage_v_measure",
                    "cluster_silhouette",
                    "cluster_bootstrap_ari_mean",
                    "cluster_bootstrap_vi_mean",
                    "cluster_noise_fraction",
                    "cluster_largest_cluster_fraction",
                ],
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "- Similarity is computed on normalized-depth profile traces with module-level reliability weights.",
            "- The hybrid method blends rank-order agreement, z-scored Pearson correlation, and a local soft-DTW alignment term.",
            "- The clustering score uses the existing lineage composite plus bootstrap stability and cluster-quality guardrails.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    method_names = _split_csv(args.methods)
    methods = []
    for name in method_names:
        if name == "paper_spearman":
            methods.append(MethodSpec(name=name, kind="paper_spearman"))
        elif name == "pearson_z":
            methods.append(MethodSpec(name=name, kind="pearson_z"))
        elif name == "rank_rbf":
            methods.append(MethodSpec(name=name, kind="rank_rbf"))
        elif name == "kendall":
            methods.append(MethodSpec(name=name, kind="kendall"))
        elif name == "hybrid":
            methods.append(MethodSpec(name=name, kind="hybrid"))
        else:
            raise ValueError(f"Unsupported method: {name}")

    dataset_specs = [
        DatasetSpec(
            name="balanced_lineage_subset",
            max_families=args.balanced_max_families,
            max_models_per_family=args.balanced_max_models_per_family,
        )
    ]
    if args.include_common_schema:
        dataset_specs.append(
            DatasetSpec(
                name="larger_common_schema_subset",
                max_families=args.common_max_families,
                max_models_per_family=args.common_max_models_per_family,
            )
        )

    summary_rows = []
    selected_rows = []
    pairwise_rows = []
    open_set_rows = []
    clustering_rows = []

    from alpha_cluster_dashboard.data import load_alpha_records
    from alpha_cluster_dashboard.features import filter_records

    all_records = load_alpha_records(args.metrics_dir)
    if not all_records:
        raise ValueError(f"No alpha metric files were found in {args.metrics_dir}")
    min_layers = min(record.num_layers for record in all_records)
    max_layers = max(record.num_layers for record in all_records)
    common_records = filter_records(
        all_records,
        schema_mode="common",
        min_layers=min_layers,
        max_layers=max_layers,
        model_query="",
        max_models=None,
        random_state=args.random_state,
    )
    if not common_records:
        raise ValueError("No common-schema records were found after filtering.")

    lineage_csv = args.lineage_csv

    for dataset_spec in dataset_specs:
        dataset_start = perf_counter()
        selected_records, selected_metadata = _balanced_records_for_dataset(
            records=common_records,
            lineage_csv=lineage_csv,
            schema_mode="common",
            min_family_size=args.min_family_size,
            max_families=dataset_spec.max_families,
            max_models_per_family=dataset_spec.max_models_per_family,
            random_state=args.random_state,
        )
        if not selected_records:
            continue

        prepared, metadata = _prepare_dataset(
            lineage_csv=lineage_csv,
            schema_mode="common",
            records=selected_records,
            profile_bins=args.profile_bins,
        )

        summary_row = _dataset_summary_frame(dataset_spec.name, prepared, metadata, args.profile_bins).iloc[0].to_dict()

        selected_frame = metadata.loc[
            :, ["display_name", "base_model", "base_model_relation", "num_layers", "schema_mode"]
        ].copy()
        selected_frame.insert(0, "dataset", dataset_spec.name)
        selected_rows.append(selected_frame)

        for method in methods:
            pairwise_row, open_set_row, clustering_row = _run_method_on_dataset(
                dataset_name=dataset_spec.name,
                prepared=prepared,
                metadata=metadata,
                method=method,
                open_set_repeats=args.open_set_repeats,
                open_set_unknown_fraction=args.open_set_unknown_fraction,
                bootstrap_rounds=args.bootstrap_rounds,
                subsample_fraction=args.subsample_fraction,
                random_state=args.random_state,
            )
            pairwise_rows.append(pairwise_row)
            open_set_rows.append(open_set_row)
            clustering_rows.append(clustering_row)

        dataset_elapsed = perf_counter() - dataset_start
        summary_row["dataset_runtime_seconds"] = float(dataset_elapsed)
        summary_rows.append(summary_row)

    if not summary_rows:
        raise ValueError("No datasets were produced. Check lineage coverage and family-size thresholds.")

    dataset_summary = pd.DataFrame(summary_rows)
    selected_models = pd.concat(selected_rows, ignore_index=True, sort=False) if selected_rows else pd.DataFrame()
    pairwise_results = pd.DataFrame(pairwise_rows)
    open_set_results = pd.DataFrame(open_set_rows)
    clustering_results = pd.DataFrame(clustering_rows)

    dataset_summary = dataset_summary.sort_values(["dataset"]).reset_index(drop=True)
    if not selected_models.empty:
        selected_models = selected_models.sort_values(["dataset", "base_model", "display_name"]).reset_index(drop=True)

    dataset_summary.to_csv(args.output_dir / "dataset_summary.csv", index=False)
    selected_models.to_csv(args.output_dir / "selected_models.csv", index=False)
    pairwise_results.to_csv(args.output_dir / "pairwise_results.csv", index=False)
    open_set_results.to_csv(args.output_dir / "open_set_results.csv", index=False)
    clustering_results.to_csv(args.output_dir / "clustering_results.csv", index=False)

    _render_report(
        output_path=args.output_dir / "report.md",
        dataset_summary=dataset_summary,
        selected_models=selected_models,
        pairwise_results=pairwise_results,
        open_set_results=open_set_results,
        clustering_results=clustering_results,
    )

    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
'''
