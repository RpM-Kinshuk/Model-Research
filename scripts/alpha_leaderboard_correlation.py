from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alpha_cluster_dashboard.data import CANONICAL_MODULE_ORDER, available_modules, load_alpha_records
from alpha_cluster_dashboard.features import build_prepared_dataset

SUMMARY_STATS = ["mean", "std", "median", "slope", "start", "end"]
REPORT_TARGETS = [
    "Average ⬆️",
    "CO₂ cost (kg)",
    "IFEval",
    "BBH",
    "MATH Lvl 5",
    "GPQA",
    "MUSR",
    "MMLU-PRO",
    "#Params (B)",
    "downloads",
    "likes",
]
DEFAULT_COVARIATES = ["#Params (B)", "Generation"]
NON_SIGNAL_AGGREGATE_FEATURES = {
    "alpha_num_layers",
    "alpha_raw_module_count",
    "alpha_valid_count",
    "alpha_total_count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correlate alpha-derived SVD metrics with leaderboard metrics for matched models."
    )
    parser.add_argument(
        "--leaderboard-csv",
        type=Path,
        default=REPO_ROOT / "data" / "sampled_leaderboard.csv",
        help="Path to sampled_leaderboard.csv",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=REPO_ROOT / "svd_results",
        help="Directory containing per-model alpha metric files (.h5 or .csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "research_runs" / "legacy" / "alpha_leaderboard",
        help="Directory where CSV and markdown outputs will be written",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=16,
        help="Normalized depth bins for the clustering-style alpha summaries",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=3,
        help="Number of alpha principal components to export",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=25,
        help="Minimum number of paired observations required for a correlation",
    )
    return parser.parse_args()


def first_non_null(series: pd.Series) -> object:
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def collapse_string_values(series: pd.Series) -> object:
    non_null = [str(value) for value in series.dropna().tolist()]
    if not non_null:
        return np.nan
    unique = list(dict.fromkeys(non_null))
    if len(unique) == 1:
        return unique[0]
    return " | ".join(unique)


def deduplicate_leaderboard(leaderboard: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    duplicate_rows = int(leaderboard["model_id"].duplicated(keep=False).sum())
    numeric_columns = []
    object_columns = []
    for column in leaderboard.columns:
        series = leaderboard[column]
        if column == "model_id":
            continue
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric_columns.append(column)
        else:
            object_columns.append(column)

    aggregation: dict[str, object] = {column: "mean" for column in numeric_columns}
    aggregation.update({column: collapse_string_values for column in object_columns})
    deduped = leaderboard.groupby("model_id", as_index=False).agg(aggregation)
    counts = leaderboard.groupby("model_id").size().rename("leaderboard_row_count").reset_index()
    deduped = deduped.merge(counts, on="model_id", how="left")
    return deduped, duplicate_rows


def sanitize_alpha_values(values: np.ndarray) -> np.ndarray:
    cleaned = np.asarray(values, dtype=float).copy()
    cleaned[~np.isfinite(cleaned)] = np.nan
    cleaned[cleaned <= 0] = np.nan
    return cleaned


def summarize_global_alpha(record_alpha: np.ndarray) -> dict[str, float]:
    cleaned = sanitize_alpha_values(record_alpha).ravel()
    total_values = int(cleaned.size)
    valid_values = cleaned[np.isfinite(cleaned)]
    if valid_values.size == 0:
        return {
            "alpha_global_mean": np.nan,
            "alpha_global_median": np.nan,
            "alpha_global_std": np.nan,
            "alpha_global_q25": np.nan,
            "alpha_global_q75": np.nan,
            "alpha_global_iqr": np.nan,
            "alpha_global_mean_signed_distance_to_2": np.nan,
            "alpha_global_mean_abs_distance_to_2": np.nan,
            "alpha_global_median_abs_distance_to_2": np.nan,
            "alpha_global_fraction_lt_2": np.nan,
            "alpha_global_fraction_gt_2": np.nan,
            "alpha_valid_fraction": 0.0,
            "alpha_valid_count": 0.0,
            "alpha_total_count": float(total_values),
        }
    q25 = float(np.nanpercentile(valid_values, 25))
    q75 = float(np.nanpercentile(valid_values, 75))
    distance_to_two = valid_values - 2.0
    return {
        "alpha_global_mean": float(np.nanmean(valid_values)),
        "alpha_global_median": float(np.nanmedian(valid_values)),
        "alpha_global_std": float(np.nanstd(valid_values)),
        "alpha_global_q25": q25,
        "alpha_global_q75": q75,
        "alpha_global_iqr": q75 - q25,
        "alpha_global_mean_signed_distance_to_2": float(np.nanmean(distance_to_two)),
        "alpha_global_mean_abs_distance_to_2": float(np.nanmean(np.abs(distance_to_two))),
        "alpha_global_median_abs_distance_to_2": float(np.nanmedian(np.abs(distance_to_two))),
        "alpha_global_fraction_lt_2": float(np.mean(valid_values < 2.0)),
        "alpha_global_fraction_gt_2": float(np.mean(valid_values > 2.0)),
        "alpha_valid_fraction": float(valid_values.size / total_values) if total_values else np.nan,
        "alpha_valid_count": float(valid_values.size),
        "alpha_total_count": float(total_values),
    }


def build_alpha_feature_tables(
    records: list,
    num_bins: int,
    pca_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[float], list[str]]:
    selected_modules = [
        module
        for module in CANONICAL_MODULE_ORDER
        if module in available_modules(records, schema_mode="canonical")
    ]
    prepared = build_prepared_dataset(
        records=records,
        schema_mode="canonical",
        selected_modules=selected_modules,
        representation="summary",
        num_bins=num_bins,
        summary_stats=SUMMARY_STATS,
        include_presence_features=True,
        sanitize_non_positive=True,
        log_transform=False,
        zscore_within_trace=False,
    )

    metadata = prepared.metadata[["model_id", "display_name", "num_layers", "raw_module_count", "common_schema"]].copy()
    metadata = metadata.rename(
        columns={
            "model_id": "alpha_model_id",
            "num_layers": "alpha_num_layers",
            "raw_module_count": "alpha_raw_module_count",
        }
    )
    detailed = pd.DataFrame(prepared.feature_matrix, columns=prepared.feature_names)
    detailed = pd.concat([metadata, detailed], axis=1)

    aggregate = metadata.copy()
    raw_summaries = pd.DataFrame(
        [summarize_global_alpha(record.alpha) for record in records]
    )
    aggregate = pd.concat([aggregate.reset_index(drop=True), raw_summaries], axis=1)

    summary_columns = [
        column
        for column in prepared.feature_names
        if not column.endswith("__present")
    ]
    presence_columns = [column for column in prepared.feature_names if column.endswith("__present")]
    detailed_summary = detailed[summary_columns]
    for stat in SUMMARY_STATS:
        stat_columns = [column for column in summary_columns if column.endswith(f"__{stat}")]
        if not stat_columns:
            continue
        aggregate[f"alpha_{stat}_across_modules"] = detailed_summary[stat_columns].mean(axis=1, skipna=True)
        aggregate[f"alpha_{stat}_module_spread"] = detailed_summary[stat_columns].std(axis=1, skipna=True)
        if stat in {"mean", "median", "start", "end"}:
            aggregate[f"alpha_{stat}_across_modules_signed_distance_to_2"] = (
                aggregate[f"alpha_{stat}_across_modules"] - 2.0
            )
            aggregate[f"alpha_{stat}_across_modules_abs_distance_to_2"] = (
                aggregate[f"alpha_{stat}_across_modules"] - 2.0
            ).abs()

    aggregate["alpha_end_minus_start_across_modules"] = (
        aggregate["alpha_end_across_modules"] - aggregate["alpha_start_across_modules"]
    )
    aggregate["alpha_presence_fraction"] = detailed[presence_columns].mean(axis=1, skipna=True) if presence_columns else 1.0

    usable_feature_columns = []
    for column in prepared.feature_names:
        series = detailed[column]
        finite = series[np.isfinite(series)]
        if finite.empty:
            continue
        if finite.nunique(dropna=True) < 2:
            continue
        usable_feature_columns.append(column)

    pca_explained_variance: list[float] = []
    if usable_feature_columns:
        imputed = SimpleImputer(strategy="median").fit_transform(detailed[usable_feature_columns])
        scaled = StandardScaler().fit_transform(imputed)
        component_count = min(pca_components, scaled.shape[0], scaled.shape[1])
        if component_count >= 1:
            pca = PCA(n_components=component_count, random_state=0)
            scores = pca.fit_transform(scaled)
            pca_explained_variance = pca.explained_variance_ratio_.tolist()
            for component_index in range(component_count):
                aggregate[f"alpha_pc{component_index + 1}"] = scores[:, component_index]
    return aggregate, detailed, pca_explained_variance, usable_feature_columns


def benjamini_hochberg(pvalues: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(pvalues), dtype=float)
    adjusted = np.full(array.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(array)
    if not np.any(finite_mask):
        return adjusted
    finite_values = array[finite_mask]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    total = ranked.size
    scaled = ranked * total / np.arange(1, total + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    scaled = np.clip(scaled, 0.0, 1.0)
    unsorted = np.empty_like(scaled)
    unsorted[order] = scaled
    adjusted[finite_mask] = unsorted
    return adjusted


def rank_array(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def residualize(response: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(response.shape[0], dtype=float), covariates])
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    return response - design @ coefficients


def compute_partial_spearman(
    x: np.ndarray,
    y: np.ndarray,
    covariates: np.ndarray | None,
    min_pairs: int,
) -> tuple[float | None, float | None, int]:
    if covariates is None or covariates.size == 0:
        return None, None, 0
    matrix = np.column_stack([x, y, covariates])
    finite_mask = np.all(np.isfinite(matrix), axis=1)
    pair_count = int(np.count_nonzero(finite_mask))
    if pair_count < min_pairs:
        return None, None, pair_count
    x_values = x[finite_mask]
    y_values = y[finite_mask]
    covariate_values = covariates[finite_mask]
    if np.nanstd(x_values) < 1e-12 or np.nanstd(y_values) < 1e-12:
        return None, None, pair_count

    ranked_x = rank_array(x_values)
    ranked_y = rank_array(y_values)
    ranked_covariates = np.column_stack([rank_array(covariate_values[:, index]) for index in range(covariate_values.shape[1])])
    x_residual = residualize(ranked_x, ranked_covariates)
    y_residual = residualize(ranked_y, ranked_covariates)
    if np.nanstd(x_residual) < 1e-12 or np.nanstd(y_residual) < 1e-12:
        return None, None, pair_count
    rho, pvalue = pearsonr(x_residual, y_residual)
    return float(rho), float(pvalue), pair_count


def correlate_pair(
    x: pd.Series,
    y: pd.Series,
    covariates: pd.DataFrame | None,
    min_pairs: int,
) -> dict[str, float | int | str | None]:
    paired = pd.concat([x, y], axis=1)
    paired.columns = ["x", "y"]
    valid = paired.replace([np.inf, -np.inf], np.nan).dropna()
    pair_count = int(len(valid))
    result: dict[str, float | int | str | None] = {
        "n_pairs": pair_count,
        "pearson_r": None,
        "pearson_p": None,
        "spearman_rho": None,
        "spearman_p": None,
        "partial_spearman_rho": None,
        "partial_spearman_p": None,
        "partial_n_pairs": None,
    }
    if pair_count < min_pairs:
        return result

    x_values = valid["x"].to_numpy(dtype=float)
    y_values = valid["y"].to_numpy(dtype=float)
    if np.nanstd(x_values) < 1e-12 or np.nanstd(y_values) < 1e-12:
        return result

    pearson_r, pearson_p = pearsonr(x_values, y_values)
    spearman_rho, spearman_p = spearmanr(x_values, y_values)
    result["pearson_r"] = float(pearson_r)
    result["pearson_p"] = float(pearson_p)
    result["spearman_rho"] = float(spearman_rho)
    result["spearman_p"] = float(spearman_p)

    if covariates is None or covariates.empty:
        return result

    covariate_frame = covariates.loc[valid.index].replace([np.inf, -np.inf], np.nan)
    partial_rho, partial_p, partial_pairs = compute_partial_spearman(
        x=x_values,
        y=y_values,
        covariates=covariate_frame.to_numpy(dtype=float),
        min_pairs=min_pairs,
    )
    result["partial_spearman_rho"] = partial_rho
    result["partial_spearman_p"] = partial_p
    result["partial_n_pairs"] = partial_pairs
    return result


def feature_group(feature_name: str) -> str:
    if feature_name.startswith("alpha_pc"):
        return "alpha_pca"
    if feature_name.startswith("alpha_global"):
        return "global_alpha"
    if feature_name.startswith("alpha_") and "__" not in feature_name:
        return "aggregate_alpha"
    if feature_name.endswith("__present"):
        return "module_presence"
    return "module_summary"


def compute_correlations(
    joined: pd.DataFrame,
    alpha_columns: list[str],
    target_columns: list[str],
    min_pairs: int,
) -> pd.DataFrame:
    rows = []
    for target_column in target_columns:
        available_covariates = [
            covariate
            for covariate in DEFAULT_COVARIATES
            if covariate in joined.columns and covariate != target_column
        ]
        covariate_frame = joined[available_covariates] if available_covariates else None
        for alpha_column in alpha_columns:
            result = correlate_pair(
                x=joined[alpha_column],
                y=joined[target_column],
                covariates=covariate_frame,
                min_pairs=min_pairs,
            )
            rows.append(
                {
                    "alpha_feature": alpha_column,
                    "feature_group": feature_group(alpha_column),
                    "target_metric": target_column,
                    "covariates": ", ".join(available_covariates),
                    **result,
                }
            )

    correlations = pd.DataFrame(rows)
    if correlations.empty:
        return correlations
    correlations["abs_spearman_rho"] = correlations["spearman_rho"].abs()
    correlations["abs_partial_spearman_rho"] = correlations["partial_spearman_rho"].abs()
    correlations["spearman_fdr_q"] = benjamini_hochberg(correlations["spearman_p"])
    correlations["partial_spearman_fdr_q"] = benjamini_hochberg(correlations["partial_spearman_p"])
    correlations = correlations.sort_values(
        by=["target_metric", "abs_spearman_rho", "abs_partial_spearman_rho"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return correlations


def select_top_by_target(correlations: pd.DataFrame, score_column: str) -> pd.DataFrame:
    rows = []
    for target_metric, group in correlations.groupby("target_metric"):
        valid = group[group[score_column].notna()].copy()
        if valid.empty:
            continue
        top = valid.sort_values(by=score_column, ascending=False).iloc[0]
        rows.append(top)
    if not rows:
        return pd.DataFrame(columns=correlations.columns)
    return pd.DataFrame(rows).sort_values(by=score_column, ascending=False).reset_index(drop=True)


def format_float(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    numeric = float(value)
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}"


def build_report(
    output_path: Path,
    joined: pd.DataFrame,
    duplicate_rows: int,
    aggregate_correlations: pd.DataFrame,
    detailed_correlations: pd.DataFrame,
    pca_explained_variance: list[float],
) -> None:
    matched_models = int(joined["model_id"].nunique())
    common_schema_models = int(joined["common_schema"].fillna(False).sum())
    key_targets = [target for target in REPORT_TARGETS if target in joined.columns]

    key_aggregate = aggregate_correlations[
        aggregate_correlations["target_metric"].isin(key_targets)
        & ~aggregate_correlations["alpha_feature"].isin(NON_SIGNAL_AGGREGATE_FEATURES)
    ].copy()
    key_detailed = detailed_correlations[detailed_correlations["target_metric"].isin(key_targets)].copy()
    best_aggregate = select_top_by_target(key_aggregate, "abs_spearman_rho")
    best_partial = select_top_by_target(key_detailed, "abs_partial_spearman_rho")

    lines = [
        "# Alpha vs Leaderboard Correlation Report",
        "",
        "## Dataset summary",
        f"- Matched unique models: {matched_models}",
        f"- Common-schema matched models: {common_schema_models}",
        f"- Duplicate leaderboard rows collapsed before matching: {duplicate_rows}",
        f"- PCA explained variance: {', '.join(format_float(value, 3) for value in pca_explained_variance) if pca_explained_variance else 'n/a'}",
        "",
        "## Strongest raw alpha correlations by target",
    ]

    if best_aggregate.empty:
        lines.append("- No raw aggregate correlations passed the minimum pair threshold.")
    else:
        for _, row in best_aggregate.iterrows():
            lines.append(
                "- "
                f"{row['target_metric']}: {row['alpha_feature']} "
                f"(Spearman={format_float(row['spearman_rho'])}, "
                f"q={format_float(row['spearman_fdr_q'])}, "
                f"n={int(row['n_pairs'])})"
            )

    lines.extend(["", "## Strongest size-adjusted alpha correlations by target"])
    if best_partial.empty:
        lines.append("- No size-adjusted correlations passed the minimum pair threshold.")
    else:
        for _, row in best_partial.iterrows():
            lines.append(
                "- "
                f"{row['target_metric']}: {row['alpha_feature']} "
                f"(partial Spearman={format_float(row['partial_spearman_rho'])}, "
                f"q={format_float(row['partial_spearman_fdr_q'])}, "
                f"covariates={row['covariates'] or 'none'}, "
                f"n={int(row['partial_n_pairs']) if pd.notna(row['partial_n_pairs']) else 0})"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "- Raw correlations can reflect model size and generation effects, so the partial-Spearman section adjusts for `#Params (B)` and `Generation` when possible.",
            "- The alpha feature space is built from the same canonical summary representation used in the clustering dashboard: per-module summary stats over sanitized normalized-depth traces plus module-presence features.",
            "- This run also includes alpha-distance-to-2 features, motivated by the HTSR/SETOL interpretation that deviations on either side of 2 can indicate different non-ideal training regimes.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    leaderboard = pd.read_csv(args.leaderboard_csv)
    leaderboard_deduped, duplicate_rows = deduplicate_leaderboard(leaderboard)

    records = load_alpha_records(args.metrics_dir)
    record_by_display_name = {record.display_name: record for record in records}
    matched_display_names = sorted(set(leaderboard_deduped["model_id"]) & set(record_by_display_name))
    if not matched_display_names:
        raise SystemExit("No common models were found between the leaderboard and alpha metric files.")

    matched_records = [record_by_display_name[display_name] for display_name in matched_display_names]
    aggregate_alpha, detailed_alpha, pca_explained_variance, usable_feature_columns = build_alpha_feature_tables(
        matched_records,
        num_bins=args.num_bins,
        pca_components=args.pca_components,
    )

    joined = leaderboard_deduped.merge(
        aggregate_alpha,
        left_on="model_id",
        right_on="display_name",
        how="inner",
        suffixes=("", "_alpha"),
    )
    joined = joined.sort_values("model_id").reset_index(drop=True)

    detailed_joined = joined[["model_id"]].merge(
        detailed_alpha,
        left_on="model_id",
        right_on="display_name",
        how="inner",
        suffixes=("", "_alpha"),
    )

    leaderboard_numeric_columns = [
        column
        for column in leaderboard_deduped.columns
        if pd.api.types.is_numeric_dtype(leaderboard_deduped[column])
        and leaderboard_deduped[column].notna().sum() >= args.min_pairs
        and column not in {"leaderboard_row_count"}
    ]
    aggregate_alpha_columns = [
        column
        for column in aggregate_alpha.columns
        if column.startswith("alpha_") and column not in {"alpha_model_id"}
    ]
    detailed_alpha_columns = usable_feature_columns.copy()

    aggregate_correlations = compute_correlations(
        joined=joined,
        alpha_columns=aggregate_alpha_columns,
        target_columns=leaderboard_numeric_columns,
        min_pairs=args.min_pairs,
    )
    detailed_correlations = compute_correlations(
        joined=detailed_joined.merge(
            joined[["model_id", *leaderboard_numeric_columns]],
            on="model_id",
            how="left",
        ),
        alpha_columns=detailed_alpha_columns,
        target_columns=leaderboard_numeric_columns,
        min_pairs=args.min_pairs,
    )

    raw_best_by_target = select_top_by_target(
        aggregate_correlations[
            ~aggregate_correlations["alpha_feature"].isin(NON_SIGNAL_AGGREGATE_FEATURES)
        ],
        "abs_spearman_rho",
    )
    partial_best_by_target = select_top_by_target(detailed_correlations, "abs_partial_spearman_rho")

    joined.to_csv(args.output_dir / "matched_models_with_alpha_features.csv", index=False)
    detailed_alpha.to_csv(args.output_dir / "alpha_module_summary_features.csv", index=False)
    aggregate_correlations.to_csv(args.output_dir / "aggregate_alpha_correlations.csv", index=False)
    detailed_correlations.to_csv(args.output_dir / "detailed_alpha_correlations.csv", index=False)
    raw_best_by_target.to_csv(args.output_dir / "top_raw_correlations_by_target.csv", index=False)
    partial_best_by_target.to_csv(args.output_dir / "top_partial_correlations_by_target.csv", index=False)
    pd.DataFrame(
        {
            "alpha_pc": [f"alpha_pc{index + 1}" for index in range(len(pca_explained_variance))],
            "explained_variance_ratio": pca_explained_variance,
        }
    ).to_csv(args.output_dir / "alpha_pca_summary.csv", index=False)
    build_report(
        output_path=args.output_dir / "report.md",
        joined=joined,
        duplicate_rows=duplicate_rows,
        aggregate_correlations=aggregate_correlations,
        detailed_correlations=detailed_correlations,
        pca_explained_variance=pca_explained_variance,
    )

    print(f"Matched unique models: {joined['model_id'].nunique()}")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
