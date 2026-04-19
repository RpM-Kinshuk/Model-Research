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
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alpha_cluster_dashboard.data import COMMON_ALPHA_MODULES, load_alpha_records
from alpha_cluster_dashboard.features import build_prepared_dataset

SUMMARY_STATS = ["mean", "std", "median", "q25", "q75", "start", "end", "delta", "slope"]
PERFORMANCE_TARGETS = [
    "Average ⬆️",
    "IFEval",
    "BBH",
    "MATH Lvl 5",
    "GPQA",
    "MUSR",
    "MMLU-PRO",
]
NON_ALPHA_METRIC_CANDIDATES = [
    "#Params (B)",
    "Generation",
    "CO₂ cost (kg)",
    "downloads",
    "likes",
    "Hub ❤️",
]
CONTROL_COVARIATES = ["#Params (B)", "Generation"]
MLP_MODULES = ["mlp.down_proj", "mlp.gate_proj", "mlp.up_proj"]
ATTN_MODULES = ["self_attn.k_proj", "self_attn.o_proj", "self_attn.q_proj", "self_attn.v_proj"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research common-schema alpha metrics versus leaderboard performance."
    )
    parser.add_argument(
        "--leaderboard-csv",
        type=Path,
        default=REPO_ROOT / "sampled_leaderboard.csv",
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
        default=REPO_ROOT / "research_runs" / "legacy" / "common_schema_performance_research",
        help="Directory where CSV and markdown outputs will be written",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=16,
        help="Normalized depth bins for common-schema alpha summaries",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=5,
        help="Number of common-schema alpha principal components to compute",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=25,
        help="Minimum number of paired observations required for a univariate correlation",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of cross-validation folds for predictive models",
    )
    return parser.parse_args()


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
        if column == "model_id":
            continue
        series = leaderboard[column]
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


def summarize_global_alpha(alpha: np.ndarray) -> dict[str, float]:
    cleaned = sanitize_alpha_values(alpha).ravel()
    valid = cleaned[np.isfinite(cleaned)]
    if valid.size == 0:
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
        }
    q25 = float(np.nanpercentile(valid, 25))
    q75 = float(np.nanpercentile(valid, 75))
    distance_to_two = valid - 2.0
    return {
        "alpha_global_mean": float(np.nanmean(valid)),
        "alpha_global_median": float(np.nanmedian(valid)),
        "alpha_global_std": float(np.nanstd(valid)),
        "alpha_global_q25": q25,
        "alpha_global_q75": q75,
        "alpha_global_iqr": q75 - q25,
        "alpha_global_mean_signed_distance_to_2": float(np.nanmean(distance_to_two)),
        "alpha_global_mean_abs_distance_to_2": float(np.nanmean(np.abs(distance_to_two))),
        "alpha_global_median_abs_distance_to_2": float(np.nanmedian(np.abs(distance_to_two))),
        "alpha_global_fraction_lt_2": float(np.mean(valid < 2.0)),
        "alpha_global_fraction_gt_2": float(np.mean(valid > 2.0)),
    }


def usable_numeric_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    usable = []
    for column in columns:
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        finite = series[np.isfinite(series)]
        if finite.empty:
            continue
        if finite.nunique(dropna=True) < 2:
            continue
        usable.append(column)
    return usable


def build_common_schema_feature_tables(
    records: list,
    num_bins: int,
    pca_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[float]]:
    prepared = build_prepared_dataset(
        records=records,
        schema_mode="common",
        selected_modules=list(COMMON_ALPHA_MODULES),
        representation="summary",
        num_bins=num_bins,
        summary_stats=SUMMARY_STATS,
        include_presence_features=False,
        sanitize_non_positive=True,
        log_transform=False,
        zscore_within_trace=False,
    )

    metadata = prepared.metadata[["model_id", "display_name", "num_layers"]].copy()
    metadata = metadata.rename(columns={"model_id": "alpha_model_id", "num_layers": "alpha_num_layers"})
    module_features = pd.DataFrame(prepared.feature_matrix, columns=prepared.feature_names)
    detailed = pd.concat([metadata.reset_index(drop=True), module_features], axis=1)

    aggregate = metadata.copy()
    aggregate = pd.concat(
        [aggregate.reset_index(drop=True), pd.DataFrame([summarize_global_alpha(record.alpha) for record in records])],
        axis=1,
    )

    for stat in SUMMARY_STATS:
        stat_columns = [f"{module}__{stat}" for module in COMMON_ALPHA_MODULES if f"{module}__{stat}" in detailed.columns]
        mlp_columns = [f"{module}__{stat}" for module in MLP_MODULES if f"{module}__{stat}" in detailed.columns]
        attn_columns = [f"{module}__{stat}" for module in ATTN_MODULES if f"{module}__{stat}" in detailed.columns]
        if stat_columns:
            aggregate[f"alpha_{stat}_across_modules"] = detailed[stat_columns].mean(axis=1, skipna=True)
            aggregate[f"alpha_{stat}_module_spread"] = detailed[stat_columns].std(axis=1, skipna=True)
            if stat in {"mean", "median", "start", "end"}:
                aggregate[f"alpha_{stat}_across_modules_signed_distance_to_2"] = (
                    aggregate[f"alpha_{stat}_across_modules"] - 2.0
                )
                aggregate[f"alpha_{stat}_across_modules_abs_distance_to_2"] = (
                    aggregate[f"alpha_{stat}_across_modules"] - 2.0
                ).abs()
        if mlp_columns:
            aggregate[f"alpha_{stat}_mlp_mean"] = detailed[mlp_columns].mean(axis=1, skipna=True)
        if attn_columns:
            aggregate[f"alpha_{stat}_attn_mean"] = detailed[attn_columns].mean(axis=1, skipna=True)
        if mlp_columns and attn_columns:
            aggregate[f"alpha_{stat}_attn_minus_mlp"] = (
                aggregate[f"alpha_{stat}_attn_mean"] - aggregate[f"alpha_{stat}_mlp_mean"]
            )

    module_feature_columns = usable_numeric_columns(detailed, list(module_features.columns))
    explained_variance: list[float] = []
    if module_feature_columns:
        imputed = SimpleImputer(strategy="median").fit_transform(detailed[module_feature_columns])
        scaled = StandardScaler().fit_transform(imputed)
        component_count = min(pca_components, scaled.shape[0], scaled.shape[1])
        if component_count >= 1:
            pca = PCA(n_components=component_count, random_state=0)
            scores = pca.fit_transform(scaled)
            explained_variance = pca.explained_variance_ratio_.tolist()
            for component_index in range(component_count):
                aggregate[f"alpha_pc{component_index + 1}"] = scores[:, component_index]
    return aggregate, detailed, explained_variance


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


def univariate_correlation_table(
    frame: pd.DataFrame,
    predictor_columns: list[str],
    target_columns: list[str],
    min_pairs: int,
) -> pd.DataFrame:
    rows = []
    for target in target_columns:
        covariate_columns = [
            column
            for column in CONTROL_COVARIATES
            if column in frame.columns and column != target
        ]
        covariates = frame[covariate_columns] if covariate_columns else None
        for predictor in predictor_columns:
            paired = frame[[predictor, target]].replace([np.inf, -np.inf], np.nan).dropna()
            pair_count = int(len(paired))
            row = {
                "predictor": predictor,
                "target_metric": target,
                "n_pairs": pair_count,
                "pearson_r": np.nan,
                "pearson_p": np.nan,
                "spearman_rho": np.nan,
                "spearman_p": np.nan,
                "partial_spearman_rho": np.nan,
                "partial_spearman_p": np.nan,
                "partial_n_pairs": np.nan,
            }
            if pair_count >= min_pairs:
                x = paired[predictor].to_numpy(dtype=float)
                y = paired[target].to_numpy(dtype=float)
                if np.nanstd(x) >= 1e-12 and np.nanstd(y) >= 1e-12:
                    pearson_r, pearson_p = pearsonr(x, y)
                    spearman_rho, spearman_p = spearmanr(x, y)
                    row["pearson_r"] = float(pearson_r)
                    row["pearson_p"] = float(pearson_p)
                    row["spearman_rho"] = float(spearman_rho)
                    row["spearman_p"] = float(spearman_p)
                    if covariates is not None and not covariates.empty:
                        partial_rho, partial_p, partial_pairs = compute_partial_spearman(
                            x=x,
                            y=y,
                            covariates=covariates.loc[paired.index].to_numpy(dtype=float),
                            min_pairs=min_pairs,
                        )
                        row["partial_spearman_rho"] = partial_rho
                        row["partial_spearman_p"] = partial_p
                        row["partial_n_pairs"] = partial_pairs
            rows.append(row)

    correlations = pd.DataFrame(rows)
    correlations["abs_spearman_rho"] = correlations["spearman_rho"].abs()
    correlations["abs_partial_spearman_rho"] = correlations["partial_spearman_rho"].abs()
    correlations["spearman_fdr_q"] = benjamini_hochberg(correlations["spearman_p"])
    correlations["partial_spearman_fdr_q"] = benjamini_hochberg(correlations["partial_spearman_p"])
    return correlations.sort_values(
        by=["target_metric", "abs_partial_spearman_rho", "abs_spearman_rho"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def evaluate_feature_family(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    cv_splits: int,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    usable_columns = usable_numeric_columns(frame, feature_columns)
    if target_column not in frame.columns:
        return {}, pd.DataFrame()
    subset = frame[usable_columns + [target_column]].replace([np.inf, -np.inf], np.nan)
    subset = subset[subset[target_column].notna()].copy()
    if len(subset) < max(40, cv_splits * 8) or not usable_columns:
        return {}, pd.DataFrame()

    x = subset[usable_columns]
    y = subset[target_column].to_numpy(dtype=float)
    split_count = min(cv_splits, len(subset))
    if split_count < 3:
        return {}, pd.DataFrame()

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ]
    )
    cv = KFold(n_splits=split_count, shuffle=True, random_state=0)
    predictions = cross_val_predict(pipeline, x, y, cv=cv)
    spearman_rho, _ = spearmanr(y, predictions)
    pearson_r, _ = pearsonr(y, predictions)
    metrics_row = {
        "target_metric": target_column,
        "n_models": int(len(subset)),
        "n_features": int(len(usable_columns)),
        "cv_r2": float(r2_score(y, predictions)),
        "cv_rmse": float(np.sqrt(mean_squared_error(y, predictions))),
        "cv_spearman": float(spearman_rho),
        "cv_pearson": float(pearson_r),
    }

    pipeline.fit(x, y)
    model = pipeline.named_steps["model"]
    coefficients = pd.DataFrame(
        {
            "target_metric": target_column,
            "feature": usable_columns,
            "coefficient": model.coef_,
        }
    )
    coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
    coefficients = coefficients.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    return metrics_row, coefficients


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
    pca_explained_variance: list[float],
    alpha_best: pd.DataFrame,
    overall_best: pd.DataFrame,
    family_scores: pd.DataFrame,
    top_coefficients: pd.DataFrame,
) -> None:
    lines = [
        "# Common-Schema Alpha vs Performance Research",
        "",
        "## Dataset summary",
        f"- Matched unique common-schema models: {joined['model_id'].nunique()}",
        f"- Duplicate leaderboard rows collapsed before matching: {duplicate_rows}",
        f"- Common-schema alpha PCs explained variance: {', '.join(format_float(value) for value in pca_explained_variance) if pca_explained_variance else 'n/a'}",
        "",
        "## Best alpha-only univariate signals",
    ]

    for target in PERFORMANCE_TARGETS:
        rows = alpha_best[alpha_best["target_metric"] == target]
        if rows.empty:
            continue
        row = rows.iloc[0]
        lines.append(
            "- "
            f"{target}: {row['predictor']} "
            f"(Spearman={format_float(row['spearman_rho'])}, "
            f"partial={format_float(row['partial_spearman_rho'])}, "
            f"q={format_float(row['partial_spearman_fdr_q'])}, "
            f"n={int(row['n_pairs'])})"
        )

    lines.extend(["", "## Best overall analysis-metric signals"])
    for target in PERFORMANCE_TARGETS:
        rows = overall_best[overall_best["target_metric"] == target]
        if rows.empty:
            continue
        row = rows.iloc[0]
        lines.append(
            "- "
            f"{target}: {row['predictor']} "
            f"(Spearman={format_float(row['spearman_rho'])}, "
            f"partial={format_float(row['partial_spearman_rho'])}, "
            f"q={format_float(row['partial_spearman_fdr_q'])})"
        )

    lines.extend(["", "## Predictive families"])
    for target in PERFORMANCE_TARGETS:
        rows = family_scores[family_scores["target_metric"] == target].sort_values("cv_r2", ascending=False)
        if rows.empty:
            continue
        best = rows.iloc[0]
        structure = rows[rows["family"] == "nonalpha_metrics"]
        delta = np.nan
        if not structure.empty:
            delta = float(best["cv_r2"] - structure.iloc[0]["cv_r2"])
        lines.append(
            "- "
            f"{target}: best family={best['family']} "
            f"(CV R2={format_float(best['cv_r2'])}, CV Spearman={format_float(best['cv_spearman'])}, "
            f"delta_vs_nonalpha={format_float(delta)})"
        )

    lines.extend(["", "## Top features from the best family per target"])
    for target in PERFORMANCE_TARGETS:
        rows = family_scores[family_scores["target_metric"] == target].sort_values("cv_r2", ascending=False)
        if rows.empty:
            continue
        best_family = rows.iloc[0]["family"]
        coef_rows = top_coefficients[
            (top_coefficients["target_metric"] == target) & (top_coefficients["family"] == best_family)
        ].head(3)
        if coef_rows.empty:
            continue
        feature_text = ", ".join(
            f"{row['feature']} ({format_float(row['coefficient'])})" for _, row in coef_rows.iterrows()
        )
        lines.append(f"- {target}: {best_family} -> {feature_text}")

    lines.extend(
        [
            "",
            "## Notes",
            "- Univariate alpha rows report both raw Spearman and partial Spearman after controlling for `#Params (B)` and `Generation` when those columns are available.",
            "- Predictive families are evaluated out-of-fold with standardized Ridge regression on the common-schema subset only.",
            "- `nonalpha_metrics` includes size and leaderboard-side analysis metrics such as params, generation, CO2, downloads, likes, and alpha-derived layer count when available.",
            "- This run also tests alpha-distance-to-2 aggregates, motivated by the HTSR/SETOL idea that alpha near 2 can indicate more ideal layer quality than either side's deviation.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    leaderboard = pd.read_csv(args.leaderboard_csv)
    leaderboard_deduped, duplicate_rows = deduplicate_leaderboard(leaderboard)

    records = [record for record in load_alpha_records(args.metrics_dir) if record.common_schema]
    record_by_display_name = {record.display_name: record for record in records}
    matched_names = sorted(set(leaderboard_deduped["model_id"]) & set(record_by_display_name))
    if not matched_names:
        raise SystemExit("No common-schema models were found between the leaderboard and alpha metric files.")

    matched_records = [record_by_display_name[name] for name in matched_names]
    aggregate_alpha, detailed_alpha, pca_explained_variance = build_common_schema_feature_tables(
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
    ).sort_values("model_id").reset_index(drop=True)

    detailed_joined = joined[["model_id"]].merge(
        detailed_alpha,
        left_on="model_id",
        right_on="display_name",
        how="inner",
        suffixes=("", "_alpha"),
    )

    aggregate_alpha_features = usable_numeric_columns(
        joined,
        [
            column
            for column in aggregate_alpha.columns
            if column.startswith("alpha_") and column not in {"alpha_model_id", "alpha_num_layers"}
        ],
    )
    alpha_pca_features = [column for column in aggregate_alpha_features if column.startswith("alpha_pc")]
    alpha_summary_features = [column for column in aggregate_alpha_features if column not in alpha_pca_features]
    module_features = usable_numeric_columns(
        detailed_joined,
        [column for column in detailed_alpha.columns if "__" in column],
    )
    nonalpha_features = usable_numeric_columns(
        joined,
        [column for column in NON_ALPHA_METRIC_CANDIDATES if column in joined.columns] + ["alpha_num_layers"],
    )

    alpha_univariate = univariate_correlation_table(
        frame=joined,
        predictor_columns=alpha_summary_features + alpha_pca_features,
        target_columns=PERFORMANCE_TARGETS,
        min_pairs=args.min_pairs,
    )
    overall_univariate = univariate_correlation_table(
        frame=joined,
        predictor_columns=alpha_summary_features + alpha_pca_features + nonalpha_features,
        target_columns=PERFORMANCE_TARGETS,
        min_pairs=args.min_pairs,
    )

    alpha_best = (
        alpha_univariate.sort_values(
            by=["target_metric", "abs_partial_spearman_rho", "abs_spearman_rho"],
            ascending=[True, False, False],
        )
        .groupby("target_metric")
        .head(1)
        .reset_index(drop=True)
    )
    overall_best = (
        overall_univariate.sort_values(
            by=["target_metric", "abs_partial_spearman_rho", "abs_spearman_rho"],
            ascending=[True, False, False],
        )
        .groupby("target_metric")
        .head(1)
        .reset_index(drop=True)
    )

    family_definitions = {
        "alpha_summary": alpha_summary_features,
        "alpha_modules": module_features,
        "nonalpha_metrics": nonalpha_features,
        "alpha_summary_plus_nonalpha": alpha_summary_features + nonalpha_features,
        "alpha_modules_plus_nonalpha": module_features + nonalpha_features,
        "alpha_all_plus_nonalpha": alpha_summary_features + alpha_pca_features + module_features + nonalpha_features,
    }

    family_rows = []
    coefficient_rows = []
    modeling_frame = detailed_joined.merge(
        joined[["model_id", *PERFORMANCE_TARGETS, *nonalpha_features, *alpha_summary_features, *alpha_pca_features]],
        on="model_id",
        how="left",
    )
    for target in PERFORMANCE_TARGETS:
        for family_name, columns in family_definitions.items():
            metrics_row, coefficients = evaluate_feature_family(
                frame=modeling_frame,
                feature_columns=list(dict.fromkeys(columns)),
                target_column=target,
                cv_splits=args.cv_splits,
            )
            if not metrics_row:
                continue
            metrics_row["family"] = family_name
            family_rows.append(metrics_row)
            if not coefficients.empty:
                coefficients["family"] = family_name
                coefficient_rows.append(coefficients)

    family_scores = pd.DataFrame(family_rows).sort_values(
        by=["target_metric", "cv_r2", "cv_spearman"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    coefficient_table = (
        pd.concat(coefficient_rows, ignore_index=True)
        if coefficient_rows
        else pd.DataFrame(columns=["target_metric", "family", "feature", "coefficient", "abs_coefficient"])
    )

    top_coefficients = (
        coefficient_table.sort_values(["target_metric", "family", "abs_coefficient"], ascending=[True, True, False])
        .groupby(["target_metric", "family"])
        .head(10)
        .reset_index(drop=True)
    )

    joined.to_csv(args.output_dir / "matched_common_schema_models.csv", index=False)
    detailed_joined.to_csv(args.output_dir / "common_schema_module_features.csv", index=False)
    alpha_univariate.to_csv(args.output_dir / "alpha_univariate_performance_correlations.csv", index=False)
    overall_univariate.to_csv(args.output_dir / "overall_univariate_performance_correlations.csv", index=False)
    family_scores.to_csv(args.output_dir / "predictive_family_scores.csv", index=False)
    top_coefficients.to_csv(args.output_dir / "top_family_coefficients.csv", index=False)
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
        pca_explained_variance=pca_explained_variance,
        alpha_best=alpha_best,
        overall_best=overall_best,
        family_scores=family_scores,
        top_coefficients=top_coefficients,
    )

    print(f"Matched common-schema models: {joined['model_id'].nunique()}")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
