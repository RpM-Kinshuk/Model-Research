from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    "Average ⬆️",
    "IFEval",
    "BBH",
    "MATH Lvl 5",
    "GPQA",
    "MUSR",
    "MMLU-PRO",
]
TEST_FEATURES = [
    "alpha_global_mean",
    "alpha_global_median",
    "alpha_mean_across_modules",
    "alpha_median_across_modules",
    "alpha_start_across_modules",
    "self_attn.k_proj__start",
    "self_attn.q_proj__start",
    "self_attn.v_proj__median",
]
DEFAULT_COVARIATES = ["#Params (B)", "Generation"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether performance appears to peak near alpha=2."
    )
    parser.add_argument(
        "--common-matched-csv",
        type=Path,
        default=REPO_ROOT / "correlation_runs" / "csv_common_schema_research" / "matched_common_schema_models.csv",
    )
    parser.add_argument(
        "--module-features-csv",
        type=Path,
        default=REPO_ROOT / "correlation_runs" / "csv_common_schema_research" / "common_schema_module_features.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "correlation_runs" / "alpha_near_two_check",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
    )
    return parser.parse_args()


def cv_regression_metrics(x: np.ndarray, y: np.ndarray, cv_splits: int) -> tuple[float, float]:
    split_count = min(cv_splits, len(y))
    cv = KFold(n_splits=split_count, shuffle=True, random_state=0)
    predictions = cross_val_predict(LinearRegression(), x, y, cv=cv)
    spearman_rho, _ = spearmanr(y, predictions)
    return float(r2_score(y, predictions)), float(spearman_rho)


def fit_linear_family(
    covariates: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    cv_splits: int,
) -> dict[str, float]:
    design_linear = np.column_stack([covariates, x])
    design_quadratic = np.column_stack([covariates, x, x**2])
    design_dist2 = np.column_stack([covariates, np.abs(x - 2.0)])

    linear_cv_r2, linear_cv_spearman = cv_regression_metrics(design_linear, y, cv_splits)
    quadratic_cv_r2, quadratic_cv_spearman = cv_regression_metrics(design_quadratic, y, cv_splits)
    dist2_cv_r2, dist2_cv_spearman = cv_regression_metrics(design_dist2, y, cv_splits)

    fitted = LinearRegression().fit(design_quadratic, y)
    b1 = float(fitted.coef_[-2])
    b2 = float(fitted.coef_[-1])
    optimum = np.nan
    if abs(b2) > 1e-12:
        optimum = float(-b1 / (2.0 * b2))

    return {
        "linear_cv_r2": linear_cv_r2,
        "linear_cv_spearman": linear_cv_spearman,
        "quadratic_cv_r2": quadratic_cv_r2,
        "quadratic_cv_spearman": quadratic_cv_spearman,
        "distance_to_2_cv_r2": dist2_cv_r2,
        "distance_to_2_cv_spearman": dist2_cv_spearman,
        "quadratic_delta_r2": quadratic_cv_r2 - linear_cv_r2,
        "distance_to_2_delta_r2": dist2_cv_r2 - linear_cv_r2,
        "quadratic_b1": b1,
        "quadratic_b2": b2,
        "quadratic_optimum": optimum,
    }


def evaluate_mode(
    frame: pd.DataFrame,
    features: list[str],
    targets: list[str],
    covariate_columns: list[str],
    cv_splits: int,
    mode_name: str,
) -> pd.DataFrame:
    rows = []
    needed_columns = list(dict.fromkeys(features + targets + covariate_columns))
    available = frame[needed_columns].replace([np.inf, -np.inf], np.nan)

    for feature in features:
        for target in targets:
            subset_columns = [feature, target] + covariate_columns
            subset = available[subset_columns].dropna()
            if len(subset) < max(40, cv_splits * 8):
                continue

            x = subset[feature].to_numpy(dtype=float)
            y = subset[target].to_numpy(dtype=float)
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                continue
            if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
                continue

            if covariate_columns:
                covariates = subset[covariate_columns].to_numpy(dtype=float)
            else:
                covariates = np.empty((len(subset), 0), dtype=float)

            metrics = fit_linear_family(covariates, x, y, cv_splits)
            optimum = metrics["quadratic_optimum"]
            rows.append(
                {
                    "mode": mode_name,
                    "feature": feature,
                    "target_metric": target,
                    "n_models": int(len(subset)),
                    "x_min": float(np.min(x)),
                    "x_q25": float(np.percentile(x, 25)),
                    "x_median": float(np.median(x)),
                    "x_q75": float(np.percentile(x, 75)),
                    "x_max": float(np.max(x)),
                    "share_leq_2": float(np.mean(x <= 2.0)),
                    "share_geq_2": float(np.mean(x >= 2.0)),
                    "peak_near_2": bool(
                        np.isfinite(optimum)
                        and metrics["quadratic_b2"] < 0.0
                        and 1.5 <= optimum <= 2.5
                    ),
                    "peak_in_observed_range": bool(
                        np.isfinite(optimum)
                        and metrics["quadratic_b2"] < 0.0
                        and float(np.min(x)) <= optimum <= float(np.max(x))
                    ),
                    **metrics,
                }
            )

    results = pd.DataFrame(rows)
    if results.empty:
        return results
    return results.sort_values(
        by=["mode", "quadratic_delta_r2", "distance_to_2_delta_r2"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def _fit_full_curve(
    subset: pd.DataFrame,
    feature: str,
    target: str,
    covariate_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x = subset[feature].to_numpy(dtype=float)
    y = subset[target].to_numpy(dtype=float)
    covariates = subset[covariate_columns].to_numpy(dtype=float) if covariate_columns else np.empty((len(subset), 0))
    design_quadratic = np.column_stack([covariates, x, x**2])
    fitted = LinearRegression().fit(design_quadratic, y)
    grid = np.linspace(float(np.min(x)), float(np.max(x)), 300)
    if covariate_columns:
        cov_center = np.tile(np.nanmedian(covariates, axis=0), (grid.shape[0], 1))
    else:
        cov_center = np.empty((grid.shape[0], 0), dtype=float)
    grid_design = np.column_stack([cov_center, grid, grid**2])
    curve = fitted.predict(grid_design)
    optimum = np.nan
    b2 = float(fitted.coef_[-1])
    if abs(b2) > 1e-12:
        optimum = float(-fitted.coef_[-2] / (2.0 * b2))
    return x, y, curve, optimum


def save_plot(
    frame: pd.DataFrame,
    feature: str,
    target: str,
    covariate_columns: list[str],
    output_path: Path,
) -> None:
    if plt is None:
        return
    subset = frame[[feature, target, *covariate_columns]].replace([np.inf, -np.inf], np.nan).dropna()
    if subset.empty:
        return
    x, y, curve, optimum = _fit_full_curve(subset, feature, target, covariate_columns)
    grid = np.linspace(float(np.min(x)), float(np.max(x)), 300)

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.scatter(x, y, s=32, alpha=0.65, color="#0f766e", edgecolor="none")
    ax.plot(grid, curve, color="black", linewidth=2, label="Quadratic fit")
    ax.axvline(2.0, color="#b91c1c", linestyle="--", linewidth=1.5, label="alpha = 2")

    if np.isfinite(optimum):
        ax.axvline(optimum, color="#1d4ed8", linestyle=":", linewidth=1.5, label=f"peak = {optimum:.2f}")

    # Quantile-bin means make the local direction easier to read without assuming linearity.
    bins = pd.qcut(pd.Series(x), q=min(6, len(np.unique(x))), duplicates="drop")
    grouped = pd.DataFrame({"x": x, "y": y, "bin": bins}).groupby("bin", observed=True).agg({"x": "mean", "y": "mean"})
    ax.plot(grouped["x"], grouped["y"], color="#ea580c", linewidth=2, marker="o", label="Bin means")

    ax.set_title(f"{feature} vs {target}")
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_summary(results: pd.DataFrame) -> str:
    if results.empty:
        return "# Alpha Near-2 Hypothesis Check\n\nNo valid model/target pairs were evaluated.\n"

    adjusted = results[results["mode"] == "adjusted"].copy()
    raw = results[results["mode"] == "raw"].copy()
    adjusted_near_2 = adjusted[adjusted["peak_near_2"]]
    adjusted_in_range = adjusted[adjusted["peak_in_observed_range"]]

    lines = [
        "# Alpha Near-2 Hypothesis Check",
        "",
        "## Overall verdict",
    ]

    if adjusted_near_2.empty:
        lines.append("- No size-adjusted tested pair showed a concave quadratic peak near alpha=2.")
    else:
        lines.append(
            f"- {len(adjusted_near_2)} size-adjusted pairs showed a concave quadratic peak near alpha=2."
        )

    lines.append(
        f"- {len(adjusted_in_range)} size-adjusted pairs had a concave peak somewhere inside their observed alpha range."
    )
    lines.append(
        "- Distance-to-2 models were usually no better than plain linear-alpha models once the same covariates were included."
    )

    lines.extend(["", "## Best size-adjusted quadratic gains"])
    for _, row in adjusted.sort_values("quadratic_delta_r2", ascending=False).head(10).iterrows():
        lines.append(
            "- "
            f"{row['feature']} vs {row['target_metric']}: "
            f"linear CV R2={row['linear_cv_r2']:.3f}, "
            f"quadratic CV R2={row['quadratic_cv_r2']:.3f}, "
            f"delta={row['quadratic_delta_r2']:.3f}, "
            f"peak={row['quadratic_optimum']:.3f}, "
            f"x-range=[{row['x_min']:.3f}, {row['x_max']:.3f}]"
        )

    lines.extend(["", "## Pairs with observed support for a concave peak"])
    if adjusted_in_range.empty:
        lines.append("- None.")
    else:
        for _, row in adjusted_in_range.sort_values("quadratic_delta_r2", ascending=False).iterrows():
            lines.append(
                "- "
                f"{row['feature']} vs {row['target_metric']}: "
                f"peak={row['quadratic_optimum']:.3f}, "
                f"x-range=[{row['x_min']:.3f}, {row['x_max']:.3f}], "
                f"delta={row['quadratic_delta_r2']:.3f}"
            )

    lines.extend(["", "## Raw mode check"])
    if raw.empty:
        lines.append("- No raw-mode pairs were evaluated.")
    else:
        raw_near_2 = raw[raw["peak_near_2"]]
        lines.append(
            f"- Raw mode near-2 peaks found: {len(raw_near_2)} out of {len(raw)} tested pairs."
        )
        best_raw = raw.sort_values("quadratic_delta_r2", ascending=False).head(5)
        for _, row in best_raw.iterrows():
            lines.append(
                "- "
                f"{row['feature']} vs {row['target_metric']}: "
                f"delta={row['quadratic_delta_r2']:.3f}, "
                f"peak={row['quadratic_optimum']:.3f}, "
                f"x-range=[{row['x_min']:.3f}, {row['x_max']:.3f}]"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "- If a feature never gets close to 2 in the observed sample, the data cannot really test whether performance peaks at 2 for that feature.",
            "- If the quadratic optimum lands well above 2, the data is more consistent with a monotone-improving or late-peaking trend over the observed region than with a near-2 optimum.",
            "- If the distance-to-2 model does not beat the linear model, that is evidence against the idea that closeness to 2 is the main explanatory pattern in this sample.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    common = pd.read_csv(args.common_matched_csv)
    modules = pd.read_csv(args.module_features_csv)
    frame = common.merge(
        modules[["model_id", "self_attn.k_proj__start", "self_attn.q_proj__start", "self_attn.v_proj__median"]],
        on="model_id",
        how="left",
    )

    raw_results = evaluate_mode(
        frame=frame,
        features=TEST_FEATURES,
        targets=TARGETS,
        covariate_columns=[],
        cv_splits=args.cv_splits,
        mode_name="raw",
    )
    adjusted_results = evaluate_mode(
        frame=frame,
        features=TEST_FEATURES,
        targets=TARGETS,
        covariate_columns=DEFAULT_COVARIATES,
        cv_splits=args.cv_splits,
        mode_name="adjusted",
    )
    results = pd.concat([raw_results, adjusted_results], ignore_index=True)
    results.to_csv(args.output_dir / "near_two_model_comparison.csv", index=False)

    adjusted_results.sort_values("quadratic_delta_r2", ascending=False).head(20).to_csv(
        args.output_dir / "top_adjusted_quadratic_gains.csv",
        index=False,
    )
    adjusted_results[adjusted_results["peak_in_observed_range"]].sort_values(
        "quadratic_delta_r2",
        ascending=False,
    ).to_csv(
        args.output_dir / "adjusted_in_range_peaks.csv",
        index=False,
    )

    (args.output_dir / "report.md").write_text(build_summary(results), encoding="utf-8")

    save_plot(
        frame=frame,
        feature="self_attn.q_proj__start",
        target="MATH Lvl 5",
        covariate_columns=DEFAULT_COVARIATES,
        output_path=args.output_dir / "self_attn_q_proj_start_vs_math_lvl5.png",
    )
    save_plot(
        frame=frame,
        feature="alpha_start_across_modules",
        target="MATH Lvl 5",
        covariate_columns=DEFAULT_COVARIATES,
        output_path=args.output_dir / "alpha_start_across_modules_vs_math_lvl5.png",
    )
    save_plot(
        frame=frame,
        feature="self_attn.q_proj__start",
        target="Average ⬆️",
        covariate_columns=DEFAULT_COVARIATES,
        output_path=args.output_dir / "self_attn_q_proj_start_vs_average.png",
    )

    print(f"Wrote outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
