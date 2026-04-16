from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_research.prediction import (
    PERFORMANCE_TARGETS,
    build_prediction_dataset,
    default_prediction_specs,
    evaluate_prediction_specs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run grouped, confound-controlled spectral prediction experiments."
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=REPO_ROOT / "svd_results",
        help="Directory containing alpha metric files.",
    )
    parser.add_argument(
        "--leaderboard-csv",
        type=Path,
        default=REPO_ROOT / "sampled_leaderboard.csv",
        help="Path to sampled_leaderboard.csv.",
    )
    parser.add_argument(
        "--lineage-csv",
        type=Path,
        default=REPO_ROOT / "sampled_metadata.csv",
        help="Path to sampled_metadata.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "research_runs" / "spectral_prediction_research",
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=32,
        help="Normalized depth bins for spectral representations.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Grouped CV splits.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=40,
        help="Minimum target sample size required for evaluation.",
    )
    parser.add_argument(
        "--profile-pca-components",
        type=int,
        default=8,
        help="Number of PCA components for the high-dimensional profile block.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default=",".join(PERFORMANCE_TARGETS),
        help="Comma-separated targets to evaluate.",
    )
    return parser.parse_args()


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _format_float(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    numeric = float(value)
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}"


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
    rows = ["| " + " | ".join(cell(value) for value in row) + " |" for row in table.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def _render_report(
    output_path: Path,
    dataset_frame: pd.DataFrame,
    blocks_frame: pd.DataFrame,
    results: pd.DataFrame,
) -> None:
    best_by_target = (
        results.sort_values(["target", "r2", "delta_r2_vs_confound"], ascending=[True, False, False])
        .groupby("target", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    spectral_gain = (
        results[results["spec"] != "confound_only"]
        .sort_values(["target", "delta_r2_vs_confound", "r2"], ascending=[True, False, False])
        .groupby("target", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    average_by_spec = (
        results.groupby("spec", as_index=False)
        .agg(
            mean_r2=("r2", "mean"),
            mean_mae=("mae", "mean"),
            mean_spearman=("spearman", "mean"),
            mean_delta_r2=("delta_r2_vs_confound", "mean"),
        )
        .sort_values(["mean_r2", "mean_delta_r2"], ascending=[False, False])
        .reset_index(drop=True)
    )

    lines = [
        "# Spectral Prediction Research",
        "",
        "## Dataset summary",
        _markdown_table(
            dataset_frame,
            [
                "n_models",
                "n_known_families",
                "n_cv_groups",
                "n_targets_available",
                "leaderboard_duplicate_rows",
            ],
        ),
        "",
        "## Feature blocks",
        _markdown_table(blocks_frame, ["block", "n_features"]),
        "",
        "## Best spec by target",
        _markdown_table(
            best_by_target,
            ["target", "spec", "r2", "mae", "rmse", "spearman", "kendall", "delta_r2_vs_confound"],
        ),
        "",
        "## Best spectral gain by target",
        _markdown_table(
            spectral_gain,
            ["target", "spec", "r2", "delta_r2_vs_confound", "spearman", "kendall"],
        ),
        "",
        "## Average performance by spec",
        _markdown_table(
            average_by_spec,
            ["spec", "mean_r2", "mean_mae", "mean_spearman", "mean_delta_r2"],
        ),
        "",
        "## Notes",
        "- Evaluation uses out-of-fold predictions under grouped CV keyed by `base_model` when known, otherwise by model identity.",
        "- `delta_r2_vs_confound` is computed target-wise relative to the `confound_only` ridge baseline on the same matched subset.",
        "- The profile block is reduced with PCA before regression to keep the model stable in the low-sample regime.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    targets = _split_csv(args.targets)
    dataset, blocks = build_prediction_dataset(
        metrics_dir=args.metrics_dir,
        leaderboard_csv=args.leaderboard_csv,
        lineage_csv=args.lineage_csv,
        num_bins=args.num_bins,
    )
    specs = default_prediction_specs()
    results, predictions = evaluate_prediction_specs(
        frame=dataset,
        blocks=blocks,
        specs=specs,
        targets=targets,
        cv_splits=args.cv_splits,
        min_samples=args.min_samples,
        profile_pca_components=args.profile_pca_components,
        random_state=0,
    )

    dataset_summary = pd.DataFrame(
        [
            {
                "n_models": int(len(dataset)),
                "n_known_families": int(dataset["base_model"].replace("unknown", np.nan).nunique())
                if "base_model" in dataset.columns
                else 0,
                "n_cv_groups": int(dataset["cv_group"].nunique()),
                "n_targets_available": int(sum(target in dataset.columns for target in targets)),
                "leaderboard_duplicate_rows": int(dataset["leaderboard_duplicate_rows"].iloc[0])
                if "leaderboard_duplicate_rows" in dataset.columns and not dataset.empty
                else 0,
            }
        ]
    )
    blocks_frame = pd.DataFrame(
        [
            {"block": "confound_numeric", "n_features": len(blocks.confound_numeric)},
            {"block": "confound_categorical", "n_features": len(blocks.confound_categorical)},
            {"block": "summary_spectral", "n_features": len(blocks.summary_spectral)},
            {"block": "profile_spectral", "n_features": len(blocks.profile_spectral)},
        ]
    )

    dataset_summary.to_csv(args.output_dir / "dataset_summary.csv", index=False)
    blocks_frame.to_csv(args.output_dir / "feature_blocks.csv", index=False)
    results.to_csv(args.output_dir / "prediction_results.csv", index=False)
    predictions.to_csv(args.output_dir / "oof_predictions.csv", index=False)

    _render_report(
        output_path=args.output_dir / "report.md",
        dataset_frame=dataset_summary,
        blocks_frame=blocks_frame,
        results=results,
    )
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
