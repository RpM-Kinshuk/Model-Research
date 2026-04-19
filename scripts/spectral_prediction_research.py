from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_research.prediction import (
    PERFORMANCE_TARGETS,
    build_prediction_dataset,
    neighborhood_predict_oof,
    structural_baseline_oof,
    summarize_predictions,
)

SUPPORTED_METHODS = (
    "structural_basics_only",
    "pl_alpha_scalar",
    "full_module_trace",
    "role_aggregated_trace",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run performance prediction cycle 1.")
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=REPO_ROOT / "svd_results",
        help="Directory containing alpha metric files.",
    )
    parser.add_argument(
        "--leaderboard-csv",
        type=Path,
        default=REPO_ROOT / "data" / "sampled_leaderboard.csv",
        help="Path to sampled_leaderboard.csv.",
    )
    parser.add_argument(
        "--lineage-csv",
        type=Path,
        default=REPO_ROOT / "data" / "sampled_metadata.csv",
        help="Path to sampled_metadata.csv.",
    )
    parser.add_argument(
        "--freeze-csv",
        type=Path,
        default=REPO_ROOT / "data" / "manual_audits" / "spectral_family_retrieval_label_freeze.csv",
        help="Path to the audited family freeze CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "research_runs" / "performance_prediction_cycle1",
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default=",".join(PERFORMANCE_TARGETS),
        help="Comma-separated targets to evaluate.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Grouped CV splits.",
    )
    return parser.parse_args(argv)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"

    table = frame.loc[:, columns].copy()

    def cell(value: object) -> str:
        if pd.isna(value):
            return "n/a"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def render_report(results: pd.DataFrame) -> str:
    best = (
        results.sort_values(["target", "r2", "spearman"], ascending=[True, False, False])
        .groupby("target", as_index=False)
        .head(3)
        .reset_index(drop=True)
    )
    return "\n".join(
        [
            "# Performance Prediction Cycle 1",
            "",
            "## Best rows by target",
            _markdown_table(
                best,
                ["target", "method", "mode", "bucket", "k", "weighted", "n", "r2", "mae", "spearman"],
            ),
            "",
            "## Notes",
            "- Raw spectral rows measure direct neighborhood signal.",
            "- Controlled rows add the structural baseline back after residual neighborhood prediction.",
        ]
    ) + "\n"


def _baseline_rows(dataset: pd.DataFrame, *, target: str, cv_splits: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = structural_baseline_oof(dataset, target=target, cv_splits=cv_splits)
    baseline = baseline.rename(columns={"baseline_prediction": "prediction"}).assign(
        target=target,
        method="structural_basics_only",
        mode="baseline",
        k=0,
        weighted=False,
    )
    summary = summarize_predictions(
        baseline,
        target=target,
        method="structural_basics_only",
        k=0,
        weighted=False,
    )
    summary["mode"] = "baseline"
    return baseline, summary


def _spectral_rows(
    dataset: pd.DataFrame,
    records_by_model_id: dict[str, object],
    *,
    target: str,
    method: str,
    cv_splits: int,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    oof_frames: list[pd.DataFrame] = []
    result_frames: list[pd.DataFrame] = []

    for k in (3, 5):
        for weighted in (False, True):
            for residualize in (False, True):
                predictions = neighborhood_predict_oof(
                    frame=dataset,
                    records_by_model_id=records_by_model_id,
                    target=target,
                    method=method,
                    k=k,
                    weighted=weighted,
                    residualize=residualize,
                    cv_splits=cv_splits,
                ).assign(
                    target=target,
                    method=method,
                    k=k,
                    weighted=weighted,
                )
                oof_frames.append(predictions)
                result_frames.append(
                    summarize_predictions(
                        predictions,
                        target=target,
                        method=method,
                        k=k,
                        weighted=weighted,
                    )
                )

    return oof_frames, result_frames


def main() -> None:
    args = parse_args()
    targets = _split_csv(args.targets)
    dataset, records_by_model_id = build_prediction_dataset(
        metrics_dir=args.metrics_dir,
        leaderboard_csv=args.leaderboard_csv,
        lineage_csv=args.lineage_csv,
        freeze_csv=args.freeze_csv,
    )

    missing_targets = [target for target in targets if target not in dataset.columns]
    if missing_targets:
        raise SystemExit(f"Missing target columns in prediction dataset: {', '.join(missing_targets)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output_dir / "prediction_dataset.csv", index=False)

    result_frames: list[pd.DataFrame] = []
    oof_frames: list[pd.DataFrame] = []
    for target in targets:
        baseline_oof, baseline_summary = _baseline_rows(
            dataset,
            target=target,
            cv_splits=args.cv_splits,
        )
        oof_frames.append(baseline_oof)
        result_frames.append(baseline_summary)

        for method in SUPPORTED_METHODS[1:]:
            target_oof, target_results = _spectral_rows(
                dataset,
                records_by_model_id,
                target=target,
                method=method,
                cv_splits=args.cv_splits,
            )
            oof_frames.extend(target_oof)
            result_frames.extend(target_results)

    results = pd.concat(result_frames, ignore_index=True)
    predictions = pd.concat(oof_frames, ignore_index=True)
    results.to_csv(args.output_dir / "prediction_results.csv", index=False)
    predictions.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    (args.output_dir / "report.md").write_text(render_report(results), encoding="utf-8")

    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
