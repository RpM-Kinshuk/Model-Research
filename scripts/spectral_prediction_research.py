from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import spectral_research.prediction as prediction_module
from spectral_research.prediction import (
    PERFORMANCE_TARGETS,
    build_prediction_dataset,
    build_pairwise_score_matrix,
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
RESULT_COLUMNS = ["target", "method", "mode", "bucket", "k", "weighted", "n", "r2", "mae", "spearman"]


def _cv_splits_arg(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("--cv-splits must be at least 2")
    return parsed


def _optimization_mode_arg(value: str) -> str:
    if value not in {"repeated_cpu", "cpu_cache_only", "cache_plus_gpu0"}:
        raise argparse.ArgumentTypeError(
            "--optimization-mode must be one of repeated_cpu, cpu_cache_only, cache_plus_gpu0"
        )
    return value


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
        type=_cv_splits_arg,
        default=5,
        help="Grouped CV splits.",
    )
    parser.add_argument(
        "--optimization-mode",
        type=_optimization_mode_arg,
        default="repeated_cpu",
        help="Prediction cache strategy.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for torch pairwise cache precompute.",
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
    if results.empty:
        return "\n".join(
            [
                "# Performance Prediction Cycle 1",
                "",
                "## Best rows by target",
                "_No rows._",
                "",
                "## Notes",
                "- No prediction rows were generated for the requested dataset/targets.",
            ]
        ) + "\n"

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


def _pairwise_cache_backend(method: str, *, optimization_mode: str, device: str) -> tuple[str, str]:
    if (
        optimization_mode == "cache_plus_gpu0"
        and method == "pl_alpha_scalar"
        and prediction_module.torch is not None
    ):
        return "torch", device
    return "cpu", "cpu"


def _build_pairwise_cache(
    dataset: pd.DataFrame,
    records_by_model_id: dict[str, object],
    *,
    method: str,
    optimization_mode: str,
    device: str,
) -> tuple[object, object]:
    backend, backend_device = _pairwise_cache_backend(
        method,
        optimization_mode=optimization_mode,
        device=device,
    )
    return build_pairwise_score_matrix(
        dataset,
        records_by_model_id,
        method=method,
        backend=backend,
        device=backend_device,
    )


def _spectral_rows(
    dataset: pd.DataFrame,
    records_by_model_id: dict[str, object],
    *,
    target: str,
    method: str,
    cv_splits: int,
    pairwise_scores: object | None = None,
    model_index: dict[str, int] | None = None,
    baseline_by_model_id: dict[str, float] | None = None,
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
                    pairwise_scores=pairwise_scores,
                    model_index=model_index,
                    baseline_by_model_id=baseline_by_model_id,
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


def _empty_results_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=RESULT_COLUMNS)


def _empty_oof_frame(targets: list[str]) -> pd.DataFrame:
    columns = ["model_id", "query_bucket", *targets, "prediction", "mode", "target", "method", "k", "weighted"]
    return pd.DataFrame(columns=list(dict.fromkeys(columns)))


def main() -> None:
    args = parse_args()
    optimization_mode = getattr(args, "optimization_mode", "repeated_cpu")
    device = getattr(args, "device", "cuda:0")
    targets = _split_csv(args.targets)
    if not targets:
        raise SystemExit("No prediction targets specified.")

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

    pairwise_cache: dict[str, tuple[object, object]] = {}
    if optimization_mode != "repeated_cpu":
        for method in SUPPORTED_METHODS[1:]:
            pairwise_cache[method] = _build_pairwise_cache(
                dataset,
                records_by_model_id,
                method=method,
                optimization_mode=optimization_mode,
                device=device,
            )

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

        baseline_by_model_id = {
            str(model_id): float(prediction)
            for model_id, prediction in zip(
                baseline_oof["model_id"],
                baseline_oof["prediction"],
                strict=False,
            )
        }

        for method in SUPPORTED_METHODS[1:]:
            cached_pairwise = pairwise_cache.get(method)
            target_oof, target_results = _spectral_rows(
                dataset,
                records_by_model_id,
                target=target,
                method=method,
                cv_splits=args.cv_splits,
                pairwise_scores=cached_pairwise[0] if cached_pairwise is not None else None,
                model_index=cached_pairwise[1] if cached_pairwise is not None else None,
                baseline_by_model_id=baseline_by_model_id,
            )
            oof_frames.extend(target_oof)
            result_frames.extend(target_results)

    results = (
        pd.concat(result_frames, ignore_index=True)
        if result_frames
        else _empty_results_frame()
    )
    if results.empty:
        results = _empty_results_frame()

    predictions = (
        pd.concat(oof_frames, ignore_index=True)
        if oof_frames
        else _empty_oof_frame(targets)
    )

    results.to_csv(args.output_dir / "prediction_results.csv", index=False)
    predictions.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    (args.output_dir / "report.md").write_text(render_report(results), encoding="utf-8")

    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
