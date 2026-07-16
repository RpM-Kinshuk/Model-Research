from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alpha_cluster_dashboard.dashboard_cache import infer_prediction_targets
from alpha_cluster_dashboard.paths import resolve_model_zoo_artifact
from spectral_research.ablation_analysis import (
    artifact_model_ids,
    load_terminal_statuses,
    summarize_run_coverage,
)
from spectral_research.ablation_method_study import (
    DEFAULT_METHODS,
    build_similarity_matrix,
    build_structural_similarity_matrix,
    evaluate_family_similarity,
    evaluate_knn_prediction,
    load_records_by_display_name,
    method_decision,
    select_balanced_study_frame,
)

DEFAULT_DATA_DIR = resolve_model_zoo_artifact("data/ablation")
DEFAULT_RUN_OUTPUT_DIR = resolve_model_zoo_artifact("output/ablation")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "research_runs" / "preliminary" / "ablation" / "ablation_method_study"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare minimal spectral similarity methods on ablation artifacts.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run-output-dir", type=Path, default=DEFAULT_RUN_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--targets", default="auto")
    parser.add_argument("--min-family-size", type=int, default=25)
    parser.add_argument("--max-families", type=int, default=10)
    parser.add_argument("--max-models-per-family", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--knn-k", type=int, default=3)
    return parser.parse_args(argv)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_per_model(args: argparse.Namespace) -> pd.DataFrame:
    selected = pd.read_csv(args.data_dir / "ablation_selected.csv")
    statuses = load_terminal_statuses(args.run_output_dir)
    stats_model_ids = artifact_model_ids(args.run_output_dir / "stats", ".csv")
    metrics_model_ids = artifact_model_ids(args.run_output_dir / "metrics", ".h5")
    per_model, _ = summarize_run_coverage(
        selected,
        statuses,
        stats_model_ids=stats_model_ids,
        metrics_model_ids=metrics_model_ids,
    )
    return per_model


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = _split_csv(args.methods)

    records_by_model = load_records_by_display_name(args.run_output_dir / "metrics")
    per_model = _load_per_model(args)
    study_frame = select_balanced_study_frame(
        per_model,
        set(records_by_model),
        min_family_size=args.min_family_size,
        max_families=args.max_families,
        max_models_per_family=args.max_models_per_family,
        random_state=args.seed,
    )
    if study_frame.empty:
        raise SystemExit("No eligible study rows after filtering.")
    targets = infer_prediction_targets(study_frame) if args.targets == "auto" else _split_csv(args.targets)

    records = [records_by_model[model_id] for model_id in study_frame["model_id"].astype(str)]
    family_labels = study_frame["coarse_family"].astype(str).tolist()
    family_rows = []
    prediction_rows = []

    for method in methods:
        start = perf_counter()
        if method == "structural_basics":
            matrix = build_structural_similarity_matrix(study_frame, records)
        else:
            matrix = build_similarity_matrix(records, method=method)
        elapsed = perf_counter() - start
        family_result = evaluate_family_similarity(matrix, family_labels)
        family_rows.append(
            {
                "method": method,
                "n_models": len(records),
                "n_families": study_frame["coarse_family"].nunique(),
                "elapsed_seconds": elapsed,
                **family_result,
            }
        )
        for target in targets:
            if target not in study_frame.columns:
                continue
            prediction_rows.append(
                {
                    "method": method,
                    "k": args.knn_k,
                    **evaluate_knn_prediction(matrix, study_frame, target=target, k=args.knn_k),
                }
            )

    family_metrics = pd.DataFrame(family_rows).sort_values(
        ["pairwise_auc", "top1_family_recall"], ascending=[False, False]
    )
    prediction_metrics = pd.DataFrame(prediction_rows).sort_values(["target", "mae", "method"])

    study_frame.to_csv(args.output_dir / "method_study_dataset.csv", index=False)
    family_metrics.to_csv(args.output_dir / "method_family_metrics.csv", index=False)
    prediction_metrics.to_csv(args.output_dir / "method_prediction_metrics.csv", index=False)
    (args.output_dir / "method_decision_report.md").write_text(
        method_decision(family_metrics, prediction_metrics),
        encoding="utf-8",
    )
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
