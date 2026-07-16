from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alpha_cluster_dashboard.paths import resolve_model_zoo_artifact
from spectral_research.ablation_analysis import (
    artifact_model_ids,
    failure_breakdown,
    group_coverage,
    load_stats_summaries,
    load_terminal_statuses,
    render_report,
    spectral_correlations,
    summarize_run_coverage,
)

DEFAULT_DATA_DIR = resolve_model_zoo_artifact("data/ablation")
DEFAULT_RUN_OUTPUT_DIR = resolve_model_zoo_artifact("output/ablation")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "research_runs" / "preliminary" / "ablation" / "ablation_spectral_analysis"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the collected ablation spectral run.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run-output-dir", type=Path, default=DEFAULT_RUN_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-correlation-pairs", type=int, default=50)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _status_summary(per_model: pd.DataFrame) -> pd.DataFrame:
    return (
        per_model.groupby(["terminal_status", "terminal_stage", "terminal_reason"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values("rows", ascending=False)
        .reset_index(drop=True)
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_path = args.data_dir / "ablation_selected.csv"
    selected = pd.read_csv(selected_path)
    statuses = load_terminal_statuses(args.run_output_dir)
    stats_model_ids = artifact_model_ids(args.run_output_dir / "stats", ".csv")
    metrics_model_ids = artifact_model_ids(args.run_output_dir / "metrics", ".h5")

    per_model, run_summary = summarize_run_coverage(
        selected,
        statuses,
        stats_model_ids=stats_model_ids,
        metrics_model_ids=metrics_model_ids,
    )
    model_summary, role_summary = load_stats_summaries(args.run_output_dir / "stats")
    joined = per_model.merge(model_summary, on="model_id", how="left")
    correlations = spectral_correlations(joined, min_pairs=args.min_correlation_pairs)

    coverage_by_type = group_coverage(per_model, "primary_type_bucket")
    coverage_by_lineage = group_coverage(per_model, "lineage_eval_group")
    coverage_by_family = group_coverage(per_model, "coarse_family")
    status_summary = _status_summary(per_model)
    failure_by_type = failure_breakdown(per_model, "primary_type_bucket")
    failure_by_family = failure_breakdown(per_model, "coarse_family")
    curation_summary = _read_json(args.data_dir / "ablation_summary.json")

    per_model.to_csv(args.output_dir / "ablation_run_per_model.csv", index=False)
    joined.to_csv(args.output_dir / "ablation_model_spectral_joined.csv", index=False)
    model_summary.to_csv(args.output_dir / "model_spectral_summary.csv", index=False)
    role_summary.to_csv(args.output_dir / "role_spectral_summary.csv", index=False)
    correlations.to_csv(args.output_dir / "spectral_score_correlations.csv", index=False)
    coverage_by_type.to_csv(args.output_dir / "coverage_by_primary_type.csv", index=False)
    coverage_by_lineage.to_csv(args.output_dir / "coverage_by_lineage_group.csv", index=False)
    coverage_by_family.to_csv(args.output_dir / "coverage_by_coarse_family.csv", index=False)
    status_summary.to_csv(args.output_dir / "terminal_status_summary.csv", index=False)
    failure_by_type.to_csv(args.output_dir / "failure_reason_by_primary_type.csv", index=False)
    failure_by_family.to_csv(args.output_dir / "failure_reason_by_coarse_family.csv", index=False)
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        render_report(
            source_data_dir=args.data_dir,
            run_output_dir=args.run_output_dir,
            curation_summary=curation_summary,
            run_summary=run_summary,
            status_summary=status_summary,
            coverage_by_type=coverage_by_type,
            coverage_by_lineage=coverage_by_lineage,
            coverage_by_family=coverage_by_family,
            correlations=correlations,
        ),
        encoding="utf-8",
    )
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
