from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_research.prediction import PERFORMANCE_TARGETS, build_prediction_dataset
from spectral_research.trend_diagnostics import (
    ANALYSIS_COLUMNS,
    TREND_FEATURE_COLUMNS,
    analyze_trend_rows,
    build_absolute_trend_frame,
    build_exact_parent_delta_frame,
    build_family_relative_delta_frame,
    render_trend_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run performance trend diagnostics.")
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
        default=REPO_ROOT / "research_runs" / "performance_trend_diagnostics",
        help="Directory where outputs will be written.",
    )
    return parser.parse_args(argv)


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _analysis_inputs() -> list[tuple[str, list[str], list[str]]]:
    absolute_features = list(TREND_FEATURE_COLUMNS)
    absolute_targets = list(PERFORMANCE_TARGETS)
    delta_features = [f"{column}_delta" for column in TREND_FEATURE_COLUMNS]
    delta_targets = [f"{target}_delta" for target in PERFORMANCE_TARGETS]
    return [
        ("absolute", absolute_features, absolute_targets),
        ("exact_parent", delta_features, delta_targets),
        ("family_relative", delta_features, delta_targets),
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    dataset, records_by_model_id = build_prediction_dataset(
        metrics_dir=args.metrics_dir,
        leaderboard_csv=args.leaderboard_csv,
        lineage_csv=args.lineage_csv,
        freeze_csv=args.freeze_csv,
    )

    absolute = build_absolute_trend_frame(dataset, records_by_model_id)
    exact_parent = build_exact_parent_delta_frame(absolute)
    family_relative = build_family_relative_delta_frame(absolute)

    table_map = {
        "absolute": absolute,
        "exact_parent": exact_parent,
        "family_relative": family_relative,
    }
    results: list[pd.DataFrame] = []
    for analysis_name, feature_columns, target_columns in _analysis_inputs():
        table = table_map[analysis_name]
        for bucket_name in ("all_scored", "heterogeneous"):
            for use_structural_controls in (False, True):
                results.append(
                    analyze_trend_rows(
                        table,
                        feature_columns=feature_columns,
                        target_columns=target_columns,
                        analysis_name=analysis_name,
                        bucket_name=bucket_name,
                        use_structural_controls=use_structural_controls,
                    )
                )

    non_empty_results = [frame for frame in results if not frame.empty]
    result_frame = pd.concat(non_empty_results, ignore_index=True) if non_empty_results else pd.DataFrame()
    if result_frame.empty:
        result_frame = pd.DataFrame(columns=list(ANALYSIS_COLUMNS))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write(absolute, args.output_dir / "absolute_trend_dataset.csv")
    _write(exact_parent, args.output_dir / "exact_parent_delta_dataset.csv")
    _write(family_relative, args.output_dir / "family_relative_delta_dataset.csv")
    _write(result_frame, args.output_dir / "trend_results.csv")
    (args.output_dir / "report.md").write_text(render_trend_report(result_frame), encoding="utf-8")

    print(f"Outputs written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
