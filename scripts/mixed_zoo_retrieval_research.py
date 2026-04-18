from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_research.pipeline import main as pipeline_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run mixed-zoo retrieval cycle 1.")
    parser.add_argument("--metrics-dir", type=Path, default=REPO_ROOT / "svd_results")
    parser.add_argument(
        "--lineage-csv",
        type=Path,
        default=REPO_ROOT / "data" / "sampled_metadata.csv",
    )
    parser.add_argument(
        "--leaderboard-csv",
        type=Path,
        default=REPO_ROOT / "data" / "sampled_leaderboard.csv",
    )
    parser.add_argument(
        "--freeze-csv",
        type=Path,
        default=REPO_ROOT / "data" / "manual_audits" / "spectral_family_retrieval_label_freeze.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "research_runs" / "mixed_zoo_retrieval_cycle1",
    )
    parser.add_argument("--min-family-size", type=int, default=5)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return pipeline_main(
        [
            "--metrics-dir",
            str(args.metrics_dir),
            "--lineage-csv",
            str(args.lineage_csv),
            "--leaderboard-csv",
            str(args.leaderboard_csv),
            "--freeze-csv",
            str(args.freeze_csv),
            "--output-dir",
            str(args.output_dir),
            "--min-family-size",
            str(args.min_family_size),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
