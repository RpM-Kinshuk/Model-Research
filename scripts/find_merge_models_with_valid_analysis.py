from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alpha_cluster_dashboard.data import _load_csv_alpha_record, _load_h5_alpha_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find models in sampled_metadata.csv with base_model_relation=merge and a "
            "non-empty, non-corrupt analysis file under svd_results."
        )
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=REPO_ROOT / "data" / "sampled_metadata.csv",
        help="Path to sampled_metadata.csv",
    )
    parser.add_argument(
        "--svd-results-dir",
        type=Path,
        default=REPO_ROOT / "svd_results",
        help="Directory to search recursively for analysis files",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "merge_models_with_valid_analysis.csv",
        help="Path to the output CSV",
    )
    return parser.parse_args()


def validate_analysis_file(path: Path) -> bool:
    try:
        if path.suffix == ".csv":
            return _load_csv_alpha_record(path) is not None
        if path.suffix in {".h5", ".hdf5"}:
            return _load_h5_alpha_record(path) is not None
    except Exception:
        return False
    return False


def gather_valid_analysis_files(svd_results_dir: Path) -> dict[str, list[Path]]:
    valid_files: dict[str, list[Path]] = {}
    for path in sorted(svd_results_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in {".csv", ".h5", ".hdf5"}:
            continue
        if not validate_analysis_file(path):
            continue
        valid_files.setdefault(path.stem, []).append(path)
    return valid_files


def choose_preferred_path(paths: list[Path]) -> Path:
    return sorted(
        paths,
        key=lambda path: (
            0 if path.suffix in {".h5", ".hdf5"} else 1,
            len(path.parts),
            str(path),
        ),
    )[0]


def main() -> int:
    args = parse_args()

    metadata = pd.read_csv(args.metadata_csv)
    merge_rows = metadata.loc[metadata["base_model_relation"].eq("merge")].copy()
    valid_files = gather_valid_analysis_files(args.svd_results_dir)

    merge_rows["analysis_stem"] = merge_rows["model_id"].astype(str).str.replace("/", "--", regex=False)
    merge_rows["valid_analysis_paths"] = merge_rows["analysis_stem"].map(valid_files)
    matched = merge_rows.loc[merge_rows["valid_analysis_paths"].notna()].copy()

    matched["analysis_path"] = matched["valid_analysis_paths"].map(choose_preferred_path).map(str)
    matched["analysis_format"] = matched["analysis_path"].map(lambda value: Path(value).suffix.lstrip("."))
    matched["analysis_file_count"] = matched["valid_analysis_paths"].map(len)
    matched["all_valid_analysis_paths"] = matched["valid_analysis_paths"].map(
        lambda paths: ";".join(str(path) for path in sorted(paths))
    )

    output_columns = list(metadata.columns) + [
        "analysis_path",
        "analysis_format",
        "analysis_file_count",
        "all_valid_analysis_paths",
    ]
    matched = matched[output_columns].sort_values("model_id").reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(args.output_csv, index=False)

    print(f"Found {len(matched)} merge models with at least one valid analysis file.")
    print(f"Output written to: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
