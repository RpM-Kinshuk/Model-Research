from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fetch_data.ablation_schema import (
    prepare_ablation_candidates,
    select_ablation_rows,
    summarize_ablation_outputs,
)


DEFAULT_OUTPUT_DIR = Path("data/ablation")
DEFAULT_PHASE2_PATH = Path("data/curated/model_zoo_phase2.csv")
CONTENTS_EVAL_SOURCE = "open_llm_leaderboard_contents"
RESULTS_EVAL_SOURCE = "open_llm_leaderboard_results"


def _require_existing_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {path}")


def _load_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _source_frame(path: Path, *, eval_source: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["eval_source"] = eval_source
    return frame


def _load_sources(contents_path: Path, results_path: Path | None = None) -> pd.DataFrame:
    sources = [_source_frame(contents_path, eval_source=CONTENTS_EVAL_SOURCE)]
    if results_path is not None:
        sources.append(_source_frame(results_path, eval_source=RESULTS_EVAL_SOURCE))
    return pd.concat(sources, ignore_index=True, sort=False)


def build_ablation_outputs(
    *,
    contents_path: Path,
    phase2_path: Path = DEFAULT_PHASE2_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    results_path: Path | None = None,
    target_size: int = 5000,
    max_eval_missing_fraction: float = 0.15,
) -> dict[str, object]:
    _require_existing_file(contents_path, "contents")
    if results_path is not None:
        _require_existing_file(results_path, "results")
    _require_existing_file(phase2_path, "phase2")

    output_dir.mkdir(parents=True, exist_ok=True)

    sources = _load_sources(contents_path, results_path)
    phase2 = _load_csv(phase2_path)
    candidate_frames = [
        prepare_ablation_candidates(group.drop(columns=["eval_source"], errors="ignore"), phase2_frame=phase2, eval_source=eval_source)
        for eval_source, group in sources.groupby("eval_source", sort=False)
    ]
    candidates = pd.concat(candidate_frames, ignore_index=True, sort=False) if candidate_frames else pd.DataFrame()
    selected, rejected = select_ablation_rows(
        candidates,
        target_size=target_size,
        max_eval_missing_fraction=max_eval_missing_fraction,
    )
    overlap_audit = candidates[candidates["phase2_overlap"].fillna(False)].copy() if "phase2_overlap" in candidates else candidates.iloc[0:0].copy()
    summary = summarize_ablation_outputs(candidates, selected, rejected)

    candidates.to_csv(output_dir / "ablation_candidates.csv", index=False)
    selected.to_csv(output_dir / "ablation_selected.csv", index=False)
    rejected.to_csv(output_dir / "ablation_rejected.csv", index=False)
    overlap_audit.to_csv(output_dir / "ablation_overlap_audit.csv", index=False)
    (output_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    return {
        "candidates": candidates,
        "selected": selected,
        "rejected": rejected,
        "overlap_audit": overlap_audit,
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the ablation curation dataset from local leaderboard CSVs.")
    parser.add_argument("--contents-path", type=Path, required=True)
    parser.add_argument("--results-path", type=Path)
    parser.add_argument("--phase2-path", type=Path, default=DEFAULT_PHASE2_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-size", type=int, default=5000)
    parser.add_argument("--max-eval-missing-fraction", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_ablation_outputs(
        contents_path=args.contents_path,
        results_path=args.results_path,
        phase2_path=args.phase2_path,
        output_dir=args.output_dir,
        target_size=args.target_size,
        max_eval_missing_fraction=args.max_eval_missing_fraction,
    )
    print(json.dumps(outputs["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
