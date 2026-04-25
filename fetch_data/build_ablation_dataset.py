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
from fetch_data.phase2_schema import canonical_model_key, parse_repo_and_revision


DEFAULT_OUTPUT_DIR = Path("data/ablation")
DEFAULT_PHASE2_PATH = Path("data/curated/model_zoo_phase2.csv")
CONTENTS_EVAL_SOURCE = "open_llm_leaderboard_contents"
RESULTS_EVAL_SOURCE = "open_llm_leaderboard_results"
METADATA_MERGE_COLUMNS = [
    "source_model",
    "base_model_relation",
    "pipeline_tag",
    "Architecture",
    "Type",
    "#Params (B)",
    "quant_method",
    "files",
    "tags",
    "availability_status",
    "Available on the hub",
]


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


def load_open_llm_leaderboard_contents(
    *,
    dataset_id: str = "open-llm-leaderboard/contents",
    load_dataset_fn=None,
) -> pd.DataFrame:
    if load_dataset_fn is None:
        from datasets import load_dataset

        load_dataset_fn = load_dataset
    dataset = load_dataset_fn(dataset_id, split="train")
    frame = dataset.to_pandas()
    frame["eval_source"] = CONTENTS_EVAL_SOURCE
    return frame


def _join_key(value: object) -> str | None:
    repo_id, _ = parse_repo_and_revision(value)
    key = canonical_model_key(repo_id, None)
    return key.lower() if key else None


def _with_join_key(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "repo_id_norm" in result.columns:
        repo_values = result["repo_id_norm"]
    else:
        repo_values = pd.Series([None] * len(result), index=result.index, dtype="object")
    if "model_id" in result.columns:
        model_values = result["model_id"]
    else:
        model_values = pd.Series([None] * len(result), index=result.index, dtype="object")
    result["_metadata_join_key"] = [
        _join_key(repo_id) or _join_key(model_id)
        for repo_id, model_id in zip(repo_values, model_values, strict=False)
    ]
    return result


def _blank_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.map(lambda value: isinstance(value, str) and not value.strip())


def _merge_metadata(source: pd.DataFrame, metadata_path: Path | None) -> pd.DataFrame:
    if metadata_path is None:
        return source

    metadata = _with_join_key(pd.read_csv(metadata_path))
    metadata = metadata[metadata["_metadata_join_key"].notna()].drop_duplicates("_metadata_join_key", keep="first")
    metadata_columns = [column for column in METADATA_MERGE_COLUMNS if column in metadata.columns]
    if not metadata_columns:
        return source

    source_with_key = _with_join_key(source)
    merged = source_with_key.merge(
        metadata[["_metadata_join_key", *metadata_columns]],
        on="_metadata_join_key",
        how="left",
        suffixes=("", "_metadata"),
    )
    for column in metadata_columns:
        metadata_column = f"{column}_metadata" if column in source.columns else column
        if metadata_column not in merged.columns:
            continue
        if column not in source.columns:
            continue
        merged.loc[_blank_mask(merged[column]), column] = merged.loc[_blank_mask(merged[column]), metadata_column]
        if metadata_column != column:
            merged = merged.drop(columns=[metadata_column])

    return merged.drop(columns=["_metadata_join_key"], errors="ignore")


def _load_sources(
    contents_path: Path | None,
    results_path: Path | None = None,
    *,
    fetch_contents: bool = False,
) -> pd.DataFrame:
    sources = []
    if contents_path is not None:
        sources.append(_source_frame(contents_path, eval_source=CONTENTS_EVAL_SOURCE))
    elif fetch_contents:
        sources.append(load_open_llm_leaderboard_contents())
    else:
        raise ValueError("Provide --contents-path or --fetch-contents.")
    if results_path is not None:
        sources.append(_source_frame(results_path, eval_source=RESULTS_EVAL_SOURCE))
    return pd.concat(sources, ignore_index=True, sort=False)


def build_ablation_outputs(
    *,
    contents_path: Path | None = None,
    phase2_path: Path = DEFAULT_PHASE2_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    results_path: Path | None = None,
    metadata_path: Path | None = None,
    fetch_contents: bool = False,
    target_size: int = 5000,
    max_eval_missing_fraction: float = 0.15,
) -> dict[str, object]:
    if contents_path is None and not fetch_contents:
        raise ValueError("Provide --contents-path or --fetch-contents.")
    if contents_path is not None:
        _require_existing_file(contents_path, "contents")
    if results_path is not None:
        _require_existing_file(results_path, "results")
    if metadata_path is not None:
        _require_existing_file(metadata_path, "metadata")
    _require_existing_file(phase2_path, "phase2")

    output_dir.mkdir(parents=True, exist_ok=True)

    sources = _merge_metadata(_load_sources(contents_path, results_path, fetch_contents=fetch_contents), metadata_path)
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
    parser.add_argument("--contents-path", type=Path)
    parser.add_argument("--results-path", type=Path)
    parser.add_argument("--fetch-contents", action="store_true")
    parser.add_argument("--metadata-path", type=Path)
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
        metadata_path=args.metadata_path,
        fetch_contents=args.fetch_contents,
        phase2_path=args.phase2_path,
        output_dir=args.output_dir,
        target_size=args.target_size,
        max_eval_missing_fraction=args.max_eval_missing_fraction,
    )
    print(json.dumps(outputs["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
