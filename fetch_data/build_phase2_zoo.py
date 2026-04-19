from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fetch_data.phase2_schema import (
    add_lineage_flags,
    apply_type_taxonomy,
    canonical_model_key,
    collapse_leaderboard_duplicates,
    load_atlas_candidates,
    normalize_repo_id,
    parse_repo_and_revision,
    seed_analysis_status,
    select_to_target,
)


def _write_parquet_if_possible(frame: pd.DataFrame, path: Path) -> None:
    try:
        frame.to_parquet(path, index=False)
    except Exception:
        # Parquet is useful but optional in environments without pyarrow/fastparquet.
        pass


def build_phase2_outputs(
    *,
    metadata_path: Path,
    leaderboard_path: Path,
    atlas_json_path: Path,
    svd_dir: Path,
    output_dir: Path,
    target_size: int,
    skip_hub_probes: bool,
    probe_fn=None,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path)
    leaderboard = pd.read_csv(leaderboard_path)
    leaderboard_canonical, leaderboard_raw = collapse_leaderboard_duplicates(leaderboard)
    atlas = load_atlas_candidates(atlas_json_path)

    metadata["repo_id_norm"] = metadata["model_id"].map(normalize_repo_id)
    metadata["revision_norm"] = metadata["model_id"].map(lambda value: parse_repo_and_revision(value)[1])
    metadata["canonical_model_key"] = [
        canonical_model_key(repo_id, revision)
        for repo_id, revision in zip(metadata["repo_id_norm"], metadata["revision_norm"], strict=False)
    ]

    combined = pd.concat([metadata, atlas], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["canonical_model_key"], keep="first")
    combined = combined.merge(
        leaderboard_canonical.drop(columns=["model_id"], errors="ignore"),
        on="canonical_model_key",
        how="left",
        suffixes=("", "_lb"),
    )
    if "model_id_lb" in combined.columns:
        combined["model_id"] = combined["model_id"].fillna(combined["model_id_lb"])
    combined["source_model"] = combined["source_model"].fillna("")
    combined = seed_analysis_status(combined, svd_dir)
    combined = add_lineage_flags(combined)
    combined = apply_type_taxonomy(combined)
    canonical = select_to_target(combined, target_size=target_size)

    if not skip_hub_probes:
        if probe_fn is not None:
            probe_callable = probe_fn
        else:
            from huggingface_hub import HfApi

            from fetch_data.hub_preflight import dry_run_small_file, probe_repo_fetchability

            api = HfApi()
            probe_callable = lambda row: probe_repo_fetchability(
                api=api,
                repo_id=row["repo_id_norm"],
                revision=row.get("revision_norm"),
                loader_scenario=row["loader_scenario"],
                base_model_name_or_path=row.get("source_model"),
                dry_run_probe=dry_run_small_file,
            )
        probe_rows = []
        for _, row in canonical.iterrows():
            probe = probe_callable(row)
            if isinstance(probe, dict):
                probe_rows.append(probe)
            else:
                probe_rows.append(probe.__dict__)
        if probe_rows:
            probe_frame = pd.DataFrame(probe_rows)
            canonical = canonical.copy()
            for column in ["availability_status", "fetchability_status", "failure_class"]:
                if column in probe_frame.columns:
                    canonical[column] = probe_frame[column].values
            probe_failures = canonical[canonical["failure_class"].notna()].copy()
        else:
            probe_failures = canonical.iloc[0:0].copy()
    else:
        probe_failures = canonical.iloc[0:0].copy()

    prediction_ready = (
        canonical[canonical["Average ⬆️"].notna()].copy()
        if "Average ⬆️" in canonical.columns
        else canonical.iloc[0:0].copy()
    )
    analysis_ready = canonical[canonical["analysis_artifact_present"]].copy()

    canonical.to_csv(output_dir / "model_zoo_phase2.csv", index=False)
    _write_parquet_if_possible(canonical, output_dir / "model_zoo_phase2.parquet")
    prediction_ready.to_csv(output_dir / "model_zoo_phase2_prediction_ready.csv", index=False)
    analysis_ready.to_csv(output_dir / "model_zoo_phase2_analysis_ready.csv", index=False)
    leaderboard_raw.to_csv(output_dir / "model_zoo_phase2_leaderboard_raw.csv", index=False)
    probe_failures.to_csv(output_dir / "model_zoo_phase2_probe_failures.csv", index=False)

    return {
        "canonical": canonical,
        "prediction_ready": prediction_ready,
        "analysis_ready": analysis_ready,
        "leaderboard_raw": leaderboard_raw,
        "probe_failures": probe_failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the phase 2 curated model zoo.")
    parser.add_argument("--metadata-path", type=Path, default=Path("data/sampled_metadata.csv"))
    parser.add_argument("--leaderboard-path", type=Path, default=Path("data/sampled_leaderboard.csv"))
    parser.add_argument("--atlas-json-path", type=Path, default=Path("fetch_data/large_nlp_data.json"))
    parser.add_argument("--svd-dir", type=Path, default=Path("svd_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/curated"))
    parser.add_argument("--target-size", type=int, default=15000)
    parser.add_argument("--skip-hub-probes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_phase2_outputs(
        metadata_path=args.metadata_path,
        leaderboard_path=args.leaderboard_path,
        atlas_json_path=args.atlas_json_path,
        svd_dir=args.svd_dir,
        output_dir=args.output_dir,
        target_size=args.target_size,
        skip_hub_probes=args.skip_hub_probes,
    )


if __name__ == "__main__":
    main()
