from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alpha_cluster_dashboard.data import load_alpha_records, records_to_frame
from spectral_research.benchmark import build_audit_manifest, build_split_artifacts
from spectral_research.pipeline import run_retrieval_pipeline, write_benchmark_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run preliminary spectral family retrieval experiments.")
    parser.add_argument("--metrics-dir", type=Path, default=REPO_ROOT / "svd_results")
    parser.add_argument("--lineage-csv", type=Path, default=REPO_ROOT / "data" / "sampled_metadata.csv")
    parser.add_argument("--leaderboard-csv", type=Path, default=REPO_ROOT / "data" / "sampled_leaderboard.csv")
    parser.add_argument(
        "--freeze-csv",
        type=Path,
        default=REPO_ROOT / "data" / "manual_audits" / "spectral_family_retrieval_label_freeze.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "research_runs" / "preliminary" / "retrieval" / "spectral_family_retrieval",
    )
    parser.add_argument("--min-family-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _seed_frame(manifest: pd.DataFrame) -> pd.DataFrame:
    family_sizes = manifest["candidate_family"].value_counts(dropna=True).rename("candidate_family_size")
    seed = manifest.merge(family_sizes, left_on="candidate_family", right_index=True, how="left")
    columns = [
        "model_id",
        "display_name",
        "candidate_family",
        "candidate_family_size",
        "source_model",
        "base_model_relation",
        "common_schema",
        "query_bucket",
        "Architecture",
        "#Params (B)",
        "Generation",
        "Type",
    ]
    available = [column for column in columns if column in seed.columns]
    return seed.loc[:, available].sort_values(
        ["candidate_family_size", "candidate_family", "display_name"],
        ascending=[False, True, True],
        na_position="last",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_alpha_records(args.metrics_dir)
    records_frame = records_to_frame(records)
    metadata = pd.read_csv(args.lineage_csv)
    leaderboard = pd.read_csv(args.leaderboard_csv)
    audited_labels = pd.read_csv(args.freeze_csv) if args.freeze_csv.exists() else pd.DataFrame(
        columns=["model_id", "audited_base_family", "label_status", "bucket_override", "audit_notes"]
    )

    manifest = build_audit_manifest(records_frame, metadata, leaderboard, audited_labels)
    seed = _seed_frame(manifest)
    seed.to_csv(args.output_dir / "tier_a_audit_seed.csv", index=False)

    split_artifacts = build_split_artifacts(
        manifest,
        min_family_size=args.min_family_size,
        calibration_fraction=0.25,
        seed=args.seed,
    )
    gallery_supported = split_artifacts["gallery_supported"]
    queries_supported = split_artifacts["queries_supported"]
    queries_unsupported = split_artifacts["queries_unsupported"]
    queries_heterogeneous = split_artifacts["queries_heterogeneous"]

    if gallery_supported.empty or queries_supported.empty:
        retrieval_metrics = pd.DataFrame(
            columns=["method", "slice", "top1_accuracy", "mrr", "recall@1", "recall@3", "recall@5", "n_queries"]
        )
        rejection_metrics = pd.DataFrame(
            columns=["method", "auroc", "auprc", "fpr95", "accepted_fraction", "accepted_precision"]
        )
        risk_coverage = pd.DataFrame(columns=["coverage", "risk", "threshold", "method"])
        neighborhood = pd.DataFrame(columns=["family", "neighbor", "rank", "score"])
    else:
        retrieval_metrics, rejection_metrics, risk_coverage, neighborhood = run_retrieval_pipeline(
            records=records,
            gallery_supported=gallery_supported,
            queries_supported=queries_supported,
            queries_unsupported=queries_unsupported,
            queries_heterogeneous=queries_heterogeneous,
        )

    split_payload = {
        "seed": args.seed,
        "min_family_size": args.min_family_size,
        "gallery_family_count": int(gallery_supported["audited_base_family"].nunique()) if not gallery_supported.empty else 0,
        "supported_query_count": int(len(queries_supported)),
        "unsupported_query_count": int(len(queries_unsupported)),
        "heterogeneous_query_count": int(len(queries_heterogeneous)),
    }
    write_benchmark_artifacts(
        output_dir=args.output_dir,
        retrieval_metrics=retrieval_metrics,
        rejection_metrics=rejection_metrics,
        risk_coverage=risk_coverage,
        gallery_supported=gallery_supported,
        queries_supported=queries_supported,
        queries_unsupported=queries_unsupported,
        queries_heterogeneous=queries_heterogeneous,
        audited_labels=audited_labels,
        split_payload=split_payload,
        neighborhood=neighborhood,
    )


if __name__ == "__main__":
    main()
