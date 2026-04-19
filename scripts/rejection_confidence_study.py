from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alpha_cluster_dashboard.data import load_alpha_records
from spectral_research.benchmark import build_audit_manifest, build_split_artifacts
from spectral_research.evaluation import open_set_metrics
from spectral_research.pipeline import (
    MATCH_SLICES,
    OVERRIDE_COLUMNS,
    _metadata_feature_map,
    _rank_query,
    _record_lookup,
    _records_frame,
    _read_csv,
    _family_prototypes,
)
from spectral_research.rejection import confidence_from_ranked_scores, precision_at_coverages

METHODS = ("full_module_trace", "attention_trace", "pl_alpha_scalar", "metadata_only")
RULES = ("top_score", "margin", "score_plus_margin")
FIXED_COVERAGES = (0.1, 0.25, 0.5)
BREAKDOWN_METHODS = ("full_module_trace", "attention_trace")
BREAKDOWN_BUCKETS = ("unsupported_unknown", "unsupported_special", "heterogeneous")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the rejection confidence mini-study on the strict family benchmark."
    )
    parser.add_argument("--metrics-dir", type=Path, default=Path("svd_results/metrics"))
    parser.add_argument(
        "--lineage-csv",
        type=Path,
        default=Path("data/sampled_metadata.csv"),
    )
    parser.add_argument(
        "--leaderboard-csv",
        type=Path,
        default=Path("data/OpenLLMLeaderboard2.csv"),
    )
    parser.add_argument(
        "--freeze-csv",
        type=Path,
        default=Path("data/manual_audits/spectral_family_retrieval_label_freeze.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research_runs/preliminary/rejection/rejection_confidence_study_strict_v1"),
    )
    parser.add_argument("--min-family-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _safe_open_set_metrics(confidence: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if confidence.size == 0 or labels.size == 0 or len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "fpr95": float("nan")}
    return open_set_metrics(confidence, labels)


def _coverage_precision(frame: pd.DataFrame, coverage: float) -> float:
    if frame.empty:
        return float("nan")
    matches = frame.loc[np.isclose(frame["coverage"].to_numpy(dtype=float), coverage), "accepted_precision"]
    if matches.empty:
        return float("nan")
    return float(matches.iloc[0])


def _build_state(args: argparse.Namespace) -> dict[str, object]:
    records = load_alpha_records(args.metrics_dir)
    records_frame = _records_frame(records)
    metadata_frame = _read_csv(args.lineage_csv)
    leaderboard_csv = args.leaderboard_csv
    if not leaderboard_csv.exists():
        fallback = Path("data/sampled_leaderboard.csv")
        if fallback.exists():
            leaderboard_csv = fallback
    leaderboard_frame = _read_csv(leaderboard_csv)
    overrides_frame = _read_csv(args.freeze_csv, columns=OVERRIDE_COLUMNS)

    audited_manifest = build_audit_manifest(
        records_frame,
        metadata_frame,
        leaderboard_frame,
        overrides_frame,
    )
    split_artifacts = build_split_artifacts(
        audited_manifest,
        min_family_size=args.min_family_size,
        seed=args.seed,
    )
    metadata_vectors = _metadata_feature_map(audited_manifest)
    records_by_model_id = _record_lookup(records)
    prototypes = _family_prototypes(
        split_artifacts["gallery_supported"],
        records_by_model_id,
        metadata_vectors,
    )
    return {
        "audited_manifest": audited_manifest,
        "split_artifacts": split_artifacts,
        "metadata_vectors": metadata_vectors,
        "records_by_model_id": records_by_model_id,
        "prototypes": prototypes,
    }


def _labeled_frames(
    queries_supported: pd.DataFrame,
    queries_unsupported: pd.DataFrame,
    queries_heterogeneous: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration = queries_supported[queries_supported["split_role"] == "calibration_supported"].copy()
    calibration["supported_label"] = 1
    calibration["bucket"] = "supported"

    supported = queries_supported[queries_supported["split_role"] == "test_supported"].copy()
    supported["supported_label"] = 1
    supported["bucket"] = "supported"

    unsupported = queries_unsupported.copy()
    unsupported["supported_label"] = 0
    unsupported["bucket"] = unsupported["query_bucket"].fillna("unsupported_unknown")

    heterogeneous = queries_heterogeneous.copy()
    heterogeneous["supported_label"] = 0
    heterogeneous["bucket"] = "heterogeneous"

    evaluation = pd.concat([supported, unsupported, heterogeneous], ignore_index=True, sort=False)
    return calibration, evaluation


def _collect_rank_rows(
    methods: tuple[str, ...],
    evaluation_rows: pd.DataFrame,
    prototypes: dict[str, object],
    records_by_model_id: dict[str, object],
    metadata_vectors: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in methods:
        for slice_name in MATCH_SLICES:
            for _, row in evaluation_rows.iterrows():
                ranked = _rank_query(
                    row,
                    method,
                    slice_name,
                    prototypes,
                    records_by_model_id,
                    metadata_vectors,
                )
                if not ranked:
                    continue
                families = [family for family, _ in ranked]
                scores = np.asarray([score for _, score in ranked], dtype=float)
                top_score = confidence_from_ranked_scores(scores, rule="top_score")
                margin = confidence_from_ranked_scores(scores, rule="margin")
                best_family = families[0]
                truth_family = row.get("audited_base_family", pd.NA)
                supported_label = int(row["supported_label"])
                rows.append(
                    {
                        "method": method,
                        "slice": slice_name,
                        "model_id": str(row["model_id"]),
                        "bucket": str(row["bucket"]),
                        "supported_label": supported_label,
                        "truth_family": truth_family,
                        "best_family": best_family,
                        "is_correct": bool(supported_label == 1 and best_family == truth_family),
                        "top_score": top_score,
                        "margin": margin,
                    }
                )
    return pd.DataFrame(rows)


def _calibration_norms(
    methods: tuple[str, ...],
    calibration_rows: pd.DataFrame,
    prototypes: dict[str, object],
    records_by_model_id: dict[str, object],
    metadata_vectors: dict[str, np.ndarray],
) -> dict[tuple[str, str], dict[str, float]]:
    norms: dict[tuple[str, str], dict[str, float]] = {}
    if calibration_rows.empty:
        return norms
    calibration_rank_rows = _collect_rank_rows(
        methods,
        calibration_rows,
        prototypes,
        records_by_model_id,
        metadata_vectors,
    )
    for (method, slice_name), group in calibration_rank_rows.groupby(["method", "slice"], sort=False):
        top_scores = group["top_score"].to_numpy(dtype=float)
        margins = group["margin"].to_numpy(dtype=float)
        score_center = float(np.nanmean(top_scores)) if top_scores.size else 0.0
        score_scale = float(np.nanstd(top_scores)) if top_scores.size else 1.0
        margin_center = float(np.nanmean(margins)) if margins.size else 0.0
        margin_scale = float(np.nanstd(margins)) if margins.size else 1.0
        norms[(str(method), str(slice_name))] = {
            "score_center": score_center,
            "score_scale": score_scale if np.isfinite(score_scale) and score_scale > 0 else 1.0,
            "margin_center": margin_center,
            "margin_scale": margin_scale if np.isfinite(margin_scale) and margin_scale > 0 else 1.0,
        }
    return norms


def _add_score_plus_margin(
    rank_rows: pd.DataFrame,
    norms: dict[tuple[str, str], dict[str, float]],
) -> pd.DataFrame:
    output = rank_rows.copy()
    values: list[float] = []
    for row in output.itertuples(index=False):
        norm = norms.get((str(row.method), str(row.slice)))
        if norm is None or not np.isfinite(float(row.top_score)) or not np.isfinite(float(row.margin)):
            values.append(float("nan"))
            continue
        second_score = float(row.top_score) - float(row.margin)
        values.append(
            confidence_from_ranked_scores(
                np.asarray([row.top_score, second_score], dtype=float),
                rule="score_plus_margin",
                score_center=norm["score_center"],
                score_scale=norm["score_scale"],
                margin_center=norm["margin_center"],
                margin_scale=norm["margin_scale"],
            )
        )
    output["score_plus_margin"] = values
    return output


def _metric_rows(rank_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[dict[str, object]] = []
    best_rule_by_method: dict[str, str] = {}
    for method in METHODS:
        best_rule = "top_score"
        best_auroc = float("-inf")
        for slice_name in MATCH_SLICES:
            method_slice = rank_rows[
                (rank_rows["method"] == method) & (rank_rows["slice"] == slice_name)
            ].copy()
            for rule in RULES:
                confidence = method_slice[rule].to_numpy(dtype=float)
                labels = method_slice["supported_label"].to_numpy(dtype=int)
                correct = method_slice["is_correct"].to_numpy(dtype=bool)
                finite = np.isfinite(confidence)
                confidence = confidence[finite]
                labels = labels[finite]
                correct = correct[finite]
                metrics = _safe_open_set_metrics(confidence, labels)
                coverage = precision_at_coverages(confidence, correct, coverages=FIXED_COVERAGES)
                rows.append(
                    {
                        "method": method,
                        "rule": rule,
                        "slice": slice_name,
                        "num_examples": int(confidence.size),
                        "auroc": metrics["auroc"],
                        "auprc": metrics["auprc"],
                        "coverage_10_precision": _coverage_precision(coverage, 0.1),
                        "coverage_25_precision": _coverage_precision(coverage, 0.25),
                        "coverage_50_precision": _coverage_precision(coverage, 0.5),
                    }
                )
                if slice_name == "all" and np.isfinite(metrics["auroc"]) and metrics["auroc"] > best_auroc:
                    best_auroc = float(metrics["auroc"])
                    best_rule = rule
        best_rule_by_method[method] = best_rule
    return pd.DataFrame(rows), best_rule_by_method


def _bucket_breakdown(rank_rows: pd.DataFrame, best_rule_by_method: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in BREAKDOWN_METHODS:
        best_rule = best_rule_by_method.get(method, "top_score")
        method_rows = rank_rows[
            (rank_rows["method"] == method) & (rank_rows["slice"] == "all")
        ].copy()
        supported = method_rows[method_rows["bucket"] == "supported"].copy()
        for bucket in BREAKDOWN_BUCKETS:
            subset = pd.concat(
                [supported, method_rows[method_rows["bucket"] == bucket].copy()],
                ignore_index=True,
                sort=False,
            )
            confidence = subset[best_rule].to_numpy(dtype=float)
            labels = subset["supported_label"].to_numpy(dtype=int)
            correct = subset["is_correct"].to_numpy(dtype=bool)
            finite = np.isfinite(confidence)
            confidence = confidence[finite]
            labels = labels[finite]
            correct = correct[finite]
            metrics = _safe_open_set_metrics(confidence, labels)
            coverage = precision_at_coverages(confidence, correct, coverages=FIXED_COVERAGES)
            rows.append(
                {
                    "method": method,
                    "rule": best_rule,
                    "bucket": bucket,
                    "num_examples": int(confidence.size),
                    "auroc": metrics["auroc"],
                    "auprc": metrics["auprc"],
                    "coverage_10_precision": _coverage_precision(coverage, 0.1),
                    "coverage_25_precision": _coverage_precision(coverage, 0.25),
                    "coverage_50_precision": _coverage_precision(coverage, 0.5),
                }
            )
    return pd.DataFrame(rows)


def _report_text(metrics_frame: pd.DataFrame, bucket_frame: pd.DataFrame) -> str:
    def _table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_No rows._"
        columns = list(frame.columns)
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        rows = []
        for row in frame.itertuples(index=False, name=None):
            values = []
            for value in row:
                if pd.isna(value):
                    values.append("n/a")
                elif isinstance(value, (float, np.floating)):
                    values.append(f"{float(value):.3f}")
                elif isinstance(value, (int, np.integer)):
                    values.append(str(int(value)))
                else:
                    values.append(str(value))
            rows.append("| " + " | ".join(values) + " |")
        return "\n".join([header, separator, *rows])

    lines = [
        "# Rejection Confidence Mini-Study",
        "",
        "This study keeps the strict benchmark and the fingerprint methods fixed.",
        "It only compares simple confidence rules: `top_score`, `margin`, and `score_plus_margin`.",
        "",
        "## Best all-slice AUROC by method",
    ]
    for method in METHODS:
        subset = metrics_frame[
            (metrics_frame["method"] == method) & (metrics_frame["slice"] == "all")
        ].sort_values("auroc", ascending=False)
        if subset.empty:
            lines.append(f"- {method}: no rows")
            continue
        row = subset.iloc[0]
        lines.append(
            f"- {method}: {row['rule']} (AUROC {row['auroc']:.3f}, AUPRC {row['auprc']:.3f}, "
            f"P@10 {row['coverage_10_precision']:.3f}, P@25 {row['coverage_25_precision']:.3f}, "
            f"P@50 {row['coverage_50_precision']:.3f})"
        )
    lines.extend(
        [
            "",
            "## Bucket breakdown",
            _table(bucket_frame),
            "",
        ]
    )
    return "\n".join(lines)


def run_study(
    *,
    metrics_dir: Path,
    lineage_csv: Path,
    leaderboard_csv: Path,
    freeze_csv: Path,
    output_dir: Path,
    min_family_size: int,
    seed: int,
) -> dict[str, object]:
    args = argparse.Namespace(
        metrics_dir=metrics_dir,
        lineage_csv=lineage_csv,
        leaderboard_csv=leaderboard_csv,
        freeze_csv=freeze_csv,
        output_dir=output_dir,
        min_family_size=min_family_size,
        seed=seed,
    )
    state = _build_state(args)
    split_artifacts = state["split_artifacts"]
    calibration_rows, evaluation_rows = _labeled_frames(
        split_artifacts["queries_supported"],
        split_artifacts["queries_unsupported"],
        split_artifacts["queries_heterogeneous"],
    )
    norms = _calibration_norms(
        METHODS,
        calibration_rows,
        state["prototypes"],
        state["records_by_model_id"],
        state["metadata_vectors"],
    )
    rank_rows = _collect_rank_rows(
        METHODS,
        evaluation_rows,
        state["prototypes"],
        state["records_by_model_id"],
        state["metadata_vectors"],
    )
    rank_rows = _add_score_plus_margin(rank_rows, norms)
    metrics_frame, best_rule_by_method = _metric_rows(rank_rows)
    bucket_frame = _bucket_breakdown(rank_rows, best_rule_by_method)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(output_dir / "confidence_rule_metrics.csv", index=False)
    bucket_frame.to_csv(output_dir / "best_rule_bucket_breakdown.csv", index=False)
    (output_dir / "report.md").write_text(_report_text(metrics_frame, bucket_frame), encoding="utf-8")
    return {
        "metrics": metrics_frame,
        "bucket_breakdown": bucket_frame,
        "best_rule_by_method": best_rule_by_method,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_study(
        metrics_dir=args.metrics_dir,
        lineage_csv=args.lineage_csv,
        leaderboard_csv=args.leaderboard_csv,
        freeze_csv=args.freeze_csv,
        output_dir=args.output_dir,
        min_family_size=args.min_family_size,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
