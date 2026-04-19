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
from spectral_research.benchmark import build_audit_manifest
from spectral_research.evaluation import risk_coverage_curve
from spectral_research.heterogeneity import (
    intrinsic_basics_feature_map,
    structural_basics_feature_map,
)
from spectral_research.pipeline import (
    OVERRIDE_COLUMNS,
    _audit_seed_frame,
    _curve_area,
    _empty_frame,
    _family_neighborhood,
    _family_prototypes,
    _markdown_table,
    _rank_query,
    _read_csv,
    _record_lookup,
    _records_frame,
    _safe_open_set_metrics,
    _safe_retrieval_metrics,
    write_benchmark_artifacts,
)

STRICT_METHODS = (
    "intrinsic_basics_only",
    "pl_alpha_scalar",
    "full_module_trace",
    "shared_role_trace",
    "role_aggregated_trace",
)
LEAN_METHODS = (
    "structural_basics_only",
    "pl_alpha_scalar",
    "full_module_trace",
    "shared_role_trace",
    "role_aggregated_trace",
)
SUPPORTED_SLICES = ("all_supported", "homogeneous_supported", "heterogeneous_supported")
SUPPORTED_SOURCE_SLICES = ("homogeneous_supported", "heterogeneous_supported")
CALIBRATION_FRACTION = 0.25
DEFAULT_SEED = 0
STRICT_DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "research_runs" / "preliminary" / "retrieval" / "mixed_zoo_retrieval_cycle1"
)
LEAN_DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "research_runs" / "preliminary" / "retrieval" / "lean_mixed_zoo_benchmark"
)


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
        default=None,
    )
    parser.add_argument(
        "--benchmark-mode",
        choices=("strict", "lean"),
        default="strict",
    )
    parser.add_argument("--min-family-size", type=int, default=5)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.output_dir is None:
        args.output_dir = (
            LEAN_DEFAULT_OUTPUT_DIR if args.benchmark_mode == "lean" else STRICT_DEFAULT_OUTPUT_DIR
        )
    return args


def build_supported_candidate_slices(
    manifest: pd.DataFrame,
    *,
    min_family_size: int,
    benchmark_mode: str,
) -> dict[str, pd.DataFrame]:
    if benchmark_mode == "strict":
        supported_statuses = {"confirmed"}
    elif benchmark_mode == "lean":
        supported_statuses = {"confirmed", "provisional"}
    else:
        raise ValueError(f"unsupported benchmark mode: {benchmark_mode}")

    homogeneous_supported = manifest[
        manifest["label_status"].isin(supported_statuses)
        & manifest["query_bucket"].eq("supported_candidate")
        & manifest["audited_base_family"].fillna("").astype(str).ne("")
        & manifest["common_schema"].fillna(False).astype(bool)
    ].copy()

    heterogeneous_supported = manifest[
        manifest["label_status"].isin(supported_statuses)
        & manifest["query_bucket"].eq("heterogeneous")
        & manifest["audited_base_family"].fillna("").astype(str).ne("")
        & ~manifest["common_schema"].fillna(False).astype(bool)
    ].copy()

    def _apply_floor(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        counts = frame["audited_base_family"].value_counts()
        keep = counts[counts >= min_family_size].index
        return frame[frame["audited_base_family"].isin(keep)].copy()

    homogeneous_supported = _apply_floor(homogeneous_supported)
    heterogeneous_supported = _apply_floor(heterogeneous_supported)
    unsupported_special = manifest[manifest["query_bucket"].eq("unsupported_special")].copy()
    unsupported_unknown = manifest[manifest["query_bucket"].eq("unsupported_unknown")].copy()
    return {
        "homogeneous_supported": homogeneous_supported.sort_values("model_id").reset_index(drop=True),
        "heterogeneous_supported": heterogeneous_supported.sort_values("model_id").reset_index(drop=True),
        "unsupported_special": unsupported_special.sort_values("model_id").reset_index(drop=True),
        "unsupported_unknown": unsupported_unknown.sort_values("model_id").reset_index(drop=True),
    }


def methods_for_benchmark_mode(benchmark_mode: str) -> tuple[str, ...]:
    if benchmark_mode == "strict":
        return STRICT_METHODS
    if benchmark_mode == "lean":
        return LEAN_METHODS
    raise ValueError(f"unsupported benchmark mode: {benchmark_mode}")


def metadata_vectors_for_benchmark_mode(
    audited_manifest: pd.DataFrame,
    benchmark_mode: str,
) -> dict[str, np.ndarray]:
    if benchmark_mode == "strict":
        return intrinsic_basics_feature_map(audited_manifest)
    if benchmark_mode == "lean":
        return structural_basics_feature_map(audited_manifest)
    raise ValueError(f"unsupported benchmark mode: {benchmark_mode}")


def build_supported_split_frames(
    slices: dict[str, pd.DataFrame],
    *,
    seed: int = DEFAULT_SEED,
    calibration_fraction: float = CALIBRATION_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    families = sorted(
        {
            str(family)
            for slice_name in SUPPORTED_SOURCE_SLICES
            for family in slices.get(slice_name, _empty_frame(["audited_base_family"]))[
                "audited_base_family"
            ]
            .dropna()
            .astype(str)
            .tolist()
        }
    )
    calibration_families: set[str] = set()
    if families:
        shuffled = np.random.default_rng(seed).permutation(families).tolist()
        calibration_count = min(
            len(shuffled),
            max(1, int(round(len(shuffled) * calibration_fraction))),
        )
        calibration_families = set(shuffled[:calibration_count])

    gallery_frames: list[pd.DataFrame] = []
    query_frames: list[pd.DataFrame] = []
    for slice_name in SUPPORTED_SOURCE_SLICES:
        slice_frame = slices.get(slice_name, _empty_frame(["model_id", "audited_base_family"])).copy()
        if slice_frame.empty:
            continue
        for family, group in slice_frame.groupby("audited_base_family", sort=True):
            ordered = group.sort_values("model_id").reset_index(drop=True)
            gallery_n = max(2, len(ordered) - max(1, len(ordered) // 3))

            gallery_part = ordered.iloc[:gallery_n].copy()
            gallery_part["source_slice"] = slice_name
            gallery_frames.append(gallery_part)

            query_part = ordered.iloc[gallery_n:].copy()
            query_part["split_role"] = (
                "calibration_supported"
                if str(family) in calibration_families
                else "test_supported"
            )
            query_part["evaluation_slice"] = slice_name
            query_frames.append(query_part)

    gallery_supported = (
        pd.concat(gallery_frames, ignore_index=True)
        if gallery_frames
        else _empty_frame(["model_id", "audited_base_family", "source_slice"])
    )
    queries_supported = (
        pd.concat(query_frames, ignore_index=True)
        if query_frames
        else _empty_frame(
            [
                "model_id",
                "audited_base_family",
                "split_role",
                "evaluation_slice",
            ]
        )
    )
    return gallery_supported, queries_supported, sorted(calibration_families)


def _unsupported_queries_frame(slices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for bucket in ("unsupported_special", "unsupported_unknown"):
        frame = slices.get(bucket, _empty_frame(["model_id"])).copy()
        if frame.empty:
            continue
        frame["unsupported_bucket"] = bucket
        frames.append(frame)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return _empty_frame(["model_id", "unsupported_bucket"])


def _calibration_top_scores(
    methods: tuple[str, ...],
    queries_supported: pd.DataFrame,
    prototypes: dict[str, object],
    records_by_model_id: dict[str, object],
    metadata_vectors: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    calibration = queries_supported[queries_supported["split_role"] == "calibration_supported"].copy()
    output: dict[str, np.ndarray] = {}
    for method in methods:
        top_scores: list[float] = []
        for _, row in calibration.iterrows():
            ranked = _rank_query(
                row,
                method,
                "all",
                prototypes,
                records_by_model_id,
                metadata_vectors,
            )
            if ranked:
                top_scores.append(float(ranked[0][1]))
        output[method] = np.asarray(top_scores, dtype=float)
    return output


def _thresholds_from_top_scores(top_scores: dict[str, np.ndarray]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for method, scores in top_scores.items():
        finite = np.asarray(scores, dtype=float)
        finite = finite[np.isfinite(finite)]
        thresholds[method] = float(np.quantile(finite, 0.1)) if finite.size else float("nan")
    return thresholds


def _mixed_zoo_retrieval_rows(
    methods: tuple[str, ...],
    queries_supported: pd.DataFrame,
    prototypes: dict[str, object],
    records_by_model_id: dict[str, object],
    metadata_vectors: dict[str, np.ndarray],
) -> pd.DataFrame:
    test_queries = queries_supported[queries_supported["split_role"] == "test_supported"].copy()
    rows: list[dict[str, object]] = []
    for method in methods:
        for slice_name in SUPPORTED_SLICES:
            if slice_name == "all_supported":
                subset = test_queries
            else:
                subset = test_queries[test_queries["evaluation_slice"] == slice_name].copy()
            rankings: list[list[str]] = []
            truths: list[str] = []
            for _, row in subset.iterrows():
                ranked = _rank_query(
                    row,
                    method,
                    "all",
                    prototypes,
                    records_by_model_id,
                    metadata_vectors,
                )
                if not ranked:
                    continue
                rankings.append([family for family, _ in ranked])
                truths.append(str(row["audited_base_family"]))
            metrics = _safe_retrieval_metrics(rankings, truths)
            rows.append(
                {
                    "method": method,
                    "slice": slice_name,
                    "num_queries": len(truths),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _mixed_zoo_rejection_outputs(
    methods: tuple[str, ...],
    thresholds: dict[str, float],
    queries_supported: pd.DataFrame,
    queries_unsupported: pd.DataFrame,
    prototypes: dict[str, object],
    records_by_model_id: dict[str, object],
    metadata_vectors: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    supported_test = queries_supported[queries_supported["split_role"] == "test_supported"].copy()
    unsupported = queries_unsupported.copy()
    supported_rows = {
        "all_supported": supported_test,
        "homogeneous_supported": supported_test[
            supported_test["evaluation_slice"] == "homogeneous_supported"
        ].copy(),
        "heterogeneous_supported": supported_test[
            supported_test["evaluation_slice"] == "heterogeneous_supported"
        ].copy(),
    }

    metric_rows: list[dict[str, object]] = []
    curve_rows: list[pd.DataFrame] = []
    for method in methods:
        threshold = thresholds.get(method, float("nan"))
        for slice_name, supported_subset in supported_rows.items():
            supported_eval = supported_subset.copy()
            supported_eval["supported_label"] = 1
            unsupported_eval = unsupported.copy()
            unsupported_eval["supported_label"] = 0
            evaluation_rows = pd.concat(
                [supported_eval, unsupported_eval],
                ignore_index=True,
                sort=False,
            )

            confidences: list[float] = []
            labels: list[int] = []
            correct: list[bool] = []
            for _, row in evaluation_rows.iterrows():
                ranked = _rank_query(
                    row,
                    method,
                    "all",
                    prototypes,
                    records_by_model_id,
                    metadata_vectors,
                )
                if not ranked:
                    continue
                best_family, best_score = ranked[0]
                supported_label = int(row["supported_label"])
                confidences.append(float(best_score))
                labels.append(supported_label)
                correct.append(
                    supported_label == 1 and best_family == str(row["audited_base_family"])
                )

            confidence_array = np.asarray(confidences, dtype=float)
            label_array = np.asarray(labels, dtype=int)
            correct_array = np.asarray(correct, dtype=bool)
            finite = np.isfinite(confidence_array)
            confidence_array = confidence_array[finite]
            label_array = label_array[finite]
            correct_array = correct_array[finite]

            metrics = _safe_open_set_metrics(confidence_array, label_array)
            curve = (
                risk_coverage_curve(confidence_array, correct_array)
                if confidence_array.size
                else _empty_frame(["coverage", "risk", "threshold"])
            )
            if not curve.empty:
                curve.insert(0, "slice", slice_name)
                curve.insert(0, "method", method)
                curve_rows.append(curve)

            accepted = (
                confidence_array >= threshold
                if np.isfinite(threshold)
                else np.zeros_like(confidence_array, dtype=bool)
            )
            accepted_precision = (
                float(np.mean(correct_array[accepted])) if accepted.any() else float("nan")
            )
            accepted_coverage = float(np.mean(accepted)) if accepted.size else float("nan")
            metric_rows.append(
                {
                    "method": method,
                    "slice": slice_name,
                    "num_examples": int(confidence_array.size),
                    "threshold": threshold,
                    "accepted_precision": accepted_precision,
                    "accepted_coverage": accepted_coverage,
                    "aurc": _curve_area(curve),
                    **metrics,
                }
            )

    risk_coverage = (
        pd.concat(curve_rows, ignore_index=True)
        if curve_rows
        else _empty_frame(["method", "slice", "coverage", "risk", "threshold"])
    )
    return pd.DataFrame(metric_rows), risk_coverage


def _mixed_zoo_split_payload(
    *,
    seed: int,
    min_family_size: int,
    gallery_supported: pd.DataFrame,
    queries_supported: pd.DataFrame,
    queries_unsupported: pd.DataFrame,
    slices: dict[str, pd.DataFrame],
    calibration_families: list[str],
) -> dict[str, object]:
    test_queries = queries_supported[queries_supported["split_role"] == "test_supported"].copy()
    return {
        "seed": seed,
        "min_family_size": min_family_size,
        "calibration_families": calibration_families,
        "test_families": sorted(test_queries["audited_base_family"].dropna().astype(str).unique().tolist()),
        "counts": {
            "gallery_supported": int(len(gallery_supported)),
            "queries_supported": int(len(queries_supported)),
            "queries_supported_test": int(len(test_queries)),
            "queries_supported_homogeneous": int(
                len(test_queries[test_queries["evaluation_slice"] == "homogeneous_supported"])
            ),
            "queries_supported_heterogeneous": int(
                len(test_queries[test_queries["evaluation_slice"] == "heterogeneous_supported"])
            ),
            "queries_unsupported": int(len(queries_unsupported)),
            "unsupported_special": int(len(slices.get("unsupported_special", pd.DataFrame()))),
            "unsupported_unknown": int(len(slices.get("unsupported_unknown", pd.DataFrame()))),
            "queries_heterogeneous": int(len(slices.get("heterogeneous_supported", pd.DataFrame()))),
        },
    }


def report_title_for_mode(benchmark_mode: str) -> str:
    if benchmark_mode == "lean":
        return "# Lean Mixed-Zoo Benchmark Report"
    return "# Mixed-Zoo Retrieval Cycle 1 Report"


def write_mixed_zoo_report(
    *,
    output_dir: Path,
    benchmark_mode: str,
    methods: tuple[str, ...],
    split_payload: dict[str, object],
    retrieval_metrics: pd.DataFrame,
    rejection_metrics: pd.DataFrame,
    neighborhood: pd.DataFrame,
) -> None:
    counts = split_payload.get("counts", {}) if isinstance(split_payload, dict) else {}
    retrieval_columns = [
        "method",
        "slice",
        "num_queries",
        "top1_accuracy",
        "mrr",
        "recall@1",
        "recall@3",
        "recall@5",
    ]
    rejection_columns = [
        "method",
        "slice",
        "num_examples",
        "threshold",
        "accepted_precision",
        "accepted_coverage",
        "auroc",
        "auprc",
        "fpr95",
        "aurc",
    ]
    neighborhood_columns = ["method", "family", "neighbor", "rank", "score"]
    lines = [
        report_title_for_mode(benchmark_mode),
        "",
        "Headline claim tested: model-intrinsic and spectral retrieval on a mixed zoo of homogeneous, heterogeneous, and unsupported models.",
        "",
        "## Included methods",
        *[f"- {method}" for method in methods],
        "",
        "## Slice summary",
        f"- benchmark_mode: {benchmark_mode}",
        f"- seed: {split_payload.get('seed', 'n/a')}",
        f"- min_family_size: {split_payload.get('min_family_size', 'n/a')}",
        f"- gallery_supported: {counts.get('gallery_supported', 'n/a')}",
        f"- queries_supported: {counts.get('queries_supported', 'n/a')}",
        f"- test_queries_homogeneous_supported: {counts.get('queries_supported_homogeneous', 'n/a')}",
        f"- test_queries_heterogeneous_supported: {counts.get('queries_supported_heterogeneous', 'n/a')}",
        f"- unsupported_special: {counts.get('unsupported_special', 'n/a')}",
        f"- unsupported_unknown: {counts.get('unsupported_unknown', 'n/a')}",
        "",
        "## Retrieval metrics",
        _markdown_table(retrieval_metrics, retrieval_columns),
        "",
        "## Rejection metrics",
        _markdown_table(rejection_metrics, rejection_columns),
        "",
        "## Family neighborhood top-k",
        _markdown_table(neighborhood, neighborhood_columns, limit=15),
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def gate_method_for_benchmark_mode(benchmark_mode: str) -> str:
    return "full_module_trace"


def write_prediction_bridge_gate(output_dir: Path, *, benchmark_mode: str = "strict") -> None:
    gate_method = gate_method_for_benchmark_mode(benchmark_mode)
    metrics = pd.read_csv(output_dir / "retrieval_metrics.csv")
    hetero = metrics[
        (metrics["slice"] == "heterogeneous_supported")
        & (metrics["method"] == gate_method)
    ]
    if hetero.empty:
        decision = "HOLD"
        reason = "No heterogeneous-supported retrieval rows were produced."
    else:
        row = hetero.iloc[0]
        num_queries = float(row.get("num_queries", 0.0))
        top1 = float(row["top1_accuracy"])
        mrr = float(row["mrr"])
        if num_queries <= 0 or not np.isfinite(top1) or not np.isfinite(mrr):
            decision = "HOLD"
            reason = "No heterogeneous-supported retrieval rows were produced."
        elif top1 >= 0.50 and mrr >= 0.65:
            decision = "GO"
            reason = "Mixed-zoo spectral neighborhoods are strong enough for a tiny prediction pilot."
        else:
            decision = "HOLD"
            reason = "Mixed-zoo retrieval is not yet strong enough to justify the prediction bridge."
    (output_dir / "prediction_bridge_gate.md").write_text(
        f"# Prediction Bridge Gate\n\n- Decision: `{decision}`\n- Reason: {reason}\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records = load_alpha_records(args.metrics_dir)
    records_frame = _records_frame(records)
    metadata_frame = _read_csv(args.lineage_csv)
    leaderboard_frame = _read_csv(args.leaderboard_csv)
    empty_overrides = _empty_frame(OVERRIDE_COLUMNS)
    seed_manifest = build_audit_manifest(
        records_frame,
        metadata_frame,
        leaderboard_frame,
        empty_overrides,
    )
    audit_seed = _audit_seed_frame(seed_manifest)
    overrides_frame = _read_csv(args.freeze_csv, columns=OVERRIDE_COLUMNS)
    audited_manifest = build_audit_manifest(
        records_frame,
        metadata_frame,
        leaderboard_frame,
        overrides_frame,
    )
    slices = build_supported_candidate_slices(
        audited_manifest,
        min_family_size=args.min_family_size,
        benchmark_mode=args.benchmark_mode,
    )
    gallery_supported, queries_supported, calibration_families = build_supported_split_frames(
        slices,
        seed=DEFAULT_SEED,
    )
    queries_unsupported = _unsupported_queries_frame(slices)
    queries_heterogeneous = slices.get("heterogeneous_supported", pd.DataFrame()).copy()

    methods = methods_for_benchmark_mode(args.benchmark_mode)
    metadata_vectors = metadata_vectors_for_benchmark_mode(
        audited_manifest,
        args.benchmark_mode,
    )
    records_by_model_id = _record_lookup(records)
    prototypes = _family_prototypes(gallery_supported, records_by_model_id, metadata_vectors)

    calibration_top_scores = _calibration_top_scores(
        methods,
        queries_supported,
        prototypes,
        records_by_model_id,
        metadata_vectors,
    )
    thresholds = _thresholds_from_top_scores(calibration_top_scores)
    retrieval_metrics = _mixed_zoo_retrieval_rows(
        methods,
        queries_supported,
        prototypes,
        records_by_model_id,
        metadata_vectors,
    )
    rejection_metrics, risk_coverage = _mixed_zoo_rejection_outputs(
        methods,
        thresholds,
        queries_supported,
        queries_unsupported,
        prototypes,
        records_by_model_id,
        metadata_vectors,
    )
    neighborhood = _family_neighborhood(methods, prototypes)
    split_payload = _mixed_zoo_split_payload(
        seed=DEFAULT_SEED,
        min_family_size=args.min_family_size,
        gallery_supported=gallery_supported,
        queries_supported=queries_supported,
        queries_unsupported=queries_unsupported,
        slices=slices,
        calibration_families=calibration_families,
    )
    write_benchmark_artifacts(
        output_dir=args.output_dir,
        audit_seed=audit_seed,
        retrieval_metrics=retrieval_metrics,
        rejection_metrics=rejection_metrics,
        risk_coverage=risk_coverage,
        gallery_supported=gallery_supported,
        queries_supported=queries_supported,
        queries_unsupported=queries_unsupported,
        queries_heterogeneous=queries_heterogeneous,
        audited_labels=audited_manifest,
        split_payload=split_payload,
        neighborhood=neighborhood,
    )
    write_mixed_zoo_report(
        output_dir=args.output_dir,
        benchmark_mode=args.benchmark_mode,
        methods=methods,
        split_payload=split_payload,
        retrieval_metrics=retrieval_metrics,
        rejection_metrics=rejection_metrics,
        neighborhood=neighborhood,
    )
    if args.output_dir.exists():
        write_prediction_bridge_gate(args.output_dir, benchmark_mode=args.benchmark_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
