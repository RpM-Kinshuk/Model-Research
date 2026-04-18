from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from alpha_cluster_dashboard.data import AlphaRecord, load_alpha_records, records_to_frame
from spectral_research.benchmark import build_audit_manifest, build_split_artifacts
from spectral_research.evaluation import open_set_metrics, retrieval_metrics, risk_coverage_curve
from spectral_research.features import (
    alpha_scalar_features,
    attention_trace_features,
    full_module_trace_features,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHODS = (
    "metadata_only",
    "pl_alpha_scalar",
    "attention_trace",
    "full_module_trace",
    "summary_module",
)
RETRIEVAL_KS = (1, 3, 5)
MATCH_SLICES = ("all", "architecture_matched", "size_matched")
ATTENTION_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")
OVERRIDE_COLUMNS = [
    "model_id",
    "audited_base_family",
    "label_status",
    "bucket_override",
    "audit_notes",
]
AUDIT_SEED_COLUMNS = [
    "model_id",
    "display_name",
    "candidate_family",
    "audited_base_family",
    "label_status",
    "query_bucket",
    "source_model",
    "base_model_relation",
    "Architecture",
    "#Params (B)",
    "Generation",
    "Type",
    "common_schema",
    "schema_name",
]


@dataclass(frozen=True)
class FamilyPrototype:
    family: str
    modules: tuple[str, ...]
    full_alpha: np.ndarray
    metadata_vector: np.ndarray
    scalar_vector: np.ndarray
    summary_vector: np.ndarray
    architecture: str | None
    size_bucket: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the audited spectral family retrieval preliminary benchmark."
    )
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
        default=REPO_ROOT / "research_runs" / "spectral_family_retrieval",
    )
    parser.add_argument("--min-family-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _empty_frame(columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _read_csv(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        if columns is None:
            raise FileNotFoundError(path)
        return _empty_frame(columns)
    frame = pd.read_csv(path)
    if columns is None:
        return frame
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, columns].copy()


def _records_frame(records: list[AlphaRecord]) -> pd.DataFrame:
    frame = records_to_frame(records)
    if not frame.empty:
        return frame
    return _empty_frame(
        [
            "model_id",
            "display_name",
            "path",
            "num_layers",
            "raw_module_count",
            "raw_modules",
            "common_schema",
            "schema_name",
        ]
    )


def _finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _first_text(series: pd.Series) -> str | None:
    for value in series.tolist():
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _audit_seed_frame(manifest: pd.DataFrame) -> pd.DataFrame:
    seed = manifest.copy()
    for column in AUDIT_SEED_COLUMNS:
        if column not in seed.columns:
            seed[column] = pd.NA
    return seed.loc[:, AUDIT_SEED_COLUMNS].sort_values(["common_schema", "display_name"], ascending=[False, True])


def _metadata_feature_map(manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    if manifest.empty:
        return {}
    features = pd.DataFrame({"model_id": manifest["model_id"].astype(str)})
    features["generation"] = pd.to_numeric(manifest.get("Generation"), errors="coerce").fillna(-1.0)
    features["params_b"] = pd.to_numeric(manifest.get("#Params (B)"), errors="coerce").fillna(-1.0)

    categorical = pd.DataFrame(
        {
            "Architecture": manifest.get("Architecture", pd.Series(index=manifest.index, dtype=object)),
            "Type": manifest.get("Type", pd.Series(index=manifest.index, dtype=object)),
            "size_bucket": manifest.get("size_bucket", pd.Series(index=manifest.index, dtype=object)),
        }
    ).fillna("unknown")
    dummies = pd.get_dummies(categorical.astype(str), prefix=["arch", "type", "size"], dtype=float)
    matrix = pd.concat([features, dummies], axis=1)
    columns = [column for column in matrix.columns if column != "model_id"]
    return {
        row.model_id: row_values
        for row, row_values in zip(
            matrix.itertuples(index=False),
            matrix.loc[:, columns].to_numpy(dtype=float),
            strict=False,
        )
    }


def _record_lookup(records: list[AlphaRecord]) -> dict[str, AlphaRecord]:
    return {record.model_id: record for record in records}


def _module_index(modules: tuple[str, ...]) -> dict[str, int]:
    return {module: index for index, module in enumerate(modules)}


def _attention_modules(modules: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(module for module in modules if module.endswith(ATTENTION_SUFFIXES))


def _align_record(record: AlphaRecord, target_modules: tuple[str, ...]) -> np.ndarray:
    aligned = np.full((record.alpha.shape[0], len(target_modules)), np.nan, dtype=float)
    source_index = _module_index(record.raw_modules)
    for target_index, module in enumerate(target_modules):
        source = source_index.get(module)
        if source is not None:
            aligned[:, target_index] = np.asarray(record.alpha[:, source], dtype=float)
    return aligned


def _stack_family_records(records: list[AlphaRecord], target_modules: tuple[str, ...]) -> np.ndarray:
    max_layers = max(record.alpha.shape[0] for record in records)
    stack = np.full((len(records), max_layers, len(target_modules)), np.nan, dtype=float)
    for row_index, record in enumerate(records):
        aligned = _align_record(record, target_modules)
        stack[row_index, : aligned.shape[0], :] = aligned
    return stack


def _summary_vector_from_alpha(alpha: np.ndarray) -> np.ndarray:
    module_means = np.nanmean(alpha, axis=0)
    module_stds = np.nanstd(alpha, axis=0)
    return np.concatenate([module_means, module_stds]).astype(float)


def _summary_vector(record: AlphaRecord, target_modules: tuple[str, ...]) -> np.ndarray:
    aligned = _align_record(record, target_modules)
    return _summary_vector_from_alpha(aligned)


def _family_prototypes(
    gallery_supported: pd.DataFrame,
    records_by_model_id: dict[str, AlphaRecord],
    metadata_vectors: dict[str, np.ndarray],
) -> dict[str, FamilyPrototype]:
    prototypes: dict[str, FamilyPrototype] = {}
    for family, group in gallery_supported.groupby("audited_base_family", sort=True):
        model_records = [
            records_by_model_id[row.model_id]
            for row in group.itertuples(index=False)
            if row.model_id in records_by_model_id
        ]
        if not model_records:
            continue
        modules = tuple(model_records[0].raw_modules)
        stack = _stack_family_records(model_records, modules)
        full_alpha = np.nanmean(stack, axis=0)
        scalar_rows = np.vstack(
            [alpha_scalar_features(record.alpha).reshape(-1) for record in model_records]
        )
        summary_rows = np.vstack([_summary_vector(record, modules) for record in model_records])
        metadata_rows = np.vstack(
            [
                metadata_vectors[row.model_id]
                for row in group.itertuples(index=False)
                if row.model_id in metadata_vectors
            ]
        )
        prototypes[str(family)] = FamilyPrototype(
            family=str(family),
            modules=modules,
            full_alpha=full_alpha,
            metadata_vector=np.nanmean(metadata_rows, axis=0),
            scalar_vector=np.nanmean(scalar_rows, axis=0),
            summary_vector=np.nanmean(summary_rows, axis=0),
            architecture=_first_text(group["Architecture"]) if "Architecture" in group.columns else None,
            size_bucket=_first_text(group["size_bucket"]) if "size_bucket" in group.columns else None,
        )
    return prototypes


def _negative_l2(left: np.ndarray, right: np.ndarray) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    mask = np.isfinite(left_array) & np.isfinite(right_array)
    if not mask.any():
        return float("nan")
    delta = left_array[mask] - right_array[mask]
    return float(-np.mean(delta * delta))


def _trace_score(
    record: AlphaRecord,
    prototype: FamilyPrototype,
    *,
    attention_only: bool,
) -> float:
    target_modules = prototype.modules
    if attention_only:
        target_modules = _attention_modules(target_modules)
    if not target_modules:
        return float("nan")
    query_trace = _align_record(record, target_modules)
    if attention_only:
        prototype_trace = attention_trace_features(prototype.full_alpha, list(prototype.modules))
    else:
        prototype_trace = full_module_trace_features(prototype.full_alpha)
    layer_count = min(query_trace.shape[0], prototype_trace.shape[0])
    if layer_count <= 0:
        return float("nan")
    return _negative_l2(query_trace[:layer_count], prototype_trace[:layer_count])


def _score_query(
    *,
    method: str,
    row: pd.Series,
    record: AlphaRecord | None,
    prototype: FamilyPrototype,
    metadata_vectors: dict[str, np.ndarray],
    parent_scores: dict[str, dict[str, float]] | None = None,
) -> float:
    if method == "hybrid_lite":
        if parent_scores is None:
            return float("nan")
        return float(parent_scores["hybrid_lite"].get(prototype.family, float("nan")))
    if method == "metadata_only":
        vector = metadata_vectors.get(str(row["model_id"]))
        if vector is None:
            return float("nan")
        return _negative_l2(vector, prototype.metadata_vector)
    if record is None:
        return float("nan")
    if method == "pl_alpha_scalar":
        vector = alpha_scalar_features(record.alpha).reshape(-1)
        return _negative_l2(vector, prototype.scalar_vector)
    if method == "summary_module":
        return _negative_l2(_summary_vector(record, prototype.modules), prototype.summary_vector)
    if method == "attention_trace":
        return _trace_score(record, prototype, attention_only=True)
    if method == "full_module_trace":
        return _trace_score(record, prototype, attention_only=False)
    raise KeyError(f"Unsupported method: {method}")


def _normalize_score_map(scores: dict[str, float]) -> dict[str, float]:
    finite_items = [(family, score) for family, score in scores.items() if np.isfinite(score)]
    if not finite_items:
        return {}
    values = np.asarray([score for _, score in finite_items], dtype=float)
    low = float(values.min())
    high = float(values.max())
    if high - low <= 1e-12:
        return {family: 0.0 for family, _ in finite_items}
    return {family: (score - low) / (high - low) for family, score in finite_items}


def _parent_scores_for_query(
    row: pd.Series,
    record: AlphaRecord | None,
    families: list[str],
    prototypes: dict[str, FamilyPrototype],
    metadata_vectors: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    parent_names = ("metadata_only", "full_module_trace")
    parent_scores: dict[str, dict[str, float]] = {}
    for method in parent_names:
        scores = {
            family: _score_query(
                method=method,
                row=row,
                record=record,
                prototype=prototypes[family],
                metadata_vectors=metadata_vectors,
            )
            for family in families
        }
        parent_scores[method] = scores

    metadata_scaled = _normalize_score_map(parent_scores["metadata_only"])
    trace_scaled = _normalize_score_map(parent_scores["full_module_trace"])
    families_with_both = sorted(set(metadata_scaled) & set(trace_scaled))
    parent_scores["hybrid_lite"] = {
        family: 0.5 * (metadata_scaled[family] + trace_scaled[family])
        for family in families_with_both
    }
    return parent_scores


def _candidate_families(
    row: pd.Series,
    slice_name: str,
    prototypes: dict[str, FamilyPrototype],
) -> list[str]:
    truth_family = row.get("audited_base_family")
    candidates: list[str] = []
    for family, prototype in prototypes.items():
        if slice_name == "architecture_matched":
            if prototype.architecture != _first_text(pd.Series([row.get("Architecture")])):
                continue
        elif slice_name == "size_matched":
            if prototype.size_bucket != _first_text(pd.Series([row.get("size_bucket")])):
                continue
        candidates.append(family)
    if slice_name != "all" and truth_family in prototypes and truth_family not in candidates:
        return []
    return sorted(candidates)


def _rank_query(
    row: pd.Series,
    method: str,
    slice_name: str,
    prototypes: dict[str, FamilyPrototype],
    records_by_model_id: dict[str, AlphaRecord],
    metadata_vectors: dict[str, np.ndarray],
) -> list[tuple[str, float]]:
    families = _candidate_families(row, slice_name, prototypes)
    if not families:
        return []
    record = records_by_model_id.get(str(row["model_id"]))
    parent_scores = None
    if method == "hybrid_lite":
        parent_scores = _parent_scores_for_query(
            row,
            record,
            families,
            prototypes,
            metadata_vectors,
        )
    scores = []
    for family in families:
        score = _score_query(
            method=method,
            row=row,
            record=record,
            prototype=prototypes[family],
            metadata_vectors=metadata_vectors,
            parent_scores=parent_scores,
        )
        if np.isfinite(score):
            scores.append((family, float(score)))
    scores.sort(key=lambda item: (-item[1], item[0]))
    return scores


def _safe_retrieval_metrics(
    rankings: list[list[str]],
    truths: list[str],
) -> dict[str, float]:
    if not rankings or not truths:
        return {
            "top1_accuracy": float("nan"),
            "mrr": float("nan"),
            "recall@1": float("nan"),
            "recall@3": float("nan"),
            "recall@5": float("nan"),
        }
    return retrieval_metrics(rankings, truths, ks=RETRIEVAL_KS)


def _safe_open_set_metrics(confidence: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if confidence.size == 0 or labels.size == 0 or len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "fpr95": float("nan")}
    return open_set_metrics(confidence, labels)


def _curve_area(curve: pd.DataFrame) -> float:
    if curve.empty:
        return float("nan")
    return float(np.trapz(curve["risk"].to_numpy(dtype=float), curve["coverage"].to_numpy(dtype=float)))


def _calibration_metrics(
    queries_supported: pd.DataFrame,
    prototypes: dict[str, FamilyPrototype],
    records_by_model_id: dict[str, AlphaRecord],
    metadata_vectors: dict[str, np.ndarray],
) -> dict[str, dict[str, object]]:
    calibration = queries_supported[queries_supported["split_role"] == "calibration_supported"].copy()
    output: dict[str, dict[str, object]] = {}
    for method in ("metadata_only", "full_module_trace", "hybrid_lite"):
        rankings: list[list[str]] = []
        truths: list[str] = []
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
            if not ranked:
                continue
            rankings.append([family for family, _ in ranked])
            truths.append(str(row["audited_base_family"]))
            top_scores.append(ranked[0][1])
        output[method] = {
            "metrics": _safe_retrieval_metrics(rankings, truths),
            "top_scores": np.asarray(top_scores, dtype=float),
        }
    return output


def _selected_methods(calibration: dict[str, dict[str, object]]) -> tuple[str, ...]:
    methods = list(DEFAULT_METHODS)
    hybrid_metrics = calibration.get("hybrid_lite", {}).get("metrics", {})
    metadata_metrics = calibration.get("metadata_only", {}).get("metrics", {})
    full_metrics = calibration.get("full_module_trace", {}).get("metrics", {})
    hybrid_mrr = float(hybrid_metrics.get("mrr", float("nan")))
    if (
        np.isfinite(hybrid_mrr)
        and hybrid_mrr > float(metadata_metrics.get("mrr", float("nan")))
        and hybrid_mrr > float(full_metrics.get("mrr", float("nan")))
    ):
        methods.append("hybrid_lite")
    return tuple(methods)


def _method_thresholds(
    methods: tuple[str, ...],
    calibration: dict[str, dict[str, object]],
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for method in methods:
        scores = np.asarray(
            calibration.get(method, {}).get("top_scores", np.asarray([], dtype=float)),
            dtype=float,
        )
        finite = scores[np.isfinite(scores)]
        thresholds[method] = (
            float(np.quantile(finite, 0.1)) if finite.size else float("nan")
        )
    return thresholds


def _retrieval_rows(
    methods: tuple[str, ...],
    queries_supported: pd.DataFrame,
    prototypes: dict[str, FamilyPrototype],
    records_by_model_id: dict[str, AlphaRecord],
    metadata_vectors: dict[str, np.ndarray],
) -> pd.DataFrame:
    test_queries = queries_supported[queries_supported["split_role"] == "test_supported"].copy()
    rows: list[dict[str, object]] = []
    for method in methods:
        for slice_name in MATCH_SLICES:
            rankings: list[list[str]] = []
            truths: list[str] = []
            for _, row in test_queries.iterrows():
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


def _rejection_outputs(
    methods: tuple[str, ...],
    threshold_by_method: dict[str, float],
    queries_supported: pd.DataFrame,
    queries_unsupported: pd.DataFrame,
    prototypes: dict[str, FamilyPrototype],
    records_by_model_id: dict[str, AlphaRecord],
    metadata_vectors: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    supported = queries_supported[queries_supported["split_role"] == "test_supported"].copy()
    unsupported = queries_unsupported.copy()
    supported["supported_label"] = 1
    unsupported["supported_label"] = 0
    evaluation_rows = pd.concat([supported, unsupported], ignore_index=True, sort=False)

    metric_rows: list[dict[str, object]] = []
    curve_rows: list[pd.DataFrame] = []
    for method in methods:
        threshold = threshold_by_method.get(method, float("nan"))
        for slice_name in MATCH_SLICES:
            confidences: list[float] = []
            labels: list[int] = []
            correct: list[bool] = []
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
                best_family, best_score = ranked[0]
                supported_label = int(row["supported_label"])
                confidences.append(best_score)
                labels.append(supported_label)
                correct.append(supported_label == 1 and best_family == row["audited_base_family"])
            confidence_array = np.asarray(confidences, dtype=float)
            label_array = np.asarray(labels, dtype=int)
            correct_array = np.asarray(correct, dtype=bool)
            finite = np.isfinite(confidence_array)
            confidence_array = confidence_array[finite]
            label_array = label_array[finite]
            correct_array = correct_array[finite]

            metrics = _safe_open_set_metrics(confidence_array, label_array)
            curve = risk_coverage_curve(confidence_array, correct_array) if confidence_array.size else _empty_frame(["coverage", "risk", "threshold"])
            if not curve.empty:
                curve.insert(0, "slice", slice_name)
                curve.insert(0, "method", method)
                curve_rows.append(curve)

            accepted = confidence_array >= threshold if np.isfinite(threshold) else np.zeros_like(confidence_array, dtype=bool)
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
                    "aurc": _curve_area(curve) if isinstance(curve, pd.DataFrame) else float("nan"),
                    **metrics,
                }
            )
    risk_coverage = (
        pd.concat(curve_rows, ignore_index=True)
        if curve_rows
        else _empty_frame(["method", "slice", "coverage", "risk", "threshold"])
    )
    return pd.DataFrame(metric_rows), risk_coverage


def _prototype_score(method: str, left: FamilyPrototype, right: FamilyPrototype) -> float:
    if method == "metadata_only":
        return _negative_l2(left.metadata_vector, right.metadata_vector)
    if method == "pl_alpha_scalar":
        return _negative_l2(left.scalar_vector, right.scalar_vector)
    if method == "summary_module":
        return _negative_l2(left.summary_vector, right.summary_vector)
    if method == "attention_trace":
        left_alpha = attention_trace_features(left.full_alpha, list(left.modules))
        right_alpha = attention_trace_features(right.full_alpha, list(right.modules))
        layer_count = min(left_alpha.shape[0], right_alpha.shape[0])
        if layer_count <= 0:
            return float("nan")
        return _negative_l2(left_alpha[:layer_count], right_alpha[:layer_count])
    if method == "full_module_trace":
        layer_count = min(left.full_alpha.shape[0], right.full_alpha.shape[0])
        if layer_count <= 0:
            return float("nan")
        return _negative_l2(left.full_alpha[:layer_count], right.full_alpha[:layer_count])
    if method == "hybrid_lite":
        metadata = _prototype_score("metadata_only", left, right)
        trace = _prototype_score("full_module_trace", left, right)
        if not np.isfinite(metadata) or not np.isfinite(trace):
            return float("nan")
        return 0.5 * (metadata + trace)
    raise KeyError(f"Unsupported method: {method}")


def _family_neighborhood(methods: tuple[str, ...], prototypes: dict[str, FamilyPrototype], topk: int = 5) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    families = sorted(prototypes)
    for method in methods:
        for family in families:
            scores = []
            for neighbor in families:
                if neighbor == family:
                    continue
                score = _prototype_score(method, prototypes[family], prototypes[neighbor])
                if np.isfinite(score):
                    scores.append((neighbor, score))
            scores.sort(key=lambda item: (-item[1], item[0]))
            for rank, (neighbor, score) in enumerate(scores[:topk], start=1):
                rows.append(
                    {
                        "method": method,
                        "family": family,
                        "neighbor": neighbor,
                        "rank": rank,
                        "score": float(score),
                    }
                )
    return pd.DataFrame(rows)


def _split_payload(
    *,
    seed: int,
    min_family_size: int,
    gallery_supported: pd.DataFrame,
    queries_supported: pd.DataFrame,
    queries_unsupported: pd.DataFrame,
    queries_heterogeneous: pd.DataFrame,
) -> dict[str, object]:
    calibration_families = sorted(
        queries_supported.loc[
            queries_supported.get("split_role").eq("calibration_supported"),
            "audited_base_family",
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    test_families = sorted(
        queries_supported.loc[
            queries_supported.get("split_role").eq("test_supported"),
            "audited_base_family",
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    return {
        "seed": seed,
        "min_family_size": min_family_size,
        "calibration_families": calibration_families,
        "test_families": test_families,
        "counts": {
            "gallery_supported": int(len(gallery_supported)),
            "queries_supported": int(len(queries_supported)),
            "queries_unsupported": int(len(queries_unsupported)),
            "queries_heterogeneous": int(len(queries_heterogeneous)),
        },
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    table = frame.copy()
    for column in columns:
        if column not in table.columns:
            table[column] = pd.NA
    table = table.loc[:, columns].copy()
    if limit is not None:
        table = table.head(limit)

    def _cell(value: object) -> str:
        if pd.isna(value):
            return "n/a"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.3f}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(_cell(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _report_text(
    *,
    methods: tuple[str, ...],
    split_payload: dict[str, object],
    retrieval_metrics_frame: pd.DataFrame,
    rejection_metrics_frame: pd.DataFrame,
    neighborhood_frame: pd.DataFrame,
    heterogeneous_queries: pd.DataFrame,
) -> str:
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
        "# Spectral Family Retrieval Preliminary Report",
        "",
        "Headline claim tested: module-aware spectral candidate generation for base-family retrieval with abstention.",
        "",
        "## Included methods",
        *[f"- {method}" for method in methods],
        "",
        "## Split summary",
        f"- seed: {split_payload.get('seed', 'n/a')}",
        f"- min_family_size: {split_payload.get('min_family_size', 'n/a')}",
        f"- gallery_supported: {counts.get('gallery_supported', 'n/a')}",
        f"- queries_supported: {counts.get('queries_supported', 'n/a')}",
        f"- queries_unsupported: {counts.get('queries_unsupported', 'n/a')}",
        f"- queries_heterogeneous: {counts.get('queries_heterogeneous', 'n/a')}",
        "",
        "## Retrieval metrics",
        _markdown_table(retrieval_metrics_frame, retrieval_columns),
        "",
        "## Rejection metrics",
        _markdown_table(rejection_metrics_frame, rejection_columns),
        "",
        "## Family neighborhood top-k",
        _markdown_table(neighborhood_frame, neighborhood_columns, limit=15),
        "",
        "## Heterogeneous stress bucket",
        f"- rows: {len(heterogeneous_queries)}",
    ]
    return "\n".join(lines) + "\n"


def write_benchmark_artifacts(
    *,
    output_dir: Path,
    audit_seed: pd.DataFrame,
    retrieval_metrics: pd.DataFrame,
    rejection_metrics: pd.DataFrame,
    risk_coverage: pd.DataFrame,
    gallery_supported: pd.DataFrame,
    queries_supported: pd.DataFrame,
    queries_unsupported: pd.DataFrame,
    queries_heterogeneous: pd.DataFrame,
    audited_labels: pd.DataFrame,
    split_payload: dict[str, object],
    neighborhood: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_seed.to_csv(output_dir / "tier_a_audit_seed.csv", index=False)
    audited_labels.to_csv(output_dir / "tier_a_audited_labels.csv", index=False)
    gallery_supported.to_csv(output_dir / "gallery_supported.csv", index=False)
    queries_supported.to_csv(output_dir / "queries_supported.csv", index=False)
    queries_unsupported.to_csv(output_dir / "queries_unsupported.csv", index=False)
    queries_heterogeneous.to_csv(output_dir / "queries_heterogeneous.csv", index=False)
    retrieval_metrics.to_csv(output_dir / "retrieval_metrics.csv", index=False)
    rejection_metrics.to_csv(output_dir / "rejection_metrics.csv", index=False)
    risk_coverage.to_csv(output_dir / "risk_coverage.csv", index=False)
    neighborhood.to_csv(output_dir / "family_neighborhood_topk.csv", index=False)
    with (output_dir / "tier_a_splits.json").open("w", encoding="utf-8") as handle:
        json.dump(split_payload, handle, indent=2, sort_keys=True)
    report = _report_text(
        methods=tuple(retrieval_metrics["method"].dropna().unique().tolist())
        if "method" in retrieval_metrics.columns
        else tuple(),
        split_payload=split_payload,
        retrieval_metrics_frame=retrieval_metrics,
        rejection_metrics_frame=rejection_metrics,
        neighborhood_frame=neighborhood,
        heterogeneous_queries=queries_heterogeneous,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def run_benchmark(
    *,
    metrics_dir: Path,
    lineage_csv: Path,
    leaderboard_csv: Path,
    freeze_csv: Path,
    output_dir: Path,
    min_family_size: int,
    seed: int,
) -> dict[str, object]:
    records = load_alpha_records(metrics_dir)
    records_frame = _records_frame(records)
    metadata_frame = _read_csv(lineage_csv)
    leaderboard_frame = _read_csv(leaderboard_csv)

    empty_overrides = _empty_frame(OVERRIDE_COLUMNS)
    seed_manifest = build_audit_manifest(
        records_frame,
        metadata_frame,
        leaderboard_frame,
        empty_overrides,
    )
    audit_seed = _audit_seed_frame(seed_manifest)
    overrides_frame = _read_csv(freeze_csv, columns=OVERRIDE_COLUMNS)
    audited_manifest = build_audit_manifest(
        records_frame,
        metadata_frame,
        leaderboard_frame,
        overrides_frame,
    )
    split_artifacts = build_split_artifacts(
        audited_manifest,
        min_family_size=min_family_size,
        seed=seed,
    )
    gallery_supported = split_artifacts["gallery_supported"]
    queries_supported = split_artifacts["queries_supported"]
    queries_unsupported = split_artifacts["queries_unsupported"]
    queries_heterogeneous = split_artifacts["queries_heterogeneous"]

    metadata_vectors = _metadata_feature_map(audited_manifest)
    records_by_model_id = _record_lookup(records)
    prototypes = _family_prototypes(gallery_supported, records_by_model_id, metadata_vectors)
    calibration = _calibration_metrics(
        queries_supported,
        prototypes,
        records_by_model_id,
        metadata_vectors,
    )
    methods = _selected_methods(calibration)
    thresholds = _method_thresholds(methods, calibration)
    retrieval_metrics_frame = _retrieval_rows(
        methods,
        queries_supported,
        prototypes,
        records_by_model_id,
        metadata_vectors,
    )
    rejection_metrics_frame, risk_coverage_frame = _rejection_outputs(
        methods,
        thresholds,
        queries_supported,
        queries_unsupported,
        prototypes,
        records_by_model_id,
        metadata_vectors,
    )
    neighborhood_frame = _family_neighborhood(methods, prototypes)
    split_payload = _split_payload(
        seed=seed,
        min_family_size=min_family_size,
        gallery_supported=gallery_supported,
        queries_supported=queries_supported,
        queries_unsupported=queries_unsupported,
        queries_heterogeneous=queries_heterogeneous,
    )
    write_benchmark_artifacts(
        output_dir=output_dir,
        audit_seed=audit_seed,
        retrieval_metrics=retrieval_metrics_frame,
        rejection_metrics=rejection_metrics_frame,
        risk_coverage=risk_coverage_frame,
        gallery_supported=gallery_supported,
        queries_supported=queries_supported,
        queries_unsupported=queries_unsupported,
        queries_heterogeneous=queries_heterogeneous,
        audited_labels=audited_manifest,
        split_payload=split_payload,
        neighborhood=neighborhood_frame,
    )
    return {
        "audit_seed": audit_seed,
        "audited_manifest": audited_manifest,
        "split_artifacts": split_artifacts,
        "retrieval_metrics": retrieval_metrics_frame,
        "rejection_metrics": rejection_metrics_frame,
        "risk_coverage": risk_coverage_frame,
        "neighborhood": neighborhood_frame,
        "methods": methods,
        "thresholds": thresholds,
        "split_payload": split_payload,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_benchmark(
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
