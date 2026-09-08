"""Bounded, weight-only gallery reassessment; never modifies a parent run.

Trace follows Who Built This Model? Tracing LLM Lineage via Spectral Fingerprints
in Weight Space (2026-08-07), Section 4, equations 2--5:
https://arxiv.org/html/2608.07786v1#S4
For each of seven roles, take sqrt(saved norm), interpolate both depth sequences
to max(L_query, L_candidate), and average their Pearson correlations. The stored
norm is sum(lambda), not ||W||_F; eigenvalue filtering may make it approximate.

The comparator is the parent's frozen interpolated DKS-rank gallery retrieval,
not fixed-tail Hill from Spectral Signatures of Large Language Models (2026-07-03):
https://arxiv.org/html/2607.03377v1#S4.SS2
No fitting, DP, score-supervised selection, or deployment threshold is performed.
Labels are read only after prediction CSVs and their digest manifest are written.
Open-set AUROC measures gallery membership, not proof of independent ancestry.

Usage: python -B -m spectral_research.study.reassessment_lineage PARENT OUTPUT
OUTPUT must be new or empty. Norm sequences are cached inside OUTPUT/cache.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from .artifacts import validate_artifact_manifest
from .contracts import canonical_model_id
from .features import CANONICAL_ROLES, _layer_index, canonical_role


TRACE = "trace_sqrt_norm_pearson"
RANK = "parent_rank32_dks"
PREDICTION_COLUMNS = (
    "model_id", "predicted_root", "top3_roots", "root_ranking", "score",
    "margin", "candidate_count", "method", "prediction_uses_metadata",
)
SOURCES = {
    "trace": "https://arxiv.org/html/2608.07786v1#S4",
    "spectral_signatures": "https://arxiv.org/html/2607.03377v1#S4.SS2",
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, mode="x")


def _identifiers(frame: pd.DataFrame, column: str = "model_id") -> pd.DataFrame:
    frame = frame.copy()
    frame[column] = frame[column].map(canonical_model_id)
    if frame[column].isna().any() or frame[column].duplicated().any():
        raise ValueError(f"Missing or duplicate {column}")
    return frame


def _trace_sequence(frame: pd.DataFrame, model_id: str) -> tuple[np.ndarray, dict]:
    """Require actual contiguous layers, not a stretched sparse-depth profile."""
    required = {"model_id", "longname", "norm"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    identity = frame["model_id"].map(canonical_model_id)
    frame = frame.loc[identity.eq(model_id)].copy()
    if frame.empty:
        raise ValueError("Requested internal model_id not present")
    n_input = len(frame)
    frame["role"] = frame["longname"].map(canonical_role)
    frame["depth"] = frame["longname"].map(_layer_index)
    # Aggregate rows, embeddings, heads, fused roles, and experts have no
    # unambiguous canonical role plus depth, and are excluded before aggregation.
    frame = frame.loc[frame["role"].notna() & np.isfinite(frame["depth"])].copy()
    n_layer = len(frame)
    frame["norm"] = pd.to_numeric(frame["norm"], errors="coerce")
    if frame.empty or not np.isfinite(frame["norm"]).all() or frame["norm"].lt(0).any():
        raise ValueError("Missing, negative, or nonfinite canonical matrix norms")
    frame = frame.drop_duplicates(["role", "depth", "norm"])
    if frame.duplicated(["role", "depth"]).any():
        raise ValueError("Conflicting matrices for one canonical role/depth")
    pivot = frame.pivot(index="depth", columns="role", values="norm").reindex(
        columns=CANONICAL_ROLES
    ).sort_index()
    if len(pivot) < 2 or pivot.isna().any().any():
        raise ValueError("Incomplete seven-role depth sequence")
    if not np.array_equal(pivot.index.to_numpy(), np.arange(len(pivot))):
        raise ValueError("Noncontiguous or truncated layer indices")
    values = np.sqrt(pivot.to_numpy(dtype=np.float64))
    _unit_profiles(values[None, :, :], len(values))
    return values, {
        "input_model_rows": n_input,
        "excluded_noncanonical_or_aggregate_rows": n_input - n_layer,
        "identical_duplicate_rows": n_layer - len(frame),
        "n_depths": len(values),
    }


def _unit_profiles(batch: np.ndarray, length: int) -> np.ndarray:
    """Interpolate on normalized native depth, then center/normalize each role."""
    native = batch.shape[1]
    if native < 2 or length < native:
        raise ValueError("Trace requires length >= native depth >= 2")
    positions = np.linspace(0.0, native - 1, length)
    lower = positions.astype(int)
    upper = np.minimum(lower + 1, native - 1)
    fraction = (positions - lower)[None, :, None]
    values = batch[:, lower, :] * (1.0 - fraction) + batch[:, upper, :] * fraction
    scale = np.max(np.abs(values), axis=1, keepdims=True)
    values = values / np.maximum(scale, np.finfo(float).tiny)
    values -= values.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norm).all() or (norm <= np.finfo(float).eps * np.sqrt(length)).any():
        raise ValueError("Constant or numerically degenerate role: Pearson undefined")
    return values / norm


def _collect_sequences(index: pd.DataFrame, requested: list[str], output: Path) -> tuple[dict, pd.DataFrame]:
    paths = index.set_index("model_id")["path"].to_dict()

    def read(model_id: str) -> tuple[str, np.ndarray | None, dict]:
        path_text = paths.get(model_id, "")
        audit = {"model_id": model_id, "path": path_text, "available": False, "reason": ""}
        try:
            if not path_text:
                raise ValueError("No frozen source-index entry")
            path = Path(path_text)
            before = path.stat()
            data = path.read_bytes()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Source artifact changed during read")
            audit["source_sha256"] = hashlib.sha256(data).hexdigest()
            audit["source_size_bytes"] = len(data)
            frame = pd.read_csv(BytesIO(data), usecols=lambda c: c in {"model_id", "longname", "norm"})
            values, detail = _trace_sequence(frame, model_id)
            audit.update(detail, available=True)
            return model_id, values, audit
        except (OSError, ValueError, pd.errors.ParserError) as error:
            audit["reason"] = f"{type(error).__name__}: {error}"
            return model_id, None, audit

    sequences, audits = {}, []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for position, (model_id, values, audit) in enumerate(executor.map(read, requested), 1):
            audits.append(audit)
            if values is not None:
                sequences[model_id] = values
            if position % 1000 == 0:
                print(f"Trace sequences: {position}/{len(requested)} read, {len(sequences)} valid", flush=True)
    audit_frame = pd.DataFrame(audits)
    _write_csv(output / "norm_sequence_audit.csv", audit_frame)
    if not sequences:
        raise ValueError("No valid Trace sequences")
    cache = output / "cache"
    cache.mkdir()
    ids = sorted(sequences)
    offsets = np.cumsum([0, *[len(sequences[model_id]) for model_id in ids]])
    np.savez_compressed(
        cache / "norm_sequences.npz", model_ids=np.asarray(ids, dtype=str),
        offsets=offsets, values=np.concatenate([sequences[model_id] for model_id in ids]),
        roles=np.asarray(CANONICAL_ROLES),
    )
    return sequences, audit_frame


def _trace_scores(sequences: dict, query_ids: list[str], gallery_ids: list[str]) -> np.ndarray:
    scores = np.empty((len(query_ids), len(gallery_ids)), dtype=np.float64)
    query_lengths = np.asarray([len(sequences[value]) for value in query_ids])
    gallery_lengths = np.asarray([len(sequences[value]) for value in gallery_ids])
    with threadpool_limits(limits=1):
        for query_length in np.unique(query_lengths):
            qi = np.flatnonzero(query_lengths == query_length)
            queries = np.stack([sequences[query_ids[i]] for i in qi])
            for gallery_length in np.unique(gallery_lengths):
                gi = np.flatnonzero(gallery_lengths == gallery_length)
                roots = np.stack([sequences[gallery_ids[i]] for i in gi])
                length = int(max(query_length, gallery_length))
                left = _unit_profiles(queries, length).reshape(len(qi), -1)
                right = _unit_profiles(roots, length).reshape(len(gi), -1)
                scores[np.ix_(qi, gi)] = left @ right.T / len(CANONICAL_ROLES)
    return np.clip(scores, -1.0, 1.0)


def _rankings(scores: np.ndarray, query_ids: list[str], gallery_ids: list[str], method: str) -> pd.DataFrame:
    if len(gallery_ids) < 2 or scores.shape != (len(query_ids), len(gallery_ids)):
        raise ValueError("At least two candidates and a matching score matrix required")
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite gallery similarities")
    order = np.argsort(-scores, axis=1, kind="stable")
    names = np.asarray(gallery_ids, dtype=str)[order]
    ranked_scores = np.take_along_axis(scores, order, axis=1)
    return pd.DataFrame({
        "model_id": query_ids, "predicted_root": names[:, 0],
        "top3_roots": ["|".join(row[:3]) for row in names],
        "root_ranking": ["|".join(row) for row in names],
        "score": ranked_scores[:, 0], "margin": ranked_scores[:, 0] - ranked_scores[:, 1],
        "candidate_count": len(gallery_ids), "method": method,
        "prediction_uses_metadata": False,
    })


def _restricted_parent_predictions(metadata: dict, parent: pd.DataFrame, gallery: list[str], common: list[str]) -> pd.DataFrame:
    if gallery == common:
        return parent.copy()
    embeddings = {}
    wanted = set(parent["model_id"]) | set(common)
    for path in metadata["caches"]["strict_seven_role"]:
        with np.load(path, allow_pickle=False) as cache:
            for model_id, row in zip(cache["model_ids"].astype(str), cache["embeddings"], strict=True):
                if model_id in wanted:
                    if model_id in embeddings:
                        raise ValueError(f"Duplicate cached parent profile: {model_id}")
                    embeddings[model_id] = row
    missing = wanted - embeddings.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} parent profiles for matched-gallery ranking")
    ids = parent["model_id"].tolist()
    queries = np.stack([embeddings[value] for value in ids])
    roots = np.stack([embeddings[value] for value in common])
    with threadpool_limits(limits=1):
        scores = queries @ roots.T / len(CANONICAL_ROLES)
    return _rankings(scores, ids, common, RANK)


def _evaluate(predictions: pd.DataFrame, labels: pd.DataFrame, gallery: list[str], operator: list[str]) -> pd.DataFrame:
    frame = predictions.merge(labels, on="model_id", how="left", validate="one_to_one")
    frame["label_available"] = frame["actual_root"].notna()
    frame["true_root_in_operator_gallery"] = frame["actual_root"].isin(operator)
    frame["true_root_profiled"] = frame["actual_root"].isin(gallery)
    frame["root_status"] = np.select(
        [~frame["label_available"], frame["true_root_profiled"], frame["true_root_in_operator_gallery"]],
        ["unlabelled", "present", "supplied_unprofiled"], default="absent_operator",
    )
    frame["top1_correct"] = frame["predicted_root"].eq(frame["actual_root"])
    frame["top3_correct"] = [
        actual in top3.split("|") if pd.notna(actual) else False
        for actual, top3 in zip(frame["actual_root"], frame["top3_roots"], strict=True)
    ]
    return frame


def _summaries(frame: pd.DataFrame, cohort: str, method: str, candidates: int) -> tuple[dict, list, list]:
    known = frame.loc[frame["true_root_profiled"]]
    by_root = known.groupby("actual_root")[["top1_correct", "top3_correct"]].mean()
    metrics = {
        "cohort": cohort, "method": method, "candidate_count": candidates,
        "n_queries": len(frame), "n_labelled": int(frame["label_available"].sum()),
        "n_present": len(known), "n_present_roots": len(by_root),
        "n_absent_operator": int(frame["root_status"].eq("absent_operator").sum()),
        "n_supplied_unprofiled": int(frame["root_status"].eq("supplied_unprofiled").sum()),
        "micro_top1": known["top1_correct"].mean(), "micro_top3": known["top3_correct"].mean(),
        "root_macro_top1": by_root["top1_correct"].mean(),
        "root_macro_top3": by_root["top3_correct"].mean(),
    }
    groups, aucs = [], []
    for status, selected in frame.groupby("root_status", sort=True):
        groups.append({
            "cohort": cohort, "method": method, "root_status": status,
            "n_queries": len(selected), "n_roots": selected["actual_root"].nunique(),
            **{f"{signal}_{name}": selected[signal].quantile(q)
               for signal in ("score", "margin")
               for name, q in (("p05", 0.05), ("median", 0.5), ("p95", 0.95))},
        })
    labelled = frame.loc[frame["label_available"]]
    for contrast in ("profiled_root_vs_all_unavailable", "profiled_root_vs_absent_operator"):
        selected = labelled if contrast.endswith("all_unavailable") else labelled.loc[
            labelled["root_status"].isin(("present", "absent_operator"))
        ]
        y = selected["true_root_profiled"].to_numpy(dtype=bool)
        root_sizes = selected.groupby("actual_root")["model_id"].transform("size")
        for signal in ("score", "margin"):
            valid = y.size > 0 and np.unique(y).size == 2
            aucs.append({
                "cohort": cohort, "method": method, "contrast": contrast,
                "signal": signal, "higher_means_present": True,
                "n_positive": int(y.sum()), "n_negative": int((~y).sum()),
                "positive_roots": selected.loc[y, "actual_root"].nunique(),
                "negative_roots": selected.loc[~y, "actual_root"].nunique(),
                "auroc": roc_auc_score(y, selected[signal]) if valid else np.nan,
                "root_balanced_auroc": roc_auc_score(y, selected[signal], sample_weight=1.0 / root_sizes) if valid else np.nan,
                "threshold_selected": False,
            })
    return metrics, groups, aucs


def run_lineage_reassessment(parent_run: Path, output_dir: Path) -> dict:
    """Compare frozen rank retrieval with Trace, with matched gallery/cohort audits.

    At most 20,000 parent queries and 64 supplied candidates; four bounded I/O
    threads and length-grouped BLAS, no all-pairs zoo matrix or dynamic programming.
    The parent prediction population is fixed without inspecting lineage labels.
    """
    start = time.monotonic()
    parent_run, output_dir = Path(parent_run).resolve(), Path(output_dir).resolve()
    metadata_path = parent_run / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    study_run = Path(metadata["sources"]["study_run"]).resolve()
    provenance_path = study_run / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    protected = [parent_run, study_run, Path(metadata["sources"]["validation_run"]).resolve()]
    protected += [Path(provenance["sources"][name]["path"]).resolve() for name in ("development_stats", "external_stats")]
    if any(output_dir == path or output_dir.is_relative_to(path) for path in protected):
        raise ValueError("Output must be outside source directories and previous runs")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Output must be a new or empty directory")

    frozen_name = "lineage_gallery_prediction_freeze.csv"
    gallery_name = "lineage_supplied_gallery.csv"
    validate_artifact_manifest(parent_run, required_files=(frozen_name, gallery_name))
    parent = _identifiers(pd.read_csv(parent_run / frozen_name, usecols=PREDICTION_COLUMNS))
    parent["method"] = RANK
    if not parent["prediction_uses_metadata"].eq(False).all():
        raise ValueError("Parent predictions are not metadata-free")
    supplied = _identifiers(pd.read_csv(parent_run / gallery_name), "gallery_model_id")
    operator = supplied["gallery_model_id"].tolist()
    gallery = supplied.loc[supplied["profile_available"].eq(True), "gallery_model_id"].tolist()
    if len(parent) > 20_000 or len(operator) > 64 or len(gallery) < 2:
        raise ValueError("Population exceeds bounded protocol or has fewer than two parent candidates")
    if set(parent["model_id"]) & set(operator):
        raise ValueError("Parent queries include an operator gallery checkpoint")
    if any(set(value.split("|")) != set(gallery) for value in parent["root_ranking"]):
        raise ValueError("Frozen parent rankings do not match the supplied candidate audit")

    index_paths = [study_run / f"{name}_stats_index.csv" for name in ("development", "external")]
    index = _identifiers(pd.concat([pd.read_csv(path, usecols=["model_id", "path"]) for path in index_paths], ignore_index=True))
    inputs = {str(path): _sha256(path) for path in (metadata_path, provenance_path, *index_paths, parent_run / frozen_name, parent_run / gallery_name)}
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = sorted(set(parent["model_id"]) | set(operator))
    sequences, norm_audit = _collect_sequences(index, requested, output_dir)
    trace_gallery = [value for value in operator if value in sequences]
    common_gallery = [value for value in gallery if value in sequences]
    common_queries = [value for value in parent["model_id"] if value in sequences]
    if len(common_gallery) < 2 or not common_queries:
        raise ValueError("Insufficient common gallery/query coverage")
    audit_by_id = norm_audit.set_index("model_id")
    coverage = supplied.rename(columns={"profile_available": "rank_profile_available"}).copy()
    coverage["trace_profile_available"] = coverage["gallery_model_id"].isin(trace_gallery)
    coverage["common_gallery"] = coverage["gallery_model_id"].isin(common_gallery)
    coverage["trace_unavailable_reason"] = coverage["gallery_model_id"].map(audit_by_id["reason"])
    coverage["trace_n_depths"] = coverage["gallery_model_id"].map(audit_by_id["n_depths"])
    _write_csv(output_dir / "gallery_candidate_coverage.csv", coverage)

    print(f"Scoring {len(common_queries)} Trace queries against {len(trace_gallery)}/{len(operator)} supplied candidates", flush=True)
    scores = _trace_scores(sequences, common_queries, trace_gallery)
    trace = _rankings(scores, common_queries, trace_gallery, TRACE)
    common_columns = [trace_gallery.index(value) for value in common_gallery]
    trace_common = _rankings(scores[:, common_columns], common_queries, common_gallery, TRACE)
    if gallery != common_gallery:
        for path in metadata["caches"]["strict_seven_role"]:
            inputs[str(Path(path))] = _sha256(Path(path))
    rank_common = _restricted_parent_predictions(metadata, parent, gallery, common_gallery)
    rank_common = rank_common.loc[rank_common["model_id"].isin(common_queries)].copy()
    frozen = {
        "parent_rank_prediction_freeze.csv": parent,
        "trace_prediction_freeze.csv": trace,
        "common_gallery_rank_prediction_freeze.csv": rank_common,
        "common_gallery_trace_prediction_freeze.csv": trace_common,
    }
    for filename, frame in frozen.items():
        _write_csv(output_dir / filename, frame)
    freeze_manifest = {
        "parent_run": str(parent_run), "sources": SOURCES,
        "implementation_sha256": _sha256(Path(__file__)),
        "feature_helpers_sha256": _sha256(Path(__file__).with_name("features.py")),
        "input_sha256": inputs, "labels_loaded": False,
        "population_rule": "parent frozen non-gallery query IDs, no label selection",
        "trace_sequence_rule": "seven unfused roles, contiguous observed layers starting at zero; no missing-depth filling",
        "interpolation_rule": "pairwise maximum native depth, linear interpolation, per-role Pearson mean",
        "tie_rule": "stable supplied gallery order", "native_dks_comparison": "not run; no DP",
        "frozen_artifacts": {name: {"rows": len(frame), "sha256": _sha256(output_dir / name)} for name, frame in frozen.items()},
        "sequence_cache_sha256": _sha256(output_dir / "cache" / "norm_sequences.npz"),
        "sequence_audit_sha256": _sha256(output_dir / "norm_sequence_audit.csv"),
    }
    _write_json(output_dir / "prediction_freeze_manifest.json", freeze_manifest)

    # This is the first access to lineage truth. Preserve absent-gallery roots.
    label_name = "lineage_validation_assignments.csv"
    label_manifest = validate_artifact_manifest(parent_run, required_files=(label_name,))
    labels = _identifiers(pd.read_csv(parent_run / label_name, usecols=["model_id", "component_root_model"]))
    labels = labels.rename(columns={"component_root_model": "actual_root"})
    labels["actual_root"] = labels["actual_root"].map(canonical_model_id)
    rows, groups, aucs, evaluated = [], [], [], []
    for cohort, predictions, candidates in (
        ("own_coverage", parent, gallery), ("own_coverage", trace, trace_gallery),
        ("common_queries_and_gallery", rank_common, common_gallery),
        ("common_queries_and_gallery", trace_common, common_gallery),
    ):
        scored = _evaluate(predictions, labels, candidates, operator)
        method = str(predictions["method"].iloc[0])
        metric, group, auc = _summaries(scored, cohort, method, len(candidates))
        rows.append(metric)
        groups.extend(group)
        aucs.extend(auc)
        evaluated.append(scored.drop(columns="root_ranking").assign(cohort=cohort))
    tables = {
        "gallery_metrics.csv": pd.DataFrame(rows),
        "open_set_groups.csv": pd.DataFrame(groups),
        "open_set_metrics.csv": pd.DataFrame(aucs),
        "evaluated_predictions.csv": pd.concat(evaluated, ignore_index=True),
    }
    for filename, frame in tables.items():
        _write_csv(output_dir / filename, frame)
    if any(_sha256(Path(path)) != digest for path, digest in inputs.items()):
        raise RuntimeError("Parent or source-index input changed during reassessment")
    summary = {
        "parent_run": str(parent_run), "output_dir": str(output_dir), "status": "complete",
        "elapsed_seconds": time.monotonic() - start, "sources": SOURCES,
        "operator_candidates": len(operator), "parent_candidates": len(gallery),
        "trace_candidates": len(trace_gallery), "common_candidates": len(common_gallery),
        "parent_queries": len(parent), "common_queries": len(common_queries),
        "label_sha256": label_manifest["artifacts"][label_name]["sha256"],
        "gallery_metrics": json.loads(tables["gallery_metrics.csv"].to_json(orient="records")),
        "open_set_metrics": json.loads(tables["open_set_metrics.csv"].to_json(orient="records")),
        "limitations": [
            "Descriptive reuse of parent labels, not a fresh external validation cohort.",
            "Gallery membership and graph-component roots are not direct parentage or proof of independence.",
            "No threshold chosen; higher score/margin means present, fixed before label access.",
            "Stored norm can omit filtered eigenvalues; checkpoint revision SHA is not available in stats.",
            "Seven-role, contiguous observed-depth restriction; entirely missing terminal blocks cannot be detected from stats alone.",
            "Supplied but unprofiled roots are separate from roots absent from the operator gallery.",
            "No native DKS/DP comparison or duplicated performance fitting.",
        ],
    }
    _write_json(output_dir / "summary.json", summary)
    print(f"Completed reassessment in {summary['elapsed_seconds']:.1f}s: {output_dir}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent_run", type=Path)
    parser.add_argument("output_dir", type=Path)
    arguments = parser.parse_args()
    result = run_lineage_reassessment(arguments.parent_run, arguments.output_dir)
    print(json.dumps({key: result[key] for key in ("output_dir", "gallery_metrics", "open_set_metrics")}, indent=2))
