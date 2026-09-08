"""Bounded, post-prediction audits of saved HTSR reassessment runs."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .artifacts import validate_artifact_manifest, write_artifact_manifest
from .contracts import canonical_model_id

__all__ = ["run_reassessment_audit"]

_KEYS = ["population", "generation", "split"]
_RAW_SAMPLE_SIZE = 32


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _text(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip()


def _sha(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    result = pd.Series("", index=frame.index, dtype=object)
    for column in columns:
        if column in frame:
            values = _text(frame[column]).str.casefold()
            values = values.mask(values.isin(("nan", "none", "null")), "")
            result = result.mask(result.eq(""), values)
    return result


def _models(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["model_id"] = frame["model_id"].map(canonical_model_id)
    if frame["model_id"].isna().any():
        raise ValueError("Audit input contains a missing model ID")
    return frame


def _metrics(frame: pd.DataFrame) -> dict:
    actual = frame["actual"].to_numpy(float)
    prediction = frame["prediction"].to_numpy(float)
    finite = np.isfinite(actual) & np.isfinite(prediction)
    a, p = actual[finite], prediction[finite]
    errors = np.abs(a - p)
    rho = (
        float(spearmanr(a, p).statistic)
        if len(a) >= 2 and np.ptp(a) > 0 and np.ptp(p) > 0
        else float("nan")
    )
    return {
        "n_eligible": len(frame),
        "n_models": len(a),
        "n_nonfinite_actual": int((~np.isfinite(actual)).sum()),
        "n_nonfinite_prediction": int((~np.isfinite(prediction)).sum()),
        "coverage": float(finite.mean()) if len(frame) else float("nan"),
        "mae": float(errors.mean()) if len(a) else float("nan"),
        "median_error": float(np.median(errors)) if len(a) else float("nan"),
        "rmse": float(np.sqrt(np.mean((a - p) ** 2))) if len(a) else float("nan"),
        "spearman": rho,
        "n_actual_outside_0_100": int(((a < 0) | (a > 100)).sum()),
        "n_prediction_outside_0_100": int(((p < 0) | (p > 100)).sum()),
        "prediction_min": float(p.min()) if len(p) else float("nan"),
        "prediction_max": float(p.max()) if len(p) else float("nan"),
        "n_train_min": int(frame["n_train"].min()),
        "n_train_max": int(frame["n_train"].max()),
    }


def _feature_hashes(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    matrix = frame[columns].to_numpy(dtype="<f8", copy=True)
    matrix[np.isnan(matrix)] = np.nan
    matrix[matrix == 0] = 0.0
    return pd.Series(
        [hashlib.sha256(row.tobytes()).hexdigest() for row in matrix],
        index=frame["model_id"],
    )


def _shared_pairs(ids: list[str], cache: dict) -> list[tuple[str, str]]:
    masks: dict[tuple, list[str]] = defaultdict(list)
    for model_id in ids:
        masks[tuple(cache[model_id][1])].append(model_id)
    pairs = set()
    for (left_mask, left), (right_mask, right) in itertools.combinations_with_replacement(
        masks.items(), 2
    ):
        shared = np.asarray(left_mask) & np.asarray(right_mask)
        if shared.sum() < 4:
            continue
        a, b = defaultdict(list), defaultdict(list)
        for model_id in left:
            a[cache[model_id][0][shared].tobytes()].append(model_id)
        for model_id in right:
            b[cache[model_id][0][shared].tobytes()].append(model_id)
        for key in a.keys() & b.keys():
            for i, j in itertools.product(a[key], b[key]):
                if i != j:
                    pairs.add(tuple(sorted((i, j))))
    return sorted(pairs)


def run_reassessment_audit(parent_run: str | Path, reassessment_run: str | Path) -> dict:
    """Write an isolated audit directory; never fit models or change source folds.

    Required manifests and consumed target-source digests fail closed. Live raw
    spectra are rehashed for a deterministic sample only; all other recorded raw
    digests remain explicitly unverified against the live files.
    """
    parent_run, reassessment_run = Path(parent_run).resolve(), Path(reassessment_run).resolve()
    digests: list[dict] = []
    manifests: list[dict] = []

    def record(path: Path, kind: str, expected: str = "", *, required: bool = False) -> str:
        observed = _digest(path) if path.is_file() else ""
        status = (
            "missing" if not observed else
            "verified" if expected and observed == expected else
            "digest_mismatch" if expected else "observed_without_recorded_digest"
        )
        digests.append({"kind": kind, "path": str(path), "expected_sha256": expected,
                        "observed_sha256": observed, "status": status})
        if required and status != "verified":
            raise ValueError(f"Required source digest failed: {path}: {status}")
        return observed

    def validate(root: Path, names: tuple[str, ...]) -> dict:
        manifest = validate_artifact_manifest(
            root, required_files=names, expected_run_id=root.name
        )
        manifests.append({"path": str(root / "artifact_manifest.json"),
                          "sha256": _digest(root / "artifact_manifest.json"),
                          "validated_files": list(names), "status": "verified_required_files"})
        for name in names:
            value = manifest["artifacts"][name]["sha256"]
            digests.append({"kind": "manifest_validated_input", "path": str(root / name),
                            "expected_sha256": value, "observed_sha256": value,
                            "status": "verified"})
        return manifest

    validate(parent_run, ("metadata.json", "performance_frozen_splits.csv"))
    validate(reassessment_run, ("metadata.json", "protocol.json", "performance_predictions.csv",
                               "performance_metrics.csv", "splits.csv", "all_matrix_features.csv"))
    parent = json.loads((parent_run / "metadata.json").read_text())
    protocol = json.loads((reassessment_run / "protocol.json").read_text())
    if Path(protocol["parent_run"]).resolve() != parent_run:
        raise ValueError("Reassessment protocol refers to a different parent run")
    record(parent_run / "artifact_manifest.json", "parent_manifest_reference",
           protocol["sources"]["parent_manifest"], required=True)
    study = Path(parent["sources"]["study_run"])
    validation = Path(parent["sources"]["validation_run"])
    validate(study, ("provenance.json", "feature_table.csv", "external_feature_table.csv"))
    validate(validation, ("metadata.json", "performance_modern_targets.csv",
                          "performance_legacy_targets.csv", "population_ledger.csv"))
    source_meta = json.loads((validation / "metadata.json").read_text())
    source_parts = []
    for name in ("phase2", "ablation"):
        info = source_meta["sources"][name]
        path = Path(info["path"])
        record(path, "target_source_csv", info["sha256"], required=True)
        frame = _models(pd.read_csv(path, low_memory=False))
        frame["source_population"] = name
        frame["source_recorded_sha"] = _sha(frame, ("model_sha", "Model sha"))
        for column in ("revision_norm", "source_model", "base_model_relation", "loader_scenario"):
            frame[column] = _text(frame.get(column, pd.Series("", index=frame.index)))
        source_parts.append(frame[["model_id", "source_population", "source_recorded_sha",
                                   "revision_norm", "source_model", "base_model_relation", "loader_scenario"]])
    sources = pd.concat(source_parts, ignore_index=True)
    if sources["model_id"].duplicated().any():
        raise ValueError("Canonical source model IDs are not unique")
    repository = Path(__file__).resolve().parents[2]
    references = protocol["sources"]
    implementations = references.get("implementations", {})
    if "script" in references:
        implementations = {**implementations, "scripts/htsr_reassessment.py": references["script"]}
    for name, expected in implementations.items():
        kind = ("historical_script_reference_not_consumed" if name == "scripts/htsr_reassessment.py"
                else "historical_implementation_reference_not_consumed")
        record(repository / name, kind, expected)
    for name, expected in references.get("inputs", {}).items():
        if name != "raw_matrix_artifacts":
            record(Path(name), "protocol_input_file", expected, required=True)
    if "reused_aggregate_run" in protocol:
        reused = Path(protocol["reused_aggregate_run"]).resolve()
        record(reused / "artifact_manifest.json", "reused_manifest_reference",
               references["reused_manifest"], required=True)
        validate(reused, ("metadata.json", "all_matrix_features.csv", "performance_predictions.csv",
                          "splits.csv", "duplicate_audit.csv"))

    predictions = _models(pd.read_csv(reassessment_run / "performance_predictions.csv", low_memory=False))
    splits = _models(pd.read_csv(reassessment_run / "splits.csv", dtype={"group": str}))
    raw = _models(pd.read_csv(reassessment_run / "all_matrix_features.csv", low_memory=False))
    expected_inventory = references.get("inputs", {}).get("raw_matrix_artifacts")
    if expected_inventory:
        observed_inventory = hashlib.sha256(
            raw[["model_id", "artifact_sha256"]].to_csv(index=False).encode()
        ).hexdigest()
        digests.append({"kind": "protocol_raw_digest_inventory", "path": str(reassessment_run / "all_matrix_features.csv"),
                        "expected_sha256": expected_inventory, "observed_sha256": observed_inventory,
                        "status": "verified" if observed_inventory == expected_inventory else "digest_mismatch"})
        if observed_inventory != expected_inventory:
            raise ValueError("Raw-file digest inventory differs from protocol inputs")
    split_keys = [*_KEYS, "model_id"]
    if predictions.duplicated([*split_keys, "method"]).any() or splits.duplicated(split_keys).any():
        raise ValueError("Duplicate prediction or split identifiers")
    aligned = predictions.merge(splits, on=split_keys, how="outer", suffixes=("", "_split"),
                                validate="many_to_one", indicator=True)
    if not aligned["_merge"].eq("both").all():
        raise ValueError("Predictions and split populations do not match")
    if not aligned["fold"].eq(aligned["fold_split"]).all():
        raise ValueError("Prediction fold differs from frozen split")
    if not aligned["group"].astype(str).eq(aligned["group_split"].astype(str)).all():
        raise ValueError("Prediction group differs from frozen split")
    if predictions.groupby(["generation", "model_id"])["actual"].nunique(dropna=False).gt(1).any():
        raise ValueError("Actual scores change between methods or splits")

    identity_parts = []
    for generation, selected in predictions.groupby("generation", sort=True):
        if generation not in ("modern", "legacy"):
            raise ValueError(f"Unknown benchmark generation: {generation}")
        targets = _models(pd.read_csv(validation / f"performance_{generation}_targets.csv"))
        targets["target_model_sha"] = _sha(targets, ("target_model_sha",))
        columns = ["model_id", "target_model_sha", "source_population", generation + "_average"]
        columns += [c for c in ("target_resolution", "target_eval_name") if c in targets]
        actual = selected[["model_id", "actual"]].drop_duplicates()
        joined = actual.merge(targets[columns], on="model_id", how="left", validate="one_to_one")
        if not np.allclose(joined["actual"], joined[generation + "_average"], rtol=0, atol=1e-10):
            raise ValueError("Saved actual values do not match the source targets")
        joined = joined.merge(sources, on=["model_id", "source_population"],
                              how="left", validate="one_to_one", indicator=True)
        if not joined["_merge"].eq("both").all():
            raise ValueError("A scored target cannot be joined to its source population")
        joined.drop(columns=["_merge", generation + "_average"], inplace=True)
        joined["generation"] = generation
        known = joined["target_model_sha"].ne("") & joined["source_recorded_sha"].ne("")
        joined["source_target_sha_status"] = np.where(
            known, np.where(joined["target_model_sha"].eq(joined["source_recorded_sha"]),
                            "recorded_sha_agrees", "recorded_sha_disagrees"), "unavailable")
        joined["tensor_identity_verified"] = False
        joined["effective_checkpoint_revision_recorded"] = False
        for population in selected["population"].unique():
            joined["in_" + population] = joined["model_id"].isin(
                selected.loc[selected["population"].eq(population), "model_id"])
        identity_parts.append(joined)
    identity = pd.concat(identity_parts, ignore_index=True)
    sha_counts = identity.loc[identity["target_model_sha"].ne("")].groupby("model_id").agg(
        generations=("generation", "nunique"), sha_count=("target_model_sha", "nunique"))
    conflicts = sha_counts.index[(sha_counts.generations > 1) & (sha_counts.sha_count > 1)]
    identity["different_recorded_sha_across_generations"] = identity["model_id"].isin(conflicts)
    identity_summary = []
    for population, members in predictions.groupby("population", sort=True):
        membership = members[["generation", "model_id"]].drop_duplicates()
        selected = membership.merge(identity, on=["generation", "model_id"], validate="one_to_one")
        cross = selected.loc[selected["target_model_sha"].ne("")].groupby("model_id").agg(
            generations=("generation", "nunique"), sha_count=("target_model_sha", "nunique"))
        for (generation, source), frame in selected.groupby(["generation", "source_population"]):
            identity_summary.append({
                "population": population, "generation": generation, "source_population": source,
                "n_models": len(frame),
                "n_source_target_sha_comparable": int(frame["source_target_sha_status"].ne("unavailable").sum()),
                "n_source_target_sha_disagreements": int(frame["source_target_sha_status"].eq("recorded_sha_disagrees").sum()),
                "n_target_sha_missing": int(frame["target_model_sha"].eq("").sum()),
                "n_explicit_source_revisions": int(frame["revision_norm"].ne("").sum()),
                "n_tensor_identities_verified": 0,
                "population_cross_generation_sha_comparable_models": int(cross.generations.gt(1).sum()),
                "population_cross_generation_sha_conflict_models": int(((cross.generations > 1) & (cross.sha_count > 1)).sum()),
            })

    metric_rows, direction_rows = [], []
    for key, frame in predictions.groupby([*_KEYS, "method"], sort=True):
        metric_rows.append({**dict(zip([*_KEYS, "method"], key)), **_metrics(frame)})
        if key[2] == "collection_holdout":
            for (fold, source), test in frame.groupby(["fold", "source_population"]):
                train = frame.loc[frame["fold"].ne(fold)]
                direction_rows.append({**dict(zip([*_KEYS, "method"], key)), "fold": int(fold),
                    "train_collection": "|".join(sorted(train["source_population"].unique())),
                    "test_collection": source, **_metrics(test)})
    metrics = pd.DataFrame(metric_rows)
    published = pd.read_csv(reassessment_run / "performance_metrics.csv")
    comparison = metrics.merge(published, on=[*_KEYS, "method"], suffixes=("", "_published"),
                               validate="one_to_one", how="outer", indicator=True)
    if not comparison["_merge"].eq("both").all():
        raise ValueError("Published metric identities do not match predictions")
    deltas = []
    for column in ("mae", "median_error", "rmse", "spearman", "n_models", "n_eligible", "coverage"):
        comparison[column + "_difference"] = comparison[column] - comparison[column + "_published"]
        if not np.allclose(comparison[column], comparison[column + "_published"],
                           rtol=0, atol=1e-5, equal_nan=True):
            raise ValueError(f"Published {column} does not match saved predictions")
        values = comparison[column + "_difference"].abs().dropna()
        if len(values):
            deltas.append(float(values.max()))
    comparison.drop(columns="_merge", inplace=True)

    control_rows = []
    for candidate, control in (("role_spectral_trees", "nonalpha_spectral_trees"),
                               ("role_spectral_ridge", "nonalpha_spectral_ridge"),
                               ("role_spectral_trees", "matrix_shape_trees")):
        for key, baseline in predictions.loc[predictions["method"].eq(control)].groupby(_KEYS, sort=True):
            selected = predictions.loc[predictions["method"].eq(candidate)]
            for column, value in zip(_KEYS, key):
                selected = selected.loc[selected[column].eq(value)]
            paired = baseline.merge(selected, on="model_id", how="outer", validate="one_to_one",
                                    suffixes=("_control", "_candidate"), indicator=True)
            if not paired["_merge"].eq("both").all() or not paired["fold_control"].eq(paired["fold_candidate"]).all():
                raise ValueError(f"Unmatched control comparison: {key}: {candidate} vs {control}")
            if not np.allclose(paired["actual_control"], paired["actual_candidate"], rtol=0, atol=1e-10):
                raise ValueError("Control comparison uses different actual scores")
            finite = np.isfinite(paired[["actual_control", "prediction_control", "prediction_candidate"]]).all(axis=1)
            scored = paired.loc[finite]
            control_mae = float((scored["actual_control"] - scored["prediction_control"]).abs().mean())
            candidate_mae = float((scored["actual_control"] - scored["prediction_candidate"]).abs().mean())
            control_rows.append({**dict(zip(_KEYS, key)), "candidate": candidate, "control": control,
                "n_eligible": len(paired), "n_paired_finite": len(scored),
                "candidate_mae": candidate_mae, "control_mae": control_mae,
                "mae_difference_candidate_minus_control": candidate_mae - control_mae,
                "evidence_role": "exploratory_paired_prediction_comparison_not_significance_test"})

    profiles = _models(pd.read_csv(parent_run / "performance_frozen_splits.csv"))
    cache = {}
    for path_text in parent["caches"]["coverage_aware"]:
        path = Path(path_text)
        record(path, "profile_cache_verified_below_against_parent_profile_hashes")
        with np.load(path, allow_pickle=False) as part:
            for model_id, values, mask in zip(part["model_ids"], part["role_embeddings"], part["role_mask"]):
                model_id = canonical_model_id(model_id)
                if model_id in cache:
                    raise ValueError("Profile caches overlap on model IDs")
                cache[model_id] = (values.astype(np.float32), mask.astype(bool))
    for row in profiles.itertuples():
        values, mask = cache[row.model_id]
        if hashlib.sha256(mask.tobytes() + values.tobytes()).hexdigest() != row.profile_hash:
            raise ValueError(f"Cache no longer matches frozen profile: {row.model_id}")
    ledger = _models(pd.read_csv(validation / "population_ledger.csv", low_memory=False))
    feature_keys = {
        "parent_profile_hash": profiles.set_index("model_id")["profile_hash"],
        "recorded_ancestry_component": ledger.set_index("model_id")["ancestry_component"],
        "all_alpha_input_hash": _feature_hashes(raw, [c for c in raw if c.startswith("all__alpha__")]),
        "all_spectral_input_hash": _feature_hashes(raw, [c for c in raw if c.startswith("all__")]),
    }
    overlap_rows, overlap_summary = [], []
    pairs_by_generation = {}
    for generation, group in splits.loc[splits["population"].eq("matched_role_population")].groupby("generation"):
        pairs_by_generation[generation] = _shared_pairs(sorted(group["model_id"].unique()), cache)
    for key, frame in splits.groupby(_KEYS, sort=True):
        label = dict(zip(_KEYS, key))
        target_sha = identity.loc[identity["generation"].eq(key[1])].set_index("model_id")["target_model_sha"]
        lookups = {"declared_split_group": frame.set_index("model_id")["group"],
                   "target_sha_postprediction": target_sha, **feature_keys}
        for kind, lookup in lookups.items():
            working = frame.assign(identity_value=frame["model_id"].map(lookup))
            working["identity_value"] = _text(working["identity_value"])
            working = working.loc[working["identity_value"].ne("")]
            counts = working.groupby("identity_value")["fold"].nunique()
            bad = working.loc[working["identity_value"].isin(counts.index[counts > 1])]
            for row in bad.itertuples():
                overlap_rows.append({**label, "identity_kind": kind, "identity_value": row.identity_value,
                                     "model_id": row.model_id, "other_model_id": "", "fold": row.fold})
            overlap_summary.append({**label, "identity_kind": kind, "n_auditable_models": len(working),
                "n_crossing_groups": int((counts > 1).sum()), "n_crossing_pairs": np.nan,
                "n_models_with_overlap": len(bad)})
        if key[0] == "matched_role_population":
            folds = frame.set_index("model_id")["fold"].to_dict()
            pairs = [(a, b) for a, b in pairs_by_generation[key[1]] if folds[a] != folds[b]]
            for a, b in pairs:
                overlap_rows.append({**label, "identity_kind": "byte_identical_shared_roles_ge4",
                                     "identity_value": "", "model_id": a, "other_model_id": b,
                                     "fold": folds[a]})
            overlap_summary.append({**label, "identity_kind": "byte_identical_shared_roles_ge4",
                "n_auditable_models": len(frame), "n_crossing_groups": np.nan,
                "n_crossing_pairs": len(pairs),
                "n_models_with_overlap": len({model_id for pair in pairs for model_id in pair})})

    sampled = set(raw.sort_values("model_id").iloc[
        np.linspace(0, len(raw) - 1, min(_RAW_SAMPLE_SIZE, len(raw)), dtype=int)
    ]["model_id"])
    for row in raw.itertuples():
        if row.model_id in sampled:
            record(Path(row.artifact_path), "raw_stats_sample", row.artifact_sha256)
        else:
            digests.append({"kind": "raw_stats_inventory", "path": row.artifact_path,
                "expected_sha256": row.artifact_sha256, "observed_sha256": "",
                "status": "recorded_not_rehashed"})
    audit_digest = _digest(Path(__file__))
    key = hashlib.sha256((audit_digest + _digest(parent_run / "artifact_manifest.json") +
                          _digest(reassessment_run / "artifact_manifest.json")).encode()).hexdigest()[:20]
    out = reassessment_run / "audit" / ("audit_" + key)
    summary = {
        "run_id": "audit_" + key, "parent_run": str(parent_run),
        "reassessment_run": str(reassessment_run), "output_dir": str(out),
        "audit_implementation_sha256": audit_digest, "status": "exploratory_postprediction_audit",
        "metadata_used_for_prediction_or_folds": False,
        "identity_interpretation": "Recorded SHA disagreement is not proof of different tensors; effective checkpoints are unverified.",
        "metric_interpretation": "Empirical held-out errors, not deployment calibration; directions are reported separately.",
        "measurement_interpretation": "Norm fields summarize retained eigenvalues; saved xmin is not the fitted tail cutoff.",
        "raw_digest_scope": "Complete recorded inventory; deterministic live-file sample only.",
        "raw_files_recorded": len(raw), "raw_files_rehashed": len(sampled),
        "consumed_input_digest_validation": "passed",
        "historical_script_drift": any(row["kind"] == "historical_script_reference_not_consumed" and
                                        row["status"] != "verified" for row in digests),
        "historical_implementation_drift": any(row["kind"] == "historical_implementation_reference_not_consumed" and
                                                row["status"] != "verified" for row in digests),
        "raw_sample_digest_mismatches": sum(row["kind"] == "raw_stats_sample" and
                                             row["status"] == "digest_mismatch" for row in digests),
        "raw_sample_missing_files": sum(row["kind"] == "raw_stats_sample" and
                                         row["status"] == "missing" for row in digests),
        "cross_generation_recorded_sha_conflict_models": len(conflicts),
        "source_target_recorded_sha_conflict_rows": int(identity["source_target_sha_status"].eq("recorded_sha_disagrees").sum()),
        "max_published_metric_difference": max(deltas, default=0.0),
        "validated_manifests": manifests,
    }
    frames = {
        "identity_join_audit.csv": identity,
        "identity_summary.csv": pd.DataFrame(identity_summary),
        "collection_direction_metrics.csv": pd.DataFrame(direction_rows),
        "prediction_metrics_audit.csv": comparison,
        "control_comparisons.csv": pd.DataFrame(control_rows, columns=[*_KEYS, "candidate", "control", "n_eligible",
            "n_paired_finite", "candidate_mae", "control_mae", "mae_difference_candidate_minus_control", "evidence_role"]),
        "overlap_audit.csv": pd.DataFrame(overlap_summary),
        "overlap_rows.csv": pd.DataFrame(overlap_rows, columns=[*_KEYS, "identity_kind", "identity_value", "model_id", "other_model_id", "fold"]),
        "source_digest_audit.csv": pd.DataFrame(digests),
    }
    gallery_file = reassessment_run / "lineage_evaluated_predictions.csv"
    if gallery_file.is_file():
        validate(reassessment_run, (gallery_file.name,))
        gallery = pd.read_csv(gallery_file)
        common = gallery.loc[gallery["cohort"].eq("common_queries_and_gallery") & gallery["true_root_profiled"]].copy()
        margins = common.pivot(index="model_id", columns="method", values="margin")
        exact_tie_ids = set(margins.index[margins.eq(0).any(axis=1)])
        tie_rows = []
        for cohort, keep in (
            ("all_common_queries", pd.Series(True, index=common.index)),
            ("either_method_has_exact_tie", common.model_id.isin(exact_tie_ids)),
            ("neither_method_has_exact_tie", ~common.model_id.isin(exact_tie_ids)),
        ):
            for method, group in common.loc[keep].groupby("method"):
                tie_rows.append({"cohort": cohort, "method": method, "n_models": len(group),
                    "n_roots": group.actual_root.nunique(), "correct_top1": int(group.top1_correct.sum()),
                    "micro_top1": float(group.top1_correct.mean()),
                    "root_macro_top1": float(group.groupby("actual_root").top1_correct.mean().mean())})
        frames["gallery_tie_sensitivity.csv"] = pd.DataFrame(tie_rows)
        summary["tie_interpretation"] = "Post-hoc exact-tie exclusion changes case mix and removes one root; it is not a causal test of tie-breaking or a deployment rejection rule."
    out.mkdir(parents=True, exist_ok=True)
    if control_rows:
        summary["alpha_specific_claim"] = "Alpha-specific improvement is not established; report the paired non-alpha controls."
    for name, frame in frames.items():
        frame.to_csv(out / name, index=False)
    summary["artifacts"] = {name: str(out / name) for name in frames}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_artifact_manifest(out, run_id=summary["run_id"])
    validate_artifact_manifest(out, required_files=(*frames, "summary.json"), expected_run_id=summary["run_id"])
    return summary
