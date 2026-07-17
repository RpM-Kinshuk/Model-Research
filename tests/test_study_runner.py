from __future__ import annotations

import errno
import json
from pathlib import Path

import pandas as pd
import pytest

from spectral_research.study.contracts import RunSettings, StudyInputs, path_digest
from spectral_research.study.runner import (
    REQUIRED_ARTIFACTS,
    _apply_external_evidence_role,
    _atomic_directory,
    _evaluate_performance,
    _feature_extractor_digest,
    _label_independent_cluster_count,
    _performance_effect_interpretation,
    _study_implementation_digest,
    run_study,
)


TASKS = ("IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO")
ROLES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def _write_stats(path: Path, model_id: str, family: int, member: int) -> None:
    rows = []
    for layer in range(2):
        for role_index, role in enumerate(ROLES):
            scale = 1.0 + 0.2 * family + 0.03 * member + 0.01 * role_index
            rows.append(
                {
                    "model_id": model_id,
                    "longname": f"model.layers.{layer}.{role}",
                    "alpha": 1.8 + 0.35 * family + 0.04 * member + 0.01 * layer,
                    "D": 0.02 + 0.001 * role_index,
                    "M": 8 + 2 * family,
                    "N": 8 + 2 * family,
                    "norm": 256.0 * scale,
                    "spectral_norm": 64.0 * scale,
                    "entropy": 0.5 + 0.05 * family,
                    "matrix_rank": 8 + 2 * family,
                    "log_alpha_norm": 1.0 + 0.1 * family,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_synthetic_inputs(tmp_path: Path) -> StudyInputs:
    stats = tmp_path / "stats"
    stats.mkdir()
    metadata_rows = []
    label_rows = []
    target_rows = []
    graph_nodes = []
    graph_edges = []

    for family in range(3):
        root = f"org-{family}/root"
        for member in range(4):
            model_id = root if member == 0 else f"org-{family}/child-{member}"
            graph_nodes.append({"id": model_id})
            metadata_rows.append(
                {
                    "model_id": model_id,
                    "source_model": root if member else "root",
                    "base_model_relation": "root" if member == 0 else "derived",
                }
            )
            label_rows.append(
                {
                    "model_id": model_id.replace("/", "--", 1),
                    "audited_base_family": root,
                    "label_status": "confirmed",
                }
            )
            if member:
                graph_edges.append({"source": root, "target": model_id})
            score = 35.0 + 12.0 * family + member
            target_rows.append(
                {
                    "model_id": model_id,
                    "model_sha": f"sha-{family}-{member}",
                    "eval_name": model_id,
                    "architecture": f"arch-{family % 2}",
                    **{task: score for task in TASKS},
                }
            )
            _write_stats(
                stats / f"{model_id.replace('/', '--')}.csv",
                model_id,
                family,
                member,
            )

    metadata = tmp_path / "metadata.csv"
    labels = tmp_path / "labels.csv"
    graph = tmp_path / "graph.json"
    targets = tmp_path / "targets.csv"
    pd.DataFrame(metadata_rows).to_csv(metadata, index=False)
    pd.DataFrame(label_rows).to_csv(labels, index=False)
    graph.write_text(
        json.dumps({"nodes": graph_nodes, "edges": graph_edges}),
        encoding="utf-8",
    )
    pd.DataFrame(target_rows).to_csv(targets, index=False)
    return StudyInputs(
        metadata=metadata,
        labels=labels,
        graph=graph,
        development_stats=stats,
        development_targets=targets,
        cache_root=tmp_path / "cache",
        output_root=tmp_path / "runs",
    )


def _settings(schema_version: str = "synthetic-study-v1") -> RunSettings:
    return RunSettings(
        schema_version=schema_version,
        random_seed=11,
        bootstrap_replicates=8,
        outer_folds=3,
        inner_folds=2,
        ridge_alphas=(0.1, 1.0),
    )


def test_cluster_count_rule_depends_only_on_unlabeled_model_count() -> None:
    assert _label_independent_cluster_count(3, rule="sqrt_half") == 2
    assert _label_independent_cluster_count(100, rule="sqrt_half") == 8

    with pytest.raises(ValueError, match="at least three"):
        _label_independent_cluster_count(2, rule="sqrt_half")
    with pytest.raises(ValueError, match="unknown cluster count rule"):
        _label_independent_cluster_count(100, rule="family_count")


def test_reused_external_evidence_is_always_descriptive() -> None:
    result = _apply_external_evidence_role(
        {
            "status": "available",
            "headline_slice": "component_disjoint",
            "component_disjoint_models": 60,
            "component_disjoint_components": 48,
        },
        role="reused_descriptive",
    )

    assert result["status"] == "descriptive_reuse"
    assert result["component_contract_status"] == "available"
    assert result["reason"] == "previous_method_development_reuse"
    assert result["evidence_role"] == "reused_descriptive"


def test_performance_effect_interpretation_respects_primary_decision() -> None:
    assert _performance_effect_interpretation(-0.2, {"passed": True}) == (
        "Adding H satisfied the primary improvement rule."
    )
    assert _performance_effect_interpretation(-0.2, {"passed": False}) == (
        "The point estimate favored H, but the primary evidence was insufficient "
        "to claim improvement."
    )
    assert _performance_effect_interpretation(0.2, {"passed": False}) == (
        "The point estimate worsened with H."
    )


def test_implementation_digest_is_stable_content_identity() -> None:
    digest = _study_implementation_digest()

    assert len(digest) == 64
    int(digest, 16)


def test_feature_extractor_digest_is_stable_content_identity() -> None:
    digest = _feature_extractor_digest()

    assert len(digest) == 64
    int(digest, 16)


def test_implementation_digest_changes_run_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_synthetic_inputs(tmp_path)
    monkeypatch.setattr(
        "spectral_research.study.runner._study_implementation_digest",
        lambda: "a" * 64,
    )
    first = run_study(inputs, _settings(), dry_run=True)
    monkeypatch.setattr(
        "spectral_research.study.runner._study_implementation_digest",
        lambda: "b" * 64,
    )
    second = run_study(inputs, _settings(), dry_run=True)

    assert first.run_id != second.run_id


def test_runner_requires_every_source_path_to_exist(tmp_path: Path) -> None:
    inputs = _write_synthetic_inputs(tmp_path)
    inputs.labels.unlink()

    with pytest.raises(FileNotFoundError, match="labels"):
        run_study(inputs, _settings(), dry_run=True)


def test_dry_run_validates_contracts_without_extracting_or_fitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_synthetic_inputs(tmp_path)

    def fail(*_args, **_kwargs):
        raise AssertionError("dry-run must not extract features or fit estimators")

    monkeypatch.setattr("spectral_research.study.runner.extract_study_tables", fail)
    monkeypatch.setattr("spectral_research.study.runner.evaluate_grouped_ridge", fail)

    result = run_study(inputs, _settings(), dry_run=True)

    assert result.dry_run is True
    assert result.run_dir is None
    assert result.summary["manifest_rows"] == 12
    assert result.summary["development_target_rows"] == 12
    assert not inputs.output_root.exists()


def test_runner_population_is_not_capped_by_metadata_lookup_rows(tmp_path: Path) -> None:
    inputs = _write_synthetic_inputs(tmp_path)
    metadata = pd.read_csv(inputs.metadata)
    metadata.iloc[:-1].to_csv(inputs.metadata, index=False)

    result = run_study(inputs, _settings(), dry_run=True)

    assert result.summary["metadata_rows"] == 11
    assert result.summary["manifest_rows"] == 12


def test_runner_writes_atomic_auditable_outputs_and_preserves_sources(
    tmp_path: Path,
) -> None:
    inputs = _write_synthetic_inputs(tmp_path)
    source_state = {
        name: (path_digest(path), path.stat().st_mtime_ns)
        for name, path in inputs.source_paths().items()
    }

    result = run_study(inputs, _settings())

    assert result.run_dir == inputs.output_root / result.run_id
    assert result.run_dir.is_dir()
    assert set(REQUIRED_ARTIFACTS).issubset(
        path.name for path in result.run_dir.iterdir()
    )
    assert not list(inputs.output_root.glob(".*.tmp-*"))

    provenance = json.loads((result.run_dir / "provenance.json").read_text())
    assert provenance["implementation_identity"]["implementation_version"] == (
        "htsr-combined-implementation-v6"
    )
    assert provenance["implementation_identity"]["implementation_digest"] == (
        _study_implementation_digest()
    )
    assert provenance["implementation_identity"]["feature_extractor_digest"] == (
        _feature_extractor_digest()
    )
    manifest = pd.read_csv(result.run_dir / "manifest.csv")
    features = pd.read_csv(result.run_dir / "feature_table.csv")
    assert provenance["counts"]["manifest_rows"] == len(manifest) == 12
    assert provenance["counts"]["development_feature_rows"] == len(features) == 12
    assert provenance["counts"]["development_target_rows"] == 12
    assert provenance["counts"]["development_candidate_rows"] == 12
    assert provenance["counts"]["development_group_eligible_rows"] == 12
    assert provenance["counts"]["development_group_eligible_components"] == 3
    assert provenance["counts"]["development_closure_eligible_rows"] == 12
    assert provenance["counts"]["development_closure_components"] == 3
    assert provenance["external"]["status"] == "unavailable"
    assert (result.run_dir / "cluster_assignments.csv").is_file()
    assert (result.run_dir / "lineage_representative_predictions.csv").is_file()

    lineage_metrics = json.loads(
        (result.run_dir / "lineage_metrics.json").read_text(encoding="utf-8")
    )
    assert {row["inferential"] for row in lineage_metrics["closed_set"]} == {False}
    assert {row["minimum_families"] for row in lineage_metrics["closed_set"]} == {10}
    assert all(
        row["ci_low"] is None and row["ci_high"] is None
        for row in lineage_metrics["closed_set_confidence_intervals"]
    )
    assert {row["inferential"] for row in lineage_metrics["open_set"]} == {False}
    assert {row["inferential"] for row in lineage_metrics["observed_family"]} == {
        False
    }
    assert {
        row["inference_reason"] for row in lineage_metrics["observed_family"]
    } == {"post_hoc_descriptive_endpoint"}
    assert {
        row["aggregation"] for row in lineage_metrics["open_set"]
    } == {"held_out_family_macro"}
    assert lineage_metrics["closed_set_contrasts"]
    assert (result.run_dir / "lineage_closed_contrasts.csv").is_file()
    combined_closed = [
        row for row in lineage_metrics["closed_set"] if row["method"] == "combined"
    ]
    assert {row["architecture_matched_n"] for row in combined_closed} == {9}
    performance_metrics = json.loads(
        (result.run_dir / "performance_metrics.json").read_text(encoding="utf-8")
    )
    secondary = performance_metrics["secondary_increments"]
    assert {row["target"] for row in secondary} == set(TASKS)
    assert {row["holm_family_size"] for row in secondary} == {6}
    assert "group_weighted_mae" in performance_metrics["primary"]
    report = (result.run_dir / "report.md").read_text(encoding="utf-8")
    assert "LOFO pseudo-unknown" in report
    assert "Cross-fitted observed-family combined top-1" in report
    assert "known reference-library attribution" in report
    assert "External transfer was unavailable for confirmatory nonreversal" in report
    assert "primary evidence was insufficient to claim improvement" in report
    assert "model-level descriptive statistics" in report
    assert "matched family recall" in report
    assert "Closed-set scope: 9 queries / 3 root families" in report
    assert "Closed-set inference status: descriptive" in report
    assert "minimum 10 independent families" in report
    assert "Primary candidate population: 12 joined models" in report
    assert "Primary graph-eligible population: 12 models / 3 graph-supported ancestry components" in report
    assert "Primary evaluated OOF population: 12 models / 3 graph-supported ancestry components" in report
    assert "Merge-retaining grouping diagnostic: 12 edge-supported models / 3 ancestry-closure components (not fitted)" in report
    assert "Development ancestry components:" not in report
    assert "conditional on the frozen grouped folds" in report
    assert "stored DKS" in report
    assert provenance["methodological_scope"]["checkpoint_matching"] == (
        "canonical_model_id; checkpoint SHA is not available in stats artifacts"
    )
    assert "single-source" in provenance["methodological_scope"]["performance_grouping"]

    for name, path in inputs.source_paths().items():
        assert (path_digest(path), path.stat().st_mtime_ns) == source_state[name]


def test_runner_excludes_nonconfirmatory_schema_from_all_fitted_analyses(
    tmp_path: Path,
) -> None:
    inputs = _write_synthetic_inputs(tmp_path)
    excluded_id = "org-0/child-1"
    stats_path = inputs.development_stats / "org-0--child-1.csv"
    stats = pd.read_csv(stats_path)
    stats = stats.loc[~stats["longname"].str.endswith("mlp.down_proj")]
    stats.to_csv(stats_path, index=False)

    result = run_study(inputs, _settings())

    features = pd.read_csv(result.run_dir / "feature_table.csv")
    assignments = pd.read_csv(result.run_dir / "cluster_assignments.csv")
    lineage = pd.read_csv(result.run_dir / "lineage_predictions.csv")
    performance = pd.read_csv(result.run_dir / "performance_predictions.csv")
    exclusions = pd.read_csv(result.run_dir / "exclusions.csv")
    provenance = json.loads((result.run_dir / "provenance.json").read_text())
    assert excluded_id in set(features["model_id"])
    assert excluded_id not in set(assignments["model_id"])
    assert excluded_id not in set(lineage["query_model_id"])
    assert excluded_id not in set(performance["model_id"])
    assert (
        exclusions.loc[exclusions["model_id"].eq(excluded_id), "reason"]
        .eq("incomplete_canonical_roles")
        .any()
    )
    assert provenance["counts"]["development_confirmatory_feature_rows"] == 11


def test_matching_run_reuses_persistent_cache_without_rewriting_features(
    tmp_path: Path,
) -> None:
    inputs = _write_synthetic_inputs(tmp_path)
    first = run_study(inputs, _settings())
    feature_path = first.run_dir / "feature_table.csv"
    first_mtime = feature_path.stat().st_mtime_ns

    second = run_study(inputs, _settings())

    assert second.run_id == first.run_id
    assert second.run_dir == first.run_dir
    assert second.cache_reused is True
    assert feature_path.stat().st_mtime_ns == first_mtime


def test_new_run_identity_reuses_feature_cache_when_data_are_unchanged(
    tmp_path: Path,
) -> None:
    inputs = _write_synthetic_inputs(tmp_path)
    first = run_study(inputs, _settings("synthetic-study-v1"))
    first_provenance = json.loads(
        (first.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    cache_feature = (
        Path(first_provenance["cache"]["development"]) / "feature_table.csv"
    )
    first_mtime = cache_feature.stat().st_mtime_ns

    second = run_study(inputs, _settings("synthetic-study-v2"))
    second_provenance = json.loads(
        (second.run_dir / "provenance.json").read_text(encoding="utf-8")
    )

    assert second.run_id != first.run_id
    assert second.cache_reused is True
    assert second_provenance["cache"]["development"] == first_provenance["cache"]["development"]
    assert cache_feature.stat().st_mtime_ns == first_mtime


def test_extractor_source_change_invalidates_feature_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_synthetic_inputs(tmp_path)
    monkeypatch.setattr(
        "spectral_research.study.runner._feature_extractor_digest",
        lambda: "a" * 64,
    )
    first = run_study(inputs, _settings("synthetic-study-v1"))
    first_provenance = json.loads(
        (first.run_dir / "provenance.json").read_text(encoding="utf-8")
    )

    monkeypatch.setattr(
        "spectral_research.study.runner._feature_extractor_digest",
        lambda: "b" * 64,
    )
    second = run_study(inputs, _settings("synthetic-study-v1"))
    second_provenance = json.loads(
        (second.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    second_metadata = json.loads(
        (
            Path(second_provenance["cache"]["development"])
            / "cache_metadata.json"
        ).read_text(encoding="utf-8")
    )

    assert second.cache_reused is False
    assert second.run_id != first.run_id
    assert second_provenance["cache"]["development"] != (
        first_provenance["cache"]["development"]
    )
    assert second_metadata["extractor_source_digest"] == "b" * 64


def test_corrupt_feature_cache_is_rejected_instead_of_reused(tmp_path: Path) -> None:
    inputs = _write_synthetic_inputs(tmp_path)
    first = run_study(inputs, _settings("synthetic-study-v1"))
    provenance = json.loads(
        (first.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    metadata_path = Path(provenance["cache"]["development"]) / "cache_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["feature_rows"] += 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="feature cache.*feature_rows"):
        run_study(inputs, _settings("synthetic-study-v2"))


def test_same_shape_feature_cache_mutation_is_rejected(tmp_path: Path) -> None:
    inputs = _write_synthetic_inputs(tmp_path)
    first = run_study(inputs, _settings("synthetic-study-v1"))
    provenance = json.loads(
        (first.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    feature_path = Path(provenance["cache"]["development"]) / "feature_table.csv"
    content = feature_path.read_bytes()
    mutated = content.replace(b"1.8", b"9.8", 1)
    assert mutated != content and len(mutated) == len(content)
    feature_path.write_bytes(mutated)

    with pytest.raises(RuntimeError, match="digest mismatch.*feature_table.csv"):
        run_study(inputs, _settings("synthetic-study-v2"))


def test_corrupt_completed_run_is_rejected_instead_of_reused(tmp_path: Path) -> None:
    inputs = _write_synthetic_inputs(tmp_path)
    first = run_study(inputs, _settings())
    provenance_path = first.run_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["run_id"] = "wrong-run-id"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(RuntimeError, match="provenance.json|run_id"):
        run_study(inputs, _settings())


def test_same_shape_artifact_mutation_is_rejected_instead_of_reused(
    tmp_path: Path,
) -> None:
    inputs = _write_synthetic_inputs(tmp_path)
    first = run_study(inputs, _settings())
    feature_path = first.run_dir / "feature_table.csv"
    content = feature_path.read_bytes()
    mutated = content.replace(b"1.8", b"9.8", 1)
    assert mutated != content
    assert len(mutated) == len(content)
    feature_path.write_bytes(mutated)

    with pytest.raises(RuntimeError, match="digest mismatch.*feature_table.csv"):
        run_study(inputs, _settings())


def test_atomic_directory_never_removes_an_existing_publication(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    marker = destination / "marker.txt"
    marker.write_text("existing", encoding="utf-8")

    published = _atomic_directory(
        destination,
        lambda temporary: (temporary / "marker.txt").write_text(
            "replacement", encoding="utf-8"
        ),
    )

    assert published is False
    assert marker.read_text(encoding="utf-8") == "existing"
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_atomic_directory_treats_concurrent_enotempty_as_a_lost_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "published"

    def concurrent_rename(_source: Path, target: Path) -> None:
        target.mkdir()
        (target / "winner.txt").write_text("winner", encoding="utf-8")
        raise OSError(errno.ENOTEMPTY, "directory not empty")

    monkeypatch.setattr("spectral_research.study.runner.os.rename", concurrent_rename)

    published = _atomic_directory(
        destination,
        lambda temporary: (temporary / "ours.txt").write_text(
            "ours", encoding="utf-8"
        ),
    )

    assert published is False
    assert (destination / "winner.txt").read_text(encoding="utf-8") == "winner"


def test_source_or_schema_change_creates_a_new_run(tmp_path: Path) -> None:
    inputs = _write_synthetic_inputs(tmp_path)
    first = run_study(inputs, _settings())

    stats_path = next(inputs.development_stats.glob("*.csv"))
    changed = pd.read_csv(stats_path)
    changed.loc[0, "alpha"] += 0.25
    changed.to_csv(stats_path, index=False)
    changed_source = run_study(inputs, _settings())
    changed_schema = run_study(inputs, _settings("synthetic-study-v2"))

    assert changed_source.run_id != first.run_id
    assert changed_schema.run_id != changed_source.run_id
    assert changed_source.run_dir.is_dir()
    assert changed_schema.run_dir.is_dir()

    first_provenance = json.loads(
        (first.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    changed_provenance = json.loads(
        (changed_source.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert changed_source.cache_reused is False
    assert changed_provenance["cache"]["development"] != first_provenance["cache"]["development"]


def test_runner_hashes_each_source_only_once_per_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import spectral_research.study.contracts as contracts_module
    import spectral_research.study.runner as runner_module

    inputs = _write_synthetic_inputs(tmp_path)
    original = contracts_module.path_digest
    calls: dict[Path, int] = {}

    def counted(path, *args, **kwargs):
        resolved = Path(path).resolve()
        calls[resolved] = calls.get(resolved, 0) + 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr(contracts_module, "path_digest", counted)
    monkeypatch.setattr(runner_module, "path_digest", counted)

    run_study(inputs, _settings())

    assert calls == {path: 1 for path in inputs.source_paths().values()}


def test_infeasible_primary_analysis_still_writes_readable_frozen_artifacts(
    tmp_path: Path,
) -> None:
    inputs = _write_synthetic_inputs(tmp_path)
    settings = RunSettings(
        schema_version="infeasible-study-v1",
        random_seed=11,
        bootstrap_replicates=8,
        outer_folds=4,
        inner_folds=2,
        ridge_alphas=(0.1, 1.0),
    )

    result = run_study(inputs, settings)

    predictions = pd.read_csv(result.run_dir / "performance_predictions.csv")
    metrics = json.loads(
        (result.run_dir / "performance_metrics.json").read_text(encoding="utf-8")
    )
    exclusions = pd.read_csv(result.run_dir / "exclusions.csv")
    assert predictions.empty
    assert {"model_id", "target", "feature_block", "actual", "prediction"}.issubset(
        predictions.columns
    )
    assert metrics["primary_decision"] == {
        "passed": False,
        "reasons": ["primary_analysis_infeasible"],
    }
    assert "infeasible_grouped_cv" in set(exclusions["reason"])


def test_external_transfer_uses_the_full_development_artifact_component_universe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development = pd.DataFrame(
        {
            "model_id": ["dev/a", "dev/b"],
            "ancestry_component": ["cc_dev_a", "cc_dev_b"],
            "component_provenance": ["graph", "graph"],
            "component_edge_count": [1, 1],
            "modern_average": [40.0, 60.0],
        }
    )
    external = pd.DataFrame(
        {
            "model_id": ["external/a"],
            "ancestry_component": ["cc_external"],
            "component_provenance": ["graph"],
            "component_edge_count": [1],
            "modern_average": [50.0],
        }
    )
    development_only = pd.DataFrame(
        {
            "model_id": ["dev/no-target"],
            "ancestry_component": ["cc_dev_unfitted"],
            "component_provenance": ["graph"],
            "component_edge_count": [1],
            "modern_average": [float("nan")],
        }
    )
    manifest = pd.concat(
        [development, development_only, external], ignore_index=True
    )
    manifest["development_artifact_available"] = [True, True, True, False]
    captured: dict[str, set[str]] = {}

    def fake_external_transfer(
        _development: pd.DataFrame,
        _external: pd.DataFrame,
        *,
        development_component_universe: pd.DataFrame,
        **_kwargs,
    ):
        captured["model_ids"] = set(
            development_component_universe["model_id"].astype(str)
        )
        predictions = pd.DataFrame(
            [
                {
                    "model_id": "external/a",
                    "ancestry_component": "cc_external",
                    "target": "modern_average",
                    "feature_block": block,
                    "actual": 50.0,
                    "prediction": prediction,
                    "external_slice": "component_disjoint",
                }
                for block, prediction in (("M+S", 48.0), ("M+S+H", 49.0))
            ]
        )
        metrics = pd.DataFrame(
            [
                {
                    "target": "modern_average",
                    "feature_block": block,
                    "n_models": 1,
                    "n_components": 1,
                    "group_weighted_mae": error,
                    "kendall_tau": float("nan"),
                    "spearman": float("nan"),
                    "r2": float("nan"),
                    "external_slice": "component_disjoint",
                }
                for block, error in (("M+S", 2.0), ("M+S+H", 1.0))
            ]
        )
        return predictions, metrics, pd.DataFrame()

    monkeypatch.setattr(
        "spectral_research.study.runner.evaluate_external_transfer",
        fake_external_transfer,
    )
    settings = RunSettings(
        schema_version="external-universe-v1",
        random_seed=3,
        bootstrap_replicates=4,
        outer_folds=3,
        inner_folds=2,
        ridge_alphas=(1.0,),
    )

    _evaluate_performance(
        development,
        settings=settings,
        manifest=manifest,
        external=external,
    )

    assert captured["model_ids"] == {"dev/a", "dev/b", "dev/no-target"}
