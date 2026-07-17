from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from alpha_cluster_dashboard.study_outputs import load_study_outputs
from spectral_research.study.artifacts import write_artifact_manifest


def _write_minimal_frozen_run(
    root: Path,
    *,
    n_models: int,
    external_status: str = "unavailable",
) -> Path:
    run = root / "htsr_test_run"
    run.mkdir()
    model_ids = [f"org/model-{index}" for index in range(n_models)]
    component_count = (n_models + 1) // 2
    pd.DataFrame(
        {
            "model_id": model_ids,
            "ancestry_component": [f"component-{index // 2}" for index in range(n_models)],
            "audited_base_family": ["org/root-a"] * n_models,
            "label_status": ["confirmed"] * n_models,
        }
    ).to_csv(run / "manifest.csv", index=False)
    pd.DataFrame(
        {
            "model_id": model_ids,
            "feature_schema_version": ["htsr-study-features-v1"] * n_models,
            "M__layer_count": [32] * n_models,
            "S__attn_q__log_rms__median": [0.25] * n_models,
            "H__attn_q__alpha__median": [2.0 + 0.1 * index for index in range(n_models)],
            "H__attn_q__alpha__iqr": [0.1] * n_models,
        }
    ).to_csv(run / "feature_table.csv", index=False)
    pd.DataFrame(
        {
            "model_id": model_ids,
            "cluster": [index % 2 for index in range(n_models)],
        }
    ).to_csv(run / "cluster_assignments.csv", index=False)
    pd.DataFrame(
        {
            "query_model_id": model_ids,
            "method": ["combined"] * n_models,
            "true_family": ["org/root-a"] * n_models,
            "predicted_family": ["org/root-a"] * n_models,
            "top_score": [0.9] * n_models,
            "correct": [True] * n_models,
            "abstained": [False] * n_models,
        }
    ).to_csv(run / "lineage_predictions.csv", index=False)
    pd.DataFrame(
        {
            "query_model_id": model_ids,
            "endpoint": ["cross_fitted_observed_family_representative"] * n_models,
            "method": ["combined"] * n_models,
            "true_family": ["org/root-a"] * n_models,
            "predicted_family": ["org/root-a"] * n_models,
            "top_score": [0.85] * n_models,
            "correct": [True] * n_models,
            "abstained": [False] * n_models,
        }
    ).to_csv(run / "lineage_representative_predictions.csv", index=False)
    pd.DataFrame(
        {
            "evaluation_scope": ["development"] * n_models,
            "model_id": model_ids,
            "ancestry_component": [f"component-{index // 2}" for index in range(n_models)],
            "target": ["modern_average"] * n_models,
            "feature_block": ["M+S+H"] * n_models,
            "actual": [40.0 + index for index in range(n_models)],
            "prediction": [40.5 + index for index in range(n_models)],
            "external_slice": [""] * n_models,
        }
    ).to_csv(run / "performance_predictions.csv", index=False)
    pd.DataFrame(columns=["stage", "model_id", "reason", "detail"]).to_csv(
        run / "exclusions.csv", index=False
    )
    (run / "lineage_metrics.json").write_text(
        json.dumps(
            {
                "closed_set": [
                    {
                        "method": "combined",
                        "metric": "root_macro_top1",
                        "estimate": 1.0,
                        "n_queries": n_models,
                        "n_families": 12,
                        "minimum_families": 10,
                        "inferential": True,
                    },
                    {
                        "method": "combined",
                        "metric": "mrr",
                        "estimate": 1.0,
                        "n_queries": n_models,
                        "n_families": 12,
                        "minimum_families": 10,
                        "inferential": True,
                    },
                ],
                "open_set": [
                    {
                        "method": "combined",
                        "metric": "auroc",
                        "estimate": 0.8,
                        "n_folds": 12,
                        "minimum_families": 10,
                        "inferential": True,
                    },
                    {
                        "method": "combined",
                        "metric": "oscr",
                        "estimate": 0.7,
                        "n_folds": 12,
                        "minimum_families": 10,
                        "inferential": True,
                    },
                ],
                "observed_family": [
                    {
                        "method": "spectral_shape",
                        "metric": "root_macro_top1",
                        "estimate": 0.96,
                        "n_queries": n_models,
                        "n_families": 12,
                        "inferential": False,
                        "inference_reason": "post_hoc_descriptive_endpoint",
                    },
                    {
                        "method": "combined",
                        "metric": "root_macro_top1",
                        "estimate": 1.0,
                        "n_queries": n_models,
                        "n_families": 12,
                        "inferential": False,
                        "inference_reason": "post_hoc_descriptive_endpoint",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run / "performance_metrics.json").write_text(
        json.dumps(
            {
                "primary": {
                    "target": "modern_average",
                    "group_weighted_mae": 0.5,
                    "spearman": 1.0,
                    "htsr_delta_mae": -0.2,
                    "n_models": n_models,
                    "n_components": component_count,
                },
                "external": {
                    "status": external_status,
                    "evidence_role": "reused_descriptive",
                    "metrics": [
                        {
                            "external_slice": "component_disjoint",
                            "target": "modern_average",
                            "feature_block": "M+S",
                            "group_weighted_mae": 0.6,
                            "n_models": n_models,
                            "n_components": component_count,
                        },
                        {
                            "external_slice": "component_disjoint",
                            "target": "modern_average",
                            "feature_block": "M+S+H",
                            "group_weighted_mae": 0.5,
                            "n_models": n_models,
                            "n_components": component_count,
                        },
                    ],
                },
                "primary_decision": {
                    "passed": False,
                    "reasons": [
                        "insufficient_mae_improvement",
                        "bootstrap_interval_includes_zero",
                    ],
                    "relative_mae_improvement": 0.04,
                },
            }
        ),
        encoding="utf-8",
    )
    (run / "provenance.json").write_text(
        json.dumps(
            {
                "run_id": run.name,
                "counts": {"manifest_rows": 398, "audit_rows": 398},
                "external": {"status": external_status},
            }
        ),
        encoding="utf-8",
    )
    (run / "report.md").write_text("# Synthetic frozen run\n", encoding="utf-8")
    write_artifact_manifest(run, run_id=run.name)
    return run


def test_study_output_loader_uses_table_counts_not_stale_provenance(
    tmp_path: Path,
) -> None:
    loaded = load_study_outputs(_write_minimal_frozen_run(tmp_path, n_models=7))

    assert loaded.summary["n_models"] == 7
    assert loaded.summary["n_feature_models"] == 7
    assert loaded.summary["n_lineage_models"] == 7
    assert loaded.summary["n_performance_models"] == 7
    assert len(loaded.atlas) == 7


def test_loader_reads_only_dashboard_feature_columns_from_full_artifact(
    tmp_path: Path,
) -> None:
    loaded = load_study_outputs(_write_minimal_frozen_run(tmp_path, n_models=3))

    assert loaded.features.columns.tolist() == [
        "model_id",
        "M__layer_count",
        "H__attn_q__alpha__median",
    ]
    assert "S__attn_q__log_rms__median" not in loaded.atlas.columns
    assert "H__attn_q__alpha__iqr" not in loaded.atlas.columns


def test_loader_validates_full_feature_header_against_artifact_manifest(
    tmp_path: Path,
) -> None:
    run = _write_minimal_frozen_run(tmp_path, n_models=3)
    manifest_path = run / "artifact_manifest.json"
    artifact_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_manifest["artifacts"]["feature_table.csv"]["columns"].remove(
        "S__attn_q__log_rms__median"
    )
    manifest_path.write_text(json.dumps(artifact_manifest), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="feature_table.csv shape does not match artifact_manifest.json",
    ):
        load_study_outputs(run)


def test_atlas_global_alpha_is_mean_log1p_not_log_of_or_raw_mean(
    tmp_path: Path,
) -> None:
    loaded = load_study_outputs(_write_minimal_frozen_run(tmp_path, n_models=2))

    observed = loaded.atlas.set_index("model_id")["overall_log_alpha"]
    assert observed["org/model-0"] == pytest.approx(float(__import__("numpy").log1p(2.0)))
    assert observed["org/model-1"] == pytest.approx(float(__import__("numpy").log1p(2.1)))


def test_dashboard_adapter_never_refits_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("frozen-output loading must not fit an estimator")

    monkeypatch.setattr("sklearn.linear_model.Ridge.fit", fail)
    monkeypatch.setattr("sklearn.cluster.KMeans.fit", fail)

    loaded = load_study_outputs(_write_minimal_frozen_run(tmp_path, n_models=4))

    assert loaded.performance_predictions["prediction"].notna().all()
    assert loaded.atlas["predicted_family"].eq("org/root-a").all()


def test_loader_falls_back_to_observed_family_predictions_for_atlas(
    tmp_path: Path,
) -> None:
    run = _write_minimal_frozen_run(tmp_path, n_models=4)
    root_predictions = pd.read_csv(run / "lineage_predictions.csv").iloc[:1]
    root_predictions.to_csv(run / "lineage_predictions.csv", index=False)
    lineage_metrics_path = run / "lineage_metrics.json"
    lineage_metrics = json.loads(lineage_metrics_path.read_text(encoding="utf-8"))
    for row in lineage_metrics["closed_set"]:
        row["n_queries"] = 1
    lineage_metrics_path.write_text(json.dumps(lineage_metrics), encoding="utf-8")
    observed_predictions = pd.read_csv(
        run / "lineage_representative_predictions.csv"
    )
    observed_predictions.loc[1, "top_score"] = 0.42
    observed_predictions.loc[1, "correct"] = False
    observed_predictions.loc[1, "abstained"] = True
    observed_predictions.to_csv(
        run / "lineage_representative_predictions.csv",
        index=False,
    )
    lineage_metrics = json.loads(lineage_metrics_path.read_text(encoding="utf-8"))
    for row in lineage_metrics["observed_family"]:
        if row["method"] == "combined":
            row["estimate"] = 0.75
    lineage_metrics_path.write_text(json.dumps(lineage_metrics), encoding="utf-8")
    write_artifact_manifest(run, run_id=run.name)

    loaded = load_study_outputs(run)
    atlas = loaded.atlas.set_index("model_id")

    assert loaded.atlas["predicted_family"].eq("org/root-a").all()
    assert loaded.atlas["lineage_endpoint"].value_counts().to_dict() == {
        "observed_family": 3,
        "canonical_root": 1,
    }
    assert atlas.loc["org/model-0", "top_score"] == pytest.approx(0.9)
    assert atlas.loc["org/model-1", "top_score"] == pytest.approx(0.42)
    assert bool(atlas.loc["org/model-1", "correct"]) is False
    assert bool(atlas.loc["org/model-1", "abstained"]) is True
    assert loaded.summary["n_lineage_models"] == 4


def test_missing_optional_external_results_render_as_unavailable(
    tmp_path: Path,
) -> None:
    loaded = load_study_outputs(_write_minimal_frozen_run(tmp_path, n_models=3))

    assert loaded.external_status == "unavailable"


def test_loader_requires_an_explicit_complete_run_directory(tmp_path: Path) -> None:
    run = _write_minimal_frozen_run(tmp_path, n_models=2)
    (run / "manifest.csv").unlink()

    with pytest.raises(ValueError, match="manifest.csv"):
        load_study_outputs(run)


@pytest.mark.parametrize(
    ("filename", "column"),
    (
        ("feature_table.csv", "M__layer_count"),
        ("cluster_assignments.csv", "cluster"),
        ("lineage_predictions.csv", "abstained"),
        ("lineage_representative_predictions.csv", "endpoint"),
        ("performance_predictions.csv", "evaluation_scope"),
    ),
)
def test_loader_rejects_missing_required_table_columns(
    tmp_path: Path,
    filename: str,
    column: str,
) -> None:
    run = _write_minimal_frozen_run(tmp_path, n_models=3)
    path = run / filename
    frame = pd.read_csv(path)
    frame.drop(columns=column).to_csv(path, index=False)
    write_artifact_manifest(run, run_id=run.name)

    with pytest.raises(
        ValueError,
        match=rf"{filename} is missing required columns.*{column}",
    ):
        load_study_outputs(run)


@pytest.mark.parametrize(
    "filename",
    (
        "lineage_predictions.csv",
        "lineage_representative_predictions.csv",
        "performance_predictions.csv",
    ),
)
def test_loader_rejects_duplicate_prediction_composite_keys(
    tmp_path: Path,
    filename: str,
) -> None:
    run = _write_minimal_frozen_run(tmp_path, n_models=3)
    path = run / filename
    frame = pd.read_csv(path)
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(path, index=False)
    write_artifact_manifest(run, run_id=run.name)

    with pytest.raises(
        ValueError,
        match=rf"{filename} contains duplicate composite key",
    ):
        load_study_outputs(run)


@pytest.mark.parametrize("invalid_cluster", (-1, 1.5, "not-a-cluster"))
def test_loader_rejects_clusters_that_are_not_nonnegative_integers(
    tmp_path: Path,
    invalid_cluster: object,
) -> None:
    run = _write_minimal_frozen_run(tmp_path, n_models=3)
    path = run / "cluster_assignments.csv"
    frame = pd.read_csv(path)
    frame["cluster"] = frame["cluster"].astype(object)
    frame.loc[0, "cluster"] = invalid_cluster
    frame.to_csv(path, index=False)
    write_artifact_manifest(run, run_id=run.name)

    with pytest.raises(
        ValueError,
        match="cluster_assignments.csv contains invalid cluster values",
    ):
        load_study_outputs(run)


def test_loader_exposes_minimal_scientific_metric_cards(tmp_path: Path) -> None:
    loaded = load_study_outputs(_write_minimal_frozen_run(tmp_path, n_models=5))

    assert loaded.lineage_summary == {
        "root_macro_top1": 1.0,
        "mrr": 1.0,
        "unknown_auroc": 0.8,
        "oscr": 0.7,
        "n_queries": 5,
        "n_families": 12,
        "n_open_folds": 12,
        "minimum_families": 10,
        "inferential": True,
        "observed_family_top1": 1.0,
        "observed_spectral_shape_top1": 0.96,
        "observed_htsr_lift": pytest.approx(0.04),
        "observed_family_queries": 5,
        "observed_family_families": 12,
    }
    assert loaded.performance_summary["group_weighted_mae"] == 0.5
    assert loaded.performance_summary["spearman"] == 1.0
    assert loaded.performance_summary["htsr_delta_mae"] == -0.2
    assert loaded.performance_summary["n_models"] == 5
    assert loaded.performance_summary["n_components"] == 3
    assert loaded.performance_summary["target"] == "modern_average"
    assert loaded.performance_summary["external_target"] == "modern_average"
    assert loaded.performance_summary["external_htsr_delta_mae"] == pytest.approx(-0.1)
    assert loaded.performance_summary["external_n_models"] == 5
    assert loaded.performance_summary["external_n_components"] == 3
    assert loaded.performance_summary["primary_passed"] is False
    assert loaded.performance_summary["primary_reasons"] == [
        "insufficient_mae_improvement",
        "bootstrap_interval_includes_zero",
    ]
    assert loaded.performance_summary["relative_mae_improvement"] == pytest.approx(
        0.04
    )
    assert loaded.performance_summary["external_status"] == "unavailable"
    assert loaded.performance_summary["external_evidence_role"] == "reused_descriptive"


def test_loader_rejects_primary_metric_prediction_count_mismatch(
    tmp_path: Path,
) -> None:
    run = _write_minimal_frozen_run(tmp_path, n_models=5)
    metrics_path = run / "performance_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["primary"]["n_models"] = 999
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    write_artifact_manifest(run, run_id=run.name)

    with pytest.raises(ValueError, match="primary performance n_models mismatch"):
        load_study_outputs(run)


def test_loader_rejects_primary_metric_value_mismatch(tmp_path: Path) -> None:
    run = _write_minimal_frozen_run(tmp_path, n_models=5)
    metrics_path = run / "performance_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["primary"]["group_weighted_mae"] = 999.0
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    write_artifact_manifest(run, run_id=run.name)

    with pytest.raises(ValueError, match="primary performance MAE mismatch"):
        load_study_outputs(run)


def test_loader_rejects_missing_primary_metric_value(tmp_path: Path) -> None:
    run = _write_minimal_frozen_run(tmp_path, n_models=5)
    metrics_path = run / "performance_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["primary"]["group_weighted_mae"] = None
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    write_artifact_manifest(run, run_id=run.name)

    with pytest.raises(ValueError, match="primary performance MAE.*missing"):
        load_study_outputs(run)


def test_loader_rejects_lineage_metric_prediction_count_mismatch(
    tmp_path: Path,
) -> None:
    run = _write_minimal_frozen_run(tmp_path, n_models=5)
    metrics_path = run / "lineage_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["closed_set"][0]["n_queries"] = 999
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    write_artifact_manifest(run, run_id=run.name)

    with pytest.raises(ValueError, match="closed-set lineage n_queries mismatch"):
        load_study_outputs(run)


def test_loader_rejects_lineage_metric_value_mismatch(tmp_path: Path) -> None:
    run = _write_minimal_frozen_run(tmp_path, n_models=5)
    metrics_path = run / "lineage_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["closed_set"][0]["estimate"] = 0.0
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    write_artifact_manifest(run, run_id=run.name)

    with pytest.raises(ValueError, match="closed-set lineage top-1 mismatch"):
        load_study_outputs(run)


def test_loader_rejects_missing_lineage_metric_value(tmp_path: Path) -> None:
    run = _write_minimal_frozen_run(tmp_path, n_models=5)
    metrics_path = run / "lineage_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["closed_set"][0]["estimate"] = None
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    write_artifact_manifest(run, run_id=run.name)

    with pytest.raises(ValueError, match="closed-set lineage top-1.*missing"):
        load_study_outputs(run)


def test_loader_rejects_unknown_representative_endpoint(tmp_path: Path) -> None:
    run = _write_minimal_frozen_run(tmp_path, n_models=5)
    path = run / "lineage_representative_predictions.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "endpoint"] = "unrelated_endpoint"
    frame.to_csv(path, index=False)
    write_artifact_manifest(run, run_id=run.name)

    with pytest.raises(ValueError, match="unexpected endpoint"):
        load_study_outputs(run)


def test_loader_rejects_missing_representative_endpoint(tmp_path: Path) -> None:
    run = _write_minimal_frozen_run(tmp_path, n_models=5)
    path = run / "lineage_representative_predictions.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "endpoint"] = pd.NA
    frame.to_csv(path, index=False)
    write_artifact_manifest(run, run_id=run.name)

    with pytest.raises(ValueError, match="unexpected endpoint"):
        load_study_outputs(run)


def test_loader_rejects_external_metric_value_mismatch(tmp_path: Path) -> None:
    run = _write_minimal_frozen_run(
        tmp_path,
        n_models=5,
        external_status="descriptive_reuse",
    )
    predictions_path = run / "performance_predictions.csv"
    development = pd.read_csv(predictions_path)
    external_rows = []
    for block, error in (("M+S", 0.6), ("M+S+H", 0.5)):
        rows = development.copy()
        rows["evaluation_scope"] = "external"
        rows["feature_block"] = block
        rows["prediction"] = rows["actual"] + error
        rows["external_slice"] = "component_disjoint"
        external_rows.append(rows)
    pd.concat([development, *external_rows], ignore_index=True).to_csv(
        predictions_path,
        index=False,
    )
    metrics_path = run / "performance_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["external"]["metrics"][1]["group_weighted_mae"] = 999.0
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    write_artifact_manifest(run, run_id=run.name)

    with pytest.raises(ValueError, match=r"external M\+S\+H MAE mismatch"):
        load_study_outputs(run)


def test_loader_rejects_artifact_mutation_even_when_shape_is_unchanged(
    tmp_path: Path,
) -> None:
    run = _write_minimal_frozen_run(tmp_path, n_models=3)
    path = run / "feature_table.csv"
    content = path.read_bytes()
    mutated = content.replace(b"2.0", b"9.0", 1)
    assert len(mutated) == len(content)
    path.write_bytes(mutated)

    with pytest.raises(ValueError, match="digest mismatch.*feature_table.csv"):
        load_study_outputs(run)


def test_loader_rejects_provenance_identity_inconsistent_with_run_directory(
    tmp_path: Path,
) -> None:
    run = _write_minimal_frozen_run(tmp_path, n_models=3)
    provenance_path = run / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["run_id"] = "htsr_wrong"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    write_artifact_manifest(run, run_id=run.name)

    with pytest.raises(ValueError, match="provenance.*run_id"):
        load_study_outputs(run)


@pytest.mark.parametrize(
    ("filename", "column"),
    (("lineage_predictions.csv", "query_model_id"), ("performance_predictions.csv", "model_id")),
)
def test_loader_rejects_prediction_models_outside_manifest(
    tmp_path: Path, filename: str, column: str
) -> None:
    run = _write_minimal_frozen_run(tmp_path, n_models=3)
    path = run / filename
    frame = pd.read_csv(path)
    frame.loc[0, column] = "outside/model"
    frame.to_csv(path, index=False)
    write_artifact_manifest(run, run_id=run.name)

    with pytest.raises(ValueError, match=f"{filename} contains models outside"):
        load_study_outputs(run)
