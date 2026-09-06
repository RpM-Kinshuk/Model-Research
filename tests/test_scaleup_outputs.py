from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from alpha_cluster_dashboard.scaleup_outputs import (
    LEADING_PERFORMANCE_METHOD,
    load_scaleup_outputs,
)
from spectral_research.study.artifacts import write_artifact_manifest


def _metric_rows(scope: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for generation, target, models, groups, mae, baseline in (
        ("modern", "modern_average", 2, 2, 0.25, 1.0),
        ("legacy", "legacy_average", 1, 1, 1.0, 1.5),
    ):
        for method, method_mae, spearman in (
            (LEADING_PERFORMANCE_METHOD, mae, 1.0),
            ("training_mean", baseline, 0.0),
        ):
            rows.append(
                {
                    "evaluation_scope": scope,
                    "generation": generation,
                    "target": target,
                    "feature_block": method,
                    "n_models": models,
                    "n_profile_groups": groups,
                    "model_weighted_mae": method_mae,
                    "rmse": method_mae,
                    "spearman": spearman,
                    "kendall_tau": spearman,
                    "r2": 0.5,
                    "prediction_inputs": (
                        "alpha_profile_and_training_scores"
                        if method == LEADING_PERFORMANCE_METHOD
                        else "training_scores_only"
                    ),
                    "lineage_metadata_used": False,
                }
            )
    return rows


def _write_scaleup_run(root: Path) -> Path:
    run = root / "weight_only_test"
    run.mkdir()
    model_ids = ["org/a", "org/b", "org/c"]
    (run / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run.name,
                "counts": {
                    "source_models": 3,
                    "profiled_models": 2,
                    "unlabelled_clusters": 2,
                },
                "method_boundary": {
                    "metadata_role": "post_freeze_validation_only",
                    "lineage_prediction_inputs": ["seven_role_alpha_profiles"],
                    "performance_prediction_inputs": [
                        "seven_role_alpha_profiles",
                        "training_benchmark_scores",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "model_id": model_ids,
            "profile_hash": ["hash-a", "hash-b", None],
            "blind_fold": [0, 1, None],
            "predicted_cluster": [4, 5, None],
            "spectral_medoid_model": ["org/a", "org/b", None],
            "nearest_neighbors": ["org/b", "org/a", ""],
            "nearest_similarity": [0.9, 0.9, None],
            "htsr_profile_available": [True, True, False],
        }
    ).to_csv(run / "population_ledger.csv", index=False)
    pd.DataFrame(
        {
            "model_id": model_ids[:2],
            "predicted_cluster": [4, 5],
            "spectral_medoid_model": ["org/a", "org/b"],
            "nearest_neighbors": ["org/b", "org/a"],
        }
    ).to_csv(run / "lineage_cluster_assignments.csv", index=False)
    pd.DataFrame(
        [
            {
                "method": "minibatch_kmeans_rank_profile",
                "n_models": 2,
                "n_clusters": 2,
                "silhouette_cosine": 0.61,
                "mean_seed_stability_ari": 0.94,
                "metadata_loaded_before_freeze": False,
            }
        ]
    ).to_csv(run / "lineage_internal_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "n_clusters": 2,
                "silhouette_cosine": 0.61,
                "selection_uses_metadata": False,
                "selected": True,
            }
        ]
    ).to_csv(run / "lineage_cluster_selection.csv", index=False)
    pd.DataFrame(
        [
            {
                "label_source": "manual_confirmed",
                "n_models": 2,
                "n_families": 2,
                "adjusted_rand": 1.0,
                "normalized_mutual_information": 1.0,
                "evidence_role": "post_freeze_metadata_validation",
            }
        ]
    ).to_csv(run / "lineage_validation_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "method": "supplied_gallery_interpolated_spearman_dks",
                "n_scored_queries": 2,
                "n_supplied_roots": 2,
                "n_profiled_roots": 2,
                "root_macro_top1": 1.0,
                "root_macro_top3": 1.0,
                "evidence_role": "post_freeze_metadata_validation",
            }
        ]
    ).to_csv(run / "lineage_gallery_metrics.csv", index=False)

    prediction_rows = [
        {
            "generation": "modern",
            "model_id": model_id,
            "profile_hash": f"hash-{index}",
            "blind_fold": index,
            "target": "modern_average",
            "feature_block": LEADING_PERFORMANCE_METHOD,
            "prediction": actual + 0.25,
            "nearest_neighbors": "org/a",
            "nearest_similarity": 0.99,
            "actual": actual,
        }
        for index, (model_id, actual) in enumerate(zip(model_ids[:2], [20.0, 21.0]))
    ]
    prediction_rows.append(
        {
            "generation": "legacy",
            "model_id": "org/a",
            "profile_hash": "hash-a",
            "blind_fold": 0,
            "target": "legacy_average",
            "feature_block": LEADING_PERFORMANCE_METHOD,
            "prediction": 54.0,
            "nearest_neighbors": "org/b",
            "nearest_similarity": 0.98,
            "actual": 55.0,
        }
    )
    pd.DataFrame(prediction_rows).to_csv(run / "performance_predictions.csv", index=False)
    pd.DataFrame(_metric_rows("blind_profile_group_5fold")).to_csv(
        run / "performance_metrics.csv", index=False
    )
    pd.DataFrame(_metric_rows("blind_spectral_cluster_holdout")).assign(
        model_weighted_mae=lambda frame: frame["model_weighted_mae"] + 2.0
    ).to_csv(run / "performance_cluster_holdout_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "generation": "modern",
                "target": "modern_average",
                "repeats": 100,
                "n_models": 2,
                "mean_mae": 0.3,
                "mean_spearman": 1.0,
            }
        ]
    ).to_csv(run / "performance_paper_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "generation": "modern",
                "n_models": 2,
                "exact_profiles_crossing_folds": 0,
            }
        ]
    ).to_csv(run / "performance_leakage_audit.csv", index=False)
    pd.DataFrame(
        [
            {
                "benchmark_generation": "modern",
                "target_resolution": "selected_source_row",
                "included": True,
                "n_models": 2,
            }
        ]
    ).to_csv(run / "performance_target_audit.csv", index=False)
    (run / "report.md").write_text("# Weight-only report\n", encoding="utf-8")
    write_artifact_manifest(run, run_id=run.name)
    return run


def test_scaleup_loader_exposes_weight_only_predictions_and_metrics(tmp_path: Path) -> None:
    loaded = load_scaleup_outputs(_write_scaleup_run(tmp_path))

    assert loaded.population["predicted_cluster"].tolist() == [4, 5, pd.NA]
    assert loaded.population["spectral_medoid_model"].tolist()[:2] == ["org/a", "org/b"]
    assert set(loaded.predictions) == {"modern_average", "legacy_average"}
    assert loaded.predictions["modern_average"]["prediction"].tolist() == [20.25, 21.25]
    assert loaded.performance["modern_average"]["model_weighted_mae"] == pytest.approx(0.25)
    assert loaded.performance["modern_average"]["delta_mae"] == pytest.approx(-0.75)
    assert loaded.performance["modern_average"]["cluster_holdout_mae"] == pytest.approx(2.25)
    assert loaded.counts["profiled_models"] == 2


def test_scaleup_loader_rejects_run_id_mismatch(tmp_path: Path) -> None:
    run = _write_scaleup_run(tmp_path)
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    metadata["run_id"] = "different-run"
    (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="run_id"):
        load_scaleup_outputs(run)


def test_scaleup_loader_rejects_pre_freeze_metadata_use(tmp_path: Path) -> None:
    run = _write_scaleup_run(tmp_path)
    internal = pd.read_csv(run / "lineage_internal_metrics.csv")
    internal["metadata_loaded_before_freeze"] = True
    internal.to_csv(run / "lineage_internal_metrics.csv", index=False)
    write_artifact_manifest(run, run_id=run.name)

    with pytest.raises(ValueError, match="metadata_loaded_before_freeze=False"):
        load_scaleup_outputs(run)
