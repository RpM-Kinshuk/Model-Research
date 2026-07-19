from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from alpha_cluster_dashboard.scaleup_outputs import (
    LEADING_PERFORMANCE_METHOD,
    load_scaleup_outputs,
)


def _write_scaleup_run(root: Path) -> Path:
    run = root / "scaleup_test"
    run.mkdir()
    model_ids = ["org/a", "org/b", "org/c"]
    (run / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run.name,
                "counts": {
                    "phase2_census": 3,
                    "phase2_htsr_artifacts": 2,
                    "phase2_strict_lineage": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "model_id": model_ids,
            "htsr_artifact_available": [True, True, False],
            "strict_cluster": [0, 1, None],
            "broad_cluster": [0, 1, None],
            "predicted_family": ["root/a", "root/b", None],
            "lineage_evidence_status": [
                "strict_qc_clustered",
                "strict_qc_clustered",
                "missing_htsr_artifact",
            ],
        }
    ).to_csv(run / "population_ledger.csv", index=False)

    modern_rows = []
    for method, offset in (("M+S+H", 1.0), (LEADING_PERFORMANCE_METHOD, 0.25)):
        for index, model_id in enumerate(model_ids[:2]):
            modern_rows.append(
                {
                    "model_id": model_id,
                    "ancestry_component": f"component-{index}",
                    "target": "modern_average",
                    "feature_block": method,
                    "outer_fold": index,
                    "actual": 20.0 + index,
                    "prediction": 20.0 + index + offset,
                }
            )
    pd.DataFrame(modern_rows).to_csv(run / "modern_pooled_predictions.csv", index=False)
    pd.DataFrame(
        [
            {
                "target": "modern_average",
                "feature_block": LEADING_PERFORMANCE_METHOD,
                "n_models": 2,
                "n_components": 2,
                "component_weighted_mae": 0.25,
                "model_weighted_mae": 0.25,
                "spearman": 1.0,
                "kendall_tau": 1.0,
            }
        ]
    ).to_csv(run / "modern_pooled_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "target": "modern_average",
                "baseline": "M+S+H",
                "candidate": LEADING_PERFORMANCE_METHOD,
                "n_models": 2,
                "n_components": 2,
                "delta_mae": -0.75,
                "ci_low": -1.0,
                "ci_high": -0.5,
                "p_value_raw": 0.01,
            }
        ]
    ).to_csv(run / "modern_pooled_increments.csv", index=False)

    pd.DataFrame(
        [
            {
                "model_id": "org/a",
                "ancestry_component": "component-0",
                "target": "legacy_average",
                "feature_block": LEADING_PERFORMANCE_METHOD,
                "outer_fold": 0,
                "actual": 55.0,
                "prediction": 54.0,
            }
        ]
    ).to_csv(run / "legacy_paired_predictions.csv", index=False)
    pd.DataFrame(
        [
            {
                "target": "legacy_average",
                "feature_block": LEADING_PERFORMANCE_METHOD,
                "n_models": 1,
                "n_components": 1,
                "component_weighted_mae": 1.0,
                "model_weighted_mae": 1.0,
                "spearman": None,
                "kendall_tau": None,
            }
        ]
    ).to_csv(run / "legacy_paired_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "target": "legacy_average",
                "baseline": "M+S+H",
                "candidate": LEADING_PERFORMANCE_METHOD,
                "n_models": 1,
                "n_components": 1,
                "delta_mae": -0.5,
                "ci_low": -0.8,
                "ci_high": -0.2,
                "p_value_raw": 0.02,
            }
        ]
    ).to_csv(run / "legacy_increments.csv", index=False)
    pd.DataFrame(
        [{"method": "broad", "n_clusters": 2, "n_clustered": 2}]
    ).to_csv(run / "lineage_broad_cluster_metrics.csv", index=False)
    (run / "report.md").write_text("# Scale-up report\n", encoding="utf-8")
    return run


def test_scaleup_loader_exposes_leading_predictions_and_metrics(tmp_path: Path) -> None:
    loaded = load_scaleup_outputs(_write_scaleup_run(tmp_path))

    assert len(loaded.population) == 3
    assert set(loaded.predictions) == {"modern_average", "legacy_average"}
    assert loaded.predictions["modern_average"]["prediction"].tolist() == [20.25, 21.25]
    assert loaded.performance["modern_average"]["group_weighted_mae"] == pytest.approx(0.25)
    assert loaded.performance["modern_average"]["delta_mae"] == pytest.approx(-0.75)
    assert loaded.counts["phase2_census"] == 3


def test_scaleup_loader_rejects_run_id_mismatch(tmp_path: Path) -> None:
    run = _write_scaleup_run(tmp_path)
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    metadata["run_id"] = "different-run"
    (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="run_id"):
        load_scaleup_outputs(run)
