from pathlib import Path

import numpy as np
import pandas as pd

from alpha_cluster_dashboard.data import AlphaRecord
from alpha_cluster_dashboard.dashboard_cache import (
    DEFAULT_ABLATION_METADATA,
    DEFAULT_CACHE_DIR,
    DashboardCacheConfig,
    build_cache_from_records,
    cache_is_valid,
    infer_prediction_targets,
    load_dashboard_cache,
    source_fingerprint,
    stable_distance_weights,
)
from alpha_cluster_dashboard.ablation import DEFAULT_ABLATION_METRICS_DIR


def test_dashboard_defaults_use_full_results_dataset():
    assert DEFAULT_ABLATION_METRICS_DIR.parts[-3:] == ("output", "results", "metrics")
    assert DEFAULT_ABLATION_METADATA.parts[-3:] == ("data", "curated", "model_zoo_phase2.csv")
    assert str(DEFAULT_CACHE_DIR).endswith("/research_runs/preliminary/results/dashboard_cache")


def _record(model_id: str, offset: float, family_scale: float = 1.0) -> AlphaRecord:
    alpha = np.array(
        [
            [1.0 + offset, 2.0 + offset, 3.0 + offset, 4.0 + offset],
            [2.0 + offset, 3.0 + offset, 4.0 + offset, 5.0 + offset],
            [3.0 + offset, 4.0 + offset, 5.0 + offset, 6.0 + offset],
        ],
        dtype=float,
    )
    return AlphaRecord(
        model_id=model_id.replace("/", "--"),
        display_name=model_id,
        path=f"/tmp/{model_id.replace('/', '--')}.h5",
        num_layers=3,
        raw_modules=("self_attn.q_proj", "self_attn.k_proj", "mlp.up_proj", "mlp.down_proj"),
        alpha=alpha * family_scale,
    )


def test_build_cache_from_records_persists_metadata_predictions_and_profiles(tmp_path: Path):
    records = [
        _record("org/a", 0.0),
        _record("org/b", 0.2),
        _record("org/c", 5.0),
        _record("org/d", 5.2),
    ]
    metadata = pd.DataFrame(
        {
            "model_id": ["org/a", "org/b", "org/c", "org/d"],
            "coarse_family": ["fam1", "fam1", "fam2", "fam2"],
            "Architecture": ["arch1", "arch1", "arch2", "arch2"],
            "Average \u2b06\ufe0f": [10.0, 12.0, 50.0, 52.0],
            "Custom Eval": [0.10, np.nan, 0.90, 0.95],
        }
    )
    (tmp_path / "predictions_Stale.csv").write_text("old", encoding="utf-8")

    cache = build_cache_from_records(
        records,
        metadata,
        tmp_path,
        config=DashboardCacheConfig(n_clusters=2, n_neighbors=1, n_bins=8),
        fingerprint={"source": "unit"},
    )

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "model_atlas.csv").exists()
    assert (tmp_path / "features.npz").exists()
    assert not (tmp_path / "predictions_Stale.csv").exists()
    assert (tmp_path / "predictions_Average.csv").exists()
    assert (tmp_path / "predictions_Custom_Eval.csv").exists()
    assert (tmp_path / "cluster_profiles.csv").exists()
    assert cache.metadata.shape[0] == 4
    assert {"embedding_x", "embedding_y", "embedding_z", "cluster"}.issubset(cache.metadata.columns)
    assert cache.predictions["Average \u2b06\ufe0f"]["absolute_error"].mean() == 2.0
    assert "Custom Eval" in cache.predictions
    custom = cache.predictions["Custom Eval"].sort_values("display_name")
    assert len(custom) == len(records)
    assert custom["prediction"].notna().all()
    assert custom["actual"].notna().sum() == 3
    assert custom.loc[custom["display_name"].eq("org/b"), "absolute_error"].isna().all()
    assert set(cache.cluster_profiles["role"]) == {"attn_k", "attn_q", "mlp_down", "mlp_up"}

    loaded = load_dashboard_cache(tmp_path)
    assert list(loaded.predictions)[:2] == ["Average \u2b06\ufe0f", "Custom Eval"]


def test_prediction_targets_are_inferred_from_data_columns():
    metadata = pd.DataFrame(
        {
            "model_id": ["a", "b", "c", "d"],
            "display_name": ["a", "b", "c", "d"],
            "Custom Eval": [0.1, 0.2, 0.8, 0.9],
            "Average \u2b06\ufe0f": [10.0, 12.0, 50.0, 52.0],
            "#Params (B)": [7.0, 7.0, 13.0, 13.0],
            "downloads": [100, 200, 300, 400],
            "raw_module_count": [4, 4, 4, 4],
            "mostly_missing": [np.nan, np.nan, np.nan, 1.0],
        }
    )

    targets = infer_prediction_targets(metadata)

    assert "Custom Eval" in targets
    assert "Average \u2b06\ufe0f" in targets
    assert "#Params (B)" not in targets
    assert "downloads" not in targets
    assert "raw_module_count" not in targets
    assert "mostly_missing" not in targets


def test_load_dashboard_cache_round_trips_outputs(tmp_path: Path):
    records = [_record("org/a", 0.0), _record("org/b", 0.2), _record("org/c", 5.0)]
    metadata = pd.DataFrame(
        {
            "model_id": ["org/a", "org/b", "org/c"],
            "coarse_family": ["fam1", "fam1", "fam2"],
            "Average \u2b06\ufe0f": [10.0, 12.0, 50.0],
        }
    )
    build_cache_from_records(
        records,
        metadata,
        tmp_path,
        config=DashboardCacheConfig(n_clusters=2, n_neighbors=1, n_bins=6),
        fingerprint={"source": "unit"},
    )

    loaded = load_dashboard_cache(tmp_path)

    assert loaded.metadata["display_name"].tolist() == ["org/a", "org/b", "org/c"]
    assert "Average \u2b06\ufe0f" in loaded.predictions
    assert loaded.features.shape[0] == 3


def test_source_fingerprint_changes_when_input_file_changes(tmp_path: Path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    first = metrics_dir / "a.h5"
    first.write_text("one")
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text("model_id\norg/a\n")

    before = source_fingerprint(metrics_dir, metadata_path, settings={"n_bins": 8})
    first.write_text("changed")
    after = source_fingerprint(metrics_dir, metadata_path, settings={"n_bins": 8})

    assert before != after
    assert not cache_is_valid(tmp_path, after)


def test_build_cache_handles_large_feature_distances_without_weight_underflow(tmp_path: Path):
    records = [
        _record("org/a", 0.0, family_scale=1.0),
        _record("org/b", 0.1, family_scale=1.0),
        _record("org/c", 1000.0, family_scale=100.0),
        _record("org/d", 1100.0, family_scale=100.0),
    ]
    metadata = pd.DataFrame(
        {
            "model_id": ["org/a", "org/b", "org/c", "org/d"],
            "Average \u2b06\ufe0f": [10.0, 12.0, 50.0, 52.0],
        }
    )

    cache = build_cache_from_records(
        records,
        metadata,
        tmp_path,
        config=DashboardCacheConfig(n_clusters=2, n_neighbors=3, n_bins=8),
        fingerprint={"source": "unit"},
    )

    predictions = cache.predictions["Average \u2b06\ufe0f"]
    assert predictions["prediction"].notna().all()


def test_stable_distance_weights_do_not_underflow_for_large_distances():
    weights = stable_distance_weights(np.array([80.0, 81.0, 82.0]))

    assert np.isfinite(weights).all()
    assert weights.sum() > 0.0
    assert weights[0] > weights[1] > weights[2]
