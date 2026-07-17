from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spectral_research.study.features import (
    ALPHA_ESTIMATOR,
    ALPHA_ESTIMATOR_VERSION,
    CANONICAL_ROLES,
    FEATURE_SCHEMA_VERSION,
    canonical_role,
    clean_alpha,
    confirmatory_schema_population,
    feature_columns,
    extract_feature_table,
    extract_alpha_trajectories,
    extract_study_tables,
    index_stats_directory,
    layer_quantities,
    load_stats_csv,
    summarize_model_features,
    summarize_trajectory,
)


def _stats_row(
    model_id: str,
    longname: str,
    *,
    alpha: float = 2.5,
    dks_distance: float = 0.03,
    matrix_rows: int = 4,
    matrix_cols: int = 5,
    frobenius_sq: float = 400.0,
    spectral_sq: float = 100.0,
    entropy: float = 0.75,
    rank: int = 4,
    log_alpha_norm: float = 1.25,
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "longname": longname,
        "alpha": alpha,
        "D": dks_distance,
        "M": matrix_rows,
        "N": matrix_cols,
        "norm": frobenius_sq,
        "spectral_norm": spectral_sq,
        "entropy": entropy,
        "matrix_rank": rank,
        "log_alpha_norm": log_alpha_norm,
        # These historical derived fields must not override corrected formulas.
        "params": 999_999,
        "log_norm": -999.0,
        "log_spectral_norm": -999.0,
        "stable_rank": -999.0,
        "alpha_weighted": -999.0,
    }


def _confirmatory_feature_row(
    model_id: str, **overrides: object
) -> dict[str, object]:
    row: dict[str, object] = {
        "model_id": model_id,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "M__canonical_role_count": len(CANONICAL_ROLES),
        "QC__canonical_schema_complete": True,
        "QC__fused_role_row_count": 0,
        "QC__unknown_role_row_count": 0,
    }
    for role in CANONICAL_ROLES:
        row[f"M__{role}__present"] = 1.0
        row[f"M__{role}__coverage"] = 1.0
    row.update(overrides)
    return row


def test_layer_quantities_use_squared_norm_semantics() -> None:
    row = layer_quantities(
        alpha=2.5,
        dks_distance=0.03,
        frobenius_sq=400.0,
        spectral_sq=100.0,
        matrix_rows=4,
        matrix_cols=5,
        entropy=1.2,
        rank=4,
        log_alpha_norm=1.25,
    )

    assert row["log_rms"] == pytest.approx(0.5 * np.log10(20.0))
    assert row["log_spectral_norm"] == pytest.approx(2.0)
    assert row["normalized_entropy"] == pytest.approx(1.2)
    assert row["stable_rank_fraction"] == pytest.approx(1.0)
    assert row["alpha_weighted"] == pytest.approx(5.0)


def test_stable_rank_fraction_uses_observed_hard_rank() -> None:
    row = layer_quantities(
        alpha=2.0,
        dks_distance=0.03,
        frobenius_sq=200.0,
        spectral_sq=100.0,
        matrix_rows=8,
        matrix_cols=8,
        entropy=0.8,
        rank=4,
    )

    assert row["stable_rank_fraction"] == pytest.approx(0.5)


@pytest.mark.parametrize("alpha", [-1.0, 0.0, np.nan, np.inf, -np.inf])
def test_invalid_alpha_is_missing(alpha: float) -> None:
    assert np.isnan(clean_alpha(alpha))


def test_invalid_squared_norm_inputs_only_invalidate_dependent_quantities() -> None:
    row = layer_quantities(
        alpha=2.0,
        dks_distance=-1.0,
        frobenius_sq=400.0,
        spectral_sq=0.0,
        matrix_rows=4,
        matrix_cols=5,
        entropy=np.inf,
        rank=0,
        log_alpha_norm=3.0,
    )

    assert np.isfinite(row["log_rms"])
    assert np.isnan(row["log_spectral_norm"])
    assert np.isnan(row["stable_rank_fraction"])
    assert np.isnan(row["alpha_weighted"])
    assert np.isnan(row["normalized_entropy"])
    assert np.isnan(row["dks_distance"])


def test_invalid_alpha_masks_all_alpha_dependent_quantities() -> None:
    row = layer_quantities(
        alpha=-1.0,
        dks_distance=0.03,
        frobenius_sq=400.0,
        spectral_sq=100.0,
        matrix_rows=4,
        matrix_cols=5,
        entropy=0.75,
        rank=4,
        log_alpha_norm=7.0,
    )

    assert np.isnan(row["alpha"])
    assert np.isnan(row["alpha_weighted"])
    assert np.isnan(row["log_alpha_norm"])
    assert row["dks_distance"] == pytest.approx(0.03)


@pytest.mark.parametrize(
    ("longname", "expected"),
    [
        ("model.layers.0.self_attn.q_proj", "attn_q"),
        ("model.layers.0.self_attn.k_proj", "attn_k"),
        ("model.layers.0.self_attn.v_proj", "attn_v"),
        ("model.layers.0.self_attn.o_proj", "attn_o"),
        ("model.layers.0.mlp.gate_proj", "mlp_gate"),
        ("model.layers.0.mlp.up_proj", "mlp_up"),
        ("model.layers.0.mlp.down_proj", "mlp_down"),
        ("transformer.h.3.attention.self.query", "attn_q"),
        ("decoder.layers.2.feed_forward.w1", "mlp_gate"),
        ("decoder.layers.2.feed_forward.w3", "mlp_up"),
        ("decoder.layers.2.feed_forward.w2", "mlp_down"),
    ],
)
def test_canonical_role_recognizes_only_unfused_confirmatory_roles(
    longname: str, expected: str
) -> None:
    assert canonical_role(longname) == expected


@pytest.mark.parametrize(
    "longname",
    [
        "model.layers.0.self_attn.qkv_proj",
        "model.layers.0.attention.query_key_value",
        "model.layers.0.mlp.gate_up_proj",
        "model.layers.0.block_sparse_moe.experts.0.w1",
        "model.layers.0.mamba.in_proj",
        "model.layers.0.cross_attn.q_proj",
        "model.layers.0.unrelated.q_proj_backup",
    ],
)
def test_canonical_role_rejects_fused_moe_cross_attention_and_unknown_schemas(
    longname: str,
) -> None:
    assert canonical_role(longname) is None


def test_normalized_depth_slope_is_invariant_to_layer_count() -> None:
    short = summarize_trajectory([1, 2, 3], [0, 1, 2])
    long = summarize_trajectory([1, 1.5, 2, 2.5, 3], [0, 1, 2, 3, 4])

    assert short["slope"] == pytest.approx(long["slope"])
    assert short["slope"] == pytest.approx(2.0)


def test_trajectory_summary_reports_iqr_and_missing_fraction() -> None:
    summary = summarize_trajectory([1.0, np.nan, 3.0], [0, 1, 2])

    assert summary["median"] == pytest.approx(2.0)
    assert summary["iqr"] == pytest.approx(1.0)
    assert summary["slope"] == pytest.approx(2.0)
    assert summary["missing_fraction"] == pytest.approx(1.0 / 3.0)


def test_trajectory_slope_requires_two_distinct_finite_depths() -> None:
    one_value = summarize_trajectory([np.nan, 2.0], [0, 1])
    one_depth = summarize_trajectory([1.0, 2.0], [3, 3])

    assert np.isnan(one_value["slope"])
    assert np.isnan(one_depth["slope"])


def test_load_stats_csv_selects_only_the_explicit_model_id(tmp_path: Path) -> None:
    path = tmp_path / "misleading--filename.csv"
    pd.DataFrame(
        [
            _stats_row("org/other", "model.layers.0.self_attn.q_proj"),
            _stats_row("org/target", "model.layers.0.self_attn.q_proj"),
            _stats_row("org/target", "model.layers.1.self_attn.q_proj"),
        ]
    ).to_csv(path, index=False)

    loaded = load_stats_csv(path, model_id="org/target", chunksize=1)

    assert loaded["model_id"].tolist() == ["org/target", "org/target"]


def test_load_stats_csv_never_infers_identity_from_the_filename(tmp_path: Path) -> None:
    path = tmp_path / "org--target.csv"
    pd.DataFrame(
        [_stats_row("org/other", "model.layers.0.self_attn.q_proj")]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="org/target"):
        load_stats_csv(path, model_id="org/target")


def test_load_stats_csv_requires_a_model_id_column(tmp_path: Path) -> None:
    path = tmp_path / "stats.csv"
    pd.DataFrame([{"longname": "model.layers.0.self_attn.q_proj"}]).to_csv(
        path, index=False
    )

    with pytest.raises(ValueError, match="model_id"):
        load_stats_csv(path, model_id="org/target")


def test_stats_directory_index_uses_internal_ids_not_filenames(tmp_path: Path) -> None:
    pd.DataFrame(
        [_stats_row("org/a", "model.layers.0.self_attn.q_proj")]
    ).to_csv(tmp_path / "misleading-name.csv", index=False)
    pd.DataFrame(
        [_stats_row("org/b", "model.layers.0.self_attn.q_proj")]
    ).to_csv(tmp_path / "another.csv", index=False)

    index, exclusions = index_stats_directory(tmp_path)

    assert index["model_id"].tolist() == ["org/a", "org/b"]
    assert index.set_index("model_id").loc["org/a", "path"].endswith(
        "misleading-name.csv"
    )
    assert exclusions.empty


def test_stats_directory_index_rejects_duplicate_internal_ids(tmp_path: Path) -> None:
    row = _stats_row("org/a", "model.layers.0.self_attn.q_proj")
    pd.DataFrame([row]).to_csv(tmp_path / "first.csv", index=False)
    pd.DataFrame([row]).to_csv(tmp_path / "second.csv", index=False)

    with pytest.raises(ValueError, match="duplicate stats artifacts.*org/a"):
        index_stats_directory(tmp_path)


def test_feature_table_records_missing_and_invalid_artifacts(tmp_path: Path) -> None:
    pd.DataFrame(
        [_stats_row("org/a", "model.layers.0.self_attn.q_proj")]
    ).to_csv(tmp_path / "a.csv", index=False)
    (tmp_path / "broken.csv").write_text("not_model_id\nvalue\n", encoding="utf-8")
    index, index_exclusions = index_stats_directory(tmp_path)

    features, exclusions = extract_feature_table(index, ["org/a", "org/missing"])

    assert features["model_id"].tolist() == ["org/a"]
    assert index_exclusions["reason"].tolist() == ["invalid_stats_header"]
    assert exclusions.loc[0, "model_id"] == "org/missing"
    assert exclusions.loc[0, "reason"] == "missing_stats_artifact"


def test_parallel_feature_extraction_is_deterministic(tmp_path: Path) -> None:
    for model_id in ("org/a", "org/b", "org/c"):
        pd.DataFrame(
            [_stats_row(model_id, "model.layers.0.self_attn.q_proj")]
        ).to_csv(tmp_path / f"{model_id[-1]}.csv", index=False)
    index, _ = index_stats_directory(tmp_path)

    serial, serial_exclusions = extract_feature_table(
        index, ["org/c", "org/a", "org/b"], workers=1
    )
    parallel, parallel_exclusions = extract_feature_table(
        index, ["org/c", "org/a", "org/b"], workers=3
    )

    pd.testing.assert_frame_equal(serial, parallel)
    pd.testing.assert_frame_equal(serial_exclusions, parallel_exclusions)


def test_combined_study_extraction_reads_each_artifact_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for model_id in ("org/a", "org/b"):
        pd.DataFrame(
            [_stats_row(model_id, "model.layers.0.self_attn.q_proj")]
        ).to_csv(tmp_path / f"{model_id[-1]}.csv", index=False)
    index, _ = index_stats_directory(tmp_path)
    original_read_csv = pd.read_csv
    reads: list[Path] = []

    def counted_read_csv(path, *args, **kwargs):
        reads.append(Path(path))
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", counted_read_csv)

    features, trajectories, feature_exclusions, trajectory_exclusions = (
        extract_study_tables(index, ["org/b", "org/a"], workers=1)
    )

    assert len(reads) == 2
    assert features["model_id"].tolist() == ["org/a", "org/b"]
    assert trajectories["model_id"].tolist() == ["org/b", "org/a"]
    assert feature_exclusions.empty
    assert trajectory_exclusions.empty


def test_alpha_trajectory_extraction_masks_sentinels_and_aggregates_duplicates(
    tmp_path: Path,
) -> None:
    rows = [
        _stats_row(
            "org/a", "model.layers.0.self_attn.q_proj", alpha=2.0
        ),
        _stats_row(
            "org/a", "model.layers.0.self_attn.q_proj", alpha=4.0
        ),
        _stats_row(
            "org/a", "model.layers.1.self_attn.q_proj", alpha=-1.0
        ),
        _stats_row(
            "org/a", "model.layers.0.self_attn.k_proj", alpha=5.0
        ),
        _stats_row("org/a", "model.layers.0.self_attn.qkv_proj", alpha=9.0),
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "a.csv", index=False)
    index, _ = index_stats_directory(tmp_path)

    trajectories, exclusions = extract_alpha_trajectories(index, ["org/a"])

    assert exclusions.empty
    assert trajectories[["role", "depth"]].to_records(index=False).tolist() == [
        ("attn_q", 0.0),
        ("attn_q", 1.0),
        ("attn_k", 0.0),
    ]
    query = trajectories.loc[trajectories["role"].eq("attn_q")].sort_values("depth")
    assert query["alpha"].iloc[0] == pytest.approx(3.0)
    assert np.isnan(query["alpha"].iloc[1])
    assert set(trajectories["alpha_estimator"]) == {ALPHA_ESTIMATOR}
    assert set(trajectories["alpha_estimator_version"]) == {
        ALPHA_ESTIMATOR_VERSION
    }


def test_alpha_trajectory_extraction_records_missing_artifacts(tmp_path: Path) -> None:
    index = pd.DataFrame(columns=["model_id", "path"])

    trajectories, exclusions = extract_alpha_trajectories(
        index, ["org/missing"], workers=1
    )

    assert trajectories.empty
    assert exclusions.loc[0, "reason"] == "missing_stats_artifact"


def test_feature_order_is_deterministic_and_d_excluded() -> None:
    columns = feature_columns()

    assert FEATURE_SCHEMA_VERSION == "htsr-study-features-v2"
    assert CANONICAL_ROLES == (
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_o",
        "mlp_gate",
        "mlp_up",
        "mlp_down",
    )
    assert columns[:6] == (
        "M__layer_count",
        "M__canonical_role_count",
        "M__canonical_role_fraction",
        "M__log10_canonical_matrix_parameter_count",
        "M__attn_q__present",
        "M__attn_q__coverage",
    )
    assert feature_columns(("H", "S")) == tuple(
        column for column in columns if column.startswith(("S__", "H__"))
    )
    assert not any("dks" in column.lower() or "__d__" in column.lower() for column in columns)


def test_confirmatory_schema_population_returns_eligible_rows_and_explicit_exclusions() -> None:
    rows = [
        _confirmatory_feature_row(
            "org/eligible-with-unknowns", QC__unknown_role_row_count=3
        ),
        _confirmatory_feature_row(
            "org/legacy-schema",
            feature_schema_version="htsr-study-features-v1",
        ),
        _confirmatory_feature_row(
            "org/missing-role",
            M__canonical_role_count=6,
            QC__canonical_schema_complete=False,
            M__mlp_down__present=0.0,
            M__mlp_down__coverage=0.0,
        ),
        _confirmatory_feature_row(
            "org/fused", QC__fused_role_row_count=1
        ),
        _confirmatory_feature_row(
            "org/shallow-role", M__attn_k__coverage=0.74
        ),
    ]

    eligible, exclusions = confirmatory_schema_population(pd.DataFrame(rows), 0.75)

    assert eligible["model_id"].tolist() == ["org/eligible-with-unknowns"]
    assert exclusions["model_id"].tolist() == [
        "org/legacy-schema",
        "org/missing-role",
        "org/fused",
        "org/shallow-role",
    ]
    assert exclusions["reason"].tolist() == [
        "incompatible_feature_schema",
        "incomplete_canonical_roles",
        "non_confirmatory_role_rows",
        "insufficient_role_coverage",
    ]
    details = exclusions.set_index("model_id")["detail"]
    assert FEATURE_SCHEMA_VERSION in details["org/legacy-schema"]
    assert "mlp_down" in details["org/missing-role"]
    assert "attn_k" in details["org/shallow-role"]


@pytest.mark.parametrize("min_role_coverage", [-0.01, 1.01, np.nan, np.inf])
def test_confirmatory_schema_population_validates_coverage_threshold(
    min_role_coverage: float,
) -> None:
    with pytest.raises(ValueError, match="min_role_coverage"):
        confirmatory_schema_population(
            pd.DataFrame([_confirmatory_feature_row("org/model")]),
            min_role_coverage,
        )


def test_confirmatory_schema_population_requires_complete_contract_columns() -> None:
    features = pd.DataFrame([_confirmatory_feature_row("org/model")]).drop(
        columns="M__attn_v__coverage"
    )

    with pytest.raises(KeyError, match="M__attn_v__coverage"):
        confirmatory_schema_population(features, 0.75)


def test_parameter_scale_uses_only_recognized_layered_matrix_dimensions() -> None:
    rows = [
        _stats_row(
            "org/model",
            "model.layers.0.self_attn.q_proj",
            matrix_rows=4,
            matrix_cols=5,
        ),
        _stats_row(
            "org/model",
            "model.layers.0.self_attn.k_proj",
            matrix_rows=3,
            matrix_cols=7,
        ),
        _stats_row(
            "org/model",
            "model.layers.0.self_attn.qkv_proj",
            matrix_rows=1_000,
            matrix_cols=1_000,
        ),
        _stats_row(
            "org/model",
            "model.layers.0.rotary_emb",
            matrix_rows=1_000,
            matrix_cols=1_000,
        ),
        _stats_row(
            "org/model",
            "model.self_attn.v_proj",
            matrix_rows=1_000,
            matrix_cols=1_000,
        ),
    ]

    record = summarize_model_features(pd.DataFrame(rows), model_id="org/model")

    assert record["M__log10_canonical_matrix_parameter_count"] == pytest.approx(
        np.log10(4 * 5 + 3 * 7)
    )


def test_parameter_scale_is_missing_when_any_canonical_matrix_size_is_invalid() -> None:
    rows = [
        _stats_row("org/model", "model.layers.0.self_attn.q_proj"),
        _stats_row(
            "org/model",
            "model.layers.0.self_attn.k_proj",
            matrix_rows=0,
        ),
    ]

    record = summarize_model_features(pd.DataFrame(rows), model_id="org/model")

    assert np.isnan(record["M__log10_canonical_matrix_parameter_count"])


def test_model_summary_recomputes_features_and_emits_role_qc() -> None:
    rows = []
    role_suffixes = {
        "attn_q": "self_attn.q_proj",
        "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj",
        "attn_o": "self_attn.o_proj",
        "mlp_gate": "mlp.gate_proj",
        "mlp_up": "mlp.up_proj",
        "mlp_down": "mlp.down_proj",
    }
    for suffix in role_suffixes.values():
        rows.append(_stats_row("org/model", f"model.layers.0.{suffix}"))
    rows.extend(
        [
            _stats_row(
                "org/model",
                "model.layers.1.self_attn.q_proj",
                alpha=-1.0,
                dks_distance=0.20,
                frobenius_sq=800.0,
            ),
            _stats_row("org/model", "model.layers.1.self_attn.qkv_proj"),
            _stats_row("org/model", "model.layers.1.rotary_emb"),
        ]
    )

    record = summarize_model_features(pd.DataFrame(rows), model_id="org/model")
    columns = feature_columns()

    assert record["model_id"] == "org/model"
    assert record["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert list(record)[2 : 2 + len(columns)] == list(columns)
    assert record["M__layer_count"] == 2
    assert record["M__canonical_role_count"] == 7
    assert record["M__canonical_role_fraction"] == pytest.approx(1.0)
    assert record["M__attn_q__coverage"] == pytest.approx(1.0)
    assert record["M__attn_k__coverage"] == pytest.approx(0.5)
    assert record["H__attn_q__alpha__missing_fraction"] == pytest.approx(0.5)
    assert record["S__attn_q__log_rms__median"] == pytest.approx(
        np.median([0.5 * np.log10(20.0), 0.5 * np.log10(40.0)])
    )
    assert record["QC__canonical_schema_complete"] is True
    assert record["QC__fused_role_row_count"] == 1
    assert record["QC__unknown_role_row_count"] == 1
    assert record["QC__attn_q__role_coverage"] == pytest.approx(1.0)
    assert record["QC__attn_q__dks_median"] == pytest.approx(0.115)
    assert record["QC__attn_q__dks_le_0_1_fraction"] == pytest.approx(0.5)
    assert not any(key.startswith(("M__", "S__", "H__")) and "dks" in key for key in record)


def test_role_coverage_uses_all_parseable_model_depths() -> None:
    stats = pd.DataFrame(
        [
            _stats_row("org/model", "model.layers.0.self_attn.q_proj"),
            _stats_row("org/model", "model.layers.1.rotary_emb"),
        ]
    )

    record = summarize_model_features(stats, model_id="org/model")

    assert record["M__layer_count"] == 2
    assert record["M__attn_q__coverage"] == pytest.approx(0.5)
    assert record["H__attn_q__alpha__missing_fraction"] == pytest.approx(0.5)
