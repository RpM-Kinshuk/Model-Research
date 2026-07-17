from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spectral_research.study.features import FEATURE_SCHEMA_VERSION
from spectral_research.study.performance import (
    MODERN_SEMANTICS,
    TargetSemanticsError,
    WeightedMedianImputer,
    _fit_weighted_pipeline,
    assess_htsr_increment,
    build_modern_targets,
    build_ridge_pipeline,
    component_sample_weights,
    confirmatory_group_population,
    evaluate_external_transfer,
    evaluate_grouped_ridge,
    evaluate_secondary_endpoints,
    external_component_slices,
    group_weighted_mae,
    grouped_outer_folds,
    holm_adjust,
    paired_component_bootstrap,
    validate_target_semantics,
)


MODERN_VALUES = {
    "IFEval": 10.0,
    "BBH": 20.0,
    "MATH Lvl 5": 30.0,
    "GPQA": 40.0,
    "MUSR": 50.0,
    "MMLU-PRO": 60.0,
}


def _target_row(model_id: str, *, shift: float = 0.0) -> dict[str, object]:
    return {
        "model_id": model_id,
        **{name: value + shift for name, value in MODERN_VALUES.items()},
    }


def _attach_semantics(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for task in MODERN_VALUES:
        output[task] = output["modern_average"]
    output["target_semantics"] = MODERN_SEMANTICS.name
    output["target_components"] = "|".join(MODERN_SEMANTICS.components)
    output["target_aggregation"] = MODERN_SEMANTICS.aggregation
    output["target_scale"] = MODERN_SEMANTICS.scale
    output["benchmark_generation"] = MODERN_SEMANTICS.generation
    output["target_outcome"] = "modern_average"
    output["target_semantics_evidence"] = "six_named_v2_components_recomputed"
    return output


def test_conflicting_duplicate_evaluations_are_excluded() -> None:
    rows = [
        _target_row("org/conflict"),
        _target_row("org/conflict", shift=1.0),
        _target_row("org/clean"),
    ]

    targets, exclusions = build_modern_targets(pd.DataFrame(rows))

    assert targets["model_id"].tolist() == ["org/clean"]
    assert exclusions.loc[0, "model_id"] == "org/conflict"
    assert exclusions.loc[0, "reason"] == "conflicting_target_rows"


def test_canonical_equivalent_target_ids_are_deduplicated_before_conflict_check() -> None:
    rows = [_target_row("org--model"), _target_row("org/model", shift=1.0)]

    targets, exclusions = build_modern_targets(pd.DataFrame(rows))

    assert targets.empty
    assert exclusions[["model_id", "reason"]].to_dict("records") == [
        {"model_id": "org/model", "reason": "conflicting_target_rows"}
    ]


def test_identical_duplicate_evaluations_collapse_deterministically() -> None:
    rows = [_target_row("org/model"), _target_row("org/model")]

    targets, exclusions = build_modern_targets(pd.DataFrame(rows))

    assert targets["model_id"].tolist() == ["org/model"]
    assert targets.loc[0, "modern_average"] == pytest.approx(35.0)
    assert targets.loc[0, "target_semantics"] == MODERN_SEMANTICS.name
    assert targets.loc[0, "target_scale"] == MODERN_SEMANTICS.scale
    assert targets.loc[0, "benchmark_generation"] == MODERN_SEMANTICS.generation
    assert exclusions.empty


def test_target_contract_preserves_checkpoint_and_architecture_provenance() -> None:
    row = _target_row("org/model")
    row.update(
        {
            "Model sha": "abc123",
            "eval_name": "org/model",
            "architecture": "llama",
        }
    )

    targets, exclusions = build_modern_targets(pd.DataFrame([row]))

    assert targets.loc[0, "target_model_sha"] == "abc123"
    assert targets.loc[0, "target_eval_name"] == "org/model"
    assert targets.loc[0, "target_architecture"] == "llama"
    assert (
        targets.loc[0, "target_semantics_evidence"]
        == "six_named_v2_components_recomputed"
    )
    assert exclusions.empty


def test_duplicate_scores_with_conflicting_checkpoint_identity_are_excluded() -> None:
    left = _target_row("org/model") | {"Model sha": "left"}
    right = _target_row("org/model") | {"Model sha": "right"}

    targets, exclusions = build_modern_targets(pd.DataFrame([left, right]))

    assert targets.empty
    assert exclusions[["model_id", "reason"]].to_dict("records") == [
        {
            "model_id": "org/model",
            "reason": "conflicting_checkpoint_identity",
        }
    ]


def test_complete_and_incomplete_duplicate_rows_are_excluded_together() -> None:
    complete = _target_row("org/model")
    incomplete = _target_row("org/model")
    incomplete["GPQA"] = np.nan

    targets, exclusions = build_modern_targets(
        pd.DataFrame([complete, incomplete])
    )

    assert targets.empty
    assert exclusions["reason"].tolist() == ["conflicting_target_rows"]


def test_rows_without_all_modern_components_are_not_coerced_from_legacy_average() -> None:
    row = _target_row("org/incomplete")
    row["GPQA"] = np.nan
    row["Average ⬆️"] = 99.0

    targets, exclusions = build_modern_targets(pd.DataFrame([row]))

    assert targets.empty
    assert exclusions.loc[0, "reason"] == "incomplete_modern_target"


def test_legacy_and_modern_semantics_are_never_pooled() -> None:
    left = _attach_semantics(pd.DataFrame({"modern_average": [35.0]}))
    right = left.copy()
    right["target_semantics"] = "legacy_six_task_mean_v1"

    with pytest.raises(TargetSemanticsError):
        validate_target_semantics(left, right, target="modern_average")


def test_target_semantics_recomputes_aggregate_and_rejects_scale_mismatch() -> None:
    left = _attach_semantics(pd.DataFrame({"modern_average": [35.0]}))
    wrong_average = left.copy()
    wrong_average["modern_average"] = 99.0
    wrong_scale = left.copy()
    wrong_scale["target_scale"] = "unit_interval"

    with pytest.raises(TargetSemanticsError, match="recomputed"):
        validate_target_semantics(left, wrong_average, target="modern_average")
    with pytest.raises(TargetSemanticsError, match="differ"):
        validate_target_semantics(left, wrong_scale, target="modern_average")


def test_target_semantics_must_equal_frozen_contract_even_when_both_sides_lie() -> None:
    left = _attach_semantics(pd.DataFrame({"modern_average": [35.0]}))
    right = left.copy()
    left["benchmark_generation"] = "invented_generation"
    right["benchmark_generation"] = "invented_generation"

    with pytest.raises(TargetSemanticsError, match="frozen modern contract"):
        validate_target_semantics(left, right, target="modern_average")


def test_target_semantics_rejects_null_contract_rows() -> None:
    left = _attach_semantics(pd.DataFrame({"modern_average": [35.0, 35.0]}))
    right = left.copy()
    right.loc[1, "target_scale"] = np.nan

    with pytest.raises(TargetSemanticsError, match="target_scale"):
        validate_target_semantics(left, right, target="modern_average")


def test_missing_group_is_not_replaced_by_model_id() -> None:
    frame = pd.DataFrame(
        {
            "model_id": ["a", "b", "c"],
            "ancestry_component": ["component-1", "", np.nan],
            "component_provenance": ["graph", "graph", "graph"],
            "component_edge_count": [1, 1, 1],
        }
    )

    eligible, exclusions = confirmatory_group_population(frame)

    assert eligible["model_id"].tolist() == ["a"]
    assert exclusions["reason"].tolist() == ["ungrouped", "ungrouped"]


def test_graph_isolate_is_quarantined_as_a_weak_group() -> None:
    frame = pd.DataFrame(
        {
            "model_id": ["a"],
            "ancestry_component": ["graph-node:a"],
            "component_provenance": ["graph"],
            "component_edge_count": [0],
        }
    )

    eligible, exclusions = confirmatory_group_population(frame)

    assert eligible.empty
    assert exclusions["reason"].tolist() == ["weak_ancestry_component"]


def test_non_graph_fallback_group_is_never_confirmatory() -> None:
    frame = pd.DataFrame(
        {
            "model_id": ["a"],
            "ancestry_component": ["a"],
            "component_provenance": ["model_id_fallback"],
            "component_edge_count": [1],
        }
    )

    eligible, exclusions = confirmatory_group_population(frame)

    assert eligible.empty
    assert exclusions["reason"].tolist() == ["weak_ancestry_component"]


def test_ancestry_component_never_crosses_outer_fold() -> None:
    frame = pd.DataFrame(
        {
            "model_id": [f"m{index}" for index in range(18)],
            "ancestry_component": [f"g{index // 3}" for index in range(18)],
        }
    )

    folds = grouped_outer_folds(frame, n_splits=3, random_state=11)

    assert len(folds) == 3
    for fold in folds:
        train_groups = set(frame.iloc[fold.train_indices]["ancestry_component"])
        test_groups = set(frame.iloc[fold.test_indices]["ancestry_component"])
        assert train_groups.isdisjoint(test_groups)
    assert sorted(np.concatenate([fold.test_indices for fold in folds]).tolist()) == list(range(len(frame)))


def test_component_sample_weights_give_every_component_equal_total_weight() -> None:
    groups = np.array(["large", "large", "large", "small"])

    weights = component_sample_weights(groups)

    assert weights[:3].sum() == pytest.approx(weights[3:].sum())
    assert weights.sum() == pytest.approx(2.0)


def test_group_weighted_mae_does_not_reward_a_large_easy_component() -> None:
    actual = np.zeros(5)
    predicted = np.array([0.0, 0.0, 0.0, 0.0, 10.0])
    groups = np.array(["large", "large", "large", "large", "small"])

    assert group_weighted_mae(actual, predicted, groups) == pytest.approx(5.0)


def test_ridge_pipeline_imputer_is_fit_only_on_training_rows() -> None:
    train = np.array([[1.0], [np.nan], [3.0]])
    target = np.array([1.0, 2.0, 3.0])
    pipeline = build_ridge_pipeline(alpha=1.0)

    pipeline.fit(train, target)

    assert pipeline.named_steps["imputer"].statistics_[0] == pytest.approx(2.0)


def test_ridge_pipeline_treats_infinite_features_as_missing() -> None:
    train = np.array([[1.0], [np.inf], [3.0]])
    target = np.array([1.0, 2.0, 3.0])
    pipeline = build_ridge_pipeline(alpha=1.0)

    pipeline.fit(train, target)

    assert np.isfinite(pipeline.predict(np.array([[-np.inf], [2.0]]))).all()


def test_component_weights_reach_imputer_scaler_and_ridge(monkeypatch) -> None:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    observed: dict[str, np.ndarray] = {}
    original_imputer = WeightedMedianImputer.fit
    original_scaler = StandardScaler.fit
    original_ridge = Ridge.fit

    def fit_imputer(self, X, y=None, sample_weight=None):
        observed["imputer"] = np.asarray(sample_weight, dtype=float)
        return original_imputer(self, X, y, sample_weight=sample_weight)

    def fit_scaler(self, X, y=None, sample_weight=None):
        observed["scaler"] = np.asarray(sample_weight, dtype=float)
        return original_scaler(self, X, y, sample_weight=sample_weight)

    def fit_ridge(self, X, y, sample_weight=None):
        observed["ridge"] = np.asarray(sample_weight, dtype=float)
        return original_ridge(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(WeightedMedianImputer, "fit", fit_imputer)
    monkeypatch.setattr(StandardScaler, "fit", fit_scaler)
    monkeypatch.setattr(Ridge, "fit", fit_ridge)
    weights = np.array([0.5, 0.5, 1.0])

    _fit_weighted_pipeline(
        build_ridge_pipeline(alpha=1.0),
        np.array([[1.0], [2.0], [3.0]]),
        np.array([1.0, 2.0, 3.0]),
        weights,
    )

    assert set(observed) == {"imputer", "scaler", "ridge"}
    assert all(np.array_equal(value, weights) for value in observed.values())


def _prediction_frame() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    rows: list[dict[str, object]] = []
    for group_index in range(12):
        group_signal = float(group_index - 5.5)
        for member in range(2):
            h_signal = group_signal + 0.1 * member
            rows.append(
                {
                    "model_id": f"g{group_index}/m{member}",
                    "ancestry_component": f"g{group_index}",
                    "component_provenance": "graph",
                    "component_edge_count": 1,
                    "component_graph_fingerprint": "graph-v1",
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "modern_average": 50.0 + 3.0 * h_signal + rng.normal(scale=0.05),
                    "M__layer_count": rng.normal(),
                    "S__attn_q__log_rms__median": rng.normal(),
                    "H__attn_q__alpha__median": h_signal,
                }
            )
    return pd.DataFrame(rows)


def test_feature_contract_rejects_prefixed_columns_outside_the_whitelist() -> None:
    frame = _prediction_frame()
    frame["M__target_copy"] = frame["modern_average"]

    with pytest.raises(ValueError, match="unapproved prefixed feature"):
        evaluate_grouped_ridge(frame, target="modern_average")


def test_feature_contract_rejects_duplicate_columns_and_null_schema_rows() -> None:
    frame = _prediction_frame()
    duplicate = pd.concat([frame, frame[["M__layer_count"]]], axis=1)
    null_schema = frame.copy()
    null_schema.loc[0, "feature_schema_version"] = np.nan

    with pytest.raises(ValueError, match="duplicate column"):
        evaluate_grouped_ridge(duplicate, target="modern_average")
    with pytest.raises(ValueError, match="feature schema"):
        evaluate_grouped_ridge(null_schema, target="modern_average")


def test_grouped_ridge_uses_identical_rows_and_folds_for_all_blocks() -> None:
    predictions, metrics, audit = evaluate_grouped_ridge(
        _prediction_frame(),
        target="modern_average",
        outer_splits=4,
        inner_splits=3,
        ridge_alphas=(0.01, 1.0, 100.0),
        random_state=7,
    )

    counts = predictions.groupby("feature_block")["model_id"].nunique()
    assert counts.nunique() == 1
    fold_maps = [
        block.set_index("model_id")["outer_fold"].sort_index()
        for _, block in predictions.groupby("feature_block")
    ]
    assert all(fold_maps[0].equals(other) for other in fold_maps[1:])
    assert set(metrics["feature_block"]) == {"M", "M+S", "M+S+H"}
    assert set(predictions["chosen_alpha"]).issubset({0.01, 1.0, 100.0})
    for fold in audit:
        assert set(fold["train_groups"]).isdisjoint(fold["test_groups"])


def test_outer_held_out_rows_never_reach_actual_pipeline_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spectral_research.study.performance as performance

    frame = _prediction_frame().copy()
    frame["M__layer_count"] = np.arange(len(frame), dtype=float)
    fitted_markers: list[set[float]] = []
    original_fit = performance._fit_weighted_pipeline

    def spy_fit(pipeline, X, y, sample_weight):
        fitted_markers.append(set(np.asarray(X, dtype=float)[:, 0]))
        return original_fit(pipeline, X, y, sample_weight)

    monkeypatch.setattr(performance, "_fit_weighted_pipeline", spy_fit)
    monkeypatch.setattr(performance, "_select_alpha", lambda *_args, **_kwargs: 1.0)

    _, _, audit = evaluate_grouped_ridge(
        frame,
        target="modern_average",
        outer_splits=3,
        inner_splits=2,
        ridge_alphas=(1.0,),
        random_state=5,
    )

    marker_by_model = frame.set_index("model_id")["M__layer_count"].to_dict()
    assert len(fitted_markers) == len(audit) * 3
    for fold_index, fold in enumerate(audit):
        train_markers = {
            marker_by_model[model_id]
            for model_id in frame.loc[
                frame["ancestry_component"].isin(fold["train_groups"]), "model_id"
            ]
        }
        test_markers = {
            marker_by_model[model_id]
            for model_id in frame.loc[
                frame["ancestry_component"].isin(fold["test_groups"]), "model_id"
            ]
        }
        for fitted in fitted_markers[fold_index * 3 : (fold_index + 1) * 3]:
            assert fitted == train_markers
            assert fitted.isdisjoint(test_markers)


def test_grouped_ridge_detects_incremental_htsr_signal() -> None:
    _, metrics, _ = evaluate_grouped_ridge(
        _prediction_frame(),
        target="modern_average",
        outer_splits=4,
        inner_splits=3,
        ridge_alphas=(0.01, 1.0, 100.0),
        random_state=7,
    )

    mae = metrics.set_index("feature_block")["group_weighted_mae"]
    assert mae["M+S+H"] < 0.2 * mae["M+S"]


def test_paired_component_bootstrap_resamples_components_not_rows() -> None:
    rows = []
    for block, error_scale in (("M+S", 2.0), ("M+S+H", 1.0)):
        for component in range(12):
            rows.append(
                {
                    "model_id": f"m{component}",
                    "ancestry_component": f"g{component}",
                    "target": "modern_average",
                    "feature_block": block,
                    "actual": 0.0,
                    "prediction": error_scale * (component + 1),
                }
            )
    predictions = pd.DataFrame(rows)

    result = paired_component_bootstrap(predictions, n_bootstrap=200, random_state=3)

    assert result["n_components"] == 12
    assert result["delta_mae"] == pytest.approx(-6.5)
    assert result["ci_high"] < 0.0
    assert result["inferential"] is True


def test_paired_bootstrap_refuses_mismatched_method_populations() -> None:
    predictions = pd.DataFrame(
        [
            {
                "model_id": "a",
                "ancestry_component": "g1",
                "target": "modern_average",
                "feature_block": "M+S",
                "actual": 1.0,
                "prediction": 2.0,
            },
            {
                "model_id": "b",
                "ancestry_component": "g2",
                "target": "modern_average",
                "feature_block": "M+S",
                "actual": 1.0,
                "prediction": 2.0,
            },
            {
                "model_id": "a",
                "ancestry_component": "g1",
                "target": "modern_average",
                "feature_block": "M+S+H",
                "actual": 1.0,
                "prediction": 1.0,
            },
        ]
    )

    with pytest.raises(ValueError, match="identical populations"):
        paired_component_bootstrap(predictions)


def test_paired_bootstrap_is_descriptive_below_component_floor() -> None:
    predictions = pd.DataFrame(
        {
            "model_id": ["a", "b"] * 2,
            "ancestry_component": ["g1", "g2"] * 2,
            "target": ["modern_average"] * 4,
            "feature_block": ["M+S"] * 2 + ["M+S+H"] * 2,
            "actual": [0.0] * 4,
            "prediction": [2.0, 2.0, 1.0, 1.0],
        }
    )

    result = paired_component_bootstrap(predictions, minimum_components=10)

    assert result["delta_mae"] == pytest.approx(-1.0)
    assert result["inferential"] is False
    assert np.isnan(result["ci_low"])
    assert np.isnan(result["ci_high"])


def test_paired_bootstrap_rejects_mixed_targets_or_external_slices() -> None:
    predictions = pd.DataFrame(
        {
            "model_id": ["a", "a", "b", "b"],
            "ancestry_component": ["g1", "g1", "g2", "g2"],
            "target": ["IFEval", "IFEval", "BBH", "BBH"],
            "feature_block": ["M+S", "M+S+H"] * 2,
            "actual": [1.0] * 4,
            "prediction": [2.0, 1.0, 2.0, 1.0],
            "external_slice": ["component_disjoint"] * 4,
        }
    )

    with pytest.raises(ValueError, match="one target"):
        paired_component_bootstrap(predictions)

    predictions["target"] = "IFEval"
    predictions.loc[2:, "external_slice"] = "component_overlap_sensitivity"
    with pytest.raises(ValueError, match="one external slice"):
        paired_component_bootstrap(predictions)


def test_secondary_endpoint_runner_applies_holm_to_exactly_six_tasks() -> None:
    frame = _attach_semantics(_prediction_frame())

    result = evaluate_secondary_endpoints(
        frame,
        outer_splits=4,
        inner_splits=3,
        ridge_alphas=(0.1, 1.0),
        bootstrap_replicates=40,
        signflip_replicates=128,
        random_state=13,
    )

    increments = result["increments"]
    assert set(increments["target"]) == set(MODERN_VALUES)
    assert len(increments) == 6
    assert increments["holm_family_size"].eq(6).all()
    assert (increments["p_value_holm"] >= increments["p_value_raw"]).all()
    assert set(result["predictions"]["target"]) == set(MODERN_VALUES)


def test_external_headline_excludes_components_seen_in_development() -> None:
    development = pd.DataFrame(
        {
            "model_id": ["dev/a", "dev/b"],
            "ancestry_component": ["a", "b"],
            "component_graph_fingerprint": ["graph-v1", "graph-v1"],
        }
    )
    external = pd.DataFrame(
        {
            "model_id": ["overlap", "strict"],
            "ancestry_component": ["b", "c"],
            "component_graph_fingerprint": ["graph-v1", "graph-v1"],
        }
    )

    strict, overlap = external_component_slices(development, external)

    assert strict["model_id"].tolist() == ["strict"]
    assert overlap["model_id"].tolist() == ["overlap"]


def test_external_slices_require_same_graph_universe_and_disjoint_model_ids() -> None:
    development = pd.DataFrame(
        {
            "model_id": ["same/model"],
            "ancestry_component": ["a"],
            "component_graph_fingerprint": ["graph-v1"],
        }
    )
    reused_id = pd.DataFrame(
        {
            "model_id": ["same/model"],
            "ancestry_component": ["z"],
            "component_graph_fingerprint": ["graph-v1"],
        }
    )
    wrong_graph = reused_id.assign(
        model_id="external/model", component_graph_fingerprint="graph-v2"
    )

    with pytest.raises(ValueError, match="model IDs overlap"):
        external_component_slices(development, reused_id)
    with pytest.raises(ValueError, match="graph fingerprint"):
        external_component_slices(development, wrong_graph)


def test_external_slices_require_complete_non_null_graph_fingerprints() -> None:
    development = pd.DataFrame(
        {
            "model_id": ["dev/a", "dev/b"],
            "ancestry_component": ["a", "b"],
            "component_graph_fingerprint": ["graph-v1", np.nan],
        }
    )
    external = pd.DataFrame(
        {
            "model_id": ["ext/c"],
            "ancestry_component": ["c"],
            "component_graph_fingerprint": ["graph-v1"],
        }
    )

    with pytest.raises(ValueError, match="complete graph fingerprint"):
        external_component_slices(development, external)


def test_external_transfer_fits_on_development_and_reports_disjoint_slices() -> None:
    development = _attach_semantics(_prediction_frame())
    external = pd.DataFrame(
        [
            {
                "model_id": "external/overlap",
                "ancestry_component": "g0",
                "component_provenance": "graph",
                "component_edge_count": 1,
                "component_graph_fingerprint": "graph-v1",
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "modern_average": 52.0,
                "M__layer_count": 0.0,
                "S__attn_q__log_rms__median": 0.0,
                "H__attn_q__alpha__median": 0.6,
            },
            {
                "model_id": "external/strict-a",
                "ancestry_component": "external-a",
                "component_provenance": "graph",
                "component_edge_count": 1,
                "component_graph_fingerprint": "graph-v1",
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "modern_average": 55.0,
                "M__layer_count": 0.2,
                "S__attn_q__log_rms__median": -0.2,
                "H__attn_q__alpha__median": 1.6,
            },
            {
                "model_id": "external/strict-b",
                "ancestry_component": "external-b",
                "component_provenance": "graph",
                "component_edge_count": 1,
                "component_graph_fingerprint": "graph-v1",
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "modern_average": 45.0,
                "M__layer_count": -0.2,
                "S__attn_q__log_rms__median": 0.2,
                "H__attn_q__alpha__median": -1.6,
            },
        ]
    )
    external = _attach_semantics(external)

    predictions, metrics, audit = evaluate_external_transfer(
        development,
        external,
        target="modern_average",
        development_component_universe=development,
        inner_splits=3,
        ridge_alphas=(0.01, 1.0),
        random_state=9,
    )

    assert set(predictions["external_slice"]) == {
        "component_disjoint",
        "component_overlap_sensitivity",
    }
    strict = predictions[predictions["external_slice"].eq("component_disjoint")]
    overlap = predictions[
        predictions["external_slice"].eq("component_overlap_sensitivity")
    ]
    assert strict["model_id"].nunique() == 2
    assert overlap["model_id"].unique().tolist() == ["external/overlap"]
    assert set(metrics["feature_block"]) == {"M", "M+S", "M+S+H"}
    assert set(audit["chosen_alpha"]).issubset({0.01, 1.0})
    assert set(audit["fit_population"]) == {"development_only"}


def test_external_transfer_rejects_incomplete_component_universe_and_schema() -> None:
    development = _attach_semantics(_prediction_frame())
    external = _attach_semantics(_prediction_frame().iloc[:2].copy())
    external["model_id"] = ["external/a", "external/b"]
    external["ancestry_component"] = ["external-a", "external-b"]
    incomplete_universe = development.loc[
        ~development["ancestry_component"].eq("g0")
    ].copy()

    with pytest.raises(ValueError, match="all fitted development components"):
        evaluate_external_transfer(
            development,
            external,
            target="modern_average",
            development_component_universe=incomplete_universe,
            inner_splits=3,
        )

    external = external.drop(columns="H__attn_q__alpha__median")
    with pytest.raises(ValueError, match="feature schema"):
        evaluate_external_transfer(
            development,
            external,
            target="modern_average",
            development_component_universe=development,
            inner_splits=3,
        )


def test_external_transfer_rejects_semantic_mismatch_before_fitting(monkeypatch) -> None:
    development = _attach_semantics(_prediction_frame())
    external = development.iloc[:2].copy()
    external["target_semantics"] = "legacy_six_task_mean_v1"

    def fail_if_called(*args, **kwargs):
        raise AssertionError("fit must not run after a semantic mismatch")

    monkeypatch.setattr("sklearn.pipeline.Pipeline.fit", fail_if_called)
    with pytest.raises(TargetSemanticsError):
        evaluate_external_transfer(
            development,
            external,
            target="modern_average",
            development_component_universe=development,
        )


def test_holm_adjust_is_monotone_in_sorted_p_values() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.20})

    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["b"] == pytest.approx(0.06)
    assert adjusted["c"] == pytest.approx(0.20)


def test_htsr_pass_rule_requires_effect_interval_and_external_nonreversal() -> None:
    passed = assess_htsr_increment(
        baseline_mae=10.0,
        htsr_mae=9.0,
        ci_low=-2.0,
        ci_high=-0.1,
        external_delta=-0.2,
        external_available=True,
        n_components=20,
    )
    unavailable = assess_htsr_increment(
        baseline_mae=10.0,
        htsr_mae=9.0,
        ci_low=-2.0,
        ci_high=-0.1,
        external_delta=np.nan,
        external_available=False,
        n_components=20,
    )

    assert passed["passed"] is True
    assert unavailable["passed"] is False
    assert "external_unavailable" in unavailable["reasons"]


def test_htsr_pass_rule_rejects_reused_external_evidence_as_confirmation() -> None:
    result = assess_htsr_increment(
        baseline_mae=10.0,
        htsr_mae=9.0,
        ci_low=-2.0,
        ci_high=-0.1,
        external_delta=-0.2,
        external_available=True,
        external_confirmatory=False,
        n_components=20,
    )

    assert result["passed"] is False
    assert "external_not_confirmatory" in result["reasons"]


def test_htsr_pass_rule_rejects_inverted_interval_and_small_component_count() -> None:
    inverted = assess_htsr_increment(
        baseline_mae=10.0,
        htsr_mae=9.0,
        ci_low=1.0,
        ci_high=-0.1,
        external_delta=-0.2,
        external_available=True,
        n_components=20,
    )
    small = assess_htsr_increment(
        baseline_mae=10.0,
        htsr_mae=9.0,
        ci_low=-2.0,
        ci_high=-0.1,
        external_delta=-0.2,
        external_available=True,
        n_components=3,
    )

    assert inverted["passed"] is False
    assert "invalid_bootstrap_interval" in inverted["reasons"]
    assert small["passed"] is False
    assert "insufficient_components" in small["reasons"]


def test_htsr_pass_rule_distinguishes_null_crossing_from_interval_supported_harm() -> None:
    crossing = assess_htsr_increment(
        baseline_mae=10.0,
        htsr_mae=10.5,
        ci_low=-0.2,
        ci_high=0.8,
        external_delta=np.nan,
        external_available=False,
        n_components=20,
    )
    harmful = assess_htsr_increment(
        baseline_mae=10.0,
        htsr_mae=11.0,
        ci_low=0.4,
        ci_high=1.7,
        external_delta=np.nan,
        external_available=False,
        n_components=20,
    )

    assert "bootstrap_interval_includes_zero" in crossing["reasons"]
    assert "bootstrap_interval_supports_harm" not in crossing["reasons"]
    assert "bootstrap_interval_supports_harm" in harmful["reasons"]
    assert "bootstrap_interval_includes_zero" not in harmful["reasons"]


def test_htsr_pass_rule_rejects_overlap_slice_as_external_confirmation() -> None:
    result = assess_htsr_increment(
        baseline_mae=10.0,
        htsr_mae=9.0,
        ci_low=-2.0,
        ci_high=-0.1,
        external_delta=-0.2,
        external_available=True,
        external_slice="component_overlap_sensitivity",
        n_components=20,
    )

    assert result["passed"] is False
    assert "external_not_component_disjoint" in result["reasons"]
