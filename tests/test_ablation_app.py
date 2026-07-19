import json
import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import alpha_cluster_dashboard.ablation_app as ablation_app

from alpha_cluster_dashboard.ablation_app import (
    PLOT_CONFIG,
    _atlas_zoom_from_control,
    _cached_config_value,
    _frozen_run_state_token,
    _category_color_map,
    _chart_theme_tokens,
    _cluster_galaxy_layout,
    _color_options,
    _dashboard_summary_cards,
    _dashboard_cache_from_scaleup,
    _dashboard_cache_from_study,
    _dashboard_theme_tokens,
    _display_model_name,
    _dynamic_hover_columns,
    _edge_limit_for_density,
    _external_status_for_target,
    _figure_axis_ranges,
    _filter_frame,
    _frozen_cluster_profiles,
    _galaxy_neighbor_edge_traces,
    _galaxy_arm_edges,
    _galaxy_arm_trace_styles,
    _lineage_quality_metrics,
    _lineage_display_frame,
    _lineage_edge_coordinates,
    _lineage_color_options,
    _method_evidence_display_tables,
    _plot_cluster_galaxy,
    _plot_config,
    _plot_galaxy,
    _panel_options,
    _plot_prediction_lineage,
    _plotly_chart,
    _performance_target_values,
    _scaleup_performance_target_values,
    _prediction_error_color_scale,
    _prediction_hover_template,
    _prediction_metric_cards,
    _prediction_counts,
    _prediction_quality_metrics,
    _prediction_lineage_frame,
    _preview_neighbors,
    _render_point_bounds,
    _settled_cluster_centers,
    _scaled_marker_size,
    _target_options,
    _study_summary_cards,
    _with_display_columns,
    render_ablation_atlas,
)
from alpha_cluster_dashboard.study_outputs import StudyOutputs
from alpha_cluster_dashboard.scaleup_outputs import ScaleupOutputs


def _frame(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "display_name": [f"model-{idx:04d}" for idx in range(rows)],
            "embedding_x": [float(idx) for idx in range(rows)],
            "embedding_y": [float(idx % 17) for idx in range(rows)],
            "embedding_z": [float(idx % 11) for idx in range(rows)],
            "cluster": [idx % 5 for idx in range(rows)],
            "coarse_family": ["fam-a" if idx % 2 == 0 else "fam-b" for idx in range(rows)],
            "overall_log_alpha": [float(idx % 7) for idx in range(rows)],
        }
    )


def _frozen_outputs(tmp_path: Path, rows: int = 6) -> StudyOutputs:
    model_ids = [f"org/model-{index}" for index in range(rows)]
    manifest = pd.DataFrame({"model_id": model_ids})
    assignments = pd.DataFrame(
        {"model_id": model_ids, "cluster": [index % 2 for index in range(rows)]}
    )
    performance = pd.DataFrame(
        {
            "model_id": model_ids,
            "target": ["modern_average"] * rows,
            "feature_block": ["M+S+H"] * rows,
            "actual": np.arange(rows, dtype=float),
            "prediction": np.arange(rows, dtype=float) + 0.2,
        }
    )
    atlas = manifest.merge(assignments, on="model_id")
    atlas["display_name"] = atlas["model_id"]
    atlas["coarse_family"] = "root/a"
    return StudyOutputs(
        run_dir=tmp_path,
        provenance={"run_id": "test"},
        manifest=manifest,
        features=manifest.copy(),
        atlas=atlas,
        cluster_assignments=assignments,
        lineage_predictions=pd.DataFrame(),
        representative_lineage_predictions=pd.DataFrame(),
        performance_predictions=performance,
        exclusions=pd.DataFrame(),
        lineage_metrics={},
        performance_metrics={},
        lineage_summary={
            "root_macro_top1": 0.75,
            "mrr": 0.8,
            "unknown_auroc": 0.7,
            "oscr": 0.6,
            "n_queries": rows,
            "n_families": 3,
            "n_open_folds": 3,
            "minimum_families": 10,
            "inferential": False,
            "observed_family_top1": 0.9,
            "observed_spectral_shape_top1": 0.9,
            "observed_htsr_lift": 0.0,
            "observed_family_queries": rows,
            "observed_family_families": 3,
        },
        performance_summary={
            "group_weighted_mae": 1.2,
            "spearman": 0.65,
            "htsr_delta_mae": -0.3,
            "relative_mae_improvement": 0.04,
            "primary_passed": False,
            "primary_reasons": ["relative_improvement_below_threshold"],
            "target": "modern_average",
            "external_target": "modern_average",
            "n_models": rows,
            "n_components": 3,
        },
        summary={
            "n_models": rows,
            "n_feature_models": rows,
            "n_clustered_models": rows,
            "n_lineage_models": 0,
            "n_performance_models": rows,
            "n_exclusions": 0,
        },
        external_status="unavailable",
    )


def _scaleup_outputs(tmp_path: Path) -> ScaleupOutputs:
    population = pd.DataFrame(
        {
            "model_id": ["org/model-0", "org/model-1", "org/new-model"],
            "htsr_artifact_available": [True, True, False],
            "strict_cluster": pd.Series([4, 5, pd.NA], dtype="Int64"),
            "broad_cluster": pd.Series([2, 3, pd.NA], dtype="Int64"),
            "predicted_family": ["root/a", "root/b", pd.NA],
            "lineage_evidence_status": [
                "strict_qc_clustered",
                "strict_qc_clustered",
                "missing_htsr_artifact",
            ],
        }
    )
    predictions = pd.DataFrame(
        {
            "model_id": ["org/model-0", "org/model-1"],
            "display_name": ["org/model-0", "org/model-1"],
            "target": ["modern_average", "modern_average"],
            "feature_block": ["alpha_profile_k5_component_distinct"] * 2,
            "actual": [20.0, 25.0],
            "prediction": [21.0, 24.5],
            "absolute_error": [1.0, 0.5],
            "nearest_neighbors": ["", ""],
        }
    )
    return ScaleupOutputs(
        run_dir=tmp_path / "scaleup_run",
        metadata={"run_id": "scaleup_run"},
        counts={
            "phase2_census": 3,
            "phase2_htsr_artifacts": 2,
            "phase2_strict_lineage": 2,
        },
        population=population,
        predictions={"modern_average": predictions},
        performance={
            "modern_average": {
                "target": "modern_average",
                "group_weighted_mae": 0.75,
                "spearman": 1.0,
                "delta_mae": -0.4,
                "ci_low": -0.7,
                "ci_high": -0.1,
                "n_models": 2,
                "n_components": 2,
            }
        },
        broad_cluster_metrics=pd.DataFrame(),
        report="# Scale-up\n",
    )


def test_live_dashboard_entrypoint_loads_frozen_outputs_without_fitting_cache() -> None:
    source = inspect.getsource(render_ablation_atlas)

    assert "_cached_study_outputs" in source
    assert "default_study_run()" in source
    assert "default_scaleup_run()" in source
    assert "_cached_scaleup_outputs" in source
    assert "load_or_build_dashboard_cache" not in source
    assert "source_fingerprint" not in source
    assert "Rendered points" not in source


def test_frozen_run_state_token_changes_when_an_artifact_changes(tmp_path: Path) -> None:
    run = tmp_path / "htsr_run"
    run.mkdir()
    artifact = run / "manifest.csv"
    artifact_manifest = run / "artifact_manifest.json"
    artifact.write_text("model_id\na/model\n", encoding="utf-8")
    artifact_manifest.write_text('{"version": 1}\n', encoding="utf-8")
    first = _frozen_run_state_token(run)

    artifact.write_text("model_id\nb/model\n", encoding="utf-8")
    artifact_manifest.write_text('{"version": 2}\n', encoding="utf-8")
    second = _frozen_run_state_token(run)

    assert first != second


def test_frozen_loader_cache_is_disk_persistent_and_used_by_entrypoint() -> None:
    module_source = inspect.getsource(ablation_app)
    entrypoint_source = inspect.getsource(render_ablation_atlas)

    assert '@st.cache_data(show_spinner=False, persist="disk")' in module_source
    assert "_cached_study_outputs(configured_path, run_state_token)" in entrypoint_source


def test_frozen_study_adapter_exposes_all_rows_and_oof_predictions(tmp_path: Path) -> None:
    outputs = _frozen_outputs(tmp_path, rows=9)

    cache = _dashboard_cache_from_study(outputs)

    assert len(cache.metadata) == 9
    assert set(cache.predictions) == {"modern_average"}
    assert len(cache.predictions["modern_average"]) == 9
    assert int(cache.cluster_summary["size"].sum()) == 9


def test_scaleup_adapter_uses_full_ledger_strict_clusters_and_alpha_predictions(
    tmp_path: Path,
) -> None:
    cache = _dashboard_cache_from_scaleup(
        _frozen_outputs(tmp_path, rows=2),
        _scaleup_outputs(tmp_path),
    )

    assert len(cache.metadata) == 3
    assert cache.metadata["cluster"].tolist() == [4, 5, pd.NA]
    assert cache.metadata["broad_cluster"].tolist() == [2, 3, pd.NA]
    assert set(cache.predictions) == {"modern_average"}
    assert cache.metadata["modern_average"].tolist()[:2] == [20.0, 25.0]
    assert int(cache.cluster_summary["size"].sum()) == 2


def test_scaleup_performance_values_report_direct_alpha_result(tmp_path: Path) -> None:
    values = _scaleup_performance_target_values(
        _scaleup_outputs(tmp_path), "modern_average"
    )

    assert values["group_weighted_mae"] == 0.75
    assert values["htsr_delta_mae"] == -0.4
    assert values["htsr_ci_low"] == -0.7
    assert values["htsr_ci_high"] == -0.1
    assert values["n_models"] == 2


def test_frozen_study_adapter_preserves_unassigned_clusters_as_missing(tmp_path: Path) -> None:
    outputs = _frozen_outputs(tmp_path, rows=5)
    atlas = outputs.atlas.copy()
    atlas.loc[atlas.index[-1], "cluster"] = np.nan
    atlas["H__q_proj__alpha__median"] = np.arange(5, dtype=float) + 1.0
    outputs = replace(outputs, atlas=atlas)

    cache = _dashboard_cache_from_study(outputs)

    assert str(cache.metadata["cluster"].dtype) == "Int64"
    assert cache.metadata.loc[cache.metadata.index[-1], "cluster"] is pd.NA
    assert -1 not in cache.cluster_summary["cluster"].tolist()
    assert -1 not in cache.cluster_profiles["cluster"].tolist()
    assert int(cache.cluster_summary["size"].sum()) == 4


def test_frozen_study_adapter_handles_missing_lineage_correctness(tmp_path: Path) -> None:
    outputs = _frozen_outputs(tmp_path, rows=3)
    outputs.atlas["correct"] = pd.Series([True, pd.NA, False], dtype="boolean")

    cache = _dashboard_cache_from_study(outputs)

    assert cache.metadata["Lineage correctness"].tolist() == [
        "correct",
        "unavailable",
        "incorrect",
    ]


def test_frozen_study_adapter_exposes_lineage_endpoint(tmp_path: Path) -> None:
    outputs = _frozen_outputs(tmp_path, rows=3)
    outputs.atlas["lineage_endpoint"] = [
        "canonical_root",
        "observed_family",
        pd.NA,
    ]

    cache = _dashboard_cache_from_study(outputs)

    assert cache.metadata["Lineage endpoint"].tolist() == [
        "canonical root",
        "observed family",
        "unavailable",
    ]


def test_frozen_study_adapter_does_not_materialize_unused_dense_features(tmp_path: Path) -> None:
    outputs = _frozen_outputs(tmp_path, rows=9)
    outputs = replace(
        outputs,
        features=pd.DataFrame(
            {
                "model_id": outputs.manifest["model_id"],
                "H__q_proj__alpha__median": np.arange(9, dtype=float),
                "S__trajectory_length": np.arange(9, dtype=float) + 10.0,
            }
        ),
    )

    cache = _dashboard_cache_from_study(outputs)

    assert cache.features.shape == (len(cache.metadata), 0)
    assert cache.features.nbytes == 0
    assert cache.feature_names == []


def test_lineage_display_keeps_every_clustered_model_without_requiring_family_prediction() -> None:
    frame = pd.DataFrame(
        {
            "model_id": ["a", "b", "c", "d"],
            "cluster": [0, 1, -1, np.nan],
            "predicted_family": ["root/a", pd.NA, pd.NA, "root/d"],
        }
    )

    displayed = _lineage_display_frame(frame)

    assert displayed["model_id"].tolist() == ["a", "b"]


def test_study_summary_cards_use_frozen_scientific_metrics(tmp_path: Path) -> None:
    cards = _study_summary_cards(_frozen_outputs(tmp_path, rows=7))
    by_label = {card["label"]: card["value"] for card in cards}

    assert by_label["Manifest models"] == "7"
    assert by_label["Root top-1"] == "75.0%"
    assert by_label["Known-family top-1"] == "90.0%"
    assert by_label["Grouped MAE"] == "1.200"
    assert by_label["HTSR dMAE"] == "-0.300"
    assert by_label["External dMAE"] == "Unavailable"
    root_card = next(card for card in cards if card["label"] == "Root top-1")
    assert root_card["delta"] == "3 families / descriptive"
    htsr_card = next(card for card in cards if card["label"] == "HTSR dMAE")
    assert htsr_card["delta"] == "claim failed"
    known_card = next(card for card in cards if card["label"] == "Known-family top-1")
    assert known_card["delta"] == "HTSR lift +0.0 pp"


def test_study_summary_cards_report_external_component_disjoint_effect(tmp_path: Path) -> None:
    outputs = _frozen_outputs(tmp_path, rows=7)
    outputs = replace(
        outputs,
        external_status="descriptive_reuse",
        performance_summary={
            **outputs.performance_summary,
            "external_htsr_delta_mae": -0.1,
            "external_n_models": 5,
            "external_n_components": 3,
            "external_evidence_role": "reused_descriptive",
        },
    )

    external = next(
        card for card in _study_summary_cards(outputs) if card["label"] == "External dMAE"
    )

    assert external["value"] == "-0.100"
    assert external["delta"] == "reused; 5 / 3"


def test_performance_target_values_include_oof_group_scope(tmp_path: Path) -> None:
    outputs = _frozen_outputs(tmp_path, rows=7)

    values = _performance_target_values(outputs, "modern_average")

    assert values["n_models"] == 7
    assert values["n_components"] == 3


def test_external_status_is_only_shown_for_its_evaluated_target(
    tmp_path: Path,
) -> None:
    outputs = replace(_frozen_outputs(tmp_path), external_status="descriptive_reuse")

    assert _external_status_for_target(outputs, "modern_average") == "Reused only"
    assert _external_status_for_target(outputs, "BBH") == "Not evaluated"


def test_frozen_lineage_scope_card_uses_compact_dynamic_counts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: list[dict[str, str]] = []
    monkeypatch.setattr(ablation_app, "_render_cards", captured.extend)
    monkeypatch.setattr(ablation_app.st, "caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ablation_app.st, "info", lambda *_args, **_kwargs: None)

    ablation_app._render_frozen_lineage(
        _frozen_outputs(tmp_path, rows=31),
        pd.DataFrame(),
        color_by="Predicted cluster",
        size_by="fixed",
        edge_density="fast",
        gesture_zoom=False,
        atlas_zoom=1.0,
    )

    scope = next(card for card in captured if card["label"] == "Queries / families")
    assert scope["value"] == "31 / 3"
    assert scope["delta"] == "Descriptive"


def test_lineage_color_options_expose_prediction_endpoint() -> None:
    frame = pd.DataFrame(
        {
            "Predicted cluster": ["C00"],
            "Predicted family": ["root/a"],
            "Lineage correctness": ["correct"],
            "Lineage endpoint": ["canonical root"],
        }
    )

    assert _lineage_color_options(frame) == [
        "Predicted cluster",
        "Predicted family",
        "Lineage endpoint",
        "Lineage correctness",
    ]


def test_atlas_renders_all_clustered_rows_without_inventing_an_unclustered_cluster() -> None:
    source = inspect.getsource(render_ablation_atlas)

    assert "atlas_frame = _lineage_display_frame(filtered)" in source
    assert "every loaded row is rendered by default" not in source


def test_frozen_cluster_profiles_parse_canonical_htsr_feature_names() -> None:
    metadata = pd.DataFrame(
        {
            "cluster": [0, 0, 1],
            "H__q_proj__alpha__median": [2.0, 4.0, 8.0],
            "H__down_proj__alpha__median": [3.0, 5.0, 9.0],
        }
    )

    profiles = _frozen_cluster_profiles(metadata)

    assert set(profiles["cluster"]) == {0, 1}
    assert set(profiles["role_name"]) == {"q_proj", "down_proj"}
    assert profiles["log_alpha"].notna().all()


def test_frozen_cluster_profiles_average_log1p_alpha_not_log1p_average_alpha() -> None:
    metadata = pd.DataFrame(
        {
            "cluster": [0, 0],
            "H__q_proj__alpha__median": [0.0, 8.0],
        }
    )

    profiles = _frozen_cluster_profiles(metadata)

    expected = float(np.mean(np.log1p([0.0, 8.0])))
    assert profiles.loc[0, "log_alpha"] == expected


def test_plotly_chart_uses_stable_streamlit_key_and_plotly_uirevision(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def capture_chart(fig, **kwargs) -> None:
        captured["fig"] = fig
        captured.update(kwargs)

    monkeypatch.setattr(ablation_app.st, "plotly_chart", capture_chart)
    figure = go.Figure()

    _plotly_chart(figure, key="atlas:run-a", scroll_zoom=True)

    assert captured["key"] == "atlas:run-a"
    assert figure.layout.uirevision == "atlas:run-a"
    assert captured["config"]["scrollZoom"] is True


def test_unreachable_live_analysis_renderers_are_removed() -> None:
    source = inspect.getsource(ablation_app)

    assert "def _render_prediction_tab(" not in source
    assert "def _render_lineage_tab(" not in source
    assert "def _render_method_evidence(" not in source


def test_filter_frame_supports_case_insensitive_model_search():
    frame = _frame(5)

    filtered = _filter_frame(frame, families=[], architectures=[], clusters=[], search_text="MODEL-0003")

    assert filtered["display_name"].tolist() == ["model-0003"]


def test_two_dimensional_galaxy_uses_webgl_traces_for_interactive_pan():
    frame = _with_display_columns(_frame(1200))
    fig = _plot_galaxy(frame, color_by="Predicted cluster", size_by="fixed", view_mode="2D microscope")

    assert {trace.type for trace in fig.data} == {"scattergl"}
    assert fig.layout.xaxis.visible is False
    assert fig.layout.showlegend is False


def test_display_columns_make_predicted_cluster_the_first_color_option():
    frame = _with_display_columns(_frame(12), target="Average")

    assert frame["Predicted cluster"].tolist()[:3] == ["C00", "C01", "C02"]
    assert "Cluster center model" in frame.columns
    assert _color_options(frame, target="Average")[0] == "Cluster center model"


def test_panel_options_are_minimal_analysis_views():
    assert _panel_options() == ["Atlas", "Prediction", "Lineage", "Profiles", "Evidence", "Data"]


def test_plot_config_makes_scroll_zoom_opt_in():
    assert PLOT_CONFIG["scrollZoom"] is False
    assert _plot_config(scroll_zoom=False)["scrollZoom"] is False
    assert _plot_config(scroll_zoom=True)["scrollZoom"] is True


def test_chart_theme_tokens_cover_dark_mode_readability():
    light = _chart_theme_tokens(dark=False)
    dark = _chart_theme_tokens(dark=True)

    assert light["template"] == "plotly_white"
    assert dark["template"] == "plotly_dark"
    assert dark["font_color"] == "#E5E7EB"
    assert dark["hover_bg"] != light["hover_bg"]


def test_dark_theme_uses_the_active_client_theme(monkeypatch):
    context = type("Context", (), {"theme": type("Theme", (), {"type": "dark"})()})()
    monkeypatch.setattr(ablation_app.st, "context", context)
    monkeypatch.setattr(ablation_app.st, "get_option", lambda _: "light")

    assert ablation_app._is_dark_theme() is True


def test_dashboard_theme_tokens_are_professional_and_mode_specific():
    light = _dashboard_theme_tokens(dark=False)
    dark = _dashboard_theme_tokens(dark=True)

    assert light["app_bg"] == "#F8FAFC"
    assert light["surface_bg"] == "#FFFFFF"
    assert dark["app_bg"] == "#020617"
    assert dark["surface_bg"] == "#0F172A"
    assert light["accent"] == dark["accent"]
    assert light["radius"] == "6px"


def test_prediction_error_scale_uses_high_contrast_dark_colors():
    dark_scale = _prediction_error_color_scale(dark=True)
    light_scale = _prediction_error_color_scale(dark=False)

    assert dark_scale != light_scale
    assert dark_scale[0][1] == "#67E8F9"
    assert dark_scale[-1][1] == "#FB7185"


def test_prediction_hover_template_is_compact_and_uses_preview_neighbors():
    preview = _preview_neighbors("org/a-very-long-model, org/b, org/c", limit=2)
    template = _prediction_hover_template()

    assert preview.endswith("+1 more")
    assert "nearest_neighbors" not in template
    assert len(template) < 240


def test_figure_axis_ranges_shrink_when_zoomed():
    frame = pd.DataFrame({"x": [-10.0, 0.0, 10.0], "y": [-5.0, 0.0, 5.0]})

    full = _figure_axis_ranges(frame, "x", "y", zoom=1.0)
    zoomed = _figure_axis_ranges(frame, "x", "y", zoom=2.0)

    assert full is None
    assert zoomed is not None
    assert zoomed["x"][1] - zoomed["x"][0] < 20.0
    assert zoomed["y"][1] - zoomed["y"][0] < 10.0


def test_atlas_zoom_control_is_more_sensitive_than_linear():
    assert _atlas_zoom_from_control(1.0) == 1.0
    assert _atlas_zoom_from_control(2.0) > 3.0
    assert _atlas_zoom_from_control(6.0) >= 18.0


def test_dark_mode_galaxy_arm_edges_are_more_visible_than_light_mode():
    light = _galaxy_arm_trace_styles(dark=False)
    dark = _galaxy_arm_trace_styles(dark=True)

    assert dark[1]["opacity"] >= light[1]["opacity"]
    assert dark[1]["line"]["width"] >= light[1]["line"]["width"]


def test_fast_edge_budget_keeps_arm_skeleton_but_bounds_neighbor_links():
    assert _edge_limit_for_density(12_000, "fast", edge_kind="arms") >= 12_000
    assert _edge_limit_for_density(12_000, "fast", edge_kind="neighbors") <= 1_500
    assert _edge_limit_for_density(12_000, "off", edge_kind="arms") == 0


def test_category_color_map_is_stable_and_covers_categories():
    frame = pd.DataFrame({"group": ["b", "a", "b", "c"]})

    first = _category_color_map(frame, "group")
    second = _category_color_map(frame, "group")

    assert first == second
    assert set(first) == {"a", "b", "c"}
    assert all(color.startswith("#") for color in first.values())


def test_fixed_marker_size_is_small_for_dense_atlas_points():
    sizes = _scaled_marker_size(_frame(10), "fixed")

    assert sizes.max() <= 2.4


def test_settled_cluster_centers_are_deterministic_and_separated():
    summary = pd.DataFrame({"cluster": [0, 1, 2], "size": [1400, 900, 500]})

    first = _settled_cluster_centers(summary)
    second = _settled_cluster_centers(summary)

    assert first[0] == second[0]
    assert first[0][0] != 0.0
    distances = [
        ((first[a][0] - first[b][0]) ** 2 + (first[a][1] - first[b][1]) ** 2) ** 0.5
        for a in first
        for b in first
        if a < b
    ]
    assert min(distances) > 9.0


def test_cluster_galaxy_layout_places_nodes_around_cluster_centers_not_pca():
    frame = _with_display_columns(_frame(12))
    frame["Average prediction"] = [float(idx) for idx in range(12)]
    frame["Average absolute_error"] = [float(idx % 3) for idx in range(12)]

    layout = _cluster_galaxy_layout(frame, target="Average")

    assert {
        "galaxy_x",
        "galaxy_y",
        "cluster_center_x",
        "cluster_center_y",
        "galaxy_radius",
        "Cluster center model",
    }.issubset(layout.columns)
    assert layout["galaxy_x"].tolist() != frame["embedding_x"].tolist()
    assert layout.groupby("cluster")["cluster_center_x"].nunique().max() == 1
    assert layout["galaxy_radius"].gt(0).all()


def test_cluster_galaxy_layout_balances_nodes_across_spiral_arms():
    frame = _with_display_columns(_frame(60))
    frame["cluster"] = 0
    frame["Average prediction"] = [float(idx) for idx in range(60)]

    layout = _cluster_galaxy_layout(frame, target="Average")
    arm_counts = layout["galaxy_arm"].value_counts()

    assert arm_counts.size >= 5
    assert int(arm_counts.max() - arm_counts.min()) <= 1


def test_cluster_galaxy_layout_keeps_subtle_spiral_progression():
    frame = _with_display_columns(_frame(120))
    frame["cluster"] = 0
    frame["Average prediction"] = [float(idx) for idx in range(120)]

    layout = _cluster_galaxy_layout(frame, target="Average")
    progressions = []
    for _, arm in layout.groupby("galaxy_arm"):
        if len(arm) < 8:
            continue
        ordered = arm.sort_values("galaxy_arm_order")
        angles = np.unwrap(ordered["galaxy_angle"].to_numpy(dtype=float))
        progressions.append(float(np.corrcoef(ordered["galaxy_arm_order"].to_numpy(dtype=float), angles)[0, 1]))

    assert progressions
    assert float(np.nanmedian(progressions)) > 0.55


def test_cluster_galaxy_layout_has_no_hollow_center():
    frame = _with_display_columns(_frame(96))
    frame["cluster"] = 0
    frame["Average prediction"] = [float(idx) for idx in range(96)]

    layout = _cluster_galaxy_layout(frame, target="Average")

    cluster_radius = 0.95 + 0.165 * (len(layout) ** 0.5)
    assert float(layout["galaxy_radius"].quantile(0.10)) < cluster_radius * 0.22


def test_cluster_galaxy_radius_is_not_a_rank_circle():
    frame = _with_display_columns(_frame(120))
    frame["cluster"] = 0
    frame["Average prediction"] = [float(idx) for idx in range(120)]

    layout = _cluster_galaxy_layout(frame, target="Average")
    rank = layout["Average prediction"].rank(method="first", pct=True)
    correlation = float(np.corrcoef(rank.to_numpy(dtype=float), layout["galaxy_radius"].to_numpy(dtype=float))[0, 1])

    assert correlation < 0.72


def test_cluster_galaxy_plot_has_center_arm_edges_and_no_3d_trace():
    frame = _with_display_columns(_frame(8))
    frame["Average prediction"] = [float(idx) for idx in range(8)]
    frame["Average absolute_error"] = [float(idx % 2) for idx in range(8)]
    layout = _cluster_galaxy_layout(frame, target="Average")

    fig = _plot_cluster_galaxy(layout, color_by="Predicted cluster", size_by="fixed", title="test")

    assert fig.data[0].mode == "lines"
    assert any(trace.name == "representative centers" for trace in fig.data)
    assert {trace.type for trace in fig.data} <= {"scattergl"}


def test_cluster_galaxy_plot_accepts_explicit_zoom_range():
    frame = _with_display_columns(_frame(24))
    frame["Average prediction"] = [float(idx) for idx in range(24)]
    layout = _cluster_galaxy_layout(frame, target="Average")

    fig = _plot_cluster_galaxy(layout, color_by="Predicted cluster", size_by="fixed", title="test", zoom=2.0)

    assert fig.layout.xaxis.range is not None
    assert fig.layout.yaxis.range is not None


def test_cluster_galaxy_arm_edges_chain_nodes_along_arms():
    frame = _with_display_columns(_frame(10))
    frame["cluster"] = 0
    frame["Average prediction"] = [float(idx) for idx in range(10)]
    layout = _cluster_galaxy_layout(frame, target="Average")

    edges = _galaxy_arm_edges(layout)
    edge_segments = list(zip(edges["x"][0::4], edges["y"][0::4], edges["x"][2::4], edges["y"][2::4]))
    center = (layout["cluster_center_x"].iloc[0], layout["cluster_center_y"].iloc[0])

    assert any((x0, y0) != center and (x1, y1) != center for x0, y0, x1, y1 in edge_segments)


def test_cluster_galaxy_arm_edges_are_visible_and_curved():
    frame = _with_display_columns(_frame(18))
    frame["cluster"] = 0
    frame["Average prediction"] = [float(idx) for idx in range(18)]
    layout = _cluster_galaxy_layout(frame, target="Average")

    edges = _galaxy_arm_edges(layout)

    assert len(edges["x"]) % 7 == 0
    assert any(value is not None for value in edges["x"])


def test_cluster_galaxy_arm_edges_are_smooth_curves_not_v_segments():
    frame = _with_display_columns(_frame(18))
    frame["cluster"] = 0
    frame["Average prediction"] = [float(idx) for idx in range(18)]
    layout = _cluster_galaxy_layout(frame, target="Average")

    edges = _galaxy_arm_edges(layout)

    first_segment = list(zip(edges["x"][:6], edges["y"][:6], strict=False))
    assert all(x is not None and y is not None for x, y in first_segment)
    assert len({(round(float(x), 6), round(float(y), 6)) for x, y in first_segment}) == 6


def test_cluster_galaxy_arm_budget_is_distributed_across_clusters():
    frame = _with_display_columns(_frame(80))
    frame["Average prediction"] = [float(idx) for idx in range(80)]
    layout = _cluster_galaxy_layout(frame, target="Average")

    edges = _galaxy_arm_edges(layout, max_edges=layout["cluster"].nunique())
    segment_starts = {
        (round(float(x), 7), round(float(y), 7))
        for x, y in zip(edges["x"][0::7], edges["y"][0::7], strict=False)
        if x is not None and y is not None
    }
    cluster_centers = {
        (round(float(row["cluster_center_x"]), 7), round(float(row["cluster_center_y"]), 7))
        for _, row in layout.drop_duplicates("cluster").iterrows()
    }

    assert cluster_centers.issubset(segment_starts)


def test_cluster_galaxy_plot_has_readable_legend_hover_and_visible_edges():
    frame = _with_display_columns(_frame(12))
    frame["Average prediction"] = [float(idx) for idx in range(12)]
    layout = _cluster_galaxy_layout(frame, target="Average")

    fig = _plot_cluster_galaxy(layout, color_by="Cluster center model", size_by="fixed", title="test")

    assert fig.layout.showlegend is False
    assert fig.layout.hoverlabel.bgcolor == "#F8FAFC"
    assert fig.layout.hoverlabel.font.color == "#0F172A"
    assert fig.data[0].line.width >= 1.0
    assert "rgba" in fig.data[0].line.color
    assert any(trace.name == "representative centers" for trace in fig.data)


def test_predicted_cluster_legend_names_representative_models(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "Predicted cluster": ["C00", "C00", "C01"],
            "Cluster center model": ["center-a", "center-a", "center-b"],
        }
    )
    rendered: list[str] = []

    monkeypatch.setattr(
        ablation_app.st,
        "markdown",
        lambda body, **_: rendered.append(str(body)),
    )

    ablation_app._render_color_legend(frame, "Predicted cluster")

    assert rendered
    assert "center-a (2)" in rendered[0]
    assert "center-b (1)" in rendered[0]
    assert "C00" not in rendered[0]


def test_high_cardinality_color_legend_is_a_compact_data_sized_preview(
    monkeypatch,
) -> None:
    frame = pd.DataFrame({"Category": [f"category-{index}" for index in range(64)]})
    rendered: list[str] = []
    monkeypatch.setattr(
        ablation_app.st,
        "markdown",
        lambda body, **_: rendered.append(str(body)),
    )

    ablation_app._render_color_legend(frame, "Category")

    assert rendered
    assert "+58 more" in rendered[0]


def test_dashboard_chrome_keeps_metric_cards_compact_on_mobile(monkeypatch) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(
        ablation_app.st,
        "markdown",
        lambda body, **_: rendered.append(str(body)),
    )

    ablation_app._render_dashboard_chrome()

    css = rendered[0]
    assert "@media (max-width: 700px)" in css
    assert "grid-template-columns: repeat(2, minmax(0, 1fr))" in css
    assert '[data-testid="stMetricDelta"] p' in css
    assert "white-space: normal !important" in css
    assert "text-overflow: clip !important" in css
    assert '[data-testid="stMetricDelta"] svg' in css
    assert "display: none" in css


def test_dashboard_chrome_leaves_theme_colors_to_streamlit(monkeypatch) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(
        ablation_app.st,
        "markdown",
        lambda body, **_: rendered.append(str(body)),
    )

    ablation_app._render_dashboard_chrome()

    base_css = rendered[0].split("@media (prefers-color-scheme: dark)", maxsplit=1)[0]
    assert ".stApp {" not in base_css
    assert '[data-testid="stSidebar"] {' not in base_css
    assert "background: #0F172A" not in base_css
    assert "color: #F8FAFC" not in base_css
    assert "prefers-color-scheme" not in rendered[0]


def test_target_options_fall_back_to_inferred_metadata_targets():
    cache = type(
        "Cache",
        (),
        {
            "predictions": {},
            "metadata": pd.DataFrame(
                {
                    "display_name": ["a", "b", "c"],
                    "Custom Eval": [0.1, 0.2, 0.3],
                    "downloads": [10, 20, 30],
                }
            ),
        },
    )()

    assert _target_options(cache) == ["Custom Eval"]


def test_dynamic_hover_columns_follow_selected_target():
    frame = _with_display_columns(_frame(3))
    frame["Custom Eval"] = [0.1, 0.2, 0.3]
    frame["Custom Eval prediction"] = [0.11, 0.18, 0.29]
    frame["Custom Eval absolute_error"] = [0.01, 0.02, 0.01]
    frame["Average \u2b06\ufe0f"] = [10.0, 20.0, 30.0]
    frame["Average \u2b06\ufe0f prediction"] = [11.0, 21.0, 31.0]

    hover = _dynamic_hover_columns(frame, "Custom Eval")

    assert "Custom Eval" in hover
    assert "Custom Eval prediction" in hover
    assert "Custom Eval absolute_error" in hover
    assert "Average \u2b06\ufe0f prediction" not in hover


def test_render_point_bounds_scale_with_dataset_size():
    small_max, small_default = _render_point_bounds(800, "fast")
    large_max, large_default = _render_point_bounds(50_000, "fast")

    assert small_max == 800
    assert small_default <= small_max
    assert large_max > 12_000
    assert large_max <= 50_000
    assert large_default <= large_max


def test_fast_render_default_is_conservative_for_startup():
    max_renderable, default_rendered = _render_point_bounds(8_892, "fast")

    assert max_renderable > default_rendered
    assert default_rendered <= 3_000


def test_cluster_galaxy_edges_are_subtle_not_dominant():
    frame = _with_display_columns(_frame(24))
    frame["Average prediction"] = [float(idx) for idx in range(24)]
    layout = _cluster_galaxy_layout(frame, target="Average")

    fig = _plot_cluster_galaxy(layout, color_by="Predicted cluster", size_by="fixed", title="test")
    contrast = next(trace for trace in fig.data if trace.name == "cluster arms contrast")
    foreground = next(trace for trace in fig.data if trace.name == "cluster arms")

    assert contrast.line.width <= 2.0
    assert contrast.opacity <= 0.60
    assert foreground.line.width <= 1.0
    assert foreground.opacity <= 0.65


def test_neighbor_edges_are_curved_and_colored_by_cluster():
    frame = _frame(4)
    frame["cluster"] = [0, 0, 1, 1]
    frame["nearest_neighbors"] = ["model-0001", "model-0000", "model-0003", "model-0002"]
    frame["Average prediction"] = [1.0, 2.0, 3.0, 4.0]
    frame = _with_display_columns(frame, target="Average")
    layout = _cluster_galaxy_layout(frame, target="Average")

    traces = _galaxy_neighbor_edge_traces(layout, color_by="Predicted cluster")

    assert len(traces) == 2
    assert all(len(trace.x) % 4 == 0 for trace in traces)
    color_map = _category_color_map(layout, "Predicted cluster")
    assert {trace.line.color for trace in traces}.issubset(set(color_map.values()))
    assert all(trace.line.width <= 0.8 for trace in traces)
    assert all(trace.opacity <= 0.45 for trace in traces)


def test_display_model_name_is_short_and_stable():
    assert _display_model_name("org/model-name-with-a-very-long-tail") == "model-name-with-a-very-long-tail"[:28] + "..."


def test_cached_config_value_prefers_manifest_setting_with_bounds(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"config": {"n_clusters": 48, "n_neighbors": 100}}), encoding="utf-8")

    assert _cached_config_value(tmp_path, "n_clusters", default=18, minimum=4, maximum=48) == 48
    assert _cached_config_value(tmp_path, "n_neighbors", default=3, minimum=1, maximum=10) == 10
    assert _cached_config_value(tmp_path, "n_bins", default=24, minimum=8, maximum=48) == 24


def test_prediction_lineage_frame_uses_prediction_shaped_coordinates():
    frame = _frame(4)
    predictions = pd.DataFrame(
        {
            "display_name": frame["display_name"],
            "actual": [10.0, 20.0, 30.0, 40.0],
            "prediction": [11.0, 18.0, 28.0, 39.0],
            "absolute_error": [1.0, 2.0, 2.0, 1.0],
            "nearest_neighbors": ["model-0001, model-0002", "model-0000", "model-0003", "model-0002"],
        }
    )
    cache = type("Cache", (), {"predictions": {"Average": predictions}})()

    lineage = _prediction_lineage_frame(cache, "Average", frame)

    assert lineage["lineage_x"].tolist() == predictions["prediction"].tolist()
    assert "lineage_y" in lineage.columns
    assert "lineage_z" in lineage.columns
    assert lineage["lineage_x"].tolist() != frame["embedding_x"].tolist()


def test_prediction_counts_separate_predicted_models_from_known_labels():
    predictions = pd.DataFrame(
        {
            "display_name": ["a", "b", "c"],
            "actual": [1.0, np.nan, 3.0],
            "prediction": [1.2, 2.1, 2.8],
            "absolute_error": [0.2, np.nan, 0.2],
        }
    )

    assert _prediction_counts(predictions) == (3, 2)


def test_prediction_quality_metrics_include_error_and_coverage():
    predictions = pd.DataFrame(
        {
            "display_name": ["a", "b", "c", "d"],
            "actual": [1.0, 2.0, np.nan, 4.0],
            "prediction": [1.2, 1.8, 3.1, 5.0],
            "absolute_error": [0.2, 0.2, np.nan, 1.0],
        }
    )

    metrics = _prediction_quality_metrics(predictions)

    assert metrics["predicted_count"] == 4
    assert metrics["labeled_count"] == 3
    assert metrics["label_coverage"] == 0.75
    assert round(metrics["mae"], 4) == round((0.2 + 0.2 + 1.0) / 3.0, 4)
    assert round(metrics["median_absolute_error"], 4) == 0.2
    assert metrics["rmse"] > metrics["mae"]
    assert metrics["spearman"] is not None
    assert metrics["r2"] is not None
    assert metrics["bias"] is not None


def test_dashboard_summary_cards_report_dataset_and_prediction_quality():
    frame = _frame(4)
    predictions = pd.DataFrame(
        {
            "display_name": frame["display_name"],
            "actual": [1.0, 2.0, np.nan, 4.0],
            "prediction": [1.2, 1.8, 3.1, 5.0],
            "absolute_error": [0.2, 0.2, np.nan, 1.0],
        }
    )
    cache = type(
        "Cache",
        (),
        {
            "metadata": _frame(8),
            "cluster_summary": pd.DataFrame({"dominant_family_share": [0.8, 0.6], "size": [3, 5]}),
            "predictions": {"Average": predictions},
        },
    )()

    cards = _dashboard_summary_cards(cache, frame, "Average")

    labels = [card["label"] for card in cards]
    assert labels == ["Filtered rows", "Clusters", "Families", "Family purity", "MAE", "R2"]
    assert cards[0]["value"] == "4"
    assert cards[0]["delta"] == "8 cached"
    assert cards[5]["value"] != "n/a"


def test_prediction_metric_cards_expose_practical_model_quality_metrics():
    predictions = pd.DataFrame(
        {
            "display_name": ["a", "b", "c", "d"],
            "actual": [1.0, 2.0, np.nan, 4.0],
            "prediction": [1.2, 1.8, 3.1, 5.0],
            "absolute_error": [0.2, 0.2, np.nan, 1.0],
        }
    )

    cards = _prediction_metric_cards(predictions)

    labels = [card["label"] for card in cards]
    assert labels == ["Predicted", "Known labels", "Coverage", "MAE", "RMSE", "R2", "Bias", "Spearman"]
    assert cards[6]["value"] != "n/a"


def test_lineage_quality_metrics_include_same_family_neighbor_rate():
    lineage = pd.DataFrame(
        {
            "display_name": ["a", "b", "c"],
            "coarse_family": ["fam-1", "fam-1", "fam-2"],
            "actual": [1.0, 2.0, np.nan],
            "prediction": [1.2, 1.8, 3.0],
            "absolute_error": [0.2, 0.2, np.nan],
            "nearest_neighbors": ["b, c", "a", "a"],
        }
    )

    metrics = _lineage_quality_metrics(lineage)

    assert metrics["predicted_count"] == 3
    assert metrics["labeled_count"] == 2
    assert metrics["same_family_link_rate"] == 0.5


def test_method_evidence_tables_do_not_label_sample_size_as_audit_rows():
    family = pd.DataFrame({"method": ["paper"], "n_models": [398], "eligible_queries": [398]})
    prediction = pd.DataFrame({"method": ["paper"], "target": ["Average"], "n": [391]})

    display_family, display_prediction = _method_evidence_display_tables(family, prediction, atlas_rows=8892)

    assert "audit_rows" not in display_family.columns
    assert "audit_rows" not in display_prediction.columns
    assert display_family.loc[0, "atlas_models"] == 8892
    assert display_family.loc[0, "study_models"] == 398
    assert display_prediction.loc[0, "labeled_study_rows"] == 391


def test_lineage_edges_follow_cached_prediction_neighbors():
    frame = pd.DataFrame(
        {
            "display_name": ["a", "b", "c"],
            "lineage_x": [1.0, 2.0, 3.0],
            "lineage_y": [0.0, 0.5, 1.0],
            "lineage_z": [0.0, 1.0, 2.0],
            "nearest_neighbors": ["b, c", "a", "a"],
        }
    )

    edges = _lineage_edge_coordinates(frame, max_neighbors=1)

    assert edges["x"][:3] == [1.0, 2.0, None]
    assert edges["y"][:3] == [0.0, 0.5, None]


def test_prediction_lineage_plot_uses_edges_and_svg_nodes():
    frame = pd.DataFrame(
        {
            "display_name": ["a", "b"],
            "lineage_x": [10.0, 12.0],
            "lineage_y": [0.0, 0.3],
            "lineage_z": [1.0, -1.0],
            "lineage_family": ["fam-a", "fam-a"],
            "actual": [11.0, 11.0],
            "prediction": [10.0, 12.0],
            "absolute_error": [1.0, 1.0],
            "nearest_neighbors": ["b", "a"],
        }
    )

    fig = _plot_prediction_lineage(frame, color_by="lineage_family", size_by="fixed", view_mode="2D microscope", target="Average")

    assert fig.data[0].mode == "lines"
    assert any(trace.type == "scattergl" and trace.mode == "markers" for trace in fig.data)
    assert fig.layout.xaxis.visible is False
