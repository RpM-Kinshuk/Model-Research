from __future__ import annotations

import html
import os
from pathlib import Path

# Mitigates OpenBLAS/OpenMP nested-parallel warnings that can appear under Streamlit.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

from alpha_cluster_dashboard.ablation import (
    DEFAULT_ABLATION_METRICS_DIR,
    DEFAULT_ABLATION_STUDY_DATASET,
    DEFAULT_ABLATION_STUDY_DIR,
    ablation_method_options,
    available_prediction_targets,
    build_ablation_similarity_matrix,
    build_knn_prediction_frame,
    load_ablation_study_dataset,
    load_method_study_tables,
    merge_ablation_metadata,
    similarity_to_distance,
)
from alpha_cluster_dashboard.ablation_app import render_ablation_atlas
from alpha_cluster_dashboard.clustering import (
    cluster_distance_matrix,
    cluster_feature_matrix,
    cluster_summary_table,
)
from alpha_cluster_dashboard.data import (
    CANONICAL_MODULE_ORDER,
    COMMON_ALPHA_MODULES,
    available_modules,
    load_alpha_records,
    module_label,
    records_to_frame,
)
from alpha_cluster_dashboard.features import build_prepared_dataset, filter_records
from alpha_cluster_dashboard.lineage import (
    build_lineage_constellation,
    build_lineage_cluster_summary,
    build_lineage_quality_metrics,
    load_lineage_metadata,
    merge_lineage_metadata,
)

SUMMARY_STATS = ["mean", "std", "median", "q25", "q75", "start", "end", "delta", "slope", "min", "max"]


@st.cache_resource(show_spinner=False)
def cached_records(metrics_dir: str):
    return load_alpha_records(metrics_dir)


@st.cache_resource(show_spinner=False)
def cached_catalog(metrics_dir: str):
    return records_to_frame(load_alpha_records(metrics_dir))


@st.cache_data(show_spinner=False)
def cached_lineage_metadata(lineage_path: str):
    return load_lineage_metadata(lineage_path)


@st.cache_data(show_spinner=False)
def cached_ablation_study_dataset(path: str):
    return load_ablation_study_dataset(path)


@st.cache_data(show_spinner=False)
def cached_method_study_tables(study_dir: str):
    return load_method_study_tables(study_dir)


def render_overview(catalog: pd.DataFrame) -> None:
    total_models = int(catalog.shape[0])
    common_models = int(catalog["common_schema"].sum())
    st.subheader("Dataset Overview")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Models", f"{total_models:,}")
    metric_columns[1].metric("Common 7-module schema", f"{common_models:,}")
    metric_columns[2].metric("Schema variants", f"{catalog['schema_name'].nunique():,}")
    metric_columns[3].metric("Layer counts", f"{catalog['num_layers'].nunique():,}")

    layer_counts = catalog["num_layers"].value_counts().sort_index().rename_axis("num_layers").reset_index(name="count")
    st.plotly_chart(
        px.bar(layer_counts, x="num_layers", y="count", title="Models by Layer Count"),
        width="stretch",
    )

    schema_counts = (
        catalog["schema_name"]
        .value_counts()
        .head(12)
        .rename_axis("schema_name")
        .reset_index(name="count")
    )
    st.plotly_chart(
        px.bar(schema_counts, x="count", y="schema_name", orientation="h", title="Top Schema Variants"),
        width="stretch",
    )


def profile_frame(prepared, labels: np.ndarray, cluster_id: int | str) -> pd.DataFrame:
    if cluster_id == "All":
        mask = np.ones(labels.shape[0], dtype=bool)
    else:
        mask = labels == cluster_id

    rows = []
    for module_index, module_name in enumerate(prepared.module_names):
        cluster_profile = np.nanmean(prepared.profiles[mask, module_index, :], axis=0)
        for depth, alpha in zip(prepared.profile_grid, cluster_profile, strict=False):
            rows.append(
                {
                    "cluster": str(cluster_id),
                    "module": module_label(module_name),
                    "depth": depth,
                    "alpha": alpha,
                }
            )
    return pd.DataFrame(rows)


def quality_metric_value(quality_metrics: pd.DataFrame, metric_key: str) -> float | None:
    row = quality_metrics.loc[quality_metrics["metric_key"] == metric_key, "value"]
    if row.empty:
        return None
    value = row.iloc[0]
    return None if pd.isna(value) else float(value)


def metric_display(row: pd.Series) -> str:
    value = row["value"]
    if pd.isna(value):
        return "n/a"
    if row["metric_key"].endswith("_fraction"):
        return f"{value:.1%}"
    if row["metric_key"].endswith("_count"):
        return f"{int(round(value))}"
    return f"{value:.4f}"


def build_lineage_export_html(
    constellation_figure,
    run_configuration: dict[str, object],
    summary_metrics: pd.DataFrame,
    clustering_quality: pd.DataFrame,
    lineage_quality: pd.DataFrame,
    cluster_centers: pd.DataFrame,
) -> bytes:
    def table_block(title: str, frame: pd.DataFrame) -> str:
        return (
            f"<section><h2>{html.escape(title)}</h2>"
            f"{frame.to_html(index=False, escape=True, border=0, classes='metric-table')}"
            "</section>"
        )

    config_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in run_configuration.items()
    )
    plot_html = pio.to_html(
        constellation_figure,
        include_plotlyjs=True,
        full_html=False,
        config={"responsive": True},
    )

    legend_rows: list[str] = []
    for trace in getattr(constellation_figure, "data", []):
        label = getattr(trace, "name", "")
        if not label:
            continue
        marker = getattr(trace, "marker", None)
        color = getattr(marker, "color", None) if marker is not None else None
        if isinstance(color, str):
            legend_rows.append(
                "<tr>"
                f"<td><span class='swatch' style='background:{html.escape(color)};'></span></td>"
                f"<td>{html.escape(label)}</td>"
                "</tr>"
            )
    legend_html = ""
    if legend_rows:
        legend_html = (
            "<section><h2>Cluster Colors</h2>"
            "<table><thead><tr><th>Color</th><th>Cluster</th></tr></thead><tbody>"
            + "".join(legend_rows)
            + "</tbody></table></section>"
        )

    page = f"""<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <title>Alpha Clustering Lineage Report</title>
    <style>
        body {{
            margin: 0;
            padding: 24px;
            font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif;
            background: #f6f7fb;
            color: #1f2430;
        }}
        .container {{
            max-width: 1280px;
            margin: 0 auto;
            display: grid;
            gap: 16px;
        }}
        .panel {{
            background: #ffffff;
            border: 1px solid #d9dfeb;
            border-radius: 10px;
            padding: 16px;
        }}
        h1 {{ margin: 0 0 8px; font-size: 24px; }}
        h2 {{ margin: 0 0 10px; font-size: 18px; }}
        p {{ margin: 0; color: #4f5d78; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #d9dfeb; padding: 6px 8px; font-size: 13px; text-align: left; }}
        th {{ background: #eef2fb; font-weight: 600; }}
        .swatch {{
            display: inline-block;
            width: 18px;
            height: 18px;
            border: 1px solid #6f7d99;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class=\"container\">
        <section class=\"panel\">
            <h1>Alpha Clustering Lineage Report</h1>
            <p>Interactive export of the current base-model constellation and metric snapshots.</p>
        </section>
        <section class=\"panel\">
            <h2>Run Configuration</h2>
            <table>
                <tbody>
                    {config_rows}
                </tbody>
            </table>
        </section>
        <section class=\"panel\">
            <h2>Current Base-Model Constellation</h2>
            {plot_html}
        </section>
        <section class=\"panel\">
            {legend_html}
            {table_block('Run Summary', summary_metrics)}
            {table_block('Clustering Quality Metrics', clustering_quality)}
            {table_block('Lineage Recovery Metrics', lineage_quality)}
            {table_block('Cluster Center Summary', cluster_centers)}
        </section>
    </div>
</body>
</html>
"""
    return page.encode("utf-8")


def main() -> None:
    st.set_page_config(page_title="Model Zoo Spectral Atlas", layout="wide")
    dashboard_mode = st.sidebar.selectbox(
        "Dashboard",
        options=["ablation", "generic"],
        index=0 if DEFAULT_ABLATION_METRICS_DIR.exists() else 1,
        format_func=lambda value: "Model Zoo Spectral Atlas" if value == "ablation" else "Generic Alpha Lab",
    )
    if dashboard_mode == "ablation":
        render_ablation_atlas()
        return

    st.title("Alpha Clustering Lab")
    st.caption(
        "Compare unsupervised clustering strategies over per-layer alpha spectra while handling models "
        "with different depths and module layouts."
    )

    is_ablation_mode = False
    metrics_dir = st.sidebar.text_input("Metrics directory", value=str(Path("svd_results") / "metrics"))
    lineage_path = st.sidebar.text_input("Lineage metadata CSV", value=str(Path("data") / "sampled_metadata.csv"))
    ablation_metadata = pd.DataFrame()
    method_family_metrics = pd.DataFrame()
    method_prediction_metrics = pd.DataFrame()
    ablation_study_dir = ""

    if not Path(metrics_dir).exists():
        st.error(f"Metrics directory not found: {metrics_dir}")
        return

    records = cached_records(metrics_dir)
    catalog = cached_catalog(metrics_dir)
    if not records:
        st.error("No .h5 alpha metric files were found.")
        return

    lineage_metadata = None
    if lineage_path:
        if Path(lineage_path).exists():
            try:
                lineage_metadata = cached_lineage_metadata(lineage_path)
            except ValueError as exc:
                st.warning(str(exc))
        else:
            st.warning(f"Lineage metadata file not found: {lineage_path}")

    ablation_metadata = pd.DataFrame()
    method_family_metrics = pd.DataFrame()
    method_prediction_metrics = pd.DataFrame()
    if is_ablation_mode:
        ablation_metadata = cached_ablation_study_dataset(ablation_dataset_path)
        if ablation_metadata.empty:
            st.warning(f"Ablation study dataset not found or empty: {ablation_dataset_path}")
        method_family_metrics, method_prediction_metrics = cached_method_study_tables(ablation_study_dir)

    render_overview(catalog)

    st.sidebar.subheader("Experiment Controls")
    schema_mode = st.sidebar.selectbox(
        "Schema handling",
        options=["common", "canonical"],
        index=1 if is_ablation_mode else 0,
        format_func=lambda value: "Common 7-module schema" if value == "common" else "Canonical roles across mixed architectures",
    )

    min_layers = int(catalog["num_layers"].min())
    max_layers = int(catalog["num_layers"].max())
    selected_layer_range = st.sidebar.slider(
        "Layer range",
        min_value=min_layers,
        max_value=max_layers,
        value=(min_layers, max_layers),
    )
    model_query = st.sidebar.text_input("Model name filter", value="")
    random_state = st.sidebar.number_input("Random seed", min_value=0, max_value=99999, value=7, step=1)

    records_for_filter = records
    if is_ablation_mode and not ablation_metadata.empty and "model_id" in ablation_metadata.columns:
        use_method_study_subset = st.sidebar.checkbox("Use method-study balanced subset", value=True)
        if use_method_study_subset:
            study_model_ids = set(ablation_metadata["model_id"].dropna().astype(str))
            records_for_filter = [record for record in records if record.display_name in study_model_ids]
            if not records_for_filter:
                st.warning("The method-study subset did not match any loaded alpha records; falling back to all records.")
                records_for_filter = records

    default_max_models = min(400 if is_ablation_mode else 300, len(records_for_filter))
    max_models = st.sidebar.slider("Max models in experiment", min_value=2, max_value=min(2500, len(records_for_filter)), value=default_max_models, step=25)

    candidate_records = filter_records(
        records_for_filter,
        schema_mode=schema_mode,
        min_layers=selected_layer_range[0],
        max_layers=selected_layer_range[1],
        model_query=model_query,
        max_models=max_models,
        random_state=random_state,
    )

    if not candidate_records:
        st.warning("The current filters produced zero models.")
        return
    if len(candidate_records) < 2:
        st.warning("Pick at least two models to run clustering.")
        return

    selectable_modules = available_modules(candidate_records, schema_mode)
    default_modules = (
        [module for module in COMMON_ALPHA_MODULES if module in selectable_modules]
        if schema_mode == "common"
        else [module for module in CANONICAL_MODULE_ORDER if module in selectable_modules]
    )
    selected_modules = st.sidebar.multiselect(
        "Modules",
        options=selectable_modules,
        default=default_modules or selectable_modules,
        format_func=module_label,
    )
    if not selected_modules:
        st.warning("Select at least one module to build features.")
        return

    representation_options = ["summary", "profile", "dtw_profile"]
    if is_ablation_mode:
        representation_options = ["method_similarity", "profile", "summary", "dtw_profile"]
    representation = st.sidebar.selectbox(
        "Representation",
        options=representation_options,
        format_func=lambda value: {
            "summary": "Summary statistics over depth",
            "profile": "Resampled alpha profiles",
            "dtw_profile": "Dynamic time warping over normalized profiles",
            "method_similarity": "Latest ablation method similarity",
        }[value],
    )
    ablation_similarity_method = "hybrid_rank"
    if is_ablation_mode:
        method_options = ablation_method_options()
        ablation_similarity_method = st.sidebar.selectbox(
            "Ablation similarity method",
            options=method_options,
            index=method_options.index("hybrid_rank") if "hybrid_rank" in method_options else 0,
            help="Use structural_basics as a leakage control, paper_spearman as the paper-faithful diagnostic, and hybrid/rank methods for prediction neighborhoods.",
        )
    num_bins = st.sidebar.slider("Normalized depth bins", min_value=8, max_value=48, value=16, step=4)
    summary_stats = st.sidebar.multiselect("Summary statistics", options=SUMMARY_STATS, default=["mean", "std", "median", "slope", "start", "end"])
    include_presence_features = st.sidebar.checkbox("Add module presence indicators", value=True)

    st.sidebar.subheader("Preprocessing")
    sanitize_non_positive = st.sidebar.checkbox("Treat alpha <= 0 as missing", value=True)
    log_transform = st.sidebar.checkbox("Apply log1p(alpha)", value=False)
    zscore_within_trace = st.sidebar.checkbox("Z-score each module trace", value=False)
    dtw_missing_penalty = st.sidebar.slider("DTW missing-module penalty", min_value=0.0, max_value=8.0, value=2.5, step=0.25)

    if representation in {"dtw_profile", "method_similarity"} and len(candidate_records) > 400:
        st.warning(
            "Distance/similarity methods scale quadratically with the number of models. Reduce `Max models` below about 400 for quicker experiments."
        )

    if representation in {"dtw_profile", "method_similarity"}:
        algorithm = st.sidebar.selectbox(
            "Clustering algorithm",
            options=["agglomerative", "dbscan"],
            format_func=lambda value: "Agglomerative" if value == "agglomerative" else "DBSCAN",
        )
        projection = "mds"
    else:
        algorithm = st.sidebar.selectbox(
            "Clustering algorithm",
            options=["kmeans", "agglomerative", "dbscan"],
            format_func=lambda value: {
                "kmeans": "K-Means",
                "agglomerative": "Agglomerative",
                "dbscan": "DBSCAN",
            }[value],
        )
        projection = st.sidebar.selectbox("2D projection", options=["pca", "tsne"], format_func=lambda value: value.upper() if value == "pca" else "t-SNE")

    max_cluster_count = max(2, min(30, len(candidate_records)))
    default_clusters = min(8, max_cluster_count)
    n_clusters = st.sidebar.slider("Target clusters", min_value=2, max_value=max_cluster_count, value=default_clusters)
    linkage = st.sidebar.selectbox("Agglomerative linkage", options=["ward", "average", "complete"], index=0)
    dbscan_eps = st.sidebar.slider("DBSCAN eps", min_value=0.05, max_value=6.0, value=1.25, step=0.05)
    dbscan_min_samples = st.sidebar.slider("DBSCAN min_samples", min_value=2, max_value=20, value=4)

    prepared_representation = "profile" if representation == "method_similarity" else representation
    try:
        prepared = build_prepared_dataset(
            records=candidate_records,
            schema_mode=schema_mode,
            selected_modules=selected_modules,
            representation=prepared_representation,
            num_bins=num_bins,
            summary_stats=summary_stats,
            include_presence_features=include_presence_features,
            sanitize_non_positive=sanitize_non_positive,
            log_transform=log_transform,
            zscore_within_trace=zscore_within_trace,
            dtw_missing_penalty=dtw_missing_penalty,
        )
    except ValueError as exc:
        st.warning(str(exc))
        return

    ablation_similarity_matrix = None
    clustering_distance_matrix = None
    if representation == "method_similarity":
        method_metadata = merge_ablation_metadata(prepared.metadata, ablation_metadata)
        with st.spinner(f"Building {ablation_similarity_method} similarity matrix"):
            ablation_similarity_matrix = build_ablation_similarity_matrix(
                candidate_records,
                ablation_similarity_method,
                metadata=method_metadata,
            )
        clustering_distance_matrix = similarity_to_distance(ablation_similarity_matrix)
        result = cluster_distance_matrix(
            clustering_distance_matrix,
            algorithm=algorithm,
            n_clusters=n_clusters,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
        )
    elif representation == "dtw_profile":
        result = cluster_distance_matrix(
            prepared.distance_matrix,
            algorithm=algorithm,
            n_clusters=n_clusters,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
        )
    else:
        result = cluster_feature_matrix(
            prepared.feature_matrix,
            algorithm=algorithm,
            n_clusters=n_clusters,
            linkage=linkage,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            projection=projection,
            random_state=random_state,
        )

    summary = cluster_summary_table(prepared.metadata, result.labels, prepared.profiles)
    enriched_metadata = merge_lineage_metadata(prepared.metadata, lineage_metadata)
    if is_ablation_mode:
        enriched_metadata = merge_ablation_metadata(enriched_metadata, ablation_metadata)
    lineage_metrics = build_lineage_quality_metrics(
        enriched_metadata,
        result.labels,
        quality_metrics=result.quality_metrics,
    )
    num_clusters = len(set(label for label in result.labels.tolist() if label != -1))
    dunn_index = quality_metric_value(result.quality_metrics, "dunn_index")
    separation_ratio = quality_metric_value(result.quality_metrics, "separation_ratio")

    st.subheader("Experiment Result")
    metrics = st.columns(6)
    metrics[0].metric("Models used", f"{prepared.metadata.shape[0]:,}")
    metrics[1].metric("Modules used", f"{len(prepared.module_names):,}")
    metrics[2].metric("Clusters found", f"{num_clusters:,}")
    metrics[3].metric("Noise points", f"{int(np.sum(result.labels == -1)):,}")
    metrics[4].metric("Silhouette", "n/a" if result.silhouette is None else f"{result.silhouette:.3f}")
    metrics[5].metric("Dunn index", "n/a" if dunn_index is None else f"{dunn_index:.3f}")
    if separation_ratio is not None:
        st.caption(f"Between/within distance ratio: {separation_ratio:.3f}")
    if is_ablation_mode:
        st.caption(
            f"Ablation mode: clustering uses `{ablation_similarity_method}` when the latest-method representation is selected. "
            "`structural_basics` is a leakage control, not a spectral method."
        )

    if is_ablation_mode:
        projection_tab, prediction_tab, method_tab, lineage_tab = st.tabs(
            ["Cluster Projection", "Prediction Neighborhoods", "Method Study", "Base-Model View"]
        )
    else:
        projection_tab, lineage_tab = st.tabs(["Cluster Projection", "Base-Model View"])

    scatter_frame = enriched_metadata.copy()
    scatter_frame["cluster"] = result.labels.astype(str)
    scatter_frame["embedding_x"] = result.embedding[:, 0]
    scatter_frame["embedding_y"] = result.embedding[:, 1]
    scatter_frame["overall_alpha"] = np.nanmean(prepared.profiles, axis=(1, 2))
    lineage_cluster_summary = build_lineage_cluster_summary(scatter_frame, result.labels)

    with projection_tab:
        hover_columns = [
            "num_layers",
            "raw_module_count",
            "overall_alpha",
            "base_model",
            "source_model",
            "base_model_relation",
            "coarse_family",
            "Architecture",
            "#Params (B)",
            "Average \u2b06\ufe0f",
            "MMLU-PRO",
            "MATH Lvl 5",
        ]
        scatter = px.scatter(
            scatter_frame,
            x="embedding_x",
            y="embedding_y",
            color="cluster",
            hover_name="display_name",
            hover_data=[column for column in hover_columns if column in scatter_frame.columns],
            title=f"{result.embedding_name} view of clustered models",
        )
        st.plotly_chart(scatter, width="stretch")

        left_column, right_column = st.columns([1.1, 0.9])
        with left_column:
            st.markdown("**Cluster Summary**")
            st.dataframe(summary, width="stretch", hide_index=True)

            st.markdown("**Clustering Quality**")
            quality_display = result.quality_metrics.copy()
            quality_display["value"] = quality_display.apply(metric_display, axis=1)
            st.dataframe(
                quality_display[["metric", "value", "optimize", "description"]],
                width="stretch",
                hide_index=True,
            )

            export_frame = scatter_frame.copy()
            export_frame["cluster"] = result.labels
            st.download_button(
                "Download cluster assignments",
                data=export_frame.to_csv(index=False).encode("utf-8"),
                file_name="alpha_cluster_assignments.csv",
                mime="text/csv",
            )

        with right_column:
            st.markdown("**Filtered Models**")
            filtered_columns = [
                "display_name",
                "coarse_family",
                "Architecture",
                "#Params (B)",
                "Average \u2b06\ufe0f",
                "MMLU-PRO",
                "MATH Lvl 5",
                "base_model",
                "base_model_relation",
                "num_layers",
                "raw_module_count",
                "path",
            ]
            st.dataframe(
                scatter_frame[[column for column in filtered_columns if column in scatter_frame.columns]],
                width="stretch",
                height=320,
                hide_index=True,
            )

        st.markdown("**Cluster Profiles**")
        cluster_options: list[int | str] = ["All"] + summary["cluster"].tolist()
        selected_cluster = st.selectbox("Cluster to inspect", options=cluster_options)
        cluster_profiles = profile_frame(prepared, result.labels, selected_cluster)
        profile_plot = px.line(
            cluster_profiles,
            x="depth",
            y="alpha",
            color="module",
            title=f"Normalized alpha profiles for cluster {selected_cluster}",
        )
        st.plotly_chart(profile_plot, width="stretch")

        st.markdown("**Module Presence Heatmap**")
        presence_frame = pd.DataFrame(
            np.nanmean(np.isfinite(prepared.profiles), axis=2),
            columns=[module_label(name) for name in prepared.module_names],
        )
        presence_frame.index = prepared.metadata["display_name"]
        heatmap = px.imshow(
            presence_frame,
            aspect="auto",
            color_continuous_scale="Viridis",
            title="Per-model module coverage after preprocessing",
        )
        st.plotly_chart(heatmap, width="stretch")

    if is_ablation_mode:
        with prediction_tab:
            st.markdown("**Prediction Neighborhoods**")
            st.caption(
                "This is leave-one-out kNN over the selected ablation similarity method. It is an association diagnostic, not a causal performance model."
            )
            targets = available_prediction_targets(scatter_frame)
            if not targets:
                st.info("No benchmark targets are available for the current ablation selection.")
            else:
                prediction_controls = st.columns([0.65, 0.35])
                prediction_target = prediction_controls[0].selectbox("Prediction target", options=targets)
                prediction_k = prediction_controls[1].slider("Neighbors", min_value=1, max_value=10, value=3)

                if ablation_similarity_matrix is None:
                    prediction_metadata = merge_ablation_metadata(prepared.metadata, ablation_metadata)
                    with st.spinner(f"Building {ablation_similarity_method} prediction neighborhoods"):
                        prediction_similarity = build_ablation_similarity_matrix(
                            candidate_records,
                            ablation_similarity_method,
                            metadata=prediction_metadata,
                        )
                else:
                    prediction_similarity = ablation_similarity_matrix

                prediction_frame = build_knn_prediction_frame(
                    scatter_frame,
                    prediction_similarity,
                    target=prediction_target,
                    k=prediction_k,
                )
                if prediction_frame.empty:
                    st.info("No finite prediction pairs are available for this target and selection.")
                else:
                    mae = float(prediction_frame["absolute_error"].mean())
                    spearman = prediction_frame["actual"].corr(prediction_frame["prediction"], method="spearman")
                    pred_cards = st.columns(3)
                    pred_cards[0].metric("Prediction rows", f"{len(prediction_frame):,}")
                    pred_cards[1].metric("MAE", f"{mae:.3f}")
                    pred_cards[2].metric("Spearman", "n/a" if pd.isna(spearman) else f"{spearman:.3f}")

                    pred_plot = px.scatter(
                        prediction_frame,
                        x="actual",
                        y="prediction",
                        hover_name="display_name",
                        hover_data=["absolute_error", "nearest_neighbors"],
                        title=f"{prediction_target}: actual vs {prediction_k}-NN prediction via {ablation_similarity_method}",
                    )
                    lower = float(np.nanmin([prediction_frame["actual"].min(), prediction_frame["prediction"].min()]))
                    upper = float(np.nanmax([prediction_frame["actual"].max(), prediction_frame["prediction"].max()]))
                    pred_plot.add_shape(type="line", x0=lower, y0=lower, x1=upper, y1=upper, line={"dash": "dash"})
                    st.plotly_chart(pred_plot, width="stretch")

                    st.markdown("**Largest Absolute Errors**")
                    st.dataframe(
                        prediction_frame.sort_values("absolute_error", ascending=False).head(30),
                        width="stretch",
                        hide_index=True,
                    )

        with method_tab:
            st.markdown("**Latest Ablation Method Study**")
            st.caption(
                "Structural metadata dominates coarse-family recovery; rank-shape spectral neighborhoods are the useful direction for benchmark prediction."
            )
            if not method_family_metrics.empty:
                st.markdown("**Family Retrieval / Leakage Control**")
                st.dataframe(method_family_metrics, width="stretch", hide_index=True)
            if not method_prediction_metrics.empty:
                st.markdown("**kNN Prediction Metrics**")
                st.dataframe(method_prediction_metrics, width="stretch", hide_index=True)

            report_path = Path(ablation_study_dir) / "method_decision_report.md"
            if report_path.exists():
                with st.expander("Read generated method-study report"):
                    st.markdown(report_path.read_text(encoding="utf-8"))

    with lineage_tab:
        if not enriched_metadata["base_model_known"].any():
            st.info("No base-model lineage is available for the current selection.")
        else:
            lineage_run_score = quality_metric_value(lineage_metrics, "lineage_run_score")
            purity_value = quality_metric_value(lineage_metrics, "lineage_cluster_purity")
            v_measure_value = quality_metric_value(lineage_metrics, "lineage_v_measure")
            anchor_recall_value = quality_metric_value(lineage_metrics, "anchor_cluster_recall")
            known_lineage_fraction = quality_metric_value(lineage_metrics, "known_lineage_fraction")
            anchored_family_count = quality_metric_value(lineage_metrics, "anchored_family_count")

            lineage_cards = st.columns(5)
            lineage_cards[0].metric(
                "Lineage run score",
                "n/a" if lineage_run_score is None else f"{lineage_run_score:.3f}",
            )
            lineage_cards[1].metric(
                "Known base-model coverage",
                "n/a" if known_lineage_fraction is None else f"{known_lineage_fraction:.1%}",
            )
            lineage_cards[2].metric(
                "Base-model purity",
                "n/a" if purity_value is None else f"{purity_value:.3f}",
            )
            lineage_cards[3].metric(
                "Lineage V-measure",
                "n/a" if v_measure_value is None else f"{v_measure_value:.3f}",
            )
            lineage_cards[4].metric(
                "Anchor cluster recall",
                "n/a" if anchor_recall_value is None else f"{anchor_recall_value:.3f}",
            )
            if anchored_family_count == 0:
                st.caption(
                    "No actual base-model anchors are present in the current filtered set, so anchor-cluster recall is unavailable and the constellation will use synthetic family centers."
                )

            metrics_column, summary_column = st.columns([1.05, 0.95])
            with metrics_column:
                st.markdown("**Lineage Recovery Metrics**")
                lineage_display = lineage_metrics.copy()
                lineage_display["value"] = lineage_display.apply(metric_display, axis=1)
                st.dataframe(
                    lineage_display[["metric", "value", "optimize", "description"]],
                    width="stretch",
                    hide_index=True,
                )

            with summary_column:
                st.markdown("**Cluster Center Summary**")
                cluster_summary_display = lineage_cluster_summary.copy()
                if not cluster_summary_display.empty:
                    cluster_summary_display["known_lineage_share"] = cluster_summary_display["known_lineage_share"].map(
                        lambda value: f"{value:.1%}"
                    )
                    cluster_summary_display["dominant_base_share"] = cluster_summary_display["dominant_base_share"].map(
                        lambda value: "n/a" if pd.isna(value) else f"{value:.1%}"
                    )
                st.dataframe(
                    cluster_summary_display.head(60),
                    width="stretch",
                    height=340,
                    hide_index=True,
                )

            lineage_controls = st.columns(2)
            min_cluster_size = lineage_controls[0].slider(
                "Minimum cluster size in constellation",
                min_value=1,
                max_value=max(1, min(25, len(scatter_frame))),
                value=2,
                key="lineage_min_cluster_size",
            )
            max_constellation_clusters = lineage_controls[1].slider(
                "Max clusters to show",
                min_value=1,
                max_value=max(1, num_clusters),
                value=min(12, max(1, num_clusters)),
                key="lineage_max_cluster_count",
            )

            constellation_frame, constellation_stats = build_lineage_constellation(
                metadata=scatter_frame,
                labels=result.labels,
                min_cluster_size=min_cluster_size,
                max_clusters=max_constellation_clusters,
            )

            if constellation_frame.empty:
                st.info("No predicted clusters match the current constellation filters.")
            else:
                constellation = px.scatter(
                    constellation_frame,
                    x="plot_x",
                    y="plot_y",
                    color="cluster_label",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    symbol="lineage_role",
                    hover_name="display_name",
                    hover_data=[
                        "base_model",
                        "source_model",
                        "base_model_relation",
                        "dominant_base_model",
                        "cluster_size",
                        "num_layers",
                    ],
                    title="Metric-driven base-model constellation view",
                )
                constellation.update_xaxes(visible=False)
                constellation.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
                st.plotly_chart(constellation, width="stretch")
                st.caption(
                    f"Showing {constellation_stats['shown_models']:,} models across "
                    f"{constellation_stats['shown_clusters']:,} predicted clusters. "
                    f"{constellation_stats['clusters_with_known_center']:,} selected clusters have a known dominant base-model center, "
                    f"and {constellation_stats['clusters_with_anchor_center']:,} use the actual anchor model at the center. "
                    f"{constellation_stats['omitted_clusters']:,} predicted clusters are omitted by the cluster-size or max-cluster filters."
                )

                clustering_quality_export = result.quality_metrics.copy()
                clustering_quality_export["value"] = clustering_quality_export.apply(metric_display, axis=1)
                clustering_quality_export = clustering_quality_export[["metric", "value", "optimize", "description"]]

                lineage_quality_export = lineage_metrics.copy()
                lineage_quality_export["value"] = lineage_quality_export.apply(metric_display, axis=1)
                lineage_quality_export = lineage_quality_export[["metric", "value", "optimize", "description"]]

                center_summary_export = lineage_cluster_summary.copy().head(60)
                if not center_summary_export.empty:
                    center_summary_export["known_lineage_share"] = center_summary_export["known_lineage_share"].map(
                        lambda value: f"{value:.1%}"
                    )
                    center_summary_export["dominant_base_share"] = center_summary_export["dominant_base_share"].map(
                        lambda value: "n/a" if pd.isna(value) else f"{value:.1%}"
                    )

                summary_metrics_export = pd.DataFrame(
                    [
                        {"metric": "Models used", "value": f"{prepared.metadata.shape[0]:,}"},
                        {"metric": "Modules used", "value": f"{len(prepared.module_names):,}"},
                        {"metric": "Clusters found", "value": f"{num_clusters:,}"},
                        {"metric": "Noise points", "value": f"{int(np.sum(result.labels == -1)):,}"},
                        {
                            "metric": "Lineage run score",
                            "value": "n/a" if lineage_run_score is None else f"{lineage_run_score:.3f}",
                        },
                    ]
                )

                run_configuration = {
                    "schema_mode": schema_mode,
                    "representation": representation,
                    "algorithm": algorithm,
                    "projection": projection,
                    "target_clusters": n_clusters,
                    "dbscan_eps": dbscan_eps,
                    "dbscan_min_samples": dbscan_min_samples,
                    "min_constellation_cluster_size": min_cluster_size,
                    "max_constellation_clusters": max_constellation_clusters,
                    "random_state": random_state,
                    "lineage_metadata_path": lineage_path,
                }

                export_payload = build_lineage_export_html(
                    constellation_figure=constellation,
                    run_configuration=run_configuration,
                    summary_metrics=summary_metrics_export,
                    clustering_quality=clustering_quality_export,
                    lineage_quality=lineage_quality_export,
                    cluster_centers=center_summary_export,
                )
                st.download_button(
                    "Download Interactive Lineage Report (HTML)",
                    data=export_payload,
                    file_name="lineage_constellation_report.html",
                    mime="text/html",
                )


if __name__ == "__main__":
    main()
