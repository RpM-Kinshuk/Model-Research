from __future__ import annotations

import colorsys
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from alpha_cluster_dashboard.dashboard_cache import (
    DashboardCache,
    infer_prediction_targets,
    recommended_cluster_count,
)
from alpha_cluster_dashboard.paths import default_scaleup_run, default_study_run
from alpha_cluster_dashboard.scaleup_outputs import (
    REQUIRED_SCALEUP_FILES,
    ScaleupOutputs,
    load_scaleup_outputs,
)
from alpha_cluster_dashboard.study_outputs import (
    REQUIRED_STUDY_FILES,
    StudyOutputs,
    load_study_outputs,
)


PLOT_HEIGHT = 760
SECONDARY_PLOT_HEIGHT = 420
TABLE_HEIGHT = 640
FIXED_MARKER_SIZE = 1.95
MAX_MARKER_SIZE = 4.4
MAX_3D_MARKER_SIZE = 5
GALAXY_EDGE_LIMIT = 9000
GALAXY_ARM_EDGE_LIMIT = 60000
DEFAULT_EDGE_DENSITY = "fast"
PLOT_CONFIG = {
    "scrollZoom": False,
    "doubleClick": "reset",
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}
EDGE_PALETTE = [
    "#2563EB",
    "#059669",
    "#D97706",
    "#7C3AED",
    "#DC2626",
    "#0891B2",
    "#65A30D",
    "#C026D3",
    "#EA580C",
    "#475569",
]


def _plot_config(*, scroll_zoom: bool = False) -> dict[str, object]:
    config = dict(PLOT_CONFIG)
    config["scrollZoom"] = bool(scroll_zoom)
    return config


def _is_dark_theme() -> bool:
    try:
        return st.get_option("theme.base") == "dark"
    except Exception:
        return False


def _chart_theme_tokens(dark: bool | None = None) -> dict[str, str]:
    dark = _is_dark_theme() if dark is None else bool(dark)
    if dark:
        return {
            "template": "plotly_dark",
            "paper_bg": "rgba(15,23,42,0)",
            "plot_bg": "rgba(15,23,42,0.62)",
            "font_color": "#E5E7EB",
            "muted_color": "#94A3B8",
            "grid_color": "rgba(148,163,184,0.18)",
            "axis_color": "rgba(226,232,240,0.56)",
            "hover_bg": "#111827",
            "hover_border": "#94A3B8",
            "hover_font": "#F8FAFC",
        }
    return {
        "template": "plotly_white",
        "paper_bg": "rgba(0,0,0,0)",
        "plot_bg": "rgba(255,255,255,0)",
        "font_color": "#0F172A",
        "muted_color": "#475569",
        "grid_color": "rgba(100,116,139,0.16)",
        "axis_color": "rgba(15,23,42,0.44)",
        "hover_bg": "#F8FAFC",
        "hover_border": "#334155",
        "hover_font": "#0F172A",
        }


def _dashboard_theme_tokens(dark: bool | None = None) -> dict[str, str]:
    dark = _is_dark_theme() if dark is None else bool(dark)
    if dark:
        return {
            "app_bg": "#020617",
            "surface_bg": "#0F172A",
            "surface_subtle": "#111827",
            "text": "#F8FAFC",
            "muted": "#94A3B8",
            "border": "rgba(148,163,184,0.22)",
            "accent": "#0891B2",
            "accent_soft": "rgba(8,145,178,0.18)",
            "shadow": "0 14px 32px rgba(0,0,0,0.26)",
            "radius": "6px",
        }
    return {
        "app_bg": "#F8FAFC",
        "surface_bg": "#FFFFFF",
        "surface_subtle": "#F1F5F9",
        "text": "#0F172A",
        "muted": "#475569",
        "border": "rgba(100,116,139,0.20)",
        "accent": "#0891B2",
        "accent_soft": "rgba(8,145,178,0.10)",
        "shadow": "0 10px 28px rgba(15,23,42,0.07)",
        "radius": "6px",
    }


def _prediction_error_color_scale(dark: bool | None = None) -> list[list[object]]:
    dark = _is_dark_theme() if dark is None else bool(dark)
    if dark:
        return [
            [0.0, "#67E8F9"],
            [0.45, "#FDE68A"],
            [1.0, "#FB7185"],
        ]
    return [
        [0.0, "#2563EB"],
        [0.50, "#D97706"],
        [1.0, "#DC2626"],
    ]


def _preview_neighbors(value: object, *, limit: int = 3) -> str:
    names = [name.strip() for name in str(value or "").split(",") if name.strip()]
    if not names:
        return "n/a"
    visible = [_display_model_name(name, limit=30) for name in names[:limit]]
    suffix = f" +{len(names) - limit} more" if len(names) > limit else ""
    return ", ".join(visible) + suffix


def _prediction_hover_template() -> str:
    return (
        "<b>%{customdata[0]}</b><br>"
        "actual %{customdata[1]:.3f}<br>"
        "predicted %{customdata[2]:.3f}<br>"
        "|error| %{customdata[3]:.3f}<br>"
        "neighbors %{customdata[4]}<extra></extra>"
    )


def _render_dashboard_chrome() -> None:
    base = _dashboard_theme_tokens()
    dark = _dashboard_theme_tokens(dark=True)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {base["app_bg"]};
            color: {base["text"]};
        }}
        .block-container {{
            padding-top: 1.0rem;
            padding-bottom: 1.55rem;
            max-width: 1400px;
        }}
        h1 {{
            letter-spacing: 0;
            font-size: 2.0rem;
            font-weight: 680;
            margin-bottom: 0.1rem;
        }}
        h2, h3 {{
            letter-spacing: 0;
        }}
        [data-testid="stMetric"] {{
            border: 1px solid {base["border"]};
            border-radius: {base["radius"]};
            padding: 0.58rem 0.70rem;
            background: {base["surface_bg"]};
            box-shadow: {base["shadow"]};
        }}
        [data-testid="stMetricLabel"] {{
            font-size: 0.74rem;
            color: {base["muted"]};
            font-weight: 560;
        }}
        [data-testid="stMetricValue"] {{
            font-size: 1.13rem;
            font-weight: 650;
            color: {base["text"]};
        }}
        [data-testid="stMetricDelta"] {{
            font-size: 0.72rem;
            line-height: 1.18;
            color: {base["muted"]};
            align-items: flex-start;
        }}
        [data-testid="stMetricDelta"] p {{
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }}
        [data-testid="stMetricDelta"] svg {{
            display: none;
        }}
        [data-testid="stSidebar"] {{
            background: {base["surface_bg"]};
            border-right: 1px solid {base["border"]};
        }}
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
            gap: 0.48rem;
        }}
        [data-testid="stSidebar"] h3 {{
            font-size: 0.86rem;
            font-weight: 680;
            margin-top: 0.65rem;
            color: {base["text"]};
        }}
        div[role="radiogroup"] {{
            gap: 0.35rem;
        }}
        div[role="radiogroup"] label {{
            border: 1px solid {base["border"]};
            border-radius: {base["radius"]};
            padding: 0.25rem 0.50rem;
            background: {base["surface_bg"]};
        }}
        div[role="radiogroup"] label:hover {{
            border-color: {base["accent"]};
            background: {base["accent_soft"]};
        }}
        .stDataFrame, [data-testid="stTable"] {{
            border-radius: {base["radius"]};
        }}
        .stPlotlyChart {{
            border: 1px solid {base["border"]};
            border-radius: {base["radius"]};
            background: {base["surface_bg"]};
            box-shadow: {base["shadow"]};
            padding: 0.2rem;
        }}
        @media (max-width: 700px) {{
            [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"]) {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.52rem;
            }}
            [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"])
                > [data-testid="stColumn"] {{
                width: auto !important;
                min-width: 0 !important;
                flex: none !important;
            }}
            [data-testid="stMetric"] {{
                min-height: 4.55rem;
            }}
        }}
        @media (prefers-color-scheme: dark) {{
            .stApp {{
                background: {dark["app_bg"]};
                color: {dark["text"]};
            }}
            [data-testid="stMetric"] {{
                border-color: {dark["border"]};
                background: {dark["surface_bg"]};
                box-shadow: {dark["shadow"]};
            }}
            [data-testid="stMetricLabel"] {{
                color: {dark["muted"]};
            }}
            [data-testid="stMetricValue"] {{
                color: {dark["text"]};
            }}
            [data-testid="stMetricDelta"] {{
                color: {dark["muted"]};
            }}
            [data-testid="stSidebar"] {{
                background: {dark["surface_bg"]};
                border-right-color: {dark["border"]};
            }}
            [data-testid="stSidebar"] h3 {{
                color: {dark["text"]};
            }}
            div[role="radiogroup"] label {{
                border-color: {dark["border"]};
                background: {dark["surface_bg"]};
            }}
            div[role="radiogroup"] label:hover {{
                border-color: {dark["accent"]};
                background: {dark["accent_soft"]};
            }}
            .stPlotlyChart {{
                border-color: {dark["border"]};
                background: {dark["surface_bg"]};
                box-shadow: {dark["shadow"]};
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _stable_category_color(value: object) -> str:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    hue = int(digest[:8], 16) / 0xFFFFFFFF
    saturation = 0.58 + 0.16 * (int(digest[8:12], 16) / 0xFFFF)
    lightness = 0.43 + 0.10 * (int(digest[12:16], 16) / 0xFFFF)
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    return f"#{int(red * 255):02X}{int(green * 255):02X}{int(blue * 255):02X}"


def _category_color_map(frame: pd.DataFrame, color_by: str) -> dict[str, str]:
    if frame.empty or color_by not in frame.columns:
        return {}
    values = frame[color_by].dropna()
    if values.empty:
        return {}

    labels = values.astype(str)
    unique_count = labels.nunique()
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_ratio = float(numeric.notna().mean()) if len(values) else 0.0
    integer_like = bool(numeric.notna().any() and np.isclose(numeric.dropna(), np.round(numeric.dropna())).all())
    categorical_numeric = color_by == "cluster" or "cluster" in color_by.lower() or (integer_like and unique_count <= 80)
    if numeric_ratio >= 0.95 and not categorical_numeric:
        return {}

    categories = sorted(labels.unique().tolist())
    return {category: _stable_category_color(category) for category in categories}


def _edge_limit_for_density(point_count: int, density: str, *, edge_kind: str = "neighbors") -> int:
    point_count = max(int(point_count), 0)
    if density == "off" or point_count == 0:
        return 0
    if edge_kind == "arms":
        if density == "dense":
            return min(GALAXY_ARM_EDGE_LIMIT, max(800, int(point_count * 1.20)))
        if density == "balanced":
            return min(40000, max(700, int(point_count * 1.05)))
        return min(24000, max(600, point_count))
    if density == "dense":
        return min(GALAXY_EDGE_LIMIT, max(600, point_count))
    if density == "balanced":
        return min(4500, max(450, int(point_count * 0.65)))
    return min(1500, max(250, int(point_count * 0.28)))


def _format_number(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, float) and abs(value) < 100:
        return f"{value:.3f}"
    return f"{value:,.0f}"


def _format_percent(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.1%}"


def _target_options(cache: DashboardCache) -> list[str]:
    return list(cache.predictions) or infer_prediction_targets(cache.metadata)


def _panel_options() -> list[str]:
    return ["Atlas", "Prediction", "Lineage", "Profiles", "Evidence", "Data"]


def _frozen_run_state_token(run_path: str | Path) -> str:
    return _run_state_token(run_path, REQUIRED_STUDY_FILES)


def _scaleup_run_state_token(run_path: str | Path) -> str:
    return _run_state_token(run_path, REQUIRED_SCALEUP_FILES)


def _run_state_token(
    run_path: str | Path,
    required_files: tuple[str, ...],
) -> str:
    root = Path(run_path).expanduser().resolve()
    state: list[tuple[str, int | None, int | None, int | None]] = []
    for name in required_files:
        try:
            metadata = (root / name).stat()
            state.append(
                (name, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns)
            )
        except OSError:
            state.append((name, None, None, None))
    try:
        artifact_manifest_digest = hashlib.sha256(
            (root / "artifact_manifest.json").read_bytes()
        ).hexdigest()
    except OSError:
        artifact_manifest_digest = "missing"
    payload = json.dumps(
        {
            "run_path": str(root),
            "files": state,
            "artifact_manifest_sha256": artifact_manifest_digest,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@st.cache_data(show_spinner=False, persist="disk")
def _cached_study_outputs(run_path: str, run_state_token: str) -> StudyOutputs:
    del run_state_token
    return load_study_outputs(run_path)


@st.cache_data(show_spinner=False, persist="disk")
def _cached_scaleup_outputs(run_path: str, run_state_token: str) -> ScaleupOutputs:
    del run_state_token
    return load_scaleup_outputs(run_path)


def _development_oof_predictions(outputs: StudyOutputs) -> dict[str, pd.DataFrame]:
    rows = outputs.performance_predictions.copy()
    if rows.empty or "model_id" not in rows.columns:
        return {}
    if "feature_block" in rows.columns:
        rows = rows.loc[rows["feature_block"].astype(str).eq("M+S+H")].copy()
    if "evaluation_scope" in rows.columns:
        rows = rows.loc[
            rows["evaluation_scope"].astype("string").str.casefold().eq("development")
        ].copy()
    elif "external_slice" in rows.columns:
        external_slice = rows["external_slice"].astype("string").fillna("").str.strip()
        rows = rows.loc[external_slice.isin(("", "development"))].copy()
    if rows.empty:
        return {}
    if "target" not in rows.columns:
        rows["target"] = str(outputs.performance_summary.get("target") or "modern_average")

    predictions: dict[str, pd.DataFrame] = {}
    for target, group in rows.groupby("target", sort=False, dropna=True):
        selected = group.drop_duplicates("model_id", keep="first").copy()
        selected["display_name"] = selected["model_id"].astype(str)
        selected["actual"] = pd.to_numeric(selected.get("actual"), errors="coerce")
        selected["prediction"] = pd.to_numeric(selected.get("prediction"), errors="coerce")
        selected["absolute_error"] = (selected["actual"] - selected["prediction"]).abs()
        selected["nearest_neighbors"] = ""
        predictions[str(target)] = selected.loc[
            :, ["display_name", "actual", "prediction", "absolute_error", "nearest_neighbors"]
        ].reset_index(drop=True)
    return predictions


def _frozen_cluster_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    if metadata.empty or "cluster" not in metadata.columns:
        return pd.DataFrame(
            columns=["cluster", "size", "dominant_family", "dominant_family_share", "mean_layers"]
        )
    metadata = metadata.copy()
    clusters = pd.to_numeric(metadata["cluster"], errors="coerce")
    metadata = metadata.loc[clusters.notna() & clusters.ge(0)].copy()
    metadata["cluster"] = clusters.loc[metadata.index].astype(int)
    if metadata.empty:
        return pd.DataFrame(
            columns=["cluster", "size", "dominant_family", "dominant_family_share", "mean_layers"]
        )
    layer_column = next(
        (column for column in ("num_layers", "M__layer_count") if column in metadata.columns),
        None,
    )
    family_column = next(
        (column for column in ("coarse_family", "predicted_family", "audited_base_family") if column in metadata.columns),
        None,
    )
    rows: list[dict[str, object]] = []
    for cluster, group in metadata.groupby("cluster", sort=True):
        families = (
            group[family_column].fillna("unknown").astype(str).value_counts()
            if family_column is not None
            else pd.Series({"unknown": len(group)})
        )
        rows.append(
            {
                "cluster": int(cluster),
                "size": int(len(group)),
                "dominant_family": str(families.index[0]),
                "dominant_family_share": float(families.iloc[0] / len(group)),
                "mean_layers": (
                    float(pd.to_numeric(group[layer_column], errors="coerce").mean())
                    if layer_column is not None
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _frozen_cluster_profiles(metadata: pd.DataFrame) -> pd.DataFrame:
    if metadata.empty or "cluster" not in metadata.columns:
        return pd.DataFrame(columns=["cluster", "role", "depth", "log_alpha", "role_name"])
    metadata = metadata.copy()
    clusters = pd.to_numeric(metadata["cluster"], errors="coerce")
    metadata = metadata.loc[clusters.notna() & clusters.ge(0)].copy()
    metadata["cluster"] = clusters.loc[metadata.index].astype(int)
    if metadata.empty:
        return pd.DataFrame(columns=["cluster", "role", "depth", "log_alpha", "role_name"])
    profile_columns: list[tuple[str, str, str]] = []
    for column in metadata.columns:
        parts = str(column).split("__")
        if len(parts) == 4 and parts[0] == "H" and parts[2:] == ["alpha", "median"]:
            profile_columns.append((str(column), parts[1], "stored DKS alpha"))
    if not profile_columns:
        return pd.DataFrame(columns=["cluster", "role", "depth", "log_alpha", "role_name"])

    role_names = list(dict.fromkeys(role_name for _, role_name, _ in profile_columns))
    role_depth = {
        role_name: index / max(len(role_names) - 1, 1)
        for index, role_name in enumerate(role_names)
    }
    rows: list[dict[str, object]] = []
    for cluster, group in metadata.groupby("cluster", sort=True):
        for column, role_name, profile_name in profile_columns:
            alpha = pd.to_numeric(group[column], errors="coerce")
            valid_alpha = alpha.loc[alpha.gt(-1.0) & alpha.notna()]
            log_alpha = float(np.log1p(valid_alpha).mean()) if not valid_alpha.empty else np.nan
            rows.append(
                {
                    "cluster": int(cluster),
                    "role": profile_name,
                    "depth": float(role_depth[role_name]),
                    "log_alpha": log_alpha,
                    "role_name": role_name,
                }
            )
    return pd.DataFrame(rows)


def _dashboard_cache_from_study(outputs: StudyOutputs) -> DashboardCache:
    """Adapt validated frozen tables to the existing 2D dashboard helpers."""

    metadata = outputs.atlas.copy().reset_index(drop=True)
    if "model_id" not in metadata.columns:
        metadata["model_id"] = metadata.get("display_name", pd.Series(index=metadata.index, dtype=str))
    metadata["model_id"] = metadata["model_id"].astype(str)
    metadata["display_name"] = metadata.get("display_name", metadata["model_id"]).fillna(metadata["model_id"]).astype(str)
    if "cluster" not in metadata.columns:
        metadata["cluster"] = pd.Series(pd.NA, index=metadata.index, dtype="Int64")
    else:
        metadata["cluster"] = pd.to_numeric(
            metadata["cluster"], errors="coerce"
        ).astype("Int64")

    predictions = _development_oof_predictions(outputs)
    for target, table in predictions.items():
        indexed = table.set_index("display_name")
        metadata[target] = metadata["display_name"].map(indexed["actual"])
        metadata[f"{target} prediction"] = metadata["display_name"].map(indexed["prediction"])
        metadata[f"{target} absolute_error"] = metadata["display_name"].map(indexed["absolute_error"])

    if "predicted_family" in metadata.columns:
        metadata["Predicted family"] = metadata["predicted_family"].fillna("unavailable").astype(str)
    if "correct" in metadata.columns:
        correctness = metadata["correct"].astype("string").str.casefold()
        metadata["Lineage correctness"] = np.select(
            [
                correctness.eq("true").fillna(False).to_numpy(dtype=bool),
                correctness.eq("false").fillna(False).to_numpy(dtype=bool),
            ],
            ["correct", "incorrect"],
            default="unavailable",
        )
    if "lineage_endpoint" in metadata.columns:
        metadata["Lineage endpoint"] = (
            metadata["lineage_endpoint"]
            .astype("string")
            .str.replace("_", " ", regex=False)
            .fillna("unavailable")
        )

    return DashboardCache(
        cache_dir=outputs.run_dir,
        metadata=metadata,
        features=np.empty((len(metadata), 0), dtype=float),
        feature_names=[],
        predictions=predictions,
        cluster_profiles=_frozen_cluster_profiles(metadata),
        cluster_summary=_frozen_cluster_summary(metadata),
        manifest={"provenance": dict(outputs.provenance), "summary": dict(outputs.summary)},
    )


def _dashboard_cache_from_scaleup(
    outputs: StudyOutputs,
    scaleup: ScaleupOutputs,
) -> DashboardCache:
    """Overlay a validated retained run with read-only scale-up results."""

    metadata = scaleup.population.copy().reset_index(drop=True)
    base = outputs.atlas.copy()
    extra_columns = [
        column
        for column in base.columns
        if column == "model_id" or column not in metadata.columns
    ]
    metadata = metadata.merge(
        base.loc[:, extra_columns],
        on="model_id",
        how="left",
        validate="one_to_one",
    )
    metadata["display_name"] = metadata["model_id"].astype(str)
    metadata["cluster"] = pd.to_numeric(
        metadata["strict_cluster"], errors="coerce"
    ).astype("Int64")
    predicted_family = metadata.get(
        "predicted_family", pd.Series(pd.NA, index=metadata.index)
    )
    audited_family = metadata.get(
        "audited_base_family", pd.Series(pd.NA, index=metadata.index)
    )
    metadata["coarse_family"] = predicted_family.fillna(audited_family).fillna(
        "unknown"
    )
    metadata["Predicted family"] = predicted_family.fillna("unavailable").astype(str)
    if "lineage_evidence_status" in metadata.columns:
        metadata["Lineage evidence"] = (
            metadata["lineage_evidence_status"]
            .astype("string")
            .str.replace("_", " ", regex=False)
            .fillna("unavailable")
        )

    predictions = {
        target: table.copy().reset_index(drop=True)
        for target, table in scaleup.predictions.items()
    }
    for target, table in predictions.items():
        indexed = table.set_index("display_name")
        metadata[target] = metadata["display_name"].map(indexed["actual"])
        metadata[f"{target} prediction"] = metadata["display_name"].map(
            indexed["prediction"]
        )
        metadata[f"{target} absolute_error"] = metadata["display_name"].map(
            indexed["absolute_error"]
        )

    return DashboardCache(
        cache_dir=scaleup.run_dir,
        metadata=metadata,
        features=np.empty((len(metadata), 0), dtype=float),
        feature_names=[],
        predictions=predictions,
        cluster_profiles=_frozen_cluster_profiles(metadata),
        cluster_summary=_frozen_cluster_summary(metadata),
        manifest={
            "provenance": dict(scaleup.metadata),
            "summary": dict(scaleup.counts),
            "base_run": str(outputs.run_dir),
        },
    )


def _study_summary_cards(
    outputs: StudyOutputs,
    scaleup: ScaleupOutputs | None = None,
) -> list[dict[str, str]]:
    if scaleup is not None:
        modern = scaleup.performance.get("modern_average", {})
        legacy = scaleup.performance.get("legacy_average", {})

        def count(key: str) -> str:
            value = scaleup.counts.get(key)
            return f"{int(value):,}" if value is not None else "n/a"

        modern_models = modern.get("n_models")
        modern_components = modern.get("n_components")
        modern_scope = (
            f"{int(modern_models):,} / {int(modern_components):,}"
            if modern_models is not None and modern_components is not None
            else "internal grouped"
        )
        return [
            {"label": "Census models", "value": count("phase2_census")},
            {"label": "HTSR artifacts", "value": count("phase2_htsr_artifacts")},
            {"label": "Strict clustered", "value": count("phase2_strict_lineage")},
            {
                "label": "Modern MAE",
                "value": _format_number(modern.get("group_weighted_mae")),
                "delta": modern_scope,
            },
            {
                "label": "Modern dMAE",
                "value": _format_number(modern.get("delta_mae")),
                "delta": "direct alpha vs M+S+H",
            },
            {
                "label": "Legacy dMAE",
                "value": _format_number(legacy.get("delta_mae")),
                "delta": "direct alpha vs M+S+H",
            },
        ]

    lineage = outputs.lineage_summary
    performance = outputs.performance_summary
    family_count = lineage.get("n_families")
    evidence_status = (
        "inferential" if lineage.get("inferential") is True else "descriptive"
    )
    lineage_scope = (
        f"{int(family_count)} families / {evidence_status}"
        if family_count is not None
        else evidence_status
    )
    external_delta = performance.get("external_htsr_delta_mae")
    external_card = {"label": "External dMAE", "value": "Unavailable"}
    if outputs.external_status in {"available", "descriptive", "descriptive_reuse"} and external_delta is not None:
        external_card["value"] = _format_number(external_delta)
        external_models = performance.get("external_n_models")
        external_groups = performance.get("external_n_components")
        if external_models is not None and external_groups is not None:
            prefix = "reused; " if outputs.external_status == "descriptive_reuse" else ""
            external_card["delta"] = f"{prefix}{int(external_models):,} / {int(external_groups):,}"
    observed_lift = lineage.get("observed_htsr_lift")
    observed_delta = "descriptive"
    if observed_lift is not None and pd.notna(observed_lift):
        observed_delta = f"HTSR lift {100.0 * float(observed_lift):+.1f} pp"
    decision_delta = (
        "claim passed" if performance.get("primary_passed") is True else "claim failed"
    )
    return [
        {"label": "Manifest models", "value": f"{len(outputs.manifest):,}"},
        {
            "label": "Root top-1",
            "value": _format_percent(lineage.get("root_macro_top1")),
            "delta": lineage_scope,
        },
        {
            "label": "Known-family top-1",
            "value": _format_percent(lineage.get("observed_family_top1")),
            "delta": observed_delta,
        },
        {"label": "Grouped MAE", "value": _format_number(performance.get("group_weighted_mae"))},
        {
            "label": "HTSR dMAE",
            "value": _format_number(performance.get("htsr_delta_mae")),
            "delta": decision_delta,
        },
        external_card,
    ]


def _manifest_model_count(cache_dir: str | Path) -> int | None:
    manifest_path = Path(cache_dir) / "manifest.json"
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8")).get("model_count")
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count > 0 else None


def _metric_file_count(metrics_dir: str | Path) -> int | None:
    root = Path(metrics_dir)
    if not root.exists():
        return None
    try:
        count = len(list(root.glob("*.h5")) + list(root.glob("*.csv")))
    except OSError:
        return None
    return count if count > 0 else None


def _cluster_slider_values(cache_dir: str | Path, metrics_dir: str | Path) -> tuple[int, int, int]:
    model_count = _manifest_model_count(cache_dir) or _metric_file_count(metrics_dir)
    default = recommended_cluster_count(model_count)
    minimum = 1 if model_count == 1 else 2
    maximum = max(minimum, min(int(model_count or default), max(default * 2, minimum)))
    value = _cached_config_value(cache_dir, "n_clusters", default=default, minimum=minimum, maximum=maximum)
    return minimum, maximum, value


def _cached_config_value(cache_dir: str | Path, key: str, *, default: int, minimum: int, maximum: int) -> int:
    manifest_path = Path(cache_dir) / "manifest.json"
    try:
        config = json.loads(manifest_path.read_text(encoding="utf-8")).get("config", {})
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default
    try:
        value = int(config.get(key, default))
    except (TypeError, ValueError):
        return default
    return int(np.clip(value, minimum, maximum))


def _display_model_name(value: object, limit: int = 31) -> str:
    text = str(value).split("/")[-1] if value is not None else "unknown"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _edge_color_for_key(value: object) -> str:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    color = EDGE_PALETTE[int(digest[:8], 16) % len(EDGE_PALETTE)]
    red = int(color[1:3], 16)
    green = int(color[3:5], 16)
    blue = int(color[5:7], 16)
    return f"rgba({red},{green},{blue},0.62)"


def _cluster_center_models(frame: pd.DataFrame, target: str | None = None) -> dict[int, str]:
    if frame.empty or "cluster" not in frame.columns:
        return {}
    clusters = pd.to_numeric(frame["cluster"], errors="coerce")
    frame = frame.loc[clusters.notna() & clusters.ge(0)].copy()
    frame["cluster"] = clusters.loc[frame.index].astype(int)
    if frame.empty:
        return {}
    prediction_column = f"{target} prediction" if target else ""
    error_column = f"{target} absolute_error" if target else ""
    numeric_candidates = [
        column
        for column in [prediction_column, target or "", error_column, "overall_log_alpha", "#Params (B)", "num_layers"]
        if column and column in frame.columns
    ]
    result: dict[int, str] = {}
    for cluster, group in frame.groupby("cluster", sort=True):
        if group.empty:
            continue
        if numeric_candidates:
            scores = pd.Series(0.0, index=group.index)
            for column in numeric_candidates:
                values = pd.to_numeric(group[column], errors="coerce")
                if not values.notna().any():
                    continue
                center = float(values.median())
                spread = float(values.quantile(0.75) - values.quantile(0.25))
                spread = max(spread, float(values.std(skipna=True) or 0.0), 1e-6)
                scores = scores + ((values.fillna(center) - center).abs() / spread)
            if error_column in group.columns:
                errors = pd.to_numeric(group[error_column], errors="coerce")
                if errors.notna().any():
                    err_spread = max(float(errors.quantile(0.75) - errors.quantile(0.25)), 1e-6)
                    scores = scores + 0.35 * (errors.fillna(float(errors.median())) / err_spread)
            chosen_index = scores.sort_values(kind="mergesort").index[0]
        else:
            chosen_index = group.index[0]
        result[int(cluster)] = str(group.loc[chosen_index, "display_name"])
    return result


def _with_display_columns(frame: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    result = frame.copy()
    if "cluster" in result.columns and "Predicted cluster" not in result.columns:
        clusters = pd.to_numeric(result["cluster"], errors="coerce")
        result["Predicted cluster"] = [
            f"C{int(value):02d}" if pd.notna(value) and value >= 0 else "unassigned"
            for value in clusters
        ]
        center_models = _cluster_center_models(result, target)
        result["Cluster center model"] = [
            _display_model_name(center_models.get(int(value), "unknown"))
            if pd.notna(value) and value >= 0
            else "unassigned"
            for value in clusters
        ]
    prediction_column = f"{target} prediction" if target else ""
    if prediction_column in result.columns and "Prediction band" not in result.columns:
        predictions = pd.to_numeric(result[prediction_column], errors="coerce")
        if predictions.notna().nunique() >= 4:
            bands = pd.qcut(predictions.rank(method="first"), q=4, labels=["low", "mid-low", "mid-high", "high"])
            result["Prediction band"] = bands.astype("object").fillna("unknown")
    return result


def _color_options(frame: pd.DataFrame, target: str | None) -> list[str]:
    options = []
    if "Cluster center model" in frame.columns:
        options.append("Cluster center model")
    if "Predicted cluster" in frame.columns:
        options.append("Predicted cluster")
    if "cluster" in frame.columns:
        options.append("cluster")
    for column in (
        "coarse_family",
        "Lineage endpoint",
        "Architecture",
        "Type",
        "Weight type",
    ):
        if column in frame.columns:
            options.append(column)
    if target and target in frame.columns:
        options.append(target)
        for suffix in ("prediction", "absolute_error"):
            column = f"{target} {suffix}"
            if column in frame.columns:
                options.append(column)
    return options


def _lineage_color_options(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in (
            "Predicted cluster",
            "Predicted family",
            "Lineage endpoint",
            "Lineage correctness",
        )
        if column in frame.columns
    ]


def _dynamic_hover_columns(frame: pd.DataFrame, target: str | None) -> list[str]:
    preferred = [
        "Predicted cluster",
        "Cluster center model",
        "coarse_family",
        "Lineage endpoint",
        "Architecture",
        "Type",
        "Weight type",
        "#Params (B)",
    ]
    if target:
        preferred.extend([target, f"{target} prediction", f"{target} absolute_error"])
    preferred.extend(
        [
            "actual",
            "prediction",
            "absolute_error",
            "nearest_neighbors",
            "lineage_family",
            "num_layers",
            "overall_log_alpha",
        ]
    )

    columns: list[str] = []
    seen: set[str] = set()
    for column in preferred:
        if column in frame.columns and column not in seen:
            columns.append(column)
            seen.add(column)
    return columns


def _filter_frame(
    frame: pd.DataFrame,
    families: list[str],
    architectures: list[str],
    clusters: list[int],
    search_text: str = "",
) -> pd.DataFrame:
    result = frame.copy()
    if families and "coarse_family" in result.columns:
        result = result[result["coarse_family"].fillna("unknown").astype(str).isin(families)]
    if architectures and "Architecture" in result.columns:
        result = result[result["Architecture"].fillna("unknown").astype(str).isin(architectures)]
    if clusters:
        result = result[pd.to_numeric(result["cluster"], errors="coerce").isin(clusters)]
    query = search_text.strip().lower()
    if query:
        searchable_columns = [column for column in ("display_name", "Model", "id") if column in result.columns]
        if searchable_columns:
            mask = pd.Series(False, index=result.index)
            for column in searchable_columns:
                mask = mask | result[column].fillna("").astype(str).str.lower().str.contains(query, regex=False)
            result = result[mask]
    return result


def _render_point_bounds(total_points: int, edge_density: str) -> tuple[int, int]:
    total = max(int(total_points), 0)
    if total <= 0:
        return 0, 0
    if edge_density == "off":
        return total, total

    scale_by_density = {"fast": 92.0, "balanced": 70.0, "dense": 54.0}
    scale = scale_by_density.get(edge_density, scale_by_density[DEFAULT_EDGE_DENSITY])
    root = float(np.sqrt(total))
    max_points = min(total, max(1, int(round(root * scale))))
    default_scale = min(scale, 3.0 * float(np.sqrt(scale)))
    default_points = min(max_points, max(1, int(round(root * default_scale))))
    return max_points, default_points


def _deterministic_jitter(value: object, scale: float = 0.16) -> float:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    unit = int(digest[:8], 16) / 0xFFFFFFFF
    return float((unit - 0.5) * scale)


def _deterministic_unit(value: object) -> float:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    return float(int(digest[:8], 16) / 0xFFFFFFFF)


def _scaled_marker_size(frame: pd.DataFrame, size_by: str) -> pd.Series:
    if size_by == "fixed" or size_by not in frame.columns:
        return pd.Series(FIXED_MARKER_SIZE, index=frame.index)
    numeric = pd.to_numeric(frame[size_by], errors="coerce")
    if not numeric.notna().any():
        return pd.Series(FIXED_MARKER_SIZE, index=frame.index)
    lower = float(numeric.quantile(0.05))
    upper = float(numeric.quantile(0.95))
    denom = max(upper - lower, 1e-8)
    return 1.8 + 3.2 * ((numeric.clip(lower, upper) - lower) / denom)


def _apply_atlas_style(fig, *, hide_axes: bool = True, dark: bool | None = None):
    tokens = _chart_theme_tokens(dark)
    marker_traces = [
        trace
        for trace in fig.data
        if "markers" in str(getattr(trace, "mode", "markers")) or getattr(trace, "type", "") in {"scatter3d"}
    ]
    fig.update_layout(
        template=tokens["template"],
        paper_bgcolor=tokens["paper_bg"],
        plot_bgcolor=tokens["paper_bg"],
        font={"color": tokens["font_color"]},
        hoverlabel={
            "bgcolor": tokens["hover_bg"],
            "bordercolor": tokens["hover_border"],
            "font": {"color": tokens["hover_font"], "size": 12},
        },
        dragmode="pan",
        showlegend=False,
    )
    if hide_axes:
        fig.update_xaxes(visible=False, showgrid=False, zeroline=False)
        fig.update_yaxes(visible=False, showgrid=False, zeroline=False)
    if "coloraxis" in fig.layout:
        fig.update_layout(coloraxis_colorbar={"thickness": 10, "len": 0.72, "tickfont": {"color": tokens["font_color"]}})
    return fig


def _apply_analytics_style(fig, *, dark: bool | None = None):
    tokens = _chart_theme_tokens(dark)
    fig.update_layout(
        template=tokens["template"],
        paper_bgcolor=tokens["paper_bg"],
        plot_bgcolor=tokens["plot_bg"],
        font={"color": tokens["font_color"]},
        hoverlabel={
            "bgcolor": tokens["hover_bg"],
            "bordercolor": tokens["hover_border"],
            "font": {"color": tokens["hover_font"], "size": 12},
        },
        margin={"l": 8, "r": 8, "t": 42, "b": 8},
    )
    fig.update_xaxes(gridcolor=tokens["grid_color"], linecolor=tokens["axis_color"], zerolinecolor=tokens["grid_color"])
    fig.update_yaxes(gridcolor=tokens["grid_color"], linecolor=tokens["axis_color"], zerolinecolor=tokens["grid_color"])
    if "coloraxis" in fig.layout:
        fig.update_layout(coloraxis_colorbar={"thickness": 10, "len": 0.72, "tickfont": {"color": tokens["font_color"]}})
    return fig


def _figure_axis_ranges(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    *,
    zoom: float = 1.0,
) -> dict[str, tuple[float, float]] | None:
    try:
        zoom_value = float(zoom)
    except (TypeError, ValueError):
        zoom_value = 1.0
    if zoom_value <= 1.0 or frame.empty or x_column not in frame.columns or y_column not in frame.columns:
        return None

    x_values = pd.to_numeric(frame[x_column], errors="coerce")
    y_values = pd.to_numeric(frame[y_column], errors="coerce")
    valid = x_values.notna() & y_values.notna()
    if not valid.any():
        return None

    ranges: dict[str, tuple[float, float]] = {}
    for key, values in {"x": x_values[valid], "y": y_values[valid]}.items():
        lower = float(values.min())
        upper = float(values.max())
        center = (lower + upper) / 2.0
        span = max(upper - lower, 1e-6)
        half_span = (span / 2.0) / zoom_value
        ranges[key] = (center - half_span, center + half_span)
    return ranges


def _atlas_zoom_from_control(value: float | int) -> float:
    try:
        control = float(value)
    except (TypeError, ValueError):
        control = 1.0
    control = float(np.clip(control, 1.0, 6.0))
    if control <= 1.0:
        return 1.0
    return float(min(24.0, 1.0 + 2.2 * ((control - 1.0) ** 1.55)))


def _apply_axis_zoom(fig, frame: pd.DataFrame, x_column: str, y_column: str, *, zoom: float) -> None:
    ranges = _figure_axis_ranges(frame, x_column, y_column, zoom=zoom)
    if ranges is None:
        return
    fig.update_xaxes(range=list(ranges["x"]))
    fig.update_yaxes(range=list(ranges["y"]))


def _galaxy_arm_trace_styles(*, dark: bool) -> list[dict[str, object]]:
    if dark:
        return [
            {
                "line": {"color": "rgba(226,232,240,0.22)", "width": 1.9},
                "opacity": 0.60,
                "name": "cluster arms contrast",
            },
            {
                "line": {"color": "rgba(148,163,184,0.62)", "width": 1.05},
                "opacity": 0.66,
                "name": "cluster arms",
            },
        ]
    return [
        {
            "line": {"color": "rgba(255,255,255,0.22)", "width": 1.8},
            "opacity": 0.55,
            "name": "cluster arms contrast",
        },
        {
            "line": {"color": "rgba(51,65,85,0.48)", "width": 0.9},
            "opacity": 0.58,
            "name": "cluster arms",
        },
    ]


def _settled_cluster_centers(summary: pd.DataFrame, iterations: int = 180) -> dict[int, tuple[float, float]]:
    if summary.empty:
        return {}
    data = summary[["cluster", "size"]].copy()
    data["cluster"] = pd.to_numeric(data["cluster"], errors="coerce").astype(int)
    data["size"] = pd.to_numeric(data["size"], errors="coerce").fillna(1.0).clip(lower=1.0)
    data = data.sort_values(["size", "cluster"], ascending=[False, True]).reset_index(drop=True)

    n_clusters = len(data)
    if n_clusters == 1:
        return {int(data.loc[0, "cluster"]): (0.0, 0.0)}

    sizes = data["size"].to_numpy(dtype=float)
    radii = 1.20 + 0.19 * np.sqrt(sizes)
    anchor_radius = 2.90 * np.sqrt(n_clusters) + float(np.mean(radii))
    positions = np.zeros((n_clusters, 2), dtype=float)
    for index, cluster in enumerate(data["cluster"].tolist()):
        digest = hashlib.sha1(f"cluster-center:{cluster}".encode("utf-8")).hexdigest()
        angle_unit = int(digest[:8], 16) / 0xFFFFFFFF
        radius_unit = int(digest[8:16], 16) / 0xFFFFFFFF
        angle = 2.0 * np.pi * angle_unit
        radius = anchor_radius * (0.28 + 0.72 * np.sqrt(radius_unit))
        positions[index] = [radius * np.cos(angle), radius * np.sin(angle)]

    anchors = positions.copy()
    velocity = np.zeros_like(positions)
    for _ in range(iterations):
        force = 0.014 * (anchors - positions)
        for left in range(n_clusters):
            delta = positions[left] - positions
            distances = np.sqrt(np.sum(delta * delta, axis=1)) + 1e-6
            desired = radii[left] + radii + 2.25
            overlap = np.maximum(desired - distances, 0.0)
            overlap[left] = 0.0
            repulsion = (delta / distances[:, None]) * overlap[:, None] * 0.070
            force[left] += repulsion.sum(axis=0)
        velocity = 0.70 * (velocity + force)
        positions += velocity

    positions -= positions.mean(axis=0)
    return {
        int(cluster): (float(position[0]), float(position[1]))
        for cluster, position in zip(data["cluster"].tolist(), positions, strict=False)
    }


def _cluster_galaxy_layout(frame: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    layout = frame.copy()
    if "cluster" not in layout.columns:
        layout["cluster"] = 0
    clusters = pd.to_numeric(layout["cluster"], errors="coerce")
    layout = layout.loc[clusters.notna() & clusters.ge(0)].copy()
    layout["cluster"] = clusters.loc[layout.index].astype(int)
    if layout.empty:
        return layout
    if "Cluster center model" not in layout.columns:
        center_models = _cluster_center_models(layout, target)
        layout["Cluster center model"] = layout["cluster"].map(lambda value: _display_model_name(center_models.get(int(value), "unknown")))
    summary = layout.groupby("cluster", as_index=False).size().rename(columns={"size": "size"})
    centers = _settled_cluster_centers(summary)
    prediction_column = f"{target} prediction" if target else ""
    error_column = f"{target} absolute_error" if target else ""

    x_values = pd.Series(np.nan, index=layout.index, dtype=float)
    y_values = pd.Series(np.nan, index=layout.index, dtype=float)
    center_x = pd.Series(np.nan, index=layout.index, dtype=float)
    center_y = pd.Series(np.nan, index=layout.index, dtype=float)
    radius_values = pd.Series(np.nan, index=layout.index, dtype=float)
    arm_values = pd.Series(-1, index=layout.index, dtype=int)
    arm_order_values = pd.Series(-1, index=layout.index, dtype=int)
    angle_values = pd.Series(np.nan, index=layout.index, dtype=float)

    for cluster, group in layout.groupby("cluster", sort=True):
        cx, cy = centers.get(int(cluster), (0.0, 0.0))
        n = len(group)
        cluster_radius = 0.95 + 0.165 * np.sqrt(max(n, 1))

        if prediction_column in group.columns:
            score = pd.to_numeric(group[prediction_column], errors="coerce")
        elif target and target in group.columns:
            score = pd.to_numeric(group[target], errors="coerce")
        else:
            score = pd.Series(np.arange(n, dtype=float), index=group.index)
        score_rank = score.rank(method="first", pct=True).fillna(0.5)

        arm_count = int(np.clip(round(np.sqrt(max(n, 1))), 3, 18))
        if n < 12:
            arm_count = int(np.clip(round(np.sqrt(max(n, 1))), 2, 5))
        ordering = (
            pd.DataFrame(
                {
                    "row_index": group.index,
                    "score": score_rank,
                    "jitter": [_deterministic_jitter(name, scale=1.0) for name in group["display_name"]],
                }
            )
            .sort_values(["jitter", "score", "row_index"])
            .reset_index(drop=True)
        )
        row_to_arm: dict[object, int] = {}
        row_to_arm_order: dict[object, int] = {}
        arm_loads = np.zeros(arm_count, dtype=int)
        arm_capacity = int(np.ceil(max(n, 1) / arm_count))
        for order, row_index in enumerate(ordering["row_index"].tolist()):
            name = str(layout.loc[row_index, "display_name"]) if "display_name" in layout.columns else str(row_index)
            preferred = int(hashlib.sha1(f"{cluster}:{name}:arm".encode("utf-8")).hexdigest()[:8], 16) % arm_count
            candidates = [(preferred + offset) % arm_count for offset in range(arm_count)]
            arm = min(candidates, key=lambda candidate: (arm_loads[candidate] >= arm_capacity, arm_loads[candidate], candidate))
            row_to_arm[row_index] = arm
            row_to_arm_order[row_index] = int(arm_loads[arm])
            arm_loads[arm] += 1

        if error_column in group.columns:
            error = pd.to_numeric(group[error_column], errors="coerce").fillna(0.0)
            error_scale = max(float(error.quantile(0.90)), 1.0)
        else:
            error = pd.Series(0.0, index=group.index)
            error_scale = 1.0

        for row_index, row in group.iterrows():
            name = row.get("display_name", row_index)
            arm = row_to_arm.get(row_index, 0)
            arm_order = row_to_arm_order.get(row_index, 0)
            rank = float(score_rank.loc[row_index])
            residual_push = 0.055 * min(float(error.loc[row_index]) / error_scale, 1.5)
            angular_jitter = _deterministic_jitter(name, scale=0.52)
            radial_jitter = _deterministic_jitter(f"{name}:radius", scale=0.09)
            arm_phase = _deterministic_jitter(f"{cluster}:{arm}:phase", scale=0.34)
            arm_curve = 1.04 + _deterministic_jitter(f"{cluster}:{arm}:curve", scale=0.32)
            branch = int(hashlib.sha1(f"{name}:branch".encode("utf-8")).hexdigest()[:2], 16) % 3 - 1
            arm_size = max(int(arm_loads[arm]), 1)
            arm_depth = (arm_order + 1.0) / (arm_size + 1.0)
            radial_unit = _deterministic_unit(f"{cluster}:{name}:radial")
            core_fraction = 0.30 if n < 24 else 0.24
            if radial_unit < core_fraction:
                core_unit = radial_unit / max(core_fraction, 1e-6)
                radius = cluster_radius * (0.035 + 0.17 * np.sqrt(core_unit)) + radial_jitter * 0.35
                angle_jitter = angular_jitter
                branch_depth = arm_depth
            else:
                branch_unit = (radial_unit - core_fraction) / max(1.0 - core_fraction, 1e-6)
                branch_depth = 0.70 * arm_depth + 0.30 * branch_unit
                radius = cluster_radius * (0.16 + 0.78 * (branch_depth**1.12)) + residual_push + radial_jitter
                angle_jitter = angular_jitter * 0.52
            spiral_turn = arm_curve * (0.18 + 0.88 * branch_depth) + 0.18 * (rank - 0.5)
            angle = 2.0 * np.pi * (arm / arm_count) + arm_phase + spiral_turn + branch * 0.16 * branch_depth + angle_jitter
            radius = max(radius, cluster_radius * 0.025)
            x_values.loc[row_index] = cx + radius * np.cos(angle)
            y_values.loc[row_index] = cy + radius * np.sin(angle)
            center_x.loc[row_index] = cx
            center_y.loc[row_index] = cy
            radius_values.loc[row_index] = radius
            arm_values.loc[row_index] = arm
            arm_order_values.loc[row_index] = arm_order
            angle_values.loc[row_index] = angle

    layout["galaxy_x"] = x_values
    layout["galaxy_y"] = y_values
    layout["cluster_center_x"] = center_x
    layout["cluster_center_y"] = center_y
    layout["galaxy_radius"] = radius_values
    layout["galaxy_arm"] = arm_values
    layout["galaxy_arm_order"] = arm_order_values
    layout["galaxy_angle"] = angle_values
    return layout.dropna(subset=["galaxy_x", "galaxy_y"])


def _galaxy_arm_edges(frame: pd.DataFrame, max_edges: int = GALAXY_EDGE_LIMIT) -> dict[str, list[float | None]]:
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    if max_edges <= 0:
        return {"x": edge_x, "y": edge_y}

    def add_curve(x0: float, y0: float, x1: float, y1: float, key: object) -> None:
        dx = x1 - x0
        dy = y1 - y0
        length = max(float(np.hypot(dx, dy)), 1e-6)
        bend = length * _deterministic_jitter(f"{key}:smooth-curve", scale=0.24)
        control_x = (x0 + x1) / 2.0 - dy / length * bend
        control_y = (y0 + y1) / 2.0 + dx / length * bend
        for t in np.linspace(0.0, 1.0, 6):
            one_minus = 1.0 - float(t)
            edge_x.append(one_minus * one_minus * x0 + 2.0 * one_minus * float(t) * control_x + float(t) * float(t) * x1)
            edge_y.append(one_minus * one_minus * y0 + 2.0 * one_minus * float(t) * control_y + float(t) * float(t) * y1)
        edge_x.append(None)
        edge_y.append(None)

    def interleave(segment_groups: list[list[tuple[float, float, float, float, object]]]) -> list[tuple[float, float, float, float, object]]:
        ordered: list[tuple[float, float, float, float, object]] = []
        offsets = [0 for _ in segment_groups]
        remaining = sum(len(group) for group in segment_groups)
        while remaining > 0:
            for group_index, group in enumerate(segment_groups):
                offset = offsets[group_index]
                if offset >= len(group):
                    continue
                ordered.append(group[offset])
                offsets[group_index] += 1
                remaining -= 1
        return ordered

    cluster_groups: list[list[tuple[float, float, float, float, object]]] = []
    if {"galaxy_arm", "galaxy_arm_order"}.issubset(frame.columns):
        sorted_frame = frame.sort_values(["cluster", "galaxy_arm", "galaxy_arm_order"])
        for _, cluster_group in sorted_frame.groupby("cluster", sort=True):
            arm_segment_groups: list[list[tuple[float, float, float, float, object]]] = []
            for _, arm_group in cluster_group.groupby("galaxy_arm", sort=True):
                if arm_group.empty:
                    continue
                previous_x = float(arm_group["cluster_center_x"].iloc[0])
                previous_y = float(arm_group["cluster_center_y"].iloc[0])
                arm_segments: list[tuple[float, float, float, float, object]] = []
                for _, row in arm_group.iterrows():
                    x1 = float(row["galaxy_x"])
                    y1 = float(row["galaxy_y"])
                    arm_segments.append((previous_x, previous_y, x1, y1, row.get("display_name", len(arm_segments))))
                    previous_x = x1
                    previous_y = y1
                if arm_segments:
                    arm_segment_groups.append(arm_segments)
            if arm_segment_groups:
                cluster_groups.append(interleave(arm_segment_groups))
    else:
        fallback_segments: list[tuple[float, float, float, float, object]] = []
        for _, row in frame.iterrows():
            fallback_segments.append(
                (
                    float(row["cluster_center_x"]),
                    float(row["cluster_center_y"]),
                    float(row["galaxy_x"]),
                    float(row["galaxy_y"]),
                    row.get("display_name", len(fallback_segments)),
                )
            )
        if fallback_segments:
            cluster_groups.append(fallback_segments)

    if not cluster_groups:
        return {"x": edge_x, "y": edge_y}

    total_edges = sum(len(group) for group in cluster_groups)
    if total_edges <= max_edges:
        for group in cluster_groups:
            for x0, y0, x1, y1, key in group:
                add_curve(x0, y0, x1, y1, key)
        return {"x": edge_x, "y": edge_y}

    offsets = [0 for _ in cluster_groups]
    edge_count = 0
    while edge_count < max_edges:
        progressed = False
        for group_index, group in enumerate(cluster_groups):
            offset = offsets[group_index]
            if offset >= len(group):
                continue
            x0, y0, x1, y1, key = group[offset]
            add_curve(x0, y0, x1, y1, key)
            offsets[group_index] += 1
            edge_count += 1
            progressed = True
            if edge_count >= max_edges:
                break
        if not progressed:
            break
    return {"x": edge_x, "y": edge_y}


def _galaxy_neighbor_edges(frame: pd.DataFrame, max_neighbors: int = 1, max_edges: int = GALAXY_EDGE_LIMIT) -> dict[str, list[float | None]]:
    if frame.empty or "nearest_neighbors" not in frame.columns:
        return {"x": [], "y": []}
    positions = frame.set_index("display_name")[["galaxy_x", "galaxy_y"]].to_dict("index")
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_count = 0
    for _, row in frame.iterrows():
        source = positions.get(str(row["display_name"]))
        if source is None:
            continue
        neighbors = [name.strip() for name in str(row.get("nearest_neighbors", "")).split(",") if name.strip()]
        for neighbor in neighbors[:max_neighbors]:
            target_position = positions.get(neighbor)
            if target_position is None:
                continue
            edge_x.extend([source["galaxy_x"], target_position["galaxy_x"], None])
            edge_y.extend([source["galaxy_y"], target_position["galaxy_y"], None])
            edge_count += 1
            if edge_count >= max_edges:
                return {"x": edge_x, "y": edge_y}
    return {"x": edge_x, "y": edge_y}


def _galaxy_neighbor_edge_traces(
    frame: pd.DataFrame,
    *,
    color_by: str,
    max_neighbors: int = 1,
    max_edges: int = GALAXY_EDGE_LIMIT,
    dark: bool = False,
) -> list[go.Scatter]:
    if frame.empty or "nearest_neighbors" not in frame.columns:
        return []
    positions = frame.set_index("display_name")[["galaxy_x", "galaxy_y"]].to_dict("index")
    color_map = _category_color_map(frame, color_by)
    edge_groups: dict[str, dict[str, list[float | None]]] = {}
    edge_count = 0

    def add_curve(group_key: str, x0: float, y0: float, x1: float, y1: float, key: object) -> None:
        dx = x1 - x0
        dy = y1 - y0
        length = max(float(np.hypot(dx, dy)), 1e-6)
        bend = _deterministic_jitter(f"neighbor:{key}", scale=0.34)
        control_x = (x0 + x1) / 2.0 - dy / length * bend
        control_y = (y0 + y1) / 2.0 + dx / length * bend
        group = edge_groups.setdefault(group_key, {"x": [], "y": []})
        group["x"].extend([x0, control_x, x1, None])
        group["y"].extend([y0, control_y, y1, None])

    for _, row in frame.iterrows():
        source = positions.get(str(row["display_name"]))
        if source is None:
            continue
        group_key = str(row[color_by]) if color_by in row.index else str(row.get("Predicted cluster", row.get("cluster", "links")))
        neighbors = [name.strip() for name in str(row.get("nearest_neighbors", "")).split(",") if name.strip()]
        for neighbor in neighbors[:max_neighbors]:
            target_position = positions.get(neighbor)
            if target_position is None:
                continue
            add_curve(
                group_key,
                float(source["galaxy_x"]),
                float(source["galaxy_y"]),
                float(target_position["galaxy_x"]),
                float(target_position["galaxy_y"]),
                f"{row['display_name']}->{neighbor}",
            )
            edge_count += 1
            if edge_count >= max_edges:
                break
        if edge_count >= max_edges:
            break

    traces: list[go.Scatter] = []
    for group_key in sorted(edge_groups):
        group = edge_groups[group_key]
        traces.append(
            go.Scattergl(
                x=group["x"],
                y=group["y"],
                mode="lines",
                line={"color": color_map.get(group_key, _edge_color_for_key(group_key)), "width": 0.85 if dark else 0.75},
                opacity=0.48 if dark else 0.38,
                hoverinfo="skip",
                name=f"{group_key} links",
                showlegend=False,
            )
        )
    return traces


def _plot_cluster_galaxy(
    frame: pd.DataFrame,
    color_by: str,
    size_by: str,
    title: str,
    target: str | None = None,
    edge_density: str = DEFAULT_EDGE_DENSITY,
    dark: bool | None = None,
    zoom: float = 1.0,
):
    dark = _is_dark_theme() if dark is None else bool(dark)
    plot_frame = frame.copy()
    plot_frame["display_size"] = _scaled_marker_size(plot_frame, size_by)
    if color_by not in plot_frame.columns:
        color_by = "Predicted cluster" if "Predicted cluster" in plot_frame.columns else "cluster"
    color_map = _category_color_map(plot_frame, color_by)
    plot_color_by = color_by
    if color_map:
        plot_color_by = f"{color_by} label"
        plot_frame[plot_color_by] = plot_frame[color_by].fillna("unknown").astype(str)

    hover_columns = _dynamic_hover_columns(plot_frame, target)

    node_fig = px.scatter(
        plot_frame,
        x="galaxy_x",
        y="galaxy_y",
        color=plot_color_by,
        size="display_size",
        size_max=MAX_MARKER_SIZE,
        hover_name="display_name",
        hover_data=hover_columns,
        color_discrete_map=color_map or None,
        render_mode="webgl",
        opacity=0.70,
        title=title,
    )

    fig = go.Figure()
    arm_edge_limit = _edge_limit_for_density(len(plot_frame), edge_density, edge_kind="arms")
    neighbor_edge_limit = _edge_limit_for_density(len(plot_frame), edge_density, edge_kind="neighbors")
    if arm_edge_limit:
        arm_edges = _galaxy_arm_edges(plot_frame, max_edges=arm_edge_limit)
        for style in _galaxy_arm_trace_styles(dark=dark):
            fig.add_trace(
                go.Scattergl(
                    x=arm_edges["x"],
                    y=arm_edges["y"],
                    mode="lines",
                    line=style["line"],
                    opacity=style["opacity"],
                    hoverinfo="skip",
                    name=style["name"],
                    showlegend=False,
                )
            )
    if neighbor_edge_limit:
        for trace in _galaxy_neighbor_edge_traces(plot_frame, color_by=color_by, max_edges=neighbor_edge_limit, dark=dark):
            fig.add_trace(trace)

    for trace in node_fig.data:
        fig.add_trace(trace)

    centers = (
        plot_frame.groupby("cluster", as_index=False)
        .agg(
            cluster_center_x=("cluster_center_x", "first"),
            cluster_center_y=("cluster_center_y", "first"),
            center_model=("Cluster center model", "first"),
            size=("display_name", "size"),
        )
        .sort_values("cluster")
    )
    fig.add_trace(
        go.Scattergl(
            x=centers["cluster_center_x"],
            y=centers["cluster_center_y"],
            mode="markers",
            marker={
                "size": np.clip(6.0 + np.sqrt(centers["size"].to_numpy(dtype=float)), 9.0, 22.0),
                "color": "rgba(15,23,42,0.94)" if dark else "rgba(255,255,255,0.92)",
                "line": {"color": "rgba(226,232,240,0.76)" if dark else "rgba(20,26,38,0.62)", "width": 1.2},
            },
            customdata=np.stack([centers["center_model"], centers["size"]], axis=-1),
            hovertemplate="center %{customdata[0]}<br>%{customdata[1]} models<extra></extra>",
            name="representative centers",
            showlegend=False,
        )
    )

    fig.update_layout(node_fig.layout)
    fig.update_layout(
        height=PLOT_HEIGHT,
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
        legend_title_text=color_by,
    )
    _apply_axis_zoom(fig, plot_frame, "galaxy_x", "galaxy_y", zoom=zoom)
    fig.update_traces(marker={"line": {"width": 0}}, selector={"mode": "markers"})
    return _apply_atlas_style(fig, hide_axes=True, dark=dark)


def _labeled_prediction_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or not {"actual", "prediction"}.issubset(predictions.columns):
        return pd.DataFrame(columns=predictions.columns)
    actual = pd.to_numeric(predictions["actual"], errors="coerce")
    predicted = pd.to_numeric(predictions["prediction"], errors="coerce")
    return predictions[actual.notna() & predicted.notna()].copy()


def _prediction_counts(predictions: pd.DataFrame) -> tuple[int, int]:
    if predictions.empty or "prediction" not in predictions.columns:
        return 0, 0
    predicted = int(pd.to_numeric(predictions["prediction"], errors="coerce").notna().sum())
    labeled = len(_labeled_prediction_rows(predictions))
    return predicted, labeled


def _prediction_quality_metrics(predictions: pd.DataFrame) -> dict[str, float | int | None]:
    predicted_count, labeled_count = _prediction_counts(predictions)
    labeled = _labeled_prediction_rows(predictions)
    metrics: dict[str, float | int | None] = {
        "predicted_count": predicted_count,
        "labeled_count": labeled_count,
        "label_coverage": (labeled_count / predicted_count) if predicted_count else None,
        "mae": None,
        "rmse": None,
        "median_absolute_error": None,
        "spearman": None,
        "bias": None,
        "r2": None,
    }
    if labeled.empty:
        return metrics

    actual = pd.to_numeric(labeled["actual"], errors="coerce")
    prediction = pd.to_numeric(labeled["prediction"], errors="coerce")
    residual = prediction - actual
    abs_error = residual.abs()
    metrics["mae"] = float(abs_error.mean())
    metrics["rmse"] = float(np.sqrt(np.mean(np.square(residual))))
    metrics["median_absolute_error"] = float(abs_error.median())
    metrics["bias"] = float(residual.mean())
    spearman = actual.corr(prediction, method="spearman")
    metrics["spearman"] = None if pd.isna(spearman) else float(spearman)
    total_variance = float(np.sum(np.square(actual - actual.mean())))
    if total_variance > 0:
        metrics["r2"] = float(1.0 - float(np.sum(np.square(residual))) / total_variance)
    return metrics


def _prediction_metric_cards(predictions: pd.DataFrame) -> list[dict[str, str]]:
    metrics = _prediction_quality_metrics(predictions)
    return [
        {"label": "Predicted", "value": f"{int(metrics['predicted_count'] or 0):,}"},
        {"label": "Known labels", "value": f"{int(metrics['labeled_count'] or 0):,}"},
        {"label": "Coverage", "value": _format_percent(metrics["label_coverage"])},
        {"label": "MAE", "value": _format_number(metrics["mae"])},
        {"label": "RMSE", "value": _format_number(metrics["rmse"])},
        {"label": "R2", "value": _format_number(metrics["r2"])},
        {"label": "Bias", "value": _format_number(metrics["bias"])},
        {"label": "Spearman", "value": _format_number(metrics["spearman"])},
    ]


def _lineage_quality_metrics(lineage: pd.DataFrame) -> dict[str, float | int | None]:
    metrics = _prediction_quality_metrics(lineage)
    metrics["same_family_link_rate"] = None
    if lineage.empty or not {"display_name", "nearest_neighbors", "coarse_family"}.issubset(lineage.columns):
        return metrics

    family_by_model = lineage.set_index("display_name")["coarse_family"].fillna("unknown").astype(str).to_dict()
    same_family = 0
    comparable_links = 0
    for _, row in lineage.iterrows():
        source = str(row["display_name"])
        source_family = family_by_model.get(source, "unknown")
        if source_family.lower() in {"", "unknown", "nan", "none", "n/a"}:
            continue
        neighbors = [name.strip() for name in str(row.get("nearest_neighbors", "")).split(",") if name.strip()]
        for neighbor in neighbors:
            target_family = family_by_model.get(neighbor)
            if target_family is None or target_family.lower() in {"", "unknown", "nan", "none", "n/a"}:
                continue
            comparable_links += 1
            same_family += int(target_family == source_family)
    if comparable_links:
        metrics["same_family_link_rate"] = same_family / comparable_links
    return metrics


def _family_purity(summary: pd.DataFrame) -> float | None:
    if summary.empty or "dominant_family_share" not in summary.columns or "size" not in summary.columns:
        return None
    shares = pd.to_numeric(summary["dominant_family_share"], errors="coerce")
    sizes = pd.to_numeric(summary["size"], errors="coerce")
    valid = shares.notna() & sizes.notna()
    if not valid.any() or float(sizes[valid].sum()) <= 0:
        return None
    return float(np.average(shares[valid], weights=sizes[valid]))


def _dashboard_summary_cards(cache: DashboardCache, frame: pd.DataFrame, target: str | None) -> list[dict[str, str]]:
    predictions = cache.predictions.get(target or "", pd.DataFrame())
    metrics = _prediction_quality_metrics(predictions)
    metadata = cache.metadata if hasattr(cache, "metadata") else pd.DataFrame()
    family_series = frame.get("coarse_family", pd.Series(dtype=object))
    cluster_series = metadata.get("cluster", pd.Series(dtype=object))
    return [
        {
            "label": "Filtered rows",
            "value": f"{len(frame):,}",
            "delta": f"{len(metadata):,} cached",
        },
        {
            "label": "Clusters",
            "value": f"{pd.to_numeric(cluster_series, errors='coerce').dropna().nunique():,}",
            "delta": "predicted groups",
        },
        {
            "label": "Families",
            "value": f"{family_series.dropna().astype(str).nunique():,}",
            "delta": "filtered view",
        },
        {
            "label": "Family purity",
            "value": _format_percent(_family_purity(cache.cluster_summary)),
            "delta": "weighted",
        },
        {
            "label": "MAE",
            "value": _format_number(metrics["mae"]),
            "delta": f"{int(metrics['labeled_count'] or 0):,} labels",
        },
        {
            "label": "R2",
            "value": _format_number(metrics["r2"]),
            "delta": _format_number(metrics["spearman"]) + " rho",
        },
    ]


def _render_metric_cards(cache: DashboardCache, frame: pd.DataFrame, target: str | None) -> None:
    cards = _dashboard_summary_cards(cache, frame, target)
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards, strict=False):
        column.metric(card["label"], card["value"], card.get("delta"), delta_color="off")


def _plotly_chart(fig, *, key: str, scroll_zoom: bool = False) -> None:
    fig.update_layout(uirevision=key)
    st.plotly_chart(
        fig,
        key=key,
        width="stretch",
        config=_plot_config(scroll_zoom=scroll_zoom),
    )


def _render_color_legend(frame: pd.DataFrame, color_by: str, max_items: int | None = None) -> None:
    color_map = _category_color_map(frame, color_by)
    if not color_map:
        return
    counts = frame[color_by].fillna("unknown").astype(str).value_counts()
    ordered = [value for value in counts.index.tolist() if value in color_map]
    if max_items is None:
        preview_size = int(np.ceil(np.log2(max(len(frame), 2))))
        max_items = max(1, min(len(ordered), preview_size))
    visible = ordered[:max_items]
    items = []
    for value in visible:
        count = int(counts[value])
        display_value = value
        if color_by == "Predicted cluster" and "Cluster center model" in frame.columns:
            cluster_values = frame[color_by].fillna("unknown").astype(str)
            representatives = (
                frame.loc[cluster_values.eq(value), "Cluster center model"]
                .dropna()
                .astype(str)
            )
            if not representatives.empty:
                display_value = representatives.iloc[0]
        items.append(
            "<span style='display:inline-flex;align-items:center;gap:5px;min-width:0;'>"
            f"<span style='width:8px;height:8px;border-radius:999px;background:{color_map[value]};"
            "display:inline-block;border:1px solid rgba(15,23,42,0.28);'></span>"
            f"<span style='white-space:nowrap;'>{html.escape(display_value)} ({count:,})</span>"
            "</span>"
        )
    if len(ordered) > max_items:
        items.append(f"<span style='opacity:0.68;'>+{len(ordered) - max_items:,} more</span>")
    st.markdown(
        "<div style='display:flex;flex-wrap:wrap;gap:7px 12px;font-size:11px;"
        "line-height:1.25;margin:-2px 0 10px;color:inherit;'>"
        + "".join(items)
        + "</div>",
        unsafe_allow_html=True,
    )


def _plot_galaxy(
    frame: pd.DataFrame,
    color_by: str,
    size_by: str,
    view_mode: str,
    target: str | None = None,
    edge_density: str = DEFAULT_EDGE_DENSITY,
    zoom: float = 1.0,
):
    layout = _cluster_galaxy_layout(frame, target=target)
    return _plot_cluster_galaxy(
        layout,
        color_by,
        size_by,
        title="Cluster galaxy",
        target=target,
        edge_density=edge_density,
        zoom=zoom,
    )


def _prediction_lineage_frame(cache: DashboardCache, target: str, frame: pd.DataFrame) -> pd.DataFrame:
    predictions = cache.predictions.get(target, pd.DataFrame())
    if predictions.empty or frame.empty:
        return pd.DataFrame()

    prediction_columns = ["display_name", "actual", "prediction", "absolute_error", "nearest_neighbors"]
    lineage = frame.merge(predictions[prediction_columns], on="display_name", how="inner")
    if lineage.empty:
        return lineage

    families = (
        lineage["coarse_family"].fillna("unknown").astype(str)
        if "coarse_family" in lineage.columns
        else pd.Series("unknown", index=lineage.index)
    )
    family_order = (
        lineage.assign(_family=families)
        .groupby("_family")["prediction"]
        .median()
        .sort_values()
        .index.tolist()
    )
    family_lane = {family: index for index, family in enumerate(family_order)}
    residual = pd.to_numeric(lineage["actual"], errors="coerce") - pd.to_numeric(lineage["prediction"], errors="coerce")
    residual_scale = float(np.nanmedian(np.abs(residual))) if np.isfinite(residual).any() else 1.0
    residual_scale = max(residual_scale, 1.0)

    lineage["lineage_family"] = families
    lineage["lineage_x"] = pd.to_numeric(lineage["prediction"], errors="coerce")
    lineage["lineage_y"] = [
        family_lane.get(family, 0) + 0.34 * np.tanh(float(error) / residual_scale) + _deterministic_jitter(name)
        for family, error, name in zip(families, residual.fillna(0.0), lineage["display_name"], strict=False)
    ]
    lineage["lineage_z"] = residual
    return lineage.replace([np.inf, -np.inf], np.nan).dropna(subset=["lineage_x", "lineage_y"])


def _lineage_edge_coordinates(frame: pd.DataFrame, *, max_neighbors: int = 2, max_edges: int = 6000, view_mode: str = "2D cluster galaxy") -> dict[str, list[float | None]]:
    if frame.empty or "nearest_neighbors" not in frame.columns:
        return {"x": [], "y": [], "z": []}

    positions = frame.set_index("display_name")[["lineage_x", "lineage_y", "lineage_z"]].to_dict("index")
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    edge_count = 0
    for _, row in frame.iterrows():
        source = str(row["display_name"])
        source_position = positions.get(source)
        if source_position is None:
            continue
        neighbors = [name.strip() for name in str(row.get("nearest_neighbors", "")).split(",") if name.strip()]
        for neighbor in neighbors[:max_neighbors]:
            target_position = positions.get(neighbor)
            if target_position is None:
                continue
            edge_x.extend([source_position["lineage_x"], target_position["lineage_x"], None])
            edge_y.extend([source_position["lineage_y"], target_position["lineage_y"], None])
            edge_count += 1
            if edge_count >= max_edges:
                return {"x": edge_x, "y": edge_y, "z": edge_z}
    return {"x": edge_x, "y": edge_y, "z": edge_z}


def _plot_prediction_lineage(
    frame: pd.DataFrame,
    color_by: str,
    size_by: str,
    view_mode: str,
    target: str,
    edge_density: str = DEFAULT_EDGE_DENSITY,
    zoom: float = 1.0,
):
    layout = _cluster_galaxy_layout(frame, target=target)
    return _plot_cluster_galaxy(
        layout,
        color_by,
        size_by,
        title=f"{target}: prediction-neighbor cluster galaxy",
        target=target,
        edge_density=edge_density,
        zoom=zoom,
    )


def _render_profiles_tab(cache: DashboardCache, selected_cluster: int) -> None:
    profiles = cache.cluster_profiles
    if profiles.empty:
        st.info("No cached cluster profiles are available.")
        return
    subset = profiles[profiles["cluster"].astype(int).eq(int(selected_cluster))]
    if subset.empty:
        st.info("No profile rows are available for the selected cluster.")
        return
    frozen_aggregates = "role_name" in subset.columns
    fig = px.line(
        subset.sort_values("depth"),
        x="depth",
        y="log_alpha",
        color="role",
        markers=frozen_aggregates,
        hover_data=["role_name"] if frozen_aggregates else None,
        title=(
            f"Cluster {selected_cluster} frozen role profile"
            if frozen_aggregates
            else f"Cluster {selected_cluster} mean role traces"
        ),
    )
    if frozen_aggregates:
        role_ticks = subset.loc[:, ["depth", "role_name"]].drop_duplicates().sort_values("depth")
        fig.update_xaxes(
            tickmode="array",
            tickvals=role_ticks["depth"].tolist(),
            ticktext=role_ticks["role_name"].tolist(),
            title_text="canonical role",
        )
        fig.update_yaxes(title_text="mean log1p(stored DKS alpha)")
    fig.update_layout(height=PLOT_HEIGHT)
    _apply_analytics_style(fig)
    provenance = cache.manifest.get("provenance", {})
    run_id = (
        provenance.get("run_id", cache.cache_dir.name)
        if isinstance(provenance, dict)
        else cache.cache_dir.name
    )
    _plotly_chart(fig, key=f"htsr:{run_id}:profiles:alpha")


def _method_evidence_display_tables(
    family: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    atlas_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    display_family = family.rename(
        columns={
            "n_models": "study_models",
            "eligible_queries": "eligible_family_queries",
        }
    ).copy()
    display_prediction = prediction.rename(columns={"n": "labeled_study_rows"}).copy()
    for frame in (display_family, display_prediction):
        if not frame.empty and "atlas_models" not in frame.columns:
            frame.insert(1, "atlas_models", int(atlas_rows))
    return display_family, display_prediction


def _frozen_metric(section: object, metric: str, *, method: str = "combined") -> float | None:
    if isinstance(section, dict):
        method_values = section.get(method)
        if isinstance(method_values, dict) and metric in method_values:
            return _finite_number(method_values.get(metric))
        return _finite_number(section.get(metric))
    if isinstance(section, list):
        for row in section:
            if not isinstance(row, dict):
                continue
            if str(row.get("method", method)) != method or str(row.get("metric", "")) != metric:
                continue
            return _finite_number(row.get("estimate", row.get("value")))
    return None


def _finite_number(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _lineage_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "cluster" in frame.columns:
        clusters = pd.to_numeric(frame["cluster"], errors="coerce")
        return frame.loc[clusters.ge(0)].copy()
    if "predicted_family" in frame.columns:
        return frame.loc[frame["predicted_family"].notna()].copy()
    return frame.iloc[0:0].copy()


def _render_cards(cards: list[dict[str, str]]) -> None:
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards, strict=False):
        column.metric(card["label"], card["value"], card.get("delta"), delta_color="off")


def _performance_target_values(outputs: StudyOutputs, target: str) -> dict[str, float | None]:
    primary = outputs.performance_metrics.get("primary", {})
    primary_target = str(primary.get("target", "modern_average")) if isinstance(primary, dict) else "modern_average"
    values: dict[str, float | None] = {
        "group_weighted_mae": None,
        "spearman": None,
        "htsr_delta_mae": None,
        "htsr_ci_low": None,
        "htsr_ci_high": None,
        "n_models": None,
        "n_components": None,
    }
    if target == primary_target:
        values.update({key: _finite_number(value) for key, value in outputs.performance_summary.items()})

    development = outputs.performance_metrics.get("development", [])
    if isinstance(development, list):
        for row in development:
            if not isinstance(row, dict):
                continue
            if str(row.get("target")) == target and str(row.get("feature_block")) == "M+S+H":
                values["group_weighted_mae"] = _finite_number(row.get("group_weighted_mae"))
                values["spearman"] = _finite_number(row.get("spearman"))
                values["n_models"] = _finite_number(row.get("n_models"))
                values["n_components"] = _finite_number(row.get("n_components"))
                break

    increment: object = outputs.performance_metrics.get("primary_increment", {})
    if target != primary_target:
        increment = next(
            (
                row
                for row in outputs.performance_metrics.get("secondary_increments", [])
                if isinstance(row, dict) and str(row.get("target")) == target
            ),
            {},
        )
    if isinstance(increment, dict):
        values["htsr_delta_mae"] = _finite_number(
            increment.get("delta_mae", increment.get("htsr_delta_mae", values["htsr_delta_mae"]))
        )
        values["htsr_ci_low"] = _finite_number(
            increment.get("ci_low", increment.get("htsr_ci_low", values["htsr_ci_low"]))
        )
        values["htsr_ci_high"] = _finite_number(
            increment.get("ci_high", increment.get("htsr_ci_high", values["htsr_ci_high"]))
        )
    return values


def _scaleup_performance_target_values(
    scaleup: ScaleupOutputs,
    target: str,
) -> dict[str, float | None]:
    record = scaleup.performance.get(target, {})
    return {
        "group_weighted_mae": _finite_number(record.get("group_weighted_mae")),
        "spearman": _finite_number(record.get("spearman")),
        "htsr_delta_mae": _finite_number(record.get("delta_mae")),
        "htsr_ci_low": _finite_number(record.get("ci_low")),
        "htsr_ci_high": _finite_number(record.get("ci_high")),
        "n_models": _finite_number(record.get("n_models")),
        "n_components": _finite_number(record.get("n_components")),
    }


def _external_status_for_target(outputs: StudyOutputs, target: str) -> str:
    evaluated_target = outputs.performance_summary.get("external_target")
    if evaluated_target is None or str(evaluated_target) != str(target):
        return "Not evaluated"
    if outputs.external_status == "descriptive_reuse":
        return "Reused only"
    return str(outputs.external_status).replace("_", " ").title()


def _render_frozen_prediction(
    outputs: StudyOutputs,
    cache: DashboardCache,
    target: str,
    scaleup: ScaleupOutputs | None = None,
) -> None:
    run_id = str(
        scaleup.metadata.get("run_id", scaleup.run_dir.name)
        if scaleup is not None
        else outputs.provenance.get("run_id", outputs.run_dir.name)
    )
    predictions = cache.predictions.get(target, pd.DataFrame())
    values = (
        _scaleup_performance_target_values(scaleup, target)
        if scaleup is not None
        else _performance_target_values(outputs, target)
    )
    interval = (
        f"[{_format_number(values['htsr_ci_low'])}, {_format_number(values['htsr_ci_high'])}]"
        if values["htsr_ci_low"] is not None and values["htsr_ci_high"] is not None
        else "n/a"
    )
    model_count = int(values["n_models"]) if values["n_models"] is not None else len(predictions)
    component_count = (
        f"{int(values['n_components']):,}"
        if values["n_components"] is not None
        else "n/a"
    )
    primary_status = "direct alpha vs M+S+H" if scaleup is not None else (
        "claim passed"
        if outputs.performance_summary.get("primary_passed") is True
        else "claim failed"
    )
    if target != "modern_average" and scaleup is None:
        primary_status = "descriptive"
    _render_cards(
        [
            {
                "label": "OOF models / groups",
                "value": f"{model_count:,} / {component_count}",
            },
            {"label": "Grouped MAE", "value": _format_number(values["group_weighted_mae"])},
            {"label": "Model Spearman", "value": _format_number(values["spearman"])},
            {
                "label": "dMAE vs baseline" if scaleup is not None else "HTSR dMAE",
                "value": _format_number(values["htsr_delta_mae"]),
                "delta": primary_status,
            },
            {"label": "95% interval", "value": interval},
            {
                "label": "Evidence",
                "value": (
                    "Internal grouped"
                    if scaleup is not None
                    else _external_status_for_target(outputs, target)
                ),
            },
        ]
    )
    if scaleup is not None:
        st.caption(
            "Scale-up out-of-fold direct alpha-profile predictions. MAE is ancestry-component "
            "weighted; Spearman is model-level descriptive."
        )
    else:
        st.caption(
            "Frozen development out-of-fold M+S+H predictions. MAE is component "
            "weighted; Spearman is model-level descriptive."
        )
    labeled = _labeled_prediction_rows(predictions)
    if labeled.empty:
        st.info("No finite frozen development predictions are available for this target.")
        return

    plot_frame = labeled.copy()
    plot_frame["hover_model"] = plot_frame["display_name"].map(
        lambda value: _display_model_name(value, limit=48)
    )
    dark = _is_dark_theme()
    scatter = px.scatter(
        plot_frame,
        x="actual",
        y="prediction",
        color="absolute_error",
        custom_data=["hover_model", "actual", "prediction", "absolute_error"],
        color_continuous_scale=_prediction_error_color_scale(dark=dark),
        title=f"{target}: out-of-fold prediction",
        render_mode="webgl",
    )
    lower = float(np.nanmin([plot_frame["actual"].min(), plot_frame["prediction"].min()]))
    upper = float(np.nanmax([plot_frame["actual"].max(), plot_frame["prediction"].max()]))
    scatter.add_shape(
        type="line",
        x0=lower,
        y0=lower,
        x1=upper,
        y1=upper,
        line={"dash": "dash", "color": "rgba(226,232,240,0.72)" if dark else "rgba(15,23,42,0.48)"},
    )
    scatter.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>actual %{customdata[1]:.3f}<br>"
            "predicted %{customdata[2]:.3f}<br>|error| %{customdata[3]:.3f}<extra></extra>"
        ),
        marker={"size": 6, "opacity": 0.84, "line": {"width": 0}},
    )
    scatter.update_layout(height=PLOT_HEIGHT, coloraxis_showscale=False, showlegend=False)
    _apply_analytics_style(scatter, dark=dark)
    _plotly_chart(
        scatter,
        key=f"htsr:{run_id}:prediction:{target}:scatter",
        scroll_zoom=False,
    )

    histogram = px.histogram(
        plot_frame,
        x="absolute_error",
        nbins=40,
        title="Absolute-error distribution",
    )
    histogram.update_layout(height=SECONDARY_PLOT_HEIGHT, showlegend=False)
    _apply_analytics_style(histogram)
    _plotly_chart(
        histogram,
        key=f"htsr:{run_id}:prediction:{target}:error-histogram",
        scroll_zoom=False,
    )
    st.dataframe(
        plot_frame.sort_values("absolute_error", ascending=False),
        width="stretch",
        height=SECONDARY_PLOT_HEIGHT,
        hide_index=True,
    )


def _render_frozen_lineage(
    outputs: StudyOutputs,
    frame: pd.DataFrame,
    *,
    color_by: str,
    size_by: str,
    edge_density: str,
    gesture_zoom: bool,
    atlas_zoom: float,
) -> None:
    run_id = str(outputs.provenance.get("run_id", outputs.run_dir.name))
    open_set = outputs.lineage_metrics.get("open_set", outputs.lineage_metrics.get("open_set_metrics", {}))
    coverage = _frozen_metric(open_set, "scorable_coverage")
    family_count = outputs.lineage_summary.get("n_families")
    query_count = outputs.lineage_summary.get("n_queries")
    lineage_status = (
        "Inferential"
        if outputs.lineage_summary.get("inferential") is True
        else "Descriptive"
    )
    _render_cards(
        [
            {"label": "Root top-1", "value": _format_percent(outputs.lineage_summary.get("root_macro_top1"))},
            {
                "label": "Known-family top-1",
                "value": _format_percent(
                    outputs.lineage_summary.get("observed_family_top1")
                ),
                "delta": "descriptive",
            },
            {"label": "Unknown AUROC", "value": _format_number(outputs.lineage_summary.get("unknown_auroc"))},
            {"label": "OSCR", "value": _format_number(outputs.lineage_summary.get("oscr"))},
            {"label": "Coverage", "value": _format_percent(coverage)},
            {
                "label": "Queries / families",
                "value": (
                    f"{int(query_count):,} / {int(family_count):,}"
                    if query_count is not None and family_count is not None
                    else "n/a"
                ),
                "delta": lineage_status,
            },
        ]
    )
    st.caption(
        "Frozen canonical-root retrieval, descriptive query-excluded known-family "
        "attribution, and leave-one-family-out open-set metrics."
    )
    lineage = _lineage_display_frame(frame)
    if lineage.empty:
        st.info("No frozen clustered models match the current filters.")
        return
    if "family_prediction_scope" in lineage.columns and lineage[
        "family_prediction_scope"
    ].notna().any():
        st.caption(
            f"{len(lineage):,} strict-QC clustered models shown. Full-gallery family "
            "routes are descriptive manual-triage suggestions, not ancestry labels."
        )
    else:
        st.caption(
            f"{len(lineage):,} clustered models shown; family predictions are available "
            "only for eligible audited queries."
        )
    _plotly_chart(
        _plot_galaxy(
            lineage,
            color_by,
            size_by,
            "2D cluster galaxy",
            edge_density=edge_density,
            zoom=atlas_zoom,
        ),
        key=f"htsr:{run_id}:lineage:galaxy",
        scroll_zoom=gesture_zoom,
    )
    _render_color_legend(lineage, color_by)
    st.dataframe(
        lineage,
        width="stretch",
        height=SECONDARY_PLOT_HEIGHT,
        hide_index=True,
    )


def _render_frozen_evidence(
    outputs: StudyOutputs,
    scaleup: ScaleupOutputs | None = None,
) -> None:
    provenance_values = []
    if "estimator_provenance" in outputs.lineage_predictions.columns:
        provenance_values = sorted(
            outputs.lineage_predictions["estimator_provenance"].dropna().astype(str).unique().tolist()
        )
    st.warning(
        "Stored alpha values use the DKS power-law estimator, not the paper's fixed-Hill estimator. "
        "The frozen tables cannot reproduce fixed-Hill estimates without raw spectral inputs."
    )
    st.markdown("**Estimator provenance**")
    st.dataframe(
        pd.DataFrame(
            {
                "field": ["feature schema", "lineage estimator provenance", "run"],
                "value": [
                    str(outputs.provenance.get("feature_schema_version", "not recorded")),
                    "; ".join(provenance_values) if provenance_values else "stored DKS artifact contract",
                    str(outputs.provenance.get("run_id", outputs.run_dir.name)),
                ],
            }
        ),
        width="stretch",
        hide_index=True,
    )
    with st.expander("Frozen lineage metrics"):
        st.json(dict(outputs.lineage_metrics))
    with st.expander("Frozen performance metrics"):
        st.json(dict(outputs.performance_metrics))
    with st.expander("Frozen exclusions"):
        st.dataframe(outputs.exclusions, width="stretch", height=SECONDARY_PLOT_HEIGHT, hide_index=True)
    report_path = outputs.run_dir / "report.md"
    with st.expander("Frozen report"):
        try:
            st.markdown(report_path.read_text(encoding="utf-8"))
        except OSError as error:
            st.error(f"The frozen report is unavailable: {error}")
    if scaleup is not None:
        with st.expander("Scale-up report", expanded=True):
            st.markdown(scaleup.report)
        with st.expander("Scale-up metadata"):
            st.json(dict(scaleup.metadata))
        with st.expander("Broad-lineage sensitivity"):
            st.dataframe(
                scaleup.broad_cluster_metrics,
                width="stretch",
                hide_index=True,
            )


def render_ablation_atlas(
    run_path: str | Path | None = None,
    scaleup_path: str | Path | None = None,
) -> None:
    _render_dashboard_chrome()
    st.title("Spectral Atlas")
    st.caption("Read-only lineage and performance evidence from retained study artifacts.")

    discovered_study = default_study_run()
    discovered_scaleup = default_scaleup_run()
    configured_path = str(
        run_path if run_path is not None else discovered_study or ""
    )
    configured_scaleup = str(
        scaleup_path if scaleup_path is not None else discovered_scaleup or ""
    )
    with st.sidebar:
        st.subheader("Data sources")
        if run_path is None:
            configured_path = st.text_input(
                "Retained study",
                value=configured_path,
                placeholder="research_runs/htsr_study/runs/<run-id>",
            ).strip()
        else:
            st.caption(configured_path)
        if scaleup_path is None:
            configured_scaleup = st.text_input(
                "Scale-up study",
                value=configured_scaleup,
                placeholder="research_runs/htsr_scaleup/runs/<run-id>",
                help="Optional overlay for the full census and leading alpha-profile predictions.",
            ).strip()
        elif configured_scaleup:
            st.caption(configured_scaleup)

    if not configured_path:
        st.warning("Frozen study unavailable")
        st.caption("Provide a retained study directory or set HTSR_STUDY_RUN.")
        return

    progress_bar = st.progress(0.0, text="0% Loading study")
    status = st.empty()
    try:
        progress_bar.progress(0.15, text="15% Validating run files")
        status.caption("Reading validated frozen CSV and JSON outputs.")
        run_state_token = _frozen_run_state_token(configured_path)
        outputs = _cached_study_outputs(configured_path, run_state_token)
    except Exception as error:
        progress_bar.empty()
        status.empty()
        st.error("Frozen study unavailable")
        st.caption(str(error))
        return

    scaleup: ScaleupOutputs | None = None
    scaleup_error: Exception | None = None
    if configured_scaleup:
        try:
            progress_bar.progress(0.55, text="55% Loading scale-up results")
            status.caption("Reading the large-scale census and direct alpha predictions.")
            scaleup_token = _scaleup_run_state_token(configured_scaleup)
            scaleup = _cached_scaleup_outputs(configured_scaleup, scaleup_token)
        except Exception as error:
            scaleup_error = error

    progress_bar.progress(0.80, text="80% Preparing display tables")
    status.caption("Adapting retained tables for the 2D viewer.")
    cache = (
        _dashboard_cache_from_scaleup(outputs, scaleup)
        if scaleup is not None
        else _dashboard_cache_from_study(outputs)
    )
    progress_bar.progress(1.0, text="100% Study loaded")
    progress_bar.empty()
    status.empty()
    if scaleup_error is not None:
        st.warning(f"Scale-up overlay unavailable; showing the retained run only. {scaleup_error}")

    _render_cards(_study_summary_cards(outputs, scaleup))
    if scaleup is not None:
        st.caption(f"Scale-up: {scaleup.run_dir} | retained base: {outputs.run_dir}")
    else:
        st.caption(f"Run: {outputs.run_dir}")

    panel = st.radio("View", options=_panel_options(), horizontal=True)
    targets = _target_options(cache)
    target = targets[0] if targets else None
    with st.sidebar:
        if targets:
            target = st.selectbox("Performance target", options=targets, index=0)

    frame = _with_display_columns(cache.metadata.copy(), target)
    with st.sidebar:
        st.subheader("View controls")
        search_text = st.text_input("Search models", value="", placeholder="name fragment")
        if panel == "Lineage":
            color_options = _lineage_color_options(frame)
        else:
            color_options = _color_options(frame, target)
        color_by = st.selectbox("Color", options=color_options or ["Predicted cluster"])
        size_options = ["fixed"]
        for column in ("#Params (B)", f"{target} absolute_error" if target else "", "overall_log_alpha"):
            if column and column in frame.columns:
                size_options.append(column)
        size_by = st.selectbox("Size", options=size_options)
        edge_density = st.selectbox("Edges", options=["fast", "balanced", "dense", "off"], index=0)
        gesture_zoom = st.checkbox("Galaxy gesture zoom", value=False)
        atlas_zoom_control = st.slider("Galaxy zoom", min_value=1.0, max_value=6.0, value=1.0, step=0.1)
        atlas_zoom = _atlas_zoom_from_control(atlas_zoom_control)

        with st.expander("Filters", expanded=False):
            families: list[str] = []
            if "coarse_family" in frame.columns:
                families = st.multiselect(
                    "Families",
                    options=sorted(frame["coarse_family"].dropna().astype(str).unique().tolist()),
                    default=[],
                )
            architecture_column = next(
                (column for column in ("Architecture", "architecture") if column in frame.columns),
                None,
            )
            architectures: list[str] = []
            if architecture_column is not None:
                architectures = st.multiselect(
                    "Architectures",
                    options=sorted(frame[architecture_column].dropna().astype(str).unique().tolist()),
                    default=[],
                )
                if architecture_column != "Architecture":
                    frame["Architecture"] = frame[architecture_column]
            clusters = st.multiselect(
                "Clusters",
                options=sorted(frame["cluster"].dropna().astype(int).unique().tolist()),
                default=[],
            )

    filtered = _filter_frame(frame, families, architectures, clusters, search_text=search_text)
    if panel == "Atlas":
        atlas_frame = _lineage_display_frame(filtered)
        if atlas_frame.empty:
            st.info("No clustered models match the current filters.")
        else:
            st.caption(
                f"{len(atlas_frame):,} clustered models rendered from "
                f"{len(filtered):,} filtered rows; Data retains every loaded row."
            )
            _plotly_chart(
                _plot_galaxy(
                    atlas_frame,
                    color_by,
                    size_by,
                    "2D cluster galaxy",
                    target=target,
                    edge_density=edge_density,
                    zoom=atlas_zoom,
                ),
                key=(
                    f"htsr:{cache.cache_dir.name}:atlas:galaxy"
                ),
                scroll_zoom=gesture_zoom,
            )
            _render_color_legend(atlas_frame, color_by)
    elif panel == "Prediction":
        if target is None:
            st.info("No out-of-fold performance target is available.")
        else:
            _render_frozen_prediction(outputs, cache, target, scaleup)
    elif panel == "Lineage":
        _render_frozen_lineage(
            outputs,
            filtered,
            color_by=color_by,
            size_by=size_by,
            edge_density=edge_density,
            gesture_zoom=gesture_zoom,
            atlas_zoom=atlas_zoom,
        )
    elif panel == "Profiles":
        cluster_options = sorted(frame["cluster"].dropna().astype(int).unique().tolist())
        if not cluster_options:
            st.info("No frozen cluster profiles are available.")
        else:
            cluster_choice = st.selectbox("Cluster", options=cluster_options, index=0)
            _render_profiles_tab(cache, int(cluster_choice))
            st.dataframe(cache.cluster_summary, width="stretch", height=SECONDARY_PLOT_HEIGHT, hide_index=True)
    elif panel == "Evidence":
        _render_frozen_evidence(outputs, scaleup)
    elif panel == "Data":
        st.caption(f"{len(filtered):,} filtered rows from {len(cache.metadata):,} loaded atlas models.")
        st.dataframe(filtered, width="stretch", height=TABLE_HEIGHT, hide_index=True)
        st.download_button(
            "Download current atlas rows",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="spectral_atlas_rows.csv",
            mime="text/csv",
        )
