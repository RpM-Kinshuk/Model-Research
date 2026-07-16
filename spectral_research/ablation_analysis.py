from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from alpha_cluster_dashboard.data import MODULE_ROLE_MAP, canonicalize_module_name


BENCHMARK_TARGETS = (
    "Average \u2b06\ufe0f",
    "IFEval",
    "BBH",
    "MATH Lvl 5",
    "GPQA",
    "MUSR",
    "MMLU-PRO",
)
SPECTRAL_FEATURE_COLUMNS = (
    "alpha_global_mean",
    "alpha_global_median",
    "alpha_global_std",
    "alpha_global_mean_abs_distance_to_2",
    "alpha_global_median_abs_distance_to_2",
    "alpha_fraction_lt_2",
    "alpha_fraction_gt_2",
    "alpha_attn_mean",
    "alpha_mlp_mean",
    "alpha_mlp_minus_attn",
    "log_norm_global_mean",
    "spectral_norm_global_mean",
    "num_layers_observed",
    "valid_alpha_count",
)
UNKNOWN_VALUES = {"", "?", "unknown", "n/a", "none", "nan"}
LAYER_PREFIXES = {"layers", "h", "layer"}
FAMILY_HINTS = (
    ("codellama", ("codellama", "code-llama", "code llama")),
    ("tinyllama", ("tinyllama", "tiny-llama", "tiny llama")),
    ("mixtral", ("mixtral",)),
    ("mistral", ("mistral",)),
    ("llama", ("llama", "meta-llama")),
    ("qwen", ("qwen",)),
    ("gemma", ("gemma",)),
    ("phi", ("phi-", "phi_", "phi2", "phi-2", "phi3", "phi-3")),
    ("yi", ("yi-", "yi_", "/yi", "yi1", "yi-1")),
    ("deepseek", ("deepseek",)),
    ("pythia", ("pythia",)),
    ("falcon", ("falcon",)),
    ("gpt-neox", ("gpt-neox", "gpt_neox", "gptneox")),
    ("gpt2", ("gpt2", "gpt-2")),
    ("opt", ("opt-", "/opt", "optforcausallm")),
    ("bloom", ("bloom",)),
    ("starcoder", ("starcoder",)),
    ("solar", ("solar",)),
    ("vicuna", ("vicuna",)),
    ("openchat", ("openchat",)),
    ("t5", ("t5",)),
)


def safe_stem_to_model_id(stem: str) -> str:
    return stem.replace("--", "/")


def artifact_model_ids(directory: Path, suffix: str) -> set[str]:
    if not directory.exists():
        return set()
    return {safe_stem_to_model_id(path.stem) for path in directory.glob(f"*{suffix}")}


def load_terminal_statuses(output_dir: Path) -> pd.DataFrame:
    status_dir = output_dir / "logs" / "terminal_status"
    rows: list[dict[str, object]] = []
    if not status_dir.exists():
        return pd.DataFrame(columns=["model_id", "status", "stage", "reason", "message", "attempt", "ts"])
    for path in sorted(status_dir.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            row = {
                "model_id": safe_stem_to_model_id(path.stem),
                "status": "invalid_json",
                "stage": "read",
                "reason": "invalid_json",
                "message": "",
                "attempt": np.nan,
                "ts": "",
            }
        row.setdefault("model_id", safe_stem_to_model_id(path.stem))
        rows.append(row)
    return pd.DataFrame(rows)


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _known_text(value: object) -> str:
    text = _text(value)
    return "" if text.lower() in UNKNOWN_VALUES else text


def _finite(values: Iterable[object], *, positive: bool = False) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if positive:
        numeric = numeric[numeric > 0]
    return numeric


def _safe_stat(values: np.ndarray, fn, default: float = float("nan")) -> float:
    if values.size == 0:
        return default
    return float(fn(values))


def _parse_layer(longname: object) -> int | None:
    text = _text(longname)
    tokens = text.split(".")
    for index in range(len(tokens) - 1):
        if tokens[index] in LAYER_PREFIXES and tokens[index + 1].isdigit():
            return int(tokens[index + 1])
    return None


def _module_from_longname(longname: object) -> str | None:
    text = _text(longname)
    if not text:
        return None
    known_modules = sorted(MODULE_ROLE_MAP, key=len, reverse=True)
    for module_name in known_modules:
        if module_name in text:
            return module_name
    return None


def _role_frame(stats: pd.DataFrame) -> pd.DataFrame:
    working = stats.copy()
    working["alpha"] = pd.to_numeric(working.get("alpha"), errors="coerce")
    working["log_norm"] = pd.to_numeric(working.get("log_norm"), errors="coerce")
    working["spectral_norm"] = pd.to_numeric(working.get("spectral_norm"), errors="coerce")
    working["layer"] = working.get("longname", pd.Series(index=working.index, dtype="object")).map(_parse_layer)
    working["raw_module"] = working.get("longname", pd.Series(index=working.index, dtype="object")).map(_module_from_longname)
    working["module_role"] = working["raw_module"].map(
        lambda value: canonicalize_module_name(value) if isinstance(value, str) else "unknown"
    )
    working = working[working["alpha"].notna() & working["module_role"].ne("unknown")].copy()
    if working.empty:
        return pd.DataFrame(
            columns=[
                "model_id",
                "module_role",
                "valid_alpha_count",
                "alpha_mean",
                "alpha_median",
                "alpha_std",
                "alpha_q25",
                "alpha_q75",
                "log_norm_mean",
                "spectral_norm_mean",
                "layer_count",
            ]
        )
    grouped = working.groupby("module_role", as_index=False)
    role_rows = grouped.agg(
        valid_alpha_count=("alpha", "count"),
        alpha_mean=("alpha", "mean"),
        alpha_median=("alpha", "median"),
        alpha_std=("alpha", "std"),
        alpha_q25=("alpha", lambda values: float(np.nanpercentile(values, 25))),
        alpha_q75=("alpha", lambda values: float(np.nanpercentile(values, 75))),
        log_norm_mean=("log_norm", "mean"),
        spectral_norm_mean=("spectral_norm", "mean"),
        layer_count=("layer", lambda values: int(pd.Series(values).dropna().nunique())),
    )
    model_id = _text(working["model_id"].dropna().iloc[0]) if "model_id" in working and working["model_id"].notna().any() else ""
    role_rows.insert(0, "model_id", model_id)
    return role_rows


def summarize_model_stats(path: Path) -> tuple[dict[str, object], pd.DataFrame]:
    stats = pd.read_csv(path)
    if "model_id" in stats.columns and stats["model_id"].notna().any():
        model_id = _text(stats["model_id"].dropna().iloc[0])
    else:
        model_id = safe_stem_to_model_id(path.stem)
    alphas = _finite(stats.get("alpha", []), positive=True)
    log_norm = _finite(stats.get("log_norm", []))
    spectral_norm = _finite(stats.get("spectral_norm", []))
    layers = [
        layer
        for layer in stats.get("longname", pd.Series(index=stats.index, dtype="object")).map(_parse_layer).tolist()
        if layer is not None
    ]
    role_summary = _role_frame(stats)
    role_summary.loc[:, "model_id"] = model_id

    attn_roles = role_summary[role_summary["module_role"].astype(str).str.startswith("attn_")]
    mlp_roles = role_summary[role_summary["module_role"].astype(str).str.startswith("mlp_")]
    attn_mean = float(attn_roles["alpha_mean"].mean()) if not attn_roles.empty else float("nan")
    mlp_mean = float(mlp_roles["alpha_mean"].mean()) if not mlp_roles.empty else float("nan")

    distance = np.abs(alphas - 2.0) if alphas.size else np.array([], dtype=float)
    row = {
        "model_id": model_id,
        "stats_path": str(path),
        "valid_alpha_count": int(alphas.size),
        "num_layers_observed": int(len(set(layers))),
        "num_module_roles_observed": int(role_summary["module_role"].nunique()) if not role_summary.empty else 0,
        "alpha_global_mean": _safe_stat(alphas, np.nanmean),
        "alpha_global_median": _safe_stat(alphas, np.nanmedian),
        "alpha_global_std": _safe_stat(alphas, np.nanstd),
        "alpha_global_q25": _safe_stat(alphas, lambda values: np.nanpercentile(values, 25)),
        "alpha_global_q75": _safe_stat(alphas, lambda values: np.nanpercentile(values, 75)),
        "alpha_global_mean_abs_distance_to_2": _safe_stat(distance, np.nanmean),
        "alpha_global_median_abs_distance_to_2": _safe_stat(distance, np.nanmedian),
        "alpha_fraction_lt_2": float(np.mean(alphas < 2.0)) if alphas.size else float("nan"),
        "alpha_fraction_gt_2": float(np.mean(alphas > 2.0)) if alphas.size else float("nan"),
        "alpha_attn_mean": attn_mean,
        "alpha_mlp_mean": mlp_mean,
        "alpha_mlp_minus_attn": float(mlp_mean - attn_mean) if np.isfinite(mlp_mean) and np.isfinite(attn_mean) else float("nan"),
        "log_norm_global_mean": _safe_stat(log_norm, np.nanmean),
        "spectral_norm_global_mean": _safe_stat(spectral_norm, np.nanmean),
    }
    return row, role_summary


def infer_coarse_family(row: pd.Series) -> str:
    fields = [
        _known_text(row.get("source_model", "")),
        _known_text(row.get("Base Model", "")),
        _known_text(row.get("Architecture", "")),
        _known_text(row.get("model_id", "")),
        _known_text(row.get("fullname", "")),
    ]
    haystack = " ".join(fields).lower()
    for family, hints in FAMILY_HINTS:
        if any(hint in haystack for hint in hints):
            return family
    architecture = _known_text(row.get("Architecture", ""))
    if architecture:
        return architecture.lower().replace("forcausallm", "").replace("model", "")
    return "unknown"


def summarize_run_coverage(
    selected: pd.DataFrame,
    statuses: pd.DataFrame,
    *,
    stats_model_ids: set[str],
    metrics_model_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    per_model = selected.copy()
    status_columns = ["model_id", "status", "stage", "reason", "message", "attempt", "ts"]
    status_frame = statuses.copy()
    for column in status_columns:
        if column not in status_frame.columns:
            status_frame[column] = pd.NA
    status_frame = status_frame.loc[:, status_columns].drop_duplicates("model_id", keep="last")
    status_frame = status_frame.rename(
        columns={
            "status": "terminal_status",
            "stage": "terminal_stage",
            "reason": "terminal_reason",
            "message": "terminal_message",
            "attempt": "terminal_attempt",
            "ts": "terminal_ts",
        }
    )
    per_model = per_model.merge(status_frame, on="model_id", how="left")
    per_model["terminal_status"] = per_model["terminal_status"].fillna("not_attempted")
    per_model["terminal_reason"] = per_model["terminal_reason"].fillna("missing_terminal_status")
    per_model["has_stats_artifact"] = per_model["model_id"].isin(stats_model_ids)
    per_model["has_metrics_artifact"] = per_model["model_id"].isin(metrics_model_ids)
    per_model["coarse_family"] = per_model.apply(infer_coarse_family, axis=1)
    selected_rows = int(len(per_model))
    success_rows = int(per_model["terminal_status"].eq("success").sum())
    stats_rows = int(per_model["has_stats_artifact"].sum())
    metrics_rows = int(per_model["has_metrics_artifact"].sum())
    summary = {
        "selected_rows": selected_rows,
        "terminal_status_rows": int(len(statuses)),
        "success_rows": success_rows,
        "failed_rows": int(per_model["terminal_status"].eq("failed").sum()),
        "not_attempted_rows": int(per_model["terminal_status"].eq("not_attempted").sum()),
        "artifact_without_terminal_rows": int(
            (per_model["terminal_status"].eq("not_attempted") & per_model["has_stats_artifact"]).sum()
        ),
        "no_terminal_no_artifact_rows": int(
            (per_model["terminal_status"].eq("not_attempted") & ~per_model["has_stats_artifact"]).sum()
        ),
        "stats_artifact_rows": stats_rows,
        "metrics_artifact_rows": metrics_rows,
        "success_rate": round(success_rows / selected_rows, 6) if selected_rows else 0.0,
        "stats_artifact_rate": round(stats_rows / selected_rows, 6) if selected_rows else 0.0,
        "metrics_artifact_rate": round(metrics_rows / selected_rows, 6) if selected_rows else 0.0,
    }
    return per_model, summary


def group_coverage(per_model: pd.DataFrame, group_column: str) -> pd.DataFrame:
    if group_column not in per_model.columns or per_model.empty:
        return pd.DataFrame(columns=[group_column, "selected_rows", "success_rows", "success_rate"])
    grouped = (
        per_model.assign(
            success=per_model["terminal_status"].eq("success"),
            failed=per_model["terminal_status"].eq("failed"),
        )
        .groupby(group_column, dropna=False)
        .agg(
            selected_rows=("model_id", "count"),
            success_rows=("success", "sum"),
            failed_rows=("failed", "sum"),
            stats_artifact_rows=("has_stats_artifact", "sum"),
            metrics_artifact_rows=("has_metrics_artifact", "sum"),
        )
        .reset_index()
    )
    grouped["success_rate"] = grouped["success_rows"] / grouped["selected_rows"]
    grouped["stats_artifact_rate"] = grouped["stats_artifact_rows"] / grouped["selected_rows"]
    return grouped.sort_values(["selected_rows", group_column], ascending=[False, True]).reset_index(drop=True)


def failure_breakdown(per_model: pd.DataFrame, group_column: str) -> pd.DataFrame:
    if group_column not in per_model.columns:
        return pd.DataFrame(columns=[group_column, "terminal_reason", "rows"])
    failed = per_model[per_model["terminal_status"].eq("failed")].copy()
    if failed.empty:
        return pd.DataFrame(columns=[group_column, "terminal_reason", "rows"])
    return (
        failed.groupby([group_column, "terminal_reason"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["rows", group_column, "terminal_reason"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def load_stats_summaries(stats_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict[str, object]] = []
    role_frames: list[pd.DataFrame] = []
    for path in sorted(stats_dir.glob("*.csv")):
        try:
            model_row, role_frame = summarize_model_stats(path)
        except (pd.errors.EmptyDataError, ValueError, OSError):
            continue
        model_rows.append(model_row)
        if not role_frame.empty:
            role_frames.append(role_frame)
    model_summary = pd.DataFrame(model_rows)
    role_summary = pd.concat(role_frames, ignore_index=True, sort=False) if role_frames else pd.DataFrame()
    return model_summary, role_summary


def spectral_correlations(joined: pd.DataFrame, *, min_pairs: int = 25) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in BENCHMARK_TARGETS:
        if target not in joined.columns:
            continue
        target_values = pd.to_numeric(joined[target], errors="coerce")
        for feature in SPECTRAL_FEATURE_COLUMNS:
            if feature not in joined.columns:
                continue
            feature_values = pd.to_numeric(joined[feature], errors="coerce")
            mask = target_values.notna() & feature_values.notna()
            n = int(mask.sum())
            if n < min_pairs:
                continue
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "n": n,
                    "pearson": float(feature_values[mask].corr(target_values[mask], method="pearson")),
                    "spearman": float(feature_values[mask].corr(target_values[mask], method="spearman")),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["abs_spearman"] = result["spearman"].abs()
    return result.sort_values(["target", "abs_spearman"], ascending=[True, False]).reset_index(drop=True)


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 12) -> str:
    if frame.empty:
        return "_No rows._"
    table = frame.loc[:, [column for column in columns if column in frame.columns]].head(max_rows).copy()

    def cell(value: object) -> str:
        if pd.isna(value):
            return "n/a"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.3f}"
        return str(value)

    header = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = ["| " + " | ".join(cell(value) for value in row) + " |" for row in table.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def render_report(
    *,
    source_data_dir: Path,
    run_output_dir: Path,
    curation_summary: dict[str, object],
    run_summary: dict[str, object],
    status_summary: pd.DataFrame,
    coverage_by_type: pd.DataFrame,
    coverage_by_lineage: pd.DataFrame,
    coverage_by_family: pd.DataFrame,
    correlations: pd.DataFrame,
) -> str:
    top_corr = correlations.sort_values("abs_spearman", ascending=False) if not correlations.empty else correlations
    lines = [
        "# Ablation Spectral Run Analysis",
        "",
        "## Sources",
        f"- Curation data: `{source_data_dir}`",
        f"- Run artifacts: `{run_output_dir}`",
        "",
        "## Executive Summary",
        f"- Curated selected rows: {curation_summary.get('selected_rows', 'n/a')}",
        f"- Terminal successes: {run_summary.get('success_rows', 0)} / {run_summary.get('selected_rows', 0)} ({run_summary.get('success_rate', 0.0):.1%})",
        f"- Stats artifacts: {run_summary.get('stats_artifact_rows', 0)} / {run_summary.get('selected_rows', 0)} ({run_summary.get('stats_artifact_rate', 0.0):.1%})",
        f"- Metrics HDF5 artifacts: {run_summary.get('metrics_artifact_rows', 0)} / {run_summary.get('selected_rows', 0)} ({run_summary.get('metrics_artifact_rate', 0.0):.1%})",
        f"- Selected rows with artifacts but no terminal JSON: {run_summary.get('artifact_without_terminal_rows', 0)}",
        f"- Selected rows with neither artifact nor terminal JSON: {run_summary.get('no_terminal_no_artifact_rows', 0)}",
        "- Hierarchy is partly present through `source_model`, `base_model_relation`, `lineage_eval_group`, and `primary_type_bucket`; this report adds a coarse fallback family label from source/model/architecture hints for stratification only.",
        "",
        "## Terminal Status",
        _markdown_table(status_summary, ["terminal_status", "terminal_stage", "terminal_reason", "rows"], max_rows=20),
        "",
        "## Coverage By Type",
        _markdown_table(
            coverage_by_type,
            ["primary_type_bucket", "selected_rows", "success_rows", "failed_rows", "success_rate", "stats_artifact_rate"],
            max_rows=20,
        ),
        "",
        "## Coverage By Lineage Group",
        _markdown_table(
            coverage_by_lineage,
            ["lineage_eval_group", "selected_rows", "success_rows", "failed_rows", "success_rate", "stats_artifact_rate"],
            max_rows=20,
        ),
        "",
        "## Top Coarse Families",
        _markdown_table(
            coverage_by_family,
            ["coarse_family", "selected_rows", "success_rows", "failed_rows", "success_rate", "stats_artifact_rate"],
            max_rows=20,
        ),
        "",
        "## Strongest Spectral-Score Correlations",
        _markdown_table(top_corr, ["target", "feature", "n", "spearman", "pearson"], max_rows=20),
        "",
        "## Interpretation Notes",
        "- The paper's most relevant lesson is to treat PL_Alpha_Hill as an ESD shape signature and inspect module families separately before over-collapsing.",
        "- This run is suitable for broad spectral/profile analysis of successful models, but success is not uniform by type. MoE and large/complex models have lower completion rates, so downstream comparisons should report coverage alongside any spectral trend.",
        "- Known lineage is sparse in the selected set; fine-tune and merge conclusions should be treated as higher-confidence where `lineage_eval_group` is known and exploratory where only coarse family hints are available.",
    ]
    return "\n".join(lines) + "\n"
