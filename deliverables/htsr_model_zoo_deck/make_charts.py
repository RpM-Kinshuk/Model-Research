#!/usr/bin/env python3
"""Build figures and compact tables for the metadata-isolated HTSR deck."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/model_zoo_deck_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "assets"
RUN_ROOT = ROOT / "research_runs/htsr_weight_only/runs"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_research.study.artifacts import validate_artifact_manifest
REQUIRED = (
    "metadata.json",
    "population_ledger.csv",
    "lineage_cluster_assignments.csv",
    "lineage_cluster_selection.csv",
    "lineage_internal_metrics.csv",
    "lineage_validation_metrics.csv",
    "lineage_gallery_metrics.csv",
    "performance_metrics.csv",
    "performance_predictions.csv",
    "performance_cluster_holdout_metrics.csv",
    "performance_cluster_holdout_predictions.csv",
)

BERKELEY = "#002676"
BOOKING = "#003B95"
DARTMOUTH = "#00693E"
GOLD = "#FDB515"
CYAN = "#00AEEF"
INK = "#14213D"
MUTED = "#667085"
GRID = "#D9E2EC"
PALE_BLUE = "#EAF2FF"
PALE_GREEN = "#E4F3EA"


def complete_run(path: Path) -> bool:
    if not path.is_dir() or not (path / "artifact_manifest.json").is_file():
        return False
    try:
        validate_artifact_manifest(
            path,
            required_files=REQUIRED,
            expected_run_id=path.name,
        )
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def select_run() -> Path:
    override = os.environ.get("HTSR_WEIGHT_ONLY_RUN")
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        if not complete_run(path):
            raise FileNotFoundError(f"Incomplete HTSR_WEIGHT_ONLY_RUN: {path}")
        return path
    candidates = [path for path in RUN_ROOT.iterdir() if complete_run(path)]
    if not candidates:
        raise FileNotFoundError(f"No complete metadata-isolated run under {RUN_ROOT}")
    return max(
        candidates,
        key=lambda path: (
            max((path / name).stat().st_mtime_ns for name in REQUIRED),
            path.name,
        ),
    )


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def finish(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def clean_axis(ax: plt.Axes, *, grid: str = "y") -> None:
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis=grid, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def average_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics[
        metrics["target"].isin(("modern_average", "legacy_average"))
    ].copy()


def row(
    metrics: pd.DataFrame, generation: str, method: str
) -> dict[str, object]:
    target = f"{generation}_average"
    selected = metrics[
        (metrics["generation"] == generation)
        & (metrics["target"] == target)
        & (metrics["feature_block"] == method)
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one {generation}/{method} metric row")
    return selected.iloc[0].to_dict()


def corpus_coverage(metadata: dict, ledger: pd.DataFrame) -> dict[str, int]:
    lineage_flag = ledger.get(
        "lineage_profile_available", ledger["htsr_profile_available"]
    ).astype(bool)
    performance_flag = ledger.get(
        "performance_profile_available", ledger["htsr_profile_available"]
    ).astype(bool)
    strict_flag = ledger.get(
        "strict_seven_role_available", pd.Series(False, index=ledger.index)
    ).astype(bool)

    def source_counts(mask: pd.Series) -> tuple[int, int]:
        selected = ledger.loc[mask, "source_collection"].astype(str)
        return (
            int(selected.str.contains("phase2").sum()),
            int(selected.str.contains("ablation").sum()),
        )

    phase2_lineage, ablation_lineage = source_counts(lineage_flag)
    phase2_performance, ablation_performance = source_counts(performance_flag)
    phase2_strict, ablation_strict = source_counts(strict_flag)
    phase2_listed = int(ledger["source_collection"].astype(str).str.contains("phase2").sum())
    ablation_listed = int(ledger["source_collection"].astype(str).str.contains("ablation").sum())

    study_run = Path(str(metadata["sources"]["study_run"]))
    provenance = json.loads((study_run / "provenance.json").read_text(encoding="utf-8"))
    phase2_analyzed = int(provenance["counts"]["development_stats_artifacts"])
    ablation_analyzed = int(provenance["counts"]["external_stats_artifacts"])

    funnels = [
        (
            "Phase 2",
            [phase2_listed, phase2_analyzed, phase2_performance, phase2_lineage, phase2_strict],
            BERKELEY,
        ),
        (
            "Ablation",
            [ablation_listed, ablation_analyzed, ablation_performance, ablation_lineage, ablation_strict],
            DARTMOUTH,
        ),
    ]
    labels = ["Listed", "Analyzed", "Performance set", "Lineage set", "Strict core"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    for ax, (title, values, accent) in zip(axes, funnels):
        bars = ax.barh(
            labels[::-1],
            values[::-1],
            color=[accent, DARTMOUTH, GOLD, BOOKING, GRID],
            height=0.58,
        )
        for bar, value in zip(bars, values[::-1]):
            ax.text(
                value + max(values) * 0.025,
                bar.get_y() + bar.get_height() / 2,
                f"{value:,}",
                va="center",
                color=INK,
                weight="bold",
            )
        ax.set_xlim(0, max(values) * 1.22)
        ax.set_title(title, loc="left", fontsize=14, pad=11)
        ax.set_xlabel("models")
        clean_axis(ax, grid="x")
    fig.suptitle(
        "Strict paper comparison and broader operational coverage",
        x=0.01,
        y=1.03,
        ha="left",
        fontsize=10,
        color=MUTED,
    )
    finish(fig, "corpus_coverage.png")
    return {
        "phase2_listed": phase2_listed,
        "phase2_analyzed": phase2_analyzed,
        "phase2_profiles": phase2_lineage,
        "phase2_performance": phase2_performance,
        "phase2_strict": phase2_strict,
        "ablation_listed": ablation_listed,
        "ablation_analyzed": ablation_analyzed,
        "ablation_profiles": ablation_lineage,
        "ablation_performance": ablation_performance,
        "ablation_strict": ablation_strict,
    }


def load_study_collections(
    metadata: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, Path]:
    validation_run = Path(str(metadata["sources"]["validation_run"]))
    validation_metadata = json.loads(
        (validation_run / "metadata.json").read_text(encoding="utf-8")
    )
    sources = validation_metadata["sources"]
    phase2 = pd.read_csv(Path(sources["phase2"]["path"]), low_memory=False)
    ablation_path = Path(sources["ablation"]["path"])
    ablation = pd.read_csv(ablation_path, low_memory=False)
    summary_path = ablation_path.parent / "ablation_summary.json"
    ablation_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return phase2, ablation, ablation_summary, Path(sources["graph"]["path"])


def collection_sources(phase2: pd.DataFrame, ablation: pd.DataFrame) -> None:
    source_specs = [
        (
            phase2["candidate_source"].value_counts(),
            {
                "atlas": "Model Atlas graph",
                "seed_metadata": "sampled metadata",
                "leaderboard_dataset": "Open LLM Leaderboard",
                "hf_search": "targeted Hub search",
            },
            f"Phase 2: {len(phase2):,} selected records",
            BERKELEY,
        ),
        (
            ablation["eval_source"].value_counts(),
            {
                "open_llm_leaderboard_results": "older leaderboard",
                "open_llm_leaderboard_contents": "current leaderboard",
            },
            f"Ablation: {len(ablation):,} disjoint models",
            DARTMOUTH,
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.25))
    for ax, (counts, names, title, color) in zip(axes, source_specs):
        counts = counts.rename(index=names).sort_values()
        shares = counts / counts.sum()
        bars = ax.barh(counts.index, shares, color=color, height=0.58)
        for bar, share in zip(bars, shares):
            ax.text(
                share + 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{share:.1%}",
                va="center",
                color=INK,
                weight="bold",
                fontsize=9,
            )
        ax.set_xlim(0, max(1.0, float(shares.max()) * 1.18))
        ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        ax.set_xlabel("share of selected dataset")
        ax.set_title(title, loc="left", fontsize=13, pad=10)
        clean_axis(ax, grid="x")
    finish(fig, "corpus_sources.png")


def model_type_distribution(phase2: pd.DataFrame, ablation: pd.DataFrame) -> None:
    order = [
        "adapter",
        "finetune",
        "quantized",
        "merge",
        "multimodal",
        "moe",
        "other_text",
        "base_source",
    ]
    labels = [
        "Adapter",
        "Fine-tune",
        "Quantized",
        "Merge",
        "Multimodal",
        "MoE",
        "Other text",
        "Base source",
    ]
    phase_shares = phase2["primary_type_bucket"].value_counts(normalize=True).reindex(order, fill_value=0)
    ablation_shares = ablation["primary_type_bucket"].value_counts(normalize=True).reindex(order, fill_value=0)
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.barh(y + 0.18, phase_shares, height=0.34, color=BERKELEY, label="Phase 2")
    ax.barh(y - 0.18, ablation_shares, height=0.34, color=DARTMOUTH, label="Ablation")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.72, float(max(phase_shares.max(), ablation_shares.max())) * 1.08))
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_xlabel("share of selected dataset")
    ax.set_title("Metadata selection buckets, not verified model types", loc="left", fontsize=12, pad=10)
    ax.legend(frameon=False, ncols=2, loc="lower right")
    clean_axis(ax, grid="x")
    finish(fig, "corpus_types.png")


def _clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _architecture_group(value: str) -> str:
    text = value.casefold()
    if not text or text in {"unknown", "?", "nan"}:
        return "Missing / unknown"
    if any(token in text for token in ("llava", "internvl", "vision", "mllama")):
        return "Vision-language"
    if "mixtral" in text:
        return "Mixtral"
    if "llama" in text:
        return "Llama"
    if "mistral" in text:
        return "Mistral"
    if "qwen" in text:
        return "Qwen"
    if "gemma" in text:
        return "Gemma"
    if "phi" in text:
        return "Phi"
    return "Other labelled"


def architecture_groups(frame: pd.DataFrame, graph_path: Path) -> pd.Series:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    graph_architecture: dict[str, str] = {}
    for node in graph.get("nodes", []):
        value = (node.get("attributes") or {}).get("architectures")
        if isinstance(value, list):
            value = value[0] if value else ""
        graph_architecture[str(node.get("id", "")).casefold()] = _clean_text(value)

    groups: list[str] = []
    for _, row_value in frame.iterrows():
        model_id = _clean_text(row_value.get("repo_id_norm")) or _clean_text(
            row_value.get("model_id")
        )
        candidates = [
            graph_architecture.get(model_id.casefold(), ""),
            _clean_text(row_value.get("Architecture_lb")),
            _clean_text(row_value.get("Architecture")),
            _clean_text(row_value.get("architecture")),
        ]
        architecture = next(
            (
                value
                for value in candidates
                if len(value) > 2 and value.casefold() not in {"unknown", "?", "nan"}
            ),
            "",
        )
        groups.append(_architecture_group(architecture))
    return pd.Series(groups, index=frame.index, dtype="object")


def architecture_distribution(
    phase2: pd.DataFrame, ablation: pd.DataFrame, graph_path: Path
) -> tuple[pd.Series, pd.Series]:
    phase_groups = architecture_groups(phase2, graph_path)
    ablation_groups = architecture_groups(ablation, graph_path)
    order = [
        "Llama",
        "Mistral",
        "Qwen",
        "Mixtral",
        "Gemma",
        "Phi",
        "Vision-language",
        "Other labelled",
        "Missing / unknown",
    ]
    phase_shares = phase_groups.value_counts(normalize=True).reindex(order, fill_value=0)
    ablation_shares = ablation_groups.value_counts(normalize=True).reindex(order, fill_value=0)
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.barh(y + 0.18, phase_shares, height=0.34, color=BERKELEY, label="Phase 2")
    ax.barh(y - 0.18, ablation_shares, height=0.34, color=DARTMOUTH, label="Ablation")
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.46, float(max(phase_shares.max(), ablation_shares.max())) * 1.08))
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_xlabel("share of selected dataset")
    ax.set_title("Architecture families from source records", loc="left", fontsize=14, pad=10)
    ax.legend(frameon=False, ncols=2, loc="lower right")
    clean_axis(ax, grid="x")
    finish(fig, "corpus_architectures.png")
    return phase_groups, ablation_groups


def collection_summary(
    phase2: pd.DataFrame,
    ablation: pd.DataFrame,
    ablation_summary: dict,
    phase_architecture: pd.Series,
    ablation_architecture: pd.Series,
) -> dict[str, object]:
    def available_count(frame: pd.DataFrame, column: str, value: str) -> int:
        return int(frame[column].fillna("").astype(str).eq(value).sum())

    def positive_parameter_count(frame: pd.DataFrame) -> int:
        return int(pd.to_numeric(frame.get("#Params (B)"), errors="coerce").gt(0).sum())

    def benchmark_count(frame: pd.DataFrame) -> int:
        columns = [c for c in frame if c.startswith("Average")]
        return int(frame[columns].apply(pd.to_numeric, errors="coerce").notna().any(axis=1).sum())

    return {
        "phase2": {
            "selected": int(len(phase2)),
            "benchmark_rows": benchmark_count(phase2),
            "lineage_metadata_rows": available_count(phase2, "lineage_status", "known"),
            "architecture_rows": int(phase_architecture.ne("Missing / unknown").sum()),
            "parameter_rows": positive_parameter_count(phase2),
        },
        "ablation": {
            "candidates": int(ablation_summary["total_candidate_rows"]),
            "selected": int(len(ablation)),
            "benchmark_rows": benchmark_count(ablation),
            "complete_benchmark_rows": available_count(
                ablation, "eval_coverage_flag", "complete"
            ),
            "lineage_metadata_rows": available_count(ablation, "lineage_status", "known"),
            "architecture_rows": int(ablation_architecture.ne("Missing / unknown").sum()),
            "parameter_rows": positive_parameter_count(ablation),
            "phase2_overlap_rows": int(ablation_summary["phase2_overlap_rows"]),
            "inaccessible_rows": int(
                ablation_summary["top_rejection_reasons"].get("repo_inaccessible", 0)
            ),
            "duplicate_rows": int(
                ablation_summary["top_rejection_reasons"].get(
                    "duplicate_lower_priority", 0
                )
            ),
        },
    }


def cluster_selection(selection: pd.DataFrame) -> None:
    selected = selection[selection["selected"].astype(bool)].iloc[0]
    fig, ax = plt.subplots(figsize=(9.5, 4.7))
    ax.plot(
        selection["n_clusters"],
        selection["silhouette_cosine"],
        color=BOOKING,
        linewidth=2,
        marker="o",
        markersize=4.5,
    )
    ax.scatter(
        [selected["n_clusters"]],
        [selected["silhouette_cosine"]],
        s=120,
        color=GOLD,
        edgecolor=INK,
        linewidth=0.8,
        zorder=3,
    )
    ax.annotate(
        f"selected: {int(selected['n_clusters'])} clusters\n"
        f"silhouette {selected['silhouette_cosine']:.3f}",
        xy=(selected["n_clusters"], selected["silhouette_cosine"]),
        xytext=(33, 0.58),
        arrowprops={"arrowstyle": "->", "color": MUTED},
        fontsize=10,
        color=INK,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": GRID},
    )
    ax.set_xlabel("candidate clusters")
    ax.set_ylabel("cosine silhouette")
    ax.set_ylim(0.25, 0.65)
    ax.set_title(
        "Cluster count selected without lineage labels", loc="left", fontsize=14, pad=11
    )
    clean_axis(ax)
    finish(fig, "cluster_selection.png")


def cluster_sizes(assignments: pd.DataFrame) -> None:
    counts = assignments["predicted_cluster"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9.5, 4.7))
    x = np.arange(len(counts))
    bars = ax.bar(x, counts.values, color=BOOKING, width=0.68)
    bars[0].set_color(DARTMOUTH)
    ax.set_xticks(x, [str(value) for value in counts.index], fontsize=8)
    ax.set_xlabel("cluster, ordered by size")
    ax.set_ylabel("models")
    ax.set_title(
        f"The {len(counts)} discovered groups are not equally sized",
        loc="left",
        fontsize=14,
    )
    clean_axis(ax)
    finish(fig, "cluster_sizes.png")


def validation_agreement(metrics: pd.DataFrame) -> None:
    order = ["manual_confirmed", "metadata_root_consensus", "metadata_graph_root"]
    data = metrics.set_index("label_source").loc[order]
    labels = [
        f"Human-reviewed\n{int(data.loc[order[0], 'n_families']):,} families",
        f"Metadata agreement\n{int(data.loc[order[1], 'n_families']):,} families",
        f"Graph roots\n{int(data.loc[order[2], 'n_families']):,} root IDs",
    ]
    measures = [
        ("adjusted_rand", "ARI", BOOKING),
        ("normalized_mutual_information", "Shared information", DARTMOUTH),
        ("cluster_purity", "Purity", GOLD),
    ]
    x = np.arange(len(data))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    for offset, (column, label, color) in zip((-width, 0, width), measures):
        bars = ax.bar(x + offset, data[column], width=width, label=label, color=color)
        for bar, value in zip(bars, data[column]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                f"{value:.2f}",
                ha="center",
                fontsize=8,
                color=INK,
            )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("agreement with labels opened after prediction")
    ax.set_title("Broad families agree more than individual root IDs", loc="left", fontsize=14)
    ax.legend(frameon=False, ncols=3, loc="upper center")
    clean_axis(ax)
    finish(fig, "validation_agreement.png")


def gallery_accuracy(metrics: pd.DataFrame) -> None:
    value = metrics.iloc[0]
    labels = [
        "Top-1 across all models",
        "Top-1, equal weight per base",
        "Top-3 across all models",
        "Top-3, equal weight per base",
    ]
    values = [
        value["micro_top1"],
        value["root_macro_top1"],
        value["micro_top3"],
        value["root_macro_top3"],
    ]
    colors = [BOOKING, BERKELEY, DARTMOUTH, GOLD]
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.56)
    for bar, score in zip(bars, values[::-1]):
        ax.text(
            score + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1%}",
            va="center",
            color=INK,
            weight="bold",
        )
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_xlabel("accuracy")
    ax.set_title("A known candidate base is usually in the top three", loc="left", fontsize=14)
    clean_axis(ax, grid="x")
    finish(fig, "gallery_accuracy.png")


def performance_summary(
    interpolation: pd.DataFrame,
    cluster_holdout: pd.DataFrame,
    leading_method: str,
) -> dict[str, dict[str, dict[str, object]]]:
    generations = ["modern", "legacy"]
    summary: dict[str, dict[str, dict[str, object]]] = {
        "interpolation": {},
        "cluster_holdout": {},
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), sharey=False)
    for ax, generation, accent in zip(axes, generations, [BOOKING, DARTMOUTH]):
        knn = row(interpolation, generation, leading_method)
        baseline = row(interpolation, generation, "training_mean")
        summary["interpolation"][generation] = {"knn": knn, "baseline": baseline}
        bars = ax.bar(
            ["3 closest\nspectral models", "Training-set\naverage"],
            [knn["model_weighted_mae"], baseline["model_weighted_mae"]],
            color=[accent, GRID],
            width=0.58,
        )
        for bar, value in zip(
            bars, [knn["model_weighted_mae"], baseline["model_weighted_mae"]]
        ):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.22,
                f"{value:.2f}",
                ha="center",
                color=INK,
                weight="bold",
            )
        ax.text(
            0,
            knn["model_weighted_mae"] * 0.56,
            f"Spearman\n{knn['spearman']:.3f}",
            ha="center",
            va="center",
            color="white",
            weight="bold",
            fontsize=9,
        )
        ax.set_title(
            f"{generation.title()} · {int(knn['n_models']):,} models",
            loc="left",
            fontsize=13,
        )
        ax.set_ylabel("mean absolute error")
        ax.set_ylim(0, max(baseline["model_weighted_mae"] * 1.16, 10.5))
        clean_axis(ax)
    fig.suptitle(
        "Each model is tested using the other four data parts",
        x=0.06,
        y=1.02,
        ha="left",
        fontsize=14,
        weight="bold",
        color=INK,
    )
    finish(fig, "performance_interpolation.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7))
    for ax, generation, accent in zip(axes, generations, [BOOKING, DARTMOUTH]):
        interp_knn = row(interpolation, generation, leading_method)
        hard_knn = row(cluster_holdout, generation, leading_method)
        hard_base = row(cluster_holdout, generation, "training_mean")
        summary["cluster_holdout"][generation] = {
            "knn": hard_knn,
            "baseline": hard_base,
        }
        labels = ["Ordinary\nfive-part test", "Whole group\nheld out", "Training-set\naverage"]
        values = [
            interp_knn["model_weighted_mae"],
            hard_knn["model_weighted_mae"],
            hard_base["model_weighted_mae"],
        ]
        bars = ax.bar(labels, values, color=[accent, GOLD, GRID], width=0.62)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(values) * 0.025,
                f"{value:.2f}",
                ha="center",
                color=INK,
                weight="bold",
                fontsize=9,
            )
        ax.text(
            1,
            hard_knn["model_weighted_mae"] * 0.55,
            f"rho={hard_knn['spearman']:.2f}",
            ha="center",
            va="center",
            color=INK,
            weight="bold",
            fontsize=9,
        )
        ax.set_title(generation.title(), loc="left", fontsize=13)
        ax.set_ylabel("mean absolute error")
        ax.set_ylim(0, max(values) * 1.16)
        clean_axis(ax)
    fig.suptitle(
        "Prediction is harder when a complete weight-derived group is absent",
        x=0.06,
        y=1.02,
        ha="left",
        fontsize=14,
        weight="bold",
        color=INK,
    )
    finish(fig, "performance_stress.png")
    return summary


def prediction_scatter(
    interpolation: pd.DataFrame,
    cluster_holdout: pd.DataFrame,
    leading_method: str,
) -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(10.8, 4.7), sharex=True, sharey=True, gridspec_kw={"wspace": 0.16}
    )
    panels = [
        (interpolation, "Ordinary five-part test", BOOKING),
        (cluster_holdout, "Whole weight group held out", GOLD),
    ]
    for ax, (frame, title, color) in zip(axes, panels):
        data = frame[
            (frame["generation"] == "modern")
            & (frame["target"] == "modern_average")
            & (frame["feature_block"] == leading_method)
        ]
        lo = float(min(data["actual"].min(), data["prediction"].min()))
        hi = float(max(data["actual"].max(), data["prediction"].max()))
        ax.scatter(
            data["actual"],
            data["prediction"],
            s=10,
            alpha=0.34,
            color=color,
            edgecolors="none",
            rasterized=True,
        )
        ax.plot([lo, hi], [lo, hi], color=DARTMOUTH, linewidth=1.5, linestyle=(0, (4, 3)))
        ax.set_title(title, loc="left", fontsize=13)
        ax.set_xlabel("observed modern average")
        clean_axis(ax, grid="both")
    axes[0].set_ylabel("predicted modern average")
    fig.suptitle(
        "Modern predictions: ordinary test versus a held-out weight group",
        x=0.06,
        y=1.02,
        ha="left",
        fontsize=14,
        weight="bold",
        color=INK,
    )
    finish(fig, "modern_prediction_scatter.png")


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def reassessment_charts(parent: Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    override = os.environ.get("HTSR_REASSESSMENT_RUN")
    candidates = [Path(override)] if override else sorted(
        (ROOT/"research_runs/htsr_reassessment/runs").glob("reassessment_*"),
        key=lambda p: p.stat().st_mtime_ns, reverse=True)
    required = ("metadata.json", "performance_metrics.csv", "performance_task_metrics.csv",
                "lineage_gallery_metrics.csv", "lineage_open_set_metrics.csv", "all_matrix_cluster_validation.csv")
    for candidate in candidates:
        try:
            validate_artifact_manifest(candidate, required_files=required, expected_run_id=candidate.name)
            metadata = json.loads((candidate/"metadata.json").read_text())
            if Path(metadata["parent_run"]).resolve() == parent.resolve():
                run = candidate
                break
        except (OSError, RuntimeError, ValueError):
            continue
    else:
        raise FileNotFoundError("No complete reassessment with gallery and task checks matches the parent")
    names = {"performance": "performance_metrics.csv", "tasks": "performance_task_metrics.csv",
             "lineage_gallery": "lineage_gallery_metrics.csv", "open_set": "lineage_open_set_metrics.csv",
             "all_matrix_clusters": "all_matrix_cluster_validation.csv"}
    tables = {name: pd.read_csv(run/file) for name, file in names.items()}
    audit_files = {"range_checks": "prediction_metrics_audit.csv",
                   "collection_directions": "collection_direction_metrics.csv",
                   "identity_summary": "identity_summary.csv", "gallery_ties": "gallery_tie_sensitivity.csv"}
    audits = sorted((run/"audit").glob("audit_*"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for audit in audits:
        try:
            validate_artifact_manifest(audit, required_files=("summary.json", *audit_files.values()), expected_run_id=audit.name)
            audit_metadata = json.loads((audit/"summary.json").read_text())
            if Path(audit_metadata["reassessment_run"]).resolve() == run.resolve():
                tables.update({key: pd.read_csv(audit/file) for key, file in audit_files.items()})
                break
        except (OSError, ValueError, RuntimeError):
            continue
    else:
        raise FileNotFoundError("A completed reassessment audit is required for range and identity claims")
    method_labels = {
        "rank_knn3": "Alpha-rank\nneighbors", "matrix_shape_trees": "Matrix-shape\ntrees",
        "nonalpha_spectral_trees": "Spectral trees\nwithout alpha", "role_spectral_trees": "Spectral trees\nwith alpha",
        "nonalpha_spectral_ridge": "Spectral ridge\nwithout alpha", "training_median": "Training\nmedian",
    }
    for name, split, methods, title in (
        ("review_performance.png", "near_group_0995",
         ["rank_knn3", "matrix_shape_trees", "nonalpha_spectral_trees", "role_spectral_trees"],
         "Near-matches kept together: adding alpha does not improve these trees"),
        ("review_stress.png", "whole_weight_cluster",
         ["rank_knn3", "nonalpha_spectral_trees", "nonalpha_spectral_ridge", "training_median"],
         "An unseen weight group raises errors; linear spectral information still helps"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
        for ax, generation in zip(axes, ("modern", "legacy")):
            data = tables["performance"].query("population == 'matched_role_population' and generation == @generation and split == @split").set_index("method").loc[methods]
            values = data.mae.to_numpy()
            bars = ax.bar(range(len(methods)), values, color=[BOOKING, GRID, DARTMOUTH, GOLD], width=0.65)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x()+bar.get_width()/2, value+max(values)*0.025, f"{value:.2f}", ha="center", color=INK, weight="bold", fontsize=10)
            ax.set_xticks(range(len(methods)), [method_labels[m] for m in methods], fontsize=9)
            ax.set_ylim(0, max(values)*1.2)
            ax.set_ylabel("Mean absolute score error")
            ax.set_title(f"{generation.title()} | {int(data.n_models.iloc[0]):,} models", loc="left", fontsize=12)
            clean_axis(ax)
        fig.suptitle(title, fontsize=12, x=0.02, y=1.03, ha="left", color=INK, weight="bold")
        finish(fig, name)

    gallery = tables["lineage_gallery"].query("cohort == 'common_queries_and_gallery'").set_index("method")
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(3)
    for offset, method, label, color in (
        (-0.18, "parent_rank32_dks", "DKS-alpha ranks", BOOKING),
        (0.18, "trace_sqrt_norm_pearson", "Trace: layer-wise norms", DARTMOUTH),
    ):
        values = gallery.loc[method, ["micro_top1", "root_macro_top1", "micro_top3"]].to_numpy(float)
        bars = ax.bar(x+offset, values, width=0.34, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value+0.015, f"{value:.2%}", ha="center", fontsize=10, weight="bold", color=INK)
    ax.set_xticks(x, ["First choice correct", "First choice: equal\nweight per base", "Root in first three"])
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(frameon=False, loc="upper center", ncols=2)
    ax.set_title(f"Same {int(gallery.n_present.iloc[0]):,} scored queries and {int(gallery.candidate_count.iloc[0]):,} candidate bases", loc="left", fontsize=13)
    clean_axis(ax)
    finish(fig, "review_gallery.png")

    tasks = tables["tasks"].query("generation == 'modern' and split == 'near_group_0995' and method == 'role_spectral_trees'").sort_values("mae")
    fig, ax = plt.subplots(figsize=(8, 4.7))
    bars = ax.barh(tasks.target, tasks.mae, color=[BOOKING]*(len(tasks)-1)+[GOLD], height=0.6)
    for bar, value in zip(bars, tasks.mae):
        ax.text(value+0.15, bar.get_y()+bar.get_height()/2, f"{value:.2f}", va="center", color=INK, weight="bold")
    ax.set_xlim(0, tasks.mae.max()*1.2)
    ax.set_xlabel("Mean absolute error: individual task, 0-100 score")
    ax.set_title("Combined spectral trees: modern tasks differ in difficulty", loc="left", fontsize=12)
    clean_axis(ax, grid="x")
    finish(fig, "review_tasks.png")
    return {"run_id": run.name, "counts": metadata["counts"], **{
        name: table.to_dict("records") for name, table in tables.items()}}, tables


def main() -> None:
    setup()
    run = select_run()
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    ledger = pd.read_csv(run / "population_ledger.csv", low_memory=False)
    assignments = pd.read_csv(run / "lineage_cluster_assignments.csv")
    selection = pd.read_csv(run / "lineage_cluster_selection.csv")
    internal = pd.read_csv(run / "lineage_internal_metrics.csv")
    validation = pd.read_csv(run / "lineage_validation_metrics.csv")
    gallery = pd.read_csv(run / "lineage_gallery_metrics.csv")
    performance = pd.read_csv(run / "performance_metrics.csv")
    performance_predictions = pd.read_csv(run / "performance_predictions.csv")
    hard_performance = pd.read_csv(run / "performance_cluster_holdout_metrics.csv")
    hard_predictions = pd.read_csv(run / "performance_cluster_holdout_predictions.csv")
    leading_method = str(metadata.get("performance_method", "alpha_profile_knn3"))
    masking = pd.read_csv(run / "role_masking_patterns.csv")
    coverage_selection = pd.read_csv(run / "role_coverage_selection.csv")

    phase2, ablation, ablation_summary, graph_path = load_study_collections(metadata)
    collection_sources(phase2, ablation)
    model_type_distribution(phase2, ablation)
    phase_architecture, ablation_architecture = architecture_distribution(
        phase2, ablation, graph_path
    )
    collections = collection_summary(
        phase2,
        ablation,
        ablation_summary,
        phase_architecture,
        ablation_architecture,
    )

    profile_sources = corpus_coverage(metadata, ledger)
    cluster_selection(selection)
    cluster_sizes(assignments)
    validation_agreement(validation)
    gallery_accuracy(gallery)
    performance_rows = performance_summary(
        performance, hard_performance, leading_method
    )
    prediction_scatter(
        performance_predictions, hard_predictions, leading_method
    )
    reassessment, reassessment_tables = reassessment_charts(run)

    summary = json_safe(
        {
            "run_id": run.name,
            "counts": metadata["counts"],
            "profile_sources": profile_sources,
            "method_boundary": metadata["method_boundary"],
            "internal": internal.iloc[0].to_dict(),
            "cluster_selection": selection.to_dict(orient="records"),
            "validation": validation.to_dict(orient="records"),
            "gallery": gallery.iloc[0].to_dict(),
            "performance": performance_rows,
            "leading_performance_method": leading_method,
            "role_masking": masking.to_dict(orient="records"),
            "role_coverage_selection": coverage_selection.to_dict(orient="records"),
            "collections": collections,
            "reassessment": reassessment,
        }
    )
    (OUT / "chart_data.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    workbook_written = False
    try:
        with pd.ExcelWriter(HERE / "htsr_study_data.xlsx") as writer:
            internal.to_excel(writer, sheet_name="lineage_internal", index=False)
            selection.to_excel(writer, sheet_name="cluster_selection", index=False)
            validation.to_excel(
                writer, sheet_name="postfreeze_validation", index=False
            )
            gallery.to_excel(writer, sheet_name="gallery_routing", index=False)
            average_rows(performance).to_excel(
                writer, sheet_name="performance_5fold", index=False
            )
            average_rows(hard_performance).to_excel(
                writer, sheet_name="performance_cluster_holdout", index=False
            )
            pd.DataFrame(collections).T.to_excel(
                writer, sheet_name="collection_summary"
            )
            for name, table in reassessment_tables.items():
                table.to_excel(writer, sheet_name=("review_"+name)[:31], index=False)
        workbook_written = True
    except ImportError:
        (HERE / "htsr_study_data.xlsx").unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "run": str(run),
                "assets": sorted(path.name for path in OUT.glob("*.png")),
                "workbook_written": workbook_written,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
