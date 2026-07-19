from __future__ import annotations

import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ZOO_ENV = "MODEL_ZOO_ROOT"
STUDY_RUN_ENV = "HTSR_STUDY_RUN"
SCALEUP_RUN_ENV = "HTSR_SCALEUP_RUN"
STUDY_RUN_MARKERS = (
    "manifest.csv",
    "feature_table.csv",
    "cluster_assignments.csv",
    "lineage_predictions.csv",
    "lineage_representative_predictions.csv",
    "lineage_metrics.json",
    "performance_predictions.csv",
    "performance_metrics.json",
    "exclusions.csv",
    "provenance.json",
    "report.md",
    "artifact_manifest.json",
)
SCALEUP_RUN_MARKERS = (
    "metadata.json",
    "population_ledger.csv",
    "modern_pooled_predictions.csv",
    "modern_pooled_metrics.csv",
    "modern_pooled_increments.csv",
    "legacy_paired_predictions.csv",
    "legacy_paired_metrics.csv",
    "legacy_increments.csv",
    "lineage_broad_cluster_metrics.csv",
    "report.md",
)


def _latest_complete_run(root: Path, markers: tuple[str, ...]) -> Path | None:
    if not root.is_dir():
        return None
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and all((path / marker).is_file() for marker in markers)
    ]
    if not candidates:
        return None

    def state(path: Path) -> tuple[int, str]:
        newest_marker = max((path / marker).stat().st_mtime_ns for marker in markers)
        return newest_marker, path.name

    return max(candidates, key=state)


def _configured_or_latest(
    environment_variable: str,
    relative_root: Path,
    markers: tuple[str, ...],
) -> Path | None:
    configured = os.environ.get(environment_variable, "").strip()
    if configured:
        return Path(configured).expanduser()
    return _latest_complete_run(PROJECT_ROOT / relative_root, markers)


def default_study_run() -> Path | None:
    return _configured_or_latest(
        STUDY_RUN_ENV,
        Path("research_runs/htsr_study/runs"),
        STUDY_RUN_MARKERS,
    )


def default_scaleup_run() -> Path | None:
    return _configured_or_latest(
        SCALEUP_RUN_ENV,
        Path("research_runs/htsr_scaleup/runs"),
        SCALEUP_RUN_MARKERS,
    )


def _root_from_dashboard_cache_manifest() -> Path | None:
    manifest_path = PROJECT_ROOT / "research_runs" / "preliminary" / "results" / "dashboard_cache" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    metrics_dir = Path(str(manifest.get("fingerprint", {}).get("metrics_dir", "")))
    suffixes = (Path("output/results/metrics"), Path("output/ablation/metrics"))
    for suffix in suffixes:
        if metrics_dir.parts[-len(suffix.parts) :] == suffix.parts and len(metrics_dir.parents) >= len(suffix.parts):
            return metrics_dir.parents[len(suffix.parts) - 1]
    return None


def model_zoo_roots() -> list[Path]:
    roots: list[Path] = []
    env_value = os.environ.get(MODEL_ZOO_ENV)
    if env_value:
        roots.append(Path(env_value).expanduser())
    roots.append(PROJECT_ROOT / "Model-Zoo")
    cached_root = _root_from_dashboard_cache_manifest()
    if cached_root is not None:
        roots.append(cached_root)

    unique_roots: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            unique_roots.append(root)
            seen.add(key)
    return unique_roots


def resolve_model_zoo_artifact(relative_path: str | Path) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        return relative
    roots = model_zoo_roots()
    for root in roots:
        candidate = root / relative
        if candidate.exists():
            return candidate
    return roots[0] / relative
