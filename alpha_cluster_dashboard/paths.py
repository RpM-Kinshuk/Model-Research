from __future__ import annotations

import os
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ZOO_ENV = "MODEL_ZOO_ROOT"


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
