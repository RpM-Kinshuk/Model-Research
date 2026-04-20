from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ZOO_ROOT = PROJECT_ROOT / "Model-Zoo"


def model_zoo_root() -> Path:
    return MODEL_ZOO_ROOT


def model_zoo_curated_table_path() -> Path:
    return MODEL_ZOO_ROOT / "data" / "curated" / "model_zoo_phase2.csv"


def model_zoo_analysis_run_path(run_name: str) -> Path:
    if not run_name or run_name in {".", ".."} or any(sep in run_name for sep in ("/", "\\")):
        raise ValueError("run_name must be a single path component")
    return MODEL_ZOO_ROOT / "analysis_runs" / "phase2" / run_name
