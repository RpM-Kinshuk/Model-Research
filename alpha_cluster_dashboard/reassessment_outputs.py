from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from alpha_cluster_dashboard.paths import PROJECT_ROOT
from spectral_research.study.artifacts import validate_artifact_manifest


SELECTION_COLUMNS = ("population", "generation", "split", "method")
METRIC_COLUMNS = ("n_models", "n_eligible", "coverage", "mae", "spearman", "rmse")
REQUIRED_REASSESSMENT_FILES = (
    "metadata.json",
    "performance_metrics.csv",
    "performance_predictions.csv",
    "artifact_manifest.json",
)
LINEAGE_FILES = ("lineage_gallery_metrics.csv", "lineage_open_set_metrics.csv")


@dataclass(frozen=True)
class ReassessmentOutputs:
    run_dir: Path
    parent_run: Path
    metadata: Mapping[str, Any]
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    lineage_gallery: pd.DataFrame = field(default_factory=pd.DataFrame)
    lineage_open_set: pd.DataFrame = field(default_factory=pd.DataFrame)


def _read_metadata(root: Path, parent: Path) -> dict[str, Any]:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("run_id") != root.name:
        raise ValueError("Reassessment metadata run_id does not match its directory")
    recorded_parent = metadata.get("parent_run")
    if not isinstance(recorded_parent, str) or Path(recorded_parent).resolve() != parent:
        raise ValueError("Reassessment belongs to a different parent run")
    sources = metadata.get("sources", {})
    parent_digest = hashlib.sha256((parent / "artifact_manifest.json").read_bytes()).hexdigest()
    if not isinstance(sources, dict) or sources.get("parent_manifest") != parent_digest:
        raise ValueError("Reassessment parent manifest hash does not match the loaded study")
    return metadata


def reassessment_runs(parent_run: str | Path, *, root: Path | None = None) -> list[Path]:
    """Find completed candidates for this parent; the loader verifies their tables."""
    parent = Path(parent_run).expanduser().resolve()
    root = root if root is not None else PROJECT_ROOT / "research_runs/htsr_reassessment/runs"
    candidates = []
    for path in sorted(root.glob("reassessment_*")):
        if not all((path / name).is_file() for name in REQUIRED_REASSESSMENT_FILES):
            continue
        try:
            validate_artifact_manifest(path, required_files=("metadata.json",), expected_run_id=path.name)
            _read_metadata(path, parent)
        except (OSError, ValueError, RuntimeError):
            continue
        candidates.append(path)
    return sorted(
        candidates,
        key=lambda path: ((path / "artifact_manifest.json").stat().st_mtime_ns, path.name),
        reverse=True,
    )


def _validate_keys(frame: pd.DataFrame, keys: tuple[str, ...], name: str) -> None:
    for column in keys:
        values = frame[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise ValueError(f"{name} contains missing {column} keys")
        frame[column] = values
    if frame.duplicated(list(keys)).any():
        raise ValueError(f"{name} contains duplicate selection keys")
    if "generation" in keys and not frame["generation"].isin(("modern", "legacy")).all():
        raise ValueError(f"{name} contains an unsupported benchmark generation")


def _load_lineage_tables(root: Path, manifest: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not any(name in manifest["artifacts"] or (root / name).exists() for name in LINEAGE_FILES):
        return pd.DataFrame(), pd.DataFrame()
    validate_artifact_manifest(root, required_files=LINEAGE_FILES, expected_run_id=root.name)
    tables = []
    for name, keys, counts, rates in (
        (LINEAGE_FILES[0], ("cohort", "method"),
         ("candidate_count", "n_queries", "n_labelled", "n_present", "n_present_roots", "n_absent_operator", "n_supplied_unprofiled"),
         ("micro_top1", "micro_top3", "root_macro_top1", "root_macro_top3")),
        (LINEAGE_FILES[1], ("cohort", "method", "contrast", "signal"),
         ("n_positive", "n_negative", "positive_roots", "negative_roots"),
         ("auroc", "root_balanced_auroc")),
    ):
        frame = pd.read_csv(root / name)
        missing = sorted(set((*keys, *counts, *rates)).difference(frame.columns))
        if missing or frame.empty:
            raise ValueError(f"{name} is empty or missing columns: {missing}")
        _validate_keys(frame, keys, name)
        for column in (*counts, *rates):
            values = pd.to_numeric(frame[column], errors="raise")
            if not np.isfinite(values).all() or values.lt(0).any():
                raise ValueError(f"Invalid {name} {column}")
            if column in counts:
                if values.mod(1).ne(0).any():
                    raise ValueError(f"Invalid {name} integer count {column}")
                values = values.astype(int)
            elif values.gt(1).any():
                raise ValueError(f"Invalid {name} probability metric {column}")
            frame[column] = values
        tables.append(frame)
    if not tables[1]["signal"].isin(("score", "margin")).all():
        raise ValueError("Unsupported lineage open-set signal")
    return tables[0], tables[1]


def load_reassessment_outputs(run_dir: str | Path, *, parent_run: str | Path) -> ReassessmentOutputs:
    root = Path(run_dir).expanduser().resolve()
    parent = Path(parent_run).expanduser().resolve()
    manifest = validate_artifact_manifest(
        root,
        required_files=REQUIRED_REASSESSMENT_FILES[:-1],
        expected_run_id=root.name,
    )
    validate_artifact_manifest(parent, required_files=("metadata.json",), expected_run_id=parent.name)
    metadata = _read_metadata(root, parent)
    metrics = pd.read_csv(root / "performance_metrics.csv", low_memory=False)
    predictions = pd.read_csv(root / "performance_predictions.csv", low_memory=False)
    for name, frame, required in (
        ("performance_metrics.csv", metrics, (*SELECTION_COLUMNS, *METRIC_COLUMNS)),
        ("performance_predictions.csv", predictions, (*SELECTION_COLUMNS, "model_id", "actual", "prediction")),
    ):
        missing = sorted(set(required).difference(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")
    if metrics.empty:
        raise ValueError("Reassessment contains no aggregate metrics")
    _validate_keys(metrics, SELECTION_COLUMNS, "performance_metrics.csv")
    _validate_keys(predictions, (*SELECTION_COLUMNS, "model_id"), "performance_predictions.csv")

    for column in METRIC_COLUMNS:
        metrics[column] = pd.to_numeric(metrics[column], errors="raise")
    for column in ("n_models", "n_eligible", "coverage"):
        if not np.isfinite(metrics[column]).all() or metrics[column].lt(0).any():
            raise ValueError(f"Invalid reassessment {column}")
    for column in ("n_models", "n_eligible"):
        if metrics[column].mod(1).ne(0).any():
            raise ValueError(f"Reassessment {column} must be an integer")
        metrics[column] = metrics[column].astype(int)
    for column in ("mae", "spearman", "rmse"):
        if np.isinf(metrics[column]).any():
            raise ValueError(f"Reassessment {column} contains infinity")
    for column in ("mae", "rmse"):
        if (metrics[column].lt(0) | (metrics[column].isna() & metrics["n_models"].gt(0))).any():
            raise ValueError(f"Invalid reassessment {column}")
    if metrics["spearman"].dropna().abs().gt(1 + 1e-9).any():
        raise ValueError("Invalid reassessment Spearman correlation")

    predictions["actual"] = pd.to_numeric(predictions["actual"], errors="raise")
    predictions["prediction"] = pd.to_numeric(predictions["prediction"], errors="raise")
    if not np.isfinite(predictions["actual"]).all() or np.isinf(predictions["prediction"]).any():
        raise ValueError("Reassessment scores must be finite; only abstained predictions may be missing")
    predictions["absolute_error"] = (predictions["actual"] - predictions["prediction"]).abs()

    # Eligible counts may include rows omitted by an unevaluable fold. Do not
    # replace the recorded denominator with the number of emitted predictions.
    counts = predictions.groupby(list(SELECTION_COLUMNS))["prediction"].agg(
        size="size", count="count",
        n_outside_0_100=lambda values: (values.lt(0) | values.gt(100)).sum(),
    )
    metric_index = pd.MultiIndex.from_frame(metrics.loc[:, list(SELECTION_COLUMNS)])
    if not counts.index.difference(metric_index).empty:
        raise ValueError("Reassessment predictions contain unreported metric selections")
    counts = counts.reindex(metric_index, fill_value=0)
    if not np.array_equal(metrics["n_models"], counts["count"]):
        raise ValueError("Reassessment n_models disagrees with finite saved predictions")
    if (metrics["n_eligible"].to_numpy() < counts["size"].to_numpy()).any():
        raise ValueError("Reassessment n_eligible is smaller than its saved prediction population")
    coverage = np.divide(
        metrics["n_models"], metrics["n_eligible"],
        out=np.zeros(len(metrics)), where=metrics["n_eligible"].to_numpy() > 0,
    )
    if not np.allclose(metrics["coverage"], coverage, rtol=1e-6, atol=1e-9):
        raise ValueError("Reassessment coverage disagrees with its recorded counts")
    metrics["n_outside_0_100"] = counts["n_outside_0_100"].to_numpy(dtype=int)
    gallery, open_set = _load_lineage_tables(root, manifest)
    return ReassessmentOutputs(root, parent, metadata, metrics, predictions, gallery, open_set)
