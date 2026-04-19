from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_cluster_dashboard.data import AlphaRecord, load_alpha_records, schema_name
from spectral_research.benchmark import build_audit_manifest

PERFORMANCE_TARGETS = ("Average ⬆️", "MMLU-PRO", "MATH Lvl 5")


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _group_key(row: pd.Series) -> str:
    for column in ("audited_base_family", "candidate_family", "model_id"):
        text = _clean_text(row.get(column))
        if text:
            return text
    return "unknown"


def _read_csv(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns or [])
    if columns is None:
        return frame
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, columns].copy()


def _lookup_targets(leaderboard: pd.DataFrame, key: str) -> pd.DataFrame:
    columns = [key, *PERFORMANCE_TARGETS]
    available = [column for column in columns if column in leaderboard.columns]
    if key not in available:
        return pd.DataFrame(columns=PERFORMANCE_TARGETS)

    frame = leaderboard.loc[:, available].copy()
    frame[key] = frame[key].map(_clean_text)
    frame = frame[frame[key].ne("")]
    if frame.empty:
        return pd.DataFrame(columns=PERFORMANCE_TARGETS)

    for column in PERFORMANCE_TARGETS:
        if column not in frame.columns:
            frame[column] = pd.NA

    row_scores = frame.loc[:, list(PERFORMANCE_TARGETS)].notna().sum(axis=1)
    ordered = frame.assign(_row_score=row_scores, _row_order=range(len(frame))).sort_values(
        [key, "_row_score", "_row_order"], ascending=[True, False, True], kind="mergesort"
    )
    best_rows = ordered.drop_duplicates(subset=[key], keep="first").set_index(key)
    return best_rows.loc[:, list(PERFORMANCE_TARGETS)]


def join_prediction_targets(manifest: pd.DataFrame, leaderboard: pd.DataFrame) -> pd.DataFrame:
    joined = manifest.copy()
    display_names = (
        joined["display_name"].map(_clean_text)
        if "display_name" in joined.columns
        else joined["model_id"].map(_clean_text)
    )

    atlas_lookup = _lookup_targets(leaderboard, "atlas_repo_id")
    fullname_lookup = _lookup_targets(leaderboard, "fullname")

    for target in PERFORMANCE_TARGETS:
        if target in atlas_lookup.columns:
            values = display_names.map(atlas_lookup[target])
        else:
            values = pd.Series(index=joined.index, dtype="object")
        if target in fullname_lookup.columns:
            values = values.combine_first(display_names.map(fullname_lookup[target]))
        joined[target] = pd.to_numeric(values, errors="coerce")

    joined["cv_group"] = joined.apply(_group_key, axis=1)
    return joined


def _records_frame(records: list[AlphaRecord]) -> pd.DataFrame:
    columns = [
        "model_id",
        "display_name",
        "path",
        "num_layers",
        "raw_module_count",
        "raw_modules",
        "common_schema",
        "schema_name",
    ]
    frame = pd.DataFrame(
        [
            {
                "model_id": record.model_id,
                "display_name": record.display_name,
                "path": record.path,
                "num_layers": record.num_layers,
                "raw_module_count": record.raw_module_count,
                "raw_modules": ", ".join(record.raw_modules),
                "common_schema": record.common_schema,
                "schema_name": schema_name(record),
            }
            for record in records
        ]
    )
    return frame.reindex(columns=columns)


def build_prediction_dataset(
    metrics_dir: str | Path,
    leaderboard_csv: str | Path,
    lineage_csv: str | Path,
    freeze_csv: str | Path,
) -> tuple[pd.DataFrame, dict[str, AlphaRecord]]:
    records = load_alpha_records(metrics_dir)
    records_frame = _records_frame(records)
    metadata = _read_csv(Path(lineage_csv))
    leaderboard = _read_csv(Path(leaderboard_csv))
    overrides = _read_csv(
        Path(freeze_csv),
        columns=["model_id", "audited_base_family", "label_status", "bucket_override", "audit_notes"],
    )

    manifest = build_audit_manifest(records_frame, metadata, leaderboard, overrides)
    joined = join_prediction_targets(manifest, leaderboard)
    joined = joined.dropna(subset=list(PERFORMANCE_TARGETS)).reset_index(drop=True)
    records_by_model_id = {record.model_id: record for record in records}
    return joined, records_by_model_id
