from __future__ import annotations

import numpy as np
import pandas as pd

SPECIAL_RELATIONS = {"merge", "adapter", "quantized"}
LEADERBOARD_COLUMNS = ["model_id", "Architecture", "#Params (B)", "Generation", "Type"]
OVERRIDE_COLUMNS = ["model_id", "audited_base_family", "label_status", "bucket_override"]
METADATA_COLUMNS = ["model_id", "source_model", "base_model_relation"]
LOW_INFORMATION_TEXT = {"unknown", "n/a", "na", "none"}


def _normalize_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _candidate_family(model_id: object, source_model: object, relation: object) -> str | None:
    source = _normalize_text(source_model)
    relation_text = _normalize_text(relation)
    model_id_text = _normalize_text(model_id)
    if source is not None and source.lower() != "unknown":
        return source
    if relation_text is not None and relation_text.lower() == "source":
        return model_id_text
    return None


def _merged_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    subset = frame.loc[:, available].copy()
    if subset.empty or "model_id" not in subset.columns:
        return subset
    if not subset["model_id"].duplicated().any():
        return subset.drop_duplicates("model_id")

    rows: list[dict[str, object]] = []
    for model_id, group in subset.groupby("model_id", sort=False):
        row: dict[str, object] = {"model_id": model_id}
        for column in available:
            if column == "model_id":
                continue
            row[column] = _best_group_value(group[column], column)
        rows.append(row)
    return pd.DataFrame(rows, columns=available)


def _best_group_value(series: pd.Series, column: str) -> object:
    non_missing = [value for value in series.tolist() if not pd.isna(value)]
    if not non_missing:
        return pd.NA

    if column in {"#Params (B)", "Generation"}:
        numeric = pd.to_numeric(pd.Series(non_missing), errors="coerce").dropna()
        if not numeric.empty:
            return float(numeric.max()) if column == "#Params (B)" else int(numeric.max())

    candidates: list[tuple[int, int, str, object]] = []
    for value in non_missing:
        text = _normalize_text(value)
        if text is None:
            continue
        lowered = text.lower()
        information = 0 if lowered in LOW_INFORMATION_TEXT else 1
        candidates.append((information, len(text), text, value))
    if not candidates:
        return non_missing[0]
    candidates.sort(reverse=True)
    return candidates[0][3]


def build_audit_manifest(
    records_frame: pd.DataFrame,
    metadata_frame: pd.DataFrame,
    leaderboard_frame: pd.DataFrame,
    overrides_frame: pd.DataFrame,
) -> pd.DataFrame:
    metadata = _merged_columns(metadata_frame, METADATA_COLUMNS)
    leaderboard = _merged_columns(leaderboard_frame, LEADERBOARD_COLUMNS)
    overrides = _merged_columns(overrides_frame, OVERRIDE_COLUMNS)

    manifest = (
        records_frame.merge(metadata, on="model_id", how="left")
        .merge(leaderboard, on="model_id", how="left")
        .merge(overrides, on="model_id", how="left")
    )
    manifest["candidate_family"] = [
        _candidate_family(model_id, source_model, relation)
        for model_id, source_model, relation in zip(
            manifest["model_id"],
            manifest.get("source_model"),
            manifest.get("base_model_relation"),
            strict=False,
        )
    ]
    manifest["audited_base_family"] = manifest["audited_base_family"].combine_first(
        manifest["candidate_family"]
    )
    manifest["label_status"] = manifest["label_status"].map(_normalize_text)
    manifest["bucket_override"] = manifest["bucket_override"].map(_normalize_text)
    manifest["label_status"] = np.where(
        manifest["label_status"].notna(),
        manifest["label_status"],
        np.where(manifest["audited_base_family"].notna(), "provisional", "unknown"),
    )

    relation = (
        manifest["base_model_relation"].map(_normalize_text).fillna("unknown").str.lower()
    )
    common_schema = manifest["common_schema"].fillna(False).astype(bool)
    manifest["query_bucket"] = np.select(
        [
            relation.isin(SPECIAL_RELATIONS),
            manifest["label_status"].eq("unknown"),
            ~common_schema,
        ],
        [
            "unsupported_special",
            "unsupported_unknown",
            "heterogeneous",
        ],
        default="supported_candidate",
    )

    params = pd.to_numeric(manifest["#Params (B)"], errors="coerce")
    manifest["size_bucket"] = pd.cut(
        params,
        bins=[-np.inf, 3.0, 13.0, np.inf],
        labels=["small", "medium", "large"],
    ).astype(object)
    manifest["query_bucket"] = manifest["bucket_override"].combine_first(manifest["query_bucket"])
    return manifest
