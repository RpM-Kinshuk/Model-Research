from __future__ import annotations

import numpy as np
import pandas as pd

SPECIAL_RELATIONS = {"merge", "adapter", "quantized"}
LEADERBOARD_COLUMNS = ["model_id", "Architecture", "#Params (B)", "Generation", "Type"]
OVERRIDE_COLUMNS = ["model_id", "audited_base_family", "label_status", "bucket_override"]
METADATA_COLUMNS = ["model_id", "source_model", "base_model_relation"]
LOW_INFORMATION_TEXT = {"unknown", "n/a", "na", "none"}
JOIN_KEY = "_join_model_id"


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


def _normalize_repo_id(value: object) -> str | None:
    text = _normalize_text(value)
    if text is None:
        return None
    return text.replace("/", "--")


def _merged_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    subset = frame.loc[:, available].copy()
    if "model_id" not in subset.columns:
        return subset
    subset[JOIN_KEY] = subset["model_id"].map(_normalize_repo_id)
    output_columns = [JOIN_KEY, *[column for column in available if column != "model_id"]]
    if subset.empty:
        return subset.loc[:, output_columns]
    if not subset[JOIN_KEY].duplicated().any():
        return subset.loc[:, output_columns].drop_duplicates(JOIN_KEY)

    rows: list[dict[str, object]] = []
    for join_model_id, group in subset.groupby(JOIN_KEY, sort=False):
        row: dict[str, object] = {JOIN_KEY: join_model_id}
        for column in available:
            if column == "model_id":
                continue
            row[column] = _best_group_value(group[column], column)
        rows.append(row)
    return pd.DataFrame(rows, columns=output_columns)


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
    records = records_frame.copy()
    records[JOIN_KEY] = records["model_id"].map(_normalize_repo_id)
    metadata = _merged_columns(metadata_frame, METADATA_COLUMNS)
    leaderboard = _merged_columns(leaderboard_frame, LEADERBOARD_COLUMNS)
    overrides = _merged_columns(overrides_frame, OVERRIDE_COLUMNS)

    manifest = (
        records.merge(metadata, on=JOIN_KEY, how="left")
        .merge(leaderboard, on=JOIN_KEY, how="left")
        .merge(overrides, on=JOIN_KEY, how="left")
    )
    identity = (
        manifest["display_name"]
        if "display_name" in manifest.columns
        else manifest["model_id"]
    )
    manifest["candidate_family"] = [
        _candidate_family(model_id, source_model, relation)
        for model_id, source_model, relation in zip(
            identity,
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
    return manifest.drop(columns=[JOIN_KEY])
def _empty_split_frame(manifest: pd.DataFrame, *, include_split_role: bool) -> pd.DataFrame:
    columns = list(manifest.columns)
    if include_split_role and "split_role" not in columns:
        columns.append("split_role")
    return pd.DataFrame(columns=columns)


def build_split_artifacts(
    manifest: pd.DataFrame,
    min_family_size: int = 8,
    calibration_fraction: float = 0.25,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    primary = manifest[
        manifest["common_schema"]
        & manifest["label_status"].eq("confirmed")
        & manifest["query_bucket"].eq("supported_candidate")
    ].copy()

    family_sizes = primary["audited_base_family"].value_counts()
    kept_families = family_sizes[family_sizes >= min_family_size].index
    if len(kept_families) < 6:
        kept_families = family_sizes[family_sizes >= 5].index
    kept_families = sorted(kept_families.tolist())
    primary = primary[primary["audited_base_family"].isin(kept_families)].copy()

    calibration_families: set[object] = set()
    if kept_families:
        shuffled = np.random.default_rng(seed).permutation(kept_families).tolist()
        calibration_count = min(
            len(shuffled),
            max(1, int(round(len(shuffled) * calibration_fraction))),
        )
        calibration_families = set(shuffled[:calibration_count])

    gallery_rows: list[pd.DataFrame] = []
    query_rows: list[pd.DataFrame] = []
    for family, group in primary.groupby("audited_base_family", sort=True):
        ordered = group.sort_values("model_id").reset_index(drop=True)
        gallery_n = max(2, len(ordered) - max(1, len(ordered) // 3))

        gallery_rows.append(ordered.iloc[:gallery_n].copy())

        query_part = ordered.iloc[gallery_n:].copy()
        query_part["split_role"] = (
            "calibration_supported" if family in calibration_families else "test_supported"
        )
        query_rows.append(query_part)

    unsupported = manifest[
        manifest["query_bucket"].isin({"unsupported_unknown", "unsupported_special"})
        | (
            manifest["label_status"].eq("confirmed")
            & ~manifest["audited_base_family"].isin(kept_families)
        )
    ].copy()
    unsupported["split_role"] = "unsupported_query"

    heterogeneous = manifest[
        ~manifest["common_schema"] & manifest["label_status"].ne("unknown")
    ].copy()
    heterogeneous["split_role"] = "heterogeneous_query"

    gallery_supported = (
        pd.concat(gallery_rows, ignore_index=True)
        if gallery_rows
        else _empty_split_frame(manifest, include_split_role=False)
    )
    queries_supported = (
        pd.concat(query_rows, ignore_index=True)
        if query_rows
        else _empty_split_frame(manifest, include_split_role=True)
    )

    return {
        "gallery_supported": gallery_supported,
        "queries_supported": queries_supported,
        "queries_unsupported": unsupported.reset_index(drop=True),
        "queries_heterogeneous": heterogeneous.reset_index(drop=True),
    }
