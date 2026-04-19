from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd


KNOWN_QUANT_METHODS = {"awq", "gptq", "bitsandbytes"}


def parse_repo_and_revision(model_id: object) -> tuple[str | None, str | None]:
    if pd.isna(model_id):
        return None, None
    value = str(model_id).strip()
    if not value:
        return None, None
    if "@" in value and not value.startswith("@"):
        repo_id, revision = value.split("@", 1)
        return normalize_repo_id(repo_id), revision.strip() or None
    return normalize_repo_id(value), None


def normalize_repo_id(raw_value: object) -> str | None:
    if pd.isna(raw_value):
        return None
    value = str(raw_value).strip()
    value = re.sub(r"^https?://(?:www\.)?huggingface\.co/", "", value, flags=re.IGNORECASE)
    value = value.split("?", 1)[0].split("#", 1)[0].strip().strip("/")
    if not value:
        return None
    if "/" not in value and "--" in value:
        owner, name = value.split("--", 1)
        if owner and name:
            value = f"{owner}/{name}"
    if "/" not in value:
        return None
    return value


def canonical_model_key(repo_id: str | None, revision: str | None) -> str | None:
    if not repo_id:
        return None
    return f"{repo_id}@{revision}" if revision else repo_id


def _completeness_score(frame: pd.DataFrame) -> pd.Series:
    architecture = (
        frame.get("Architecture", pd.Series(index=frame.index, dtype="object"))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    architecture_known = architecture.ne("") & architecture.str.lower().ne("unknown")
    params_known = pd.to_numeric(frame.get("#Params (B)"), errors="coerce").fillna(0).gt(0)
    type_value = (
        frame.get("Type", pd.Series(index=frame.index, dtype="object"))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    type_known = type_value.ne("") & type_value.str.lower().ne("other")
    average_known = pd.to_numeric(frame.get("Average ⬆️"), errors="coerce").notna()
    return (
        architecture_known.astype(int)
        + params_known.astype(int)
        + type_known.astype(int)
        + average_known.astype(int)
    )


def collapse_leaderboard_duplicates(leaderboard: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = leaderboard.copy()
    repo_revision = raw["model_id"].map(parse_repo_and_revision)
    raw["repo_id_norm"] = [item[0] for item in repo_revision]
    raw["revision_norm"] = [item[1] for item in repo_revision]
    raw["canonical_model_key"] = [
        canonical_model_key(repo_id, revision)
        for repo_id, revision in zip(raw["repo_id_norm"], raw["revision_norm"], strict=False)
    ]
    raw["completeness_score"] = _completeness_score(raw)
    raw["average_numeric"] = pd.to_numeric(raw.get("Average ⬆️"), errors="coerce").fillna(-1.0)
    canonical = (
        raw.sort_values(
            ["canonical_model_key", "completeness_score", "average_numeric"],
            ascending=[True, False, False],
        )
        .drop_duplicates(subset=["canonical_model_key"], keep="first")
        .reset_index(drop=True)
    )
    return canonical, raw


def seed_analysis_status(frame: pd.DataFrame, svd_dir: Path) -> pd.DataFrame:
    result = frame.copy()
    existing = {path.stem for path in svd_dir.glob("*.csv")}
    safe_names = (
        result["model_id"]
        .astype(str)
        .str.replace("/", "--", regex=False)
        .str.replace("@", "__", regex=False)
    )
    result["analysis_artifact_present"] = safe_names.isin(existing)
    result["analysis_artifact_path"] = [
        str(svd_dir / f"{safe_name}.csv") if present else None
        for safe_name, present in zip(safe_names, result["analysis_artifact_present"], strict=False)
    ]
    return result


def classify_primary_type(row: pd.Series) -> str:
    relation = str(row.get("base_model_relation", "")).strip().lower()
    if relation == "source":
        return "base_source"
    if relation in {"adapter", "lora", "peft"}:
        return "adapter"
    if relation == "merge":
        return "merge"
    if relation == "quantized":
        return "quantized"
    if relation == "finetune":
        return "finetune"
    if row.get("pipeline_tag") == "image-text-to-text":
        return "multimodal"
    return "unknown"


def classify_loader_scenario_from_row(row: pd.Series) -> str:
    files = {str(item).lower() for item in row.get("files", [])}
    architecture = str(row.get("Architecture", "")).lower()
    relation = str(row.get("base_model_relation", "")).lower()
    if "adapter_config.json" in files or relation in {"adapter", "lora", "peft"}:
        return "adapter_requires_base"
    if any(name.endswith(".gguf") for name in files):
        return "quantized_alt_format"
    if row.get("pipeline_tag") == "image-text-to-text" or "conditionalgeneration" in architecture:
        return "multimodal_transformers"
    if any(method in str(row.get("quant_method", "")).lower() for method in KNOWN_QUANT_METHODS):
        return "quantized_transformers_native"
    if relation == "quantized":
        return "quantized_transformers_native"
    return "standard_transformers"


def load_atlas_candidates(atlas_json_path: Path) -> pd.DataFrame:
    data = json.loads(atlas_json_path.read_text())
    rows = []
    for node in data.get("nodes", []):
        attrs = node.get("attributes", {})
        rows.append(
            {
                "model_id": node["id"],
                "source_model": attrs.get("base_model_clean"),
                "base_model_relation": attrs.get("base_model_relation", "unknown"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "model_id",
                "source_model",
                "base_model_relation",
                "repo_id_norm",
                "revision_norm",
                "canonical_model_key",
            ]
        )
    repo_revision = frame["model_id"].map(parse_repo_and_revision)
    frame["repo_id_norm"] = [item[0] for item in repo_revision]
    frame["revision_norm"] = [item[1] for item in repo_revision]
    frame["canonical_model_key"] = [
        canonical_model_key(repo_id, revision)
        for repo_id, revision in zip(frame["repo_id_norm"], frame["revision_norm"], strict=False)
    ]
    return frame


def add_lineage_flags(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    source = result.get("source_model", pd.Series(index=result.index, dtype="object")).fillna("")
    result["known_lineage_flag"] = source.ne("") & source.ne("unknown") & source.ne("N/A")
    return result


def apply_type_taxonomy(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["primary_type_bucket"] = result.apply(classify_primary_type, axis=1)
    result["loader_scenario"] = result.apply(classify_loader_scenario_from_row, axis=1)
    result["is_adapter"] = result["primary_type_bucket"].eq("adapter")
    result["is_merge"] = result["primary_type_bucket"].eq("merge")
    result["is_quantized"] = result["primary_type_bucket"].eq("quantized")
    result["is_multimodal"] = result["loader_scenario"].eq("multimodal_transformers")
    return result


def select_to_target(frame: pd.DataFrame, target_size: int) -> pd.DataFrame:
    result = frame.copy()
    result["selection_priority"] = (
        result["analysis_artifact_present"].fillna(False).astype(int) * 4
        + result["known_lineage_flag"].fillna(False).astype(int) * 2
        + result["primary_type_bucket"].isin({"adapter", "quantized", "merge", "multimodal"}).astype(int)
    )
    ordered = (
        result.sort_values(
            ["primary_type_bucket", "selection_priority", "canonical_model_key"],
            ascending=[True, False, True],
        )
        .drop_duplicates(subset=["canonical_model_key"], keep="first")
        .reset_index(drop=True)
    )

    buckets = {
        bucket: bucket_frame.reset_index(drop=True)
        for bucket, bucket_frame in ordered.groupby("primary_type_bucket", sort=True)
    }
    selected_rows = []
    round_index = 0
    while len(selected_rows) < target_size:
        progressed = False
        for bucket in sorted(buckets):
            bucket_frame = buckets[bucket]
            if round_index < len(bucket_frame):
                selected_rows.append(bucket_frame.iloc[round_index].to_dict())
                progressed = True
                if len(selected_rows) >= target_size:
                    break
        if not progressed:
            break
        round_index += 1

    return pd.DataFrame(selected_rows).reset_index(drop=True)
