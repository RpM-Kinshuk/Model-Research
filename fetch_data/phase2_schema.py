from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd


KNOWN_QUANT_METHODS = {"awq", "gptq", "bitsandbytes"}
FINETUNE_TYPE_HINTS = {"chat", "instruct", "sft", "dpo", "preference"}
FINETUNE_NAME_HINTS = {"chat", "instruct", "sft", "dpo", "rlhf", "reasoning"}
MOE_HINTS = {"moe", "mixtral", "qwen2moe"}
UNKNOWN_LINEAGE_VALUES = {"", "unknown", "n/a", "none"}
TYPE_TARGET_WEIGHTS = {
    "base_source": 0.14,
    "finetune": 0.28,
    "merge": 0.14,
    "adapter": 0.14,
    "quantized": 0.14,
    "moe": 0.08,
    "multimodal": 0.05,
    "other_text": 0.03,
}


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


def _text_value(row: pd.Series, key: str) -> str:
    value = row.get(key, "")
    if pd.isna(value):
        return ""
    return str(value).strip()


def _lower_value(row: pd.Series, key: str) -> str:
    return _text_value(row, key).lower()


def _tag_tokens(row: pd.Series) -> set[str]:
    tags = row.get("tags", [])
    if isinstance(tags, str):
        return {token.strip().lower() for token in tags.split(",") if token.strip()}
    if isinstance(tags, (list, tuple, set)):
        return {str(token).strip().lower() for token in tags if str(token).strip()}
    return set()


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
    relation = _lower_value(row, "base_model_relation")
    type_value = _lower_value(row, "Type")
    pipeline_tag = _lower_value(row, "pipeline_tag")
    architecture = _lower_value(row, "Architecture")
    model_id = _lower_value(row, "model_id")
    source_model = _lower_value(row, "source_model")
    quant_method = _lower_value(row, "quant_method")
    tags = _tag_tokens(row)
    files = {str(item).lower() for item in row.get("files", [])}

    if pipeline_tag == "image-text-to-text" or "conditionalgeneration" in architecture:
        return "multimodal"
    if relation in {"adapter", "lora", "peft"} or "adapter_config.json" in files or tags & {"adapter", "peft", "lora"}:
        return "adapter"
    if any(name.endswith(".gguf") for name in files):
        return "quantized"
    if relation == "merge" or "merge" in tags:
        return "merge"
    if quant_method in KNOWN_QUANT_METHODS or relation == "quantized" or tags & KNOWN_QUANT_METHODS:
        return "quantized"
    if any(hint in architecture or hint in model_id or hint in tags for hint in MOE_HINTS):
        return "moe"
    if relation == "source":
        return "base_source"
    if relation == "finetune":
        return "finetune"
    if type_value in FINETUNE_TYPE_HINTS:
        return "finetune"
    if any(hint in model_id for hint in FINETUNE_NAME_HINTS):
        return "finetune"
    if source_model and source_model not in UNKNOWN_LINEAGE_VALUES and relation not in {"", "source"}:
        return "finetune"
    if pipeline_tag in {"text-generation", "text2text-generation"}:
        return "other_text"
    if pipeline_tag:
        return "other_text"
    return "other_text"


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
        architectures = attrs.get("architectures") or []
        rows.append(
            {
                "model_id": node["id"],
                "source_model": attrs.get("base_model_clean"),
                "base_model_relation": attrs.get("base_model_relation", "unknown"),
                "pipeline_tag": attrs.get("pipeline_tag"),
                "Architecture": architectures[0] if architectures else None,
                "tags": attrs.get("tags", []),
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
    source = result.get("source_model", pd.Series(index=result.index, dtype="object")).fillna("").astype(str).str.strip()
    model = result.get("model_id", pd.Series(index=result.index, dtype="object")).map(
        normalize_repo_id
    )
    normalized_source = source.map(normalize_repo_id)
    known_lineage = (
        ~source.str.lower().isin(UNKNOWN_LINEAGE_VALUES)
        & normalized_source.notna()
        & ~normalized_source.str.casefold().eq(model.str.casefold())
    )
    result["known_lineage_flag"] = known_lineage
    result["lineage_status"] = known_lineage.map({True: "known", False: "unknown"})
    if "Average ⬆️" in result.columns:
        eval_present = pd.to_numeric(result["Average ⬆️"], errors="coerce").notna()
    else:
        eval_present = pd.Series(False, index=result.index)
    result["eval_status"] = eval_present.map({True: "eval_present", False: "eval_missing"})
    return result


def apply_type_taxonomy(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["primary_type_bucket"] = result.apply(classify_primary_type, axis=1)
    result["loader_scenario"] = result.apply(classify_loader_scenario_from_row, axis=1)
    result["is_adapter"] = result["primary_type_bucket"].eq("adapter")
    result["is_merge"] = result["primary_type_bucket"].eq("merge")
    result["is_quantized"] = result["primary_type_bucket"].eq("quantized")
    result["is_moe"] = result["primary_type_bucket"].eq("moe")
    result["is_multimodal"] = result["loader_scenario"].eq("multimodal_transformers")
    return result


def _quota_plan(counts_by_type: pd.Series, target_size: int) -> dict[str, int]:
    quotas: dict[str, int] = {}
    remaining = target_size
    fractional_parts: list[tuple[float, str]] = []
    for bucket, available in counts_by_type.items():
        weight = TYPE_TARGET_WEIGHTS.get(bucket, 0.03)
        raw_quota = target_size * weight
        assigned = min(available, int(raw_quota))
        quotas[bucket] = assigned
        remaining -= assigned
        fractional_parts.append((raw_quota - int(raw_quota), bucket))
    for _, bucket in sorted(fractional_parts, reverse=True):
        if remaining <= 0:
            break
        if quotas[bucket] < counts_by_type[bucket]:
            quotas[bucket] += 1
            remaining -= 1
    if remaining > 0:
        for bucket in counts_by_type.sort_values(ascending=False).index:
            if remaining <= 0:
                break
            available = counts_by_type[bucket]
            room = available - quotas[bucket]
            if room <= 0:
                continue
            take = min(room, remaining)
            quotas[bucket] += take
            remaining -= take
    return quotas


def _balanced_bucket_pick(bucket_frame: pd.DataFrame, quota: int) -> pd.DataFrame:
    if quota <= 0 or bucket_frame.empty:
        return bucket_frame.iloc[0:0].copy()
    ordered = bucket_frame.sort_values(
        ["selection_priority", "canonical_model_key"],
        ascending=[False, True],
    ).reset_index(drop=True)
    cells = {
        key: group.reset_index(drop=True)
        for key, group in ordered.groupby(["lineage_status", "eval_status"], sort=True)
    }
    selected_rows = []
    round_index = 0
    while len(selected_rows) < quota:
        progressed = False
        for key in sorted(cells):
            group = cells[key]
            if round_index < len(group):
                selected_rows.append(group.iloc[round_index].to_dict())
                progressed = True
                if len(selected_rows) >= quota:
                    break
        if not progressed:
            break
        round_index += 1
    return pd.DataFrame(selected_rows)


def select_to_target(frame: pd.DataFrame, target_size: int) -> pd.DataFrame:
    result = frame.copy()
    result["selection_priority"] = (
        result["analysis_artifact_present"].fillna(False).astype(int) * 8
        + result["eval_status"].eq("eval_present").astype(int) * 4
        + result["lineage_status"].eq("unknown").astype(int) * 2
        + result["primary_type_bucket"].isin({"adapter", "quantized", "merge", "moe", "multimodal"}).astype(int)
    )
    ordered = (
        result.sort_values(
            ["primary_type_bucket", "selection_priority", "canonical_model_key"],
            ascending=[True, False, True],
        )
        .drop_duplicates(subset=["canonical_model_key"], keep="first")
        .reset_index(drop=True)
    )
    counts_by_type = ordered["primary_type_bucket"].value_counts()
    quotas = _quota_plan(counts_by_type, target_size=min(target_size, len(ordered)))

    picked_frames = []
    picked_keys: set[str] = set()
    for bucket, quota in quotas.items():
        bucket_frame = ordered[ordered["primary_type_bucket"] == bucket].reset_index(drop=True)
        picked = _balanced_bucket_pick(bucket_frame, quota)
        if not picked.empty:
            picked_frames.append(picked)
            picked_keys.update(picked["canonical_model_key"].tolist())

    selected = pd.concat(picked_frames, ignore_index=True, sort=False) if picked_frames else ordered.iloc[0:0].copy()
    if len(selected) < min(target_size, len(ordered)):
        leftovers = ordered[~ordered["canonical_model_key"].isin(picked_keys)].sort_values(
            ["selection_priority", "canonical_model_key"],
            ascending=[False, True],
        )
        needed = min(target_size, len(ordered)) - len(selected)
        selected = pd.concat([selected, leftovers.head(needed)], ignore_index=True, sort=False)

    return selected.reset_index(drop=True)
