from __future__ import annotations

import pandas as pd

from fetch_data.phase2_schema import canonical_model_key, normalize_repo_id


DEFAULT_HF_SUPPLEMENT_SEARCHES = ["moe", "mixtral", "qwen2moe", "AWQ", "GPTQ", "GGUF"]


def _card_base_model(card_data) -> str | None:
    if card_data is None:
        return None
    if isinstance(card_data, dict):
        value = card_data.get("base_model") or card_data.get("base_models")
    else:
        value = getattr(card_data, "base_model", None)
        if value is None:
            value = getattr(card_data, "base_models", None)
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _supplement_relation(model_info) -> str:
    tags = {str(tag).strip().lower() for tag in getattr(model_info, "tags", [])}
    model_id = str(getattr(model_info, "modelId", "")).lower()
    config = getattr(model_info, "config", {}) or {}
    quant_config = config.get("quantization_config", {}) if isinstance(config, dict) else {}
    quant_method = str(quant_config.get("quant_method", "")).lower()

    if {"adapter", "peft", "lora"} & tags:
        return "adapter"
    if "merge" in tags:
        return "merge"
    if quant_method in {"awq", "gptq", "bitsandbytes"} or {"awq", "gptq", "gguf"} & tags:
        return "quantized"
    if any(token in model_id for token in ("awq", "gptq", "gguf")):
        return "quantized"
    base_model = _card_base_model(getattr(model_info, "cardData", None))
    if base_model:
        return "finetune"
    return "unknown"


def _supplement_row(model_info, *, candidate_source: str) -> dict[str, object]:
    config = getattr(model_info, "config", {}) or {}
    architectures = config.get("architectures", []) if isinstance(config, dict) else []
    quant_config = config.get("quantization_config", {}) if isinstance(config, dict) else {}
    model_id = str(getattr(model_info, "modelId"))
    repo_id_norm = normalize_repo_id(model_id)
    return {
        "model_id": model_id,
        "repo_id_norm": repo_id_norm,
        "revision_norm": None,
        "canonical_model_key": canonical_model_key(repo_id_norm, None),
        "source_model": _card_base_model(getattr(model_info, "cardData", None)) or "",
        "base_model_relation": _supplement_relation(model_info),
        "pipeline_tag": getattr(model_info, "pipeline_tag", None),
        "Architecture": architectures[0] if architectures else None,
        "quant_method": quant_config.get("quant_method"),
        "downloads": getattr(model_info, "downloads", None),
        "likes": getattr(model_info, "likes", None),
        "tags": list(getattr(model_info, "tags", []) or []),
        "candidate_source": candidate_source,
    }


def collect_hf_supplementary_candidates(
    *,
    api,
    per_query_limit: int = 100,
    searches: list[str] | None = None,
    include_multimodal: bool = True,
) -> pd.DataFrame:
    search_terms = searches or DEFAULT_HF_SUPPLEMENT_SEARCHES
    rows: list[dict[str, object]] = []

    for search in search_terms:
        for model_info in api.list_models(
            search=search,
            limit=per_query_limit,
            full=True,
            cardData=True,
            fetch_config=True,
            token=True,
        ):
            rows.append(_supplement_row(model_info, candidate_source="hf_search"))

    if include_multimodal:
        for model_info in api.list_models(
            pipeline_tag="image-text-to-text",
            limit=per_query_limit,
            full=True,
            cardData=True,
            fetch_config=True,
            token=True,
        ):
            rows.append(_supplement_row(model_info, candidate_source="hf_search"))

    if not rows:
        return pd.DataFrame(
            columns=[
                "model_id",
                "repo_id_norm",
                "revision_norm",
                "canonical_model_key",
                "source_model",
                "base_model_relation",
                "pipeline_tag",
                "Architecture",
                "quant_method",
                "downloads",
                "likes",
                "tags",
                "candidate_source",
            ]
        )

    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(subset=["canonical_model_key"], keep="first").reset_index(drop=True)
    return frame
