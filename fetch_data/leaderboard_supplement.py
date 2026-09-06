from __future__ import annotations

import re

import pandas as pd

from fetch_data.phase2_schema import canonical_model_key, normalize_repo_id


HF_LINK_PATTERN = re.compile(r"huggingface\.co/([^\"?#]+)")
MERGE_DISPLAY_SUFFIX = re.compile(r"\s+\(merge\)\s*$", re.IGNORECASE)
LEADERBOARD_SCORE_COLUMNS = (
    "Average ⬆️",
    "IFEval Raw",
    "IFEval",
    "BBH Raw",
    "BBH",
    "MATH Lvl 5 Raw",
    "MATH Lvl 5",
    "GPQA Raw",
    "GPQA",
    "MUSR Raw",
    "MUSR",
    "MMLU-PRO Raw",
    "MMLU-PRO",
)
KNOWN_PRECISIONS = {
    "bf16",
    "bfloat16",
    "float16",
    "float32",
    "fp16",
    "fp32",
    "int4",
    "int8",
    "4bit",
    "8bit",
}


def _parse_model_string(model_str: str) -> tuple[str, str | None]:
    value = str(model_str).strip()
    if "@" in value and not value.startswith("@"):
        repo_id, revision = value.split("@", 1)
        return repo_id.strip(), revision.strip() or None
    return value, None


def _normalize_leaderboard_repo_id(raw_value: object) -> str | None:
    if pd.isna(raw_value):
        return None
    value = str(raw_value).strip()
    if not value:
        return None
    match = HF_LINK_PATTERN.search(value)
    if match:
        value = match.group(1).strip()
    value = MERGE_DISPLAY_SUFFIX.sub("", value)
    repo_id, _ = _parse_model_string(value)
    return normalize_repo_id(repo_id)


def _eval_name_to_repo_id(eval_name: object) -> str | None:
    if pd.isna(eval_name):
        return None
    value = str(eval_name).strip()
    if not value:
        return None
    parts = value.split("_")
    if len(parts) < 2:
        return None
    if parts[-1].lower() in KNOWN_PRECISIONS:
        parts = parts[:-1]
    if len(parts) < 2:
        return None
    return normalize_repo_id(f"{parts[0]}/{'_'.join(parts[1:])}")


def build_leaderboard_repo_ids(frame: pd.DataFrame) -> pd.Series:
    repo_ids = pd.Series([None] * len(frame), index=frame.index, dtype="object")
    if "Model" in frame.columns:
        repo_ids = frame["Model"].map(_normalize_leaderboard_repo_id)
    if "Base Model" in frame.columns:
        base_ids = frame["Base Model"].map(_normalize_leaderboard_repo_id)
        repo_ids = repo_ids.where(repo_ids.notna(), base_ids)
    if "eval_name" in frame.columns:
        eval_ids = frame["eval_name"].map(_eval_name_to_repo_id)
        repo_ids = repo_ids.where(repo_ids.notna(), eval_ids)
    return repo_ids


def leaderboard_frame_to_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["repo_id_norm"] = build_leaderboard_repo_ids(result)
    result["canonical_model_key"] = result["repo_id_norm"].map(lambda repo_id: canonical_model_key(repo_id, None))
    base_model = (
        result["Base Model"].map(_normalize_leaderboard_repo_id)
        if "Base Model" in result.columns
        else pd.Series([None] * len(result), index=result.index)
    )
    self_reference = base_model.str.casefold().eq(result["repo_id_norm"].str.casefold())
    base_model = base_model.mask(self_reference)
    merged_flag = (
        result["Merged"].fillna(False).astype(bool)
        if "Merged" in result.columns
        else pd.Series(False, index=result.index)
    )
    if "Base Model" in result.columns:
        merged_flag |= result["Base Model"].fillna("").astype(str).str.contains(
            MERGE_DISPLAY_SUFFIX
        )
    if "Type" in result.columns:
        merged_flag |= result["Type"].fillna("").astype(str).str.contains(
            "merge", case=False, regex=False
        )

    relation = pd.Series("unknown", index=result.index, dtype="object")
    relation = relation.where(~merged_flag, "merge")
    relation = relation.where(~(base_model.notna() & ~merged_flag), "finetune")

    candidate_columns = {
            "model_id": result["repo_id_norm"],
            "repo_id_norm": result["repo_id_norm"],
            "revision_norm": None,
            "canonical_model_key": result["canonical_model_key"],
            "source_model": base_model.fillna(""),
            "base_model_relation": relation,
            "Architecture": result.get("Architecture"),
            "Type": result.get("Type"),
            "#Params (B)": result.get("#Params (B)"),
            "Average ⬆️": result.get("Average ⬆️"),
            "candidate_source": "leaderboard_dataset",
    }
    for column in (
        "eval_name",
        "Model sha",
        "Generation",
        *LEADERBOARD_SCORE_COLUMNS,
    ):
        if column in result.columns:
            candidate_columns[column] = result[column]
    candidates = pd.DataFrame(candidate_columns)
    candidates = candidates[candidates["canonical_model_key"].notna()].drop_duplicates(
        subset=["canonical_model_key"], keep="first"
    )
    return candidates.reset_index(drop=True)


def load_leaderboard_supplementary_candidates(dataset_id: str = "open-llm-leaderboard/contents") -> pd.DataFrame:
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split="train")
    return leaderboard_frame_to_candidates(dataset.to_pandas())
