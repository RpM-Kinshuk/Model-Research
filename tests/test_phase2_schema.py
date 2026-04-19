from pathlib import Path

import pandas as pd

from fetch_data.phase2_schema import (
    canonical_model_key,
    classify_loader_scenario_from_row,
    classify_primary_type,
    collapse_leaderboard_duplicates,
    normalize_repo_id,
    parse_repo_and_revision,
    seed_analysis_status,
)


def test_parse_repo_and_revision_splits_explicit_revision():
    repo_id, revision = parse_repo_and_revision("org/model@refs/pr/7")
    assert repo_id == "org/model"
    assert revision == "refs/pr/7"


def test_normalize_repo_id_collapses_hf_urls_and_owner_model_escape():
    assert normalize_repo_id("https://huggingface.co/owner/model") == "owner/model"
    assert normalize_repo_id("owner--model") == "owner/model"


def test_canonical_model_key_uses_revision_when_present():
    assert canonical_model_key("owner/model", None) == "owner/model"
    assert canonical_model_key("owner/model", "rev-a") == "owner/model@rev-a"


def test_collapse_leaderboard_duplicates_prefers_more_complete_row():
    leaderboard = pd.DataFrame(
        [
            {
                "model_id": "owner/model",
                "Precision": "float16",
                "Architecture": "LlamaForCausalLM",
                "#Params (B)": 8.0,
                "Average ⬆️": 30.0,
                "Type": "chat",
            },
            {
                "model_id": "owner/model",
                "Precision": "bfloat16",
                "Architecture": "Unknown",
                "#Params (B)": 0.0,
                "Average ⬆️": 28.0,
                "Type": "other",
            },
        ]
    )
    canonical, raw = collapse_leaderboard_duplicates(leaderboard)
    assert len(raw) == 2
    assert len(canonical) == 1
    row = canonical.iloc[0]
    assert row["Architecture"] == "LlamaForCausalLM"
    assert row["canonical_model_key"] == "owner/model"


def test_seed_analysis_status_marks_existing_svd_outputs(tmp_path: Path):
    (tmp_path / "owner--model.csv").write_text("alpha\n1.2\n")
    frame = pd.DataFrame({"model_id": ["owner/model", "other/model"]})
    status = seed_analysis_status(frame, tmp_path)
    assert status.loc[status["model_id"] == "owner/model", "analysis_artifact_present"].item() is True
    assert status.loc[status["model_id"] == "other/model", "analysis_artifact_present"].item() is False


def test_classify_primary_type_and_loader_scenario_cover_core_cases():
    adapter_row = pd.Series(
        {
            "base_model_relation": "adapter",
            "pipeline_tag": "text-generation",
            "files": ["adapter_config.json", "adapter_model.safetensors"],
        }
    )
    gguf_row = pd.Series(
        {
            "base_model_relation": "quantized",
            "pipeline_tag": "text-generation",
            "files": ["model.Q4_K_M.gguf", "tokenizer.json"],
        }
    )
    multimodal_row = pd.Series(
        {
            "base_model_relation": "finetune",
            "pipeline_tag": "image-text-to-text",
            "files": ["config.json", "preprocessor_config.json"],
            "Architecture": "MllamaForConditionalGeneration",
        }
    )
    assert classify_primary_type(adapter_row) == "adapter"
    assert classify_loader_scenario_from_row(adapter_row) == "adapter_requires_base"
    assert classify_loader_scenario_from_row(gguf_row) == "quantized_alt_format"
    assert classify_loader_scenario_from_row(multimodal_row) == "multimodal_transformers"
