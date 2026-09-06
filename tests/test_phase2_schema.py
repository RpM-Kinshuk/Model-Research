from pathlib import Path

import pandas as pd

from fetch_data.phase2_schema import (
    add_lineage_flags,
    canonical_model_key,
    classify_loader_scenario_from_row,
    classify_primary_type,
    collapse_leaderboard_duplicates,
    load_atlas_candidates,
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


def test_add_lineage_flags_assigns_lineage_and_eval_status():
    frame = pd.DataFrame(
        [
            {"model_id": "org/known", "source_model": "base/model", "Average ⬆️": 42.0},
            {"model_id": "org/unknown", "source_model": "unknown", "Average ⬆️": None},
            {"model_id": "org/self", "source_model": "org/self", "Average ⬆️": 1.0},
        ]
    )
    flagged = add_lineage_flags(frame)
    assert flagged.loc[flagged["model_id"] == "org/known", "lineage_status"].item() == "known"
    assert flagged.loc[flagged["model_id"] == "org/known", "eval_status"].item() == "eval_present"
    assert flagged.loc[flagged["model_id"] == "org/unknown", "lineage_status"].item() == "unknown"
    assert flagged.loc[flagged["model_id"] == "org/unknown", "eval_status"].item() == "eval_missing"
    assert flagged.loc[flagged["model_id"] == "org/self", "lineage_status"].item() == "unknown"


def test_classify_primary_type_uses_content_hints_when_relation_is_ambiguous():
    chat_row = pd.Series(
        {
            "base_model_relation": "unknown_annotated",
            "source_model": "org/base",
            "Type": "chat",
            "pipeline_tag": "text-generation",
            "files": ["config.json", "model.safetensors"],
        }
    )
    moe_row = pd.Series(
        {
            "base_model_relation": "unknown",
            "source_model": "unknown",
            "Type": "other",
            "pipeline_tag": "text-generation",
            "Architecture": "Qwen2MoeForCausalLM",
            "model_id": "org/qwen2moe-experimental",
            "files": ["config.json", "model.safetensors"],
        }
    )
    fallback_row = pd.Series(
        {
            "base_model_relation": "unknown",
            "source_model": "unknown",
            "Type": "other",
            "pipeline_tag": "text-generation",
            "files": ["config.json", "model.safetensors"],
        }
    )
    assert classify_primary_type(chat_row) == "finetune"
    assert classify_primary_type(moe_row) == "moe"
    assert classify_primary_type(fallback_row) == "other_text"


def test_load_atlas_candidates_keeps_extra_type_signals(tmp_path: Path):
    atlas_path = tmp_path / "large_nlp_data.json"
    atlas_path.write_text(
        """
        {
          "nodes": [
            {
              "id": "org/multimodal-model",
              "attributes": {
                "base_model_clean": "unknown",
                "base_model_relation": "unknown",
                "pipeline_tag": "image-text-to-text",
                "architectures": ["MllamaForConditionalGeneration"],
                "tags": ["vision", "multimodal"]
              }
            }
          ],
          "edges": []
        }
        """
    )
    frame = load_atlas_candidates(atlas_path)
    row = frame.iloc[0]
    assert row["pipeline_tag"] == "image-text-to-text"
    assert row["Architecture"] == "MllamaForConditionalGeneration"
    assert row["tags"] == ["vision", "multimodal"]
