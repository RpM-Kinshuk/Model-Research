import pandas as pd

from fetch_data.ablation_schema import (
    build_phase2_duplicate_keys,
    classify_esd_suitability,
    mark_phase2_overlaps,
    normalize_ablation_candidates,
)


def test_normalize_ablation_candidates_adds_eval_and_identity_fields():
    frame = pd.DataFrame(
        [
            {
                "model_id": "Owner/Model@main",
                "Model sha": "abc123",
                "Average ⬆️": 42.5,
                "IFEval": 50.0,
                "BBH": 30.0,
                "MATH Lvl 5": 10.0,
                "GPQA": 20.0,
                "MUSR": 25.0,
                "MMLU-PRO": 60.0,
            }
        ]
    )

    normalized = normalize_ablation_candidates(frame, eval_source="oll_contents")
    row = normalized.iloc[0]

    assert row["repo_id_norm"] == "Owner/Model"
    assert row["revision_norm"] == "main"
    assert row["model_id_norm"] == "owner/model"
    assert row["canonical_model_key"] == "Owner/Model@main"
    assert row["model_sha"] == "abc123"
    assert row["average_numeric"] == 42.5
    assert row["eval_task_count"] == 6
    assert row["eval_coverage_flag"] == "complete"
    assert row["eval_status"] == "eval_present"
    assert row["eval_source"] == "oll_contents"


def test_normalize_ablation_candidates_marks_partial_and_missing_eval():
    frame = pd.DataFrame(
        [
            {"model_id": "org/partial", "Average ⬆️": 10.0, "IFEval": 50.0},
            {"model_id": "org/missing", "Average ⬆️": None},
        ]
    )

    normalized = normalize_ablation_candidates(frame, eval_source="oll_contents")

    assert normalized.loc[normalized["model_id"] == "org/partial", "eval_coverage_flag"].item() == "partial"
    assert normalized.loc[normalized["model_id"] == "org/missing", "eval_coverage_flag"].item() == "missing"
    assert normalized.loc[normalized["model_id"] == "org/missing", "eval_status"].item() == "eval_missing"


def test_mark_phase2_overlaps_matches_case_insensitive_repo_key_and_sha():
    phase2 = pd.DataFrame(
        [
            {"model_id": "Org/Seen", "canonical_model_key": "Org/Seen", "Model sha": "sha-seen"},
        ]
    )
    candidates = normalize_ablation_candidates(
        pd.DataFrame(
            [
                {"model_id": "org/seen", "Average ⬆️": 1.0},
                {"model_id": "org/sha-match", "Model sha": "sha-seen", "Average ⬆️": 2.0},
                {"model_id": "org/new", "Average ⬆️": 3.0},
            ]
        ),
        eval_source="oll_contents",
    )

    marked = mark_phase2_overlaps(candidates, build_phase2_duplicate_keys(phase2))

    assert marked.loc[marked["model_id"] == "org/seen", "phase2_overlap"].item() is True
    assert marked.loc[marked["model_id"] == "org/sha-match", "phase2_overlap"].item() is True
    assert marked.loc[marked["model_id"] == "org/new", "phase2_overlap"].item() is False


def test_classify_esd_suitability_uses_loader_and_lineage_signals():
    dense = pd.Series({"primary_type_bucket": "finetune", "loader_scenario": "standard_transformers"})
    moe = pd.Series({"primary_type_bucket": "moe", "loader_scenario": "standard_transformers"})
    gguf = pd.Series({"primary_type_bucket": "quantized", "loader_scenario": "quantized_alt_format", "files": ["model.gguf"]})
    adapter_missing_base = pd.Series(
        {"primary_type_bucket": "adapter", "loader_scenario": "adapter_requires_base", "source_model": "unknown"}
    )

    assert classify_esd_suitability(dense) == "good"
    assert classify_esd_suitability(moe) == "usable_with_caution"
    assert classify_esd_suitability(gguf) == "separate_or_exclude"
    assert classify_esd_suitability(adapter_missing_base) == "separate_or_exclude"
