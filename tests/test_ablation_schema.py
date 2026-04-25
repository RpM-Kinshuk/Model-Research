import pandas as pd

from fetch_data.ablation_schema import (
    assign_lineage_eval_group,
    build_phase2_duplicate_keys,
    classify_esd_suitability,
    mark_phase2_overlaps,
    normalize_ablation_candidates,
    prepare_ablation_candidates,
    select_ablation_rows,
    summarize_ablation_outputs,
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


def test_normalize_ablation_candidates_marks_task_scores_without_average_as_eval_present():
    frame = pd.DataFrame(
        [
            {"model_id": "org/partial", "Average ⬆️": None, "IFEval": 50.0},
            {
                "model_id": "org/complete",
                "Average ⬆️": None,
                "IFEval": 50.0,
                "BBH": 30.0,
                "MATH Lvl 5": 10.0,
                "GPQA": 20.0,
                "MUSR": 25.0,
                "MMLU-PRO": 60.0,
            },
        ]
    )

    normalized = normalize_ablation_candidates(frame, eval_source="oll_contents")

    assert normalized.loc[normalized["model_id"] == "org/partial", "eval_coverage_flag"].item() == "partial"
    assert normalized.loc[normalized["model_id"] == "org/partial", "eval_status"].item() == "eval_present"
    assert normalized.loc[normalized["model_id"] == "org/complete", "eval_coverage_flag"].item() == "complete"
    assert normalized.loc[normalized["model_id"] == "org/complete", "eval_status"].item() == "eval_present"


def test_normalize_ablation_candidates_materializes_model_id_from_fallback_identifier():
    frame = pd.DataFrame(
        [
            {"fullname": "Owner/Fallback@main", "Average ⬆️": 12.0},
        ]
    )

    normalized = normalize_ablation_candidates(frame, eval_source="oll_contents")
    row = normalized.iloc[0]

    assert row["model_id"] == "Owner/Fallback@main"
    assert row["repo_id_norm"] == "Owner/Fallback"
    assert row["revision_norm"] == "main"
    assert row["canonical_model_key"] == "Owner/Fallback@main"


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


def test_classify_esd_suitability_splits_comma_separated_file_string():
    row = pd.Series(
        {
            "primary_type_bucket": "finetune",
            "loader_scenario": "standard_transformers",
            "files": "model.gguf,tokenizer.json",
        }
    )

    assert classify_esd_suitability(row) == "separate_or_exclude"


def test_assign_lineage_eval_group_names_comparable_scenarios():
    assert assign_lineage_eval_group(pd.Series({"base_model_relation": "finetune", "source_model": "org/base"})) == "same_base_finetune"
    assert assign_lineage_eval_group(pd.Series({"base_model_relation": "merge", "source_model": "org/base"})) == "known_merge_family"
    assert assign_lineage_eval_group(pd.Series({"base_model_relation": "adapter", "source_model": "org/base"})) == "adapter_versus_merged_base_pair"
    assert assign_lineage_eval_group(pd.Series({"base_model_relation": "source", "source_model": "org/base"})) == "base_model_family_variant"
    assert assign_lineage_eval_group(pd.Series({"base_model_relation": "unknown", "source_model": "unknown"})) == "unrelated_negative_pair"


def test_prepare_ablation_candidates_adds_type_lineage_suitability_and_overlap():
    raw = pd.DataFrame(
        [
            {
                "model_id": "org/new-chat",
                "source_model": "org/base",
                "base_model_relation": "finetune",
                "Type": "chat",
                "Average ⬆️": 12.0,
                "pipeline_tag": "text-generation",
            },
            {
                "model_id": "org/old",
                "source_model": "unknown",
                "base_model_relation": "unknown",
                "Average ⬆️": 11.0,
                "pipeline_tag": "text-generation",
            },
        ]
    )
    phase2 = pd.DataFrame([{"model_id": "org/old"}])

    prepared = prepare_ablation_candidates(raw, phase2_frame=phase2, eval_source="oll_contents")

    assert prepared.loc[prepared["model_id"] == "org/new-chat", "known_lineage_flag"].item() is True
    assert prepared.loc[prepared["model_id"] == "org/new-chat", "primary_type_bucket"].item() == "finetune"
    assert prepared.loc[prepared["model_id"] == "org/new-chat", "esd_suitability"].item() == "good"
    assert prepared.loc[prepared["model_id"] == "org/new-chat", "lineage_eval_group"].item() == "same_base_finetune"
    assert prepared.loc[prepared["model_id"] == "org/new-chat", "lineage_source"].item() == "metadata"
    assert prepared.loc[prepared["model_id"] == "org/new-chat", "lineage_confidence"].item() == "medium"
    assert prepared.loc[prepared["model_id"] == "org/old", "lineage_eval_group"].item() == "unrelated_negative_pair"
    assert prepared.loc[prepared["model_id"] == "org/old", "lineage_confidence"].item() == "unknown"
    assert prepared.loc[prepared["model_id"] == "org/old", "phase2_overlap"].item() is True


def test_select_ablation_rows_rejects_overlaps_and_caps_missing_eval():
    raw = pd.DataFrame(
        [
            {"model_id": "org/eval-a", "source_model": "org/base", "base_model_relation": "finetune", "Average ⬆️": 1.0},
            {"model_id": "org/eval-b", "source_model": "org/base", "base_model_relation": "merge", "Average ⬆️": 2.0},
            {"model_id": "org/missing-a", "source_model": "unknown", "base_model_relation": "unknown", "Average ⬆️": None},
            {"model_id": "org/missing-b", "source_model": "unknown", "base_model_relation": "unknown", "Average ⬆️": None},
            {"model_id": "org/phase2", "source_model": "org/base", "base_model_relation": "finetune", "Average ⬆️": 3.0},
            {"model_id": "org/gguf", "source_model": "org/base", "base_model_relation": "quantized", "Average ⬆️": 4.0, "files": ["model.gguf"]},
        ]
    )
    prepared = prepare_ablation_candidates(
        raw,
        phase2_frame=pd.DataFrame([{"model_id": "org/phase2"}]),
        eval_source="oll_contents",
    )

    selected, rejected = select_ablation_rows(prepared, target_size=4, max_eval_missing_fraction=0.25)

    assert set(selected["model_id"]) == {"org/eval-a", "org/eval-b", "org/missing-a"}
    assert selected["eval_status"].eq("eval_missing").sum() == 1
    assert rejected.loc[rejected["model_id"] == "org/phase2", "rejection_reason"].item() == "phase2_duplicate"
    assert rejected.loc[rejected["model_id"] == "org/gguf", "rejection_reason"].item() == "unsupported_artifact_format"
    assert rejected.loc[rejected["model_id"] == "org/missing-b", "rejection_reason"].item() == "missing_eval_over_cap"


def test_select_ablation_rows_rejects_only_lower_priority_duplicates():
    raw = pd.DataFrame(
        [
            {
                "model_id": "org/dup",
                "source_model": "org/base",
                "base_model_relation": "finetune",
                "Average ⬆️": 9.0,
                "IFEval": 1.0,
                "BBH": 1.0,
                "MATH Lvl 5": 1.0,
                "GPQA": 1.0,
                "MUSR": 1.0,
                "MMLU-PRO": 1.0,
                "Model sha": "higher-priority",
            },
            {
                "model_id": "org/dup",
                "source_model": "org/base",
                "base_model_relation": "finetune",
                "Average ⬆️": 10.0,
                "Model sha": "lower-priority",
            },
            {"model_id": "org/other", "source_model": "org/base", "base_model_relation": "merge", "Average ⬆️": 8.0},
        ]
    )
    prepared = prepare_ablation_candidates(raw, phase2_frame=pd.DataFrame(), eval_source="oll_contents")

    selected, rejected = select_ablation_rows(prepared, target_size=2)

    assert set(selected["model_id"]) == {"org/dup", "org/other"}
    assert selected.loc[selected["model_id"] == "org/dup", "model_sha"].item() == "higher-priority"
    assert rejected.loc[rejected["model_sha"] == "lower-priority", "rejection_reason"].item() == "duplicate_lower_priority"


def test_select_ablation_rows_marks_unselected_eval_present_as_type_quota_pressure():
    raw = pd.DataFrame(
        [
            {"model_id": "org/a", "source_model": "org/base", "base_model_relation": "finetune", "Average ⬆️": 3.0},
            {"model_id": "org/b", "source_model": "org/base", "base_model_relation": "finetune", "Average ⬆️": 2.0},
            {"model_id": "org/c", "source_model": "org/base", "base_model_relation": "finetune", "Average ⬆️": 1.0},
        ]
    )
    prepared = prepare_ablation_candidates(raw, phase2_frame=pd.DataFrame(), eval_source="oll_contents")

    selected, rejected = select_ablation_rows(prepared, target_size=2)

    assert set(selected["model_id"]) == {"org/a", "org/b"}
    assert rejected.loc[rejected["model_id"] == "org/c", "eval_status"].item() == "eval_present"
    assert rejected.loc[rejected["model_id"] == "org/c", "rejection_reason"].item() == "type_quota_pressure"


def test_summarize_ablation_outputs_reports_core_metrics():
    candidates = prepare_ablation_candidates(
        pd.DataFrame(
            [
                {"model_id": "org/a", "source_model": "org/base", "base_model_relation": "finetune", "Average ⬆️": 1.0},
                {"model_id": "org/b", "source_model": "unknown", "base_model_relation": "unknown", "Average ⬆️": None},
                {"model_id": "org/c", "source_model": "org/base", "base_model_relation": "merge", "Average ⬆️": 2.0},
            ]
        ),
        phase2_frame=pd.DataFrame(),
        eval_source="oll_contents",
    )
    selected, rejected = select_ablation_rows(candidates, target_size=2, max_eval_missing_fraction=0.5)

    summary = summarize_ablation_outputs(candidates, selected, rejected)

    assert summary["total_candidate_rows"] == 3
    assert summary["selected_rows"] == 2
    assert summary["eval_labelled_rows"] == 2
    assert summary["missing_eval_percentage"] == 0.0
    assert summary["phase2_overlap_rows"] == 0
    assert summary["counts_by_eval_source"] == {"oll_contents": 2}
    assert summary["counts_by_primary_type_bucket"] == {"finetune": 1, "merge": 1}
    assert summary["counts_by_esd_suitability"] == {"good": 2}
    assert summary["counts_by_lineage_status"] == {"known": 2}
    assert summary["top_rejection_reasons"] == {"missing_eval_over_cap": 1}
