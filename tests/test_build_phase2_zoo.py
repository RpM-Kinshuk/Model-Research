import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from fetch_data.build_phase2_zoo import build_phase2_outputs


def test_build_phase2_outputs_ingests_seed_and_writes_views(tmp_path: Path):
    metadata_path = tmp_path / "sampled_metadata.csv"
    leaderboard_path = tmp_path / "sampled_leaderboard.csv"
    atlas_path = tmp_path / "large_nlp_data.json"
    svd_dir = tmp_path / "svd_results"
    output_dir = tmp_path / "curated"
    svd_dir.mkdir()
    (svd_dir / "org--seed-model.csv").write_text("alpha\n1.0\n")

    pd.DataFrame(
        [
            {"model_id": "org/seed-model", "source_model": "base/model", "base_model_relation": "finetune"},
            {"model_id": "org/adapter-model", "source_model": "base/model", "base_model_relation": "adapter"},
        ]
    ).to_csv(metadata_path, index=False)

    pd.DataFrame(
        [
            {
                "model_id": "org/seed-model",
                "Architecture": "LlamaForCausalLM",
                "#Params (B)": 8.0,
                "Type": "chat",
                "Average ⬆️": 30.0,
                "pipeline_tag": "text-generation",
            },
            {
                "model_id": "org/seed-model",
                "Architecture": "Unknown",
                "#Params (B)": 0.0,
                "Type": "other",
                "Average ⬆️": 10.0,
                "pipeline_tag": "text-generation",
            },
        ]
    ).to_csv(leaderboard_path, index=False)

    atlas_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "org/seed-model", "attributes": {"base_model_clean": "base/model", "base_model_relation": "finetune"}},
                    {"id": "org/new-merge", "attributes": {"base_model_clean": "base/model", "base_model_relation": "merge"}},
                    {"id": "org/new-unknown", "attributes": {"base_model_clean": "unknown", "base_model_relation": "unknown"}},
                ],
                "edges": [],
            }
        )
    )

    outputs = build_phase2_outputs(
        metadata_path=metadata_path,
        leaderboard_path=leaderboard_path,
        atlas_json_path=atlas_path,
        svd_dir=svd_dir,
        output_dir=output_dir,
        target_size=3,
        skip_hub_probes=True,
    )
    assert (output_dir / "model_zoo_phase2.csv").exists()
    assert (output_dir / "model_zoo_phase2_prediction_ready.csv").exists()
    assert outputs["canonical"]["canonical_model_key"].nunique() == len(outputs["canonical"])
    assert set(outputs["canonical"]["model_id"]) == {"org/seed-model", "org/adapter-model", "org/new-merge"}
    assert outputs["canonical"].loc[outputs["canonical"]["model_id"] == "org/seed-model", "analysis_artifact_present"].item() is True


def test_build_phase2_outputs_keeps_known_and_unknown_when_available(tmp_path: Path):
    metadata_path = tmp_path / "sampled_metadata.csv"
    leaderboard_path = tmp_path / "sampled_leaderboard.csv"
    atlas_path = tmp_path / "large_nlp_data.json"
    svd_dir = tmp_path / "svd_results"
    output_dir = tmp_path / "curated"
    svd_dir.mkdir()

    pd.DataFrame(
        [{"model_id": "org/base", "source_model": "org/base", "base_model_relation": "source"}]
    ).to_csv(metadata_path, index=False)
    pd.DataFrame([], columns=["model_id", "Architecture", "#Params (B)", "Type", "Average ⬆️", "pipeline_tag"]).to_csv(leaderboard_path, index=False)
    atlas_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "org/base", "attributes": {"base_model_clean": "org/base", "base_model_relation": "source"}},
                    {"id": "org/known-ft", "attributes": {"base_model_clean": "org/base", "base_model_relation": "finetune"}},
                    {"id": "org/unknown-ft", "attributes": {"base_model_clean": "unknown", "base_model_relation": "finetune"}},
                ],
                "edges": [],
            }
        )
    )
    outputs = build_phase2_outputs(
        metadata_path=metadata_path,
        leaderboard_path=leaderboard_path,
        atlas_json_path=atlas_path,
        svd_dir=svd_dir,
        output_dir=output_dir,
        target_size=3,
        skip_hub_probes=True,
    )
    finetunes = outputs["canonical"][outputs["canonical"]["primary_type_bucket"] == "finetune"]
    assert finetunes["known_lineage_flag"].nunique() == 2


def test_build_phase2_outputs_records_probe_failures(tmp_path: Path):
    metadata_path = tmp_path / "sampled_metadata.csv"
    leaderboard_path = tmp_path / "sampled_leaderboard.csv"
    atlas_path = tmp_path / "large_nlp_data.json"
    svd_dir = tmp_path / "svd_results"
    output_dir = tmp_path / "curated"
    svd_dir.mkdir()

    pd.DataFrame(
        [{"model_id": "org/model", "source_model": "base/model", "base_model_relation": "adapter"}]
    ).to_csv(metadata_path, index=False)
    pd.DataFrame([], columns=["model_id", "Architecture", "#Params (B)", "Type", "Average ⬆️", "pipeline_tag"]).to_csv(leaderboard_path, index=False)
    atlas_path.write_text(json.dumps({"nodes": [], "edges": []}))

    fake_probe = Mock(
        return_value={
            "availability_status": "public",
            "fetchability_status": "public_but_structurally_incomplete",
            "failure_class": "adapter_base_unresolved",
        }
    )

    outputs = build_phase2_outputs(
        metadata_path=metadata_path,
        leaderboard_path=leaderboard_path,
        atlas_json_path=atlas_path,
        svd_dir=svd_dir,
        output_dir=output_dir,
        target_size=1,
        skip_hub_probes=False,
        probe_fn=fake_probe,
    )
    assert fake_probe.called
    assert (output_dir / "model_zoo_phase2_probe_failures.csv").exists()
    assert outputs["probe_failures"]["failure_class"].iloc[0] == "adapter_base_unresolved"
