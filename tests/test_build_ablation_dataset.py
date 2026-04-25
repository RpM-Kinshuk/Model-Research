import json
from pathlib import Path

import pandas as pd
import pytest

from fetch_data.build_ablation_dataset import build_ablation_outputs, load_open_llm_leaderboard_contents


def test_load_open_llm_leaderboard_contents_uses_injected_loader():
    calls = []

    class FakeDataset:
        def to_pandas(self):
            return pd.DataFrame([{"model_id": "org/fetched", "Average ⬆️": 9.0}])

    def fake_loader(dataset_id, split):
        calls.append((dataset_id, split))
        return FakeDataset()

    frame = load_open_llm_leaderboard_contents(load_dataset_fn=fake_loader)

    assert calls == [("open-llm-leaderboard/contents", "train")]
    assert frame.loc[0, "model_id"] == "org/fetched"
    assert frame.loc[0, "eval_source"] == "open_llm_leaderboard_contents"


def test_build_ablation_outputs_writes_expected_files(tmp_path: Path):
    contents_path = tmp_path / "contents.csv"
    phase2_path = tmp_path / "phase2.csv"
    output_dir = tmp_path / "ablation"

    pd.DataFrame(
        [
            {"model_id": "org/eval-a", "source_model": "org/base", "base_model_relation": "finetune", "Average ⬆️": 1.0, "IFEval": 2.0},
            {"model_id": "org/eval-b", "source_model": "org/base", "base_model_relation": "merge", "Average ⬆️": 2.0, "BBH": 3.0},
            {"model_id": "org/missing", "source_model": "unknown", "base_model_relation": "unknown", "Average ⬆️": None},
            {"model_id": "org/phase2", "source_model": "org/base", "base_model_relation": "finetune", "Average ⬆️": 4.0},
        ]
    ).to_csv(contents_path, index=False)
    pd.DataFrame([{"model_id": "org/phase2"}]).to_csv(phase2_path, index=False)

    outputs = build_ablation_outputs(
        contents_path=contents_path,
        phase2_path=phase2_path,
        output_dir=output_dir,
        target_size=3,
        max_eval_missing_fraction=0.34,
    )

    assert (output_dir / "ablation_candidates.csv").exists()
    assert (output_dir / "ablation_selected.csv").exists()
    assert (output_dir / "ablation_rejected.csv").exists()
    assert (output_dir / "ablation_overlap_audit.csv").exists()
    assert (output_dir / "ablation_summary.json").exists()
    assert set(outputs) == {"candidates", "selected", "rejected", "overlap_audit", "summary"}
    assert set(outputs["selected"]["model_id"]) == {"org/eval-a", "org/eval-b", "org/missing"}
    assert outputs["rejected"].loc[outputs["rejected"]["model_id"] == "org/phase2", "rejection_reason"].item() == "phase2_duplicate"

    summary = json.loads((output_dir / "ablation_summary.json").read_text())
    assert summary["selected_rows"] == 3
    assert summary["eval_labelled_rows"] == 2
    assert summary["phase2_overlap_rows"] == 1


def test_build_ablation_outputs_combines_optional_results_source(tmp_path: Path):
    contents_path = tmp_path / "contents.csv"
    results_path = tmp_path / "results.csv"
    phase2_path = tmp_path / "phase2.csv"
    output_dir = tmp_path / "ablation"

    pd.DataFrame([{"model_id": "org/content", "Average ⬆️": 1.0}]).to_csv(contents_path, index=False)
    pd.DataFrame([{"model_id": "org/result", "Average ⬆️": 2.0, "MMLU-PRO": 5.0}]).to_csv(results_path, index=False)
    pd.DataFrame([], columns=["model_id"]).to_csv(phase2_path, index=False)

    outputs = build_ablation_outputs(
        contents_path=contents_path,
        results_path=results_path,
        phase2_path=phase2_path,
        output_dir=output_dir,
        target_size=5,
    )

    assert set(outputs["candidates"]["model_id"]) == {"org/content", "org/result"}
    assert set(outputs["selected"]["eval_source"]) == {"open_llm_leaderboard_contents", "open_llm_leaderboard_results"}


def test_build_ablation_outputs_merges_metadata_path_before_suitability(tmp_path: Path):
    contents_path = tmp_path / "contents.csv"
    metadata_path = tmp_path / "metadata.csv"
    phase2_path = tmp_path / "phase2.csv"
    output_dir = tmp_path / "ablation"

    pd.DataFrame([{"model_id": "org/gguf", "Average ⬆️": 5.0}]).to_csv(contents_path, index=False)
    pd.DataFrame(
        [
            {
                "model_id": "org/gguf",
                "source_model": "org/base",
                "base_model_relation": "quantized",
                "files": "model.gguf,tokenizer.json",
            }
        ]
    ).to_csv(metadata_path, index=False)
    pd.DataFrame([], columns=["model_id"]).to_csv(phase2_path, index=False)

    outputs = build_ablation_outputs(
        contents_path=contents_path,
        metadata_path=metadata_path,
        phase2_path=phase2_path,
        output_dir=output_dir,
        target_size=1,
    )

    assert outputs["selected"].empty
    assert outputs["rejected"].loc[0, "rejection_reason"] == "unsupported_artifact_format"


def test_build_ablation_outputs_requires_phase2_path_before_creating_output_dir(tmp_path: Path):
    contents_path = tmp_path / "contents.csv"
    phase2_path = tmp_path / "missing-phase2.csv"
    output_dir = tmp_path / "ablation"

    pd.DataFrame([{"model_id": "org/content", "Average ⬆️": 1.0}]).to_csv(contents_path, index=False)

    with pytest.raises(FileNotFoundError):
        build_ablation_outputs(
            contents_path=contents_path,
            phase2_path=phase2_path,
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_build_ablation_outputs_requires_contents_path_before_creating_output_dir(tmp_path: Path):
    contents_path = tmp_path / "missing-contents.csv"
    phase2_path = tmp_path / "phase2.csv"
    output_dir = tmp_path / "ablation"

    pd.DataFrame([], columns=["model_id"]).to_csv(phase2_path, index=False)

    with pytest.raises(FileNotFoundError):
        build_ablation_outputs(
            contents_path=contents_path,
            phase2_path=phase2_path,
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_build_ablation_outputs_requires_metadata_path_before_creating_output_dir(tmp_path: Path):
    contents_path = tmp_path / "contents.csv"
    metadata_path = tmp_path / "missing-metadata.csv"
    phase2_path = tmp_path / "phase2.csv"
    output_dir = tmp_path / "ablation"

    pd.DataFrame([{"model_id": "org/content", "Average ⬆️": 1.0}]).to_csv(contents_path, index=False)
    pd.DataFrame([], columns=["model_id"]).to_csv(phase2_path, index=False)

    with pytest.raises(FileNotFoundError):
        build_ablation_outputs(
            contents_path=contents_path,
            metadata_path=metadata_path,
            phase2_path=phase2_path,
            output_dir=output_dir,
        )

    assert not output_dir.exists()
