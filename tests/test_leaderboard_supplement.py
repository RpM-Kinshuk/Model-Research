import pandas as pd

from fetch_data.leaderboard_supplement import (
    build_leaderboard_repo_ids,
    leaderboard_frame_to_candidates,
)


def test_build_leaderboard_repo_ids_prefers_model_and_falls_back_to_base_or_eval_name():
    frame = pd.DataFrame(
        [
            {"Model": '<a href="https://huggingface.co/org/model">org/model</a>', "Base Model": None, "eval_name": None},
            {"Model": None, "Base Model": "base/model", "eval_name": None},
            {"Model": None, "Base Model": None, "eval_name": "org_model_fp16"},
        ]
    )
    repo_ids = build_leaderboard_repo_ids(frame)
    assert list(repo_ids) == ["org/model", "base/model", "org/model"]


def test_leaderboard_frame_to_candidates_creates_eval_rich_rows():
    frame = pd.DataFrame(
        [
            {
                "Model": "org/model-chat",
                "Base Model": "org/model-base",
                "Type": "chat",
                "Merged": False,
                "Architecture": "LlamaForCausalLM",
                "#Params (B)": 8.0,
                "Average ⬆️": 42.0,
                "IFEval": 41.0,
                "BBH": 43.0,
            },
            {
                "Model": "org/model-merge",
                "Base Model": "org/model-base",
                "Type": "chat",
                "Merged": True,
                "Architecture": "LlamaForCausalLM",
                "#Params (B)": 8.0,
                "Average ⬆️": 39.0,
                "IFEval": 38.0,
                "BBH": 40.0,
            },
        ]
    )
    candidates = leaderboard_frame_to_candidates(frame)
    assert set(candidates["model_id"]) == {"org/model-chat", "org/model-merge"}
    assert set(candidates["candidate_source"]) == {"leaderboard_dataset"}
    assert candidates.loc[candidates["model_id"] == "org/model-chat", "base_model_relation"].item() == "finetune"
    assert candidates.loc[candidates["model_id"] == "org/model-merge", "base_model_relation"].item() == "merge"
    assert candidates["Average ⬆️"].notna().all()
    assert candidates[["IFEval", "BBH"]].notna().all().all()


def test_leaderboard_candidate_rejects_displayed_self_parent():
    frame = pd.DataFrame(
        [
            {
                "Model": "org/model-merge",
                "Base Model": "org/model-merge (Merge)",
                "Type": "base merges and moerges",
                "Merged": False,
                "Average ⬆️": 39.0,
            }
        ]
    )
    candidate = leaderboard_frame_to_candidates(frame).iloc[0]
    assert candidate["source_model"] == ""
    assert candidate["base_model_relation"] == "merge"
