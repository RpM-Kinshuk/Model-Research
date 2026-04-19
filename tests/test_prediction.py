from __future__ import annotations

import unittest

import pandas as pd

from spectral_research.prediction import _group_key, join_prediction_targets


class PredictionDatasetTests(unittest.TestCase):
    def test_join_prediction_targets_uses_display_name_to_atlas_repo_id(self) -> None:
        manifest = pd.DataFrame(
            [
                {
                    "model_id": "repo--model-a",
                    "display_name": "repo/model-a",
                    "audited_base_family": "fam/a",
                    "candidate_family": "fam/a",
                    "label_status": "confirmed",
                    "query_bucket": "supported_candidate",
                    "num_layers": 24,
                    "raw_module_count": 7,
                    "common_schema": True,
                    "schema_name": "common_llm_7",
                }
            ]
        )
        leaderboard = pd.DataFrame(
            [
                {
                    "atlas_repo_id": "repo/model-a",
                    "fullname": "irrelevant",
                    "Average ⬆️": 33.0,
                    "MMLU-PRO": 31.0,
                    "MATH Lvl 5": 12.0,
                }
            ]
        )

        joined = join_prediction_targets(manifest, leaderboard)

        self.assertEqual(len(joined), 1)
        self.assertEqual(float(joined.iloc[0]["Average ⬆️"]), 33.0)

    def test_join_prediction_targets_falls_back_to_fullname(self) -> None:
        manifest = pd.DataFrame(
            [
                {
                    "model_id": "repo--model-b",
                    "display_name": "repo/model-b",
                    "audited_base_family": "",
                    "candidate_family": "fam/b",
                    "label_status": "provisional",
                    "query_bucket": "heterogeneous",
                    "num_layers": 32,
                    "raw_module_count": 5,
                    "common_schema": False,
                    "schema_name": "bert_like_5",
                }
            ]
        )
        leaderboard = pd.DataFrame(
            [
                {
                    "atlas_repo_id": "other/repo",
                    "fullname": "repo/model-b",
                    "Average ⬆️": 27.0,
                    "MMLU-PRO": 20.0,
                    "MATH Lvl 5": 4.0,
                }
            ]
        )

        joined = join_prediction_targets(manifest, leaderboard)

        self.assertEqual(len(joined), 1)
        self.assertEqual(float(joined.iloc[0]["MMLU-PRO"]), 20.0)

    def test_group_key_prefers_audited_family_then_candidate_then_model(self) -> None:
        row_a = pd.Series(
            {"model_id": "m-a", "audited_base_family": "fam/a", "candidate_family": "cand/a"}
        )
        row_b = pd.Series({"model_id": "m-b", "audited_base_family": "", "candidate_family": "cand/b"})
        row_c = pd.Series({"model_id": "m-c", "audited_base_family": "", "candidate_family": ""})

        self.assertEqual(_group_key(row_a), "fam/a")
        self.assertEqual(_group_key(row_b), "cand/b")
        self.assertEqual(_group_key(row_c), "m-c")
