from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from alpha_cluster_dashboard.data import AlphaRecord, COMMON_ALPHA_MODULES, schema_name
from spectral_research.prediction import (
    _group_key,
    join_prediction_targets,
    neighborhood_predict_oof,
    summarize_predictions,
    structural_baseline_oof,
)


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

    def test_join_prediction_targets_uses_best_duplicate_leaderboard_row(self) -> None:
        manifest = pd.DataFrame(
            [
                {
                    "model_id": "repo--model-c",
                    "display_name": "repo/model-c",
                    "audited_base_family": "fam/c",
                    "candidate_family": "fam/c",
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
                    "atlas_repo_id": "repo/model-c",
                    "fullname": "repo/model-c",
                    "Average ⬆️": 10.0,
                    "MMLU-PRO": 11.0,
                    "MATH Lvl 5": pd.NA,
                },
                {
                    "atlas_repo_id": "repo/model-c",
                    "fullname": "repo/model-c",
                    "Average ⬆️": 12.0,
                    "MMLU-PRO": 13.0,
                    "MATH Lvl 5": 14.0,
                },
            ]
        )

        joined = join_prediction_targets(manifest, leaderboard)

        self.assertEqual(float(joined.iloc[0]["Average ⬆️"]), 12.0)
        self.assertEqual(float(joined.iloc[0]["MMLU-PRO"]), 13.0)
        self.assertEqual(float(joined.iloc[0]["MATH Lvl 5"]), 14.0)

    def test_group_key_prefers_audited_family_then_candidate_then_model(self) -> None:
        row_a = pd.Series(
            {"model_id": "m-a", "audited_base_family": "fam/a", "candidate_family": "cand/a"}
        )
        row_b = pd.Series({"model_id": "m-b", "audited_base_family": "", "candidate_family": "cand/b"})
        row_c = pd.Series({"model_id": "m-c", "audited_base_family": "", "candidate_family": ""})

        self.assertEqual(_group_key(row_a), "fam/a")
        self.assertEqual(_group_key(row_b), "cand/b")
        self.assertEqual(_group_key(row_c), "m-c")

    def test_build_prediction_dataset_uses_local_schema_labeling(self) -> None:
        records = [
            AlphaRecord(
                model_id="common-a",
                display_name="repo/common-a",
                path="/tmp/common-a.h5",
                num_layers=2,
                raw_modules=COMMON_ALPHA_MODULES,
                alpha=np.array([[1.0] * len(COMMON_ALPHA_MODULES), [1.0] * len(COMMON_ALPHA_MODULES)]),
            ),
            AlphaRecord(
                model_id="hetero-a",
                display_name="repo/hetero-a",
                path="/tmp/hetero-a.h5",
                num_layers=2,
                raw_modules=("self_attn.q_proj", "mlp.up_proj"),
                alpha=np.array([[1.0, 2.0], [1.0, 2.0]]),
            ),
        ]
        leaderboard = pd.DataFrame(
            [
                {
                    "atlas_repo_id": "repo/common-a",
                    "fullname": "repo/common-a",
                    "Average ⬆️": 1.0,
                    "MMLU-PRO": 2.0,
                    "MATH Lvl 5": 3.0,
                },
                {
                    "atlas_repo_id": "repo/hetero-a",
                    "fullname": "repo/hetero-a",
                    "Average ⬆️": 4.0,
                    "MMLU-PRO": 5.0,
                    "MATH Lvl 5": 6.0,
                },
            ]
        )
        captured = {}

        def fake_build_audit_manifest(records_frame, metadata_frame, leaderboard_frame, overrides_frame):
            captured["records_frame"] = records_frame.copy()
            return records_frame.copy()

        with patch("spectral_research.prediction.load_alpha_records", return_value=records), patch(
            "spectral_research.prediction.build_audit_manifest",
            side_effect=fake_build_audit_manifest,
        ), patch("spectral_research.prediction._read_csv", side_effect=[pd.DataFrame(), leaderboard, pd.DataFrame()]):
            from spectral_research.prediction import build_prediction_dataset

            joined, records_by_model_id = build_prediction_dataset(
                metrics_dir="/tmp/metrics",
                leaderboard_csv="/tmp/leaderboard.csv",
                lineage_csv="/tmp/lineage.csv",
                freeze_csv="/tmp/freeze.csv",
            )

        records_frame = captured["records_frame"]
        self.assertEqual(records_frame.loc[records_frame["model_id"].eq("common-a"), "schema_name"].iloc[0], "common_llm_7")
        self.assertEqual(
            records_frame.loc[records_frame["model_id"].eq("hetero-a"), "schema_name"].iloc[0],
            schema_name(records[1]),
        )
        self.assertEqual(set(records_by_model_id), {"common-a", "hetero-a"})
        self.assertEqual(len(joined), 2)


class PredictionModelTests(unittest.TestCase):
    def _records(self) -> dict[str, AlphaRecord]:
        return {
            "m1": AlphaRecord(
                "m1",
                "repo/m1",
                "/tmp/m1",
                2,
                ("self_attn.q_proj", "mlp.down_proj"),
                np.array([[1.0, 2.0], [1.0, 2.0]]),
            ),
            "m2": AlphaRecord(
                "m2",
                "repo/m2",
                "/tmp/m2",
                2,
                ("self_attn.q_proj", "mlp.down_proj"),
                np.array([[1.1, 2.1], [1.1, 2.1]]),
            ),
            "m3": AlphaRecord(
                "m3",
                "repo/m3",
                "/tmp/m3",
                2,
                ("self_attn.q_proj", "mlp.down_proj"),
                np.array([[9.0, 9.0], [9.0, 9.0]]),
            ),
            "m4": AlphaRecord(
                "m4",
                "repo/m4",
                "/tmp/m4",
                2,
                ("self_attn.q_proj", "mlp.down_proj"),
                np.array([[9.2, 9.1], [9.1, 9.2]]),
            ),
        }

    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "model_id": "m1",
                    "display_name": "repo/m1",
                    "audited_base_family": "fam/a",
                    "candidate_family": "fam/a",
                    "cv_group": "fam/a",
                    "label_status": "confirmed",
                    "query_bucket": "supported_candidate",
                    "num_layers": 2,
                    "raw_module_count": 2,
                    "common_schema": True,
                    "schema_name": "common",
                    "Average ⬆️": 10.0,
                },
                {
                    "model_id": "m2",
                    "display_name": "repo/m2",
                    "audited_base_family": "fam/a",
                    "candidate_family": "fam/a",
                    "cv_group": "fam/a",
                    "label_status": "confirmed",
                    "query_bucket": "supported_candidate",
                    "num_layers": 2,
                    "raw_module_count": 2,
                    "common_schema": True,
                    "schema_name": "common",
                    "Average ⬆️": 12.0,
                },
                {
                    "model_id": "m3",
                    "display_name": "repo/m3",
                    "audited_base_family": "fam/b",
                    "candidate_family": "fam/b",
                    "cv_group": "fam/b",
                    "label_status": "provisional",
                    "query_bucket": "heterogeneous",
                    "num_layers": 2,
                    "raw_module_count": 2,
                    "common_schema": False,
                    "schema_name": "bert_like",
                    "Average ⬆️": 90.0,
                },
                {
                    "model_id": "m4",
                    "display_name": "repo/m4",
                    "audited_base_family": "fam/b",
                    "candidate_family": "fam/b",
                    "cv_group": "fam/b",
                    "label_status": "provisional",
                    "query_bucket": "heterogeneous",
                    "num_layers": 2,
                    "raw_module_count": 2,
                    "common_schema": False,
                    "schema_name": "bert_like",
                    "Average ⬆️": 92.0,
                },
            ]
        )

    def test_neighborhood_predict_oof_returns_predictions_for_role_aggregated_trace(self) -> None:
        predictions = neighborhood_predict_oof(
            frame=self._frame(),
            records_by_model_id=self._records(),
            target="Average ⬆️",
            method="role_aggregated_trace",
            k=1,
            weighted=False,
            residualize=False,
        )

        self.assertEqual(len(predictions), 4)
        self.assertTrue(predictions["prediction"].notna().all())

    def test_structural_baseline_oof_returns_prediction_column(self) -> None:
        baseline = structural_baseline_oof(self._frame(), target="Average ⬆️")

        self.assertEqual(len(baseline), 4)
        self.assertIn("baseline_prediction", baseline.columns)

    def test_summarize_predictions_accepts_structural_baseline_output(self) -> None:
        baseline = structural_baseline_oof(self._frame(), target="Average ⬆️")

        summary = summarize_predictions(
            baseline,
            target="Average ⬆️",
            method="structural_basics_only",
            k=0,
            weighted=False,
        )

        self.assertEqual(set(summary["bucket"]), {"supported_candidate", "heterogeneous"})
        self.assertTrue(summary["mae"].notna().all())
