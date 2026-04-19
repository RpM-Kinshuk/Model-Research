from __future__ import annotations

import importlib.util
import tempfile
import unittest
import warnings
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from alpha_cluster_dashboard.data import AlphaRecord, COMMON_ALPHA_MODULES, schema_name
from spectral_research.prediction import (
    _group_key,
    build_pairwise_score_matrix,
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

    def test_summarize_predictions_skips_missing_targets(self) -> None:
        predictions = pd.DataFrame(
            [
                {"query_bucket": "supported_candidate", "Average ⬆️": 10.0, "prediction": 11.0, "mode": "raw_spectral"},
                {"query_bucket": "supported_candidate", "Average ⬆️": np.nan, "prediction": 13.0, "mode": "raw_spectral"},
                {"query_bucket": "supported_candidate", "Average ⬆️": 14.0, "prediction": 15.0, "mode": "raw_spectral"},
            ]
        )

        summary = summarize_predictions(
            predictions,
            target="Average ⬆️",
            method="pl_alpha_scalar",
            k=1,
            weighted=False,
        )

        self.assertEqual(int(summary.iloc[0]["n"]), 2)
        self.assertTrue(np.isfinite(float(summary.iloc[0]["mae"])))

    def test_summarize_predictions_handles_constant_inputs_without_warning(self) -> None:
        predictions = pd.DataFrame(
            [
                {"query_bucket": "supported_candidate", "Average ⬆️": 10.0, "prediction": 5.0, "mode": "raw_spectral"},
                {"query_bucket": "supported_candidate", "Average ⬆️": 10.0, "prediction": 5.0, "mode": "raw_spectral"},
            ]
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("error")
            summary = summarize_predictions(
                predictions,
                target="Average ⬆️",
                method="pl_alpha_scalar",
                k=1,
                weighted=False,
            )

        self.assertEqual(caught, [])
        self.assertTrue(np.isnan(float(summary.iloc[0]["spearman"])))


class PredictionOptimizationTests(unittest.TestCase):
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

    def test_build_pairwise_score_matrix_returns_symmetric_cpu_matrix(self) -> None:
        matrix, model_index = build_pairwise_score_matrix(
            self._frame(),
            self._records(),
            method="pl_alpha_scalar",
            backend="cpu",
        )

        self.assertEqual(matrix.shape, (4, 4))
        self.assertEqual(set(model_index), {"m1", "m2", "m3", "m4"})
        self.assertTrue(np.allclose(matrix, matrix.T, equal_nan=True))
        self.assertEqual(model_index["m1"], 0)
        self.assertEqual(model_index["m4"], 3)

    def test_cached_neighborhood_predict_oof_matches_uncached_path(self) -> None:
        frame = self._frame()
        records_by_model_id = self._records()
        uncached = neighborhood_predict_oof(
            frame=frame,
            records_by_model_id=records_by_model_id,
            target="Average ⬆️",
            method="pl_alpha_scalar",
            k=1,
            weighted=False,
            residualize=True,
        )

        matrix, model_index = build_pairwise_score_matrix(
            frame,
            records_by_model_id,
            method="pl_alpha_scalar",
            backend="cpu",
        )
        baseline = structural_baseline_oof(frame, target="Average ⬆️")
        baseline_by_model_id = {
            str(model_id): float(prediction)
            for model_id, prediction in zip(
                baseline["model_id"],
                baseline["baseline_prediction"],
                strict=False,
            )
        }

        with patch("spectral_research.prediction._pair_score", side_effect=AssertionError("cache miss")):
            cached = neighborhood_predict_oof(
                frame=frame,
                records_by_model_id=records_by_model_id,
                target="Average ⬆️",
                method="pl_alpha_scalar",
                k=1,
                weighted=False,
                residualize=True,
                pairwise_scores=matrix,
                model_index=model_index,
                baseline_by_model_id=baseline_by_model_id,
            )

        np.testing.assert_allclose(
            cached["prediction"].to_numpy(dtype=float),
            uncached["prediction"].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
        )

    def test_cached_full_module_trace_keeps_legitimate_nan_scores(self) -> None:
        frame = pd.DataFrame(
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
                    "Average ⬆️": 20.0,
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
                    "Average ⬆️": 30.0,
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
                    "Average ⬆️": 40.0,
                },
                {
                    "model_id": "m5",
                    "display_name": "repo/m5",
                    "audited_base_family": "fam/c",
                    "candidate_family": "fam/c",
                    "cv_group": "fam/c",
                    "label_status": "confirmed",
                    "query_bucket": "supported_candidate",
                    "num_layers": 2,
                    "raw_module_count": 2,
                    "common_schema": True,
                    "schema_name": "common",
                    "Average ⬆️": 50.0,
                },
                {
                    "model_id": "m6",
                    "display_name": "repo/m6",
                    "audited_base_family": "fam/c",
                    "candidate_family": "fam/c",
                    "cv_group": "fam/c",
                    "label_status": "confirmed",
                    "query_bucket": "supported_candidate",
                    "num_layers": 2,
                    "raw_module_count": 2,
                    "common_schema": True,
                    "schema_name": "common",
                    "Average ⬆️": 60.0,
                },
            ]
        )
        records_by_model_id = {
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
                ("self_attn.k_proj", "mlp.up_proj"),
                np.array([[3.0, 4.0], [3.0, 4.0]]),
            ),
            "m3": AlphaRecord(
                "m3",
                "repo/m3",
                "/tmp/m3",
                2,
                ("self_attn.q_proj", "mlp.down_proj"),
                np.array([[1.1, 2.1], [1.1, 2.1]]),
            ),
            "m4": AlphaRecord(
                "m4",
                "repo/m4",
                "/tmp/m4",
                2,
                ("self_attn.v_proj", "mlp.gate_proj"),
                np.array([[5.0, 6.0], [5.0, 6.0]]),
            ),
            "m5": AlphaRecord(
                "m5",
                "repo/m5",
                "/tmp/m5",
                2,
                ("self_attn.k_proj", "mlp.up_proj"),
                np.array([[3.1, 4.1], [3.1, 4.1]]),
            ),
            "m6": AlphaRecord(
                "m6",
                "repo/m6",
                "/tmp/m6",
                2,
                ("self_attn.v_proj", "mlp.gate_proj"),
                np.array([[5.1, 6.1], [5.1, 6.1]]),
            ),
        }

        uncached = neighborhood_predict_oof(
            frame=frame,
            records_by_model_id=records_by_model_id,
            target="Average ⬆️",
            method="full_module_trace",
            k=1,
            weighted=False,
            residualize=False,
        )
        matrix, model_index = build_pairwise_score_matrix(
            frame,
            records_by_model_id,
            method="full_module_trace",
            backend="cpu",
        )
        self.assertTrue(np.isnan(matrix[model_index["m1"], model_index["m4"]]))
        self.assertTrue(np.isfinite(matrix[model_index["m1"], model_index["m3"]]))

        with patch("spectral_research.prediction._pair_score", side_effect=AssertionError("cache miss")):
            cached = neighborhood_predict_oof(
                frame=frame,
                records_by_model_id=records_by_model_id,
                target="Average ⬆️",
                method="full_module_trace",
                k=1,
                weighted=False,
                residualize=False,
                pairwise_scores=matrix,
                model_index=model_index,
            )

        np.testing.assert_allclose(
            cached["prediction"].to_numpy(dtype=float),
            uncached["prediction"].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )


class PredictionCliTests(unittest.TestCase):
    def _load_module(self):
        script_path = (
            Path(__file__).resolve().parents[1] / "scripts" / "spectral_prediction_research.py"
        )
        spec = importlib.util.spec_from_file_location("spectral_prediction_research_cli", script_path)

        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_prediction_cli_accepts_freeze_csv_targets_and_output_dir(self) -> None:
        module = self._load_module()

        args = module.parse_args(
            [
                "--freeze-csv",
                "freeze.csv",
                "--targets",
                "Average ⬆️,MMLU-PRO,MATH Lvl 5",
                "--output-dir",
                "out",
            ]
        )

        self.assertEqual(args.freeze_csv, Path("freeze.csv"))
        self.assertEqual(args.targets, "Average ⬆️,MMLU-PRO,MATH Lvl 5")
        self.assertEqual(args.output_dir, Path("out"))

    def test_render_report_includes_best_rows_table_and_notes(self) -> None:
        module = self._load_module()

        results = pd.DataFrame(
            [
                {
                    "target": "Average ⬆️",
                    "method": "structural_basics_only",
                    "mode": "baseline",
                    "bucket": "supported_candidate",
                    "k": 0,
                    "weighted": False,
                    "n": 10,
                    "r2": 0.1,
                    "mae": 1.0,
                    "spearman": 0.2,
                },
                {
                    "target": "Average ⬆️",
                    "method": "role_aggregated_trace",
                    "mode": "spectral_plus_structural",
                    "bucket": "supported_candidate",
                    "k": 5,
                    "weighted": True,
                    "n": 10,
                    "r2": 0.3,
                    "mae": 0.8,
                    "spearman": 0.4,
                },
                {
                    "target": "MMLU-PRO",
                    "method": "pl_alpha_scalar",
                    "mode": "raw_spectral",
                    "bucket": "heterogeneous",
                    "k": 3,
                    "weighted": False,
                    "n": 8,
                    "r2": 0.2,
                    "mae": 1.2,
                    "spearman": 0.5,
                },
            ]
        )

        report = module.render_report(results)

        self.assertIn("# Performance Prediction Cycle 1", report)
        self.assertIn("## Best rows by target", report)
        self.assertIn("role_aggregated_trace", report)
        self.assertIn("- Raw spectral rows measure direct neighborhood signal.", report)

    def test_prediction_cli_rejects_invalid_cv_splits(self) -> None:
        module = self._load_module()

        with self.assertRaises(SystemExit):
            module.parse_args(["--cv-splits", "1"])

    def test_main_rejects_blank_targets_cleanly(self) -> None:
        module = self._load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(
                metrics_dir=Path("metrics"),
                leaderboard_csv=Path("leaderboard.csv"),
                lineage_csv=Path("lineage.csv"),
                freeze_csv=Path("freeze.csv"),
                output_dir=Path(tmpdir),
                targets=" , ",
                cv_splits=5,
            )
            with patch.object(module, "parse_args", return_value=args):
                with self.assertRaisesRegex(SystemExit, "No prediction targets specified"):
                    module.main()

    def test_main_writes_stable_outputs_for_empty_dataset(self) -> None:
        module = self._load_module()

        empty_dataset = pd.DataFrame(
            columns=["model_id", "query_bucket", "Average ⬆️", "MMLU-PRO", "MATH Lvl 5"]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            args = Namespace(
                metrics_dir=Path("metrics"),
                leaderboard_csv=Path("leaderboard.csv"),
                lineage_csv=Path("lineage.csv"),
                freeze_csv=Path("freeze.csv"),
                output_dir=output_dir,
                targets="Average ⬆️,MMLU-PRO",
                cv_splits=5,
            )
            with patch.object(module, "parse_args", return_value=args), patch.object(
                module,
                "build_prediction_dataset",
                return_value=(empty_dataset, {}),
            ):
                module.main()

            self.assertTrue((output_dir / "prediction_dataset.csv").exists())
            self.assertTrue((output_dir / "prediction_results.csv").exists())
            self.assertTrue((output_dir / "oof_predictions.csv").exists())
            oof_predictions = pd.read_csv(output_dir / "oof_predictions.csv")
            self.assertIn("Average ⬆️", oof_predictions.columns)
            self.assertIn("MMLU-PRO", oof_predictions.columns)
            self.assertEqual(
                [column for column in oof_predictions.columns if column not in {"Average ⬆️", "MMLU-PRO"}],
                ["model_id", "query_bucket", "prediction", "mode", "target", "method", "k", "weighted"],
            )
            self.assertEqual(len(oof_predictions), 0)
            report = (output_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("# Performance Prediction Cycle 1", report)
            self.assertIn("_No rows._", report)
