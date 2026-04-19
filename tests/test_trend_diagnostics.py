from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from alpha_cluster_dashboard.data import AlphaRecord
import scripts.performance_trend_diagnostics as performance_trend_diagnostics
from spectral_research.trend_diagnostics import (
    build_absolute_trend_frame,
    build_exact_parent_delta_frame,
    build_family_relative_delta_frame,
    analyze_trend_rows,
    render_trend_report,
)


class TrendDiagnosticsTableTests(unittest.TestCase):
    def _records(self) -> dict[str, AlphaRecord]:
        return {
            "parent-a": AlphaRecord(
                "parent-a",
                "repo/parent-a",
                "/tmp/parent-a",
                2,
                ("self_attn.q_proj", "mlp.down_proj"),
                np.array([[1.0, 2.0], [2.0, 3.0]], dtype=float),
            ),
            "child-a": AlphaRecord(
                "child-a",
                "repo/child-a",
                "/tmp/child-a",
                2,
                ("self_attn.q_proj", "mlp.down_proj"),
                np.array([[2.0, 3.0], [4.0, 5.0]], dtype=float),
            ),
            "peer-b": AlphaRecord(
                "peer-b",
                "repo/peer-b",
                "/tmp/peer-b",
                2,
                ("self_attn.q_proj", "mlp.down_proj"),
                np.array([[0.5, 1.5], [1.5, 2.5]], dtype=float),
            ),
        }

    def _dataset(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "model_id": "parent-a",
                    "display_name": "repo/parent-a",
                    "source_model": "",
                    "base_model_relation": "",
                    "audited_base_family": "fam/a",
                    "candidate_family": "fam/a",
                    "query_bucket": "supported_candidate",
                    "common_schema": True,
                    "schema_name": "common_llm_7",
                    "num_layers": 2,
                    "raw_module_count": 2,
                    "Average ⬆️": 10.0,
                    "MMLU-PRO": 11.0,
                    "MATH Lvl 5": 12.0,
                },
                {
                    "model_id": "child-a",
                    "display_name": "repo/child-a",
                    "source_model": "parent-a",
                    "base_model_relation": "finetune",
                    "audited_base_family": "fam/a",
                    "candidate_family": "fam/a",
                    "query_bucket": "supported_candidate",
                    "common_schema": True,
                    "schema_name": "common_llm_7",
                    "num_layers": 2,
                    "raw_module_count": 2,
                    "Average ⬆️": 15.0,
                    "MMLU-PRO": 16.0,
                    "MATH Lvl 5": 20.0,
                },
                {
                    "model_id": "peer-b",
                    "display_name": "repo/peer-b",
                    "source_model": "",
                    "base_model_relation": "",
                    "audited_base_family": "fam/a",
                    "candidate_family": "fam/a",
                    "query_bucket": "heterogeneous",
                    "common_schema": False,
                    "schema_name": "bert_like_5",
                    "num_layers": 2,
                    "raw_module_count": 2,
                    "Average ⬆️": 7.0,
                    "MMLU-PRO": 8.0,
                    "MATH Lvl 5": 6.0,
                },
            ]
        )

    def test_build_absolute_trend_frame_adds_global_level_start_end_and_delta(self) -> None:
        feature_frame = build_absolute_trend_frame(self._dataset(), self._records())

        row = feature_frame.loc[feature_frame["model_id"].eq("child-a")].iloc[0]
        self.assertAlmostEqual(float(row["trend_global_mean"]), 3.5)
        self.assertAlmostEqual(float(row["trend_global_start"]), 2.5)
        self.assertAlmostEqual(float(row["trend_global_end"]), 4.5)
        self.assertAlmostEqual(float(row["trend_global_end_minus_start"]), 2.0)

    def test_build_exact_parent_delta_frame_uses_direct_source_model_when_present(self) -> None:
        feature_frame = build_absolute_trend_frame(self._dataset(), self._records())
        delta_frame = build_exact_parent_delta_frame(feature_frame)

        self.assertEqual(len(delta_frame), 1)
        row = delta_frame.iloc[0]
        self.assertEqual(row["model_id"], "child-a")
        self.assertEqual(row["parent_model_id"], "parent-a")
        self.assertAlmostEqual(float(row["trend_global_mean_delta"]), 1.5)
        self.assertAlmostEqual(float(row["MMLU-PRO_delta"]), 5.0)

    def test_build_exact_parent_delta_frame_resolves_slash_form_source_model_with_normalized_ids(self) -> None:
        dataset = pd.DataFrame(
            [
                {
                    "model_id": "repo--parent-a",
                    "display_name": "repo/parent-a",
                    "source_model": "",
                    "base_model_relation": "",
                    "audited_base_family": "fam/a",
                    "candidate_family": "fam/a",
                    "query_bucket": "supported_candidate",
                    "common_schema": True,
                    "schema_name": "common_llm_7",
                    "num_layers": 2,
                    "raw_module_count": 2,
                    "Average ⬆️": 10.0,
                    "MMLU-PRO": 11.0,
                    "MATH Lvl 5": 12.0,
                },
                {
                    "model_id": "repo--child-a",
                    "display_name": "repo/child-a",
                    "source_model": "repo/parent-a",
                    "base_model_relation": "finetune",
                    "audited_base_family": "fam/a",
                    "candidate_family": "fam/a",
                    "query_bucket": "supported_candidate",
                    "common_schema": True,
                    "schema_name": "common_llm_7",
                    "num_layers": 2,
                    "raw_module_count": 2,
                    "Average ⬆️": 15.0,
                    "MMLU-PRO": 16.0,
                    "MATH Lvl 5": 20.0,
                },
            ]
        )
        records = {
            "repo--parent-a": AlphaRecord(
                "repo--parent-a",
                "repo/parent-a",
                "/tmp/repo--parent-a",
                2,
                ("self_attn.q_proj", "mlp.down_proj"),
                np.array([[1.0, 2.0], [2.0, 3.0]], dtype=float),
            ),
            "repo--child-a": AlphaRecord(
                "repo--child-a",
                "repo/child-a",
                "/tmp/repo--child-a",
                2,
                ("self_attn.q_proj", "mlp.down_proj"),
                np.array([[2.0, 3.0], [4.0, 5.0]], dtype=float),
            ),
        }

        feature_frame = build_absolute_trend_frame(dataset, records)
        delta_frame = build_exact_parent_delta_frame(feature_frame)

        self.assertEqual(len(delta_frame), 1)
        row = delta_frame.iloc[0]
        self.assertEqual(row["model_id"], "repo--child-a")
        self.assertEqual(row["source_model"], "repo/parent-a")
        self.assertEqual(row["parent_model_id"], "repo--parent-a")
        self.assertAlmostEqual(float(row["MMLU-PRO_delta"]), 5.0)
        self.assertAlmostEqual(float(row["trend_global_mean_delta"]), 1.5)

    def test_build_family_relative_delta_frame_uses_family_median_anchor(self) -> None:
        feature_frame = build_absolute_trend_frame(self._dataset(), self._records())
        delta_frame = build_family_relative_delta_frame(feature_frame)

        row = delta_frame.loc[delta_frame["model_id"].eq("child-a")].iloc[0]
        self.assertEqual(row["family_anchor"], "fam/a")
        self.assertAlmostEqual(float(row["MMLU-PRO_delta"]), 5.0)
        self.assertIn("trend_global_end_minus_start_delta", delta_frame.columns)

    def test_build_absolute_trend_frame_returns_stable_empty_schema_when_no_records_match(self) -> None:
        dataset = self._dataset().assign(model_id=["missing-a", "missing-b", "missing-c"])

        feature_frame = build_absolute_trend_frame(dataset, self._records())

        self.assertEqual(len(feature_frame), 0)
        self.assertIn("model_id", feature_frame.columns)
        self.assertIn("trend_global_mean", feature_frame.columns)
        self.assertIn("trend_global_end_minus_start", feature_frame.columns)

    def test_build_exact_parent_delta_frame_returns_stable_empty_schema_when_no_pairs_exist(self) -> None:
        feature_frame = build_absolute_trend_frame(self._dataset(), self._records()).loc[
            lambda frame: frame["model_id"].eq("parent-a")
        ]

        delta_frame = build_exact_parent_delta_frame(feature_frame)

        self.assertEqual(len(delta_frame), 0)
        self.assertIn("parent_model_id", delta_frame.columns)
        self.assertIn("trend_global_mean_delta", delta_frame.columns)
        self.assertIn("MATH Lvl 5_delta", delta_frame.columns)

    def test_build_family_relative_delta_frame_normalizes_family_keys_and_uses_public_schema(self) -> None:
        dataset = pd.DataFrame(
            [
                {
                    "model_id": "family-a-1",
                    "display_name": "repo/family-a-1",
                    "source_model": "",
                    "base_model_relation": "",
                    "audited_base_family": " fam/a ",
                    "candidate_family": "",
                    "query_bucket": "supported_candidate",
                    "common_schema": True,
                    "schema_name": "common_llm_7",
                    "num_layers": 2,
                    "raw_module_count": 2,
                    "Average ⬆️": 10.0,
                    "MMLU-PRO": 20.0,
                    "MATH Lvl 5": 30.0,
                },
                {
                    "model_id": "family-a-2",
                    "display_name": "repo/family-a-2",
                    "source_model": "",
                    "base_model_relation": "",
                    "audited_base_family": "fam/a",
                    "candidate_family": "  fam/a  ",
                    "query_bucket": "supported_candidate",
                    "common_schema": True,
                    "schema_name": "common_llm_7",
                    "num_layers": 2,
                    "raw_module_count": 2,
                    "Average ⬆️": 14.0,
                    "MMLU-PRO": 22.0,
                    "MATH Lvl 5": 34.0,
                },
            ]
        )
        records = {
            "family-a-1": AlphaRecord(
                "family-a-1",
                "repo/family-a-1",
                "/tmp/family-a-1",
                2,
                ("self_attn.q_proj", "mlp.down_proj"),
                np.array([[1.0, 2.0], [2.0, 4.0]], dtype=float),
            ),
            "family-a-2": AlphaRecord(
                "family-a-2",
                "repo/family-a-2",
                "/tmp/family-a-2",
                2,
                ("self_attn.q_proj", "mlp.down_proj"),
                np.array([[2.0, 3.0], [4.0, 5.0]], dtype=float),
            ),
        }

        feature_frame = build_absolute_trend_frame(dataset, records)
        delta_frame = build_family_relative_delta_frame(feature_frame)

        self.assertEqual(len(delta_frame), 2)
        self.assertTrue((delta_frame["family_anchor"] == "fam/a").all())
        self.assertNotIn("_family_anchor", delta_frame.columns)
        self.assertIn("trend_global_mean_delta", delta_frame.columns)

    def test_build_family_relative_delta_frame_returns_stable_empty_schema_when_no_usable_family_exists(
        self,
    ) -> None:
        dataset = pd.DataFrame(
            [
                {
                    "model_id": "lonely-a",
                    "display_name": "repo/lonely-a",
                    "source_model": "",
                    "base_model_relation": "",
                    "audited_base_family": "fam/a",
                    "candidate_family": "fam/a",
                    "query_bucket": "supported_candidate",
                    "common_schema": True,
                    "schema_name": "common_llm_7",
                    "num_layers": 2,
                    "raw_module_count": 2,
                    "Average ⬆️": 10.0,
                    "MMLU-PRO": 20.0,
                    "MATH Lvl 5": 30.0,
                }
            ]
        )
        records = {
            "lonely-a": AlphaRecord(
                "lonely-a",
                "repo/lonely-a",
                "/tmp/lonely-a",
                2,
                ("self_attn.q_proj", "mlp.down_proj"),
                np.array([[1.0, 2.0], [2.0, 4.0]], dtype=float),
            )
        }

        feature_frame = build_absolute_trend_frame(dataset, records)
        delta_frame = build_family_relative_delta_frame(feature_frame)

        self.assertEqual(len(delta_frame), 0)
        self.assertIn("family_anchor", delta_frame.columns)
        self.assertIn("Average ⬆️_delta", delta_frame.columns)


class TrendDiagnosticsAnalysisTests(unittest.TestCase):
    def test_analyze_trend_rows_returns_descriptive_spearman_only(self) -> None:
        frame = pd.DataFrame(
            {
                "model_id": ["a", "b", "c", "d"],
                "query_bucket": [
                    "supported_candidate",
                    "supported_candidate",
                    "heterogeneous",
                    "heterogeneous",
                ],
                "trend_global_end_minus_start": [0.0, 1.0, 2.0, 3.0],
                "MMLU-PRO": [10.0, 20.0, 30.0, 40.0],
                "num_layers": [2, 2, 2, 2],
                "raw_module_count": [2, 2, 2, 2],
                "common_schema": [True, True, False, False],
                "schema_name": ["common", "common", "bert_like", "bert_like"],
            }
        )

        results = analyze_trend_rows(
            frame,
            feature_columns=["trend_global_end_minus_start"],
            target_columns=["MMLU-PRO"],
            analysis_name="absolute",
            bucket_name="all_scored",
            use_structural_controls=False,
        )

        self.assertEqual(len(results), 1)
        row = results.iloc[0]
        self.assertEqual(row["analysis"], "absolute")
        self.assertEqual(row["bucket"], "all_scored")
        self.assertEqual(row["feature"], "trend_global_end_minus_start")
        self.assertAlmostEqual(float(row["spearman"]), 1.0)
        self.assertEqual(
            list(results.columns),
            ["analysis", "bucket", "target", "feature", "control_mode", "n", "spearman"],
        )

    def test_analyze_trend_rows_structural_adjusted_spearman_residualizes_controls(self) -> None:
        frame = pd.DataFrame(
            {
                "model_id": [f"model-{index}" for index in range(8)],
                "query_bucket": ["supported_candidate"] * 8,
                "trend_global_end_minus_start": [0.9, 1.1, 1.9, 2.1, 2.9, 3.1, 3.9, 4.1],
                "MMLU-PRO": [1.1, 0.9, 2.1, 1.9, 3.1, 2.9, 4.1, 3.9],
                "num_layers": [1, 1, 2, 2, 3, 3, 4, 4],
                "raw_module_count": [10, 10, 20, 20, 30, 30, 40, 40],
                "common_schema": [True] * 8,
                "schema_name": ["common"] * 8,
            }
        )

        raw = analyze_trend_rows(
            frame,
            feature_columns=["trend_global_end_minus_start"],
            target_columns=["MMLU-PRO"],
            analysis_name="absolute",
            bucket_name="all_scored",
            use_structural_controls=False,
        )
        adjusted = analyze_trend_rows(
            frame,
            feature_columns=["trend_global_end_minus_start"],
            target_columns=["MMLU-PRO"],
            analysis_name="absolute",
            bucket_name="all_scored",
            use_structural_controls=True,
        )

        self.assertGreater(float(raw.iloc[0]["spearman"]), 0.8)
        self.assertLess(float(adjusted.iloc[0]["spearman"]), -0.8)

    def test_render_trend_report_includes_three_sections(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "analysis": "absolute",
                    "bucket": "all_scored",
                    "target": "MMLU-PRO",
                    "feature": "trend_global_end_minus_start",
                    "control_mode": "raw",
                    "n": 12,
                    "spearman": 0.5,
                },
                {
                    "analysis": "exact_parent",
                    "bucket": "all_scored",
                    "target": "MMLU-PRO",
                    "feature": "trend_global_end_minus_start_delta",
                    "control_mode": "raw",
                    "n": 7,
                    "spearman": 0.4,
                },
                {
                    "analysis": "family_relative",
                    "bucket": "heterogeneous",
                    "target": "MATH Lvl 5",
                    "feature": "trend_global_end_minus_start_delta",
                    "control_mode": "raw",
                    "n": 9,
                    "spearman": 0.3,
                },
            ]
        )

        report = render_trend_report(rows)

        self.assertIn("# Performance Trend Diagnostics", report)
        self.assertIn("## Top Absolute-Trend Rows By Target", report)
        self.assertIn("## Top Exact-Parent Rows By Target", report)
        self.assertIn("## Top Family-Relative Rows By Target", report)
        self.assertIn("| bucket | target | feature | control_mode | n | spearman |", report)
        self.assertNotIn("r2", report)
        self.assertNotIn("mae", report)


class PerformanceTrendDiagnosticsCliTests(unittest.TestCase):
    def test_main_writes_descriptive_outputs_with_canonical_parent_ids(self) -> None:
        dataset = pd.DataFrame(
            [
                {
                    "model_id": "repo--parent-a",
                    "display_name": "repo/parent-a",
                    "source_model": "",
                    "base_model_relation": "",
                    "audited_base_family": "fam/a",
                    "candidate_family": "fam/a",
                    "query_bucket": "supported_candidate",
                    "common_schema": True,
                    "schema_name": "common_llm_7",
                    "num_layers": 2,
                    "raw_module_count": 2,
                    "Average ⬆️": 10.0,
                    "MMLU-PRO": 11.0,
                    "MATH Lvl 5": 12.0,
                },
                {
                    "model_id": "repo--child-a",
                    "display_name": "repo/child-a",
                    "source_model": "repo/parent-a",
                    "base_model_relation": "finetune",
                    "audited_base_family": "fam/a",
                    "candidate_family": "fam/a",
                    "query_bucket": "supported_candidate",
                    "common_schema": True,
                    "schema_name": "common_llm_7",
                    "num_layers": 2,
                    "raw_module_count": 2,
                    "Average ⬆️": 15.0,
                    "MMLU-PRO": 16.0,
                    "MATH Lvl 5": 20.0,
                },
                {
                    "model_id": "peer-b",
                    "display_name": "repo/peer-b",
                    "source_model": "",
                    "base_model_relation": "",
                    "audited_base_family": "fam/a",
                    "candidate_family": "fam/a",
                    "query_bucket": "heterogeneous",
                    "common_schema": False,
                    "schema_name": "bert_like_5",
                    "num_layers": 3,
                    "raw_module_count": 3,
                    "Average ⬆️": 7.0,
                    "MMLU-PRO": 8.0,
                    "MATH Lvl 5": 6.0,
                },
                {
                    "model_id": "peer-c",
                    "display_name": "repo/peer-c",
                    "source_model": "",
                    "base_model_relation": "",
                    "audited_base_family": "fam/b",
                    "candidate_family": "fam/b",
                    "query_bucket": "heterogeneous",
                    "common_schema": False,
                    "schema_name": "bert_like_5",
                    "num_layers": 4,
                    "raw_module_count": 4,
                    "Average ⬆️": 9.0,
                    "MMLU-PRO": 10.0,
                    "MATH Lvl 5": 11.0,
                },
            ]
        )
        records = {
            "repo--parent-a": AlphaRecord(
                "repo--parent-a",
                "repo/parent-a",
                "/tmp/repo--parent-a",
                2,
                ("self_attn.q_proj", "mlp.down_proj"),
                np.array([[1.0, 2.0], [2.0, 3.0]], dtype=float),
            ),
            "repo--child-a": AlphaRecord(
                "repo--child-a",
                "repo/child-a",
                "/tmp/repo--child-a",
                2,
                ("self_attn.q_proj", "mlp.down_proj"),
                np.array([[2.0, 3.0], [4.0, 5.0]], dtype=float),
            ),
            "peer-b": AlphaRecord(
                "peer-b",
                "repo/peer-b",
                "/tmp/peer-b",
                2,
                ("self_attn.q_proj", "mlp.down_proj"),
                np.array([[0.5, 1.5], [1.5, 2.5]], dtype=float),
            ),
            "peer-c": AlphaRecord(
                "peer-c",
                "repo/peer-c",
                "/tmp/peer-c",
                2,
                ("self_attn.q_proj", "mlp.down_proj"),
                np.array([[0.75, 1.75], [1.75, 2.75]], dtype=float),
            ),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                performance_trend_diagnostics,
                "build_prediction_dataset",
                return_value=(dataset, records),
            ):
                exit_code = performance_trend_diagnostics.main(["--output-dir", temp_dir])

            self.assertEqual(exit_code, 0)
            output_dir = performance_trend_diagnostics.Path(temp_dir)
            for file_name in [
                "absolute_trend_dataset.csv",
                "exact_parent_delta_dataset.csv",
                "family_relative_delta_dataset.csv",
                "trend_results.csv",
                "report.md",
            ]:
                self.assertTrue((output_dir / file_name).exists(), file_name)

            results = pd.read_csv(output_dir / "trend_results.csv")
            self.assertEqual(
                list(results.columns),
                ["analysis", "bucket", "target", "feature", "control_mode", "n", "spearman"],
            )

            exact_parent = pd.read_csv(output_dir / "exact_parent_delta_dataset.csv")
            self.assertIn("source_model", exact_parent.columns)
            self.assertIn("parent_model_id", exact_parent.columns)
            row = exact_parent.loc[exact_parent["model_id"].eq("repo--child-a")].iloc[0]
            self.assertEqual(row["source_model"], "repo/parent-a")
            self.assertEqual(row["parent_model_id"], "repo--parent-a")
