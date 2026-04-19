from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from alpha_cluster_dashboard.data import AlphaRecord
from spectral_research.trend_diagnostics import (
    build_absolute_trend_frame,
    build_exact_parent_delta_frame,
    build_family_relative_delta_frame,
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

    def test_build_family_relative_delta_frame_uses_family_median_anchor(self) -> None:
        feature_frame = build_absolute_trend_frame(self._dataset(), self._records())
        delta_frame = build_family_relative_delta_frame(feature_frame)

        row = delta_frame.loc[delta_frame["model_id"].eq("child-a")].iloc[0]
        self.assertEqual(row["family_anchor"], "fam/a")
        self.assertAlmostEqual(float(row["MMLU-PRO_delta"]), 5.0)
        self.assertIn("trend_global_end_minus_start_delta", delta_frame.columns)
