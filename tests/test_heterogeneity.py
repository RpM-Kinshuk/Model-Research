from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from spectral_research.heterogeneity import (
    canonical_role,
    intrinsic_basics_feature_map,
    role_aggregated_trace_score,
    shared_role_trace_score,
)


class HeterogeneityTests(unittest.TestCase):
    def test_canonical_role_maps_attention_and_mlp_suffixes(self) -> None:
        self.assertEqual(canonical_role("self_attn.q_proj"), "attn_q")
        self.assertEqual(canonical_role("self_attn.o_proj"), "attn_o")
        self.assertEqual(canonical_role("mlp.down_proj"), "mlp_down")
        self.assertIsNone(canonical_role("totally_unknown"))

    def test_intrinsic_basics_feature_map_ignores_external_catalog_fields(self) -> None:
        manifest = pd.DataFrame(
            [
                {
                    "model_id": "model/a",
                    "Architecture": "LlamaForCausalLM",
                    "#Params (B)": 8.03,
                    "num_layers": 32,
                    "schema_name": "common_llm_7",
                    "raw_module_count": 7,
                    "common_schema": True,
                    "Type": "chat",
                    "Generation": 9,
                },
                {
                    "model_id": "model/b",
                    "Architecture": "LlamaForCausalLM",
                    "#Params (B)": 8.03,
                    "num_layers": 32,
                    "schema_name": "common_llm_7",
                    "raw_module_count": 7,
                    "common_schema": True,
                    "Type": "merge",
                    "Generation": 0,
                },
            ]
        )

        vectors = intrinsic_basics_feature_map(manifest)

        np.testing.assert_allclose(vectors["model/a"], vectors["model/b"])

    def test_intrinsic_basics_feature_map_fails_when_no_intrinsic_features_exist(self) -> None:
        manifest = pd.DataFrame([{"model_id": "model/a"}])

        with self.assertRaises(ValueError):
            intrinsic_basics_feature_map(manifest)

    def test_intrinsic_basics_feature_map_returns_empty_map_for_empty_manifest(self) -> None:
        manifest = pd.DataFrame(columns=["model_id"])

        self.assertEqual(intrinsic_basics_feature_map(manifest), {})

    def test_shared_role_trace_score_uses_only_overlapping_roles(self) -> None:
        query_alpha = np.array([[1.0, 2.0], [2.0, 3.0]])
        query_modules = ("self_attn.q_proj", "mlp.down_proj")
        proto_alpha = np.array([[1.0, 99.0, 2.0], [2.0, 98.0, 3.0]])
        proto_modules = ("self_attn.q_proj", "self_attn.k_proj", "mlp.down_proj")

        score = shared_role_trace_score(
            query_alpha=query_alpha,
            query_modules=query_modules,
            proto_alpha=proto_alpha,
            proto_modules=proto_modules,
            overlap_penalty=0.2,
        )

        self.assertTrue(np.isfinite(score))
        self.assertGreater(score, -1.0)

    def test_shared_role_trace_score_handles_different_layer_counts(self) -> None:
        query_alpha = np.array([[1.0, 2.0], [2.0, 3.0]])
        query_modules = ("self_attn.q_proj", "mlp.down_proj")
        proto_alpha = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        proto_modules = ("self_attn.q_proj", "mlp.down_proj")

        score = shared_role_trace_score(
            query_alpha=query_alpha,
            query_modules=query_modules,
            proto_alpha=proto_alpha,
            proto_modules=proto_modules,
            overlap_penalty=0.2,
        )

        self.assertTrue(np.isfinite(score))

    def test_shared_role_trace_score_rejects_alpha_module_width_mismatch(self) -> None:
        query_alpha = np.array([[1.0, 2.0]])
        query_modules = ("self_attn.q_proj",)
        proto_alpha = np.array([[1.0, 2.0]])
        proto_modules = ("self_attn.q_proj", "mlp.down_proj")

        with self.assertRaises(ValueError):
            shared_role_trace_score(
                query_alpha=query_alpha,
                query_modules=query_modules,
                proto_alpha=proto_alpha,
                proto_modules=proto_modules,
                overlap_penalty=0.2,
            )

    def test_role_aggregated_trace_score_handles_schema_mismatch(self) -> None:
        query_alpha = np.array([[1.0, 2.0], [2.0, 4.0]])
        query_modules = ("self_attn.q_proj", "self_attn.o_proj")
        proto_alpha = np.array([[1.0, 1.5, 3.0], [2.0, 2.5, 4.0]])
        proto_modules = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj")

        score = role_aggregated_trace_score(
            query_alpha=query_alpha,
            query_modules=query_modules,
            proto_alpha=proto_alpha,
            proto_modules=proto_modules,
            overlap_penalty=0.2,
        )

        self.assertTrue(np.isfinite(score))

    def test_role_aggregated_trace_score_handles_different_layer_counts(self) -> None:
        query_alpha = np.array([[1.0, 2.0], [2.0, 4.0]])
        query_modules = ("self_attn.q_proj", "self_attn.o_proj")
        proto_alpha = np.array([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]])
        proto_modules = ("self_attn.q_proj", "self_attn.o_proj")

        score = role_aggregated_trace_score(
            query_alpha=query_alpha,
            query_modules=query_modules,
            proto_alpha=proto_alpha,
            proto_modules=proto_modules,
            overlap_penalty=0.2,
        )

        self.assertTrue(np.isfinite(score))

    def test_role_aggregated_trace_score_rejects_alpha_module_width_mismatch(self) -> None:
        query_alpha = np.array([[1.0, 2.0]])
        query_modules = ("self_attn.q_proj",)
        proto_alpha = np.array([[1.0, 2.0]])
        proto_modules = ("self_attn.q_proj", "mlp.down_proj")

        with self.assertRaises(ValueError):
            role_aggregated_trace_score(
                query_alpha=query_alpha,
                query_modules=query_modules,
                proto_alpha=proto_alpha,
                proto_modules=proto_modules,
                overlap_penalty=0.2,
            )


if __name__ == "__main__":
    unittest.main()
