from __future__ import annotations

import unittest

import numpy as np

from spectral_research.features import (
    alpha_scalar_features,
    attention_trace_features,
    full_module_trace_features,
)


class FeatureTests(unittest.TestCase):
    def test_attention_trace_keeps_only_attention_modules(self) -> None:
        alpha = np.arange(24, dtype=float).reshape(3, 8)
        modules = [
            "mlp.down_proj",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "other",
        ]

        feat = attention_trace_features(alpha, modules)

        self.assertEqual(feat.shape, (3, 4))
        np.testing.assert_array_equal(feat, alpha[:, 1:5])

    def test_scalar_features_return_fixed_width(self) -> None:
        alpha = np.array([[1.0, 2.0], [3.0, 5.0]])

        feat = alpha_scalar_features(alpha)

        self.assertEqual(feat.shape, (1, 6))
        np.testing.assert_allclose(feat[0], np.array([2.75, 1.47901995, 2.5, 1.5, 4.0, 2.5]))

    def test_full_module_trace_preserves_module_count(self) -> None:
        alpha = np.ones((4, 7), dtype=np.float32)

        feat = full_module_trace_features(alpha)

        self.assertEqual(feat.shape, (4, 7))
        self.assertEqual(feat.dtype, np.dtype(float))
        np.testing.assert_array_equal(feat, alpha.astype(float))
