from __future__ import annotations

import unittest

import numpy as np

from spectral_research.evaluation import (
    open_set_metrics,
    retrieval_metrics,
    risk_coverage_curve,
)


class EvaluationTests(unittest.TestCase):
    def test_retrieval_metrics_compute_mrr_and_recall(self) -> None:
        rankings = [["fam-a", "fam-b"], ["fam-b", "fam-a"]]
        truths = ["fam-a", "fam-a"]

        metrics = retrieval_metrics(rankings, truths, ks=(1, 2))

        self.assertAlmostEqual(metrics["top1_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["mrr"], 0.75)
        self.assertAlmostEqual(metrics["recall@2"], 1.0)

    def test_open_set_metrics_include_auroc_auprc_and_fpr95(self) -> None:
        scores = np.array([0.9, 0.8, 0.3, 0.2], dtype=float)
        labels = np.array([1, 1, 0, 0], dtype=int)

        metrics = open_set_metrics(scores, labels)

        self.assertIn("auroc", metrics)
        self.assertIn("auprc", metrics)
        self.assertIn("fpr95", metrics)
        self.assertGreaterEqual(metrics["auroc"], 0.0)
        self.assertLessEqual(metrics["auroc"], 1.0)

    def test_risk_coverage_curve_is_monotone_in_coverage(self) -> None:
        confidence = np.array([0.9, 0.8, 0.4], dtype=float)
        correct = np.array([1, 0, 1], dtype=int)

        curve = risk_coverage_curve(confidence, correct)

        self.assertTrue((curve["coverage"].diff().fillna(0) >= 0).all())
        self.assertEqual(list(curve.columns), ["coverage", "risk", "threshold"])


if __name__ == "__main__":
    unittest.main()
