from __future__ import annotations

import unittest

import numpy as np

from spectral_research.rejection import (
    confidence_from_ranked_scores,
    precision_at_coverages,
)


class RejectionConfidenceTests(unittest.TestCase):
    def test_confidence_from_ranked_scores_returns_top_score(self) -> None:
        scores = np.array([0.8, 0.5, 0.1], dtype=float)

        self.assertAlmostEqual(
            confidence_from_ranked_scores(scores, rule="top_score"),
            0.8,
        )

    def test_confidence_from_ranked_scores_returns_margin(self) -> None:
        scores = np.array([0.8, 0.5, 0.1], dtype=float)

        self.assertAlmostEqual(
            confidence_from_ranked_scores(scores, rule="margin"),
            0.3,
        )

    def test_confidence_from_ranked_scores_returns_score_plus_margin(self) -> None:
        scores = np.array([0.8, 0.5, 0.1], dtype=float)

        confidence = confidence_from_ranked_scores(
            scores,
            rule="score_plus_margin",
            score_center=0.6,
            score_scale=0.1,
            margin_center=0.2,
            margin_scale=0.1,
        )

        self.assertAlmostEqual(confidence, 1.5)

    def test_precision_at_coverages_reports_requested_rows(self) -> None:
        confidence = np.array([0.9, 0.8, 0.4, 0.1], dtype=float)
        correct = np.array([1, 1, 0, 0], dtype=bool)

        frame = precision_at_coverages(
            confidence,
            correct,
            coverages=(0.25, 0.5, 1.0),
        )

        self.assertEqual(list(frame["coverage"]), [0.25, 0.5, 1.0])
        self.assertAlmostEqual(
            frame.loc[frame["coverage"] == 0.25, "accepted_precision"].iloc[0],
            1.0,
        )
        self.assertAlmostEqual(
            frame.loc[frame["coverage"] == 0.5, "accepted_precision"].iloc[0],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
