from __future__ import annotations

import math

import numpy as np
import pandas as pd


def confidence_from_ranked_scores(
    scores: np.ndarray,
    rule: str = "top_score",
    *,
    score_center: float = 0.0,
    score_scale: float = 1.0,
    margin_center: float = 0.0,
    margin_scale: float = 1.0,
) -> float:
    score_array = np.asarray(scores, dtype=float)
    if score_array.size == 0:
        return float("nan")

    top_score = float(score_array[0])
    second_score = float(score_array[1]) if score_array.size > 1 else float("nan")
    margin = top_score - second_score if np.isfinite(second_score) else float("nan")

    if rule == "top_score":
        return top_score
    if rule == "margin":
        return margin
    if rule == "score_plus_margin":
        safe_score_scale = score_scale if score_scale > 0 else 1.0
        safe_margin_scale = margin_scale if margin_scale > 0 else 1.0
        score_term = (top_score - score_center) / safe_score_scale
        margin_term = (margin - margin_center) / safe_margin_scale
        return 0.5 * (score_term + margin_term)
    raise KeyError(f"Unsupported confidence rule: {rule}")


def precision_at_coverages(
    confidence: np.ndarray,
    correct: np.ndarray,
    *,
    coverages: tuple[float, ...] = (0.1, 0.25, 0.5),
) -> pd.DataFrame:
    confidence_array = np.asarray(confidence, dtype=float)
    correct_array = np.asarray(correct, dtype=bool)
    if confidence_array.size == 0:
        return pd.DataFrame(
            columns=["coverage", "accepted_precision", "threshold", "accepted_count"]
        )

    order = np.argsort(-confidence_array, kind="mergesort")
    sorted_confidence = confidence_array[order]
    sorted_correct = correct_array[order]
    total = int(sorted_confidence.size)

    rows: list[dict[str, float]] = []
    for coverage in coverages:
        accepted_count = min(total, max(1, int(math.ceil(total * coverage))))
        accepted = sorted_correct[:accepted_count]
        precision = float(np.mean(accepted))
        rows.append(
            {
                "coverage": float(coverage),
                "accepted_count": int(accepted_count),
                "threshold": float(sorted_confidence[accepted_count - 1]),
                "precision": precision,
                "accepted_precision": precision,
            }
        )

    return pd.DataFrame(
        rows,
        columns=["coverage", "accepted_precision", "threshold", "accepted_count"],
    )
