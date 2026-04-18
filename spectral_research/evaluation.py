from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def retrieval_metrics(
    rankings: list[list[str]],
    truths: list[str],
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    total = len(truths)
    top1_hits = 0
    reciprocal_sum = 0.0
    recall_hits = {k: 0 for k in ks}

    for ranking, truth in zip(rankings, truths, strict=False):
        for rank, family in enumerate(ranking, start=1):
            if family != truth:
                continue
            if rank == 1:
                top1_hits += 1
            reciprocal_sum += 1.0 / rank
            for k in ks:
                if rank <= k:
                    recall_hits[k] += 1
            break

    metrics = {
        "top1_accuracy": top1_hits / total,
        "mrr": reciprocal_sum / total,
    }
    for k in ks:
        metrics[f"recall@{k}"] = recall_hits[k] / total
    return metrics


def open_set_metrics(
    confidence: np.ndarray,
    supported_label: np.ndarray,
) -> dict[str, float]:
    confidence_array = np.asarray(confidence, dtype=float)
    label_array = np.asarray(supported_label, dtype=int)

    fpr, tpr, _ = roc_curve(label_array, confidence_array)
    fpr95_idx = np.where(tpr >= 0.95)[0]
    fpr95 = float(fpr[fpr95_idx[0]]) if len(fpr95_idx) else float("nan")

    return {
        "auroc": float(roc_auc_score(label_array, confidence_array)),
        "auprc": float(average_precision_score(label_array, confidence_array)),
        "fpr95": fpr95,
    }


def risk_coverage_curve(confidence: np.ndarray, correct: np.ndarray) -> pd.DataFrame:
    confidence_array = np.asarray(confidence, dtype=float)
    correct_array = np.asarray(correct, dtype=bool)
    order = np.argsort(-confidence_array)

    sorted_confidence = confidence_array[order]
    sorted_correct = correct_array[order]

    rows: list[dict[str, float]] = []
    errors = 0
    total = len(sorted_correct)

    for index, is_correct in enumerate(sorted_correct, start=1):
        errors += int(not is_correct)
        rows.append(
            {
                "coverage": index / total,
                "risk": errors / index,
                "threshold": float(sorted_confidence[index - 1]),
            }
        )

    return pd.DataFrame(rows, columns=["coverage", "risk", "threshold"])
