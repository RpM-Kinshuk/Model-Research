import numpy as np
import pandas as pd

from spectral_research.ablation_method_study import (
    build_structural_similarity_matrix,
    evaluate_family_similarity,
    evaluate_knn_prediction,
    pair_similarity,
)
from alpha_cluster_dashboard.data import AlphaRecord


def test_paper_spearman_prefers_matching_depth_order():
    left = {"attn_q": np.array([1.0, 2.0, 3.0, 4.0])}
    same_order = {"attn_q": np.array([10.0, 20.0, 30.0, 40.0])}
    reversed_order = {"attn_q": np.array([40.0, 30.0, 20.0, 10.0])}

    assert pair_similarity(left, same_order, method="paper_spearman") > 0.99
    assert pair_similarity(left, reversed_order, method="paper_spearman") < -0.99


def test_rank_rbf_prefers_matching_rank_shape():
    left = {"mlp_up": np.array([1.0, 4.0, 2.0, 3.0])}
    same = {"mlp_up": np.array([10.0, 40.0, 20.0, 30.0])}
    different = {"mlp_up": np.array([40.0, 10.0, 30.0, 20.0])}

    assert pair_similarity(left, same, method="rank_rbf") > pair_similarity(left, different, method="rank_rbf")


def test_role_alpha_wasserstein_is_distributional_not_ordered():
    left = {"mlp_up": np.array([1.0, 2.0, 5.0, 6.0])}
    same_distribution = {"mlp_up": np.array([6.0, 1.0, 5.0, 2.0])}
    shifted_distribution = {"mlp_up": np.array([10.0, 11.0, 14.0, 15.0])}

    assert pair_similarity(left, same_distribution, method="role_alpha_wasserstein") == 0.0
    assert pair_similarity(left, same_distribution, method="role_alpha_wasserstein") > pair_similarity(
        left, shifted_distribution, method="role_alpha_wasserstein"
    )


def test_structural_basics_prefers_matching_metadata_shape():
    frame = pd.DataFrame(
        {
            "Architecture": ["llama", "llama", "gpt-neox"],
            "#Params (B)": [7.0, 7.2, 20.0],
            "Precision": ["float16", "float16", "bfloat16"],
        }
    )
    records = [
        AlphaRecord(str(index), str(index), "", 32 if index < 2 else 44, ("a", "b"), np.ones((2, 2)))
        for index in range(3)
    ]

    matrix = build_structural_similarity_matrix(frame, records)

    assert matrix[0, 1] > matrix[0, 2]


def test_evaluate_family_similarity_reports_auc_and_top1_recall():
    matrix = np.array(
        [
            [1.0, 0.9, 0.1, 0.2],
            [0.9, 1.0, 0.1, 0.2],
            [0.1, 0.1, 1.0, 0.8],
            [0.2, 0.2, 0.8, 1.0],
        ]
    )
    labels = ["a", "a", "b", "b"]

    result = evaluate_family_similarity(matrix, labels)

    assert result["pairwise_auc"] == 1.0
    assert result["top1_family_recall"] == 1.0


def test_evaluate_knn_prediction_uses_leave_one_out_neighbors():
    matrix = np.array(
        [
            [1.0, 0.9, 0.1, 0.2],
            [0.9, 1.0, 0.1, 0.2],
            [0.1, 0.1, 1.0, 0.8],
            [0.2, 0.2, 0.8, 1.0],
        ]
    )
    frame = pd.DataFrame({"target": [10.0, 12.0, 50.0, 52.0]})

    result = evaluate_knn_prediction(matrix, frame, target="target", k=1)

    assert result["n"] == 4
    assert result["mae"] == 2.0
    assert result["spearman"] > 0.5
