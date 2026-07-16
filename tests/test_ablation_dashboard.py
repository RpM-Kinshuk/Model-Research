import numpy as np
import pandas as pd

from alpha_cluster_dashboard.ablation import (
    build_knn_prediction_frame,
    merge_ablation_metadata,
    similarity_to_distance,
)


def test_merge_ablation_metadata_joins_by_display_name():
    metadata = pd.DataFrame(
        {
            "display_name": ["org/a", "org/b"],
            "num_layers": [2, 3],
        }
    )
    ablation = pd.DataFrame(
        {
            "model_id": ["org/a", "org/c"],
            "coarse_family": ["llama", "qwen"],
            "Average \u2b06\ufe0f": [70.0, 80.0],
        }
    )

    result = merge_ablation_metadata(metadata, ablation)

    assert result.loc[0, "coarse_family"] == "llama"
    assert result.loc[0, "Average \u2b06\ufe0f"] == 70.0
    assert pd.isna(result.loc[1, "coarse_family"])


def test_similarity_to_distance_keeps_nearest_pairs_small():
    similarity = np.array(
        [
            [1.0, 0.8, -0.2],
            [0.8, 1.0, 0.1],
            [-0.2, 0.1, 1.0],
        ]
    )

    distance = similarity_to_distance(similarity)

    assert np.allclose(np.diag(distance), 0.0)
    assert distance[0, 1] < distance[0, 2]
    assert distance[1, 2] < distance[0, 2]


def test_build_knn_prediction_frame_uses_leave_one_out_neighbors():
    frame = pd.DataFrame(
        {
            "display_name": ["a", "b", "c", "d"],
            "Average \u2b06\ufe0f": [10.0, 12.0, 50.0, 52.0],
        }
    )
    similarity = np.array(
        [
            [1.0, 0.9, 0.1, 0.2],
            [0.9, 1.0, 0.1, 0.2],
            [0.1, 0.1, 1.0, 0.8],
            [0.2, 0.2, 0.8, 1.0],
        ]
    )

    result = build_knn_prediction_frame(frame, similarity, target="Average \u2b06\ufe0f", k=1)

    assert result["prediction"].tolist() == [12.0, 10.0, 52.0, 50.0]
    assert result["nearest_neighbors"].tolist() == ["b", "a", "d", "c"]
    assert result["absolute_error"].mean() == 2.0
