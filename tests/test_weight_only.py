import numpy as np
import pandas as pd

from spectral_research.study.weight_only import (
    ProfileCorpus,
    cluster_holdout_split_table,
    cross_validated_performance_freeze,
    profile_split_table,
)


def _corpus() -> ProfileCorpus:
    rng = np.random.default_rng(9)
    matrix = rng.normal(size=(8, 14)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return ProfileCorpus(
        model_ids=np.asarray([f"org/model-{index}" for index in range(8)]),
        embeddings=matrix,
        cache_paths=(),
    )


def test_exact_spectral_profiles_stay_in_one_fold() -> None:
    corpus = _corpus()
    corpus.embeddings[1] = corpus.embeddings[0]

    splits = profile_split_table(corpus, seed=17, n_folds=5)

    duplicate = splits.iloc[:2]
    assert duplicate["profile_hash"].nunique() == 1
    assert duplicate["blind_fold"].nunique() == 1


def test_spectral_cluster_holdout_keeps_clusters_whole() -> None:
    corpus = _corpus()
    profile_splits = profile_split_table(corpus, seed=17, n_folds=2)
    assignments = pd.DataFrame(
        {
            "model_id": corpus.model_ids,
            "predicted_cluster": [0, 0, 0, 1, 1, 2, 2, 3],
        }
    )

    splits = cluster_holdout_split_table(
        profile_splits, assignments, n_folds=2
    )

    assert len(splits) == len(corpus.model_ids)
    assert splits.groupby("predicted_cluster")["blind_fold"].nunique().max() == 1
    assert set(splits["blind_fold"]) == {0, 1}


def test_performance_prediction_ignores_unrequested_metadata_columns() -> None:
    corpus = _corpus()
    splits = pd.DataFrame(
        {
            "model_id": corpus.model_ids,
            "profile_hash": [f"hash-{index}" for index in range(8)],
            "blind_fold": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    targets = pd.DataFrame(
        {
            "model_id": corpus.model_ids,
            "score": np.arange(8, dtype=float),
            "architecture": ["secret-a", "secret-b"] * 4,
            "lineage": ["family-a"] * 8,
        }
    )

    full = cross_validated_performance_freeze(
        corpus, splits, targets, ("score",), generation="test"
    )[0]
    stripped = cross_validated_performance_freeze(
        corpus,
        splits,
        targets.loc[:, ["model_id", "score"]],
        ("score",),
        generation="test",
    )[0]

    pd.testing.assert_frame_equal(full, stripped)
    assert "architecture" not in full.columns
    assert "lineage" not in full.columns
