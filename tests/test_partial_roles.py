from __future__ import annotations

import numpy as np
import pandas as pd

from spectral_research.study.features import CANONICAL_ROLES
from spectral_research.study.partial_roles import (
    PartialRoleCorpus,
    extend_strict_clusters,
    pairwise_role_similarity,
    select_reliable_role_patterns,
)


def _corpus(model_ids: list[str], masks: list[list[bool]]) -> PartialRoleCorpus:
    values = np.zeros((len(model_ids), len(CANONICAL_ROLES), 4), dtype=np.float32)
    for model, mask in enumerate(masks):
        for role, present in enumerate(mask):
            if present:
                values[model, role, model % 4] = 1.0
    return PartialRoleCorpus(
        model_ids=np.asarray(model_ids),
        role_embeddings=values,
        role_mask=np.asarray(masks, dtype=bool),
        cache_paths=(),
    )


def test_role_similarity_rejects_too_little_overlap() -> None:
    left = _corpus(["left"], [[True, True, True, True, False, False, False]])
    right = _corpus(["right"], [[True, False, False, False, True, True, True]])

    similarity, shared = pairwise_role_similarity(
        left, right, average_shared=True, minimum_shared_roles=4
    )

    assert shared.item() == 1
    assert np.isneginf(similarity.item())


def test_lineage_admits_only_mask_patterns_that_pass_calibration() -> None:
    corpus = _corpus(
        ["complete", "reliable", "weak"],
        [
            [True] * 7,
            [True, True, True, True, False, False, False],
            [True, False, False, False, True, True, True],
        ],
    )
    patterns = pd.DataFrame(
        {
            "role_mask": [
                "attn_q|attn_k|attn_v|attn_o",
                "attn_q|mlp_gate|mlp_up|mlp_down",
            ],
            "full_nearest_retained_in_top10": [0.96, 0.71],
        }
    )

    selected = select_reliable_role_patterns(
        corpus, patterns, minimum_retention=0.90
    )

    assert selected.model_ids.tolist() == ["complete", "reliable"]


def test_partial_models_attach_without_changing_core_assignments() -> None:
    corpus = _corpus(
        ["a", "b", "c", "d", "new"],
        [
            [True] * 7,
            [True] * 7,
            [True] * 7,
            [True] * 7,
            [True, True, True, True, False, False, False],
        ],
    )
    strict = pd.DataFrame(
        {
            "model_id": ["a", "b", "c", "d"],
            "predicted_cluster": [0, 0, 1, 1],
            "spectral_medoid_model": ["a", "a", "c", "c"],
            "nearest_neighbors": ["b", "a", "d", "c"],
            "nearest_similarity": [0.8, 0.8, 0.8, 0.8],
        }
    )

    extended = extend_strict_clusters(strict["model_id"], strict, corpus)

    core = extended.set_index("model_id").loc[["a", "b", "c", "d"]]
    assert core["predicted_cluster"].tolist() == [0, 0, 1, 1]
    assert core["assignment_status"].eq("strict_core").all()
    assert extended.set_index("model_id").at["new", "assignment_status"] == "coverage_extension"
