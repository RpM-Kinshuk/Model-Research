import numpy as np
import pandas as pd

from scripts.htsr_scaleup_study import _profile_centroid_predictions
from spectral_research.study.scaleup import (
    consensus_roots,
    map_overlap_to_phase2,
    reconcile_phase2_legacy_targets,
    reconcile_phase2_modern_targets,
)


MODERN = ("IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO")
LEGACY = ("ARC", "HellaSwag", "MMLU", "TruthfulQA", "Winogrande", "GSM8K")


def _scores(columns, start):
    return {column: float(start + offset) for offset, column in enumerate(columns)}


def test_modern_reconciliation_uses_unique_recorded_average():
    phase2 = pd.DataFrame(
        [{"model_id": "org/model", "canonical_model_key": "org/model", "Average ⬆️": 12.5}]
    )
    overlap = pd.DataFrame(
        [
            {"model_id": "org/model", "canonical_model_key": "org/model", "Average ⬆️": 12.5, **_scores(MODERN, 10)},
            {"model_id": "org/model", "canonical_model_key": "org/model", "Average ⬆️": 22.5, **_scores(MODERN, 20)},
        ]
    )
    mapped = map_overlap_to_phase2(phase2, overlap)
    targets, audit = reconcile_phase2_modern_targets(
        phase2,
        mapped,
        pd.DataFrame(columns=["model_id", *MODERN]),
        tasks=MODERN,
    )
    assert targets.loc[0, "IFEval"] == 10.0
    assert audit.loc[0, "reason"] == "unique_reported_average"


def test_legacy_reconciliation_rejects_disagreeing_evaluations():
    phase2 = pd.DataFrame(
        [{"model_id": "org/model", "canonical_model_key": "org/model"}]
    )
    overlap = pd.DataFrame(
        [
            {"model_id": "org/model", "canonical_model_key": "org/model", **_scores(LEGACY, 10)},
            {"model_id": "org/model", "canonical_model_key": "org/model", **_scores(LEGACY, 20)},
        ]
    )
    mapped = map_overlap_to_phase2(phase2, overlap)
    targets, audit = reconcile_phase2_legacy_targets(mapped, tasks=LEGACY)
    assert targets.empty
    assert audit.loc[0, "reason"] == "conflicting_score_rows"


def test_consensus_roots_accept_same_root_merge_and_reject_conflict():
    edges = pd.DataFrame(
        [
            {"parent_id": "root/a", "child_id": "branch/a1"},
            {"parent_id": "root/a", "child_id": "branch/a2"},
            {"parent_id": "branch/a1", "child_id": "merge/same"},
            {"parent_id": "branch/a2", "child_id": "merge/same"},
            {"parent_id": "root/b", "child_id": "merge/conflict"},
            {"parent_id": "root/a", "child_id": "merge/conflict"},
        ]
    )
    roots = consensus_roots(edges, graph_nodes=[]).set_index("model_id")
    assert roots.at["merge/same", "consensus_root_model"] == "root/a"
    assert pd.isna(roots.at["merge/conflict", "consensus_root_model"])


def test_profile_branch_predictions_keep_unscorable_queries():
    population = pd.DataFrame(
        [
            {"model_id": "a/one", "component_root_model": "root/a", "root_branch_model": "branch/a1", "branch_fold": 0},
            {"model_id": "a/two", "component_root_model": "root/a", "root_branch_model": "branch/a2", "branch_fold": 1},
            {"model_id": "a/missing", "component_root_model": "root/a", "root_branch_model": "branch/a3", "branch_fold": 0},
            {"model_id": "b/one", "component_root_model": "root/b", "root_branch_model": "branch/b1", "branch_fold": 0},
            {"model_id": "b/two", "component_root_model": "root/b", "root_branch_model": "branch/b2", "branch_fold": 1},
        ]
    )
    left = np.tile(np.r_[1.0, np.zeros(31)], 7).astype(np.float32)
    right = np.tile(np.r_[0.0, 1.0, np.zeros(30)], 7).astype(np.float32)
    embeddings = {
        "a/one": left,
        "a/two": left,
        "b/one": right,
        "b/two": right,
    }

    predictions = _profile_centroid_predictions(population, embeddings)

    assert len(predictions) == len(population)
    missing = predictions.loc[predictions["model_id"].eq("a/missing")].iloc[0]
    assert not missing["scorable"]
    assert missing["unscorable_reason"] == "query_profile_unavailable"
