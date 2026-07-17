from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import spectral_research.study.lineage as lineage

from spectral_research.study.features import (
    ALPHA_ESTIMATOR,
    ALPHA_ESTIMATOR_VERSION,
    CANONICAL_ROLES,
)
from spectral_research.study.lineage import (
    LINEAGE_METHODS,
    PAPER_STYLE_ALPHA_DKS,
    PAPER_STYLE_ALPHA_DKS_PROVENANCE,
    bootstrap_closed_set_metrics,
    bootstrap_closed_set_contrasts,
    build_root_gallery,
    closed_set_metrics,
    cluster_lineage_diagnostics,
    compute_open_set_metrics,
    evaluate_exact_parent_diagnostic,
    evaluate_unknown_family_rejection,
    fit_gallery_scaler,
    paper_style_alpha_similarity,
    rank_confirmed_queries,
    root_macro_top1,
    shared_feature_similarity,
    unknown_family_folds,
    _root_macro_purity,
)


def _manifest(
    families: tuple[str, ...] = ("org/a", "org/b", "org/c"),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, family in enumerate(families):
        architecture = "arch-even" if index % 2 == 0 else "arch-odd"
        rows.extend(
            [
                {
                    "model_id": family,
                    "audited_base_family": family,
                    "label_status": "confirmed",
                    "artifact_available": True,
                    "architecture": architecture,
                },
                {
                    "model_id": f"derived/{family.rsplit('/', 1)[-1]}",
                    "audited_base_family": family,
                    "label_status": "confirmed",
                    "artifact_available": True,
                    "architecture": architecture,
                },
            ]
        )
    return pd.DataFrame(rows)


def _features(manifest: pd.DataFrame) -> pd.DataFrame:
    family_values = {"org/a": 0.0, "org/b": 10.0, "org/c": 20.0}
    rows = []
    for row in manifest.itertuples(index=False):
        family = str(row.audited_base_family)
        base = family_values[family]
        is_root = str(row.model_id) == family
        rows.append(
            {
                "model_id": row.model_id,
                "M__depth": base + (0.0 if is_root else 0.2),
                "M__width": base + (0.0 if is_root else 0.1),
                "S__shape": base + (0.0 if is_root else 0.2),
                "H__alpha": base + (0.0 if is_root else 0.1),
            }
        )
    return pd.DataFrame(rows)


def _trajectories(manifest: pd.DataFrame) -> pd.DataFrame:
    patterns = {
        "org/a": np.array([1.0, 2.0, 3.0, 4.0]),
        "org/b": np.array([4.0, 3.0, 2.0, 1.0]),
        "org/c": np.array([1.0, 4.0, 2.0, 3.0]),
    }
    rows = []
    for model in manifest.itertuples(index=False):
        values = patterns[str(model.audited_base_family)]
        for role_index, role in enumerate(CANONICAL_ROLES):
            for depth, alpha in enumerate(values):
                rows.append(
                    {
                        "model_id": model.model_id,
                        "role": role,
                        "depth": depth,
                        "alpha": alpha + role_index * 0.01,
                        "alpha_estimator": ALPHA_ESTIMATOR,
                        "alpha_estimator_version": ALPHA_ESTIMATOR_VERSION,
                    }
                )
    return pd.DataFrame(rows)


def _one_trace(values: list[float], depths: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "role": role,
                "depth": depth,
                "alpha": alpha,
                "alpha_estimator": ALPHA_ESTIMATOR,
                "alpha_estimator_version": ALPHA_ESTIMATOR_VERSION,
            }
            for role in CANONICAL_ROLES
            for depth, alpha in zip(depths, values, strict=True)
        ]
    )


def test_gallery_requires_actual_available_canonical_root() -> None:
    manifest = pd.DataFrame(
        [
            {
                "model_id": "fam--root",
                "audited_base_family": "fam/root",
                "label_status": "confirmed",
                "artifact_available": True,
            },
            {
                "model_id": "derived/child",
                "audited_base_family": "fam/root",
                "label_status": "confirmed",
                "artifact_available": True,
            },
            {
                "model_id": "orphan/child",
                "audited_base_family": "orphan/root",
                "label_status": "confirmed",
                "artifact_available": True,
            },
            {
                "model_id": "cold--root",
                "audited_base_family": "cold/root",
                "label_status": "confirmed",
                "artifact_available": False,
            },
        ]
    )

    gallery, exclusions = build_root_gallery(manifest)

    assert gallery["model_id"].tolist() == ["fam/root"]
    assert gallery["root_family"].tolist() == ["fam/root"]
    reasons = set(exclusions["reason"])
    assert "descendant_is_not_root" in reasons
    assert "missing_canonical_root" in reasons
    assert "root_feature_or_artifact_unavailable" in reasons


def test_cross_fitted_family_representatives_exclude_each_query() -> None:
    manifest = _manifest()

    candidates, predictions, exclusions = (
        lineage.rank_cross_fitted_family_representatives(
            manifest,
            _features(manifest),
            _trajectories(manifest),
            architecture_column="architecture",
        )
    )

    assert exclusions.empty
    assert predictions.groupby("method")["query_model_id"].nunique().eq(6).all()
    assert predictions["candidate_count"].eq(3).all()
    assert candidates["query_model_id"].ne(
        candidates["candidate_representative_id"]
    ).all()
    assert candidates.groupby(["method", "query_model_id"]).size().eq(3).all()
    combined = predictions.loc[predictions["method"].eq("combined")]
    assert lineage.root_macro_top1(combined) == pytest.approx(1.0)


def test_cross_fitted_query_label_does_not_change_its_rankings() -> None:
    manifest = _manifest()
    query_id = "derived/a"
    first_candidates, _, _ = lineage.rank_cross_fitted_family_representatives(
        manifest,
        _features(manifest),
        _trajectories(manifest),
    )
    permuted = manifest.copy()
    permuted.loc[permuted["model_id"].eq(query_id), "audited_base_family"] = "org/b"
    second_candidates, _, _ = lineage.rank_cross_fitted_family_representatives(
        permuted,
        _features(manifest),
        _trajectories(manifest),
    )

    columns = [
        "method",
        "candidate_family",
        "candidate_representative_id",
        "score",
        "rank",
    ]
    first = first_candidates.loc[
        first_candidates["query_model_id"].eq(query_id), columns
    ].reset_index(drop=True)
    second = second_candidates.loc[
        second_candidates["query_model_id"].eq(query_id), columns
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)


def test_cross_fitted_family_endpoint_reports_family_macro_metrics() -> None:
    manifest = _manifest()

    result = lineage.evaluate_cross_fitted_family_representatives(
        manifest,
        _features(manifest),
        _trajectories(manifest),
        bootstrap_replicates=8,
        random_state=3,
    )

    assert set(result) == {
        "candidates",
        "predictions",
        "metrics",
        "confidence_intervals",
        "contrasts",
        "exclusions",
    }
    assert set(result["metrics"]["method"]) == set(LINEAGE_METHODS)
    assert set(result["metrics"]["n_queries"]) == {6}
    assert set(result["metrics"]["n_families"]) == {3}
    assert result["predictions"]["endpoint"].eq(
        "cross_fitted_observed_family_representative"
    ).all()


def test_gallery_accepts_explicit_feature_availability_without_inferring_files() -> None:
    manifest = _manifest(("org/a",))
    manifest = manifest.drop(columns="artifact_available")

    gallery, exclusions = build_root_gallery(
        manifest,
        available_model_ids=["org--a", "derived/a"],
    )

    assert gallery["model_id"].tolist() == ["org/a"]
    assert exclusions.empty


def test_gallery_requires_an_explicit_availability_contract() -> None:
    with pytest.raises(ValueError, match="availability"):
        build_root_gallery(_manifest(("org/a",)).drop(columns="artifact_available"))


def test_paper_style_alpha_is_depth_aligned_role_exact_and_dks_named() -> None:
    query = _one_trace([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
    same = _one_trace([1.0, 2.0, 3.0, 4.0, 5.0], [0, 1, 2, 3, 4])
    reversed_root = _one_trace([5.0, 4.0, 3.0, 2.0, 1.0], [0, 1, 2, 3, 4])

    same_score = paper_style_alpha_similarity(query, same)
    reversed_score = paper_style_alpha_similarity(query, reversed_root)

    assert same_score == pytest.approx(1.0)
    assert reversed_score == pytest.approx(-1.0)
    assert PAPER_STYLE_ALPHA_DKS == "paper_style_alpha_dks"
    assert "stored_dks" in PAPER_STYLE_ALPHA_DKS_PROVENANCE
    assert "not_fixed_hill" in PAPER_STYLE_ALPHA_DKS_PROVENANCE


def test_paper_style_alpha_requires_all_roles_and_masks_failure_sentinels() -> None:
    complete = _one_trace([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
    missing_role = complete.loc[complete["role"] != "mlp_down"].copy()
    invalid = complete.copy()
    invalid.loc[invalid["role"] == "attn_q", "alpha"] = -1.0

    assert np.isnan(paper_style_alpha_similarity(complete, missing_role))
    assert np.isnan(paper_style_alpha_similarity(complete, invalid))


def test_paper_style_cross_depth_alignment_skips_extra_layers_without_stretching_roles() -> None:
    short = _one_trace([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
    long = _one_trace([100.0, 1.0, 2.0, 3.0, -100.0], [0, 1, 2, 3, 4])

    assert paper_style_alpha_similarity(short, long) == pytest.approx(1.0)


def test_paper_style_alpha_rejects_missing_or_mixed_estimator_provenance() -> None:
    valid = _one_trace([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
    missing = valid.drop(columns="alpha_estimator")
    mixed = valid.copy()
    mixed.loc[mixed.index[0], "alpha_estimator"] = "fixed_hill"

    with pytest.raises(ValueError, match="alpha_estimator"):
        paper_style_alpha_similarity(valid, missing)
    with pytest.raises(ValueError, match="mixed|expected"):
        paper_style_alpha_similarity(valid, mixed)


def test_gallery_scaler_uses_gallery_rows_only() -> None:
    features = pd.DataFrame(
        [
            {"model_id": "r/one", "M__x": 0.0, "M__y": 2.0},
            {"model_id": "r/two", "M__x": 2.0, "M__y": 4.0},
            {"model_id": "query/outlier", "M__x": 10_000.0, "M__y": 10_000.0},
        ]
    )

    scaler = fit_gallery_scaler(
        features,
        gallery_model_ids=["r/one", "r/two"],
        columns=["M__x", "M__y"],
    )

    assert scaler.gallery_model_ids == ("r/one", "r/two")
    assert scaler.center == pytest.approx((1.0, 3.0))


def test_shared_feature_similarity_abstains_below_explicit_coverage() -> None:
    features = pd.DataFrame(
        [
            {"model_id": "r/one", "M__x": 0.0, "M__y": 2.0},
            {"model_id": "r/two", "M__x": 2.0, "M__y": 4.0},
        ]
    )
    scaler = fit_gallery_scaler(
        features,
        gallery_model_ids=["r/one", "r/two"],
        columns=["M__x", "M__y"],
    )

    result = shared_feature_similarity(
        pd.Series({"M__x": 1.0, "M__y": np.nan}),
        pd.Series({"M__x": 0.0, "M__y": 2.0}),
        scaler,
        min_coverage=0.75,
    )

    assert result.coverage == pytest.approx(0.5)
    assert np.isnan(result.score)
    assert result.abstained is True


def test_ranking_uses_all_methods_all_roots_and_one_common_query_population() -> None:
    manifest = _manifest()
    features = _features(manifest)
    trajectories = _trajectories(manifest)

    candidates, predictions, exclusions = rank_confirmed_queries(
        manifest,
        features,
        trajectories,
        architecture_column="architecture",
        min_feature_coverage=1.0,
        min_pair_coverage=1.0,
    )

    queries = set(manifest.loc[manifest["model_id"].str.startswith("derived/"), "model_id"])
    roots = {"org/a", "org/b", "org/c"}
    assert tuple(predictions["method"].drop_duplicates()) == LINEAGE_METHODS
    assert set(predictions["query_model_id"]) == queries
    assert set(candidates["candidate_root_id"]) == roots
    assert len(candidates) == len(LINEAGE_METHODS) * len(queries) * len(roots)
    assert len(predictions) == len(LINEAGE_METHODS) * len(queries)
    assert predictions["correct"].all()
    assert not predictions["abstained"].any()
    assert exclusions.empty
    paper_rows = predictions.loc[predictions["method"].eq(PAPER_STYLE_ALPHA_DKS)]
    assert set(paper_rows["estimator_provenance"]) == {
        PAPER_STYLE_ALPHA_DKS_PROVENANCE
    }


def test_common_intersection_excludes_a_query_from_every_method() -> None:
    manifest = _manifest()
    features = _features(manifest)
    trajectories = _trajectories(manifest)
    features.loc[features["model_id"].eq("derived/a"), "H__alpha"] = np.nan

    _, predictions, exclusions = rank_confirmed_queries(
        manifest,
        features,
        trajectories,
        min_feature_coverage=1.0,
        min_pair_coverage=1.0,
    )

    assert "derived/a" not in set(predictions["query_model_id"])
    excluded = exclusions.loc[exclusions["model_id"].eq("derived/a")]
    assert excluded["reason"].tolist() == ["query_ineligible_common_intersection"]


def test_architecture_matched_rerank_is_restricted_to_matching_roots() -> None:
    manifest = _manifest(("org/a", "org/b"))
    features = _features(manifest)
    trajectories = _trajectories(manifest)
    features.loc[features["model_id"].eq("derived/a"), ["M__depth", "M__width"]] = [
        9.8,
        9.9,
    ]

    candidates, predictions, _ = rank_confirmed_queries(
        manifest,
        features,
        trajectories,
        architecture_column="architecture",
        min_feature_coverage=1.0,
        min_pair_coverage=1.0,
    )
    row = predictions.loc[
        predictions["method"].eq("config")
        & predictions["query_model_id"].eq("derived/a")
    ].iloc[0]
    restricted = candidates.loc[
        candidates["method"].eq("config")
        & candidates["query_model_id"].eq("derived/a")
        & candidates["architecture_match"]
    ]

    assert row["predicted_family"] == "org/b"
    assert row["architecture_matched_predicted_family"] == "org/a"
    assert bool(row["architecture_matched_correct"]) is True
    assert restricted["candidate_family"].tolist() == ["org/a"]
    assert restricted["architecture_rank"].tolist() == [1]


def test_architecture_rerank_counts_wrong_match_when_true_root_architecture_differs() -> None:
    manifest = _manifest(("org/a", "org/b"))
    manifest.loc[manifest["model_id"].eq("org/a"), "architecture"] = "arch-root-a"
    manifest.loc[manifest["model_id"].eq("derived/a"), "architecture"] = "arch-query"
    manifest.loc[manifest["model_id"].eq("org/b"), "architecture"] = "arch-query"
    manifest.loc[manifest["model_id"].eq("derived/b"), "architecture"] = "arch-query"

    _, predictions, _ = rank_confirmed_queries(
        manifest,
        _features(manifest),
        _trajectories(manifest),
        architecture_column="architecture",
        min_feature_coverage=1.0,
        min_pair_coverage=1.0,
    )
    row = predictions.loc[
        predictions["method"].eq("config")
        & predictions["query_model_id"].eq("derived/a")
    ].iloc[0]

    assert bool(row["architecture_matched_eligible"]) is True
    assert row["architecture_matched_predicted_family"] == "org/b"
    assert bool(row["architecture_matched_correct"]) is False


def test_architecture_rerank_rejects_label_or_provenance_columns() -> None:
    manifest = _manifest(("org/a", "org/b"))

    with pytest.raises(ValueError, match="architecture_column"):
        rank_confirmed_queries(
            manifest,
            _features(manifest),
            _trajectories(manifest),
            architecture_column="audited_base_family",
        )


def test_common_intersection_ignores_optional_log_alpha_norm_and_missingness_indicators() -> None:
    manifest = _manifest(("org/a", "org/b"))
    rows = []
    for row in manifest.itertuples(index=False):
        family_value = 0.0 if row.audited_base_family == "org/a" else 10.0
        rows.append(
            {
                "model_id": row.model_id,
                "M__layer_count": 2.0,
                "S__attn_q__log_rms__median": family_value,
                "S__attn_q__log_rms__missing_fraction": 0.0,
                "H__attn_q__alpha__median": family_value + 2.0,
                "H__attn_q__alpha__missing_fraction": 0.0,
                "H__attn_q__log_alpha_norm__median": np.nan,
                "H__attn_q__log_alpha_norm__missing_fraction": 1.0,
            }
        )

    _, predictions, _ = rank_confirmed_queries(
        manifest,
        pd.DataFrame(rows),
        _trajectories(manifest),
        min_feature_coverage=0.8,
        min_pair_coverage=0.4,
    )

    assert set(predictions["query_model_id"]) == {"derived/a", "derived/b"}


def test_root_macro_accuracy_does_not_reward_large_families() -> None:
    predictions = pd.DataFrame(
        [
            *[
                {"true_family": "large/root", "correct": True, "true_rank": 1}
                for _ in range(9)
            ],
            {"true_family": "small/root", "correct": False, "true_rank": 2},
        ]
    )

    assert root_macro_top1(predictions) == pytest.approx(0.5)
    metrics = closed_set_metrics(predictions)
    assert metrics["root_macro_top1"] == pytest.approx(0.5)
    assert metrics["micro_top1"] == pytest.approx(0.9)
    assert metrics["mrr"] == pytest.approx(0.75)


def test_closed_set_metrics_report_recall_mrr_and_architecture_error() -> None:
    predictions = pd.DataFrame(
        [
            {
                "true_family": "a/root",
                "correct": True,
                "true_rank": 1,
                "architecture_matched_eligible": True,
                "architecture_matched_correct": True,
            },
            {
                "true_family": "b/root",
                "correct": False,
                "true_rank": 4,
                "architecture_matched_eligible": True,
                "architecture_matched_correct": False,
            },
            {
                "true_family": "c/root",
                "correct": False,
                "true_rank": 6,
                "architecture_matched_eligible": False,
                "architecture_matched_correct": False,
            },
        ]
    )

    metrics = closed_set_metrics(predictions)

    assert metrics["recall@3"] == pytest.approx(1.0 / 3.0)
    assert metrics["recall@5"] == pytest.approx(2.0 / 3.0)
    assert metrics["mrr"] == pytest.approx((1.0 + 0.25 + 1.0 / 6.0) / 3.0)
    assert metrics["architecture_matched_error"] == pytest.approx(0.5)


def test_family_bootstrap_is_deterministic_and_reports_units() -> None:
    predictions = pd.DataFrame(
        [
            {"method": "config", "true_family": "a", "correct": True, "true_rank": 1},
            {"method": "config", "true_family": "a", "correct": False, "true_rank": 2},
            {"method": "config", "true_family": "b", "correct": False, "true_rank": 3},
            {"method": "config", "true_family": "c", "correct": True, "true_rank": 1},
        ]
    )

    left = bootstrap_closed_set_metrics(
        predictions, bootstrap_replicates=80, random_state=17
    )
    right = bootstrap_closed_set_metrics(
        predictions, bootstrap_replicates=80, random_state=17
    )

    pd.testing.assert_frame_equal(left, right)
    assert set(left["bootstrap_unit"]) == {"root_family"}
    assert set(left["bootstrap_replicates"]) == {80}
    assert set(left["n_families"]) == {3}
    top1 = left.loc[left["metric"].eq("root_macro_top1")].iloc[0]
    assert top1["ci_low"] <= top1["estimate"] <= top1["ci_high"]


def test_family_bootstrap_uses_paired_draws_across_methods() -> None:
    base = pd.DataFrame(
        [
            {"true_family": "a", "correct": True, "true_rank": 1},
            {"true_family": "b", "correct": False, "true_rank": 2},
            {"true_family": "c", "correct": False, "true_rank": 4},
            {"true_family": "d", "correct": True, "true_rank": 1},
        ]
    )
    predictions = pd.concat(
        [base.assign(method="config"), base.assign(method="combined")],
        ignore_index=True,
    )

    intervals = bootstrap_closed_set_metrics(
        predictions, bootstrap_replicates=37, random_state=19
    )
    config = intervals.loc[
        intervals["method"].eq("config"), ["metric", "ci_low", "ci_high"]
    ].reset_index(drop=True)
    combined = intervals.loc[
        intervals["method"].eq("combined"), ["metric", "ci_low", "ci_high"]
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(config, combined)


def test_family_bootstrap_vectorizes_replicates_without_recomputing_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spectral_research.study.lineage as lineage_module

    predictions = pd.DataFrame(
        [
            {
                "method": "combined",
                "true_family": family,
                "correct": index % 2 == 0,
                "true_rank": index + 1,
            }
            for index, family in enumerate(("a", "b", "c", "d"))
        ]
    )
    original = lineage_module._closed_set_metrics
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(lineage_module, "_closed_set_metrics", counted)

    bootstrap_closed_set_metrics(
        predictions,
        bootstrap_replicates=80,
        random_state=19,
    )

    assert calls == 1


def test_family_bootstrap_reports_paired_combined_minus_paper_contrasts() -> None:
    rows = []
    for family_index, family in enumerate(("a", "b", "c", "d")):
        rows.extend(
            [
                {
                    "method": PAPER_STYLE_ALPHA_DKS,
                    "query_model_id": f"{family}/q",
                    "true_family": family,
                    "correct": family_index == 0,
                    "true_rank": 1 if family_index == 0 else 3,
                },
                {
                    "method": "combined",
                    "query_model_id": f"{family}/q",
                    "true_family": family,
                    "correct": True,
                    "true_rank": 1,
                },
            ]
        )

    contrasts = bootstrap_closed_set_contrasts(
        pd.DataFrame(rows), bootstrap_replicates=80, random_state=23
    )

    top1 = contrasts.loc[contrasts["metric"].eq("root_macro_top1")].iloc[0]
    assert top1["contrast"] == "combined-paper_style_alpha_dks"
    assert top1["estimate"] == pytest.approx(0.75)
    assert top1["ci_low"] <= top1["estimate"] <= top1["ci_high"]


def test_leave_one_family_out_never_leaves_held_root_in_gallery() -> None:
    manifest = _manifest()

    folds = list(unknown_family_folds(manifest))

    assert len(folds) == 3
    for fold in folds:
        assert fold.held_out_family not in set(fold.gallery["root_family"])
        assert set(fold.unknown_queries["root_family"]) == {fold.held_out_family}
        assert fold.held_out_family not in set(fold.known_queries["root_family"])


def test_open_set_metrics_reward_correct_known_unknown_ordering() -> None:
    labels = np.array([1, 1, 0, 0], dtype=int)
    correct = np.array([True, True, False, False])

    good = compute_open_set_metrics(
        np.array([0.9, 0.8, 0.2, 0.1]), labels, correct
    )
    bad = compute_open_set_metrics(
        np.array([0.2, 0.1, 0.9, 0.8]), labels, correct
    )

    assert good["oscr"] == pytest.approx(1.0)
    assert good["auroc"] == pytest.approx(1.0)
    assert good["auprc"] == pytest.approx(1.0)
    assert good["fpr95"] == pytest.approx(0.0)
    assert good["risk_coverage_area"] < bad["risk_coverage_area"]


def test_open_set_metrics_count_abstentions_as_rejections_and_report_coverage() -> None:
    metrics = compute_open_set_metrics(
        np.array([0.9, np.nan, 0.1, np.nan]),
        np.array([1, 1, 0, 0], dtype=int),
        np.array([True, False, False, False]),
    )

    assert metrics["scorable_coverage"] == pytest.approx(0.5)
    assert metrics["abstention_rate"] == pytest.approx(0.5)
    assert metrics["auroc"] < 1.0


def test_unknown_family_evaluation_exposes_fold_gallery_and_all_metrics() -> None:
    manifest = _manifest()
    result = evaluate_unknown_family_rejection(
        manifest,
        _features(manifest),
        _trajectories(manifest),
        min_feature_coverage=1.0,
        min_pair_coverage=1.0,
    )

    assert set(result) == {
        "fold_gallery",
        "scores",
        "fold_metrics",
        "metrics",
        "risk_coverage",
        "exclusions",
    }
    fold_gallery = result["fold_gallery"]
    assert not (
        fold_gallery["held_out_family"] == fold_gallery["gallery_family"]
    ).any()
    assert set(result["metrics"]["metric"]) == {
        "oscr",
        "auroc",
        "auprc",
        "fpr95",
        "risk_coverage_area",
        "scorable_coverage",
        "abstention_rate",
    }
    assert set(result["scores"]["method"]) == set(LINEAGE_METHODS)
    assert result["metrics"]["aggregation"].eq("held_out_family_macro").all()
    assert result["fold_metrics"]["held_out_family"].nunique() == 3


def test_open_set_ranking_does_not_retain_unused_candidate_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spectral_research.study.lineage as lineage_module

    original = lineage_module._rank_queries_with_gallery
    collection_flags: list[object] = []

    def observed(*args, **kwargs):
        collection_flags.append(kwargs.get("collect_candidates"))
        return original(*args, **kwargs)

    monkeypatch.setattr(lineage_module, "_rank_queries_with_gallery", observed)
    manifest = _manifest()
    lineage_module.evaluate_unknown_family_rejection(
        manifest,
        _features(manifest),
        _trajectories(manifest),
        min_feature_coverage=1.0,
        min_pair_coverage=1.0,
    )

    assert collection_flags and set(collection_flags) == {False}


def test_clustering_fits_without_labels_then_scores_frozen_assignments() -> None:
    features = pd.DataFrame(
        [
            {"model_id": "a/1", "M__x": 0.0, "S__x": 0.0, "H__x": 0.0},
            {"model_id": "a/2", "M__x": 0.1, "S__x": 0.2, "H__x": 0.1},
            {"model_id": "a/3", "M__x": 0.2, "S__x": 0.1, "H__x": 0.2},
            {"model_id": "b/1", "M__x": 9.8, "S__x": 10.0, "H__x": 9.9},
            {"model_id": "b/2", "M__x": 10.0, "S__x": 9.8, "H__x": 10.1},
            {"model_id": "b/3", "M__x": 10.2, "S__x": 10.1, "H__x": 10.0},
        ]
    )
    labels = pd.DataFrame(
        {
            "model_id": features["model_id"],
            "audited_base_family": ["root/a"] * 3 + ["root/b"] * 3,
            "label_status": ["confirmed"] * 6,
        }
    )
    permuted = labels.copy()
    permuted["audited_base_family"] = ["x", "y", "x", "y", "x", "y"]

    result = cluster_lineage_diagnostics(
        features, labels, n_clusters=2, random_state=7, min_feature_coverage=1.0
    )
    permuted_result = cluster_lineage_diagnostics(
        features, permuted, n_clusters=2, random_state=7, min_feature_coverage=1.0
    )

    pd.testing.assert_frame_equal(result["assignments"], permuted_result["assignments"])
    metrics = result["metrics"].iloc[0]
    assert metrics["ari"] == pytest.approx(1.0)
    assert metrics["nmi"] == pytest.approx(1.0)
    assert metrics["root_macro_purity"] == pytest.approx(1.0)
    assert metrics["silhouette"] > 0.9


def test_cluster_count_rule_uses_post_coverage_population() -> None:
    features = pd.DataFrame(
        [
            {
                "model_id": f"model/{index}",
                "M__x": float(index),
                "S__x": float(index) if index < 8 else np.nan,
                "H__x": float(index) if index < 8 else np.nan,
            }
            for index in range(10)
        ]
    )
    labels = pd.DataFrame(
        {
            "model_id": features["model_id"],
            "audited_base_family": ["root/a"] * 5 + ["root/b"] * 5,
            "label_status": ["confirmed"] * 10,
        }
    )

    result = cluster_lineage_diagnostics(
        features,
        labels,
        n_clusters=None,
        cluster_count_rule="sqrt_half",
        random_state=7,
        min_feature_coverage=0.8,
    )

    assert len(result["assignments"]) == 8
    assert result["metrics"].iloc[0]["n_clusters"] == 2


def test_root_macro_cluster_purity_penalizes_total_cluster_collapse() -> None:
    assignments = pd.DataFrame(
        {
            "root_family": ["a", "a", "b", "b", "c", "c"],
            "cluster": [0, 0, 0, 0, 0, 0],
        }
    )

    assert _root_macro_purity(assignments) == pytest.approx(1.0 / 3.0)


def test_exact_parent_requires_single_parent_and_two_same_family_negatives() -> None:
    manifest = pd.DataFrame(
        [
            {"model_id": "fam/parent", "audited_base_family": "fam/root", "label_status": "confirmed"},
            {"model_id": "fam/child", "audited_base_family": "fam/root", "label_status": "confirmed"},
            {"model_id": "fam/n1", "audited_base_family": "fam/root", "label_status": "confirmed"},
            {"model_id": "fam/n2", "audited_base_family": "fam/root", "label_status": "confirmed"},
            {"model_id": "multi/child", "audited_base_family": "fam/root", "label_status": "confirmed"},
            {"model_id": "small/parent", "audited_base_family": "small/root", "label_status": "confirmed"},
            {"model_id": "small/child", "audited_base_family": "small/root", "label_status": "confirmed"},
        ]
    )
    values = {
        "fam/parent": 0.0,
        "fam/child": 0.1,
        "fam/n1": 5.0,
        "fam/n2": 9.0,
        "multi/child": 0.2,
        "small/parent": 1.0,
        "small/child": 1.1,
    }
    features = pd.DataFrame(
        [
            {"model_id": model_id, "M__x": value, "S__x": value, "H__x": value}
            for model_id, value in values.items()
        ]
    )
    edges = pd.DataFrame(
        [
            {"parent_id": "fam/parent", "child_id": "fam/child"},
            {"parent_id": "fam/parent", "child_id": "multi/child"},
            {"parent_id": "fam/n1", "child_id": "multi/child"},
            {"parent_id": "small/parent", "child_id": "small/child"},
        ]
    )

    result = evaluate_exact_parent_diagnostic(
        edges,
        manifest,
        features,
        min_feature_coverage=1.0,
        min_pair_coverage=1.0,
    )

    assert result["predictions"]["child_id"].tolist() == ["fam/child"]
    prediction = result["predictions"].iloc[0]
    assert prediction["candidate_count"] == 3
    assert prediction["true_parent_rank"] == 1
    assert prediction["reciprocal_rank"] == pytest.approx(1.0)
    assert prediction["claim_scope"] == "sensitivity_only"
    assert prediction["edge_provenance"] == "graph_export_unverified_direct_parent"
    assert result["metrics"].loc[0, "estimator_provenance"] == (
        "candidate_gallery_scaling_excludes_query_child"
    )
    assert set(result["candidates"]["candidate_id"]) == {
        "fam/parent",
        "fam/n1",
        "fam/n2",
    }
    reasons = set(result["exclusions"]["reason"])
    assert "ambiguous_multiple_graph_parents" in reasons
    assert "insufficient_same_family_hard_negatives" in reasons


def test_exact_parent_requires_two_scorable_hard_negatives() -> None:
    model_ids = ("fam/parent", "fam/child", "fam/n1", "fam/n2")
    manifest = pd.DataFrame(
        {
            "model_id": model_ids,
            "audited_base_family": ["fam/root"] * len(model_ids),
            "label_status": ["confirmed"] * len(model_ids),
        }
    )
    features = pd.DataFrame(
        [
            {
                "model_id": model_id,
                "M__x": value if side == "x" else np.nan,
                "M__y": value if side == "y" else np.nan,
                "S__x": value if side == "x" else np.nan,
                "S__y": value if side == "y" else np.nan,
                "H__x": value if side == "x" else np.nan,
                "H__y": value if side == "y" else np.nan,
            }
            for model_id, value, side in (
                ("fam/parent", 0.0, "x"),
                ("fam/child", 0.1, "x"),
                ("fam/n1", 5.0, "x"),
                ("fam/n2", 9.0, "y"),
            )
        ]
    )
    edges = pd.DataFrame(
        [{"parent_id": "fam/parent", "child_id": "fam/child"}]
    )

    result = evaluate_exact_parent_diagnostic(
        edges,
        manifest,
        features,
        min_feature_coverage=0.5,
        min_pair_coverage=0.5,
    )

    assert result["predictions"].empty
    assert result["exclusions"]["reason"].tolist() == [
        "insufficient_scorable_same_family_hard_negatives"
    ]
