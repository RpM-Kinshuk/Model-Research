from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pandas as pd
import pytest

import spectral_research.study.contracts as contracts

from spectral_research.study.contracts import (
    RunSettings,
    StudyInputs,
    build_connected_components,
    build_model_manifest,
    build_provenance,
    file_digest,
    load_graph_edges,
    load_graph_nodes,
    normalize_graph_edges,
    path_digest,
    study_run_id,
    write_provenance,
)


def _inputs(tmp_path: Path, **overrides: Path | None) -> StudyInputs:
    values: dict[str, Path | None] = {
        "metadata": tmp_path / "metadata.csv",
        "labels": tmp_path / "labels.csv",
        "graph": tmp_path / "graph.json",
        "development_stats": tmp_path / "development_stats",
        "development_targets": tmp_path / "development_targets.csv",
        "cache_root": tmp_path / "cache",
        "output_root": tmp_path / "runs",
        "external_stats": None,
        "external_targets": None,
    }
    values.update(overrides)
    return StudyInputs(**values)


def test_study_inputs_are_explicit_resolved_and_immutable(tmp_path):
    inputs = _inputs(tmp_path)

    assert inputs.metadata == (tmp_path / "metadata.csv").resolve()
    assert inputs.source_paths() == {
        "metadata": (tmp_path / "metadata.csv").resolve(),
        "labels": (tmp_path / "labels.csv").resolve(),
        "graph": (tmp_path / "graph.json").resolve(),
        "development_stats": (tmp_path / "development_stats").resolve(),
        "development_targets": (tmp_path / "development_targets.csv").resolve(),
    }
    with pytest.raises(FrozenInstanceError):
        inputs.metadata = tmp_path / "other.csv"


def test_study_inputs_require_external_stats_and_targets_together(tmp_path):
    with pytest.raises(ValueError, match="external_stats and external_targets"):
        _inputs(tmp_path, external_stats=tmp_path / "external_stats")


@pytest.mark.parametrize("destination_name", ["cache_root", "output_root"])
def test_study_inputs_reject_destinations_inside_source_trees(
    tmp_path: Path, destination_name: str
) -> None:
    source = tmp_path / "development_stats"

    with pytest.raises(ValueError, match="source and destination paths must be disjoint"):
        _inputs(tmp_path, **{destination_name: source / "generated"})


def test_study_inputs_reject_overlapping_destination_roots(tmp_path: Path) -> None:
    cache_root = tmp_path / "generated"

    with pytest.raises(ValueError, match="cache_root and output_root must be disjoint"):
        _inputs(
            tmp_path,
            cache_root=cache_root,
            output_root=cache_root / "runs",
        )


def test_run_settings_are_serializable_and_immutable():
    settings = RunSettings(
        schema_version="study-v1",
        random_seed=17,
        bootstrap_replicates=200,
    )

    assert settings.to_dict() == {
        "schema_version": "study-v1",
        "random_seed": 17,
        "bootstrap_replicates": 200,
        "signflip_replicates": 10_000,
        "outer_folds": 5,
        "inner_folds": 4,
        "ridge_alphas": [0.01, 0.1, 1.0, 10.0, 100.0],
        "minimum_components": 10,
        "minimum_lineage_families": 10,
        "min_feature_coverage": 0.8,
        "min_pair_coverage": 0.8,
        "min_role_coverage": 0.8,
        "cluster_count_rule": "sqrt_half",
        "external_evidence_role": "confirmatory_locked",
    }
    with pytest.raises(FrozenInstanceError):
        settings.random_seed = 18


def test_file_digest_depends_on_content_not_absolute_location(tmp_path):
    left = tmp_path / "left.csv"
    right = tmp_path / "nested" / "right.csv"
    right.parent.mkdir()
    left.write_text("model_id\na\n", encoding="utf-8")
    right.write_text("model_id\na\n", encoding="utf-8")

    assert file_digest(left) == file_digest(right)

    right.write_text("model_id\nb\n", encoding="utf-8")
    assert file_digest(left) != file_digest(right)


def test_directory_digest_depends_on_relative_names_and_content_not_location(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "a.csv").write_text("model_id\na\n", encoding="utf-8")
    (right / "a.csv").write_text("model_id\na\n", encoding="utf-8")

    assert path_digest(left) == path_digest(right)

    (right / "a.csv").write_text("model_id\nb\n", encoding="utf-8")
    assert path_digest(left) != path_digest(right)


def test_run_id_changes_with_source_bytes_or_settings_but_not_root_location(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "source.csv").write_text("model_id\na\n", encoding="utf-8")
    (second / "source.csv").write_text("model_id\na\n", encoding="utf-8")
    settings = RunSettings("study-v1", random_seed=7, bootstrap_replicates=100)

    first_id = study_run_id({"source": first}, settings)
    assert first_id == study_run_id({"source": second}, settings)

    (second / "source.csv").write_text("model_id\nb\n", encoding="utf-8")
    assert first_id != study_run_id({"source": second}, settings)
    assert first_id != study_run_id(
        {"source": first},
        RunSettings("study-v2", random_seed=7, bootstrap_replicates=100),
    )


def test_run_id_includes_every_execution_setting(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("model_id\na\n", encoding="utf-8")
    baseline = RunSettings("study-v1", random_seed=7, bootstrap_replicates=100)
    changed_cv = RunSettings(
        "study-v1",
        random_seed=7,
        bootstrap_replicates=100,
        outer_folds=3,
    )
    changed_grid = RunSettings(
        "study-v1",
        random_seed=7,
        bootstrap_replicates=100,
        ridge_alphas=(0.1, 1.0),
    )

    baseline_id = study_run_id({"source": source}, baseline)
    assert study_run_id({"source": source}, changed_cv) != baseline_id
    assert study_run_id({"source": source}, changed_grid) != baseline_id


def test_run_id_can_include_explicit_implementation_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("model_id\na\n", encoding="utf-8")
    settings = RunSettings("study-v1", random_seed=7, bootstrap_replicates=100)

    first = study_run_id(
        {"source": source}, settings, identity={"implementation": "v1"}
    )
    second = study_run_id(
        {"source": source}, settings, identity={"implementation": "v2"}
    )

    assert first != second


def test_load_graph_edges_normalizes_real_export_shape(tmp_path):
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "nodes": [{"id": "org/root"}, {"id": "org/child"}],
                "edges": [
                    {"source": " org/root ", "target": "org/child", "id": "7"},
                    {"source": "org/root", "target": "org/child", "id": "8"},
                ],
            }
        ),
        encoding="utf-8",
    )

    edges = load_graph_edges(graph_path)

    assert edges.to_dict("records") == [
        {"parent_id": "org/root", "child_id": "org/child"}
    ]

    assert load_graph_nodes(graph_path) == ("org/child", "org/root")


def test_graph_contract_exposes_node_metadata_without_a_second_population_file(
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "org/child",
                        "attributes": {
                            "base_model_clean": "org/root",
                            "base_model_relation": "finetune",
                            "architectures": "ExampleForCausalLM",
                        },
                    },
                    {"id": "org/root", "attributes": {}},
                ],
                "edges": [{"source": "org/root", "target": "org/child"}],
            }
        ),
        encoding="utf-8",
    )

    edges, nodes, metadata = contracts.load_graph_contract(graph_path)

    assert edges.to_dict("records") == [
        {"parent_id": "org/root", "child_id": "org/child"}
    ]
    assert nodes == ("org/child", "org/root")
    child = metadata.set_index("model_id").loc["org/child"]
    assert child["source_model"] == "org/root"
    assert child["base_model_relation"] == "finetune"
    assert child["graph_architecture"] == "ExampleForCausalLM"


def test_analysis_metadata_uses_uncapped_data_union_and_ignores_lookup_only_rows() -> None:
    metadata = pd.DataFrame(
        [
            {
                "model_id": "org/labeled",
                "source_model": "explicit/root",
                "base_model_relation": "manual",
            },
            {
                "model_id": "org/lookup-only",
                "source_model": "org/root",
                "base_model_relation": "finetune",
            },
        ]
    )
    graph_metadata = pd.DataFrame(
        [
            {
                "model_id": "org/stats",
                "source_model": "org/root",
                "base_model_relation": "adapter",
                "graph_architecture": "GraphArch",
            },
            {
                "model_id": "org/labeled",
                "source_model": "graph/root",
                "base_model_relation": "graph",
                "graph_architecture": "GraphArch",
            },
        ]
    )
    population_tables = {
        "development_stats": pd.DataFrame({"model_id": ["org/stats"]}),
        "development_targets": pd.DataFrame({"model_id": ["org/target"]}),
        "labels": pd.DataFrame({"model_id": ["org--labeled"]}),
    }

    derived, exclusions = contracts.derive_analysis_metadata(
        metadata,
        graph_metadata,
        population_tables,
    )

    assert derived["model_id"].tolist() == [
        "org/labeled",
        "org/stats",
        "org/target",
    ]
    by_id = derived.set_index("model_id")
    assert by_id.loc["org/labeled", "source_model"] == "explicit/root"
    assert by_id.loc["org/stats", "graph_architecture"] == "GraphArch"
    assert by_id.loc["org/target", "metadata_origin"] == "fallback"
    assert exclusions.loc[0, "model_id"] == "org/lookup-only"
    assert exclusions.loc[0, "reason"] == "metadata_without_analysis_source"


def test_graph_self_loops_are_quarantined_from_ancestry_evidence() -> None:
    normalized = normalize_graph_edges(
        pd.DataFrame(
            {
                "source": ["org/a", "org/a"],
                "target": ["org/a", "org/b"],
            }
        )
    )

    assert normalized.to_dict("records") == [
        {"parent_id": "org/a", "child_id": "org/b"}
    ]


def test_connected_components_include_paths_through_unselected_nodes():
    edges = pd.DataFrame(
        {
            "parent_id": ["org/root", "org/root", "other/root"],
            "child_id": ["org/a", "org/b", "other/child"],
        }
    )

    graph_nodes = ["org/root", "org/a", "org/b", "other/root", "other/child", "org/isolated"]
    first = build_connected_components(
        ["org/a", "org/b", "org/isolated"], edges, graph_nodes=graph_nodes
    )
    second = build_connected_components(
        ["org/isolated", "org/b", "org/a"],
        edges.iloc[::-1],
        graph_nodes=reversed(graph_nodes),
    )

    first_by_id = first.set_index("model_id")["ancestry_component"].to_dict()
    second_by_id = second.set_index("model_id")["ancestry_component"].to_dict()
    assert first_by_id == second_by_id
    assert first_by_id["org/a"] == first_by_id["org/b"]
    assert first_by_id["org/a"] != first_by_id["org/isolated"]


def test_connected_components_do_not_invent_groups_for_models_absent_from_graph():
    edges = pd.DataFrame(
        {"parent_id": ["org/root"], "child_id": ["org/child"]}
    )

    components = build_connected_components(
        ["org/child", "org/not-in-graph"],
        edges,
        graph_nodes=["org/root", "org/child"],
    ).set_index("model_id")

    assert components.loc["org/child", "component_provenance"] == "graph"
    assert pd.notna(components.loc["org/child", "ancestry_component"])
    assert components.loc["org/child", "component_size"] == 2
    assert components.loc["org/child", "component_edge_count"] == 1
    assert components.loc["org/not-in-graph", "component_provenance"] == ""
    assert pd.isna(components.loc["org/not-in-graph", "ancestry_component"])
    assert components.loc["org/not-in-graph", "component_size"] == 0
    assert components.loc["org/not-in-graph", "component_edge_count"] == 0


def test_graph_isolate_is_identified_as_weak_component() -> None:
    components = build_connected_components(
        ["org/isolate"],
        pd.DataFrame(columns=["parent_id", "child_id"]),
        graph_nodes=["org/isolate"],
    )

    assert components.loc[0, "component_provenance"] == "graph"
    assert components.loc[0, "component_size"] == 1
    assert components.loc[0, "component_edge_count"] == 0


def test_manifest_leakage_groups_ignore_unselected_common_descendants() -> None:
    metadata = pd.DataFrame(
        {
            "model_id": [
                "org/root-a",
                "org/root-b",
                "org/clean-child",
            ]
        }
    )
    edges = pd.DataFrame(
        {
            "parent_id": [
                "org/root-a",
                "org/root-a",
                "org/root-b",
                "org/merge",
            ],
            "child_id": [
                "org/clean-child",
                "org/merge",
                "org/merge",
                "org/post-merge",
            ],
        }
    )

    manifest, _ = build_model_manifest(
        metadata,
        pd.DataFrame(columns=["model_id"]),
        edges,
        graph_nodes=[
            "org/root-a",
            "org/root-b",
            "org/clean-child",
            "org/merge",
            "org/post-merge",
        ],
    )
    by_id = manifest.set_index("model_id")

    assert by_id.loc["org/root-a", "ancestry_component"] == by_id.loc[
        "org/clean-child", "ancestry_component"
    ]
    assert by_id.loc["org/root-a", "ancestry_component"] != by_id.loc[
        "org/root-b", "ancestry_component"
    ]
    assert by_id["component_grouping_method"].eq("single_source_root_v1").all()
    assert by_id.loc["org/root-a", "ancestor_closure_component"] != by_id.loc[
        "org/root-b", "ancestor_closure_component"
    ]


def test_manifest_leakage_groups_join_selected_merge_with_all_ancestors() -> None:
    metadata = pd.DataFrame(
        {
            "model_id": [
                "org/root-a",
                "org/root-b",
                "org/clean",
                "org/merge",
                "org/post-merge",
            ]
        }
    )
    edges = pd.DataFrame(
        {
            "parent_id": [
                "org/root-a",
                "org/root-a",
                "org/root-b",
                "org/merge",
            ],
            "child_id": [
                "org/clean",
                "org/merge",
                "org/merge",
                "org/post-merge",
            ],
        }
    )
    manifest, _ = build_model_manifest(
        metadata,
        pd.DataFrame(columns=["model_id"]),
        edges,
        graph_metadata=pd.DataFrame(
            {
                "model_id": metadata["model_id"],
                "base_model_relation": [
                    "source",
                    "source",
                    "finetune",
                    "merge",
                    "finetune",
                ],
            }
        ),
    )
    by_id = manifest.set_index("model_id")

    assert by_id["ancestry_component"].nunique() == 2
    assert pd.isna(by_id.loc["org/merge", "ancestry_component"])
    assert pd.isna(by_id.loc["org/post-merge", "ancestry_component"])
    assert by_id["ancestor_closure_component"].nunique() == 1
    assert by_id["ancestor_closure_component"].notna().all()


def test_manifest_joins_artifact_style_label_ids_to_canonical_repo_ids():
    metadata = pd.DataFrame([{"model_id": "org/root"}])
    labels = pd.DataFrame(
        [
            {
                "model_id": "org--root",
                "audited_base_family": "org/root",
                "label_status": "confirmed",
            }
        ]
    )

    manifest, exclusions = build_model_manifest(
        metadata,
        labels,
        pd.DataFrame(columns=["parent_id", "child_id"]),
        graph_nodes=["org/root"],
    )

    assert manifest.loc[0, "model_id"] == "org/root"
    assert manifest.loc[0, "label_status"] == "confirmed"
    assert exclusions.empty


def test_manifest_records_exclusions_and_uses_explicit_model_ids():
    metadata = pd.DataFrame(
        [
            {
                "model_id": "org/root",
                "artifact_path": "/stats/not-the-root-name.csv",
            },
            {
                "model_id": "org/child",
                "artifact_path": "/stats/also-not-the-child-name.csv",
            },
            {"model_id": "", "artifact_path": "/stats/guessable-name.csv"},
        ]
    )
    labels = pd.DataFrame(
        [
            {
                "model_id": "org/root",
                "root_family": "family-a",
                "label_provenance": "manual-freeze",
            },
            {
                "model_id": "org/child",
                "root_family": "family-a",
                "label_provenance": "manual-freeze",
            },
        ]
    )
    edges = pd.DataFrame(
        {"parent_id": ["org/root"], "child_id": ["org/child"]}
    )

    manifest, exclusions = build_model_manifest(metadata, labels, edges)

    assert manifest["model_id"].tolist() == ["org/child", "org/root"]
    assert manifest.set_index("model_id").loc["org/root", "artifact_path"] == (
        "/stats/not-the-root-name.csv"
    )
    assert manifest["ancestry_component"].nunique() == 1
    assert manifest["label_disposition"].tolist() == ["matched", "matched"]
    assert exclusions.to_dict("records") == [
        {
            "source_table": "metadata",
            "source_row": 2,
            "model_id": None,
            "reason": "missing_model_id",
        }
    ]


def test_manifest_records_labels_without_metadata():
    metadata = pd.DataFrame([{"model_id": "org/root"}])
    labels = pd.DataFrame(
        [
            {"model_id": "org/root", "root_family": "family-a"},
            {"model_id": "org/orphan", "root_family": "family-b"},
        ]
    )
    edges = pd.DataFrame(columns=["parent_id", "child_id"])

    manifest, exclusions = build_model_manifest(metadata, labels, edges)

    assert manifest["model_id"].tolist() == ["org/root"]
    assert exclusions.loc[0].to_dict() == {
        "source_table": "labels",
        "source_row": 1,
        "model_id": "org/orphan",
        "reason": "label_without_metadata",
    }


def test_manifest_exclusions_preserve_original_source_rows_after_filtering():
    metadata = pd.DataFrame([{"model_id": "org/root"}])
    labels = pd.DataFrame(
        [
            {"model_id": "org/root", "root_family": "family-a"},
            {"model_id": "", "root_family": "unknown"},
            {"model_id": "org/orphan", "root_family": "family-b"},
        ]
    )

    _, exclusions = build_model_manifest(
        metadata,
        labels,
        pd.DataFrame(columns=["parent_id", "child_id"]),
    )

    orphan = exclusions.loc[exclusions["reason"] == "label_without_metadata"].iloc[0]
    assert orphan["source_row"] == 2


@pytest.mark.parametrize("table_name", ["metadata", "labels"])
def test_manifest_rejects_duplicate_canonical_ids(table_name):
    metadata = pd.DataFrame([{"model_id": "org/a"}])
    labels = pd.DataFrame([{"model_id": "org/a", "root_family": "a"}])
    duplicate = pd.DataFrame(
        [
            {"model_id": "org/a", "root_family": "a"},
            {"model_id": " org/a ", "root_family": "a"},
        ]
    )
    if table_name == "metadata":
        metadata = duplicate.drop(columns="root_family")
    else:
        labels = duplicate

    with pytest.raises(ValueError, match=rf"duplicate canonical model_id.*{table_name}"):
        build_model_manifest(
            metadata,
            labels,
            pd.DataFrame(columns=["parent_id", "child_id"]),
        )


def test_provenance_serialization_records_sources_rows_and_exclusions(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text("model_id\norg/a\n", encoding="utf-8")
    exclusions = pd.DataFrame(
        [
            {
                "source_table": "labels",
                "source_row": 3,
                "model_id": "org/orphan",
                "reason": "label_without_metadata",
            }
        ]
    )
    settings = RunSettings("study-v1", random_seed=17, bootstrap_replicates=200)

    provenance = build_provenance(
        {"metadata": metadata_path},
        row_counts={"metadata": 1},
        settings=settings,
        manifest_rows=1,
        exclusions=exclusions,
    )
    output_path = write_provenance(tmp_path / "run" / "provenance.json", provenance)

    serialized = json.loads(output_path.read_text(encoding="utf-8"))
    assert serialized == provenance
    assert serialized["sources"]["metadata"] == {
        "path": str(metadata_path.resolve()),
        "sha256": file_digest(metadata_path),
        "rows": 1,
    }
    assert serialized["manifest"] == {
        "included_rows": 1,
        "excluded_rows": 1,
        "exclusions_by_reason": {"label_without_metadata": 1},
    }
