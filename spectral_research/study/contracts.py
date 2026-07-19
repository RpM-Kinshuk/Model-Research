from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class StudyInputs:
    """Explicit source and destination paths for one study run."""

    metadata: Path
    labels: Path
    graph: Path
    development_stats: Path
    development_targets: Path
    cache_root: Path
    output_root: Path
    external_stats: Path | None = None
    external_targets: Path | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "metadata",
            "labels",
            "graph",
            "development_stats",
            "development_targets",
            "cache_root",
            "output_root",
            "external_stats",
            "external_targets",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    Path(value).expanduser().resolve(),
                )
        if (self.external_stats is None) != (self.external_targets is None):
            raise ValueError(
                "external_stats and external_targets must be supplied together"
            )

        sources = self.source_paths()
        destinations = {
            "cache_root": self.cache_root,
            "output_root": self.output_root,
        }
        for destination_name, destination in destinations.items():
            for source_name, source in sources.items():
                if (
                    destination == source
                    or destination.is_relative_to(source)
                    or source.is_relative_to(destination)
                ):
                    raise ValueError(
                        "source and destination paths must be disjoint: "
                        f"{destination_name} overlaps {source_name}"
                    )
        if (
            self.cache_root == self.output_root
            or self.cache_root.is_relative_to(self.output_root)
            or self.output_root.is_relative_to(self.cache_root)
        ):
            raise ValueError("cache_root and output_root must be disjoint")

    def source_paths(self) -> dict[str, Path]:
        paths = {
            "metadata": self.metadata,
            "labels": self.labels,
            "graph": self.graph,
            "development_stats": self.development_stats,
            "development_targets": self.development_targets,
        }
        if self.external_stats is not None and self.external_targets is not None:
            paths["external_stats"] = self.external_stats
            paths["external_targets"] = self.external_targets
        return paths


@dataclass(frozen=True)
class RunSettings:
    """Reproducibility settings that contribute to a study run identity."""

    schema_version: str
    random_seed: int
    bootstrap_replicates: int
    signflip_replicates: int = 10_000
    outer_folds: int = 5
    inner_folds: int = 4
    ridge_alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    minimum_components: int = 10
    minimum_lineage_families: int = 10
    min_feature_coverage: float = 0.8
    min_pair_coverage: float = 0.8
    min_role_coverage: float = 0.8
    cluster_count_rule: str = "sqrt_half"
    external_evidence_role: str = "confirmatory_locked"
    holdout_repeats: int = 0
    holdout_test_fraction: float = 0.3

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise ValueError("schema_version must be non-empty")
        for field_name in (
            "random_seed",
            "bootstrap_replicates",
            "signflip_replicates",
            "outer_folds",
            "inner_folds",
            "minimum_components",
            "minimum_lineage_families",
            "holdout_repeats",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{field_name} must be an integer")
            object.__setattr__(self, field_name, int(value))
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        if self.bootstrap_replicates <= 0:
            raise ValueError("bootstrap_replicates must be positive")
        if self.signflip_replicates <= 0:
            raise ValueError("signflip_replicates must be positive")
        if self.outer_folds < 2:
            raise ValueError("outer_folds must be at least two")
        if self.inner_folds < 2:
            raise ValueError("inner_folds must be at least two")
        if self.minimum_components < 2:
            raise ValueError("minimum_components must be at least two")
        if self.minimum_lineage_families < 2:
            raise ValueError("minimum_lineage_families must be at least two")
        if self.holdout_repeats < 0:
            raise ValueError("holdout_repeats must be non-negative")

        for field_name in (
            "min_feature_coverage",
            "min_pair_coverage",
            "min_role_coverage",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{field_name} must be a real number")
            normalized = float(value)
            if not math.isfinite(normalized) or not 0.0 < normalized <= 1.0:
                raise ValueError(f"{field_name} must be finite and in (0, 1]")
            object.__setattr__(self, field_name, normalized)
        if self.cluster_count_rule != "sqrt_half":
            raise ValueError("cluster_count_rule must be 'sqrt_half'")
        if self.external_evidence_role not in {
            "confirmatory_locked",
            "reused_descriptive",
        }:
            raise ValueError(
                "external_evidence_role must be 'confirmatory_locked' or "
                "'reused_descriptive'"
            )
        if (
            isinstance(self.holdout_test_fraction, bool)
            or not isinstance(self.holdout_test_fraction, Real)
            or not math.isfinite(float(self.holdout_test_fraction))
            or not 0.0 < float(self.holdout_test_fraction) < 1.0
        ):
            raise ValueError("holdout_test_fraction must be finite and in (0, 1)")
        object.__setattr__(
            self, "holdout_test_fraction", float(self.holdout_test_fraction)
        )

        alphas = tuple(self.ridge_alphas)
        if not alphas:
            raise ValueError("ridge_alphas must contain at least one value")
        if any(
            isinstance(alpha, bool)
            or not isinstance(alpha, Real)
            or not math.isfinite(float(alpha))
            or float(alpha) <= 0.0
            for alpha in alphas
        ):
            raise ValueError("ridge_alphas must contain only finite positive values")
        normalized_alphas = tuple(float(alpha) for alpha in alphas)
        if len(normalized_alphas) != len(set(normalized_alphas)):
            raise ValueError("ridge_alphas must not contain duplicates")
        object.__setattr__(self, "ridge_alphas", normalized_alphas)

    def to_dict(self) -> dict[str, Any]:
        """Return every setting that determines run identity or inference."""

        serialized = asdict(self)
        serialized["ridge_alphas"] = list(self.ridge_alphas)
        if self.holdout_repeats == 0:
            serialized.pop("holdout_repeats")
            serialized.pop("holdout_test_fraction")
        return serialized

    def serialized(self) -> dict[str, Any]:
        """Backward-compatible alias for the complete settings contract."""

        return self.to_dict()


def file_digest(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest of file bytes."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def path_digest(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash one file or a directory tree by relative names and file bytes."""

    source = Path(path)
    if source.is_file():
        return file_digest(source, chunk_size=chunk_size)
    if not source.is_dir():
        raise FileNotFoundError(source)

    digest = hashlib.sha256()
    digest.update(b"directory-v1\0")
    files = sorted(candidate for candidate in source.rglob("*") if candidate.is_file())
    for candidate in files:
        relative = candidate.relative_to(source).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = candidate.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with candidate.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    return digest.hexdigest()


def study_run_id(
    source_paths: Mapping[str, str | Path],
    settings: RunSettings,
    *,
    source_digests: Mapping[str, str] | None = None,
    identity: Mapping[str, str] | None = None,
) -> str:
    """Return a location-independent identity for sources and frozen settings."""

    if source_digests is None:
        digests = {
            name: path_digest(source_paths[name]) for name in sorted(source_paths)
        }
    else:
        if set(source_digests) != set(source_paths):
            raise ValueError("source_digests must match source_paths exactly")
        digests = {name: str(source_digests[name]) for name in sorted(source_paths)}
    payload = {
        "identity": {
            str(name): str(value)
            for name, value in sorted((identity or {}).items())
        },
        "settings": settings.serialized(),
        "sources": digests,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "htsr_" + hashlib.sha256(encoded).hexdigest()[:20]


def canonical_model_id(value: object) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        raise ValueError("model identifiers must be scalar values") from None
    normalized = str(value).strip()
    if "/" not in normalized and "--" in normalized:
        normalized = normalized.replace("--", "/", 1)
    return normalized or None


def _edge_endpoint(value: object) -> object:
    if isinstance(value, Mapping):
        return value.get("id")
    return value


def normalize_graph_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """Normalize graph exports to unique parent/child canonical identifiers."""

    if {"parent_id", "child_id"}.issubset(edges.columns):
        parent_column, child_column = "parent_id", "child_id"
    elif {"source", "target"}.issubset(edges.columns):
        parent_column, child_column = "source", "target"
    else:
        raise ValueError(
            "graph edges require parent_id/child_id or source/target columns"
        )

    normalized_rows: list[dict[str, str]] = []
    for position, (_, row) in enumerate(edges.iterrows()):
        parent_id = canonical_model_id(_edge_endpoint(row[parent_column]))
        child_id = canonical_model_id(_edge_endpoint(row[child_column]))
        if parent_id is None or child_id is None:
            raise ValueError(f"graph edge at row {position} has a missing endpoint")
        if parent_id == child_id:
            continue
        normalized_rows.append({"parent_id": parent_id, "child_id": child_id})

    if not normalized_rows:
        return pd.DataFrame(columns=["parent_id", "child_id"])
    return (
        pd.DataFrame(normalized_rows)
        .drop_duplicates(ignore_index=True)
        .sort_values(["parent_id", "child_id"], kind="stable")
        .reset_index(drop=True)
    )


def load_graph_contract(
    path: str | Path,
) -> tuple[pd.DataFrame, tuple[str, ...], pd.DataFrame]:
    """Load graph edges, node IDs, and lineage metadata in one parse."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("graph JSON must contain an object")
    edge_records = payload.get("edges", payload.get("links"))
    node_records = payload.get("nodes")
    if not isinstance(edge_records, list):
        raise ValueError("graph JSON must contain an edges or links list")
    if not isinstance(node_records, list):
        raise ValueError("graph JSON must contain a nodes list")

    metadata_rows: list[dict[str, str]] = []
    seen_nodes: set[str] = set()
    for position, node in enumerate(node_records):
        value = node.get("id") if isinstance(node, Mapping) else node
        model_id = canonical_model_id(value)
        if model_id is None:
            raise ValueError(f"graph node at row {position} has a missing identifier")
        if model_id in seen_nodes:
            raise ValueError(f"duplicate canonical graph node: {model_id}")
        seen_nodes.add(model_id)
        attributes = node.get("attributes", {}) if isinstance(node, Mapping) else {}
        if not isinstance(attributes, Mapping):
            attributes = {}

        def node_value(key: str, fallback: str = "unknown") -> str:
            raw = node.get(key) if isinstance(node, Mapping) else None
            if raw is None:
                raw = attributes.get(key)
            text = str(raw).strip() if raw is not None else ""
            return text if text and text.casefold() not in {"nan", "none", "n/a"} else fallback

        metadata_rows.append(
            {
                "model_id": model_id,
                "source_model": node_value("base_model_clean"),
                "base_model_relation": node_value("base_model_relation"),
                "graph_architecture": node_value("architectures"),
            }
        )

    edges = normalize_graph_edges(pd.DataFrame.from_records(edge_records))
    metadata = pd.DataFrame.from_records(
        metadata_rows,
        columns=[
            "model_id",
            "source_model",
            "base_model_relation",
            "graph_architecture",
        ],
    ).sort_values("model_id", kind="stable", ignore_index=True)
    return edges, tuple(sorted(seen_nodes)), metadata


def load_graph_edges(path: str | Path) -> pd.DataFrame:
    """Load the repository's JSON graph export and normalize its edges."""

    return load_graph_contract(path)[0]


def load_graph_nodes(path: str | Path) -> tuple[str, ...]:
    """Load canonical node identifiers, including graph-defined isolates."""

    return load_graph_contract(path)[1]


class _DisjointSet:
    def __init__(self, nodes: Iterable[str]) -> None:
        self.parent = {node: node for node in nodes}

    def find(self, node: str) -> str:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def build_connected_components(
    model_ids: Iterable[object],
    graph_edges: pd.DataFrame,
    *,
    graph_nodes: Iterable[object] | None = None,
) -> pd.DataFrame:
    """Assign deterministic undirected ancestry components to selected models."""

    selected: list[str] = []
    for position, value in enumerate(model_ids):
        model_id = canonical_model_id(value)
        if model_id is None:
            raise ValueError(f"model_ids contains a missing value at position {position}")
        selected.append(model_id)
    if len(selected) != len(set(selected)):
        raise ValueError("model_ids contains duplicate canonical identifiers")

    edges = normalize_graph_edges(graph_edges)
    all_nodes = set(edges["parent_id"].tolist()) | set(edges["child_id"].tolist())
    if graph_nodes is not None:
        for position, value in enumerate(graph_nodes):
            model_id = canonical_model_id(value)
            if model_id is None:
                raise ValueError(f"graph_nodes contains a missing value at position {position}")
            all_nodes.add(model_id)
    disjoint_set = _DisjointSet(all_nodes)
    for edge in edges.itertuples(index=False):
        disjoint_set.union(edge.parent_id, edge.child_id)

    members_by_root: dict[str, list[str]] = {}
    for node in sorted(all_nodes):
        members_by_root.setdefault(disjoint_set.find(node), []).append(node)
    component_ids = {
        root: "cc_"
        + hashlib.sha256("\0".join(members).encode("utf-8")).hexdigest()[:16]
        for root, members in members_by_root.items()
    }
    component_sizes = {root: len(members) for root, members in members_by_root.items()}
    component_edge_counts = {root: 0 for root in members_by_root}
    for edge in edges.itertuples(index=False):
        component_edge_counts[disjoint_set.find(edge.parent_id)] += 1
    rows = []
    for model_id in sorted(selected):
        in_graph = model_id in all_nodes
        rows.append(
            {
                "model_id": model_id,
                "ancestry_component": (
                    component_ids[disjoint_set.find(model_id)] if in_graph else pd.NA
                ),
                "component_provenance": "graph" if in_graph else "",
                "component_size": (
                    component_sizes[disjoint_set.find(model_id)] if in_graph else 0
                ),
                "component_edge_count": (
                    component_edge_counts[disjoint_set.find(model_id)] if in_graph else 0
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "model_id",
            "ancestry_component",
            "component_provenance",
            "component_size",
            "component_edge_count",
        ],
    )


def build_single_source_components(
    model_ids: Iterable[object],
    graph_edges: pd.DataFrame,
    *,
    graph_nodes: Iterable[object] | None = None,
    graph_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Resolve clean single-root lineages and quarantine merge ancestry."""

    selected = []
    for position, value in enumerate(model_ids):
        model_id = canonical_model_id(value)
        if model_id is None:
            raise ValueError(f"model_ids contains a missing value at position {position}")
        selected.append(model_id)
    if len(selected) != len(set(selected)):
        raise ValueError("model_ids contains duplicate canonical identifiers")

    edges = normalize_graph_edges(graph_edges)
    all_nodes = set(edges["parent_id"]) | set(edges["child_id"])
    if graph_nodes is not None:
        for position, value in enumerate(graph_nodes):
            model_id = canonical_model_id(value)
            if model_id is None:
                raise ValueError(
                    f"graph_nodes contains a missing value at position {position}"
                )
            all_nodes.add(model_id)
    parents: dict[str, set[str]] = {node: set() for node in all_nodes}
    for edge in edges.itertuples(index=False):
        parents[edge.child_id].add(edge.parent_id)

    initial_reason = {
        node: "ambiguous_multiple_parents"
        for node, values in parents.items()
        if len(values) > 1
    }
    if graph_metadata is not None and not graph_metadata.empty:
        _require_graph_metadata = {"model_id", "base_model_relation"}
        missing = sorted(_require_graph_metadata - set(graph_metadata.columns))
        if missing:
            raise ValueError(f"graph_metadata is missing columns: {missing}")
        for row in graph_metadata.loc[
            :, ["model_id", "base_model_relation"]
        ].itertuples(index=False):
            model_id = canonical_model_id(row.model_id)
            relation = str(row.base_model_relation).strip().casefold()
            if model_id in all_nodes and relation == "merge":
                initial_reason.setdefault(model_id, "ambiguous_declared_merge")

    roots: dict[str, str | None] = {}
    reasons: dict[str, str] = {}
    for start in sorted(all_nodes):
        if start in roots:
            continue
        path: list[str] = []
        positions: dict[str, int] = {}
        current = start
        terminal_reason = ""
        while True:
            if current in roots:
                resolved = roots[current]
                terminal_reason = reasons.get(current, "")
                break
            if current in initial_reason:
                resolved = None
                terminal_reason = initial_reason[current]
                roots[current] = None
                reasons[current] = terminal_reason
                break
            if current in positions:
                resolved = None
                terminal_reason = "ancestry_cycle"
                break
            positions[current] = len(path)
            path.append(current)
            if not parents[current]:
                resolved = current
                break
            current = next(iter(parents[current]))
        for node in reversed(path):
            roots[node] = resolved
            if resolved is None:
                reasons[node] = (
                    "ancestry_cycle"
                    if terminal_reason == "ancestry_cycle"
                    else "ambiguous_ancestor"
                )

    members_by_root: dict[str, list[str]] = {}
    for node in sorted(all_nodes):
        root = roots[node]
        if root is not None:
            members_by_root.setdefault(root, []).append(node)
    component_ids = {
        root: "root_"
        + hashlib.sha256("\0".join(members).encode("utf-8")).hexdigest()[:16]
        for root, members in members_by_root.items()
    }
    edge_counts = {root: 0 for root in members_by_root}
    for edge in edges.itertuples(index=False):
        root = roots[edge.parent_id]
        if root is not None and root == roots[edge.child_id]:
            edge_counts[root] += 1

    rows = []
    for model_id in sorted(selected):
        in_graph = model_id in all_nodes
        root = roots.get(model_id)
        rows.append(
            {
                "model_id": model_id,
                "ancestry_component": (
                    component_ids[root] if root is not None else pd.NA
                ),
                "component_provenance": "graph" if root is not None else "",
                "component_size": len(members_by_root[root]) if root is not None else 0,
                "component_edge_count": edge_counts[root] if root is not None else 0,
                "component_root_model": root if root is not None else pd.NA,
                "component_grouping_method": "single_source_root_v1",
                "component_exclusion_reason": (
                    reasons.get(model_id, "")
                    if in_graph
                    else "model_absent_from_graph"
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def build_ancestor_closure_components(
    model_ids: Iterable[object],
    graph_edges: pd.DataFrame,
    *,
    graph_nodes: Iterable[object] | None = None,
    analysis_universe: Iterable[object] | None = None,
) -> pd.DataFrame:
    """Group analyzed models by shared directed ancestry, excluding downstream bridges."""

    selected: list[str] = []
    for position, value in enumerate(model_ids):
        model_id = canonical_model_id(value)
        if model_id is None:
            raise ValueError(f"model_ids contains a missing value at position {position}")
        selected.append(model_id)
    if len(selected) != len(set(selected)):
        raise ValueError("model_ids contains duplicate canonical identifiers")

    edges = normalize_graph_edges(graph_edges)
    all_nodes = set(edges["parent_id"]) | set(edges["child_id"])
    if graph_nodes is not None:
        for position, value in enumerate(graph_nodes):
            model_id = canonical_model_id(value)
            if model_id is None:
                raise ValueError(
                    f"graph_nodes contains a missing value at position {position}"
                )
            all_nodes.add(model_id)

    universe_values = selected if analysis_universe is None else analysis_universe
    universe: set[str] = set()
    for position, value in enumerate(universe_values):
        model_id = canonical_model_id(value)
        if model_id is None:
            raise ValueError(
                f"analysis_universe contains a missing value at position {position}"
            )
        universe.add(model_id)

    parents: dict[str, set[str]] = {node: set() for node in all_nodes}
    for edge in edges.itertuples(index=False):
        parents[edge.child_id].add(edge.parent_id)

    closure = {model_id for model_id in universe if model_id in all_nodes}
    frontier = list(closure)
    while frontier:
        child = frontier.pop()
        for parent in parents[child]:
            if parent not in closure:
                closure.add(parent)
                frontier.append(parent)

    disjoint_set = _DisjointSet(closure)
    closure_edges: list[tuple[str, str]] = []
    for edge in edges.itertuples(index=False):
        if edge.parent_id in closure and edge.child_id in closure:
            disjoint_set.union(edge.parent_id, edge.child_id)
            closure_edges.append((edge.parent_id, edge.child_id))

    members_by_root: dict[str, list[str]] = {}
    for node in sorted(closure):
        members_by_root.setdefault(disjoint_set.find(node), []).append(node)
    component_ids = {
        root: "ac_"
        + hashlib.sha256("\0".join(members).encode("utf-8")).hexdigest()[:16]
        for root, members in members_by_root.items()
    }
    component_edge_counts = {root: 0 for root in members_by_root}
    for parent, _ in closure_edges:
        component_edge_counts[disjoint_set.find(parent)] += 1

    rows = []
    for model_id in sorted(selected):
        in_graph = model_id in all_nodes
        in_closure = model_id in closure
        root = disjoint_set.find(model_id) if in_closure else None
        rows.append(
            {
                "model_id": model_id,
                "ancestry_component": (
                    component_ids[root] if in_closure and root is not None else pd.NA
                ),
                "component_provenance": (
                    "graph" if in_closure and root is not None else ""
                ),
                "component_size": (
                    len(members_by_root[root]) if root is not None else 0
                ),
                "component_edge_count": (
                    component_edge_counts[root] if root is not None else 0
                ),
                "component_grouping_method": "analysis_ancestor_closure_v1",
                "component_exclusion_reason": (
                    ""
                    if in_closure
                    else (
                        "outside_analysis_ancestor_closure"
                        if in_graph
                        else "model_absent_from_graph"
                    )
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


_EXCLUSION_COLUMNS = ["source_table", "source_row", "model_id", "reason"]


def _prepare_id_table(
    frame: pd.DataFrame, source_table: str
) -> tuple[pd.DataFrame, dict[str, int], list[dict[str, object]]]:
    if "model_id" not in frame.columns:
        raise ValueError(f"{source_table} requires a model_id column")

    valid_rows: list[pd.Series] = []
    source_rows: dict[str, int] = {}
    exclusions: list[dict[str, object]] = []
    for position, (_, row) in enumerate(frame.iterrows()):
        model_id = canonical_model_id(row["model_id"])
        if model_id is None:
            exclusions.append(
                {
                    "source_table": source_table,
                    "source_row": position,
                    "model_id": None,
                    "reason": "missing_model_id",
                }
            )
            continue
        copied = row.copy()
        copied["model_id"] = model_id
        valid_rows.append(copied)
        source_rows[model_id] = position

    valid = pd.DataFrame(valid_rows, columns=frame.columns).reset_index(drop=True)
    duplicate_mask = valid["model_id"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_ids = sorted(valid.loc[duplicate_mask, "model_id"].unique())
        raise ValueError(
            f"duplicate canonical model_id in {source_table}: {duplicate_ids}"
        )
    return valid, source_rows, exclusions


def derive_analysis_metadata(
    metadata: pd.DataFrame,
    graph_metadata: pd.DataFrame,
    population_tables: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build an uncapped metadata table from models present in study sources."""

    clean_metadata, metadata_source_rows, exclusions = _prepare_id_table(
        metadata, "metadata"
    )
    clean_graph, _, graph_exclusions = _prepare_id_table(
        graph_metadata, "graph_metadata"
    )
    if graph_exclusions:
        raise ValueError("graph metadata contains a missing model identifier")

    sources_by_model: dict[str, set[str]] = {}
    for source_name, frame in sorted(population_tables.items()):
        if "model_id" not in frame.columns:
            raise ValueError(f"{source_name} requires a model_id column")
        for value in frame["model_id"]:
            model_id = canonical_model_id(value)
            if model_id is not None:
                sources_by_model.setdefault(model_id, set()).add(str(source_name))

    required_ids = set(sources_by_model)
    for row in clean_metadata.loc[
        ~clean_metadata["model_id"].isin(required_ids)
    ].itertuples(index=False):
        exclusions.append(
            {
                "source_table": "metadata",
                "source_row": metadata_source_rows[str(row.model_id)],
                "model_id": str(row.model_id),
                "reason": "metadata_without_analysis_source",
            }
        )

    graph_by_id = clean_graph.set_index("model_id", verify_integrity=True)
    explicit_by_id = clean_metadata.set_index("model_id", verify_integrity=True)
    output_columns = list(
        dict.fromkeys(
            [
                "model_id",
                *[column for column in clean_graph.columns if column != "model_id"],
                *[column for column in clean_metadata.columns if column != "model_id"],
                "metadata_origin",
                "population_sources",
            ]
        )
    )

    def meaningful(value: object) -> bool:
        if value is None or bool(pd.isna(value)):
            return False
        return str(value).strip().casefold() not in {"", "unknown", "n/a", "nan", "none"}

    rows: list[dict[str, object]] = []
    for model_id in sorted(required_ids):
        row: dict[str, object] = {"model_id": model_id}
        has_graph = model_id in graph_by_id.index
        has_explicit = model_id in explicit_by_id.index
        if has_graph:
            for column, value in graph_by_id.loc[model_id].items():
                row[str(column)] = value
        if has_explicit:
            for column, value in explicit_by_id.loc[model_id].items():
                if meaningful(value):
                    row[str(column)] = value
        row["source_model"] = (
            row.get("source_model") if meaningful(row.get("source_model")) else "unknown"
        )
        row["base_model_relation"] = (
            row.get("base_model_relation")
            if meaningful(row.get("base_model_relation"))
            else "unknown"
        )
        row["metadata_origin"] = (
            "explicit+graph"
            if has_explicit and has_graph
            else "explicit"
            if has_explicit
            else "graph"
            if has_graph
            else "fallback"
        )
        row["population_sources"] = "|".join(sorted(sources_by_model[model_id]))
        rows.append(row)

    derived = pd.DataFrame.from_records(rows, columns=output_columns)
    exclusion_frame = pd.DataFrame(exclusions, columns=_EXCLUSION_COLUMNS)
    return derived, exclusion_frame


def build_model_manifest(
    metadata: pd.DataFrame,
    labels: pd.DataFrame,
    graph_edges: pd.DataFrame,
    *,
    graph_nodes: Iterable[object] | None = None,
    grouping_universe: Iterable[object] | None = None,
    graph_metadata: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join metadata and labels while returning every rejected row explicitly."""

    clean_metadata, _, exclusions = _prepare_id_table(metadata, "metadata")
    clean_labels, label_source_rows, label_exclusions = _prepare_id_table(
        labels, "labels"
    )
    exclusions.extend(label_exclusions)

    metadata_ids = set(clean_metadata["model_id"])
    orphan_mask = ~clean_labels["model_id"].isin(metadata_ids)
    for row in clean_labels.loc[orphan_mask].itertuples(index=False):
        exclusions.append(
            {
                "source_table": "labels",
                "source_row": label_source_rows[row.model_id],
                "model_id": row.model_id,
                "reason": "label_without_metadata",
            }
        )
    matched_labels = clean_labels.loc[~orphan_mask].copy()

    manifest = clean_metadata.merge(
        matched_labels,
        on="model_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_label"),
        indicator="_label_join",
    )
    manifest["label_disposition"] = manifest["_label_join"].map(
        {"both": "matched", "left_only": "unlabeled"}
    )
    manifest = manifest.drop(columns="_label_join")
    components = build_single_source_components(
        manifest["model_id"],
        graph_edges,
        graph_nodes=graph_nodes,
        graph_metadata=graph_metadata,
    )
    closure_components = build_ancestor_closure_components(
        manifest["model_id"],
        graph_edges,
        graph_nodes=graph_nodes,
        analysis_universe=grouping_universe,
    ).rename(
        columns={
            "ancestry_component": "ancestor_closure_component",
            "component_provenance": "ancestor_closure_provenance",
            "component_size": "ancestor_closure_size",
            "component_edge_count": "ancestor_closure_edge_count",
            "component_grouping_method": "ancestor_closure_method",
            "component_exclusion_reason": "ancestor_closure_exclusion_reason",
        }
    )
    components = components.merge(
        closure_components,
        on="model_id",
        how="left",
        validate="one_to_one",
    )
    manifest = manifest.merge(
        components,
        on="model_id",
        how="left",
        validate="one_to_one",
    ).sort_values("model_id", kind="stable", ignore_index=True)

    exclusion_frame = pd.DataFrame(exclusions, columns=_EXCLUSION_COLUMNS)
    return manifest, exclusion_frame


def build_provenance(
    source_paths: Mapping[str, str | Path],
    *,
    row_counts: Mapping[str, int],
    settings: RunSettings,
    manifest_rows: int,
    exclusions: pd.DataFrame,
    source_digests: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, JSON-serializable provenance record."""

    if source_digests is not None and set(source_digests) != set(source_paths):
        raise ValueError("source_digests must match source_paths exactly")
    sources: dict[str, dict[str, Any]] = {}
    for name in sorted(source_paths):
        path = Path(source_paths[name]).expanduser().resolve()
        digest = (
            str(source_digests[name])
            if source_digests is not None
            else path_digest(path)
        )
        record: dict[str, Any] = {"path": str(path), "sha256": digest}
        if name in row_counts:
            record["rows"] = int(row_counts[name])
        sources[name] = record

    reason_counts = {
        str(reason): int(count)
        for reason, count in sorted(exclusions["reason"].value_counts().items())
    }
    return {
        "schema_version": settings.schema_version,
        "settings": settings.serialized(),
        "sources": sources,
        "manifest": {
            "included_rows": int(manifest_rows),
            "excluded_rows": int(len(exclusions)),
            "exclusions_by_reason": reason_counts,
        },
    }


def write_provenance(path: str | Path, provenance: Mapping[str, Any]) -> Path:
    """Serialize provenance as stable, human-readable JSON."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
