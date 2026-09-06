from __future__ import annotations

"""Deterministic reconciliation and graph helpers for the scaled HTSR study."""

from collections import defaultdict, deque
import hashlib
from os import PathLike
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .contracts import (
    build_single_source_components,
    canonical_model_id,
    normalize_graph_edges,
)


STRONG_RELATIONS = frozenset(
    {"adapter", "finetune", "quantized", "finetune_quantized"}
)
REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")


def clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        return ""
    return str(value).strip()


def canonical(value: object) -> str:
    return canonical_model_id(value) or ""


def sha256_file(path: str | PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_lookup(frame: pd.DataFrame, column: str, *, casefold: bool) -> dict[str, str]:
    values: dict[str, set[str]] = defaultdict(set)
    if column not in frame.columns:
        return {}
    for model_id, raw in frame.loc[:, ["model_id", column]].itertuples(index=False):
        value = clean_text(raw)
        if not value:
            continue
        key = value.casefold() if casefold else value
        values[key].add(str(model_id))
    return {key: next(iter(model_ids)) for key, model_ids in values.items() if len(model_ids) == 1}


def map_overlap_to_phase2(
    phase2: pd.DataFrame,
    overlap: pd.DataFrame,
) -> pd.DataFrame:
    """Map score rows to exactly one Phase 2 model without fuzzy matching."""

    phase = phase2.copy()
    phase["model_id"] = phase["model_id"].map(canonical)
    exact_model = _unique_lookup(phase, "model_id", casefold=False)
    exact_key = _unique_lookup(phase, "canonical_model_key", casefold=False)
    folded_model = _unique_lookup(phase, "model_id", casefold=True)
    folded_key = _unique_lookup(phase, "canonical_model_key", casefold=True)

    sha_values: dict[str, set[str]] = defaultdict(set)
    for sha_column in ("model_sha", "Model sha"):
        if sha_column not in phase.columns:
            continue
        for model_id, raw in phase.loc[:, ["model_id", sha_column]].itertuples(index=False):
            value = clean_text(raw).casefold()
            if value:
                sha_values[value].add(str(model_id))
    sha_lookup = {
        key: next(iter(model_ids))
        for key, model_ids in sha_values.items()
        if len(model_ids) == 1
    }

    rows: list[dict[str, object]] = []
    for source_row, row in overlap.reset_index(drop=True).iterrows():
        candidates = (
            ("exact_model_id", exact_model, clean_text(row.get("model_id"))),
            ("exact_canonical_key", exact_key, clean_text(row.get("canonical_model_key"))),
            ("casefold_model_id", folded_model, clean_text(row.get("model_id")).casefold()),
            (
                "casefold_canonical_key",
                folded_key,
                clean_text(row.get("canonical_model_key")).casefold(),
            ),
            (
                "unique_checkpoint_sha",
                sha_lookup,
                (
                    clean_text(row.get("model_sha"))
                    or clean_text(row.get("Model sha"))
                ).casefold(),
            ),
        )
        mapped = ""
        method = "unmapped"
        for candidate_method, lookup, key in candidates:
            if key and key in lookup:
                mapped = lookup[key]
                method = candidate_method
                break
        record = row.to_dict()
        record.update(
            {
                "source_row": int(source_row),
                "mapped_model_id": mapped,
                "mapping_method": method,
            }
        )
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def _finite_tasks(frame: pd.DataFrame, tasks: Sequence[str]) -> pd.DataFrame:
    values = frame.loc[:, tasks].apply(pd.to_numeric, errors="coerce")
    valid = values.notna().all(axis=1) & values.ge(0.0).all(axis=1) & values.le(100.0).all(axis=1)
    result = frame.loc[valid].copy()
    result.loc[:, tasks] = values.loc[valid]
    return result


def _vectors_equal(frame: pd.DataFrame, tasks: Sequence[str], *, atol: float = 1e-12) -> bool:
    values = frame.loc[:, tasks].to_numpy(dtype=float)
    return len(values) > 0 and np.allclose(
        values,
        values[[0]],
        rtol=0.0,
        atol=atol,
        equal_nan=False,
    )


def _row_sha(row: pd.Series) -> str:
    return clean_text(row.get("model_sha")) or clean_text(row.get("Model sha"))


def _selected_target_row(
    row: pd.Series,
    *,
    model_id: str,
    tasks: Sequence[str],
    generation: str,
    resolution: str,
) -> dict[str, object]:
    result = {
        "model_id": model_id,
        **{task: float(row[task]) for task in tasks},
        "target_model_sha": _row_sha(row),
        "target_eval_name": clean_text(row.get("eval_name")),
        "target_architecture": clean_text(row.get("Architecture"))
        or clean_text(row.get("architecture")),
        "target_source": "ablation_overlap_audit",
        "target_source_row": int(row["source_row"]),
        "target_resolution": resolution,
        "source_population": "phase2",
        "benchmark_generation": generation,
    }
    return result


def reconcile_phase2_modern_targets(
    phase2: pd.DataFrame,
    mapped_overlap: pd.DataFrame,
    retained_targets: pd.DataFrame,
    *,
    tasks: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recover current-leaderboard task rows with explicit conflict handling."""

    complete = _finite_tasks(mapped_overlap, tasks)
    complete = complete.loc[complete["mapped_model_id"].astype(str).ne("")].copy()
    retained = retained_targets.copy().set_index("model_id", verify_integrity=True)
    phase = phase2.copy().set_index("model_id", verify_integrity=True)
    accepted: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []

    for model_id, group in complete.groupby("mapped_model_id", sort=True):
        group = group.sort_values("source_row", kind="stable")
        selected: pd.Series | None = None
        resolution = ""
        if _vectors_equal(group, tasks):
            selected = group.iloc[0]
            resolution = "single_or_identical_score_vector"
        elif model_id in retained.index:
            expected = retained.loc[model_id]
            score_match = np.isclose(
                group.loc[:, tasks].to_numpy(float),
                expected.loc[list(tasks)].to_numpy(float)[None, :],
                rtol=0.0,
                atol=1e-12,
            ).all(axis=1)
            sha_match = group.apply(_row_sha, axis=1).eq(
                clean_text(expected.get("target_model_sha"))
            )
            eval_match = group.get("eval_name", pd.Series("", index=group.index)).map(
                clean_text
            ).eq(clean_text(expected.get("target_eval_name")))
            matched = group.loc[score_match & sha_match & eval_match]
            if len(matched) == 1:
                selected = matched.iloc[0]
                resolution = "retained_target_identity"
        if selected is None and model_id in phase.index:
            phase_row = phase.loc[model_id]
            average = pd.to_numeric(
                pd.Series(
                    [phase_row.get("Average ⬆️"), phase_row.get("Average ⬆️_lb")]
                ),
                errors="coerce",
            ).dropna()
            candidate_average = pd.to_numeric(group.get("Average ⬆️"), errors="coerce")
            matches = np.zeros(len(group), dtype=bool)
            for value in average:
                matches |= np.isclose(
                    candidate_average.to_numpy(float),
                    float(value),
                    rtol=0.0,
                    atol=1e-10,
                )
            matched = group.loc[matches]
            if len(matched) == 1:
                selected = matched.iloc[0]
                resolution = "unique_reported_average"

        status = "accepted" if selected is not None else "rejected"
        reason = resolution if selected is not None else "conflicting_score_rows"
        audit.append(
            {
                "model_id": model_id,
                "benchmark_generation": "modern",
                "status": status,
                "reason": reason,
                "candidate_rows": int(len(group)),
                "mapping_methods": "|".join(sorted(set(group["mapping_method"].astype(str)))),
                "selected_source_row": (
                    int(selected["source_row"]) if selected is not None else pd.NA
                ),
            }
        )
        if selected is not None:
            accepted.append(
                _selected_target_row(
                    selected,
                    model_id=model_id,
                    tasks=tasks,
                    generation="modern",
                    resolution=resolution,
                )
            )

    recovered = pd.DataFrame.from_records(accepted)
    missing_retained = retained.loc[~retained.index.isin(set(recovered.get("model_id", [])))]
    for model_id, row in missing_retained.iterrows():
        recovered = pd.concat(
            [
                recovered,
                pd.DataFrame(
                    [
                        {
                            "model_id": model_id,
                            **{task: float(row[task]) for task in tasks},
                            "target_model_sha": clean_text(row.get("target_model_sha")),
                            "target_eval_name": clean_text(row.get("target_eval_name")),
                            "target_architecture": clean_text(row.get("target_architecture")),
                            "target_source": "retained_development_targets",
                            "target_source_row": pd.NA,
                            "target_resolution": "retained_target_only",
                            "source_population": "phase2",
                            "benchmark_generation": "modern",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    if recovered["model_id"].duplicated().any():
        raise ValueError("modern reconciliation produced duplicate Phase 2 model IDs")
    return recovered.sort_values("model_id", kind="stable", ignore_index=True), pd.DataFrame(audit)


def reconcile_phase2_legacy_targets(
    mapped_overlap: pd.DataFrame,
    *,
    tasks: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recover old-leaderboard rows only when duplicate evaluations agree."""

    complete = _finite_tasks(mapped_overlap, tasks)
    complete = complete.loc[complete["mapped_model_id"].astype(str).ne("")].copy()
    accepted: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for model_id, group in complete.groupby("mapped_model_id", sort=True):
        group = group.sort_values("source_row", kind="stable")
        agrees = _vectors_equal(group, tasks)
        audit.append(
            {
                "model_id": model_id,
                "benchmark_generation": "legacy",
                "status": "accepted" if agrees else "rejected",
                "reason": "identical_score_vector" if agrees else "conflicting_score_rows",
                "candidate_rows": int(len(group)),
                "mapping_methods": "|".join(sorted(set(group["mapping_method"].astype(str)))),
                "selected_source_row": int(group.iloc[0]["source_row"]) if agrees else pd.NA,
            }
        )
        if agrees:
            accepted.append(
                _selected_target_row(
                    group.iloc[0],
                    model_id=model_id,
                    tasks=tasks,
                    generation="legacy",
                    resolution="single_or_identical_score_vector",
                )
            )
    columns = [
        "model_id",
        *tasks,
        "target_model_sha",
        "target_eval_name",
        "target_architecture",
        "target_source",
        "target_source_row",
        "target_resolution",
        "source_population",
        "benchmark_generation",
    ]
    targets = pd.DataFrame.from_records(accepted, columns=columns)
    return targets.sort_values("model_id", kind="stable", ignore_index=True), pd.DataFrame(audit)


def metadata_edges(
    frames: Iterable[tuple[str, pd.DataFrame]],
    base_edges: pd.DataFrame,
    graph_nodes: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit strong local parent declarations and return represented additions."""

    source_frames = list(frames)
    base = normalize_graph_edges(base_edges)
    base_pairs = set(base.itertuples(index=False, name=None))
    atlas_nodes = set(graph_nodes)
    population_ids = {
        source_name: set(
            frame["model_id"].map(canonical).loc[lambda value: value.ne("")]
        )
        for source_name, frame in source_frames
    }
    all_population_ids = set().union(*population_ids.values())

    rows: list[dict[str, object]] = []
    accepted: list[dict[str, str]] = []
    for source_name, frame in source_frames:
        represented = atlas_nodes | population_ids[source_name]
        if source_name != "phase2":
            represented |= all_population_ids
        for source_row, row in frame.reset_index(drop=True).iterrows():
            child = canonical(row.get("model_id"))
            parent_raw = clean_text(row.get("source_model"))
            displayed_merge = bool(
                re.search(r"\s+\(merge\)\s*$", parent_raw, re.IGNORECASE)
            )
            parent_clean = re.sub(
                r"\s+\(merge\)\s*$", "", parent_raw, flags=re.IGNORECASE
            )
            parent = canonical(parent_clean)
            relation = clean_text(row.get("base_model_relation")).casefold()
            reason = "accepted"
            if child and parent and parent.casefold() == child.casefold():
                reason = "displayed_self_merge" if displayed_merge else "self_reference"
            elif relation not in STRONG_RELATIONS:
                reason = "relation_not_strong"
            elif not REPO_ID.fullmatch(parent_clean):
                reason = "invalid_parent_id"
            elif not child:
                reason = "invalid_child_id"
            elif parent not in represented:
                reason = "parent_not_represented"
            elif child in atlas_nodes and (parent, child) not in base_pairs:
                reason = "atlas_parent_conflict"
            if reason == "accepted":
                accepted.append({"parent_id": parent, "child_id": child})
            pair = (parent, child)
            rows.append(
                {
                    "source": source_name,
                    "source_row": int(source_row),
                    "model_id": child,
                    "source_model": parent,
                    "relation": relation,
                    "status": "accepted" if reason == "accepted" else "rejected",
                    "reason": reason,
                    "already_in_atlas": pair in base_pairs if parent and child else False,
                }
            )
    accepted_frame = (
        pd.DataFrame.from_records(accepted, columns=["parent_id", "child_id"])
        .drop_duplicates(ignore_index=True)
        .sort_values(["parent_id", "child_id"], kind="stable", ignore_index=True)
    )
    augmented = normalize_graph_edges(pd.concat([base, accepted_frame], ignore_index=True))
    return augmented, accepted_frame, pd.DataFrame.from_records(rows)


def corroborated_edges(
    base_edges: pd.DataFrame,
    graph_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Keep Atlas edges also named and strongly typed in child metadata."""

    pairs = set(normalize_graph_edges(base_edges).itertuples(index=False, name=None))
    rows: list[dict[str, str]] = []
    for row in graph_metadata.itertuples(index=False):
        child = canonical(row.model_id)
        parent = canonical(row.source_model)
        relation = clean_text(row.base_model_relation).casefold()
        if (
            relation in STRONG_RELATIONS
            and parent
            and child
            and parent != child
            and (parent, child) in pairs
        ):
            rows.append({"parent_id": parent, "child_id": child})
    return normalize_graph_edges(pd.DataFrame.from_records(rows))


def propagated_component_labels(
    model_ids: Sequence[str],
    edges: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    graph_nodes: Iterable[str] | None = None,
    graph_metadata: pd.DataFrame | None = None,
    family_column: str = "audited_base_family",
) -> pd.DataFrame:
    """Name clean graph components only when all audit anchors agree."""

    components = build_single_source_components(
        model_ids,
        edges,
        graph_nodes=graph_nodes,
        graph_metadata=graph_metadata,
    )
    anchors = labels.loc[
        labels[family_column].notna(), ["model_id", family_column]
    ].copy()
    anchors["model_id"] = anchors["model_id"].map(canonical)
    named = components.merge(anchors, on="model_id", how="left", validate="one_to_one")
    grouped = (
        named.dropna(subset=["ancestry_component", family_column])
        .groupby("ancestry_component")[family_column]
        .agg(lambda values: tuple(sorted(set(map(str, values)))))
    )
    component_family = {
        component: values[0] for component, values in grouped.items() if len(values) == 1
    }
    named["metadata_family"] = named["ancestry_component"].map(component_family)
    named["metadata_family_conflict"] = named["ancestry_component"].map(
        {component: len(values) > 1 for component, values in grouped.items()}
    ).eq(True)
    return named.drop(columns=[family_column])


def consensus_roots(
    graph_edges: pd.DataFrame,
    graph_nodes: Iterable[str],
) -> pd.DataFrame:
    """Resolve a root when every directed parent path ends at the same root."""

    edges = normalize_graph_edges(graph_edges)
    nodes = set(graph_nodes) | set(edges["parent_id"]) | set(edges["child_id"])
    parents: dict[str, set[str]] = {node: set() for node in nodes}
    children: dict[str, set[str]] = {node: set() for node in nodes}
    for parent, child in edges.itertuples(index=False, name=None):
        parents[child].add(parent)
        children[parent].add(child)
    remaining = {node: len(parents[node]) for node in nodes}
    queue = deque(sorted(node for node, count in remaining.items() if count == 0))
    root: dict[str, str | None] = {}
    while queue:
        node = queue.popleft()
        if not parents[node]:
            root[node] = node
        else:
            values = {root.get(parent) for parent in parents[node]}
            root[node] = next(iter(values)) if len(values) == 1 and None not in values else None
        for child in sorted(children[node]):
            remaining[child] -= 1
            if remaining[child] == 0:
                queue.append(child)
    for node in nodes:
        root.setdefault(node, None)
    return pd.DataFrame(
        {
            "model_id": sorted(nodes),
            "consensus_root_model": [root[node] for node in sorted(nodes)],
            "parent_count": [len(parents[node]) for node in sorted(nodes)],
            "consensus_status": [
                "resolved" if root[node] is not None else "conflicting_or_cyclic"
                for node in sorted(nodes)
            ],
        }
    )


def name_consensus_families(
    roots: pd.DataFrame,
    labels: pd.DataFrame,
    model_ids: Sequence[str],
) -> pd.DataFrame:
    lookup = roots.set_index("model_id")["consensus_root_model"]
    anchors = labels.loc[labels["audited_base_family"].notna()].copy()
    anchors["model_id"] = anchors["model_id"].map(canonical)
    anchors["consensus_root_model"] = anchors["model_id"].map(lookup)
    names = (
        anchors.dropna(subset=["consensus_root_model"])
        .groupby("consensus_root_model")["audited_base_family"]
        .agg(lambda values: tuple(sorted(set(map(str, values)))))
    )
    root_family = {root: values[0] for root, values in names.items() if len(values) == 1}
    selected = pd.DataFrame({"model_id": list(model_ids)})
    selected["consensus_root_model"] = selected["model_id"].map(lookup)
    selected["metadata_family"] = selected["consensus_root_model"].map(root_family)
    selected["metadata_family_conflict"] = selected["consensus_root_model"].map(
        {root: len(values) > 1 for root, values in names.items()}
    ).eq(True)
    return selected


def first_root_branch(
    components: pd.DataFrame,
    edges: pd.DataFrame,
) -> pd.DataFrame:
    """Find the first child below each clean single-source root."""

    parent = {
        child: parent
        for parent, child in normalize_graph_edges(edges).itertuples(index=False, name=None)
    }

    def resolve(model_id: str, root: object) -> tuple[object, int | None]:
        if pd.isna(root):
            return pd.NA, None
        if model_id == root:
            return pd.NA, 0
        current = model_id
        depth = 0
        seen: set[str] = set()
        while current in parent and parent[current] != root:
            if current in seen:
                return pd.NA, None
            seen.add(current)
            current = parent[current]
            depth += 1
        if parent.get(current) == root:
            return current, depth + 1
        return pd.NA, None

    result = components.copy()
    resolved = [
        resolve(str(model_id), root)
        for model_id, root in zip(
            result["model_id"], result["component_root_model"], strict=False
        )
    ]
    result["root_branch_model"] = [value[0] for value in resolved]
    result["path_depth"] = [value[1] for value in resolved]
    return result


def balanced_branch_folds(
    population: pd.DataFrame,
    *,
    n_splits: int,
) -> pd.Series:
    """Assign each root branch once while balancing total held-out rows."""

    loads = np.zeros(n_splits, dtype=int)
    assignments: dict[str, int] = {}
    for _, root_frame in population.groupby("component_root_model", sort=True):
        sizes = root_frame.groupby("root_branch_model").size().sort_values(
            ascending=False, kind="stable"
        )
        used: set[int] = set()
        for branch, size in sizes.items():
            choices = [fold for fold in range(n_splits) if fold not in used]
            if not choices:
                choices = list(range(n_splits))
            fold = min(choices, key=lambda value: (loads[value], value))
            assignments[str(branch)] = fold
            loads[fold] += int(size)
            used.add(fold)
    return population["root_branch_model"].astype(str).map(assignments).astype(int)


def matched_family_recall(labels: Sequence[object], clusters: Sequence[object]) -> float:
    frame = pd.DataFrame({"family": labels, "cluster": clusters}).dropna()
    families = sorted(frame["family"].astype(str).unique())
    groups = sorted(frame["cluster"].unique())
    if not families or not groups:
        return float("nan")
    table = np.zeros((len(families), len(groups)), dtype=float)
    family_index = {name: index for index, name in enumerate(families)}
    group_index = {name: index for index, name in enumerate(groups)}
    for family, group in frame.itertuples(index=False, name=None):
        table[family_index[str(family)], group_index[group]] += 1
    rows, columns = linear_sum_assignment(-table)
    correct = np.zeros(len(families), dtype=float)
    correct[rows] = table[rows, columns]
    totals = table.sum(axis=1)
    return float(np.divide(correct, totals, out=np.zeros_like(correct), where=totals > 0).mean())
