from __future__ import annotations

import numpy as np
import pandas as pd

ROLE_MAP = {
    "q_proj": "attn_q",
    "k_proj": "attn_k",
    "v_proj": "attn_v",
    "o_proj": "attn_o",
    "down_proj": "mlp_down",
    "gate_proj": "mlp_gate",
    "up_proj": "mlp_up",
}

_INTRINSIC_COLUMNS = (
    "Architecture",
    "#Params (B)",
    "num_layers",
    "schema_name",
    "raw_module_count",
    "common_schema",
)


def canonical_role(module_name: str) -> str | None:
    for suffix, role in ROLE_MAP.items():
        if module_name.endswith(suffix):
            return role
    return None


def _validate_alpha_modules(alpha: np.ndarray, modules: tuple[str, ...]) -> np.ndarray:
    arr = np.asarray(alpha, dtype=float)
    if arr.ndim != 2:
        raise ValueError("alpha must be a 2D array")
    if arr.shape[1] != len(modules):
        raise ValueError("alpha width must match the number of modules")
    return arr


def intrinsic_basics_feature_map(manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    frame = manifest.copy()
    if "model_id" not in frame.columns:
        return {}

    intrinsic_columns = [column for column in _INTRINSIC_COLUMNS if column in frame.columns]
    if not intrinsic_columns:
        raise ValueError("no intrinsic feature columns are available")
    intrinsic = frame.loc[:, intrinsic_columns].copy()

    numeric_columns = ["#Params (B)", "num_layers", "raw_module_count"]
    numeric = pd.DataFrame(index=frame.index)
    for column in numeric_columns:
        if column in intrinsic.columns:
            numeric[column] = pd.to_numeric(intrinsic[column], errors="coerce").fillna(-1.0)

    categorical_columns = [
        column for column in intrinsic.columns if column not in numeric.columns
    ]
    if categorical_columns:
        categorical = intrinsic.loc[:, categorical_columns].fillna("unknown").astype(str)
        categorical = pd.get_dummies(categorical, prefix=categorical_columns, dtype=float)
    else:
        categorical = pd.DataFrame(index=frame.index)

    features = pd.concat([numeric, categorical], axis=1)
    if features.empty:
        features = pd.DataFrame(index=frame.index)

    vectors: dict[str, np.ndarray] = {}
    for row in pd.concat([frame[["model_id"]], features], axis=1).itertuples(index=False):
        vectors[str(row.model_id)] = np.asarray(row[1:], dtype=float)
    return vectors


def _role_traces(alpha: np.ndarray, modules: tuple[str, ...]) -> dict[str, np.ndarray]:
    arr = _validate_alpha_modules(alpha, modules)
    grouped: dict[str, list[np.ndarray]] = {}
    for index, module in enumerate(modules):
        role = canonical_role(module)
        if role is None:
            continue
        grouped.setdefault(role, []).append(arr[:, index])

    return {role: np.nanmean(np.vstack(columns), axis=0) for role, columns in grouped.items()}


def _group_traces(role_traces: dict[str, np.ndarray], roles: tuple[str, ...]) -> np.ndarray | None:
    columns = [role_traces[role] for role in roles if role in role_traces]
    if not columns:
        return None
    return np.nanmean(np.vstack(columns), axis=0)


def _negative_l2(left: np.ndarray, right: np.ndarray) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    mask = np.isfinite(left_array) & np.isfinite(right_array)
    if not mask.any():
        return float("nan")
    delta = left_array[mask] - right_array[mask]
    return float(-np.mean(delta * delta))


def shared_role_trace_score(
    *,
    query_alpha: np.ndarray,
    query_modules: tuple[str, ...],
    proto_alpha: np.ndarray,
    proto_modules: tuple[str, ...],
    overlap_penalty: float,
) -> float:
    query_roles = _role_traces(query_alpha, query_modules)
    proto_roles = _role_traces(proto_alpha, proto_modules)
    shared_roles = sorted(set(query_roles) & set(proto_roles))
    if not shared_roles:
        return float("nan")

    scores = [_negative_l2(query_roles[role], proto_roles[role]) for role in shared_roles]
    overlap = len(shared_roles) / max(len(query_roles), len(proto_roles))
    return float(np.nanmean(scores) - overlap_penalty * (1.0 - overlap))


def role_aggregated_trace_score(
    *,
    query_alpha: np.ndarray,
    query_modules: tuple[str, ...],
    proto_alpha: np.ndarray,
    proto_modules: tuple[str, ...],
    overlap_penalty: float,
) -> float:
    groups = {
        "attention": ("attn_q", "attn_k", "attn_v", "attn_o"),
        "mlp": ("mlp_down", "mlp_gate", "mlp_up"),
    }
    query_roles = _role_traces(query_alpha, query_modules)
    proto_roles = _role_traces(proto_alpha, proto_modules)

    query_grouped = {
        group: trace
        for group, roles in groups.items()
        if (trace := _group_traces(query_roles, roles)) is not None
    }
    proto_grouped = {
        group: trace
        for group, roles in groups.items()
        if (trace := _group_traces(proto_roles, roles)) is not None
    }

    shared_groups = sorted(set(query_grouped) & set(proto_grouped))
    if not shared_groups:
        return float("nan")

    scores = [_negative_l2(query_grouped[group], proto_grouped[group]) for group in shared_groups]
    overlap = len(shared_groups) / max(len(query_grouped), len(proto_grouped))
    return float(np.nanmean(scores) - overlap_penalty * (1.0 - overlap))


__all__ = [
    "ROLE_MAP",
    "canonical_role",
    "intrinsic_basics_feature_map",
    "role_aggregated_trace_score",
    "shared_role_trace_score",
]
