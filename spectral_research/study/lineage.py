from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    normalized_mutual_info_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from spectral_research.study.contracts import (
    canonical_model_id,
    normalize_graph_edges,
)
from spectral_research.study.features import (
    ALPHA_ESTIMATOR,
    ALPHA_ESTIMATOR_VERSION,
    CANONICAL_ROLES,
    clean_alpha,
)


CONFIG_METHOD = "config"
SPECTRAL_SHAPE_METHOD = "spectral_shape"
PAPER_STYLE_ALPHA_DKS = "paper_style_alpha_dks"
COMBINED_METHOD = "combined"
LINEAGE_METHODS = (
    CONFIG_METHOD,
    SPECTRAL_SHAPE_METHOD,
    PAPER_STYLE_ALPHA_DKS,
    COMBINED_METHOD,
)
PAPER_STYLE_ALPHA_DKS_PROVENANCE = (
    "paper_eq2_spearman_dp_depth_alignment_stored_dks_not_fixed_hill"
)
GALLERY_SCALED_PROVENANCE = "gallery_only_robust_scaling_shared_finite_features"

_FEATURE_PREFIXES = {
    CONFIG_METHOD: ("M__",),
    SPECTRAL_SHAPE_METHOD: ("S__",),
    COMBINED_METHOD: ("M__", "S__", "H__"),
}
_REQUIRED_BLOCK_PREFIXES = ("M__", "S__", "H__")
_AVAILABILITY_COLUMNS = (
    "feature_available",
    "features_available",
    "artifact_available",
    "analysis_artifact_present",
)
_EXCLUSION_COLUMNS = ("stage", "root_family", "model_id", "reason", "detail")
_ARCHITECTURE_COLUMNS = frozenset(
    {
        "architecture",
        "Architecture",
        "model_architecture",
        "architecture_class",
        "target_architecture",
    }
)


@dataclass(frozen=True)
class GalleryScaler:
    """Robust feature scaling fitted only on the supplied gallery rows."""

    columns: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    gallery_model_ids: tuple[str, ...]


@dataclass(frozen=True)
class SimilarityResult:
    score: float
    coverage: float
    abstained: bool


@dataclass(frozen=True)
class UnknownFamilyFold:
    held_out_family: str
    gallery: pd.DataFrame
    known_queries: pd.DataFrame
    unknown_queries: pd.DataFrame


@dataclass(frozen=True)
class _PreparedPopulation:
    manifest: pd.DataFrame
    features: pd.DataFrame
    trajectories: pd.DataFrame
    gallery: pd.DataFrame
    queries: pd.DataFrame
    common_model_ids: frozenset[str]
    method_columns: Mapping[str, tuple[str, ...]]
    exclusions: pd.DataFrame


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _canonical_id(value: object, *, field: str) -> str:
    model_id = canonical_model_id(value)
    if model_id is None:
        raise ValueError(f"{field} contains a missing identifier")
    return model_id


def _canonical_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        manifest,
        ("model_id", "audited_base_family", "label_status"),
        "manifest",
    )
    working = manifest.copy(deep=True)
    working["model_id"] = [
        _canonical_id(value, field="manifest.model_id")
        for value in working["model_id"]
    ]
    duplicate = working["model_id"].duplicated(keep=False)
    if duplicate.any():
        identifiers = sorted(working.loc[duplicate, "model_id"].unique())
        raise ValueError(f"manifest has duplicate canonical model_id values: {identifiers}")
    working["root_family"] = [
        canonical_model_id(value) for value in working["audited_base_family"]
    ]
    working["_confirmed"] = (
        working["label_status"].astype("string").str.strip().str.casefold().eq("confirmed")
        & working["root_family"].notna()
    )
    working["is_root"] = working["model_id"].eq(working["root_family"])
    return working


def _canonical_features(features: pd.DataFrame) -> pd.DataFrame:
    _require_columns(features, ("model_id",), "features")
    working = features.copy(deep=True)
    working["model_id"] = [
        _canonical_id(value, field="features.model_id")
        for value in working["model_id"]
    ]
    duplicate = working["model_id"].duplicated(keep=False)
    if duplicate.any():
        identifiers = sorted(working.loc[duplicate, "model_id"].unique())
        raise ValueError(f"features has duplicate canonical model_id values: {identifiers}")
    return working


def _canonical_trajectories(trajectories: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        trajectories,
        (
            "model_id",
            "role",
            "depth",
            "alpha",
            "alpha_estimator",
            "alpha_estimator_version",
        ),
        "alpha trajectories",
    )
    working = trajectories.copy(deep=True)
    working["model_id"] = [
        _canonical_id(value, field="alpha trajectories.model_id")
        for value in working["model_id"]
    ]
    working["role"] = working["role"].astype("string").str.strip()
    working["depth"] = pd.to_numeric(working["depth"], errors="coerce")
    working["alpha"] = working["alpha"].map(clean_alpha)
    _validate_alpha_provenance(working, name="alpha trajectories")
    return working


def _validate_alpha_provenance(frame: pd.DataFrame, *, name: str) -> None:
    _require_columns(
        frame,
        ("alpha_estimator", "alpha_estimator_version"),
        name,
    )
    expected = {
        "alpha_estimator": ALPHA_ESTIMATOR,
        "alpha_estimator_version": ALPHA_ESTIMATOR_VERSION,
    }
    for column, expected_value in expected.items():
        values = frame[column].astype("string").str.strip()
        unique = sorted(values.dropna().unique().tolist())
        if values.isna().any() or values.eq("").any() or unique != [expected_value]:
            raise ValueError(
                f"{name} has missing, mixed, or unexpected {column}; "
                f"expected {expected_value!r}, found {unique}"
            )


def _empty_exclusions() -> pd.DataFrame:
    return pd.DataFrame(columns=_EXCLUSION_COLUMNS)


def _exclusion(
    *,
    stage: str,
    root_family: str | None,
    model_id: str | None,
    reason: str,
    detail: str = "",
) -> dict[str, object]:
    return {
        "stage": stage,
        "root_family": root_family,
        "model_id": model_id,
        "reason": reason,
        "detail": detail,
    }


def _as_available(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().casefold() in {
            "1",
            "true",
            "yes",
            "available",
            "present",
        }
    return bool(value)


def _available_model_ids(
    manifest: pd.DataFrame,
    *,
    available_model_ids: Iterable[object] | None,
    availability_column: str | None,
) -> set[str]:
    if available_model_ids is not None:
        return {
            _canonical_id(value, field="available_model_ids")
            for value in available_model_ids
        }

    selected_column = availability_column
    if selected_column is None:
        selected_column = next(
            (column for column in _AVAILABILITY_COLUMNS if column in manifest.columns),
            None,
        )
    if selected_column is None:
        raise ValueError(
            "gallery construction requires an explicit availability contract"
        )
    if selected_column not in manifest.columns:
        raise ValueError(f"availability column is absent: {selected_column}")
    return set(
        manifest.loc[manifest[selected_column].map(_as_available), "model_id"]
    )


def build_root_gallery(
    manifest: pd.DataFrame,
    *,
    available_model_ids: Iterable[object] | None = None,
    availability_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one available, audited canonical root per confirmed family.

    A descendant is never promoted when the exact canonical root is absent. Both
    root IDs and artifact-style ``org--model`` IDs are normalized before the exact
    identity check.
    """

    working = _canonical_manifest(manifest)
    available = _available_model_ids(
        working,
        available_model_ids=available_model_ids,
        availability_column=availability_column,
    )
    confirmed = working.loc[working["_confirmed"]].copy()
    gallery_rows: list[pd.Series] = []
    exclusions: list[dict[str, object]] = []

    for family, family_rows in confirmed.groupby("root_family", sort=True):
        roots = family_rows.loc[family_rows["model_id"].eq(family)]
        if roots.empty:
            exclusions.append(
                _exclusion(
                    stage="gallery",
                    root_family=str(family),
                    model_id=None,
                    reason="missing_canonical_root",
                    detail="no confirmed row exactly matches audited_base_family",
                )
            )
            for model_id in sorted(family_rows["model_id"]):
                exclusions.append(
                    _exclusion(
                        stage="gallery",
                        root_family=str(family),
                        model_id=str(model_id),
                        reason="descendant_is_not_root",
                        detail="descendants cannot substitute for the canonical root",
                    )
                )
            continue
        if len(roots) != 1:
            raise ValueError(f"family {family!r} has more than one canonical root row")

        root = roots.iloc[0].copy()
        root_id = str(root["model_id"])
        if root_id not in available:
            exclusions.append(
                _exclusion(
                    stage="gallery",
                    root_family=str(family),
                    model_id=root_id,
                    reason="root_feature_or_artifact_unavailable",
                    detail="canonical root is absent from the explicit availability set",
                )
            )
            continue
        gallery_rows.append(root)

    gallery = pd.DataFrame(gallery_rows, columns=working.columns)
    if not gallery.empty:
        gallery = gallery.sort_values("root_family", kind="stable", ignore_index=True)
    gallery = gallery.drop(columns=["_confirmed"], errors="ignore")
    exclusion_frame = pd.DataFrame(exclusions, columns=_EXCLUSION_COLUMNS)
    return gallery, exclusion_frame


def _validate_coverage(value: float, *, name: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must be in (0, 1]")
    return numeric


def _finite_number(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if np.isfinite(numeric) else float("nan")


def fit_gallery_scaler(
    features: pd.DataFrame,
    *,
    gallery_model_ids: Iterable[object],
    columns: Sequence[str],
) -> GalleryScaler:
    """Fit robust center and scale without observing queries or labels."""

    working = _canonical_features(features)
    return _fit_gallery_scaler_from_canonical(
        working,
        gallery_model_ids=gallery_model_ids,
        columns=columns,
    )


def _fit_gallery_scaler_from_canonical(
    working: pd.DataFrame,
    *,
    gallery_model_ids: Iterable[object],
    columns: Sequence[str],
) -> GalleryScaler:
    requested_columns = tuple(dict.fromkeys(str(column) for column in columns))
    if not requested_columns:
        raise ValueError("at least one feature column is required")
    _require_columns(working, requested_columns, "features")

    gallery_ids = tuple(
        dict.fromkeys(
            _canonical_id(value, field="gallery_model_ids")
            for value in gallery_model_ids
        )
    )
    if not gallery_ids:
        raise ValueError("gallery_model_ids must not be empty")
    indexed = working.set_index("model_id", verify_integrity=True)
    missing = sorted(set(gallery_ids).difference(indexed.index))
    if missing:
        raise ValueError(f"gallery models are missing feature rows: {missing}")

    numeric = indexed.loc[list(gallery_ids), list(requested_columns)].apply(
        pd.to_numeric, errors="coerce"
    )
    kept_columns: list[str] = []
    centers: list[float] = []
    scales: list[float] = []
    for column in requested_columns:
        values = numeric[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        center = float(np.median(finite))
        q25, q75 = np.percentile(finite, [25.0, 75.0])
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale <= np.finfo(float).eps:
            scale = float(np.std(finite, ddof=0))
        if not np.isfinite(scale) or scale <= np.finfo(float).eps:
            scale = 1.0
        kept_columns.append(column)
        centers.append(center)
        scales.append(scale)
    if not kept_columns:
        raise ValueError("no feature column is finite in the gallery")
    return GalleryScaler(
        columns=tuple(kept_columns),
        center=tuple(centers),
        scale=tuple(scales),
        gallery_model_ids=gallery_ids,
    )


def shared_feature_similarity(
    query: Mapping[str, object] | pd.Series,
    root: Mapping[str, object] | pd.Series,
    scaler: GalleryScaler,
    *,
    min_coverage: float,
) -> SimilarityResult:
    """Return inverse standardized RMS distance over shared finite features."""

    required = _validate_coverage(min_coverage, name="min_coverage")
    query_values = np.asarray(
        [_finite_number(query.get(column, np.nan)) for column in scaler.columns],
        dtype=float,
    )
    root_values = np.asarray(
        [_finite_number(root.get(column, np.nan)) for column in scaler.columns],
        dtype=float,
    )
    shared = np.isfinite(query_values) & np.isfinite(root_values)
    coverage = float(np.mean(shared)) if shared.size else 0.0
    if not shared.any() or coverage < required:
        return SimilarityResult(float("nan"), coverage, True)
    scales = np.asarray(scaler.scale, dtype=float)[shared]
    standardized_delta = (query_values[shared] - root_values[shared]) / scales
    distance = float(np.sqrt(np.mean(np.square(standardized_delta))))
    return SimilarityResult(1.0 / (1.0 + distance), coverage, False)


def _role_trajectory(frame: pd.DataFrame, role: str) -> tuple[np.ndarray, np.ndarray] | None:
    selected = frame.loc[frame["role"].astype("string").eq(role)].copy()
    if selected.empty:
        return None
    depths = pd.to_numeric(selected["depth"], errors="coerce").to_numpy(dtype=float)
    alpha = np.asarray([clean_alpha(value) for value in selected["alpha"]], dtype=float)
    valid = np.isfinite(depths) & np.isfinite(alpha)
    if np.count_nonzero(valid) < 2:
        return None
    grouped = (
        pd.DataFrame({"depth": depths[valid], "alpha": alpha[valid]})
        .groupby("depth", sort=True, as_index=False)["alpha"]
        .median()
    )
    if len(grouped) < 2:
        return None
    depth_values = grouped["depth"].to_numpy(dtype=float)
    alpha_values = grouped["alpha"].to_numpy(dtype=float)
    if np.ptp(alpha_values) <= np.finfo(float).eps:
        return None
    return depth_values, alpha_values


def _signature_matrix(frame: pd.DataFrame, *, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    _require_columns(
        frame,
        (
            "role",
            "depth",
            "alpha",
            "alpha_estimator",
            "alpha_estimator_version",
        ),
        name,
    )
    _validate_alpha_provenance(frame, name=name)
    working = frame.loc[:, ["role", "depth", "alpha"]].copy()
    working["role"] = working["role"].astype("string").str.strip()
    working["depth"] = pd.to_numeric(working["depth"], errors="coerce")
    working["alpha"] = working["alpha"].map(clean_alpha)
    working = working.loc[
        working["role"].isin(CANONICAL_ROLES)
        & np.isfinite(working["depth"])
        & np.isfinite(working["alpha"])
    ]
    if working.empty:
        return None
    pivot = working.pivot_table(
        index="depth",
        columns="role",
        values="alpha",
        aggfunc="median",
        observed=False,
    ).reindex(columns=CANONICAL_ROLES)
    pivot = pivot.dropna(axis=0, how="any").sort_index()
    if len(pivot) < 2:
        return None
    matrix = pivot.to_numpy(dtype=float)
    if np.any(np.ptp(matrix, axis=0) <= np.finfo(float).eps):
        return None
    return pivot.index.to_numpy(dtype=float), matrix


def _order_preserving_layer_pairs(
    query: np.ndarray,
    root: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match every layer of the shorter signature injectively to the longer one."""

    if query.ndim != 2 or root.ndim != 2 or query.shape[1] != root.shape[1]:
        raise ValueError("signature matrices must be two-dimensional with equal width")
    query_is_shorter = len(query) <= len(root)
    short = query if query_is_shorter else root
    long = root if query_is_shorter else query
    combined = np.vstack([short, long])
    center = np.mean(combined, axis=0)
    scale = np.std(combined, axis=0)
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    short_z = (short - center) / scale
    long_z = (long - center) / scale
    cost = np.mean(np.abs(short_z[:, None, :] - long_z[None, :, :]), axis=2)

    m, n = cost.shape
    cumulative = np.full((m + 1, n + 1), np.inf, dtype=float)
    cumulative[0, :] = 0.0
    choose_match = np.zeros((m + 1, n + 1), dtype=bool)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if j < i:
                continue
            match = cumulative[i - 1, j - 1] + cost[i - 1, j - 1]
            skip = cumulative[i, j - 1]
            if match <= skip:
                cumulative[i, j] = match
                choose_match[i, j] = True
            else:
                cumulative[i, j] = skip

    short_indices: list[int] = []
    long_indices: list[int] = []
    i, j = m, n
    while i > 0 and j > 0:
        if choose_match[i, j]:
            short_indices.append(i - 1)
            long_indices.append(j - 1)
            i -= 1
            j -= 1
        else:
            j -= 1
    if i != 0:
        raise RuntimeError("order-preserving layer alignment failed")
    short_index = np.asarray(short_indices[::-1], dtype=int)
    long_index = np.asarray(long_indices[::-1], dtype=int)
    return (
        (short_index, long_index)
        if query_is_shorter
        else (long_index, short_index)
    )


def paper_style_alpha_similarity(
    query: pd.DataFrame,
    root: pd.DataFrame,
) -> float:
    """Mean seven-role Spearman correlation using stored DKS alpha.

    Equal-depth signatures use the paper's Eq. 2 directly. Unequal-depth
    signatures use the paper's z-scored, order-preserving dynamic-programming
    alignment. The stored estimator is DKS, not fixed-tail Hill.
    """

    query_signature = _signature_matrix(query, name="query trajectory")
    root_signature = _signature_matrix(root, name="root trajectory")
    if query_signature is None or root_signature is None:
        return float("nan")
    _, query_matrix = query_signature
    _, root_matrix = root_signature
    if len(query_matrix) == len(root_matrix):
        query_index = np.arange(len(query_matrix), dtype=int)
        root_index = np.arange(len(root_matrix), dtype=int)
    else:
        query_index, root_index = _order_preserving_layer_pairs(
            query_matrix, root_matrix
        )

    correlations: list[float] = []
    for role_index in range(len(CANONICAL_ROLES)):
        aligned_query = query_matrix[query_index, role_index]
        aligned_root = root_matrix[root_index, role_index]
        correlation = float(spearmanr(aligned_query, aligned_root).statistic)
        if not np.isfinite(correlation):
            return float("nan")
        correlations.append(correlation)
    return float(np.mean(correlations))


def _feature_columns(features: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    result = {
        method: tuple(
            column
            for column in features.columns
            if column.startswith(prefixes)
        )
        for method, prefixes in _FEATURE_PREFIXES.items()
    }
    for method, columns in result.items():
        if not columns:
            raise ValueError(f"features contain no columns for lineage method {method!r}")
    for prefix in _REQUIRED_BLOCK_PREFIXES:
        if not any(column.startswith(prefix) for column in features.columns):
            raise ValueError(f"features contain no required {prefix[:-2]} block")
    return result


def _feature_complete_ids(
    features: pd.DataFrame,
    *,
    min_feature_coverage: float,
) -> set[str]:
    required = _validate_coverage(
        min_feature_coverage, name="min_feature_coverage"
    )
    block_columns: dict[str, list[str]] = {}
    for prefix in _REQUIRED_BLOCK_PREFIXES:
        columns = [
            column
            for column in features.columns
            if column.startswith(prefix) and not column.endswith("__missing_fraction")
        ]
        if prefix == "H__":
            columns = [
                column for column in columns if "__log_alpha_norm__" not in column
            ]
        block_columns[prefix] = columns
    complete = np.ones(len(features), dtype=bool)
    for prefix, columns in block_columns.items():
        if not columns:
            raise ValueError(f"features contain no required {prefix[:-2]} measurements")
        active = [
            column
            for column in columns
            if np.isfinite(
                pd.to_numeric(features[column], errors="coerce").to_numpy(dtype=float)
            ).any()
        ]
        if not active:
            raise ValueError(f"features contain no finite {prefix[:-2]} measurements")
        numeric = features.loc[:, active].apply(pd.to_numeric, errors="coerce")
        coverage = np.mean(np.isfinite(numeric.to_numpy(dtype=float)), axis=1)
        complete &= coverage >= required
    return set(features.loc[complete, "model_id"].astype(str))


def _trajectory_complete_ids(trajectories: pd.DataFrame) -> set[str]:
    complete: set[str] = set()
    for model_id, frame in trajectories.groupby("model_id", sort=True):
        if all(_role_trajectory(frame, role) is not None for role in CANONICAL_ROLES):
            complete.add(str(model_id))
    return complete


def _prepare_population(
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    trajectories: pd.DataFrame,
    *,
    min_feature_coverage: float,
) -> _PreparedPopulation:
    clean_manifest = _canonical_manifest(manifest)
    clean_features = _canonical_features(features)
    clean_trajectories = _canonical_trajectories(trajectories)
    method_columns = _feature_columns(clean_features)
    feature_ids = set(clean_features["model_id"])
    initial_gallery, gallery_exclusions = build_root_gallery(
        clean_manifest,
        available_model_ids=feature_ids,
    )

    common_ids = _feature_complete_ids(
        clean_features,
        min_feature_coverage=min_feature_coverage,
    ).intersection(_trajectory_complete_ids(clean_trajectories))
    exclusions = gallery_exclusions.to_dict("records")

    gallery = initial_gallery.loc[
        initial_gallery["model_id"].isin(common_ids)
    ].copy()
    ineligible_roots = initial_gallery.loc[
        ~initial_gallery["model_id"].isin(common_ids)
    ]
    for row in ineligible_roots.itertuples(index=False):
        exclusions.append(
            _exclusion(
                stage="common_intersection",
                root_family=str(row.root_family),
                model_id=str(row.model_id),
                reason="root_ineligible_common_intersection",
                detail="root fails at least one M/S/H or seven-role DKS contract",
            )
        )

    represented_families = set(gallery["root_family"])
    confirmed_nonroots = clean_manifest.loc[
        clean_manifest["_confirmed"] & ~clean_manifest["is_root"]
    ].copy()
    for row in confirmed_nonroots.itertuples(index=False):
        if row.root_family not in represented_families:
            exclusions.append(
                _exclusion(
                    stage="common_intersection",
                    root_family=str(row.root_family),
                    model_id=str(row.model_id),
                    reason="query_family_without_gallery_root",
                    detail="the audited family has no eligible canonical root",
                )
            )
        elif row.model_id not in common_ids:
            exclusions.append(
                _exclusion(
                    stage="common_intersection",
                    root_family=str(row.root_family),
                    model_id=str(row.model_id),
                    reason="query_ineligible_common_intersection",
                    detail="query fails at least one M/S/H or seven-role DKS contract",
                )
            )

    queries = confirmed_nonroots.loc[
        confirmed_nonroots["root_family"].isin(represented_families)
        & confirmed_nonroots["model_id"].isin(common_ids)
    ].copy()
    families_with_queries = set(queries["root_family"])
    without_queries = gallery.loc[
        ~gallery["root_family"].isin(families_with_queries)
    ]
    for row in without_queries.itertuples(index=False):
        exclusions.append(
            _exclusion(
                stage="common_intersection",
                root_family=str(row.root_family),
                model_id=str(row.model_id),
                reason="no_eligible_non_root_query",
                detail="confirmatory families require at least one eligible descendant",
            )
        )
    gallery = gallery.loc[
        gallery["root_family"].isin(families_with_queries)
    ].sort_values("root_family", kind="stable", ignore_index=True)
    retained_families = set(gallery["root_family"])
    queries = queries.loc[queries["root_family"].isin(retained_families)].sort_values(
        "model_id", kind="stable", ignore_index=True
    )
    retained_ids = set(clean_manifest.loc[
        clean_manifest["root_family"].isin(retained_families)
        & clean_manifest["model_id"].isin(common_ids),
        "model_id",
    ])
    return _PreparedPopulation(
        manifest=clean_manifest.drop(columns=["_confirmed"], errors="ignore"),
        features=clean_features,
        trajectories=clean_trajectories,
        gallery=gallery,
        queries=queries,
        common_model_ids=frozenset(retained_ids),
        method_columns=method_columns,
        exclusions=pd.DataFrame(exclusions, columns=_EXCLUSION_COLUMNS),
    )


def _method_provenance(method: str) -> str:
    return (
        PAPER_STYLE_ALPHA_DKS_PROVENANCE
        if method == PAPER_STYLE_ALPHA_DKS
        else GALLERY_SCALED_PROVENANCE
    )


def _architecture(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().casefold()


def _validate_architecture_column(architecture_column: str | None) -> None:
    if architecture_column is not None and architecture_column not in _ARCHITECTURE_COLUMNS:
        raise ValueError(
            "architecture_column must be a validated architecture field; "
            f"received {architecture_column!r}"
        )


def _rank_queries_with_gallery(
    prepared: _PreparedPopulation,
    queries: pd.DataFrame,
    gallery: pd.DataFrame,
    *,
    min_pair_coverage: float,
    architecture_column: str | None,
    feature_index: pd.DataFrame | None = None,
    trajectory_groups: Mapping[str, pd.DataFrame] | None = None,
    paper_similarity_cache: dict[tuple[str, str], float] | None = None,
    collect_candidates: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_coverage = _validate_coverage(
        min_pair_coverage, name="min_pair_coverage"
    )
    _validate_architecture_column(architecture_column)
    if architecture_column is not None:
        _require_columns(queries, (architecture_column,), "query manifest")
        _require_columns(gallery, (architecture_column,), "gallery")
    if gallery.empty:
        raise ValueError("lineage ranking requires at least one gallery root")

    if feature_index is None:
        feature_index = prepared.features.set_index("model_id", verify_integrity=True)
    if trajectory_groups is None:
        trajectory_groups = {
            str(model_id): frame
            for model_id, frame in prepared.trajectories.groupby("model_id", sort=False)
        }
    if paper_similarity_cache is None:
        paper_similarity_cache = {}
    scalers = {
        method: _fit_gallery_scaler_from_canonical(
            prepared.features,
            gallery_model_ids=gallery["model_id"],
            columns=prepared.method_columns[method],
        )
        for method in (CONFIG_METHOD, SPECTRAL_SHAPE_METHOD, COMBINED_METHOD)
    }

    candidate_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for method in LINEAGE_METHODS:
        provenance = _method_provenance(method)
        for query in queries.sort_values("model_id", kind="stable").itertuples(index=False):
            query_values = query._asdict()
            query_id = str(query.model_id)
            true_family = str(query.root_family)
            query_architecture = (
                _architecture(query_values[architecture_column])
                if architecture_column is not None
                else ""
            )
            local_candidates: list[dict[str, object]] = []
            for root in gallery.itertuples(index=False):
                root_values = root._asdict()
                root_id = str(root.model_id)
                root_family = str(root.root_family)
                root_architecture = (
                    _architecture(root_values[architecture_column])
                    if architecture_column is not None
                    else ""
                )
                if method == PAPER_STYLE_ALPHA_DKS:
                    cache_key = (query_id, root_id)
                    if cache_key not in paper_similarity_cache:
                        paper_similarity_cache[cache_key] = paper_style_alpha_similarity(
                            trajectory_groups[query_id], trajectory_groups[root_id]
                        )
                    score = paper_similarity_cache[cache_key]
                    coverage = 1.0 if np.isfinite(score) else 0.0
                    abstained = not np.isfinite(score)
                else:
                    similarity = shared_feature_similarity(
                        feature_index.loc[query_id],
                        feature_index.loc[root_id],
                        scalers[method],
                        min_coverage=pair_coverage,
                    )
                    score = similarity.score
                    coverage = similarity.coverage
                    abstained = similarity.abstained
                architecture_match = bool(
                    query_architecture
                    and root_architecture
                    and query_architecture == root_architecture
                )
                local_candidates.append(
                    {
                        "method": method,
                        "query_model_id": query_id,
                        "true_family": true_family,
                        "candidate_root_id": root_id,
                        "candidate_family": root_family,
                        "score": score,
                        "coverage": coverage,
                        "pair_abstained": abstained,
                        "rank": pd.NA,
                        "is_true_family": root_family == true_family,
                        "query_architecture": query_architecture,
                        "candidate_architecture": root_architecture,
                        "architecture_match": architecture_match,
                        "architecture_rank": pd.NA,
                        "estimator_provenance": provenance,
                    }
                )

            finite_order = sorted(
                (
                    index
                    for index, row in enumerate(local_candidates)
                    if np.isfinite(float(row["score"]))
                ),
                key=lambda index: (
                    -float(local_candidates[index]["score"]),
                    str(local_candidates[index]["candidate_family"]),
                    str(local_candidates[index]["candidate_root_id"]),
                ),
            )
            for rank, index in enumerate(finite_order, start=1):
                local_candidates[index]["rank"] = rank
            architecture_order = [
                index
                for index in finite_order
                if bool(local_candidates[index]["architecture_match"])
            ]
            for rank, index in enumerate(architecture_order, start=1):
                local_candidates[index]["architecture_rank"] = rank

            best = local_candidates[finite_order[0]] if finite_order else None
            second = local_candidates[finite_order[1]] if len(finite_order) > 1 else None
            true_candidate = next(
                (
                    row
                    for row in local_candidates
                    if bool(row["is_true_family"]) and pd.notna(row["rank"])
                ),
                None,
            )
            architecture_best = (
                local_candidates[architecture_order[0]] if architecture_order else None
            )
            architecture_truth = next(
                (
                    row
                    for row in local_candidates
                    if bool(row["is_true_family"])
                    and pd.notna(row["architecture_rank"])
                ),
                None,
            )
            predicted_family = best["candidate_family"] if best is not None else pd.NA
            architecture_predicted = (
                architecture_best["candidate_family"]
                if architecture_best is not None
                else pd.NA
            )
            prediction_rows.append(
                {
                    "method": method,
                    "query_model_id": query_id,
                    "true_family": true_family,
                    "predicted_family": predicted_family,
                    "predicted_root_id": (
                        best["candidate_root_id"] if best is not None else pd.NA
                    ),
                    "top_score": float(best["score"]) if best is not None else float("nan"),
                    "score_margin": (
                        float(best["score"]) - float(second["score"])
                        if best is not None and second is not None
                        else float("nan")
                    ),
                    "abstained": best is None,
                    "true_rank": true_candidate["rank"] if true_candidate else pd.NA,
                    "candidate_count": len(local_candidates),
                    "correct": bool(best is not None and predicted_family == true_family),
                    "architecture_matched_predicted_family": architecture_predicted,
                    "architecture_matched_true_rank": (
                        architecture_truth["architecture_rank"]
                        if architecture_truth is not None
                        else pd.NA
                    ),
                    "architecture_matched_eligible": architecture_best is not None,
                    "architecture_matched_correct": bool(
                        architecture_best is not None
                        and architecture_predicted == true_family
                    ),
                    "estimator_provenance": provenance,
                }
            )
            if collect_candidates:
                candidate_rows.extend(local_candidates)

    candidates = pd.DataFrame(candidate_rows)
    predictions = pd.DataFrame(prediction_rows)
    return candidates, predictions


def rank_confirmed_queries(
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    alpha_trajectories: pd.DataFrame,
    *,
    architecture_column: str | None = None,
    min_feature_coverage: float = 0.8,
    min_pair_coverage: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rank every eligible confirmed non-root query against every root."""

    prepared = _prepare_population(
        manifest,
        features,
        alpha_trajectories,
        min_feature_coverage=min_feature_coverage,
    )
    candidates, predictions = _rank_queries_with_gallery(
        prepared,
        prepared.queries,
        prepared.gallery,
        min_pair_coverage=min_pair_coverage,
        architecture_column=architecture_column,
    )
    return candidates, predictions, prepared.exclusions.copy()


def rank_cross_fitted_family_representatives(
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    alpha_trajectories: pd.DataFrame,
    *,
    architecture_column: str | None = None,
    min_feature_coverage: float = 0.8,
    min_pair_coverage: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attribute audited models to query-excluded observed-family medoids.

    This descriptive endpoint targets a known reference library. It does not
    substitute descendants for canonical roots in the primary lineage endpoint.
    """

    _validate_architecture_column(architecture_column)
    clean_manifest = _canonical_manifest(manifest)
    clean_features = _canonical_features(features)
    clean_trajectories = _canonical_trajectories(alpha_trajectories)
    method_columns = _feature_columns(clean_features)
    common_ids = _feature_complete_ids(
        clean_features,
        min_feature_coverage=min_feature_coverage,
    ).intersection(_trajectory_complete_ids(clean_trajectories))
    confirmed = clean_manifest.loc[clean_manifest["_confirmed"]].copy()
    population = confirmed.loc[confirmed["model_id"].isin(common_ids)].copy()
    population = population.sort_values("model_id", kind="stable", ignore_index=True)
    exclusions = [
        _exclusion(
            stage="cross_fitted_representative",
            root_family=str(row.root_family),
            model_id=str(row.model_id),
            reason="representative_ineligible_common_intersection",
            detail="model fails at least one M/S/H or seven-role DKS contract",
        )
        for row in confirmed.loc[~confirmed["model_id"].isin(common_ids)].itertuples(
            index=False
        )
    ]
    if population.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(
            exclusions, columns=_EXCLUSION_COLUMNS
        )
    if architecture_column is not None:
        _require_columns(population, (architecture_column,), "representative manifest")

    feature_index = clean_features.set_index("model_id", verify_integrity=True)
    trajectory_groups = {
        str(model_id): frame
        for model_id, frame in clean_trajectories.groupby("model_id", sort=False)
    }
    paper_cache: dict[tuple[str, str], float] = {}

    def paper_score(left: str, right: str) -> float:
        key = tuple(sorted((left, right)))
        if key not in paper_cache:
            paper_cache[key] = paper_style_alpha_similarity(
                trajectory_groups[left], trajectory_groups[right]
            )
        return paper_cache[key]

    candidate_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for query in population.itertuples(index=False):
        query_values = query._asdict()
        query_id = str(query.model_id)
        true_family = str(query.root_family)
        references = population.loc[~population["model_id"].eq(query_id)].copy()
        family_references = {
            str(family): sorted(group["model_id"].astype(str).tolist())
            for family, group in references.groupby("root_family", sort=True)
        }
        if true_family not in family_references:
            exclusions.append(
                _exclusion(
                    stage="cross_fitted_representative",
                    root_family=true_family,
                    model_id=query_id,
                    reason="query_without_family_reference",
                    detail="query exclusion leaves no observed member of its family",
                )
            )
            continue
        reference_ids = references["model_id"].astype(str).tolist()
        scalers = {
            method: _fit_gallery_scaler_from_canonical(
                clean_features,
                gallery_model_ids=reference_ids,
                columns=method_columns[method],
            )
            for method in (CONFIG_METHOD, SPECTRAL_SHAPE_METHOD, COMBINED_METHOD)
        }

        for method in LINEAGE_METHODS:
            local_candidates: list[dict[str, object]] = []
            for family, family_ids in family_references.items():
                if method == PAPER_STYLE_ALPHA_DKS:
                    representative_id = min(
                        family_ids,
                        key=lambda candidate: (
                            -float(
                                np.nanmedian(
                                    [
                                        paper_score(candidate, peer)
                                        for peer in family_ids
                                        if peer != candidate
                                    ]
                                )
                            )
                            if len(family_ids) > 1
                            else 0.0,
                            candidate,
                        ),
                    )
                    score = paper_score(query_id, representative_id)
                    coverage = 1.0 if np.isfinite(score) else 0.0
                    abstained = not np.isfinite(score)
                else:
                    scaler = scalers[method]
                    values = feature_index.loc[
                        family_ids, list(scaler.columns)
                    ].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                    standardized = (
                        values - np.asarray(scaler.center, dtype=float)
                    ) / np.asarray(scaler.scale, dtype=float)
                    center = np.nanmedian(standardized, axis=0)
                    shared = np.isfinite(standardized) & np.isfinite(center)[None, :]
                    coverage_by_model = shared.mean(axis=1)
                    squared = np.where(
                        shared, np.square(standardized - center), np.nan
                    )
                    distance = np.sqrt(np.nanmean(squared, axis=1))
                    distance[
                        (coverage_by_model < min_pair_coverage)
                        | ~np.isfinite(distance)
                    ] = np.inf
                    representative_index = min(
                        range(len(family_ids)),
                        key=lambda index: (
                            float(distance[index]), family_ids[index]
                        ),
                    )
                    representative_id = family_ids[representative_index]
                    similarity = shared_feature_similarity(
                        feature_index.loc[query_id],
                        feature_index.loc[representative_id],
                        scaler,
                        min_coverage=min_pair_coverage,
                    )
                    score = similarity.score
                    coverage = similarity.coverage
                    abstained = similarity.abstained
                representative = population.loc[
                    population["model_id"].eq(representative_id)
                ].iloc[0]
                query_architecture = (
                    _architecture(query_values[architecture_column])
                    if architecture_column is not None
                    else ""
                )
                representative_architecture = (
                    _architecture(representative[architecture_column])
                    if architecture_column is not None
                    else ""
                )
                local_candidates.append(
                    {
                        "endpoint": "cross_fitted_observed_family_representative",
                        "method": method,
                        "query_model_id": query_id,
                        "true_family": true_family,
                        "candidate_family": family,
                        "candidate_representative_id": representative_id,
                        "candidate_root_id": representative_id,
                        "score": score,
                        "coverage": coverage,
                        "pair_abstained": abstained,
                        "rank": pd.NA,
                        "is_true_family": family == true_family,
                        "architecture_match": bool(
                            query_architecture
                            and representative_architecture
                            and query_architecture == representative_architecture
                        ),
                        "estimator_provenance": _method_provenance(method),
                        "endpoint_provenance": "query_excluded_scaling_and_observed_family_medoid",
                    }
                )

            finite_order = sorted(
                (
                    index
                    for index, row in enumerate(local_candidates)
                    if np.isfinite(float(row["score"]))
                ),
                key=lambda index: (
                    -float(local_candidates[index]["score"]),
                    str(local_candidates[index]["candidate_family"]),
                    str(local_candidates[index]["candidate_representative_id"]),
                ),
            )
            for rank, index in enumerate(finite_order, start=1):
                local_candidates[index]["rank"] = rank
            best = local_candidates[finite_order[0]] if finite_order else None
            second = local_candidates[finite_order[1]] if len(finite_order) > 1 else None
            true_candidate = next(
                (
                    row
                    for row in local_candidates
                    if row["is_true_family"] and pd.notna(row["rank"])
                ),
                None,
            )
            architecture_order = [
                index
                for index in finite_order
                if local_candidates[index]["architecture_match"]
            ]
            architecture_best = (
                local_candidates[architecture_order[0]] if architecture_order else None
            )
            predicted_family = best["candidate_family"] if best is not None else pd.NA
            prediction_rows.append(
                {
                    "endpoint": "cross_fitted_observed_family_representative",
                    "method": method,
                    "query_model_id": query_id,
                    "true_family": true_family,
                    "predicted_family": predicted_family,
                    "predicted_representative_id": (
                        best["candidate_representative_id"] if best else pd.NA
                    ),
                    "predicted_root_id": (
                        best["candidate_representative_id"] if best else pd.NA
                    ),
                    "top_score": float(best["score"]) if best else float("nan"),
                    "score_margin": (
                        float(best["score"]) - float(second["score"])
                        if best is not None and second is not None
                        else float("nan")
                    ),
                    "abstained": best is None,
                    "true_rank": true_candidate["rank"] if true_candidate else pd.NA,
                    "candidate_count": len(local_candidates),
                    "correct": bool(best is not None and predicted_family == true_family),
                    "architecture_matched_eligible": architecture_best is not None,
                    "architecture_matched_correct": bool(
                        architecture_best is not None
                        and architecture_best["candidate_family"] == true_family
                    ),
                    "estimator_provenance": _method_provenance(method),
                    "endpoint_provenance": "query_excluded_scaling_and_observed_family_medoid",
                }
            )
            candidate_rows.extend(local_candidates)

    return (
        pd.DataFrame(candidate_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(exclusions, columns=_EXCLUSION_COLUMNS),
    )


def root_macro_top1(
    predictions: pd.DataFrame,
    *,
    family_column: str = "true_family",
) -> float:
    _require_columns(predictions, (family_column, "correct"), "predictions")
    if predictions.empty:
        return float("nan")
    family_accuracy = predictions.assign(
        _correct=predictions["correct"].astype(bool).astype(float)
    ).groupby(family_column, sort=True)["_correct"].mean()
    return float(family_accuracy.mean()) if not family_accuracy.empty else float("nan")


def _closed_set_metrics(
    predictions: pd.DataFrame,
    *,
    family_column: str,
) -> dict[str, float]:
    _require_columns(
        predictions,
        (family_column, "correct", "true_rank"),
        "predictions",
    )
    if predictions.empty:
        return {
            "root_macro_top1": float("nan"),
            "recall@3": float("nan"),
            "recall@5": float("nan"),
            "mrr": float("nan"),
            "micro_top1": float("nan"),
            "architecture_matched_error": float("nan"),
            "n_queries": 0.0,
            "n_families": 0.0,
            "architecture_matched_n": 0.0,
        }
    working = predictions.copy()
    working["_rank"] = pd.to_numeric(working["true_rank"], errors="coerce")
    working["_valid_rank"] = np.isfinite(working["_rank"]) & working["_rank"].gt(0)
    working["_reciprocal_rank"] = np.where(
        working["_valid_rank"], 1.0 / working["_rank"], 0.0
    )
    working["_correct"] = working["correct"].astype(bool)
    family_rows: list[dict[str, float]] = []
    for _, family in working.groupby(family_column, sort=True):
        architecture_eligible = (
            family["architecture_matched_eligible"].astype(bool)
            if "architecture_matched_eligible" in family.columns
            else pd.Series(False, index=family.index)
        )
        architecture_correct = (
            family["architecture_matched_correct"].astype(bool)
            if "architecture_matched_correct" in family.columns
            else pd.Series(False, index=family.index)
        )
        family_rows.append(
            {
                "recall@3": float(
                    np.mean(family["_valid_rank"] & family["_rank"].le(3))
                ),
                "recall@5": float(
                    np.mean(family["_valid_rank"] & family["_rank"].le(5))
                ),
                "mrr": float(family["_reciprocal_rank"].mean()),
                "architecture_matched_error": (
                    float(np.mean(~architecture_correct.loc[architecture_eligible]))
                    if architecture_eligible.any()
                    else float("nan")
                ),
            }
        )
    family_metrics = pd.DataFrame(family_rows)
    architecture_eligible = (
        working["architecture_matched_eligible"].astype(bool).to_numpy()
        if "architecture_matched_eligible" in working.columns
        else np.zeros(len(working), dtype=bool)
    )
    return {
        "root_macro_top1": root_macro_top1(
            predictions, family_column=family_column
        ),
        "recall@3": float(family_metrics["recall@3"].mean()),
        "recall@5": float(family_metrics["recall@5"].mean()),
        "mrr": float(family_metrics["mrr"].mean()),
        "micro_top1": float(working["_correct"].mean()),
        "architecture_matched_error": (
            float(family_metrics["architecture_matched_error"].mean())
            if family_metrics["architecture_matched_error"].notna().any()
            else float("nan")
        ),
        "n_queries": float(len(predictions)),
        "n_families": float(predictions[family_column].nunique()),
        "architecture_matched_n": float(np.count_nonzero(architecture_eligible)),
    }


def closed_set_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """Compute one method's closed-set metrics."""

    if "method" in predictions.columns and predictions["method"].nunique() > 1:
        raise ValueError("closed_set_metrics accepts one method at a time")
    return _closed_set_metrics(predictions, family_column="true_family")


def closed_set_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    methods = (
        predictions["method"].drop_duplicates().tolist()
        if "method" in predictions.columns
        else [""]
    )
    rows: list[dict[str, object]] = []
    for method in methods:
        selected = (
            predictions.loc[predictions["method"].eq(method)]
            if "method" in predictions.columns
            else predictions
        )
        metrics = closed_set_metrics(selected)
        for metric in (
            "root_macro_top1",
            "recall@3",
            "recall@5",
            "mrr",
            "micro_top1",
            "architecture_matched_error",
        ):
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "estimate": metrics[metric],
                    "n_queries": int(metrics["n_queries"]),
                    "n_families": int(metrics["n_families"]),
                    "architecture_matched_n": int(metrics["architecture_matched_n"]),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_closed_set_metrics(
    predictions: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    random_state: int,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Bootstrap complete root families, never individual query rows."""

    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    _require_columns(predictions, ("true_family", "correct", "true_rank"), "predictions")
    methods = (
        predictions["method"].drop_duplicates().tolist()
        if "method" in predictions.columns
        else [""]
    )
    metric_names = (
        "root_macro_top1",
        "recall@3",
        "recall@5",
        "mrr",
        "micro_top1",
        "architecture_matched_error",
    )
    rows: list[dict[str, object]] = []
    alpha = (1.0 - confidence) / 2.0

    for method in methods:
        # Methods share the same family draws, enabling paired comparisons.
        rng = np.random.default_rng(random_state)
        selected = (
            predictions.loc[predictions["method"].eq(method)].copy()
            if "method" in predictions.columns
            else predictions.copy()
        )
        families = sorted(selected["true_family"].dropna().astype(str).unique())
        if not families:
            continue
        estimates = closed_set_metrics(selected)
        family_rows: list[dict[str, float]] = []
        family_values = selected["true_family"].astype(str)
        for family in families:
            frame = selected.loc[family_values.eq(family)]
            rank = pd.to_numeric(frame["true_rank"], errors="coerce").to_numpy(
                dtype=float
            )
            valid_rank = np.isfinite(rank) & (rank > 0.0)
            correct = frame["correct"].astype(bool).to_numpy(dtype=bool)
            architecture_eligible = (
                frame["architecture_matched_eligible"]
                .astype(bool)
                .to_numpy(dtype=bool)
                if "architecture_matched_eligible" in frame.columns
                else np.zeros(len(frame), dtype=bool)
            )
            architecture_correct = (
                frame["architecture_matched_correct"]
                .astype(bool)
                .to_numpy(dtype=bool)
                if "architecture_matched_correct" in frame.columns
                else np.zeros(len(frame), dtype=bool)
            )
            family_rows.append(
                {
                    "n_queries": float(len(frame)),
                    "correct_count": float(np.count_nonzero(correct)),
                    "root_macro_top1": float(np.mean(correct)),
                    "recall@3": float(np.mean(valid_rank & (rank <= 3.0))),
                    "recall@5": float(np.mean(valid_rank & (rank <= 5.0))),
                    "mrr": float(
                        np.mean(np.where(valid_rank, 1.0 / rank, 0.0))
                    ),
                    "architecture_matched_error": (
                        float(np.mean(~architecture_correct[architecture_eligible]))
                        if np.any(architecture_eligible)
                        else float("nan")
                    ),
                }
            )
        family_statistics = pd.DataFrame(family_rows)
        sampled = rng.choice(
            len(families),
            size=(bootstrap_replicates, len(families)),
            replace=True,
        )
        draws: dict[str, np.ndarray] = {}
        for name in ("root_macro_top1", "recall@3", "recall@5", "mrr"):
            values = family_statistics[name].to_numpy(dtype=float)
            draws[name] = values[sampled].mean(axis=1)
        query_counts = family_statistics["n_queries"].to_numpy(dtype=float)
        correct_counts = family_statistics["correct_count"].to_numpy(dtype=float)
        draws["micro_top1"] = (
            correct_counts[sampled].sum(axis=1)
            / query_counts[sampled].sum(axis=1)
        )
        architecture_error = family_statistics[
            "architecture_matched_error"
        ].to_numpy(dtype=float)[sampled]
        architecture_count = np.isfinite(architecture_error).sum(axis=1)
        draws["architecture_matched_error"] = np.divide(
            np.nansum(architecture_error, axis=1),
            architecture_count,
            out=np.full(bootstrap_replicates, np.nan, dtype=float),
            where=architecture_count > 0,
        )

        for name in metric_names:
            values = draws[name]
            finite = values[np.isfinite(values)]
            ci_low = float(np.quantile(finite, alpha)) if finite.size else float("nan")
            ci_high = (
                float(np.quantile(finite, 1.0 - alpha))
                if finite.size
                else float("nan")
            )
            rows.append(
                {
                    "method": method,
                    "metric": name,
                    "estimate": estimates[name],
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "confidence": confidence,
                    "bootstrap_unit": "root_family",
                    "bootstrap_replicates": bootstrap_replicates,
                    "n_families": len(families),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_closed_set_contrasts(
    predictions: pd.DataFrame,
    *,
    challenger: str = COMBINED_METHOD,
    reference: str = PAPER_STYLE_ALPHA_DKS,
    bootstrap_replicates: int,
    random_state: int,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Paired family-bootstrap contrasts on one common query population."""

    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    _require_columns(
        predictions,
        ("method", "query_model_id", "true_family", "correct", "true_rank"),
        "predictions",
    )
    selected = predictions.loc[predictions["method"].isin((challenger, reference))]
    by_method = {
        method: frame.copy()
        for method, frame in selected.groupby("method", sort=False)
    }
    if set(by_method) != {challenger, reference}:
        raise ValueError("both challenger and reference predictions are required")
    key_columns = ["query_model_id", "true_family"]
    keys = {
        method: set(map(tuple, frame[key_columns].itertuples(index=False, name=None)))
        for method, frame in by_method.items()
    }
    if keys[challenger] != keys[reference]:
        raise ValueError("paired lineage contrasts require identical query populations")

    metric_names = (
        "root_macro_top1",
        "recall@3",
        "recall@5",
        "mrr",
        "architecture_matched_error",
    )
    families = sorted(selected["true_family"].astype(str).unique())
    family_deltas: dict[str, list[float]] = {metric: [] for metric in metric_names}
    for family in families:
        metrics = {}
        for method in (challenger, reference):
            frame = by_method[method].loc[
                by_method[method]["true_family"].astype(str).eq(family)
            ]
            metrics[method] = closed_set_metrics(frame)
        for metric in metric_names:
            family_deltas[metric].append(
                float(metrics[challenger][metric]) - float(metrics[reference][metric])
            )

    rng = np.random.default_rng(random_state)
    alpha = (1.0 - confidence) / 2.0
    rows: list[dict[str, object]] = []
    for metric in metric_names:
        values = np.asarray(family_deltas[metric], dtype=float)
        finite = np.isfinite(values)
        values = values[finite]
        if values.size:
            draws = rng.choice(
                values, size=(bootstrap_replicates, len(values)), replace=True
            ).mean(axis=1)
            estimate = float(np.mean(values))
            ci_low = float(np.quantile(draws, alpha))
            ci_high = float(np.quantile(draws, 1.0 - alpha))
        else:
            estimate = ci_low = ci_high = float("nan")
        rows.append(
            {
                "contrast": f"{challenger}-{reference}",
                "challenger": challenger,
                "reference": reference,
                "metric": metric,
                "estimate": estimate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "confidence": confidence,
                "bootstrap_unit": "root_family",
                "bootstrap_replicates": bootstrap_replicates,
                "n_families": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def evaluate_closed_set_lineage(
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    alpha_trajectories: pd.DataFrame,
    *,
    architecture_column: str | None = None,
    min_feature_coverage: float = 0.8,
    min_pair_coverage: float = 0.8,
    bootstrap_replicates: int = 2_000,
    random_state: int = 0,
) -> dict[str, pd.DataFrame]:
    candidates, predictions, exclusions = rank_confirmed_queries(
        manifest,
        features,
        alpha_trajectories,
        architecture_column=architecture_column,
        min_feature_coverage=min_feature_coverage,
        min_pair_coverage=min_pair_coverage,
    )
    return {
        "candidates": candidates,
        "predictions": predictions,
        "metrics": closed_set_metric_table(predictions),
        "confidence_intervals": bootstrap_closed_set_metrics(
            predictions,
            bootstrap_replicates=bootstrap_replicates,
            random_state=random_state,
        ),
        "contrasts": bootstrap_closed_set_contrasts(
            predictions,
            bootstrap_replicates=bootstrap_replicates,
            random_state=random_state,
        ),
        "exclusions": exclusions,
    }


def evaluate_cross_fitted_family_representatives(
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    alpha_trajectories: pd.DataFrame,
    *,
    architecture_column: str | None = None,
    min_feature_coverage: float = 0.8,
    min_pair_coverage: float = 0.8,
    bootstrap_replicates: int = 2_000,
    random_state: int = 0,
) -> dict[str, pd.DataFrame]:
    """Evaluate descriptive attribution to query-excluded family medoids."""

    candidates, predictions, exclusions = (
        rank_cross_fitted_family_representatives(
            manifest,
            features,
            alpha_trajectories,
            architecture_column=architecture_column,
            min_feature_coverage=min_feature_coverage,
            min_pair_coverage=min_pair_coverage,
        )
    )
    if predictions.empty:
        empty = pd.DataFrame()
        return {
            "candidates": candidates,
            "predictions": predictions,
            "metrics": empty.copy(),
            "confidence_intervals": empty.copy(),
            "contrasts": empty.copy(),
            "exclusions": exclusions,
        }
    return {
        "candidates": candidates,
        "predictions": predictions,
        "metrics": closed_set_metric_table(predictions),
        "confidence_intervals": bootstrap_closed_set_metrics(
            predictions,
            bootstrap_replicates=bootstrap_replicates,
            random_state=random_state,
        ),
        "contrasts": bootstrap_closed_set_contrasts(
            predictions,
            bootstrap_replicates=bootstrap_replicates,
            random_state=random_state,
        ),
        "exclusions": exclusions,
    }


def unknown_family_folds(
    manifest: pd.DataFrame,
    *,
    gallery: pd.DataFrame | None = None,
    available_model_ids: Iterable[object] | None = None,
    eligible_model_ids: Iterable[object] | None = None,
) -> Iterable[UnknownFamilyFold]:
    """Yield leave-one-root-family-out pseudo-unknown folds."""

    working = _canonical_manifest(manifest)
    if gallery is None:
        gallery, _ = build_root_gallery(
            working,
            available_model_ids=available_model_ids,
        )
    else:
        gallery = gallery.copy(deep=True)
        _require_columns(gallery, ("model_id", "root_family"), "gallery")
        gallery["model_id"] = [
            _canonical_id(value, field="gallery.model_id")
            for value in gallery["model_id"]
        ]
        gallery["root_family"] = [
            _canonical_id(value, field="gallery.root_family")
            for value in gallery["root_family"]
        ]
    eligible = (
        {
            _canonical_id(value, field="eligible_model_ids")
            for value in eligible_model_ids
        }
        if eligible_model_ids is not None
        else set(working["model_id"])
    )
    families = tuple(sorted(gallery["root_family"].astype(str).unique()))
    confirmed = working.loc[
        working["_confirmed"]
        & working["root_family"].isin(families)
        & working["model_id"].isin(eligible)
    ].copy()
    for held_family in families:
        fold_gallery = gallery.loc[
            ~gallery["root_family"].eq(held_family)
        ].sort_values("root_family", kind="stable", ignore_index=True)
        unknown = confirmed.loc[
            confirmed["root_family"].eq(held_family)
        ].sort_values("model_id", kind="stable", ignore_index=True)
        known = confirmed.loc[
            ~confirmed["root_family"].eq(held_family)
            & ~confirmed["is_root"]
            & confirmed["root_family"].isin(fold_gallery["root_family"])
        ].sort_values("model_id", kind="stable", ignore_index=True)
        yield UnknownFamilyFold(
            held_out_family=held_family,
            gallery=fold_gallery.drop(columns=["_confirmed"], errors="ignore"),
            known_queries=known.drop(columns=["_confirmed"], errors="ignore"),
            unknown_queries=unknown.drop(columns=["_confirmed"], errors="ignore"),
        )


def _oscr_score(
    confidence: np.ndarray,
    known_label: np.ndarray,
    correct: np.ndarray,
) -> float:
    known_count = int(np.count_nonzero(known_label == 1))
    unknown_count = int(np.count_nonzero(known_label == 0))
    if known_count == 0 or unknown_count == 0:
        return float("nan")
    thresholds = np.concatenate(
        ([np.inf], np.sort(np.unique(confidence))[::-1], [-np.inf])
    )
    fpr: list[float] = []
    ccr: list[float] = []
    for threshold in thresholds:
        accepted = confidence >= threshold
        fpr.append(float(np.count_nonzero(accepted & (known_label == 0))) / unknown_count)
        ccr.append(
            float(np.count_nonzero(accepted & (known_label == 1) & correct))
            / known_count
        )
    order = np.argsort(np.asarray(fpr), kind="stable")
    return float(trapezoid(np.asarray(ccr)[order], np.asarray(fpr)[order]))


def _risk_coverage_curve(
    confidence: np.ndarray,
    correct: np.ndarray,
    *,
    total_count: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    total = len(confidence) if total_count is None else int(total_count)
    if total < len(confidence):
        raise ValueError("total_count cannot be smaller than scorable confidence rows")
    if total == 0:
        return pd.DataFrame(columns=["coverage", "risk", "threshold"])
    for threshold in np.sort(np.unique(confidence))[::-1]:
        accepted = confidence >= threshold
        accepted_count = int(np.count_nonzero(accepted))
        rows.append(
            {
                "coverage": accepted_count / total,
                "risk": float(np.mean(~correct[accepted])),
                "threshold": float(threshold),
            }
        )
    return pd.DataFrame(rows, columns=["coverage", "risk", "threshold"])


def _risk_coverage_area(curve: pd.DataFrame) -> float:
    if curve.empty:
        return float("nan")
    ordered = curve.sort_values("coverage", kind="stable")
    coverage = ordered["coverage"].to_numpy(dtype=float)
    risk = ordered["risk"].to_numpy(dtype=float)
    increments = np.diff(np.concatenate(([0.0], coverage)))
    area = float(np.sum(increments * risk))
    return area + max(0.0, 1.0 - float(np.max(coverage)))


def compute_open_set_metrics(
    confidence: np.ndarray,
    known_label: np.ndarray,
    correct: np.ndarray,
) -> dict[str, float]:
    """Compute threshold-free open-set and selective-risk metrics."""

    scores = np.asarray(confidence, dtype=float)
    labels = np.asarray(known_label, dtype=int)
    correctness = np.asarray(correct, dtype=bool)
    if not (scores.shape == labels.shape == correctness.shape):
        raise ValueError("confidence, known_label, and correct must have equal shapes")
    if set(np.unique(labels)) != {0, 1}:
        return {
            "oscr": float("nan"),
            "auroc": float("nan"),
            "auprc": float("nan"),
            "fpr95": float("nan"),
            "risk_coverage_area": float("nan"),
            "scorable_coverage": float(np.mean(np.isfinite(scores))) if scores.size else 0.0,
            "abstention_rate": float(np.mean(~np.isfinite(scores))) if scores.size else 0.0,
        }
    finite = np.isfinite(scores)
    finite_scores = scores[finite]
    if finite_scores.size:
        spread = float(np.ptp(finite_scores))
        floor = float(np.min(finite_scores) - max(0.05 * spread, 1e-12))
    else:
        floor = 0.0
    ranking_scores = np.where(finite, scores, floor)
    fpr, tpr, _ = roc_curve(labels, ranking_scores)
    reaches_95 = np.flatnonzero(tpr >= 0.95)
    curve = _risk_coverage_curve(
        scores[finite], correctness[finite], total_count=len(scores)
    )
    return {
        "oscr": _oscr_score(ranking_scores, labels, correctness),
        "auroc": float(roc_auc_score(labels, ranking_scores)),
        "auprc": float(average_precision_score(labels, ranking_scores)),
        "fpr95": float(fpr[reaches_95[0]]) if reaches_95.size else float("nan"),
        "risk_coverage_area": _risk_coverage_area(curve),
        "scorable_coverage": float(np.mean(finite)),
        "abstention_rate": float(np.mean(~finite)),
    }


def evaluate_unknown_family_rejection(
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    alpha_trajectories: pd.DataFrame,
    *,
    architecture_column: str | None = None,
    min_feature_coverage: float = 0.8,
    min_pair_coverage: float = 0.8,
) -> dict[str, pd.DataFrame]:
    """Evaluate leave-one-family-out pseudo-unknown rejection without calibration."""

    prepared = _prepare_population(
        manifest,
        features,
        alpha_trajectories,
        min_feature_coverage=min_feature_coverage,
    )
    score_rows: list[dict[str, object]] = []
    fold_gallery_rows: list[dict[str, str]] = []
    exclusions = prepared.exclusions.to_dict("records")
    folds = unknown_family_folds(
        prepared.manifest,
        gallery=prepared.gallery,
        eligible_model_ids=prepared.common_model_ids,
    )
    feature_index = prepared.features.set_index("model_id", verify_integrity=True)
    trajectory_groups = {
        str(model_id): frame
        for model_id, frame in prepared.trajectories.groupby("model_id", sort=False)
    }
    paper_similarity_cache: dict[tuple[str, str], float] = {}
    for fold in folds:
        if fold.gallery.empty or fold.known_queries.empty or fold.unknown_queries.empty:
            exclusions.append(
                _exclusion(
                    stage="open_set",
                    root_family=fold.held_out_family,
                    model_id=None,
                    reason="ineligible_unknown_family_fold",
                    detail="fold requires gallery roots, known queries, and unknown queries",
                )
            )
            continue
        for root in fold.gallery.itertuples(index=False):
            fold_gallery_rows.append(
                {
                    "held_out_family": fold.held_out_family,
                    "gallery_family": str(root.root_family),
                    "gallery_root_id": str(root.model_id),
                }
            )
        queries = pd.concat(
            [
                fold.known_queries.assign(_known_label=1),
                fold.unknown_queries.assign(_known_label=0),
            ],
            ignore_index=True,
            sort=False,
        )
        _, predictions = _rank_queries_with_gallery(
            prepared,
            queries,
            fold.gallery,
            min_pair_coverage=min_pair_coverage,
            architecture_column=architecture_column,
            feature_index=feature_index,
            trajectory_groups=trajectory_groups,
            paper_similarity_cache=paper_similarity_cache,
            collect_candidates=False,
        )
        known_by_id = queries.set_index("model_id")["_known_label"].to_dict()
        for prediction in predictions.itertuples(index=False):
            known = int(known_by_id[prediction.query_model_id])
            score_rows.append(
                {
                    "held_out_family": fold.held_out_family,
                    "method": prediction.method,
                    "query_model_id": prediction.query_model_id,
                    "true_family": prediction.true_family,
                    "known_label": known,
                    "predicted_family": prediction.predicted_family,
                    "confidence": prediction.top_score,
                    "correct": bool(known == 1 and prediction.correct),
                    "abstained": prediction.abstained,
                    "estimator_provenance": prediction.estimator_provenance,
                }
            )

    scores = pd.DataFrame(score_rows)
    fold_metric_rows: list[dict[str, object]] = []
    curve_rows: list[pd.DataFrame] = []
    if not scores.empty:
        for (method, held_out_family), selected in scores.groupby(
            ["method", "held_out_family"], sort=True
        ):
            metrics = compute_open_set_metrics(
                selected["confidence"].to_numpy(dtype=float),
                selected["known_label"].to_numpy(dtype=int),
                selected["correct"].to_numpy(dtype=bool),
            )
            for metric, estimate in metrics.items():
                fold_metric_rows.append(
                    {
                        "method": method,
                        "held_out_family": held_out_family,
                        "metric": metric,
                        "estimate": estimate,
                        "n_known": int(selected["known_label"].sum()),
                        "n_unknown": int((1 - selected["known_label"]).sum()),
                        "n_scorable": int(
                            np.isfinite(selected["confidence"].to_numpy(dtype=float)).sum()
                        ),
                        "n_abstained": int(
                            (~np.isfinite(selected["confidence"].to_numpy(dtype=float))).sum()
                        ),
                    }
                )
            finite = np.isfinite(selected["confidence"].to_numpy(dtype=float))
            curve = _risk_coverage_curve(
                selected.loc[finite, "confidence"].to_numpy(dtype=float),
                selected.loc[finite, "correct"].to_numpy(dtype=bool),
                total_count=len(selected),
            )
            if not curve.empty:
                curve.insert(0, "method", method)
                curve.insert(1, "held_out_family", held_out_family)
                curve_rows.append(curve)
    fold_metrics = pd.DataFrame(
        fold_metric_rows,
        columns=[
            "method",
            "held_out_family",
            "metric",
            "estimate",
            "n_known",
            "n_unknown",
            "n_scorable",
            "n_abstained",
        ],
    )
    aggregate_rows: list[dict[str, object]] = []
    if not fold_metrics.empty:
        for (method, metric), group in fold_metrics.groupby(
            ["method", "metric"], sort=True
        ):
            values = pd.to_numeric(group["estimate"], errors="coerce")
            aggregate_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "estimate": float(values.mean()) if values.notna().any() else float("nan"),
                    "n_folds": int(group["held_out_family"].nunique()),
                    "n_known": int(group["n_known"].sum()),
                    "n_unknown": int(group["n_unknown"].sum()),
                    "n_scorable": int(group["n_scorable"].sum()),
                    "n_abstained": int(group["n_abstained"].sum()),
                    "aggregation": "held_out_family_macro",
                }
            )
    metrics_frame = pd.DataFrame(
        aggregate_rows,
        columns=[
            "method",
            "metric",
            "estimate",
            "n_folds",
            "n_known",
            "n_unknown",
            "n_scorable",
            "n_abstained",
            "aggregation",
        ],
    )
    risk_coverage = (
        pd.concat(curve_rows, ignore_index=True)
        if curve_rows
        else pd.DataFrame(
            columns=["method", "held_out_family", "coverage", "risk", "threshold"]
        )
    )
    return {
        "fold_gallery": pd.DataFrame(
            fold_gallery_rows,
            columns=["held_out_family", "gallery_family", "gallery_root_id"],
        ),
        "scores": scores,
        "fold_metrics": fold_metrics,
        "metrics": metrics_frame,
        "risk_coverage": risk_coverage,
        "exclusions": pd.DataFrame(exclusions, columns=_EXCLUSION_COLUMNS),
    }


def _root_macro_purity(assignments: pd.DataFrame) -> float:
    if assignments.empty:
        return float("nan")
    families = sorted(assignments["root_family"].dropna().astype(str).unique())
    clusters = sorted(assignments["cluster"].dropna().unique())
    if not families or not clusters:
        return float("nan")
    contingency = np.zeros((len(families), len(clusters)), dtype=float)
    family_index = {family: index for index, family in enumerate(families)}
    cluster_index = {cluster: index for index, cluster in enumerate(clusters)}
    for row in assignments.itertuples(index=False):
        family = str(row.root_family)
        if family in family_index and row.cluster in cluster_index:
            contingency[family_index[family], cluster_index[row.cluster]] += 1.0
    matched_families, matched_clusters = linear_sum_assignment(-contingency)
    correct = np.zeros(len(families), dtype=float)
    correct[matched_families] = contingency[matched_families, matched_clusters]
    totals = contingency.sum(axis=1)
    recalls = np.divide(correct, totals, out=np.zeros_like(correct), where=totals > 0)
    return float(np.mean(recalls))


def label_independent_cluster_count(
    model_count: int, *, rule: str = "sqrt_half"
) -> int:
    if model_count < 3:
        raise ValueError("clustering requires at least three eligible models")
    if rule != "sqrt_half":
        raise ValueError(f"unknown cluster count rule: {rule}")
    proposed = int(np.ceil(np.sqrt(model_count / 2.0)))
    return min(model_count - 1, max(2, proposed))


def cluster_lineage_diagnostics(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    n_clusters: int | None,
    random_state: int,
    min_feature_coverage: float = 0.8,
    cluster_count_rule: str = "sqrt_half",
) -> dict[str, pd.DataFrame]:
    """Fit K-means without labels, then evaluate frozen assignments."""

    if n_clusters is not None and n_clusters < 2:
        raise ValueError("n_clusters must be at least two and predeclared")
    clean_features = _canonical_features(features)
    columns = tuple(
        column
        for column in clean_features.columns
        if column.startswith(_REQUIRED_BLOCK_PREFIXES)
    )
    if not columns:
        raise ValueError("clustering requires M/S/H feature columns")
    required = _validate_coverage(
        min_feature_coverage, name="min_feature_coverage"
    )
    numeric = clean_features.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    coverage = np.mean(np.isfinite(numeric.to_numpy(dtype=float)), axis=1)
    eligible_mask = coverage >= required
    eligible_features = clean_features.loc[eligible_mask].copy()
    numeric = numeric.loc[eligible_mask].copy()
    if n_clusters is None:
        n_clusters = label_independent_cluster_count(
            len(eligible_features), rule=cluster_count_rule
        )
    if len(eligible_features) <= n_clusters:
        raise ValueError("clustering requires more eligible models than clusters")
    active = numeric.columns[np.isfinite(numeric.to_numpy(dtype=float)).any(axis=0)]
    if len(active) == 0:
        raise ValueError("clustering has no finite feature columns")
    matrix = numeric.loc[:, active].to_numpy(dtype=float)
    medians = np.nanmedian(matrix, axis=0)
    missing_rows, missing_columns = np.where(~np.isfinite(matrix))
    matrix[missing_rows, missing_columns] = medians[missing_columns]
    standardized = StandardScaler().fit_transform(matrix)

    # No label table is read above this line.
    cluster = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20,
    ).fit_predict(standardized)
    assignments = pd.DataFrame(
        {
            "model_id": eligible_features["model_id"].to_numpy(),
            "cluster": cluster.astype(int),
        }
    ).sort_values("model_id", kind="stable", ignore_index=True)

    clean_labels = _canonical_manifest(labels)
    confirmed_labels = clean_labels.loc[
        clean_labels["_confirmed"], ["model_id", "root_family"]
    ]
    scored = assignments.merge(
        confirmed_labels,
        on="model_id",
        how="inner",
        validate="one_to_one",
    )
    ari = (
        float(adjusted_rand_score(scored["root_family"], scored["cluster"]))
        if len(scored) >= 2
        else float("nan")
    )
    nmi = (
        float(normalized_mutual_info_score(scored["root_family"], scored["cluster"]))
        if len(scored) >= 2
        else float("nan")
    )
    silhouette = (
        float(silhouette_score(standardized, cluster))
        if 1 < np.unique(cluster).size < len(cluster)
        else float("nan")
    )
    metrics = pd.DataFrame(
        [
            {
                "method": "combined_unsupervised_kmeans",
                "n_clusters": n_clusters,
                "n_clustered": len(assignments),
                "n_labeled": len(scored),
                "ari": ari,
                "nmi": nmi,
                "root_macro_purity": _root_macro_purity(scored),
                "silhouette": silhouette,
            }
        ]
    )
    exclusion_rows = [
        _exclusion(
            stage="clustering",
            root_family=None,
            model_id=str(model_id),
            reason="insufficient_feature_coverage",
            detail=f"combined coverage is below {required:g}",
        )
        for model_id in clean_features.loc[~eligible_mask, "model_id"]
    ]
    return {
        "assignments": assignments,
        "metrics": metrics,
        "exclusions": pd.DataFrame(exclusion_rows, columns=_EXCLUSION_COLUMNS),
    }


def evaluate_exact_parent_diagnostic(
    graph_edges: pd.DataFrame,
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    *,
    min_feature_coverage: float = 0.8,
    min_pair_coverage: float = 0.8,
) -> dict[str, pd.DataFrame]:
    """Rank unambiguous graph parents against same-family hard negatives."""

    clean_manifest = _canonical_manifest(manifest)
    clean_features = _canonical_features(features)
    columns = tuple(
        column
        for column in clean_features.columns
        if column.startswith(_REQUIRED_BLOCK_PREFIXES)
    )
    if not columns:
        raise ValueError("exact-parent diagnostic requires M/S/H feature columns")
    eligible_ids = _feature_complete_ids(
        clean_features,
        min_feature_coverage=min_feature_coverage,
    )
    confirmed = clean_manifest.loc[
        clean_manifest["_confirmed"] & clean_manifest["model_id"].isin(eligible_ids)
    ].copy()
    manifest_index = confirmed.set_index("model_id", verify_integrity=True)
    feature_index = clean_features.set_index("model_id", verify_integrity=True)
    edges = normalize_graph_edges(graph_edges)
    parent_sets = edges.groupby("child_id", sort=True)["parent_id"].agg(
        lambda values: tuple(sorted(set(values)))
    )
    ambiguous_children = {
        str(child_id)
        for child_id, parents in parent_sets.items()
        if len(parents) != 1
    }
    pair_coverage = _validate_coverage(
        min_pair_coverage, name="min_pair_coverage"
    )

    candidates: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    for child_id, parents in parent_sets.items():
        child_id = str(child_id)
        if len(parents) != 1:
            exclusions.append(
                _exclusion(
                    stage="exact_parent",
                    root_family=(
                        str(manifest_index.at[child_id, "root_family"])
                        if child_id in manifest_index.index
                        else None
                    ),
                    model_id=child_id,
                    reason="ambiguous_multiple_graph_parents",
                    detail=f"graph contains {len(parents)} distinct parents",
                )
            )
            continue
        parent_id = str(parents[0])
        if child_id not in manifest_index.index or parent_id not in manifest_index.index:
            exclusions.append(
                _exclusion(
                    stage="exact_parent",
                    root_family=None,
                    model_id=child_id,
                    reason="child_or_true_parent_ineligible",
                    detail="both graph endpoints require confirmed labels and features",
                )
            )
            continue
        family = str(manifest_index.at[child_id, "root_family"])
        if str(manifest_index.at[parent_id, "root_family"]) != family:
            exclusions.append(
                _exclusion(
                    stage="exact_parent",
                    root_family=family,
                    model_id=child_id,
                    reason="true_parent_outside_audited_family",
                    detail=parent_id,
                )
            )
            continue
        family_ids = sorted(
            manifest_index.loc[manifest_index["root_family"].eq(family)].index.astype(str)
        )
        hard_negatives = [
            model_id
            for model_id in family_ids
            if model_id not in {child_id, parent_id}
            and model_id not in ambiguous_children
        ]
        if len(hard_negatives) < 2:
            exclusions.append(
                _exclusion(
                    stage="exact_parent",
                    root_family=family,
                    model_id=child_id,
                    reason="insufficient_same_family_hard_negatives",
                    detail=f"found {len(hard_negatives)}; require at least 2",
                )
            )
            continue

        candidate_ids = [parent_id, *hard_negatives]
        scaler = _fit_gallery_scaler_from_canonical(
            clean_features,
            gallery_model_ids=candidate_ids,
            columns=columns,
        )
        local: list[dict[str, object]] = []
        for candidate_id in candidate_ids:
            similarity = shared_feature_similarity(
                feature_index.loc[child_id],
                feature_index.loc[candidate_id],
                scaler,
                min_coverage=pair_coverage,
            )
            local.append(
                {
                    "child_id": child_id,
                    "root_family": family,
                    "true_parent_id": parent_id,
                    "candidate_id": candidate_id,
                    "is_true_parent": candidate_id == parent_id,
                    "score": similarity.score,
                    "coverage": similarity.coverage,
                    "rank": pd.NA,
                    "claim_scope": "sensitivity_only",
                    "edge_provenance": "graph_export_unverified_direct_parent",
                    "estimator_provenance": "candidate_gallery_scaling_excludes_query_child",
                }
            )
        true_row = next(row for row in local if bool(row["is_true_parent"]))
        if not np.isfinite(float(true_row["score"])):
            exclusions.append(
                _exclusion(
                    stage="exact_parent",
                    root_family=family,
                    model_id=child_id,
                    reason="true_parent_pair_below_coverage",
                    detail=f"shared feature coverage is below {pair_coverage:g}",
                )
            )
            continue
        scorable_negatives = [
            row
            for row in local
            if not bool(row["is_true_parent"]) and np.isfinite(float(row["score"]))
        ]
        if len(scorable_negatives) < 2:
            exclusions.append(
                _exclusion(
                    stage="exact_parent",
                    root_family=family,
                    model_id=child_id,
                    reason="insufficient_scorable_same_family_hard_negatives",
                    detail=f"found {len(scorable_negatives)}; require at least 2",
                )
            )
            continue
        local = [true_row, *scorable_negatives]
        order = sorted(
            (
                index
                for index, row in enumerate(local)
                if np.isfinite(float(row["score"]))
            ),
            key=lambda index: (-float(local[index]["score"]), local[index]["candidate_id"]),
        )
        for rank, index in enumerate(order, start=1):
            local[index]["rank"] = rank
        true_rank = int(true_row["rank"])
        predictions.append(
            {
                "child_id": child_id,
                "root_family": family,
                "true_parent_id": parent_id,
                "candidate_count": len(local),
                "true_parent_rank": true_rank,
                "reciprocal_rank": 1.0 / true_rank,
                "top1_correct": true_rank == 1,
                "claim_scope": "sensitivity_only",
                "edge_provenance": "graph_export_unverified_direct_parent",
                "estimator_provenance": "candidate_gallery_scaling_excludes_query_child",
            }
        )
        candidates.extend(local)

    prediction_frame = pd.DataFrame(predictions).sort_values(
        "child_id", kind="stable", ignore_index=True
    ) if predictions else pd.DataFrame(
        columns=[
            "child_id",
            "root_family",
            "true_parent_id",
            "candidate_count",
            "true_parent_rank",
            "reciprocal_rank",
            "top1_correct",
            "claim_scope",
            "edge_provenance",
            "estimator_provenance",
        ]
    )
    family_metrics = (
        prediction_frame.groupby("root_family", sort=True).agg(
            top1_accuracy=("top1_correct", "mean"),
            mrr=("reciprocal_rank", "mean"),
        )
        if not prediction_frame.empty
        else pd.DataFrame(columns=["top1_accuracy", "mrr"])
    )
    metrics = pd.DataFrame(
        [
            {
                "n_children": len(prediction_frame),
                "top1_accuracy": (
                    float(prediction_frame["top1_correct"].mean())
                    if not prediction_frame.empty
                    else float("nan")
                ),
                "mrr": (
                    float(prediction_frame["reciprocal_rank"].mean())
                    if not prediction_frame.empty
                    else float("nan")
                ),
                "root_macro_top1": (
                    float(family_metrics["top1_accuracy"].mean())
                    if not family_metrics.empty
                    else float("nan")
                ),
                "root_macro_mrr": (
                    float(family_metrics["mrr"].mean())
                    if not family_metrics.empty
                    else float("nan")
                ),
                "claim_scope": "sensitivity_only",
                "edge_provenance": "graph_export_unverified_direct_parent",
                "estimator_provenance": "candidate_gallery_scaling_excludes_query_child",
            }
        ]
    )
    return {
        "candidates": pd.DataFrame(candidates),
        "predictions": prediction_frame,
        "metrics": metrics,
        "exclusions": pd.DataFrame(exclusions, columns=_EXCLUSION_COLUMNS),
    }


__all__ = [
    "COMBINED_METHOD",
    "CONFIG_METHOD",
    "GALLERY_SCALED_PROVENANCE",
    "GalleryScaler",
    "LINEAGE_METHODS",
    "PAPER_STYLE_ALPHA_DKS",
    "PAPER_STYLE_ALPHA_DKS_PROVENANCE",
    "SPECTRAL_SHAPE_METHOD",
    "SimilarityResult",
    "UnknownFamilyFold",
    "bootstrap_closed_set_metrics",
    "build_root_gallery",
    "closed_set_metric_table",
    "closed_set_metrics",
    "cluster_lineage_diagnostics",
    "compute_open_set_metrics",
    "evaluate_closed_set_lineage",
    "evaluate_exact_parent_diagnostic",
    "evaluate_unknown_family_rejection",
    "fit_gallery_scaler",
    "paper_style_alpha_similarity",
    "rank_confirmed_queries",
    "root_macro_top1",
    "shared_feature_similarity",
    "unknown_family_folds",
]
