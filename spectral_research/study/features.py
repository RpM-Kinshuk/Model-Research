from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from spectral_research.study.contracts import canonical_model_id


FEATURE_SCHEMA_VERSION = "htsr-study-features-v2"
ALPHA_ESTIMATOR = "weightwatcher_dks_power_law"
ALPHA_ESTIMATOR_VERSION = "stored_artifact_contract_v1"
CANONICAL_ROLES = (
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_o",
    "mlp_gate",
    "mlp_up",
    "mlp_down",
)
FEATURE_BLOCKS = ("M", "S", "H")
TRAJECTORY_STATISTICS = ("median", "iqr", "slope", "missing_fraction")
DKS_SENSITIVITY_MAX = 0.1

_BLOCK_QUANTITIES: Mapping[str, tuple[str, ...]] = {
    "M": ("matrix_rows", "matrix_cols"),
    "S": (
        "log_rms",
        "log_spectral_norm",
        "normalized_entropy",
        "stable_rank_fraction",
    ),
    "H": ("alpha", "alpha_weighted", "log_alpha_norm"),
}
_GLOBAL_M_FEATURES = (
    "M__layer_count",
    "M__canonical_role_count",
    "M__canonical_role_fraction",
    "M__log10_canonical_matrix_parameter_count",
)

_ROLE_PATTERNS: Mapping[str, tuple[re.Pattern[str], ...]] = {
    "attn_q": (
        re.compile(r"(?:^|\.)(?:self_attn|self_attention|attention|attn)\.q_proj$"),
        re.compile(r"(?:^|\.)attention\.self\.query$"),
        re.compile(r"(?:^|\.)attention\.wq$"),
    ),
    "attn_k": (
        re.compile(r"(?:^|\.)(?:self_attn|self_attention|attention|attn)\.k_proj$"),
        re.compile(r"(?:^|\.)attention\.self\.key$"),
        re.compile(r"(?:^|\.)attention\.wk$"),
    ),
    "attn_v": (
        re.compile(r"(?:^|\.)(?:self_attn|self_attention|attention|attn)\.v_proj$"),
        re.compile(r"(?:^|\.)attention\.self\.value$"),
        re.compile(r"(?:^|\.)attention\.wv$"),
    ),
    "attn_o": (
        re.compile(r"(?:^|\.)(?:self_attn|self_attention|attention|attn)\.(?:o_proj|out_proj)$"),
        re.compile(r"(?:^|\.)attention\.output\.dense$"),
        re.compile(r"(?:^|\.)attention\.(?:wo|dense)$"),
        re.compile(r"(?:^|\.)attn\.c_proj$"),
    ),
    "mlp_gate": (
        re.compile(r"(?:^|\.)mlp\.gate_proj$"),
        re.compile(r"(?:^|\.)feed_forward\.w1$"),
    ),
    "mlp_up": (
        re.compile(r"(?:^|\.)mlp\.(?:up_proj|fc1|c_fc|dense_h_to_4h)$"),
        re.compile(r"(?:^|\.)feed_forward\.w3$"),
        re.compile(r"(?:^|\.)intermediate\.dense$"),
    ),
    "mlp_down": (
        re.compile(r"(?:^|\.)mlp\.(?:down_proj|fc2|c_proj|dense_4h_to_h)$"),
        re.compile(r"(?:^|\.)feed_forward\.w2$"),
        re.compile(r"(?:^|\.)output\.dense$"),
    ),
}
_NON_CONFIRMATORY_MARKERS = (
    "cross_attn",
    "cross_attention",
    "encoder_attn",
    "qkv",
    "query_key_value",
    "c_attn",
    "wqkv",
    "gate_up",
    "block_sparse_moe",
    ".moe.",
    ".experts.",
    ".expert.",
    "switch_mlp",
    "mamba",
    ".ssm.",
    "recurrent",
)
_LAYER_PATTERN = re.compile(
    r"(?:^|\.)(?:layers|layer|h|block)\.(\d+)(?:\.|$)", re.IGNORECASE
)


def _as_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if np.isfinite(numeric) else float("nan")


def _positive_float(value: object) -> float:
    numeric = _as_float(value)
    return numeric if numeric > 0 else float("nan")


def _nonnegative_float(value: object) -> float:
    numeric = _as_float(value)
    return numeric if numeric >= 0 else float("nan")


def clean_alpha(alpha: object) -> float:
    """Return a finite positive alpha, masking the stored failure sentinel."""

    return _positive_float(alpha)


def _is_non_confirmatory(longname: object) -> bool:
    text = str(longname).strip().lower() if longname is not None else ""
    return any(marker in text for marker in _NON_CONFIRMATORY_MARKERS)


def canonical_role(longname: object) -> str | None:
    """Map an unfused transformer matrix name to one of seven study roles."""

    if longname is None or pd.isna(longname):
        return None
    text = str(longname).strip().lower()
    if not text or _is_non_confirmatory(text):
        return None
    for role in CANONICAL_ROLES:
        if any(pattern.search(text) for pattern in _ROLE_PATTERNS[role]):
            return role
    return None


def _layer_index(longname: object) -> float:
    if longname is None or pd.isna(longname):
        return float("nan")
    match = _LAYER_PATTERN.search(str(longname).strip())
    return float(match.group(1)) if match else float("nan")


def layer_quantities(
    *,
    alpha: object,
    dks_distance: object,
    frobenius_sq: object,
    spectral_sq: object,
    matrix_rows: object,
    matrix_cols: object,
    entropy: object,
    rank: object,
    log_alpha_norm: object = np.nan,
) -> dict[str, float]:
    """Recompute layer quantities from the saved squared-norm contract.

    The artifact's ``entropy`` field is already normalized by log(rank). Values
    above one are preserved and surfaced through model QC rather than clipped.
    """

    rows = _positive_float(matrix_rows)
    cols = _positive_float(matrix_cols)
    frobenius = _positive_float(frobenius_sq)
    spectral = _positive_float(spectral_sq)
    cleaned_alpha = clean_alpha(alpha)
    matrix_rank = _positive_float(rank)

    log_rms = float("nan")
    if np.isfinite(frobenius) and np.isfinite(rows) and np.isfinite(cols):
        log_rms = 0.5 * (np.log10(frobenius) - np.log10(rows * cols))

    log_spectral = np.log10(spectral) if np.isfinite(spectral) else float("nan")
    stable_rank_fraction = float("nan")
    if (
        np.isfinite(frobenius)
        and np.isfinite(spectral)
        and np.isfinite(matrix_rank)
    ):
        stable_rank_fraction = (frobenius / spectral) / matrix_rank

    normalized_entropy = _nonnegative_float(entropy)
    if not np.isfinite(matrix_rank) or matrix_rank <= 1:
        normalized_entropy = float("nan")

    alpha_weighted = float("nan")
    if np.isfinite(cleaned_alpha) and np.isfinite(log_spectral):
        alpha_weighted = cleaned_alpha * log_spectral

    cleaned_log_alpha_norm = _as_float(log_alpha_norm)
    if not np.isfinite(cleaned_alpha):
        cleaned_log_alpha_norm = float("nan")

    return {
        "matrix_rows": rows,
        "matrix_cols": cols,
        "log_rms": log_rms,
        "log_spectral_norm": log_spectral,
        "normalized_entropy": normalized_entropy,
        "stable_rank_fraction": stable_rank_fraction,
        "alpha": cleaned_alpha,
        "alpha_weighted": alpha_weighted,
        "log_alpha_norm": cleaned_log_alpha_norm,
        "dks_distance": _nonnegative_float(dks_distance),
        "matrix_rank": matrix_rank,
    }


def summarize_trajectory(
    values: Iterable[object], depths: Iterable[object]
) -> dict[str, float]:
    """Summarize a trajectory after scaling the model's full depth to [0, 1]."""

    value_array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(
        dtype=float
    )
    depth_array = pd.to_numeric(pd.Series(list(depths)), errors="coerce").to_numpy(
        dtype=float
    )
    if value_array.shape != depth_array.shape:
        raise ValueError("values and depths must have the same length")
    if value_array.size == 0:
        return {
            "median": float("nan"),
            "iqr": float("nan"),
            "slope": float("nan"),
            "missing_fraction": 1.0,
        }

    finite_depth = np.isfinite(depth_array)
    valid = np.isfinite(value_array) & finite_depth
    finite_values = value_array[valid]
    missing_fraction = 1.0 - float(np.count_nonzero(valid)) / float(value_array.size)
    if finite_values.size == 0:
        return {
            "median": float("nan"),
            "iqr": float("nan"),
            "slope": float("nan"),
            "missing_fraction": missing_fraction,
        }

    q25, q75 = np.percentile(finite_values, [25.0, 75.0])
    slope = float("nan")
    all_finite_depths = depth_array[finite_depth]
    depth_min = float(np.min(all_finite_depths))
    depth_max = float(np.max(all_finite_depths))
    if finite_values.size >= 2 and depth_max > depth_min:
        normalized_depth = (depth_array[valid] - depth_min) / (depth_max - depth_min)
        if np.unique(normalized_depth).size >= 2:
            design = np.column_stack([normalized_depth, np.ones_like(normalized_depth)])
            slope = float(np.linalg.lstsq(design, finite_values, rcond=None)[0][0])

    return {
        "median": float(np.median(finite_values)),
        "iqr": float(q75 - q25),
        "slope": slope,
        "missing_fraction": missing_fraction,
    }


def feature_columns(blocks: Sequence[str] = FEATURE_BLOCKS) -> tuple[str, ...]:
    """Return feature names in schema order, independent of caller block order."""

    requested = set(blocks)
    unknown = requested.difference(FEATURE_BLOCKS)
    if unknown:
        raise ValueError(f"unknown feature blocks: {sorted(unknown)}")

    columns: list[str] = []
    for block in FEATURE_BLOCKS:
        if block not in requested:
            continue
        if block == "M":
            columns.extend(_GLOBAL_M_FEATURES)
        for role in CANONICAL_ROLES:
            if block == "M":
                columns.extend((f"M__{role}__present", f"M__{role}__coverage"))
            for quantity in _BLOCK_QUANTITIES[block]:
                columns.extend(
                    f"{block}__{role}__{quantity}__{statistic}"
                    for statistic in TRAJECTORY_STATISTICS
                )
    return tuple(columns)


_SCHEMA_EXCLUSION_COLUMNS = ("model_id", "reason", "detail")


def confirmatory_schema_population(
    features: pd.DataFrame,
    min_role_coverage: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select complete unfused seven-role rows under the current feature schema."""

    threshold = _as_float(min_role_coverage)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("min_role_coverage must be finite and between 0 and 1")

    present_columns = [f"M__{role}__present" for role in CANONICAL_ROLES]
    coverage_columns = [f"M__{role}__coverage" for role in CANONICAL_ROLES]
    required = {
        "model_id",
        "feature_schema_version",
        "M__canonical_role_count",
        "QC__canonical_schema_complete",
        "QC__fused_role_row_count",
        *present_columns,
        *coverage_columns,
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise KeyError(f"Missing confirmatory schema columns: {missing}")

    versions = features["feature_schema_version"].astype("string").str.strip()
    schema_ok = versions.eq(FEATURE_SCHEMA_VERSION).fillna(False)
    role_counts = pd.to_numeric(
        features["M__canonical_role_count"], errors="coerce"
    )
    schema_complete = features["QC__canonical_schema_complete"].eq(True).fillna(
        False
    )
    role_present = features.loc[:, present_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    roles_ok = (
        role_counts.eq(len(CANONICAL_ROLES))
        & schema_complete
        & role_present.eq(1.0).all(axis=1)
    )
    fused_counts = pd.to_numeric(
        features["QC__fused_role_row_count"], errors="coerce"
    )
    unfused = fused_counts.eq(0)
    role_coverage = features.loc[:, coverage_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    coverage_ok = role_coverage.ge(threshold).all(axis=1)
    eligible_mask = schema_ok & roles_ok & unfused & coverage_ok

    exclusions: list[dict[str, str]] = []
    for index, row in features.loc[~eligible_mask].iterrows():
        model_value = row["model_id"]
        model_id = "" if pd.isna(model_value) else str(model_value).strip()
        if not bool(schema_ok.loc[index]):
            reason = "incompatible_feature_schema"
            detail = (
                f"feature_schema_version={row['feature_schema_version']!r}; "
                f"expected {FEATURE_SCHEMA_VERSION!r}"
            )
        elif not bool(roles_ok.loc[index]):
            missing_roles = [
                role
                for role, column in zip(CANONICAL_ROLES, present_columns)
                if role_present.loc[index, column] != 1.0
            ]
            detail = (
                f"requires all {len(CANONICAL_ROLES)} canonical roles; "
                f"observed_count={role_counts.loc[index]!r}; "
                f"missing_roles={missing_roles}"
            )
            reason = "incomplete_canonical_roles"
        elif not bool(unfused.loc[index]):
            reason = "non_confirmatory_role_rows"
            detail = (
                "QC__fused_role_row_count must be zero; "
                f"found {fused_counts.loc[index]!r}"
            )
        else:
            below = [
                f"{role}={role_coverage.loc[index, column]!r}"
                for role, column in zip(CANONICAL_ROLES, coverage_columns)
                if not bool(role_coverage.loc[index, column] >= threshold)
            ]
            reason = "insufficient_role_coverage"
            detail = (
                f"per-role coverage must be >= {threshold:g}; below threshold: "
                + ", ".join(below)
            )
        exclusions.append(
            {"model_id": model_id, "reason": reason, "detail": detail}
        )

    eligible = features.loc[eligible_mask].copy().reset_index(drop=True)
    exclusion_frame = pd.DataFrame(exclusions, columns=_SCHEMA_EXCLUSION_COLUMNS)
    return eligible, exclusion_frame


def load_stats_csv(
    path: str | Path, *, model_id: str, chunksize: int = 50_000
) -> pd.DataFrame:
    """Load only rows for an explicit canonical ID; filenames are never identity."""

    requested_id = str(model_id).strip()
    if not requested_id:
        raise ValueError("model_id must be a non-empty explicit identifier")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")

    selected: list[pd.DataFrame] = []
    for chunk in pd.read_csv(Path(path), chunksize=chunksize, low_memory=False):
        if "model_id" not in chunk.columns:
            raise ValueError(f"stats CSV is missing required model_id column: {path}")
        identity = chunk["model_id"].astype("string").str.strip()
        matching = chunk.loc[identity.eq(requested_id)].copy()
        if not matching.empty:
            matching.loc[:, "model_id"] = requested_id
            selected.append(matching)
    if not selected:
        raise ValueError(f"model_id {requested_id!r} not found in stats CSV: {path}")
    return pd.concat(selected, ignore_index=True)


_ARTIFACT_EXCLUSION_COLUMNS = ("model_id", "path", "reason", "detail")


def index_stats_directory(
    stats_dir: str | Path,
    *,
    workers: int = 16,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Index CSV artifacts by their internal canonical identifier."""

    root = Path(stats_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if workers <= 0:
        raise ValueError("workers must be positive")
    rows: list[dict[str, str]] = []
    exclusions: list[dict[str, str]] = []

    def read_identity(path: Path) -> tuple[Path, str | None, Exception | None]:
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or "model_id" not in reader.fieldnames:
                    raise ValueError("missing model_id column")
                first = next(reader, None)
            if first is None:
                return path, None, EOFError("artifact has a header but no rows")
            return path, canonical_model_id(first.get("model_id")), None
        except (OSError, UnicodeError, ValueError, csv.Error) as error:
            return path, None, error

    paths = sorted(root.glob("*.csv"))
    with ThreadPoolExecutor(max_workers=min(workers, max(len(paths), 1))) as executor:
        identities = executor.map(read_identity, paths)
        for path, model_id, error in identities:
            if error is not None:
                reason = (
                    "empty_stats_artifact"
                    if isinstance(error, EOFError)
                    else "invalid_stats_header"
                )
                exclusions.append(
                    {
                        "model_id": "",
                        "path": str(path),
                        "reason": reason,
                        "detail": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            if model_id is None:
                exclusions.append(
                    {
                        "model_id": "",
                        "path": str(path),
                        "reason": "missing_stats_model_id",
                        "detail": "first artifact row has no canonical model_id",
                    }
                )
                continue
            rows.append({"model_id": model_id, "path": str(path)})

    index = pd.DataFrame(rows, columns=["model_id", "path"])
    if not index.empty:
        duplicate = index["model_id"].duplicated(keep=False)
        if duplicate.any():
            model_ids = sorted(index.loc[duplicate, "model_id"].unique())
            raise ValueError(f"duplicate stats artifacts for model_id: {model_ids}")
        index = index.sort_values("model_id").reset_index(drop=True)
    exclusion_frame = pd.DataFrame(exclusions, columns=_ARTIFACT_EXCLUSION_COLUMNS)
    return index, exclusion_frame


def extract_feature_table(
    stats_index: pd.DataFrame,
    model_ids: Iterable[object],
    *,
    workers: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract deterministic model summaries and record unavailable artifacts."""

    required = {"model_id", "path"}
    missing_columns = sorted(required - set(stats_index.columns))
    if missing_columns:
        raise ValueError(f"stats index is missing columns: {missing_columns}")
    if stats_index["model_id"].duplicated().any():
        raise ValueError("stats index contains duplicate model_id values")
    if workers <= 0:
        raise ValueError("workers must be positive")
    paths = stats_index.set_index("model_id")["path"].astype(str).to_dict()

    requested: list[str] = []
    for position, value in enumerate(model_ids):
        model_id = canonical_model_id(value)
        if model_id is None:
            raise ValueError(f"model_ids contains a missing value at position {position}")
        requested.append(model_id)
    if len(requested) != len(set(requested)):
        raise ValueError("model_ids contains duplicate canonical identifiers")

    def process_model(
        model_id: str,
    ) -> tuple[dict[str, object] | None, dict[str, str] | None]:
        path_text = paths.get(model_id)
        if path_text is None:
            return None, {
                    "model_id": model_id,
                    "path": "",
                    "reason": "missing_stats_artifact",
                    "detail": "no indexed CSV contains this model_id",
                }
        path = Path(path_text)
        try:
            stats = pd.read_csv(path, low_memory=False)
            if "model_id" in stats.columns:
                stats["model_id"] = stats["model_id"].map(canonical_model_id)
            return summarize_model_features(stats, model_id=model_id), None
        except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
            return None, {
                    "model_id": model_id,
                    "path": str(path),
                    "reason": "invalid_stats_artifact",
                    "detail": f"{type(error).__name__}: {error}",
                }

    records: list[dict[str, object]] = []
    exclusions: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=min(workers, max(len(requested), 1))) as executor:
        for record, exclusion in executor.map(process_model, requested):
            if record is not None:
                records.append(record)
            if exclusion is not None:
                exclusions.append(exclusion)
    features = pd.DataFrame(records)
    if not features.empty:
        features = features.sort_values("model_id").reset_index(drop=True)
    exclusion_frame = pd.DataFrame(exclusions, columns=_ARTIFACT_EXCLUSION_COLUMNS)
    return features, exclusion_frame


def _alpha_trajectory_from_stats(stats: pd.DataFrame, model_id: str) -> pd.DataFrame:
    working = _model_rows(stats, model_id)
    working["role"] = working["longname"].map(canonical_role)
    working["depth"] = working["longname"].map(_layer_index)
    working["alpha"] = working.get(
        "alpha", pd.Series(index=working.index, dtype=float)
    ).map(clean_alpha)
    working = working.loc[
        working["role"].notna() & np.isfinite(working["depth"]),
        ["role", "depth", "alpha"],
    ].copy()
    if working.empty:
        raise ValueError("no canonical alpha trajectory rows")
    grouped = working.groupby(
        ["role", "depth"], as_index=False, sort=False
    )["alpha"].median()
    grouped.insert(0, "model_id", model_id)
    grouped["alpha_estimator"] = ALPHA_ESTIMATOR
    grouped["alpha_estimator_version"] = ALPHA_ESTIMATOR_VERSION
    grouped["_role_order"] = grouped["role"].map(
        {role: index for index, role in enumerate(CANONICAL_ROLES)}
    )
    return (
        grouped.sort_values(["_role_order", "depth"])
        .drop(columns="_role_order")
        .reset_index(drop=True)
    )


def _study_table_worker(
    task: tuple[str, str],
) -> tuple[
    str,
    dict[str, object] | None,
    pd.DataFrame | None,
    dict[str, str] | None,
    dict[str, str] | None,
]:
    model_id, path_text = task
    path = Path(path_text)
    try:
        stats = pd.read_csv(path, low_memory=False)
        if "model_id" in stats.columns:
            stats["model_id"] = stats["model_id"].map(canonical_model_id)
    except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        detail = f"{type(error).__name__}: {error}"
        return (
            model_id,
            None,
            None,
            {
                "model_id": model_id,
                "path": str(path),
                "reason": "invalid_stats_artifact",
                "detail": detail,
            },
            {
                "model_id": model_id,
                "path": str(path),
                "reason": "invalid_alpha_trajectory",
                "detail": detail,
            },
        )

    feature: dict[str, object] | None = None
    feature_exclusion: dict[str, str] | None = None
    trajectory: pd.DataFrame | None = None
    trajectory_exclusion: dict[str, str] | None = None
    try:
        feature = summarize_model_features(stats, model_id=model_id)
    except ValueError as error:
        feature_exclusion = {
            "model_id": model_id,
            "path": str(path),
            "reason": "invalid_stats_artifact",
            "detail": f"{type(error).__name__}: {error}",
        }
    try:
        trajectory = _alpha_trajectory_from_stats(stats, model_id)
    except ValueError as error:
        trajectory_exclusion = {
            "model_id": model_id,
            "path": str(path),
            "reason": "invalid_alpha_trajectory",
            "detail": f"{type(error).__name__}: {error}",
        }
    return (
        model_id,
        feature,
        trajectory,
        feature_exclusion,
        trajectory_exclusion,
    )


def extract_study_tables(
    stats_index: pd.DataFrame,
    model_ids: Iterable[object],
    *,
    workers: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract feature summaries and alpha trajectories with one read per model."""

    required = {"model_id", "path"}
    missing_columns = sorted(required - set(stats_index.columns))
    if missing_columns:
        raise ValueError(f"stats index is missing columns: {missing_columns}")
    if stats_index["model_id"].duplicated().any():
        raise ValueError("stats index contains duplicate model_id values")
    if workers <= 0:
        raise ValueError("workers must be positive")
    paths = stats_index.set_index("model_id")["path"].astype(str).to_dict()

    requested: list[str] = []
    for position, value in enumerate(model_ids):
        model_id = canonical_model_id(value)
        if model_id is None:
            raise ValueError(f"model_ids contains a missing value at position {position}")
        requested.append(model_id)
    if len(requested) != len(set(requested)):
        raise ValueError("model_ids contains duplicate canonical identifiers")

    tasks = [(model_id, paths[model_id]) for model_id in requested if model_id in paths]
    if workers == 1:
        extracted = list(map(_study_table_worker, tasks))
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, max(len(tasks), 1))
        ) as executor:
            extracted = list(executor.map(_study_table_worker, tasks))
    by_id = {result[0]: result[1:] for result in extracted}

    feature_records: list[dict[str, object]] = []
    trajectory_frames: list[pd.DataFrame] = []
    feature_exclusions: list[dict[str, str]] = []
    trajectory_exclusions: list[dict[str, str]] = []
    for model_id in requested:
        if model_id not in paths:
            feature_exclusions.append(
                {
                    "model_id": model_id,
                    "path": "",
                    "reason": "missing_stats_artifact",
                    "detail": "no indexed CSV contains this model_id",
                }
            )
            trajectory_exclusions.append(
                {
                    "model_id": model_id,
                    "path": "",
                    "reason": "missing_stats_artifact",
                    "detail": "no indexed CSV contains this model_id",
                }
            )
            continue
        feature, trajectory, feature_exclusion, trajectory_exclusion = by_id[model_id]
        if feature is not None:
            feature_records.append(feature)
        if trajectory is not None:
            trajectory_frames.append(trajectory)
        if feature_exclusion is not None:
            feature_exclusions.append(feature_exclusion)
        if trajectory_exclusion is not None:
            trajectory_exclusions.append(trajectory_exclusion)

    features = pd.DataFrame(feature_records)
    if not features.empty:
        features = features.sort_values("model_id").reset_index(drop=True)
    trajectories = (
        pd.concat(trajectory_frames, ignore_index=True)
        if trajectory_frames
        else pd.DataFrame(
            columns=[
                "model_id",
                "role",
                "depth",
                "alpha",
                "alpha_estimator",
                "alpha_estimator_version",
            ]
        )
    )
    return (
        features,
        trajectories,
        pd.DataFrame(feature_exclusions, columns=_ARTIFACT_EXCLUSION_COLUMNS),
        pd.DataFrame(trajectory_exclusions, columns=_ARTIFACT_EXCLUSION_COLUMNS),
    )


def extract_alpha_trajectories(
    stats_index: pd.DataFrame,
    model_ids: Iterable[object],
    *,
    workers: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract clean seven-role alpha trajectories for lineage comparisons."""

    required = {"model_id", "path"}
    missing_columns = sorted(required - set(stats_index.columns))
    if missing_columns:
        raise ValueError(f"stats index is missing columns: {missing_columns}")
    if stats_index["model_id"].duplicated().any():
        raise ValueError("stats index contains duplicate model_id values")
    if workers <= 0:
        raise ValueError("workers must be positive")
    paths = stats_index.set_index("model_id")["path"].astype(str).to_dict()

    requested: list[str] = []
    for position, value in enumerate(model_ids):
        model_id = canonical_model_id(value)
        if model_id is None:
            raise ValueError(f"model_ids contains a missing value at position {position}")
        requested.append(model_id)
    if len(requested) != len(set(requested)):
        raise ValueError("model_ids contains duplicate canonical identifiers")

    def process_model(
        model_id: str,
    ) -> tuple[pd.DataFrame | None, dict[str, str] | None]:
        path_text = paths.get(model_id)
        if path_text is None:
            return None, {
                "model_id": model_id,
                "path": "",
                "reason": "missing_stats_artifact",
                "detail": "no indexed CSV contains this model_id",
            }
        path = Path(path_text)
        try:
            stats = pd.read_csv(path, low_memory=False)
            if "model_id" in stats.columns:
                stats["model_id"] = stats["model_id"].map(canonical_model_id)
            return _alpha_trajectory_from_stats(stats, model_id), None
        except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
            return None, {
                "model_id": model_id,
                "path": str(path),
                "reason": "invalid_alpha_trajectory",
                "detail": f"{type(error).__name__}: {error}",
            }

    frames: list[pd.DataFrame] = []
    exclusions: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=min(workers, max(len(requested), 1))) as executor:
        for frame, exclusion in executor.map(process_model, requested):
            if frame is not None:
                frames.append(frame)
            if exclusion is not None:
                exclusions.append(exclusion)
    trajectories = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=[
                "model_id",
                "role",
                "depth",
                "alpha",
                "alpha_estimator",
                "alpha_estimator_version",
            ]
        )
    )
    exclusion_frame = pd.DataFrame(exclusions, columns=_ARTIFACT_EXCLUSION_COLUMNS)
    return trajectories, exclusion_frame


def _model_rows(stats: pd.DataFrame, model_id: str) -> pd.DataFrame:
    if "model_id" not in stats.columns:
        raise ValueError("stats frame is missing required model_id column")
    if "longname" not in stats.columns:
        raise ValueError("stats frame is missing required longname column")
    requested_id = str(model_id).strip()
    if not requested_id:
        raise ValueError("model_id must be a non-empty explicit identifier")
    identity = stats["model_id"].astype("string").str.strip()
    selected = stats.loc[identity.eq(requested_id)].copy()
    if selected.empty:
        raise ValueError(f"model_id {requested_id!r} not found in stats frame")
    selected.loc[:, "model_id"] = requested_id
    return selected


def _column_value(row: pd.Series, column: str) -> object:
    return row[column] if column in row.index else np.nan


def summarize_model_features(
    stats: pd.DataFrame, *, model_id: str
) -> dict[str, object]:
    """Build one deterministic M/S/H record plus non-predictive QC fields."""

    working = _model_rows(stats, model_id)
    working.loc[:, "_role"] = working["longname"].map(canonical_role)
    working.loc[:, "_layer"] = working["longname"].map(_layer_index)
    working.loc[:, "_non_confirmatory"] = working["longname"].map(
        _is_non_confirmatory
    )

    recognized = working.loc[working["_role"].notna()].copy()
    layered = recognized.loc[np.isfinite(recognized["_layer"].to_numpy(dtype=float))].copy()
    layer_quantities_by_row: list[dict[str, float]] = []
    for _, row in layered.iterrows():
        layer_quantities_by_row.append(
            layer_quantities(
                alpha=_column_value(row, "alpha"),
                dks_distance=_column_value(row, "D"),
                frobenius_sq=_column_value(row, "norm"),
                spectral_sq=_column_value(row, "spectral_norm"),
                matrix_rows=_column_value(row, "M"),
                matrix_cols=_column_value(row, "N"),
                entropy=_column_value(row, "entropy"),
                rank=_column_value(row, "matrix_rank"),
                log_alpha_norm=_column_value(row, "log_alpha_norm"),
            )
        )
    if layer_quantities_by_row:
        quantities = pd.DataFrame(layer_quantities_by_row, index=layered.index)
        layered = layered.drop(
            columns=[column for column in quantities.columns if column in layered.columns]
        ).join(quantities)

    all_depths = tuple(sorted(working["_layer"].dropna().unique().tolist()))
    record: dict[str, object] = {
        "model_id": str(model_id).strip(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }
    record.update({column: float("nan") for column in feature_columns()})
    observed_roles = set(layered["_role"].astype(str))
    record["M__layer_count"] = len(all_depths)
    record["M__canonical_role_count"] = len(observed_roles)
    record["M__canonical_role_fraction"] = len(observed_roles) / len(CANONICAL_ROLES)
    if not layered.empty:
        matrix_rows = pd.to_numeric(layered["matrix_rows"], errors="coerce").to_numpy(
            dtype=float
        )
        matrix_cols = pd.to_numeric(layered["matrix_cols"], errors="coerce").to_numpy(
            dtype=float
        )
        with np.errstate(over="ignore", invalid="ignore"):
            parameter_counts = matrix_rows * matrix_cols
            total_parameters = float(np.sum(parameter_counts))
        if np.isfinite(parameter_counts).all() and total_parameters > 0:
            record["M__log10_canonical_matrix_parameter_count"] = float(
                np.log10(total_parameters)
            )

    for role in CANONICAL_ROLES:
        role_rows = layered.loc[layered["_role"].eq(role)].copy()
        grouped = role_rows.groupby("_layer", sort=True).median(numeric_only=True)
        role_depths = set(grouped.index.tolist())
        coverage = len(role_depths) / len(all_depths) if all_depths else 0.0
        record[f"M__{role}__present"] = float(bool(role_depths))
        record[f"M__{role}__coverage"] = coverage

        for block in FEATURE_BLOCKS:
            for quantity in _BLOCK_QUANTITIES[block]:
                values = [
                    grouped.at[depth, quantity]
                    if depth in grouped.index and quantity in grouped.columns
                    else float("nan")
                    for depth in all_depths
                ]
                summary = summarize_trajectory(values, all_depths)
                for statistic in TRAJECTORY_STATISTICS:
                    record[f"{block}__{role}__{quantity}__{statistic}"] = summary[
                        statistic
                    ]

    non_confirmatory = working["_role"].isna() & working["_non_confirmatory"]
    unknown_role = working["_role"].isna() & ~working["_non_confirmatory"]
    record.update(
        {
            "QC__row_count": int(len(working)),
            "QC__recognized_role_row_count": int(len(recognized)),
            "QC__unlayered_role_row_count": int(
                np.count_nonzero(~np.isfinite(recognized["_layer"].to_numpy(dtype=float)))
            ),
            "QC__fused_role_row_count": int(non_confirmatory.sum()),
            "QC__unknown_role_row_count": int(unknown_role.sum()),
            "QC__canonical_schema_complete": observed_roles == set(CANONICAL_ROLES),
        }
    )
    for role in CANONICAL_ROLES:
        role_rows = layered.loc[layered["_role"].eq(role)].copy()
        role_depths = set(role_rows["_layer"].tolist())
        role_coverage = len(role_depths) / len(all_depths) if all_depths else 0.0
        dks = (
            pd.to_numeric(role_rows.get("dks_distance"), errors="coerce").to_numpy(
                dtype=float
            )
            if not role_rows.empty
            else np.array([], dtype=float)
        )
        finite_dks = dks[np.isfinite(dks)]
        alpha_values = (
            pd.to_numeric(role_rows.get("alpha"), errors="coerce").to_numpy(dtype=float)
            if not role_rows.empty
            else np.array([], dtype=float)
        )
        entropy_values = (
            pd.to_numeric(
                role_rows.get("normalized_entropy"), errors="coerce"
            ).to_numpy(dtype=float)
            if not role_rows.empty
            else np.array([], dtype=float)
        )
        finite_entropy = entropy_values[np.isfinite(entropy_values)]
        denominator = len(role_rows)
        record[f"QC__{role}__role_coverage"] = role_coverage
        record[f"QC__{role}__row_count"] = denominator
        record[f"QC__{role}__valid_alpha_fraction"] = (
            float(np.count_nonzero(np.isfinite(alpha_values))) / denominator
            if denominator
            else float("nan")
        )
        record[f"QC__{role}__dks_valid_fraction"] = (
            float(finite_dks.size) / denominator if denominator else float("nan")
        )
        record[f"QC__{role}__dks_median"] = (
            float(np.median(finite_dks)) if finite_dks.size else float("nan")
        )
        record[f"QC__{role}__dks_le_0_1_fraction"] = (
            float(np.mean(finite_dks <= DKS_SENSITIVITY_MAX))
            if finite_dks.size
            else float("nan")
        )
        record[f"QC__{role}__entropy_out_of_range_fraction"] = (
            float(np.mean(finite_entropy > 1.0))
            if finite_entropy.size
            else float("nan")
        )
    return record
