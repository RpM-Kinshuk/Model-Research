from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spectral_research.study.contracts import canonical_model_id
from spectral_research.study.features import FEATURE_SCHEMA_VERSION, feature_columns


MODERN_TASKS = (
    "IFEval",
    "BBH",
    "MATH Lvl 5",
    "GPQA",
    "MUSR",
    "MMLU-PRO",
)


@dataclass(frozen=True)
class TargetSemantics:
    name: str
    generation: str
    components: tuple[str, ...]
    aggregation: str
    scale: str
    outcome: str


MODERN_SEMANTICS = TargetSemantics(
    name="open_llm_leaderboard_v2_six_task_mean",
    generation="open_llm_leaderboard_v2",
    components=MODERN_TASKS,
    aggregation="arithmetic_mean",
    scale="percentage_points",
    outcome="modern_average",
)


class TargetSemanticsError(ValueError):
    pass


def _clean_identifier(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _canonical_identifier(value: object) -> str:
    model_id = canonical_model_id(value)
    return model_id or ""


def _exclusion_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["model_id", "reason", "detail"])


def build_modern_targets(
    frame: pd.DataFrame,
    *,
    model_column: str = "model_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {model_column, *MODERN_TASKS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise TargetSemanticsError(f"Missing modern target columns: {missing}")

    optional_columns = [
        column
        for column in (
            "model_sha",
            "Model sha",
            "eval_name",
            "architecture",
            "Architecture",
            "Architecture_lb",
        )
        if column in frame.columns
    ]
    working = frame.loc[:, [model_column, *MODERN_TASKS, *optional_columns]].copy()
    working[model_column] = working[model_column].map(_canonical_identifier)
    for column in MODERN_TASKS:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    sha_columns = [
        column for column in ("model_sha", "Model sha") if column in working.columns
    ]
    working["_target_model_sha"] = ""
    working["_checkpoint_conflict"] = False
    for index, row in working.iterrows():
        values = {
            _clean_identifier(row[column])
            for column in sha_columns
            if _clean_identifier(row[column])
        }
        if len(values) > 1:
            working.at[index, "_checkpoint_conflict"] = True
        elif values:
            working.at[index, "_target_model_sha"] = next(iter(values))
    working["_target_eval_name"] = (
        working["eval_name"].map(_clean_identifier)
        if "eval_name" in working.columns
        else ""
    )
    architecture_columns = [
        column
        for column in ("architecture", "Architecture", "Architecture_lb")
        if column in working.columns
    ]
    working["_target_architecture"] = ""
    for column in architecture_columns:
        values = working[column].map(_clean_identifier)
        valid = values.ne("") & ~values.str.casefold().isin(("unknown", "?", "nan"))
        fill = working["_target_architecture"].eq("") & valid
        working.loc[fill, "_target_architecture"] = values.loc[fill]

    exclusions: list[dict[str, object]] = []
    missing_id = working[model_column].eq("")
    for _ in range(int(missing_id.sum())):
        exclusions.append(
            {"model_id": "", "reason": "missing_model_id", "detail": "target row has no canonical identifier"}
        )
    working = working.loc[~missing_id].copy()

    kept: list[pd.Series] = []
    for model_id, group in working.groupby(model_column, sort=True):
        checkpoint_values = set(
            group["_target_model_sha"].loc[
                group["_target_model_sha"].astype(str).ne("")
            ]
        )
        if group["_checkpoint_conflict"].any() or len(checkpoint_values) > 1:
            exclusions.append(
                {
                    "model_id": str(model_id),
                    "reason": "conflicting_checkpoint_identity",
                    "detail": "duplicate target rows disagree on source model SHA",
                }
            )
            continue
        values = group.loc[:, MODERN_TASKS].to_numpy(dtype=float)
        if len(group) > 1 and not np.allclose(
            values, values[[0]], rtol=0.0, atol=1e-12, equal_nan=True
        ):
            exclusions.append(
                {
                    "model_id": str(model_id),
                    "reason": "conflicting_target_rows",
                    "detail": f"{len(group)} rows disagree on at least one modern task",
                }
            )
            continue
        row = group.iloc[0]
        row_values = row.loc[list(MODERN_TASKS)].to_numpy(dtype=float)
        if not np.isfinite(row_values).all():
            exclusions.append(
                {
                    "model_id": str(model_id),
                    "reason": "incomplete_modern_target",
                    "detail": "all six modern task scores are required",
                }
            )
            continue
        if np.any((row_values < 0.0) | (row_values > 100.0)):
            exclusions.append(
                {
                    "model_id": str(model_id),
                    "reason": "out_of_range_modern_target",
                    "detail": "modern task scores must use percentage-point scale",
                }
            )
            continue
        kept.append(row)

    output_columns = [
        "model_id",
        *MODERN_TASKS,
        "modern_average",
        "target_semantics",
        "target_components",
        "target_aggregation",
        "target_scale",
        "benchmark_generation",
        "target_outcome",
        "target_semantics_evidence",
        "target_model_sha",
        "target_checkpoint_status",
        "target_eval_name",
        "target_architecture",
    ]
    if not kept:
        return pd.DataFrame(columns=output_columns), _exclusion_frame(exclusions)

    targets = pd.DataFrame(kept).rename(columns={model_column: "model_id"})
    targets["modern_average"] = targets.loc[:, MODERN_TASKS].mean(axis=1)
    targets["target_semantics"] = MODERN_SEMANTICS.name
    targets["target_components"] = "|".join(MODERN_SEMANTICS.components)
    targets["target_aggregation"] = MODERN_SEMANTICS.aggregation
    targets["target_scale"] = MODERN_SEMANTICS.scale
    targets["benchmark_generation"] = MODERN_SEMANTICS.generation
    targets["target_outcome"] = MODERN_SEMANTICS.outcome
    targets["target_semantics_evidence"] = "six_named_v2_components_recomputed"
    targets["target_model_sha"] = targets.pop("_target_model_sha")
    targets["target_checkpoint_status"] = np.where(
        targets["target_model_sha"].astype(str).ne(""), "recorded", "unavailable"
    )
    targets["target_eval_name"] = targets.pop("_target_eval_name")
    targets["target_architecture"] = targets.pop("_target_architecture")
    targets = targets.loc[:, output_columns].sort_values("model_id").reset_index(drop=True)
    return targets, _exclusion_frame(exclusions)


_SEMANTIC_COLUMNS = (
    "target_semantics",
    "target_components",
    "target_aggregation",
    "target_scale",
    "benchmark_generation",
    "target_outcome",
    "target_semantics_evidence",
)


def _semantic_contract(frame: pd.DataFrame) -> tuple[str, ...]:
    contract: list[str] = []
    for column in _SEMANTIC_COLUMNS:
        if column not in frame.columns:
            raise TargetSemanticsError(f"{column} column is required")
        values = frame[column].astype("string").str.strip()
        unique = values.dropna().unique().tolist()
        if values.isna().any() or values.eq("").any() or len(unique) != 1:
            raise TargetSemanticsError(f"Expected one {column}, found {unique}")
        contract.append(unique[0])
    return tuple(contract)


def _validate_recomputed_target(frame: pd.DataFrame, target: str) -> None:
    required = {target, *MODERN_TASKS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise TargetSemanticsError(f"Cannot verify target semantics; missing columns: {missing}")
    components = frame.loc[:, MODERN_TASKS].apply(pd.to_numeric, errors="coerce")
    outcome = pd.to_numeric(frame[target], errors="coerce")
    values = components.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 100.0)):
        raise TargetSemanticsError("Modern component scores do not use finite percentage-point scale")
    recomputed = components.mean(axis=1).to_numpy(dtype=float)
    if not np.isfinite(outcome).all() or not np.allclose(
        outcome.to_numpy(dtype=float), recomputed, rtol=0.0, atol=1e-10
    ):
        raise TargetSemanticsError(f"{target} does not match the recomputed six-task mean")


def validate_target_semantics(
    development: pd.DataFrame,
    external: pd.DataFrame,
    *,
    target: str,
) -> None:
    development_semantics = _semantic_contract(development)
    external_semantics = _semantic_contract(external)
    expected_semantics = (
        MODERN_SEMANTICS.name,
        "|".join(MODERN_SEMANTICS.components),
        MODERN_SEMANTICS.aggregation,
        MODERN_SEMANTICS.scale,
        MODERN_SEMANTICS.generation,
        MODERN_SEMANTICS.outcome,
        "six_named_v2_components_recomputed",
    )
    if development_semantics != expected_semantics or external_semantics != expected_semantics:
        raise TargetSemanticsError(
            "Target semantics differ from the frozen modern contract: "
            f"expected={expected_semantics}, development={development_semantics}, "
            f"external={external_semantics}"
        )
    if development_semantics != external_semantics:
        raise TargetSemanticsError(
            f"Target semantics differ: development={development_semantics}, external={external_semantics}"
        )
    if target not in (MODERN_SEMANTICS.outcome, *MODERN_TASKS):
        raise TargetSemanticsError(
            f"Requested outcome {target!r} is not in the frozen modern contract"
        )
    if target == MODERN_SEMANTICS.outcome:
        _validate_recomputed_target(development, target)
        _validate_recomputed_target(external, target)
    else:
        for name, frame in (("development", development), ("external", external)):
            values = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 100.0)):
                raise TargetSemanticsError(
                    f"{name} target {target!r} is not finite percentage-point data"
                )


def confirmatory_group_population(
    frame: pd.DataFrame,
    *,
    group_column: str = "ancestry_component",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if group_column not in frame.columns:
        raise KeyError(f"Missing group column: {group_column}")
    required = {"component_provenance", "component_edge_count"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing ancestry evidence columns: {missing}")
    groups = frame[group_column].map(_clean_identifier)
    graph_provenance = frame["component_provenance"].map(_clean_identifier).eq("graph")
    edge_count = pd.to_numeric(frame["component_edge_count"], errors="coerce").fillna(0)
    has_group = groups.ne("")
    eligible_mask = has_group & graph_provenance & edge_count.ge(1)
    eligible = frame.loc[eligible_mask].copy()
    eligible[group_column] = groups.loc[eligible_mask]
    exclusions = []
    for index, row in frame.loc[~eligible_mask].iterrows():
        ungrouped = not bool(has_group.loc[index])
        exclusions.append(
            {
                "model_id": _clean_identifier(row.get("model_id")),
                "reason": "ungrouped" if ungrouped else "weak_ancestry_component",
                "detail": (
                    "no graph-derived ancestry component"
                    if ungrouped
                    else "component lacks graph edge evidence or graph provenance"
                ),
            }
        )
    return eligible.reset_index(drop=True), _exclusion_frame(exclusions)


@dataclass(frozen=True)
class GroupedFold:
    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray


def grouped_outer_folds(
    frame: pd.DataFrame,
    *,
    n_splits: int,
    random_state: int,
    group_column: str = "ancestry_component",
) -> list[GroupedFold]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    if group_column not in frame.columns:
        raise KeyError(f"Missing group column: {group_column}")
    groups = frame[group_column].map(_clean_identifier)
    if groups.eq("").any():
        raise ValueError("Grouped folds require a non-empty ancestry component for every row")

    group_sizes = groups.value_counts(sort=False)
    group_count = len(group_sizes)
    if group_count < 2:
        raise ValueError("At least two ancestry components are required")
    if group_count < n_splits:
        raise ValueError(
            f"Requested {n_splits} grouped folds but only {group_count} components are available"
        )
    effective_splits = int(n_splits)

    rng = np.random.default_rng(random_state)
    shuffled = list(group_sizes.index[rng.permutation(group_count)])
    ordered_groups = sorted(shuffled, key=lambda group: -int(group_sizes[group]))
    fold_groups: list[list[str]] = [[] for _ in range(effective_splits)]
    fold_sizes = np.zeros(effective_splits, dtype=int)
    for group in ordered_groups:
        fold_index = int(np.argmin(fold_sizes))
        fold_groups[fold_index].append(str(group))
        fold_sizes[fold_index] += int(group_sizes[group])

    all_indices = np.arange(len(frame), dtype=int)
    folds: list[GroupedFold] = []
    for fold_id, held_groups in enumerate(fold_groups):
        test_mask = groups.isin(held_groups).to_numpy()
        folds.append(
            GroupedFold(
                fold_id=fold_id,
                train_indices=all_indices[~test_mask],
                test_indices=all_indices[test_mask],
            )
        )
    return folds


def component_sample_weights(groups: Sequence[object]) -> np.ndarray:
    group_values = pd.Series(groups, dtype="object").map(_clean_identifier)
    if group_values.eq("").any():
        raise ValueError("All sample-weight groups must be non-empty")
    counts = group_values.value_counts()
    weights = group_values.map(lambda value: 1.0 / float(counts[value])).to_numpy(dtype=float)
    return weights


def group_weighted_mae(
    actual: Sequence[float],
    predicted: Sequence[float],
    groups: Sequence[object],
) -> float:
    values = pd.DataFrame(
        {
            "actual": np.asarray(actual, dtype=float),
            "predicted": np.asarray(predicted, dtype=float),
            "group": pd.Series(groups, dtype="object").map(_clean_identifier),
        }
    )
    valid = np.isfinite(values["actual"]) & np.isfinite(values["predicted"]) & values["group"].ne("")
    values = values.loc[valid].copy()
    if values.empty:
        return float("nan")
    values["absolute_error"] = (values["actual"] - values["predicted"]).abs()
    return float(values.groupby("group")["absolute_error"].mean().mean())


class FiniteVarianceSelector(BaseEstimator, TransformerMixin):
    """Select finite, non-constant training columns without seeing held-out rows."""

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "FiniteVarianceSelector":
        values = np.asarray(x, dtype=float)
        values = np.where(np.isfinite(values), values, np.nan)
        keep: list[int] = []
        for index in range(values.shape[1]):
            finite = values[np.isfinite(values[:, index]), index]
            if finite.size >= 2 and float(np.ptp(finite)) > 0.0:
                keep.append(index)
        self.keep_indices_ = np.asarray(keep, dtype=int)
        self.fallback_ = self.keep_indices_.size == 0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        values = np.where(np.isfinite(values), values, np.nan)
        if self.fallback_:
            return np.zeros((values.shape[0], 1), dtype=float)
        return values[:, self.keep_indices_]


class WeightedMedianImputer(BaseEstimator, TransformerMixin):
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "WeightedMedianImputer":
        values = np.asarray(x, dtype=float)
        values = np.where(np.isfinite(values), values, np.nan)
        weights = (
            np.ones(values.shape[0], dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        if weights.shape != (values.shape[0],):
            raise ValueError("sample_weight shape does not match rows")
        statistics: list[float] = []
        for column in range(values.shape[1]):
            finite = np.isfinite(values[:, column]) & np.isfinite(weights) & (weights > 0)
            if not finite.any():
                statistics.append(0.0)
                continue
            column_values = values[finite, column]
            column_weights = weights[finite]
            order = np.argsort(column_values, kind="mergesort")
            sorted_values = column_values[order]
            sorted_weights = column_weights[order]
            positions = (np.cumsum(sorted_weights) - 0.5 * sorted_weights) / float(
                sorted_weights.sum()
            )
            statistics.append(float(np.interp(0.5, positions, sorted_values)))
        self.statistics_ = np.asarray(statistics, dtype=float)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float).copy()
        values[~np.isfinite(values)] = np.nan
        missing = np.isnan(values)
        if missing.any():
            values[missing] = np.take(self.statistics_, np.where(missing)[1])
        return values


def build_ridge_pipeline(*, alpha: float) -> Pipeline:
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Ridge alpha must be finite and positive")
    return Pipeline(
        steps=[
            ("selector", FiniteVarianceSelector()),
            ("imputer", WeightedMedianImputer()),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )


def _fit_weighted_pipeline(
    pipeline: Pipeline,
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
) -> Pipeline:
    pipeline.fit(
        x,
        y,
        imputer__sample_weight=sample_weight,
        scaler__sample_weight=sample_weight,
        ridge__sample_weight=sample_weight,
    )
    return pipeline


def _feature_blocks(frame: pd.DataFrame) -> dict[str, list[str]]:
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated(keep=False)].unique().tolist()
        raise ValueError(f"feature table contains duplicate column labels: {duplicates}")
    if "feature_schema_version" not in frame.columns:
        raise ValueError("feature_schema_version is required")
    versions = frame["feature_schema_version"].astype("string").str.strip()
    if versions.isna().any() or versions.eq("").any() or not versions.eq(
        FEATURE_SCHEMA_VERSION
    ).all():
        raise ValueError(
            f"feature schema must be {FEATURE_SCHEMA_VERSION!r} on every row; "
            f"found {sorted(versions.dropna().unique().tolist())}"
        )
    approved = set(feature_columns())
    prefixed = {
        column for column in frame.columns if column.startswith(("M__", "S__", "H__"))
    }
    unapproved = sorted(prefixed - approved)
    if unapproved:
        raise ValueError(f"unapproved prefixed feature columns: {unapproved}")
    m_columns = [column for column in feature_columns(("M",)) if column in frame.columns]
    s_columns = [column for column in feature_columns(("S",)) if column in frame.columns]
    h_columns = [column for column in feature_columns(("H",)) if column in frame.columns]
    if not m_columns:
        raise ValueError("No M structural features are available")
    if not s_columns:
        raise ValueError("No S non-HTSR spectral features are available")
    if not h_columns:
        raise ValueError("No H HTSR features are available")
    return {
        "M": m_columns,
        "M+S": [*m_columns, *s_columns],
        "M+S+H": [*m_columns, *s_columns, *h_columns],
    }


def _validate_matching_feature_schema(
    development: pd.DataFrame,
    external: pd.DataFrame,
) -> None:
    try:
        development_blocks = _feature_blocks(development)
        external_blocks = _feature_blocks(external)
    except ValueError as error:
        raise ValueError(f"Invalid development/external feature schema: {error}") from error
    development_columns = tuple(development_blocks["M+S+H"])
    external_columns = tuple(external_blocks["M+S+H"])
    if development_columns != external_columns:
        raise ValueError(
            "External feature schema differs from development feature schema: "
            f"development={development_columns}, external={external_columns}"
        )


def _select_alpha(
    train_frame: pd.DataFrame,
    *,
    target: str,
    feature_columns: Sequence[str],
    group_column: str,
    inner_splits: int,
    ridge_alphas: Sequence[float],
    random_state: int,
) -> float:
    folds = grouped_outer_folds(
        train_frame,
        n_splits=inner_splits,
        random_state=random_state,
        group_column=group_column,
    )
    x = train_frame.loc[:, feature_columns].to_numpy(dtype=float)
    y = train_frame[target].to_numpy(dtype=float)
    groups = train_frame[group_column].astype(str).to_numpy()
    candidates: list[tuple[float, float]] = []
    for alpha in ridge_alphas:
        predictions = np.full(len(train_frame), np.nan, dtype=float)
        for fold in folds:
            pipeline = build_ridge_pipeline(alpha=float(alpha))
            weights = component_sample_weights(groups[fold.train_indices])
            _fit_weighted_pipeline(
                pipeline,
                x[fold.train_indices],
                y[fold.train_indices],
                weights,
            )
            predictions[fold.test_indices] = pipeline.predict(x[fold.test_indices])
        candidates.append((group_weighted_mae(y, predictions, groups), float(alpha)))
    return min(candidates, key=lambda item: (item[0], item[1]))[1]


def _rank_correlation(function: object, actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) < 2 or np.unique(actual).size < 2 or np.unique(predicted).size < 2:
        return float("nan")
    result = function(actual, predicted)
    return float(result.statistic)


def evaluate_grouped_ridge(
    frame: pd.DataFrame,
    *,
    target: str,
    group_column: str = "ancestry_component",
    outer_splits: int = 5,
    inner_splits: int = 4,
    ridge_alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    random_state: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    if target not in frame.columns:
        raise KeyError(f"Missing target: {target}")
    if "model_id" not in frame.columns:
        raise KeyError("model_id is required")
    alphas = tuple(float(alpha) for alpha in ridge_alphas)
    if not alphas:
        raise ValueError("At least one Ridge alpha is required")

    population, _ = confirmatory_group_population(frame, group_column=group_column)
    population[target] = pd.to_numeric(population[target], errors="coerce")
    population = population.loc[np.isfinite(population[target])].reset_index(drop=True)
    if population["model_id"].astype(str).duplicated().any():
        raise ValueError("Prediction population contains duplicate model_id values")

    feature_blocks = _feature_blocks(population)
    folds = grouped_outer_folds(
        population,
        n_splits=outer_splits,
        random_state=random_state,
        group_column=group_column,
    )
    rows: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for fold in folds:
        train = population.iloc[fold.train_indices].reset_index(drop=True)
        test = population.iloc[fold.test_indices].reset_index(drop=True)
        train_groups = train[group_column].astype(str).to_numpy()
        audit.append(
            {
                "outer_fold": fold.fold_id,
                "train_groups": sorted(set(train_groups)),
                "test_groups": sorted(set(test[group_column].astype(str))),
                "n_train": len(train),
                "n_test": len(test),
            }
        )
        for block_name, columns in feature_blocks.items():
            chosen_alpha = _select_alpha(
                train,
                target=target,
                feature_columns=columns,
                group_column=group_column,
                inner_splits=inner_splits,
                ridge_alphas=alphas,
                random_state=random_state + fold.fold_id,
            )
            pipeline = build_ridge_pipeline(alpha=chosen_alpha)
            _fit_weighted_pipeline(
                pipeline,
                train.loc[:, columns].to_numpy(dtype=float),
                train[target].to_numpy(dtype=float),
                component_sample_weights(train_groups),
            )
            predicted = pipeline.predict(test.loc[:, columns].to_numpy(dtype=float))
            for position, (_, row) in enumerate(test.iterrows()):
                rows.append(
                    {
                        "model_id": str(row["model_id"]),
                        group_column: str(row[group_column]),
                        "target": target,
                        "feature_block": block_name,
                        "outer_fold": fold.fold_id,
                        "chosen_alpha": chosen_alpha,
                        "actual": float(row[target]),
                        "prediction": float(predicted[position]),
                    }
                )

    predictions = pd.DataFrame(rows)
    metric_rows: list[dict[str, object]] = []
    for block_name, block in predictions.groupby("feature_block", sort=False):
        actual = block["actual"].to_numpy(dtype=float)
        predicted = block["prediction"].to_numpy(dtype=float)
        groups = block[group_column].astype(str).to_numpy()
        metric_rows.append(
            {
                "target": target,
                "feature_block": block_name,
                "n_models": len(block),
                "n_components": block[group_column].nunique(),
                "group_weighted_mae": group_weighted_mae(actual, predicted, groups),
                "kendall_tau": _rank_correlation(kendalltau, actual, predicted),
                "spearman": _rank_correlation(spearmanr, actual, predicted),
                "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else float("nan"),
            }
        )
    return predictions, pd.DataFrame(metric_rows), audit


@dataclass
class FittedRidge:
    feature_block: str
    feature_columns: tuple[str, ...]
    chosen_alpha: float
    pipeline: Pipeline


def _fit_final_ridge_models(
    development: pd.DataFrame,
    *,
    target: str,
    group_column: str,
    inner_splits: int,
    ridge_alphas: Sequence[float],
    random_state: int,
) -> tuple[dict[str, FittedRidge], pd.DataFrame]:
    feature_blocks = _feature_blocks(development)
    groups = development[group_column].astype(str).to_numpy()
    fitted: dict[str, FittedRidge] = {}
    audit_rows: list[dict[str, object]] = []
    for block_name, columns in feature_blocks.items():
        alpha = _select_alpha(
            development,
            target=target,
            feature_columns=columns,
            group_column=group_column,
            inner_splits=inner_splits,
            ridge_alphas=ridge_alphas,
            random_state=random_state,
        )
        pipeline = build_ridge_pipeline(alpha=alpha)
        _fit_weighted_pipeline(
            pipeline,
            development.loc[:, columns].to_numpy(dtype=float),
            development[target].to_numpy(dtype=float),
            component_sample_weights(groups),
        )
        fitted[block_name] = FittedRidge(
            feature_block=block_name,
            feature_columns=tuple(columns),
            chosen_alpha=alpha,
            pipeline=pipeline,
        )
        audit_rows.append(
            {
                "feature_block": block_name,
                "chosen_alpha": alpha,
                "n_models": len(development),
                "n_components": development[group_column].nunique(),
                "fit_population": "development_only",
            }
        )
    return fitted, pd.DataFrame(audit_rows)


def _predict_with_fitted_models(
    models: Mapping[str, FittedRidge],
    frame: pd.DataFrame,
    *,
    target: str,
    group_column: str,
    external_slice: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for block_name, fitted in models.items():
        missing = sorted(set(fitted.feature_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"External feature schema is missing columns: {missing}")
        predicted = fitted.pipeline.predict(
            frame.loc[:, fitted.feature_columns].to_numpy(dtype=float)
        )
        for position, (_, row) in enumerate(frame.iterrows()):
            rows.append(
                {
                    "model_id": str(row["model_id"]),
                    group_column: str(row[group_column]),
                    "target": target,
                    "feature_block": block_name,
                    "chosen_alpha": fitted.chosen_alpha,
                    "actual": float(row[target]),
                    "prediction": float(predicted[position]),
                    "external_slice": external_slice,
                }
            )
    return pd.DataFrame(rows)


def _prediction_metric_rows(
    predictions: pd.DataFrame,
    *,
    target: str,
    group_column: str,
    slice_column: str | None = None,
) -> pd.DataFrame:
    groupers = ["feature_block"] if slice_column is None else [slice_column, "feature_block"]
    rows: list[dict[str, object]] = []
    for key, block in predictions.groupby(groupers, sort=False):
        if slice_column is None:
            slice_value = None
            block_name = str(key)
        else:
            slice_value, block_name = key
        actual = block["actual"].to_numpy(dtype=float)
        predicted = block["prediction"].to_numpy(dtype=float)
        groups = block[group_column].astype(str).to_numpy()
        row: dict[str, object] = {
            "target": target,
            "feature_block": str(block_name),
            "n_models": len(block),
            "n_components": block[group_column].nunique(),
            "group_weighted_mae": group_weighted_mae(actual, predicted, groups),
            "kendall_tau": _rank_correlation(kendalltau, actual, predicted),
            "spearman": _rank_correlation(spearmanr, actual, predicted),
            "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else float("nan"),
        }
        if slice_column is not None:
            row[slice_column] = str(slice_value)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_external_transfer(
    development: pd.DataFrame,
    external: pd.DataFrame,
    *,
    target: str,
    development_component_universe: pd.DataFrame,
    group_column: str = "ancestry_component",
    inner_splits: int = 4,
    ridge_alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    random_state: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit on development only, then evaluate disjoint and overlap external slices."""

    validate_target_semantics(development, external, target=target)
    _validate_matching_feature_schema(development, external)
    if development["model_id"].astype(str).duplicated().any():
        raise ValueError("Development population contains duplicate model_id values")
    if external["model_id"].astype(str).duplicated().any():
        raise ValueError("External population contains duplicate model_id values")
    development_population, _ = confirmatory_group_population(
        development, group_column=group_column
    )
    external_population, _ = confirmatory_group_population(
        external, group_column=group_column
    )
    for population_name, population in (
        ("development", development_population),
        ("external", external_population),
    ):
        if target not in population.columns:
            raise KeyError(f"Missing {population_name} target: {target}")
        population[target] = pd.to_numeric(population[target], errors="coerce")
        population.drop(
            index=population.index[~np.isfinite(population[target].to_numpy(dtype=float))],
            inplace=True,
        )
        population.reset_index(drop=True, inplace=True)
    if development_population.empty:
        raise ValueError("Development population is empty")
    if external_population.empty:
        raise ValueError("External population is empty")

    universe_groups = set(
        development_component_universe[group_column].dropna().astype(str)
    ) if group_column in development_component_universe.columns else set()
    fitted_groups = set(development_population[group_column].astype(str))
    if not fitted_groups.issubset(universe_groups):
        missing_groups = sorted(fitted_groups - universe_groups)
        raise ValueError(
            "development_component_universe must contain all fitted development "
            f"components; missing {missing_groups}"
        )
    strict, overlap = external_component_slices(
        development_component_universe,
        external_population,
        group_column=group_column,
    )

    models, audit = _fit_final_ridge_models(
        development_population,
        target=target,
        group_column=group_column,
        inner_splits=inner_splits,
        ridge_alphas=tuple(float(alpha) for alpha in ridge_alphas),
        random_state=random_state,
    )
    prediction_frames: list[pd.DataFrame] = []
    if not strict.empty:
        prediction_frames.append(
            _predict_with_fitted_models(
                models,
                strict,
                target=target,
                group_column=group_column,
                external_slice="component_disjoint",
            )
        )
    if not overlap.empty:
        prediction_frames.append(
            _predict_with_fitted_models(
                models,
                overlap,
                target=target,
                group_column=group_column,
                external_slice="component_overlap_sensitivity",
            )
        )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = _prediction_metric_rows(
        predictions,
        target=target,
        group_column=group_column,
        slice_column="external_slice",
    )
    return predictions, metrics, audit


def paired_component_bootstrap(
    predictions: pd.DataFrame,
    *,
    baseline_block: str = "M+S",
    htsr_block: str = "M+S+H",
    group_column: str = "ancestry_component",
    n_bootstrap: int = 2000,
    random_state: int = 7,
    minimum_components: int = 10,
) -> dict[str, float | int | bool]:
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if minimum_components < 2:
        raise ValueError("minimum_components must be at least two")
    if "target" in predictions.columns:
        targets = predictions["target"].astype("string").fillna("").unique().tolist()
        if len(targets) != 1:
            raise ValueError(f"paired bootstrap requires one target; found {targets}")
    if "external_slice" in predictions.columns:
        slices = (
            predictions["external_slice"].astype("string").fillna("development").unique().tolist()
        )
        if len(slices) != 1:
            raise ValueError(
                f"paired bootstrap requires one external slice; found {slices}"
            )
    key_columns = ["model_id", group_column]
    for optional in ("target", "external_slice"):
        if optional in predictions.columns:
            key_columns.append(optional)
    baseline = predictions.loc[predictions["feature_block"].eq(baseline_block), key_columns + ["actual", "prediction"]]
    htsr = predictions.loc[predictions["feature_block"].eq(htsr_block), key_columns + ["actual", "prediction"]]
    for name, block in ((baseline_block, baseline), (htsr_block, htsr)):
        if block.duplicated(key_columns).any():
            raise ValueError(f"{name} contains duplicate prediction keys")
        if not np.isfinite(block[["actual", "prediction"]].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} contains non-finite predictions or targets")
    baseline_keys = set(map(tuple, baseline[key_columns].itertuples(index=False, name=None)))
    htsr_keys = set(map(tuple, htsr[key_columns].itertuples(index=False, name=None)))
    if baseline_keys != htsr_keys:
        raise ValueError("Baseline and HTSR methods must have identical populations")
    paired = baseline.merge(htsr, on=key_columns, suffixes=("_baseline", "_htsr"), validate="one_to_one")
    if paired.empty:
        raise ValueError("No paired baseline and HTSR predictions")
    if not np.allclose(paired["actual_baseline"], paired["actual_htsr"], rtol=0.0, atol=1e-12):
        raise ValueError("Paired methods do not share identical targets")
    paired["delta"] = (
        (paired["actual_htsr"] - paired["prediction_htsr"]).abs()
        - (paired["actual_baseline"] - paired["prediction_baseline"]).abs()
    )
    component_delta = paired.groupby(group_column)["delta"].mean().to_numpy(dtype=float)
    inferential = len(component_delta) >= minimum_components
    if inferential:
        rng = np.random.default_rng(random_state)
        samples = rng.choice(
            component_delta, size=(n_bootstrap, len(component_delta)), replace=True
        ).mean(axis=1)
        ci_low = float(np.quantile(samples, 0.025))
        ci_high = float(np.quantile(samples, 0.975))
    else:
        ci_low = float("nan")
        ci_high = float("nan")
    return {
        "n_models": len(paired),
        "n_components": len(component_delta),
        "delta_mae": float(component_delta.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "inferential": inferential,
    }


def _paired_component_deltas(
    predictions: pd.DataFrame,
    *,
    baseline_block: str,
    htsr_block: str,
    group_column: str,
) -> np.ndarray:
    key_columns = ["model_id", group_column]
    for optional in ("target", "external_slice"):
        if optional in predictions.columns:
            key_columns.append(optional)
    baseline = predictions.loc[
        predictions["feature_block"].eq(baseline_block),
        key_columns + ["actual", "prediction"],
    ]
    htsr = predictions.loc[
        predictions["feature_block"].eq(htsr_block),
        key_columns + ["actual", "prediction"],
    ]
    paired = baseline.merge(
        htsr,
        on=key_columns,
        suffixes=("_baseline", "_htsr"),
        validate="one_to_one",
    )
    if len(paired) != len(baseline) or len(paired) != len(htsr):
        raise ValueError("Baseline and HTSR methods must have identical populations")
    delta = (
        (paired["actual_htsr"] - paired["prediction_htsr"]).abs()
        - (paired["actual_baseline"] - paired["prediction_baseline"]).abs()
    )
    return paired.assign(_delta=delta).groupby(group_column)["_delta"].mean().to_numpy(dtype=float)


def paired_component_signflip_pvalue(
    predictions: pd.DataFrame,
    *,
    baseline_block: str = "M+S",
    htsr_block: str = "M+S+H",
    group_column: str = "ancestry_component",
    n_replicates: int = 10_000,
    random_state: int = 7,
    minimum_components: int = 10,
) -> float:
    """Two-sided paired randomization test over ancestry-component effects."""

    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive")
    component_delta = _paired_component_deltas(
        predictions,
        baseline_block=baseline_block,
        htsr_block=htsr_block,
        group_column=group_column,
    )
    if len(component_delta) < minimum_components:
        return float("nan")
    observed = abs(float(np.mean(component_delta)))
    rng = np.random.default_rng(random_state)
    signs = rng.choice(
        np.array([-1.0, 1.0]), size=(n_replicates, len(component_delta))
    )
    randomized = np.abs(np.mean(signs * component_delta, axis=1))
    return float((1 + np.count_nonzero(randomized >= observed)) / (n_replicates + 1))


def evaluate_secondary_endpoints(
    frame: pd.DataFrame,
    *,
    group_column: str = "ancestry_component",
    outer_splits: int = 5,
    inner_splits: int = 4,
    ridge_alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    bootstrap_replicates: int = 2_000,
    signflip_replicates: int = 10_000,
    random_state: int = 7,
    minimum_components: int = 10,
) -> dict[str, object]:
    """Evaluate the frozen six secondary tasks and one Holm correction family."""

    missing = sorted(set(MODERN_TASKS) - set(frame.columns))
    if missing:
        raise TargetSemanticsError(f"Missing modern secondary targets: {missing}")
    prediction_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    increments: list[dict[str, object]] = []
    raw_p_values: dict[str, float] = {}
    for endpoint_index, target in enumerate(MODERN_TASKS):
        endpoint_seed = random_state + endpoint_index
        predictions, metrics, audit = evaluate_grouped_ridge(
            frame,
            target=target,
            group_column=group_column,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            ridge_alphas=ridge_alphas,
            random_state=random_state,
        )
        prediction_frames.append(predictions)
        metric_frames.append(metrics)
        audit_rows.extend({"target": target, **row} for row in audit)
        interval = paired_component_bootstrap(
            predictions,
            group_column=group_column,
            n_bootstrap=bootstrap_replicates,
            random_state=endpoint_seed,
            minimum_components=minimum_components,
        )
        p_value = paired_component_signflip_pvalue(
            predictions,
            group_column=group_column,
            n_replicates=signflip_replicates,
            random_state=endpoint_seed,
            minimum_components=minimum_components,
        )
        raw_p_values[target] = p_value
        increments.append(
            {
                "target": target,
                **interval,
                "p_value_raw": p_value,
            }
        )
    if all(np.isfinite(value) for value in raw_p_values.values()):
        adjusted = holm_adjust(raw_p_values)
    else:
        adjusted = {target: float("nan") for target in MODERN_TASKS}
    for row in increments:
        row["p_value_holm"] = adjusted[str(row["target"])]
        row["holm_family_size"] = len(MODERN_TASKS)
    return {
        "predictions": pd.concat(prediction_frames, ignore_index=True),
        "metrics": pd.concat(metric_frames, ignore_index=True),
        "increments": pd.DataFrame(increments),
        "audit": audit_rows,
    }


def external_component_slices(
    development: pd.DataFrame,
    external: pd.DataFrame,
    *,
    group_column: str = "ancestry_component",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"model_id", group_column, "component_graph_fingerprint"}
    for name, frame in (("development", development), ("external", external)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} component table is missing columns: {missing}")
    fingerprints: dict[str, set[str]] = {}
    for name, frame in (("development", development), ("external", external)):
        values = frame["component_graph_fingerprint"].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise ValueError(f"{name} requires a complete graph fingerprint on every row")
        fingerprints[name] = set(values.tolist())
    development_fingerprints = fingerprints["development"]
    external_fingerprints = fingerprints["external"]
    if (
        len(development_fingerprints) != 1
        or len(external_fingerprints) != 1
        or development_fingerprints != external_fingerprints
    ):
        raise ValueError(
            "Development and external components require one matching graph fingerprint"
        )
    model_overlap = set(development["model_id"].astype(str)) & set(
        external["model_id"].astype(str)
    )
    if model_overlap:
        raise ValueError(f"Development and external model IDs overlap: {sorted(model_overlap)}")
    development_groups = set(development[group_column].dropna().astype(str))
    overlap_mask = external[group_column].astype(str).isin(development_groups)
    strict = external.loc[~overlap_mask].copy().reset_index(drop=True)
    overlap = external.loc[overlap_mask].copy().reset_index(drop=True)
    return strict, overlap


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    invalid = {
        name: value
        for name, value in p_values.items()
        if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0
    }
    if invalid:
        raise ValueError(f"p-values must be finite and in [0, 1]: {invalid}")
    ordered = sorted(p_values.items(), key=lambda item: (float(item[1]), item[0]))
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        candidate = min(1.0, float(value) * (count - index))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def assess_htsr_increment(
    *,
    baseline_mae: float,
    htsr_mae: float,
    ci_low: float,
    ci_high: float,
    external_delta: float,
    external_available: bool,
    external_confirmatory: bool = True,
    external_slice: str = "component_disjoint",
    n_components: int,
    minimum_relative_improvement: float = 0.05,
    minimum_components: int = 10,
) -> dict[str, object]:
    reasons: list[str] = []
    relative_improvement = (
        (float(baseline_mae) - float(htsr_mae)) / float(baseline_mae)
        if np.isfinite(baseline_mae) and baseline_mae > 0 and np.isfinite(htsr_mae)
        else float("nan")
    )
    if not np.isfinite(relative_improvement) or relative_improvement < minimum_relative_improvement:
        reasons.append("insufficient_mae_improvement")
    valid_interval = np.isfinite(ci_low) and np.isfinite(ci_high) and ci_low <= ci_high
    if not valid_interval:
        reasons.append("invalid_bootstrap_interval")
    elif ci_low <= 0 <= ci_high:
        reasons.append("bootstrap_interval_includes_zero")
    elif ci_low > 0:
        reasons.append("bootstrap_interval_supports_harm")
    if n_components < minimum_components:
        reasons.append("insufficient_components")
    if not external_available:
        reasons.append("external_unavailable")
    elif not external_confirmatory:
        reasons.append("external_not_confirmatory")
    elif external_slice != "component_disjoint":
        reasons.append("external_not_component_disjoint")
    elif not np.isfinite(external_delta) or external_delta > 0:
        reasons.append("external_reversal")
    return {
        "passed": not reasons,
        "relative_mae_improvement": relative_improvement,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "external_delta": float(external_delta),
        "reasons": reasons,
    }
