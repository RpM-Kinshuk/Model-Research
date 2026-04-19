# Performance Prediction Cycle 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a small performance-prediction pilot that uses structural and spectral model neighborhoods to predict `Average ⬆️`, `MMLU-PRO`, and `MATH Lvl 5` on the current joined scored subset.

**Architecture:** Add a focused `spectral_research.prediction` module that rebuilds the current audited mixed-zoo manifest, joins score targets via `display_name -> atlas_repo_id` with `fullname` fallback, and evaluates simple grouped-CV predictors. The main path is neighborhood prediction for `structural_basics_only`, `pl_alpha_scalar`, `full_module_trace`, and `role_aggregated_trace`, reported both raw and after a small structural-control residualization. A repaired CLI writes dataset, result, and OOF artifacts plus a short report.

**Tech Stack:** Python, pandas, numpy, scikit-learn, existing `alpha_cluster_dashboard.data`, existing `spectral_research` feature and heterogeneity helpers, `unittest`

---

## File Structure

- Create: `spectral_research/prediction.py`
  - Own the joined prediction dataset, pairwise similarity helpers, grouped CV evaluation, neighborhood prediction, structural baseline, and report-friendly result tables.
- Create: `tests/test_prediction.py`
  - Own dataset-join, grouping, neighborhood, and residualization regression tests.
- Modify: `scripts/spectral_prediction_research.py`
  - Repair the broken script import path and align the CLI/report with the new Cycle 1 pilot.
- Modify: `docs/superpowers/specs/2026-04-18-spectral-family-retrieval-preliminary-findings.md`
  - Append the Cycle 1 prediction outcome after the run.

## Task 1: Build The Joined Prediction Dataset

**Files:**
- Create: `spectral_research/prediction.py`
- Test: `tests/test_prediction.py`

- [ ] **Step 1: Write the failing dataset-join tests**

```python
import unittest
from pathlib import Path
import tempfile

import pandas as pd

from spectral_research.prediction import (
    PERFORMANCE_TARGETS,
    _group_key,
    join_prediction_targets,
)


class PredictionDatasetTests(unittest.TestCase):
    def test_join_prediction_targets_uses_display_name_to_atlas_repo_id(self) -> None:
        manifest = pd.DataFrame(
            [
                {
                    "model_id": "repo--model-a",
                    "display_name": "repo/model-a",
                    "audited_base_family": "fam/a",
                    "candidate_family": "fam/a",
                    "label_status": "confirmed",
                    "query_bucket": "supported_candidate",
                    "num_layers": 24,
                    "raw_module_count": 7,
                    "common_schema": True,
                    "schema_name": "common_llm_7",
                }
            ]
        )
        leaderboard = pd.DataFrame(
            [
                {
                    "atlas_repo_id": "repo/model-a",
                    "fullname": "irrelevant",
                    "Average ⬆️": 33.0,
                    "MMLU-PRO": 31.0,
                    "MATH Lvl 5": 12.0,
                }
            ]
        )

        joined = join_prediction_targets(manifest, leaderboard)

        self.assertEqual(len(joined), 1)
        self.assertEqual(float(joined.iloc[0]["Average ⬆️"]), 33.0)

    def test_join_prediction_targets_falls_back_to_fullname(self) -> None:
        manifest = pd.DataFrame(
            [
                {
                    "model_id": "repo--model-b",
                    "display_name": "repo/model-b",
                    "audited_base_family": "",
                    "candidate_family": "fam/b",
                    "label_status": "provisional",
                    "query_bucket": "heterogeneous",
                    "num_layers": 32,
                    "raw_module_count": 5,
                    "common_schema": False,
                    "schema_name": "bert_like_5",
                }
            ]
        )
        leaderboard = pd.DataFrame(
            [
                {
                    "atlas_repo_id": "other/repo",
                    "fullname": "repo/model-b",
                    "Average ⬆️": 27.0,
                    "MMLU-PRO": 20.0,
                    "MATH Lvl 5": 4.0,
                }
            ]
        )

        joined = join_prediction_targets(manifest, leaderboard)

        self.assertEqual(len(joined), 1)
        self.assertEqual(float(joined.iloc[0]["MMLU-PRO"]), 20.0)

    def test_group_key_prefers_audited_family_then_candidate_then_model(self) -> None:
        row_a = pd.Series({"model_id": "m-a", "audited_base_family": "fam/a", "candidate_family": "cand/a"})
        row_b = pd.Series({"model_id": "m-b", "audited_base_family": "", "candidate_family": "cand/b"})
        row_c = pd.Series({"model_id": "m-c", "audited_base_family": "", "candidate_family": ""})

        self.assertEqual(_group_key(row_a), "fam/a")
        self.assertEqual(_group_key(row_b), "cand/b")
        self.assertEqual(_group_key(row_c), "m-c")
```

- [ ] **Step 2: Run the dataset tests to verify they fail**

Run: `conda run -n esd_ind python -m unittest tests.test_prediction.PredictionDatasetTests -v`

Expected: `FAIL` with `ModuleNotFoundError: No module named 'spectral_research.prediction'`

- [ ] **Step 3: Write the minimal dataset/join implementation**

```python
# spectral_research/prediction.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_cluster_dashboard.data import AlphaRecord, load_alpha_records
from spectral_research.benchmark import build_audit_manifest
from spectral_research.features import alpha_scalar_features
from spectral_research.heterogeneity import structural_basics_feature_map, role_aggregated_trace_score

PERFORMANCE_TARGETS = ("Average ⬆️", "MMLU-PRO", "MATH Lvl 5")


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _group_key(row: pd.Series) -> str:
    for column in ("audited_base_family", "candidate_family", "model_id"):
        text = _clean_text(row.get(column))
        if text:
            return text
    return "unknown"


def join_prediction_targets(manifest: pd.DataFrame, leaderboard: pd.DataFrame) -> pd.DataFrame:
    score_columns = ["atlas_repo_id", "fullname", *PERFORMANCE_TARGETS]
    right = leaderboard.loc[:, [column for column in score_columns if column in leaderboard.columns]].copy()

    by_atlas = manifest.merge(
        right,
        left_on="display_name",
        right_on="atlas_repo_id",
        how="left",
    )
    missing = by_atlas["Average ⬆️"].isna()
    if missing.any() and "fullname" in right.columns:
        fallback = manifest.loc[missing].merge(
            right,
            left_on="display_name",
            right_on="fullname",
            how="left",
        )
        for column in PERFORMANCE_TARGETS:
            by_atlas.loc[missing, column] = fallback[column].to_numpy()

    for column in PERFORMANCE_TARGETS:
        by_atlas[column] = pd.to_numeric(by_atlas[column], errors="coerce")
    by_atlas["cv_group"] = by_atlas.apply(_group_key, axis=1)
    return by_atlas


def build_prediction_dataset(
    *,
    metrics_dir: Path,
    leaderboard_csv: Path,
    lineage_csv: Path,
    freeze_csv: Path,
) -> tuple[pd.DataFrame, dict[str, AlphaRecord]]:
    records = load_alpha_records(metrics_dir)
    records_frame = pd.DataFrame(
        [
            {
                "model_id": record.model_id,
                "display_name": record.display_name,
                "path": record.path,
                "num_layers": record.num_layers,
                "raw_module_count": record.raw_module_count,
                "raw_modules": ", ".join(record.raw_modules),
                "common_schema": record.common_schema,
                "schema_name": f"schema_{record.raw_module_count}",
            }
            for record in records
        ]
    )
    metadata = pd.read_csv(lineage_csv)
    leaderboard = pd.read_csv(leaderboard_csv)
    overrides = pd.read_csv(freeze_csv)
    manifest = build_audit_manifest(records_frame, metadata, leaderboard, overrides)
    joined = join_prediction_targets(manifest, leaderboard)
    joined = joined.dropna(subset=list(PERFORMANCE_TARGETS)).reset_index(drop=True)
    records_by_model_id = {record.model_id: record for record in records}
    return joined, records_by_model_id
```

- [ ] **Step 4: Run the dataset tests to verify they pass**

Run: `conda run -n esd_ind python -m unittest tests.test_prediction.PredictionDatasetTests -v`

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 add \
  spectral_research/prediction.py \
  tests/test_prediction.py
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 commit -m "feat: add prediction dataset join"
```

## Task 2: Add Neighborhood And Structural-Control Predictors

**Files:**
- Modify: `spectral_research/prediction.py`
- Test: `tests/test_prediction.py`

- [ ] **Step 1: Extend the failing tests for neighborhood prediction**

```python
from alpha_cluster_dashboard.data import AlphaRecord
from spectral_research.prediction import (
    neighborhood_predict_oof,
    structural_baseline_oof,
)


class PredictionModelTests(unittest.TestCase):
    def _records(self) -> dict[str, AlphaRecord]:
        return {
            "m1": AlphaRecord("m1", "repo/m1", "/tmp/m1", 2, ("self_attn.q_proj", "mlp.down_proj"), np.array([[1.0, 2.0], [1.0, 2.0]])),
            "m2": AlphaRecord("m2", "repo/m2", "/tmp/m2", 2, ("self_attn.q_proj", "mlp.down_proj"), np.array([[1.1, 2.1], [1.1, 2.1]])),
            "m3": AlphaRecord("m3", "repo/m3", "/tmp/m3", 2, ("self_attn.q_proj", "mlp.down_proj"), np.array([[9.0, 9.0], [9.0, 9.0]])),
            "m4": AlphaRecord("m4", "repo/m4", "/tmp/m4", 2, ("self_attn.q_proj", "mlp.down_proj"), np.array([[9.2, 9.1], [9.1, 9.2]])),
        }

    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"model_id": "m1", "display_name": "repo/m1", "audited_base_family": "fam/a", "candidate_family": "fam/a", "label_status": "confirmed", "query_bucket": "supported_candidate", "num_layers": 2, "raw_module_count": 2, "common_schema": True, "schema_name": "common", "Average ⬆️": 10.0},
                {"model_id": "m2", "display_name": "repo/m2", "audited_base_family": "fam/a", "candidate_family": "fam/a", "label_status": "confirmed", "query_bucket": "supported_candidate", "num_layers": 2, "raw_module_count": 2, "common_schema": True, "schema_name": "common", "Average ⬆️": 12.0},
                {"model_id": "m3", "display_name": "repo/m3", "audited_base_family": "fam/b", "candidate_family": "fam/b", "label_status": "provisional", "query_bucket": "heterogeneous", "num_layers": 2, "raw_module_count": 2, "common_schema": False, "schema_name": "bert_like", "Average ⬆️": 90.0},
                {"model_id": "m4", "display_name": "repo/m4", "audited_base_family": "fam/b", "candidate_family": "fam/b", "label_status": "provisional", "query_bucket": "heterogeneous", "num_layers": 2, "raw_module_count": 2, "common_schema": False, "schema_name": "bert_like", "Average ⬆️": 92.0},
            ]
        )

    def test_neighborhood_predict_oof_returns_predictions_for_role_aggregated_trace(self) -> None:
        predictions = neighborhood_predict_oof(
            frame=self._frame(),
            records_by_model_id=self._records(),
            target="Average ⬆️",
            method="role_aggregated_trace",
            k=1,
            weighted=False,
            residualize=False,
        )
        self.assertEqual(len(predictions), 4)
        self.assertTrue(predictions["prediction"].notna().all())

    def test_structural_baseline_oof_returns_prediction_column(self) -> None:
        baseline = structural_baseline_oof(self._frame(), target="Average ⬆️")
        self.assertEqual(len(baseline), 4)
        self.assertIn("baseline_prediction", baseline.columns)
```

- [ ] **Step 2: Run the model tests to verify they fail**

Run: `conda run -n esd_ind python -m unittest tests.test_prediction.PredictionModelTests -v`

Expected: `FAIL` with missing `neighborhood_predict_oof` / `structural_baseline_oof`

- [ ] **Step 3: Implement the minimal prediction helpers**

```python
# spectral_research/prediction.py
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr


def _negative_l2(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if not mask.any():
        return float("nan")
    delta = left[mask] - right[mask]
    return float(-np.mean(delta * delta))


def _pair_score(
    *,
    method: str,
    left_row: pd.Series,
    right_row: pd.Series,
    left_record: AlphaRecord | None,
    right_record: AlphaRecord | None,
    structural_vectors: dict[str, np.ndarray],
) -> float:
    if method == "structural_basics_only":
        return _negative_l2(structural_vectors[left_row["model_id"]], structural_vectors[right_row["model_id"]])
    if method == "pl_alpha_scalar":
        return _negative_l2(
            alpha_scalar_features(left_record.alpha).reshape(-1),
            alpha_scalar_features(right_record.alpha).reshape(-1),
        )
    if method == "full_module_trace":
        if tuple(left_record.raw_modules) != tuple(right_record.raw_modules):
            return float("nan")
        depth = min(left_record.alpha.shape[0], right_record.alpha.shape[0])
        return _negative_l2(left_record.alpha[:depth], right_record.alpha[:depth])
    if method == "role_aggregated_trace":
        return role_aggregated_trace_score(
            query_alpha=left_record.alpha,
            query_modules=tuple(left_record.raw_modules),
            proto_alpha=right_record.alpha,
            proto_modules=tuple(right_record.raw_modules),
            overlap_penalty=0.2,
        )
    raise KeyError(method)


def structural_baseline_oof(frame: pd.DataFrame, *, target: str, cv_splits: int = 5) -> pd.DataFrame:
    vectors = structural_basics_feature_map(frame)
    groups = frame["cv_group"].astype(str).to_numpy()
    y = frame[target].to_numpy(dtype=float)
    splitter = GroupKFold(n_splits=min(cv_splits, pd.Series(groups).nunique()))
    out = frame[["model_id", "query_bucket", target]].copy()
    out["baseline_prediction"] = np.nan
    for train_idx, test_idx in splitter.split(frame, y, groups):
        x_train = np.vstack([vectors[frame.iloc[i]["model_id"]] for i in train_idx])
        x_test = np.vstack([vectors[frame.iloc[i]["model_id"]] for i in test_idx])
        model = Ridge(alpha=1.0)
        model.fit(x_train, y[train_idx])
        out.iloc[test_idx, out.columns.get_loc("baseline_prediction")] = model.predict(x_test)
    return out


def neighborhood_predict_oof(
    *,
    frame: pd.DataFrame,
    records_by_model_id: dict[str, AlphaRecord],
    target: str,
    method: str,
    k: int,
    weighted: bool,
    residualize: bool,
    cv_splits: int = 5,
) -> pd.DataFrame:
    structural_vectors = structural_basics_feature_map(frame)
    baseline = structural_baseline_oof(frame, target=target, cv_splits=cv_splits)
    working_target = (
        frame[target].to_numpy(dtype=float) - baseline["baseline_prediction"].to_numpy(dtype=float)
        if residualize
        else frame[target].to_numpy(dtype=float)
    )
    groups = frame["cv_group"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=min(cv_splits, pd.Series(groups).nunique()))
    out = frame[["model_id", "query_bucket", target]].copy()
    out["prediction"] = np.nan
    out["mode"] = "spectral_plus_structural" if residualize else "raw_spectral"
    for train_idx, test_idx in splitter.split(frame, working_target, groups):
        train = frame.iloc[train_idx].reset_index(drop=True)
        for held_out in test_idx:
            row = frame.iloc[held_out]
            scored = []
            for _, train_row in train.iterrows():
                score = _pair_score(
                    method=method,
                    left_row=row,
                    right_row=train_row,
                    left_record=records_by_model_id.get(str(row["model_id"])),
                    right_record=records_by_model_id.get(str(train_row["model_id"])),
                    structural_vectors=structural_vectors,
                )
                if np.isfinite(score):
                    scored.append((float(score), float(working_target[frame.index.get_loc(train_row.name) if False else train_idx[train.index[train["model_id"] == train_row["model_id"]][0]]])))
            scored.sort(key=lambda item: item[0], reverse=True)
            top = scored[:k]
            if not top:
                continue
            if weighted:
                raw = np.asarray([score for score, _ in top], dtype=float)
                weights = np.exp(raw - raw.max())
                pred = float(np.average([value for _, value in top], weights=weights))
            else:
                pred = float(np.mean([value for _, value in top]))
            if residualize:
                pred += float(baseline.loc[baseline["model_id"].eq(row["model_id"]), "baseline_prediction"].iloc[0])
            out.loc[out["model_id"].eq(row["model_id"]), "prediction"] = pred
    return out
```

- [ ] **Step 4: Add result aggregation helpers in the same module**

```python
def summarize_predictions(predictions: pd.DataFrame, *, target: str, method: str, k: int, weighted: bool) -> pd.DataFrame:
    rows = []
    for bucket, bucket_frame in predictions.groupby("query_bucket", dropna=False):
        valid = bucket_frame["prediction"].notna()
        if not valid.any():
            continue
        y_true = bucket_frame.loc[valid, target].to_numpy(dtype=float)
        y_pred = bucket_frame.loc[valid, "prediction"].to_numpy(dtype=float)
        rho, _ = spearmanr(y_true, y_pred)
        rows.append(
            {
                "target": target,
                "method": method,
                "mode": bucket_frame["mode"].iloc[0],
                "bucket": bucket,
                "k": k,
                "weighted": weighted,
                "n": int(valid.sum()),
                "r2": float(r2_score(y_true, y_pred)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "spearman": float(rho),
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 5: Run the prediction tests to verify they pass**

Run: `conda run -n esd_ind python -m unittest tests.test_prediction -v`

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 add \
  spectral_research/prediction.py \
  tests/test_prediction.py
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 commit -m "feat: add neighborhood prediction pilot"
```

## Task 3: Repair The Prediction CLI And Reporting

**Files:**
- Modify: `scripts/spectral_prediction_research.py`
- Test: `tests/test_prediction.py`

- [ ] **Step 1: Add failing CLI/report tests**

```python
import importlib.util


class PredictionCliTests(unittest.TestCase):
    def test_prediction_cli_accepts_freeze_csv_and_targets(self) -> None:
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "spectral_prediction_research.py"
        spec = importlib.util.spec_from_file_location("spectral_prediction_research_cli", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        args = module.parse_args(
            [
                "--freeze-csv",
                "freeze.csv",
                "--targets",
                "Average ⬆️,MMLU-PRO,MATH Lvl 5",
                "--output-dir",
                "out",
            ]
        )

        self.assertEqual(args.freeze_csv, Path("freeze.csv"))
        self.assertEqual(args.output_dir, Path("out"))
```

- [ ] **Step 2: Run the CLI test to verify it fails**

Run: `conda run -n esd_ind python -m unittest tests.test_prediction.PredictionCliTests -v`

Expected: `FAIL` because `parse_args` does not accept `--freeze-csv`

- [ ] **Step 3: Repair the script around the new module**

```python
# scripts/spectral_prediction_research.py
from spectral_research.prediction import (
    PERFORMANCE_TARGETS,
    build_prediction_dataset,
    neighborhood_predict_oof,
    summarize_predictions,
    structural_baseline_oof,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run performance prediction cycle 1.")
    parser.add_argument("--metrics-dir", type=Path, default=REPO_ROOT / "svd_results")
    parser.add_argument("--leaderboard-csv", type=Path, default=REPO_ROOT / "data" / "sampled_leaderboard.csv")
    parser.add_argument("--lineage-csv", type=Path, default=REPO_ROOT / "data" / "sampled_metadata.csv")
    parser.add_argument("--freeze-csv", type=Path, default=REPO_ROOT / "data" / "manual_audits" / "spectral_family_retrieval_label_freeze.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "research_runs" / "performance_prediction_cycle1")
    parser.add_argument("--targets", type=str, default=",".join(PERFORMANCE_TARGETS))
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    dataset, records_by_model_id = build_prediction_dataset(
        metrics_dir=args.metrics_dir,
        leaderboard_csv=args.leaderboard_csv,
        lineage_csv=args.lineage_csv,
        freeze_csv=args.freeze_csv,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output_dir / "prediction_dataset.csv", index=False)

    result_frames = []
    oof_frames = []
    for target in targets:
        baseline = structural_baseline_oof(dataset, target=target)
        baseline = baseline.assign(target=target, method="structural_basics_only", mode="baseline", k=0, weighted=False)
        oof_frames.append(baseline)
        result_frames.append(summarize_predictions(baseline.rename(columns={"baseline_prediction": "prediction"}), target=target, method="structural_basics_only", k=0, weighted=False))
        for method in ("pl_alpha_scalar", "full_module_trace", "role_aggregated_trace"):
            for k in (3, 5):
                for weighted in (False, True):
                    raw = neighborhood_predict_oof(frame=dataset, records_by_model_id=records_by_model_id, target=target, method=method, k=k, weighted=weighted, residualize=False)
                    ctl = neighborhood_predict_oof(frame=dataset, records_by_model_id=records_by_model_id, target=target, method=method, k=k, weighted=weighted, residualize=True)
                    raw = raw.assign(target=target, method=method, k=k, weighted=weighted)
                    ctl = ctl.assign(target=target, method=method, k=k, weighted=weighted)
                    oof_frames.extend([raw, ctl])
                    result_frames.extend([
                        summarize_predictions(raw, target=target, method=method, k=k, weighted=weighted),
                        summarize_predictions(ctl, target=target, method=method, k=k, weighted=weighted),
                    ])
    results = pd.concat(result_frames, ignore_index=True)
    predictions = pd.concat(oof_frames, ignore_index=True)
    results.to_csv(args.output_dir / "prediction_results.csv", index=False)
    predictions.to_csv(args.output_dir / "oof_predictions.csv", index=False)
```

- [ ] **Step 4: Add a small report renderer and save the report**

```python
def render_report(results: pd.DataFrame) -> str:
    best = results.sort_values(["target", "r2", "spearman"], ascending=[True, False, False]).groupby("target", as_index=False).head(3)
    return "\n".join(
        [
            "# Performance Prediction Cycle 1",
            "",
            "## Best rows by target",
            best.to_markdown(index=False),
            "",
            "## Notes",
            "- Raw spectral rows measure direct neighborhood signal.",
            "- Controlled rows add the structural baseline back after residual neighborhood prediction.",
        ]
    ) + "\n"


(args.output_dir / "report.md").write_text(render_report(results), encoding="utf-8")
```

- [ ] **Step 5: Run the CLI test to verify it passes**

Run: `conda run -n esd_ind python -m unittest tests.test_prediction.PredictionCliTests -v`

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 add \
  scripts/spectral_prediction_research.py \
  tests/test_prediction.py \
  spectral_research/prediction.py
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 commit -m "feat: add prediction research runner"
```

## Task 4: Run The Pilot And Update The Findings Note

**Files:**
- Modify: `docs/superpowers/specs/2026-04-18-spectral-family-retrieval-preliminary-findings.md`
- Modify: `research_runs/performance_prediction_cycle1/*` (generated outputs)

- [ ] **Step 1: Run the focused test suite**

Run: `conda run -n esd_ind python -m unittest tests.test_prediction tests.test_pipeline tests.test_heterogeneity -v`

Expected: `OK`

- [ ] **Step 2: Run the prediction pilot**

Run:

```bash
conda run -n esd_ind python scripts/spectral_prediction_research.py \
  --output-dir research_runs/performance_prediction_cycle1 \
  --targets "Average ⬆️,MMLU-PRO,MATH Lvl 5"
```

Expected:

- `research_runs/performance_prediction_cycle1/prediction_dataset.csv`
- `research_runs/performance_prediction_cycle1/prediction_results.csv`
- `research_runs/performance_prediction_cycle1/oof_predictions.csv`
- `research_runs/performance_prediction_cycle1/report.md`

- [ ] **Step 3: Summarize the outcome in the findings note**

```markdown
## Performance Prediction Cycle 1

We then ran a small performance-prediction pilot on the joined scored subset.

- joined scored models: `388`
- targets: `Average ⬆️`, `MMLU-PRO`, `MATH Lvl 5`
- main method family: neighborhood prediction from structural and spectral similarities
- controls: structural baseline and structurally controlled spectral prediction

The question was not whether prediction is solved.
The question was whether the current zoo representation carries any useful predictive signal at all.

[Insert the best observed rows from `prediction_results.csv` here.]

Interpretation:

- if the spectral methods clearly beat the structural control on at least one target, keep prediction as an active research direction
- if they do not, keep the pilot as a useful negative result and do not force prediction into the paper
```

- [ ] **Step 4: Verify the report and results files exist**

Run:

```bash
test -f /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_prediction_cycle1/report.md
test -f /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_prediction_cycle1/prediction_results.csv
test -f /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_prediction_cycle1/oof_predictions.csv
```

Expected: exit code `0`

- [ ] **Step 5: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 add \
  docs/superpowers/specs/2026-04-18-spectral-family-retrieval-preliminary-findings.md \
  research_runs/performance_prediction_cycle1 \
  scripts/spectral_prediction_research.py \
  spectral_research/prediction.py \
  tests/test_prediction.py
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 commit -m "run performance prediction cycle 1"
```

## Self-Review

- Spec coverage:
  - joined scored dataset: Task 1
  - neighborhood prediction main path: Task 2
  - with/without structural controls: Task 2
  - repaired CLI and outputs: Task 3
  - findings consolidation after run: Task 4
- Placeholder scan:
  - no `TODO`, `TBD`, or undefined file paths remain
- Type consistency:
  - `build_prediction_dataset`, `structural_baseline_oof`, `neighborhood_predict_oof`, and `summarize_predictions` are used consistently across tasks
