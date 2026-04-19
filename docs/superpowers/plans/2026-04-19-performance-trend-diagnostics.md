# Performance Trend Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one small diagnostic pipeline that measures whether simple absolute and lineage-relative spectral trend summaries correlate with benchmark performance on the existing scored mixed-zoo dataset.

**Architecture:** Reuse the current prediction dataset join from `spectral_research.prediction`, add a focused `spectral_research.trend_diagnostics` module for feature/delta table construction plus tiny correlation/regression utilities, then expose the cycle through one dedicated CLI script that writes compact CSVs and a short markdown report. Keep exact-parent and family-relative analyses separate all the way through the outputs.

**Tech Stack:** Python, pandas, numpy, scipy, scikit-learn, unittest/pytest, existing `esd_ind` conda env

---

## File Structure

- Create: `spectral_research/trend_diagnostics.py`
  - Responsibility: build absolute trend features from scored spectral records, construct exact-parent and family-relative delta tables, run compact trend statistics, and return report-ready tables.
- Create: `scripts/performance_trend_diagnostics.py`
  - Responsibility: CLI entrypoint that loads the joined scored dataset, runs the three diagnostics, writes CSV artifacts, and renders a short markdown report.
- Create: `tests/test_trend_diagnostics.py`
  - Responsibility: unit tests for feature extraction, exact-parent pairing, family-relative anchors, and compact stats behavior.
- Modify: `docs/superpowers/specs/2026-04-18-spectral-family-retrieval-preliminary-findings.md`
  - Responsibility: add one short section summarizing the trend-diagnostics outcome after the run is complete.

The plan deliberately does **not** modify the current predictor stack or prediction-cycle scripts.

### Task 1: Add Core Trend-Diagnostics Tables

**Files:**
- Create: `spectral_research/trend_diagnostics.py`
- Create: `tests/test_trend_diagnostics.py`

- [ ] **Step 1: Write the failing table-construction tests**

Add these tests to `tests/test_trend_diagnostics.py`:

```python
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from alpha_cluster_dashboard.data import AlphaRecord
from spectral_research.trend_diagnostics import (
    build_absolute_trend_frame,
    build_exact_parent_delta_frame,
    build_family_relative_delta_frame,
)


class TrendDiagnosticsTableTests(unittest.TestCase):
    def _records(self) -> dict[str, AlphaRecord]:
        return {
            "parent-a": AlphaRecord(
                "parent-a",
                "repo/parent-a",
                "/tmp/parent-a",
                2,
                ("self_attn.q_proj", "mlp.down_proj"),
                np.array([[1.0, 2.0], [2.0, 3.0]], dtype=float),
            ),
            "child-a": AlphaRecord(
                "child-a",
                "repo/child-a",
                "/tmp/child-a",
                2,
                ("self_attn.q_proj", "mlp.down_proj"),
                np.array([[2.0, 3.0], [4.0, 5.0]], dtype=float),
            ),
            "peer-b": AlphaRecord(
                "peer-b",
                "repo/peer-b",
                "/tmp/peer-b",
                2,
                ("self_attn.q_proj", "mlp.down_proj"),
                np.array([[0.5, 1.5], [1.5, 2.5]], dtype=float),
            ),
        }

    def _dataset(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "model_id": "parent-a",
                    "display_name": "repo/parent-a",
                    "source_model": "",
                    "base_model_relation": "",
                    "audited_base_family": "fam/a",
                    "candidate_family": "fam/a",
                    "query_bucket": "supported_candidate",
                    "common_schema": True,
                    "schema_name": "common_llm_7",
                    "num_layers": 2,
                    "raw_module_count": 2,
                    "Average ⬆️": 10.0,
                    "MMLU-PRO": 11.0,
                    "MATH Lvl 5": 12.0,
                },
                {
                    "model_id": "child-a",
                    "display_name": "repo/child-a",
                    "source_model": "parent-a",
                    "base_model_relation": "finetune",
                    "audited_base_family": "fam/a",
                    "candidate_family": "fam/a",
                    "query_bucket": "supported_candidate",
                    "common_schema": True,
                    "schema_name": "common_llm_7",
                    "num_layers": 2,
                    "raw_module_count": 2,
                    "Average ⬆️": 15.0,
                    "MMLU-PRO": 16.0,
                    "MATH Lvl 5": 20.0,
                },
                {
                    "model_id": "peer-b",
                    "display_name": "repo/peer-b",
                    "source_model": "",
                    "base_model_relation": "",
                    "audited_base_family": "fam/a",
                    "candidate_family": "fam/a",
                    "query_bucket": "heterogeneous",
                    "common_schema": False,
                    "schema_name": "bert_like_5",
                    "num_layers": 2,
                    "raw_module_count": 2,
                    "Average ⬆️": 7.0,
                    "MMLU-PRO": 8.0,
                    "MATH Lvl 5": 6.0,
                },
            ]
        )

    def test_build_absolute_trend_frame_adds_global_level_start_end_and_delta(self) -> None:
        feature_frame = build_absolute_trend_frame(self._dataset(), self._records())

        row = feature_frame.loc[feature_frame["model_id"].eq("child-a")].iloc[0]
        self.assertAlmostEqual(float(row["trend_global_mean"]), 3.5)
        self.assertAlmostEqual(float(row["trend_global_start"]), 2.5)
        self.assertAlmostEqual(float(row["trend_global_end"]), 4.5)
        self.assertAlmostEqual(float(row["trend_global_end_minus_start"]), 2.0)

    def test_build_exact_parent_delta_frame_uses_direct_source_model_when_present(self) -> None:
        feature_frame = build_absolute_trend_frame(self._dataset(), self._records())
        delta_frame = build_exact_parent_delta_frame(feature_frame)

        self.assertEqual(len(delta_frame), 1)
        row = delta_frame.iloc[0]
        self.assertEqual(row["model_id"], "child-a")
        self.assertEqual(row["parent_model_id"], "parent-a")
        self.assertAlmostEqual(float(row["trend_global_mean_delta"]), 1.5)
        self.assertAlmostEqual(float(row["MMLU-PRO_delta"]), 5.0)

    def test_build_family_relative_delta_frame_uses_family_median_anchor(self) -> None:
        feature_frame = build_absolute_trend_frame(self._dataset(), self._records())
        delta_frame = build_family_relative_delta_frame(feature_frame)

        row = delta_frame.loc[delta_frame["model_id"].eq("child-a")].iloc[0]
        self.assertEqual(row["family_anchor"], "fam/a")
        self.assertAlmostEqual(float(row["MMLU-PRO_delta"]), 5.0)
        self.assertIn("trend_global_end_minus_start_delta", delta_frame.columns)
```

- [ ] **Step 2: Run the new tests to confirm the missing module/functions fail**

Run:

```bash
cd /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2
conda run -n esd_ind python -m unittest tests.test_trend_diagnostics.TrendDiagnosticsTableTests -v
```

Expected: FAIL with `ModuleNotFoundError` for `spectral_research.trend_diagnostics` or missing-function import errors.

- [ ] **Step 3: Write the minimal trend-table implementation**

Create `spectral_research/trend_diagnostics.py` with these core pieces:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_cluster_dashboard.data import AlphaRecord

PERFORMANCE_TARGETS = ("Average ⬆️", "MMLU-PRO", "MATH Lvl 5")
STRUCTURAL_CONTROL_COLUMNS = ("num_layers", "raw_module_count", "common_schema", "schema_name")
TREND_FEATURE_COLUMNS = (
    "trend_global_mean",
    "trend_global_start",
    "trend_global_end",
    "trend_global_end_minus_start",
)


def _safe_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _record_trend_features(record: AlphaRecord) -> dict[str, float]:
    alpha = np.asarray(record.alpha, dtype=float)
    alpha[~np.isfinite(alpha)] = np.nan
    if alpha.size == 0:
        return {column: float("nan") for column in TREND_FEATURE_COLUMNS}

    start_level = _safe_mean(alpha[0])
    end_level = _safe_mean(alpha[-1])
    global_mean = _safe_mean(alpha.reshape(-1))
    return {
        "trend_global_mean": global_mean,
        "trend_global_start": start_level,
        "trend_global_end": end_level,
        "trend_global_end_minus_start": end_level - start_level
        if np.isfinite(start_level) and np.isfinite(end_level)
        else float("nan"),
    }


def build_absolute_trend_frame(
    dataset: pd.DataFrame,
    records_by_model_id: dict[str, AlphaRecord],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in dataset.to_dict(orient="records"):
        model_id = str(row["model_id"])
        record = records_by_model_id.get(model_id)
        if record is None:
            continue
        rows.append({**row, **_record_trend_features(record)})
    return pd.DataFrame(rows)


def build_exact_parent_delta_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    by_model_id = feature_frame.set_index("model_id", drop=False)
    rows: list[dict[str, object]] = []
    for row in feature_frame.to_dict(orient="records"):
        source_model = str(row.get("source_model", "") or "").strip()
        if not source_model or source_model not in by_model_id.index:
            continue
        parent = by_model_id.loc[source_model]
        delta_row = {**row, "parent_model_id": source_model}
        for column in TREND_FEATURE_COLUMNS:
            delta_row[f"{column}_delta"] = float(row[column]) - float(parent[column])
        for target in PERFORMANCE_TARGETS:
            delta_row[f"{target}_delta"] = float(row[target]) - float(parent[target])
        rows.append(delta_row)
    return pd.DataFrame(rows)


def build_family_relative_delta_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    anchor_key = feature_frame["audited_base_family"].fillna("").replace("", pd.NA)
    anchored = feature_frame.assign(_family_anchor=anchor_key.fillna(feature_frame["candidate_family"]))
    rows: list[dict[str, object]] = []
    for family, family_frame in anchored.groupby("_family_anchor", dropna=False):
        if pd.isna(family) or len(family_frame) < 2:
            continue
        trend_anchor = family_frame.loc[:, list(TREND_FEATURE_COLUMNS)].median(numeric_only=True)
        target_anchor = family_frame.loc[:, list(PERFORMANCE_TARGETS)].median(numeric_only=True)
        for row in family_frame.to_dict(orient="records"):
            delta_row = {**row, "family_anchor": family}
            for column in TREND_FEATURE_COLUMNS:
                delta_row[f"{column}_delta"] = float(row[column]) - float(trend_anchor[column])
            for target in PERFORMANCE_TARGETS:
                delta_row[f"{target}_delta"] = float(row[target]) - float(target_anchor[target])
            rows.append(delta_row)
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Re-run the table tests and confirm they pass**

Run:

```bash
cd /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2
conda run -n esd_ind python -m unittest tests.test_trend_diagnostics.TrendDiagnosticsTableTests -v
```

Expected: PASS for all 3 tests.

- [ ] **Step 5: Commit the core tables**

```bash
cd /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2
git add spectral_research/trend_diagnostics.py tests/test_trend_diagnostics.py
git commit -m "feat: add performance trend diagnostic tables"
```

### Task 2: Add Compact Statistics And CLI Runner

**Files:**
- Modify: `spectral_research/trend_diagnostics.py`
- Create: `scripts/performance_trend_diagnostics.py`
- Modify: `tests/test_trend_diagnostics.py`

- [ ] **Step 1: Add failing tests for statistics and report generation**

Append these tests to `tests/test_trend_diagnostics.py`:

```python
from spectral_research.trend_diagnostics import (
    analyze_trend_rows,
    render_trend_report,
)


class TrendDiagnosticsAnalysisTests(unittest.TestCase):
    def test_analyze_trend_rows_returns_spearman_and_r2(self) -> None:
        frame = pd.DataFrame(
            {
                "model_id": ["a", "b", "c", "d"],
                "query_bucket": ["supported_candidate", "supported_candidate", "heterogeneous", "heterogeneous"],
                "trend_global_end_minus_start": [0.0, 1.0, 2.0, 3.0],
                "MMLU-PRO": [10.0, 20.0, 30.0, 40.0],
                "num_layers": [2, 2, 2, 2],
                "raw_module_count": [2, 2, 2, 2],
                "common_schema": [True, True, False, False],
                "schema_name": ["common", "common", "bert_like", "bert_like"],
            }
        )

        results = analyze_trend_rows(
            frame,
            feature_columns=["trend_global_end_minus_start"],
            target_columns=["MMLU-PRO"],
            analysis_name="absolute",
            bucket_name="all_scored",
            use_structural_controls=False,
        )

        self.assertEqual(len(results), 1)
        row = results.iloc[0]
        self.assertEqual(row["analysis"], "absolute")
        self.assertEqual(row["bucket"], "all_scored")
        self.assertEqual(row["feature"], "trend_global_end_minus_start")
        self.assertAlmostEqual(float(row["spearman"]), 1.0)
        self.assertGreater(float(row["r2"]), 0.90)

    def test_render_trend_report_includes_three_sections(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "analysis": "absolute",
                    "bucket": "all_scored",
                    "target": "MMLU-PRO",
                    "feature": "trend_global_end_minus_start",
                    "control_mode": "raw",
                    "n": 12,
                    "spearman": 0.5,
                    "r2": 0.2,
                    "mae": 1.1,
                },
                {
                    "analysis": "exact_parent",
                    "bucket": "all_scored",
                    "target": "MMLU-PRO",
                    "feature": "trend_global_end_minus_start_delta",
                    "control_mode": "raw",
                    "n": 7,
                    "spearman": 0.4,
                    "r2": 0.1,
                    "mae": 1.2,
                },
                {
                    "analysis": "family_relative",
                    "bucket": "heterogeneous",
                    "target": "MATH Lvl 5",
                    "feature": "trend_global_end_minus_start_delta",
                    "control_mode": "raw",
                    "n": 9,
                    "spearman": 0.3,
                    "r2": 0.05,
                    "mae": 1.3,
                },
            ]
        )

        report = render_trend_report(rows)

        self.assertIn("# Performance Trend Diagnostics", report)
        self.assertIn("## Best Absolute-Trend Rows", report)
        self.assertIn("## Best Exact-Parent Rows", report)
        self.assertIn("## Best Family-Relative Rows", report)
```

- [ ] **Step 2: Run the tests and confirm they fail on missing analysis helpers**

Run:

```bash
cd /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2
conda run -n esd_ind python -m unittest tests.test_trend_diagnostics.TrendDiagnosticsAnalysisTests -v
```

Expected: FAIL with import errors for `analyze_trend_rows` and `render_trend_report`.

- [ ] **Step 3: Extend the trend module with tiny-fit stats and report rendering**

Add these functions to `spectral_research/trend_diagnostics.py`:

```python
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def _bucket_mask(frame: pd.DataFrame, bucket_name: str) -> pd.Series:
    if bucket_name == "all_scored":
        return pd.Series(True, index=frame.index)
    if bucket_name == "heterogeneous":
        return frame["query_bucket"].eq("heterogeneous")
    return frame["query_bucket"].eq(bucket_name)


def _structural_design(frame: pd.DataFrame) -> pd.DataFrame:
    design = pd.DataFrame(index=frame.index)
    design["num_layers"] = pd.to_numeric(frame["num_layers"], errors="coerce")
    design["raw_module_count"] = pd.to_numeric(frame["raw_module_count"], errors="coerce")
    design["common_schema"] = frame["common_schema"].astype(float)
    schema_dummies = pd.get_dummies(frame["schema_name"].fillna("unknown"), prefix="schema")
    return pd.concat([design, schema_dummies], axis=1).fillna(0.0)


def analyze_trend_rows(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_columns: list[str],
    analysis_name: str,
    bucket_name: str,
    use_structural_controls: bool,
) -> pd.DataFrame:
    bucket_frame = frame.loc[_bucket_mask(frame, bucket_name)].copy()
    rows: list[dict[str, object]] = []
    if bucket_frame.empty:
        return pd.DataFrame(columns=["analysis", "bucket", "target", "feature", "control_mode", "n", "spearman", "r2", "mae"])

    structural = _structural_design(bucket_frame) if use_structural_controls else None
    for feature in feature_columns:
        for target in target_columns:
            subset = bucket_frame.loc[:, [feature, target]].copy()
            if structural is not None:
                subset = subset.join(structural)
            subset = subset.dropna()
            if len(subset) < 4:
                continue

            x = subset[[feature]].to_numpy(dtype=float)
            y = subset[target].to_numpy(dtype=float)
            if use_structural_controls:
                controls = subset.drop(columns=[feature, target]).to_numpy(dtype=float)
                x = np.hstack([x, controls])

            model = LinearRegression()
            model.fit(x, y)
            predictions = model.predict(x)
            spearman = spearmanr(subset[feature].to_numpy(dtype=float), y, nan_policy="omit").statistic
            rows.append(
                {
                    "analysis": analysis_name,
                    "bucket": bucket_name,
                    "target": target,
                    "feature": feature,
                    "control_mode": "structural_adjusted" if use_structural_controls else "raw",
                    "n": int(len(subset)),
                    "spearman": float(spearman),
                    "r2": float(r2_score(y, predictions)),
                    "mae": float(mean_absolute_error(y, predictions)),
                }
            )
    return pd.DataFrame(rows)


def render_trend_report(results: pd.DataFrame) -> str:
    def best_rows(analysis: str) -> pd.DataFrame:
        subset = results.loc[results["analysis"].eq(analysis)].copy()
        if subset.empty:
            return subset
        return (
            subset.sort_values(["target", "r2", "spearman"], ascending=[True, False, False])
            .groupby("target", as_index=False)
            .head(3)
        )

    def markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_No rows._"
        columns = ["bucket", "target", "feature", "control_mode", "n", "spearman", "r2", "mae"]
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = []
        for row in frame.loc[:, columns].itertuples(index=False, name=None):
            body.append("| " + " | ".join(f\"{value:.3f}\" if isinstance(value, float) else str(value) for value in row) + " |")
        return "\n".join([header, separator, *body])

    return "\n".join(
        [
            "# Performance Trend Diagnostics",
            "",
            "## Best Absolute-Trend Rows",
            markdown_table(best_rows("absolute")),
            "",
            "## Best Exact-Parent Rows",
            markdown_table(best_rows("exact_parent")),
            "",
            "## Best Family-Relative Rows",
            markdown_table(best_rows("family_relative")),
            "",
            "## Notes",
            "- Exact-parent and family-relative evidence are reported separately on purpose.",
            "- Raw and lightweight structural-adjusted variants are both included when enough rows are available.",
        ]
    ) + "\n"
```

- [ ] **Step 4: Add the CLI runner**

Create `scripts/performance_trend_diagnostics.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_research.prediction import build_prediction_dataset, PERFORMANCE_TARGETS
from spectral_research.trend_diagnostics import (
    TREND_FEATURE_COLUMNS,
    analyze_trend_rows,
    build_absolute_trend_frame,
    build_exact_parent_delta_frame,
    build_family_relative_delta_frame,
    render_trend_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run performance trend diagnostics.")
    parser.add_argument("--metrics-dir", type=Path, default=REPO_ROOT / "svd_results")
    parser.add_argument("--leaderboard-csv", type=Path, default=REPO_ROOT / "data" / "sampled_leaderboard.csv")
    parser.add_argument("--lineage-csv", type=Path, default=REPO_ROOT / "data" / "sampled_metadata.csv")
    parser.add_argument(
        "--freeze-csv",
        type=Path,
        default=REPO_ROOT / "data" / "manual_audits" / "spectral_family_retrieval_label_freeze.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "research_runs" / "performance_trend_diagnostics",
    )
    return parser.parse_args(argv)


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset, records_by_model_id = build_prediction_dataset(
        metrics_dir=args.metrics_dir,
        leaderboard_csv=args.leaderboard_csv,
        lineage_csv=args.lineage_csv,
        freeze_csv=args.freeze_csv,
    )
    absolute = build_absolute_trend_frame(dataset, records_by_model_id)
    exact_parent = build_exact_parent_delta_frame(absolute)
    family_relative = build_family_relative_delta_frame(absolute)

    results = []
    for bucket in ("all_scored", "heterogeneous"):
        for controlled in (False, True):
            results.append(
                analyze_trend_rows(
                    absolute,
                    feature_columns=list(TREND_FEATURE_COLUMNS),
                    target_columns=list(PERFORMANCE_TARGETS),
                    analysis_name="absolute",
                    bucket_name=bucket,
                    use_structural_controls=controlled,
                )
            )
            results.append(
                analyze_trend_rows(
                    exact_parent,
                    feature_columns=[f"{column}_delta" for column in TREND_FEATURE_COLUMNS],
                    target_columns=[f"{target}_delta" for target in PERFORMANCE_TARGETS],
                    analysis_name="exact_parent",
                    bucket_name=bucket,
                    use_structural_controls=controlled,
                )
            )
            results.append(
                analyze_trend_rows(
                    family_relative,
                    feature_columns=[f"{column}_delta" for column in TREND_FEATURE_COLUMNS],
                    target_columns=[f"{target}_delta" for target in PERFORMANCE_TARGETS],
                    analysis_name="family_relative",
                    bucket_name=bucket,
                    use_structural_controls=controlled,
                )
            )

    result_frame = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write(absolute, args.output_dir / "absolute_trend_dataset.csv")
    _write(exact_parent, args.output_dir / "exact_parent_delta_dataset.csv")
    _write(family_relative, args.output_dir / "family_relative_delta_dataset.csv")
    _write(result_frame, args.output_dir / "trend_results.csv")
    (args.output_dir / "report.md").write_text(render_trend_report(result_frame), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Re-run focused tests and then the full prediction/trend test module**

Run:

```bash
cd /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2
conda run -n esd_ind python -m unittest tests.test_trend_diagnostics.TrendDiagnosticsAnalysisTests -v
conda run -n esd_ind python -m unittest tests.test_trend_diagnostics -v
```

Expected: PASS for the new analysis/report tests and the earlier table tests.

- [ ] **Step 6: Commit the analysis and CLI**

```bash
cd /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2
git add spectral_research/trend_diagnostics.py scripts/performance_trend_diagnostics.py tests/test_trend_diagnostics.py
git commit -m "feat: add performance trend diagnostics runner"
```

### Task 3: Run The Diagnostic Cycle And Update Findings

**Files:**
- Modify: `docs/superpowers/specs/2026-04-18-spectral-family-retrieval-preliminary-findings.md`
- Create: `research_runs/performance_trend_diagnostics/absolute_trend_dataset.csv`
- Create: `research_runs/performance_trend_diagnostics/exact_parent_delta_dataset.csv`
- Create: `research_runs/performance_trend_diagnostics/family_relative_delta_dataset.csv`
- Create: `research_runs/performance_trend_diagnostics/trend_results.csv`
- Create: `research_runs/performance_trend_diagnostics/report.md`

- [ ] **Step 1: Run the full relevant test suite before the experiment**

Run:

```bash
cd /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2
conda run -n esd_ind env PYTHONPATH=. pytest -q
```

Expected: PASS, with the current project-level pattern of a mostly green suite and only known skips if any.

- [ ] **Step 2: Execute the diagnostics**

Run:

```bash
cd /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2
conda run -n esd_ind python scripts/performance_trend_diagnostics.py \
  --output-dir research_runs/performance_trend_diagnostics
```

Expected: exit code `0` and the five output files written under `research_runs/performance_trend_diagnostics/`.

- [ ] **Step 3: Inspect the strongest rows before writing the findings note**

Run:

```bash
cd /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2
python - <<'PY'
import pandas as pd
results = pd.read_csv('research_runs/performance_trend_diagnostics/trend_results.csv')
for analysis in ('absolute', 'exact_parent', 'family_relative'):
    subset = results[results['analysis'].eq(analysis)]
    if subset.empty:
        print(f'[{analysis}] no rows')
        continue
    best = subset.sort_values(['target', 'r2', 'spearman'], ascending=[True, False, False]).groupby('target').head(3)
    print(f'[{analysis}]')
    print(best[['bucket', 'target', 'feature', 'control_mode', 'n', 'spearman', 'r2', 'mae']].to_string(index=False))
    print()
PY
```

Expected: a short printed summary of the top rows for each of the three analyses.

- [ ] **Step 4: Update the consolidated findings note with one small trend-diagnostics section**

Append a short section to `docs/superpowers/specs/2026-04-18-spectral-family-retrieval-preliminary-findings.md` in this style:

```md
## Performance Trend Diagnostics

We then ran a small diagnostic cycle on simple spectral trend summaries rather than another larger prediction stack.

Outputs:

- [absolute_trend_dataset.csv](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_trend_diagnostics/absolute_trend_dataset.csv)
- [exact_parent_delta_dataset.csv](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_trend_diagnostics/exact_parent_delta_dataset.csv)
- [family_relative_delta_dataset.csv](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_trend_diagnostics/family_relative_delta_dataset.csv)
- [trend_results.csv](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_trend_diagnostics/trend_results.csv)
- [report.md](/scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2/research_runs/performance_trend_diagnostics/report.md)

Interpretation:

- summarize the strongest absolute-trend result on `MMLU-PRO` or `MATH Lvl 5`
- summarize whether exact-parent deltas had enough clean support to matter
- summarize whether family-relative deltas look broad but noisy or genuinely promising
- state clearly whether trend-style signals are worth carrying into later prediction work
```

Keep this section short and factual.

- [ ] **Step 5: Re-run the report-producing command if the note update required any code tweak, otherwise verify files exist**

Run:

```bash
cd /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2
test -f research_runs/performance_trend_diagnostics/report.md
test -f research_runs/performance_trend_diagnostics/trend_results.csv
```

Expected: both shell checks succeed with exit code `0`.

- [ ] **Step 6: Commit the run artifacts and findings update**

```bash
cd /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2
git add \
  docs/superpowers/specs/2026-04-18-spectral-family-retrieval-preliminary-findings.md \
  research_runs/performance_trend_diagnostics \
  && git commit -m "run performance trend diagnostics"
```

## Self-Review

Spec coverage check:
- `absolute trend` diagnostic: covered in Task 1 feature extraction and Task 2 absolute analysis.
- `exact-parent trend` diagnostic: covered in Task 1 delta table construction and Task 2 exact-parent analysis.
- `family-relative trend` diagnostic: covered in Task 1 family anchor table and Task 2 family-relative analysis.
- same three targets: wired directly in Task 2 CLI through `PERFORMANCE_TARGETS`.
- compact outputs/report only: enforced in Task 2 report renderer and Task 3 artifact set.
- no predictor-stack expansion: no task touches `neighborhood_predict_oof` or current prediction methods.

Placeholder scan:
- no `TODO`, `TBD`, or “similar to Task N” shortcuts.
- all tasks include concrete files, commands, and code snippets.

Type consistency check:
- trend feature names are consistent across tasks:
  - absolute: `trend_global_mean`, `trend_global_start`, `trend_global_end`, `trend_global_end_minus_start`
  - relative: same names with `_delta`
- analysis names are consistent across tasks:
  - `absolute`
  - `exact_parent`
  - `family_relative`
- output paths are consistent throughout:
  - `research_runs/performance_trend_diagnostics/...`
