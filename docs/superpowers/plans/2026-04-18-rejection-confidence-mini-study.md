# Rejection Confidence Mini-Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a very small follow-up study that compares simple confidence rules for rejection on the existing strict family-retrieval benchmark.

**Architecture:** Keep the benchmark and fingerprint methods fixed. Add one focused helper module for rejection-confidence rules plus one narrow script that rebuilds the strict benchmark state, computes `top_score`, `margin`, and `score_plus_margin`, and writes comparison tables. Do not change the main retrieval pipeline or add new fingerprints.

**Tech Stack:** Python 3.10, NumPy, pandas, built-in `unittest`, existing `spectral_research.pipeline` internals.

---

## File Structure

**Create:**

- `spectral_research/rejection.py`
  - confidence-rule helpers and fixed-coverage precision helpers
- `tests/test_rejection.py`
  - TDD coverage for the helper module
- `scripts/rejection_confidence_study.py`
  - one-off script for the mini-study

**Modify:**

- `tests/test_pipeline.py`
  - add one small CLI test for the new script if needed

**Outputs produced at runtime:**

- `research_runs/rejection_confidence_study_strict_v1/confidence_rule_metrics.csv`
- `research_runs/rejection_confidence_study_strict_v1/best_rule_bucket_breakdown.csv`
- `research_runs/rejection_confidence_study_strict_v1/report.md`

## Task 1: Add Confidence-Rule Helpers

**Files:**

- Create: `spectral_research/rejection.py`
- Create: `tests/test_rejection.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rejection.py
from __future__ import annotations

import unittest

import numpy as np

from spectral_research.rejection import (
    confidence_from_ranked_scores,
    precision_at_coverages,
)


class RejectionRuleTests(unittest.TestCase):
    def test_confidence_from_ranked_scores_supports_top_score_and_margin(self) -> None:
        scores = np.array([0.8, 0.5, 0.1], dtype=float)

        self.assertAlmostEqual(
            confidence_from_ranked_scores(scores, rule="top_score"),
            0.8,
        )
        self.assertAlmostEqual(
            confidence_from_ranked_scores(scores, rule="margin"),
            0.3,
        )

    def test_score_plus_margin_uses_simple_normalized_combination(self) -> None:
        scores = np.array([0.8, 0.5, 0.1], dtype=float)

        confidence = confidence_from_ranked_scores(
            scores,
            rule="score_plus_margin",
            score_center=0.6,
            score_scale=0.1,
            margin_center=0.2,
            margin_scale=0.1,
        )

        self.assertAlmostEqual(confidence, 1.5)

    def test_precision_at_coverages_reports_requested_coverages(self) -> None:
        confidence = np.array([0.9, 0.8, 0.4, 0.1], dtype=float)
        correct = np.array([1, 1, 0, 0], dtype=bool)

        frame = precision_at_coverages(confidence, correct, coverages=(0.25, 0.5, 1.0))

        self.assertEqual(list(frame["coverage"]), [0.25, 0.5, 1.0])
        self.assertAlmostEqual(frame.loc[frame["coverage"] == 0.25, "accepted_precision"].iloc[0], 1.0)
        self.assertAlmostEqual(frame.loc[frame["coverage"] == 0.5, "accepted_precision"].iloc[0], 1.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n esd_ind python -m unittest tests.test_rejection -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'spectral_research.rejection'`

- [ ] **Step 3: Write the minimal helper module**

```python
# spectral_research/rejection.py
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def confidence_from_ranked_scores(
    scores: np.ndarray,
    rule: str,
    *,
    score_center: float = 0.0,
    score_scale: float = 1.0,
    margin_center: float = 0.0,
    margin_scale: float = 1.0,
) -> float:
    score_array = np.asarray(scores, dtype=float)
    if score_array.size == 0:
        return float("nan")

    top_score = float(score_array[0])
    second_score = float(score_array[1]) if score_array.size > 1 else float("-inf")
    margin = top_score - second_score if np.isfinite(second_score) else float("nan")

    if rule == "top_score":
        return top_score
    if rule == "margin":
        return margin
    if rule == "score_plus_margin":
        safe_score_scale = score_scale if score_scale > 0 else 1.0
        safe_margin_scale = margin_scale if margin_scale > 0 else 1.0
        z_score = (top_score - score_center) / safe_score_scale
        z_margin = (margin - margin_center) / safe_margin_scale
        return 0.5 * (z_score + z_margin)
    raise KeyError(f"Unsupported confidence rule: {rule}")


def precision_at_coverages(
    confidence: np.ndarray,
    correct: np.ndarray,
    *,
    coverages: tuple[float, ...] = (0.1, 0.25, 0.5),
) -> pd.DataFrame:
    confidence_array = np.asarray(confidence, dtype=float)
    correct_array = np.asarray(correct, dtype=bool)
    if confidence_array.size == 0:
        return pd.DataFrame(columns=["coverage", "accepted_precision", "threshold", "accepted_count"])

    order = np.argsort(-confidence_array)
    sorted_confidence = confidence_array[order]
    sorted_correct = correct_array[order]
    total = len(sorted_confidence)

    rows: list[dict[str, float]] = []
    for coverage in coverages:
        accepted_count = max(1, int(math.ceil(total * coverage)))
        accepted = sorted_correct[:accepted_count]
        rows.append(
            {
                "coverage": float(coverage),
                "accepted_precision": float(np.mean(accepted)),
                "threshold": float(sorted_confidence[accepted_count - 1]),
                "accepted_count": int(accepted_count),
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n esd_ind python -m unittest tests.test_rejection -v`

Expected: PASS with `Ran 3 tests`

- [ ] **Step 5: Commit**

```bash
git add spectral_research/rejection.py tests/test_rejection.py
git commit -m "feat: add rejection confidence helpers"
```

## Task 2: Add The Narrow Rejection Study Script

**Files:**

- Create: `scripts/rejection_confidence_study.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Add the failing script test**

```python
    def test_rejection_confidence_study_cli_accepts_output_dir(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1] / "scripts" / "rejection_confidence_study.py"
        )
        spec = importlib.util.spec_from_file_location("rejection_confidence_study_cli", script_path)

        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        args = module.parse_args(["--output-dir", "out"])

        self.assertEqual(args.output_dir, Path("out"))
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `conda run -n esd_ind python -m unittest tests.test_pipeline.PipelineTests.test_rejection_confidence_study_cli_accepts_output_dir -v`

Expected: FAIL because `scripts/rejection_confidence_study.py` does not exist

- [ ] **Step 3: Write the study script**

```python
# scripts/rejection_confidence_study.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spectral_research.evaluation import open_set_metrics
from spectral_research.pipeline import (
    MATCH_SLICES,
    _calibration_metrics,
    _family_prototypes,
    _metadata_feature_map,
    _rank_query,
    _read_alpha_records,
    _read_csv,
    _record_lookup,
    build_audit_manifest,
    build_split_artifacts,
)
from spectral_research.benchmark import OVERRIDE_COLUMNS
from spectral_research.rejection import confidence_from_ranked_scores, precision_at_coverages


METHODS = ("full_module_trace", "attention_trace", "pl_alpha_scalar", "metadata_only")
RULES = ("top_score", "margin", "score_plus_margin")
FIXED_COVERAGES = (0.1, 0.25, 0.5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, default=Path("svd_results/metrics"))
    parser.add_argument("--lineage-csv", type=Path, default=Path("data/sampled_metadata.csv"))
    parser.add_argument("--leaderboard-csv", type=Path, default=Path("data/OpenLLMLeaderboard2.csv"))
    parser.add_argument(
        "--freeze-csv",
        type=Path,
        default=Path("data/manual_audits/spectral_family_retrieval_label_freeze.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research_runs/rejection_confidence_study_strict_v1"),
    )
    parser.add_argument("--min-family-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)
```

```python
def _collect_rank_rows(
    methods: tuple[str, ...],
    queries_supported: pd.DataFrame,
    queries_unsupported: pd.DataFrame,
    queries_heterogeneous: pd.DataFrame,
    prototypes: dict[str, object],
    records_by_model_id: dict[str, object],
    metadata_vectors: dict[str, np.ndarray],
) -> pd.DataFrame:
    supported = queries_supported[queries_supported["split_role"] == "test_supported"].copy()
    supported["supported_label"] = 1
    supported["bucket"] = "supported"

    unsupported = queries_unsupported.copy()
    unsupported["supported_label"] = 0
    unsupported["bucket"] = unsupported["query_bucket"].fillna("unsupported_unknown")

    heterogeneous = queries_heterogeneous.copy()
    heterogeneous["supported_label"] = 0
    heterogeneous["bucket"] = "heterogeneous"

    evaluation_rows = pd.concat([supported, unsupported, heterogeneous], ignore_index=True, sort=False)
    rows: list[dict[str, object]] = []

    for method in methods:
        for slice_name in MATCH_SLICES:
            for _, row in evaluation_rows.iterrows():
                ranked = _rank_query(
                    row,
                    method,
                    slice_name,
                    prototypes,
                    records_by_model_id,
                    metadata_vectors,
                )
                if not ranked:
                    continue
                families = [family for family, _ in ranked]
                scores = np.asarray([score for _, score in ranked], dtype=float)
                best_family = families[0]
                rows.append(
                    {
                        "method": method,
                        "slice": slice_name,
                        "model_id": row["model_id"],
                        "bucket": row["bucket"],
                        "supported_label": int(row["supported_label"]),
                        "truth_family": row.get("audited_base_family", ""),
                        "best_family": best_family,
                        "is_correct": int(row["supported_label"] == 1 and best_family == row["audited_base_family"]),
                        "top_score": confidence_from_ranked_scores(scores, rule="top_score"),
                        "margin": confidence_from_ranked_scores(scores, rule="margin"),
                    }
                )
    return pd.DataFrame(rows)
```

```python
def _add_score_plus_margin(rank_rows: pd.DataFrame) -> pd.DataFrame:
    output = rank_rows.copy()
    output["score_plus_margin"] = np.nan

    for (method, slice_name), group in output.groupby(["method", "slice"], sort=False):
        calibration = group[group["bucket"] == "supported"]
        score_center = float(calibration["top_score"].mean())
        score_scale = float(calibration["top_score"].std(ddof=0)) or 1.0
        margin_center = float(calibration["margin"].mean())
        margin_scale = float(calibration["margin"].std(ddof=0)) or 1.0

        output.loc[group.index, "score_plus_margin"] = [
            confidence_from_ranked_scores(
                np.asarray([top_score, top_score - margin], dtype=float),
                rule="score_plus_margin",
                score_center=score_center,
                score_scale=score_scale,
                margin_center=margin_center,
                margin_scale=margin_scale,
            )
            for top_score, margin in zip(group["top_score"], group["margin"], strict=False)
        ]

    return output
```

```python
def _metric_rows(rank_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    bucket_rows: list[dict[str, object]] = []

    best_rule_by_method: dict[str, str] = {}
    for method in METHODS:
        best_rule = None
        best_auroc = float("-inf")
        for rule in RULES:
            method_slice = rank_rows[(rank_rows["method"] == method) & (rank_rows["slice"] == "all")]
            confidence = method_slice[rule].to_numpy(dtype=float)
            labels = method_slice["supported_label"].to_numpy(dtype=int)
            correct = method_slice["is_correct"].to_numpy(dtype=bool)
            metrics = open_set_metrics(confidence, labels)
            fixed = precision_at_coverages(confidence, correct, coverages=FIXED_COVERAGES)
            metric_rows.append(
                {
                    "method": method,
                    "rule": rule,
                    "slice": "all",
                    "num_examples": int(len(method_slice)),
                    "auroc": metrics["auroc"],
                    "auprc": metrics["auprc"],
                    "coverage_10_precision": float(fixed.loc[fixed["coverage"] == 0.1, "accepted_precision"].iloc[0]),
                    "coverage_25_precision": float(fixed.loc[fixed["coverage"] == 0.25, "accepted_precision"].iloc[0]),
                    "coverage_50_precision": float(fixed.loc[fixed["coverage"] == 0.5, "accepted_precision"].iloc[0]),
                }
            )
            if metrics["auroc"] > best_auroc:
                best_auroc = float(metrics["auroc"])
                best_rule = rule
        best_rule_by_method[method] = best_rule or "top_score"

    for method in ("full_module_trace", "attention_trace"):
        best_rule = best_rule_by_method[method]
        subset = rank_rows[(rank_rows["method"] == method) & (rank_rows["slice"] == "all")]
        for bucket, group in subset.groupby("bucket", sort=True):
            confidence = group[best_rule].to_numpy(dtype=float)
            correct = group["is_correct"].to_numpy(dtype=bool)
            fixed = precision_at_coverages(confidence, correct, coverages=FIXED_COVERAGES)
            bucket_rows.append(
                {
                    "method": method,
                    "rule": best_rule,
                    "bucket": bucket,
                    "num_examples": int(len(group)),
                    "coverage_10_precision": float(fixed.loc[fixed["coverage"] == 0.1, "accepted_precision"].iloc[0]),
                    "coverage_25_precision": float(fixed.loc[fixed["coverage"] == 0.25, "accepted_precision"].iloc[0]),
                    "coverage_50_precision": float(fixed.loc[fixed["coverage"] == 0.5, "accepted_precision"].iloc[0]),
                }
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(bucket_rows)
```

Implementation requirements:

- rebuild the same audited benchmark state as `spectral_similarity_research.py`
- use only `METHODS`
- compute ranked family scores for supported test queries plus unsupported queries
- derive:
  - `top_score`
  - `margin`
  - `score_plus_margin`
- estimate normalization for `score_plus_margin` from calibration-supported queries only
- write:
  - `confidence_rule_metrics.csv`
  - `best_rule_bucket_breakdown.csv`
  - `report.md`

The metrics file must include at least:

```python
[
    "method",
    "rule",
    "slice",
    "num_examples",
    "auroc",
    "auprc",
    "coverage_10_precision",
    "coverage_25_precision",
    "coverage_50_precision",
]
```

The bucket breakdown must be written only for the best rule on:

- `full_module_trace`
- `attention_trace`

and only on the `all` slice.

- [ ] **Step 4: Run the targeted tests**

Run: `conda run -n esd_ind python -m unittest tests.test_rejection tests.test_pipeline -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/rejection_confidence_study.py tests/test_pipeline.py
git commit -m "feat: add rejection confidence study script"
```

## Task 3: Run The Mini-Study And Save The Results

**Files:**

- Runtime only: `research_runs/rejection_confidence_study_strict_v1/*`

- [ ] **Step 1: Run the study**

Run: `conda run -n esd_ind python scripts/rejection_confidence_study.py --output-dir research_runs/rejection_confidence_study_strict_v1`

Expected: exit `0`, output folder contains:

- `confidence_rule_metrics.csv`
- `best_rule_bucket_breakdown.csv`
- `report.md`

- [ ] **Step 2: Verify the results are usable**

Run:

```bash
conda run -n esd_ind python -c "import pandas as pd; m=pd.read_csv('research_runs/rejection_confidence_study_strict_v1/confidence_rule_metrics.csv'); print(m.sort_values(['method','slice','auroc'], ascending=[True,True,False]).to_string(index=False)); b=pd.read_csv('research_runs/rejection_confidence_study_strict_v1/best_rule_bucket_breakdown.csv'); print(); print(b.to_string(index=False))"
```

Expected:

- at least one non-`NaN` result for each `method/rule/slice`
- clear comparison between `top_score`, `margin`, and `score_plus_margin`

- [ ] **Step 3: Interpret with the agreed keep/kill rule**

The summary must say one of:

- `margin improves rejection enough to keep as a support result`
- `score_plus_margin improves rejection enough to keep as a support result`
- `simple confidence rules do not materially improve rejection`

- [ ] **Step 4: Commit**

```bash
git add research_runs/rejection_confidence_study_strict_v1
git commit -m "run rejection confidence mini study"
```
