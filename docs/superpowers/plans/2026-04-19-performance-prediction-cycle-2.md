# Performance Prediction Cycle 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one small mixed-zoo prediction follow-up that tests whether a simple routed spectral neighborhood method can improve heterogeneous prediction without damaging the stronger slices.

**Architecture:** Reuse the current joined scored dataset, grouped-CV pipeline, optimized runner, and existing methods. Add exactly one new prediction method, `schema_routed_trace`, that uses `full_module_trace` when exact-schema support is sufficient and otherwise falls back to `role_aggregated_trace`. Evaluate it on the same three targets, with `MMLU-PRO` and `MATH Lvl 5` treated as the real tests and `Average ⬆️` kept as a sanity target. Summarize Cycle 2 strictly as a comparison against Cycle 1, not as a broader prediction-system expansion.

**Tech Stack:** Python, pandas, numpy, existing `spectral_research.prediction` module, existing `spectral_research.heterogeneity` helpers, `unittest`

---

## File Structure

- Modify: `spectral_research/prediction.py`
  - Add the `schema_routed_trace` pairwise path and keep the routing rule local and deterministic.
- Modify: `scripts/spectral_prediction_research.py`
  - Include `schema_routed_trace` in the supported method ladder and keep the reporting format unchanged.
- Modify: `tests/test_prediction.py`
  - Add focused routing tests and preserve Cycle 1 behavior.
- Modify: `docs/superpowers/specs/2026-04-18-spectral-family-retrieval-preliminary-findings.md`
  - Append a short Cycle 2 comparison note after the run.

## Task 1: Add The Routed Spectral Prediction Method

**Files:**
- Modify: `spectral_research/prediction.py`
- Modify: `tests/test_prediction.py`

- [ ] **Step 1: Write the failing routing tests**

```python
from spectral_research.prediction import build_pairwise_score_matrix, neighborhood_predict_oof


class PredictionRoutingTests(unittest.TestCase):
    def test_schema_routed_trace_matches_full_trace_when_exact_scores_exist(self) -> None:
        frame = self._frame()
        records = self._records()

        full_matrix, _ = build_pairwise_score_matrix(
            frame,
            records,
            method="full_module_trace",
            backend="cpu",
        )
        routed_matrix, _ = build_pairwise_score_matrix(
            frame,
            records,
            method="schema_routed_trace",
            backend="cpu",
        )

        finite = np.isfinite(full_matrix)
        np.testing.assert_allclose(
            routed_matrix[finite],
            full_matrix[finite],
            rtol=0.0,
            atol=0.0,
        )

    def test_schema_routed_trace_falls_back_to_role_aggregated_when_full_trace_is_nan(self) -> None:
        frame = self._heterogeneous_frame()
        records = self._heterogeneous_records()

        full_matrix, index = build_pairwise_score_matrix(
            frame,
            records,
            method="full_module_trace",
            backend="cpu",
        )
        role_matrix, _ = build_pairwise_score_matrix(
            frame,
            records,
            method="role_aggregated_trace",
            backend="cpu",
        )
        routed_matrix, _ = build_pairwise_score_matrix(
            frame,
            records,
            method="schema_routed_trace",
            backend="cpu",
        )

        i = index["m1"]
        j = index["m2"]
        self.assertTrue(np.isnan(full_matrix[i, j]))
        self.assertTrue(np.isfinite(role_matrix[i, j]))
        self.assertEqual(routed_matrix[i, j], role_matrix[i, j])
```

- [ ] **Step 2: Run the focused routing tests to verify they fail**

Run: `conda run -n esd_ind python -m unittest tests.test_prediction.PredictionRoutingTests -v`

Expected: `FAIL` with unsupported `schema_routed_trace`

- [ ] **Step 3: Implement the minimal routed method**

```python
# spectral_research/prediction.py
def _pair_score(
    *,
    method: str,
    left_row: pd.Series,
    right_row: pd.Series,
    left_record: AlphaRecord | None,
    right_record: AlphaRecord | None,
    structural_vectors: dict[str, np.ndarray],
) -> float:
    left_model_id = str(left_row["model_id"])
    right_model_id = str(right_row["model_id"])

    if method == "structural_basics_only":
        left_vector = structural_vectors.get(left_model_id)
        right_vector = structural_vectors.get(right_model_id)
        if left_vector is None or right_vector is None:
            return float("nan")
        return _negative_l2(left_vector, right_vector)

    if left_record is None or right_record is None:
        return float("nan")

    if method == "pl_alpha_scalar":
        return _negative_l2(
            alpha_scalar_features(left_record.alpha).reshape(-1),
            alpha_scalar_features(right_record.alpha).reshape(-1),
        )

    if method == "full_module_trace":
        if tuple(left_record.raw_modules) != tuple(right_record.raw_modules):
            return float("nan")
        depth = min(left_record.alpha.shape[0], right_record.alpha.shape[0])
        if depth <= 0:
            return float("nan")
        return _negative_l2(left_record.alpha[:depth], right_record.alpha[:depth])

    if method == "role_aggregated_trace":
        return role_aggregated_trace_score(
            query_alpha=np.asarray(left_record.alpha, dtype=float),
            query_modules=tuple(left_record.raw_modules),
            proto_alpha=np.asarray(right_record.alpha, dtype=float),
            proto_modules=tuple(right_record.raw_modules),
            overlap_penalty=0.2,
        )

    if method == "schema_routed_trace":
        full_score = _pair_score(
            method="full_module_trace",
            left_row=left_row,
            right_row=right_row,
            left_record=left_record,
            right_record=right_record,
            structural_vectors=structural_vectors,
        )
        if np.isfinite(full_score):
            return full_score
        return _pair_score(
            method="role_aggregated_trace",
            left_row=left_row,
            right_row=right_row,
            left_record=left_record,
            right_record=right_record,
            structural_vectors=structural_vectors,
        )
    raise KeyError(f"Unsupported method: {method}")
```

- [ ] **Step 4: Run the focused routing tests to verify they pass**

Run: `conda run -n esd_ind python -m unittest tests.test_prediction.PredictionRoutingTests -v`

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 add \
  spectral_research/prediction.py \
  tests/test_prediction.py
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 commit -m "feat: add schema routed prediction trace"
```

## Task 2: Wire The Routed Method Into The Prediction Runner

**Files:**
- Modify: `scripts/spectral_prediction_research.py`
- Modify: `tests/test_prediction.py`

- [ ] **Step 1: Extend the failing CLI/reporting test**

```python
class PredictionCliCycle2Tests(unittest.TestCase):
    def test_supported_methods_include_schema_routed_trace(self) -> None:
        module = self._load_module()
        self.assertIn("schema_routed_trace", module.SUPPORTED_METHODS)
```

- [ ] **Step 2: Run the focused CLI test to verify it fails**

Run: `conda run -n esd_ind python -m unittest tests.test_prediction.PredictionCliCycle2Tests -v`

Expected: `FAIL` because `schema_routed_trace` is absent

- [ ] **Step 3: Add the method to the runner**

```python
# scripts/spectral_prediction_research.py
SUPPORTED_METHODS = (
    "structural_basics_only",
    "pl_alpha_scalar",
    "full_module_trace",
    "role_aggregated_trace",
    "schema_routed_trace",
)
```

Keep:

- the same target loop
- the same optimization-mode behavior
- the same output/report schema

- [ ] **Step 4: Run the full prediction test module**

Run: `conda run -n esd_ind python -m unittest tests.test_prediction -v`

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 add \
  scripts/spectral_prediction_research.py \
  tests/test_prediction.py
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 commit -m "feat: wire schema routed prediction method"
```

## Task 3: Run Cycle 2 And Record The Comparison

**Files:**
- Modify: `docs/superpowers/specs/2026-04-18-spectral-family-retrieval-preliminary-findings.md`

- [ ] **Step 1: Verify tests and clear stale runners**

Run:

```bash
pkill -f 'spectral_prediction_research\.py' || true
conda run -n esd_ind env PYTHONPATH=. pytest -q
```

Expected:

- no stale prediction runner remains
- test output ends with `passed`

- [ ] **Step 2: Run the Cycle 2 pilot with the kept optimized mode**

Run:

```bash
conda run -n esd_ind python scripts/spectral_prediction_research.py \
  --targets "Average ⬆️,MMLU-PRO,MATH Lvl 5" \
  --optimization-mode cpu_cache_only \
  --output-dir research_runs/performance_prediction_cycle2
```

Expected:

- the run completes successfully
- outputs exist:
  - `research_runs/performance_prediction_cycle2/prediction_dataset.csv`
  - `research_runs/performance_prediction_cycle2/prediction_results.csv`
  - `research_runs/performance_prediction_cycle2/oof_predictions.csv`
  - `research_runs/performance_prediction_cycle2/report.md`

- [ ] **Step 3: Write a short Cycle 2 comparison note**

Append a short section to the findings note covering:

```markdown
### Prediction Cycle 2 Note

- added `schema_routed_trace` as a simple mixed-zoo prediction rule
- main interpretation emphasis stayed on `MMLU-PRO` and `MATH Lvl 5`
- `Average ⬆️` remained a sanity target only
- compare Cycle 2 against Cycle 1 with emphasis on:
  - heterogeneous slice change
  - whether the routed method beat `full_module_trace` and `role_aggregated_trace`
  - whether any gains came without clear regression on the stronger slices
```

Keep it factual and short.

- [ ] **Step 4: Verify the key comparison directly from the CSV**

Run:

```bash
python - <<'PY'
import pandas as pd

cycle1 = pd.read_csv("research_runs/performance_prediction_cycle1_optimized/prediction_results.csv")
cycle2 = pd.read_csv("research_runs/performance_prediction_cycle2/prediction_results.csv")

for bucket in ["heterogeneous", "supported_candidate", "unsupported_special", "unsupported_unknown"]:
    print("\\nBUCKET", bucket)
    for target in ["Average ⬆️", "MMLU-PRO", "MATH Lvl 5"]:
        c1 = cycle1[(cycle1["bucket"] == bucket) & (cycle1["target"] == target)].sort_values(["r2", "spearman"], ascending=[False, False]).head(1)
        c2 = cycle2[(cycle2["bucket"] == bucket) & (cycle2["target"] == target)].sort_values(["r2", "spearman"], ascending=[False, False]).head(1)
        print(target, "cycle1", c1[["method", "mode", "r2", "spearman"]].to_dict("records"))
        print(target, "cycle2", c2[["method", "mode", "r2", "spearman"]].to_dict("records"))
PY
```

Expected:

- direct visibility into whether `schema_routed_trace` improved the heterogeneous slice
- no guessing from memory or report prose

- [ ] **Step 5: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 add \
  docs/superpowers/specs/2026-04-18-spectral-family-retrieval-preliminary-findings.md \
  research_runs/performance_prediction_cycle2
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 commit -m "run performance prediction cycle 2"
```

## Self-Review

- Spec coverage:
  - the one new routed method is implemented in Task 1
  - the runner integration is handled in Task 2
  - the Cycle 2 run and comparison writeup are handled in Task 3
  - the broader `absolute trend` and `lineage-relative trend` note remains deferred by design and is not incorrectly pulled into this cycle
- Placeholder scan:
  - no `TODO`, `TBD`, or unresolved placeholder text remains
- Type consistency:
  - the plan consistently uses `schema_routed_trace`, existing target names, existing output paths, and the kept optimization mode `cpu_cache_only`
