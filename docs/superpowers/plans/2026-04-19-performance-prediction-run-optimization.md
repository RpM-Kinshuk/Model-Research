# Performance Prediction Run Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the current performance-prediction pilot cheap enough to rerun quickly by caching pairwise scores once per method, then test a tiny optimization ladder and keep the simplest promising path.

**Architecture:** Extend `spectral_research.prediction` with a reusable pairwise-score cache and cache-aware neighborhood prediction path. Keep the research logic fixed. Add one optional `torch` backend for vectorizable spectral precompute, wire the CLI with a small `--optimization-mode` switch, run a short timing bakeoff, then rerun the full pilot once with the best simple mode and update the findings note.

**Tech Stack:** Python, pandas, numpy, torch, scikit-learn, existing `spectral_research` helpers, `unittest`, shell timing via `/usr/bin/time`

---

## File Structure

- Modify: `spectral_research/prediction.py`
  - Add pairwise score-matrix builders, cache-aware neighborhood prediction, and an optional `torch` precompute path.
- Modify: `tests/test_prediction.py`
  - Add coverage for cached-vs-uncached equivalence, pairwise-matrix shape/content, and torch-backend parity.
- Modify: `scripts/spectral_prediction_research.py`
  - Add a small optimization-mode CLI switch and reuse precomputed caches per run.
- Modify: `docs/superpowers/specs/2026-04-18-spectral-family-retrieval-preliminary-findings.md`
  - Append the chosen optimized run mode, bakeoff outcome, and confirmation that metrics stayed consistent.

## Task 1: Add A Reusable Pairwise Cache On CPU

**Files:**
- Modify: `spectral_research/prediction.py`
- Modify: `tests/test_prediction.py`

- [ ] **Step 1: Write the failing cache tests**

```python
from spectral_research.prediction import (
    build_pairwise_score_matrix,
    neighborhood_predict_oof,
    structural_baseline_oof,
)


class PredictionOptimizationTests(unittest.TestCase):
    def test_build_pairwise_score_matrix_is_square_and_symmetric(self) -> None:
        frame = self._frame()
        records = self._records()

        matrix, model_index = build_pairwise_score_matrix(
            frame,
            records,
            method="pl_alpha_scalar",
            backend="cpu",
        )

        self.assertEqual(matrix.shape, (len(frame), len(frame)))
        self.assertEqual(set(model_index), set(frame["model_id"]))
        self.assertTrue(np.allclose(matrix, matrix.T, equal_nan=True))

    def test_cached_neighborhood_predictions_match_uncached_path(self) -> None:
        frame = self._frame()
        records = self._records()
        baseline = structural_baseline_oof(frame, target="Average ⬆️", cv_splits=2)
        baseline_by_model_id = dict(
            zip(baseline["model_id"], baseline["baseline_prediction"], strict=False)
        )
        matrix, model_index = build_pairwise_score_matrix(
            frame,
            records,
            method="pl_alpha_scalar",
            backend="cpu",
        )

        uncached = neighborhood_predict_oof(
            frame,
            records,
            target="Average ⬆️",
            method="pl_alpha_scalar",
            k=3,
            weighted=True,
            residualize=True,
            cv_splits=2,
        )
        cached = neighborhood_predict_oof(
            frame,
            records,
            target="Average ⬆️",
            method="pl_alpha_scalar",
            k=3,
            weighted=True,
            residualize=True,
            cv_splits=2,
            pairwise_scores=matrix,
            model_index=model_index,
            baseline_by_model_id=baseline_by_model_id,
        )

        pd.testing.assert_series_equal(
            uncached["prediction"],
            cached["prediction"],
            check_names=False,
            check_exact=False,
            atol=1e-12,
            rtol=0.0,
        )
```

- [ ] **Step 2: Run the optimization tests to verify they fail**

Run: `conda run -n esd_ind python -m unittest tests.test_prediction.PredictionOptimizationTests -v`

Expected: `FAIL` with missing `build_pairwise_score_matrix` and unexpected keyword arguments in `neighborhood_predict_oof`

- [ ] **Step 3: Add the CPU pairwise cache and cache-aware neighborhood path**

```python
# spectral_research/prediction.py
def build_pairwise_score_matrix(
    frame: pd.DataFrame,
    records_by_model_id: dict[str, AlphaRecord],
    *,
    method: str,
    backend: str = "cpu",
    device: str = "cpu",
) -> tuple[np.ndarray, dict[str, int]]:
    model_ids = frame["model_id"].fillna("").astype(str).tolist()
    model_index = {model_id: idx for idx, model_id in enumerate(model_ids)}
    matrix = np.full((len(model_ids), len(model_ids)), np.nan, dtype=float)
    structural_vectors = structural_basics_feature_map(frame)
    rows = [frame.iloc[idx] for idx in range(len(frame))]

    for idx in range(len(model_ids)):
        matrix[idx, idx] = 0.0

    for left_idx, left_row in enumerate(rows):
        left_model_id = model_ids[left_idx]
        left_record = records_by_model_id.get(left_model_id)
        for right_idx in range(left_idx + 1, len(model_ids)):
            right_row = rows[right_idx]
            right_model_id = model_ids[right_idx]
            score = _pair_score(
                method=method,
                left_row=left_row,
                right_row=right_row,
                left_record=left_record,
                right_record=records_by_model_id.get(right_model_id),
                structural_vectors=structural_vectors,
            )
            matrix[left_idx, right_idx] = score
            matrix[right_idx, left_idx] = score

    return matrix, model_index


def neighborhood_predict_oof(
    frame: pd.DataFrame,
    records_by_model_id: dict[str, AlphaRecord],
    *,
    target: str,
    method: str,
    k: int,
    weighted: bool,
    residualize: bool,
    cv_splits: int = 5,
    pairwise_scores: np.ndarray | None = None,
    model_index: dict[str, int] | None = None,
    baseline_by_model_id: dict[str, float] | None = None,
) -> pd.DataFrame:
    out = frame[["model_id", "query_bucket", target]].copy()
    out["prediction"] = np.nan
    out["mode"] = "spectral_plus_structural" if residualize else "raw_spectral"
    if out.empty:
        return out

    structural_vectors = structural_basics_feature_map(frame)
    if baseline_by_model_id is None:
        baseline = structural_baseline_oof(frame, target=target, cv_splits=cv_splits)
        baseline_by_model_id = {
            str(model_id): float(prediction)
            for model_id, prediction in zip(
                baseline["model_id"],
                baseline["baseline_prediction"],
                strict=False,
            )
        }
    target_by_model_id = {
        str(model_id): float(value)
        for model_id, value in zip(frame["model_id"], frame[target], strict=False)
    }
    working_target_by_model_id = {
        model_id: target_by_model_id[model_id] - baseline_by_model_id[model_id]
        if residualize
        else target_by_model_id[model_id]
        for model_id in target_by_model_id
    }
    groups = _cv_groups(frame)
    splitter = _group_kfold(frame, cv_splits=cv_splits)
    splits = (
        list(splitter.split(frame, frame[target].to_numpy(dtype=float), groups))
        if splitter is not None
        else [(np.arange(len(frame), dtype=int), np.arange(len(frame), dtype=int))]
    )
    prediction_column = out.columns.get_loc("prediction")
    for train_idx, test_idx in splits:
        for test_position in test_idx:
            test_row = frame.iloc[test_position]
            test_model_id = str(test_row["model_id"])
            if pairwise_scores is not None and model_index is not None and test_model_id in model_index:
                score_row = pairwise_scores[model_index[test_model_id], train_idx]
                neighbor_pairs = [
                    (float(score), working_target_by_model_id[str(frame.iloc[idx]["model_id"])])
                    for score, idx in zip(score_row, train_idx, strict=False)
                    if np.isfinite(score)
                ]
            else:
                test_record = records_by_model_id.get(test_model_id)
                neighbor_pairs = []
                for idx in train_idx:
                    train_row = frame.iloc[idx]
                    train_model_id = str(train_row["model_id"])
                    score = _pair_score(
                        method=method,
                        left_row=test_row,
                        right_row=train_row,
                        left_record=test_record,
                        right_record=records_by_model_id.get(train_model_id),
                        structural_vectors=structural_vectors,
                    )
                    if np.isfinite(score):
                        neighbor_pairs.append((float(score), working_target_by_model_id[train_model_id]))
            neighbor_pairs.sort(key=lambda item: item[0], reverse=True)
            top_neighbors = neighbor_pairs[:k]
            if not top_neighbors:
                continue
            prediction = _weighted_mean(top_neighbors) if weighted else float(
                np.mean([value for _, value in top_neighbors])
            )
            if residualize:
                prediction += baseline_by_model_id[test_model_id]
            out.iloc[test_position, prediction_column] = prediction
    return out
```

- [ ] **Step 4: Run the focused optimization tests to verify they pass**

Run: `conda run -n esd_ind python -m unittest tests.test_prediction.PredictionOptimizationTests -v`

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 add \
  spectral_research/prediction.py \
  tests/test_prediction.py
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 commit -m "feat: cache pairwise prediction scores"
```

## Task 2: Add The Small GPU Path And CLI Optimization Modes

**Files:**
- Modify: `spectral_research/prediction.py`
- Modify: `tests/test_prediction.py`
- Modify: `scripts/spectral_prediction_research.py`

- [ ] **Step 1: Extend the failing tests for torch parity and CLI args**

```python
class PredictionOptimizationModeTests(unittest.TestCase):
    def test_torch_pairwise_backend_matches_cpu_for_scalar_features(self) -> None:
        frame = self._frame()
        records = self._records()

        cpu_matrix, cpu_index = build_pairwise_score_matrix(
            frame,
            records,
            method="pl_alpha_scalar",
            backend="cpu",
        )
        torch_matrix, torch_index = build_pairwise_score_matrix(
            frame,
            records,
            method="pl_alpha_scalar",
            backend="torch",
            device="cpu",
        )

        self.assertEqual(cpu_index, torch_index)
        self.assertTrue(np.allclose(cpu_matrix, torch_matrix, equal_nan=True))

    def test_prediction_cli_accepts_optimization_mode_and_device(self) -> None:
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "spectral_prediction_research.py"
        spec = importlib.util.spec_from_file_location("spectral_prediction_research_cli", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        args = module.parse_args(
            [
                "--optimization-mode",
                "cache_plus_gpu0",
                "--device",
                "cuda:0",
            ]
        )

        self.assertEqual(args.optimization_mode, "cache_plus_gpu0")
        self.assertEqual(args.device, "cuda:0")
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `conda run -n esd_ind python -m unittest tests.test_prediction.PredictionOptimizationModeTests -v`

Expected: `FAIL` with missing CLI args and unsupported `backend="torch"`

- [ ] **Step 3: Add the optional torch backend and CLI wiring**

```python
# spectral_research/prediction.py
try:
    import torch
except ImportError:  # pragma: no cover - torch is present in esd_ind, but keep CPU fallback
    torch = None


def _torch_negative_l2_matrix(vectors: np.ndarray, *, device: str) -> np.ndarray:
    tensor = torch.as_tensor(vectors, dtype=torch.float32, device=device)
    mask = torch.isfinite(tensor)
    filled = torch.nan_to_num(tensor, nan=0.0)
    diffs = filled[:, None, :] - filled[None, :, :]
    shared = mask[:, None, :] & mask[None, :, :]
    counts = shared.sum(dim=-1)
    sq_error = (diffs * diffs).masked_fill(~shared, 0.0).sum(dim=-1)
    out = -sq_error / torch.clamp(counts, min=1)
    out[counts == 0] = torch.nan
    return out.detach().cpu().numpy()


def _build_cpu_pairwise_score_matrix(
    frame: pd.DataFrame,
    records_by_model_id: dict[str, AlphaRecord],
    *,
    method: str,
) -> tuple[np.ndarray, dict[str, int]]:
    model_ids = frame["model_id"].fillna("").astype(str).tolist()
    model_index = {model_id: idx for idx, model_id in enumerate(model_ids)}
    matrix = np.full((len(model_ids), len(model_ids)), np.nan, dtype=float)
    structural_vectors = structural_basics_feature_map(frame)
    rows = [frame.iloc[idx] for idx in range(len(frame))]

    for idx in range(len(model_ids)):
        matrix[idx, idx] = 0.0

    for left_idx, left_row in enumerate(rows):
        left_record = records_by_model_id.get(model_ids[left_idx])
        for right_idx in range(left_idx + 1, len(model_ids)):
            score = _pair_score(
                method=method,
                left_row=left_row,
                right_row=rows[right_idx],
                left_record=left_record,
                right_record=records_by_model_id.get(model_ids[right_idx]),
                structural_vectors=structural_vectors,
            )
            matrix[left_idx, right_idx] = score
            matrix[right_idx, left_idx] = score

    return matrix, model_index


def build_pairwise_score_matrix(
    frame: pd.DataFrame,
    records_by_model_id: dict[str, AlphaRecord],
    *,
    method: str,
    backend: str = "cpu",
    device: str = "cpu",
) -> tuple[np.ndarray, dict[str, int]]:
    model_ids = frame["model_id"].fillna("").astype(str).tolist()
    model_index = {model_id: idx for idx, model_id in enumerate(model_ids)}
    if backend == "torch" and torch is not None and method == "pl_alpha_scalar":
        vectors = np.vstack(
            [
                alpha_scalar_features(records_by_model_id[str(model_id)].alpha).reshape(-1)
                for model_id in frame["model_id"]
            ]
        )
        return _torch_negative_l2_matrix(vectors, device=device), model_index
    return _build_cpu_pairwise_score_matrix(frame, records_by_model_id, method=method)
```

```python
# scripts/spectral_prediction_research.py
parser.add_argument(
    "--optimization-mode",
    choices=("repeated_cpu", "cpu_cache_only", "cache_plus_gpu0"),
    default="repeated_cpu",
    help="How to build neighbor scores for the prediction run.",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="Torch device for spectral precompute when optimization mode uses GPU.",
)

def main() -> None:
    args = parse_args()
    targets = _split_csv(args.targets)
    dataset, records_by_model_id = build_prediction_dataset(
        metrics_dir=args.metrics_dir,
        leaderboard_csv=args.leaderboard_csv,
        lineage_csv=args.lineage_csv,
        freeze_csv=args.freeze_csv,
    )
    pairwise_cache: dict[str, tuple[np.ndarray, dict[str, int]]] = {}
    if args.optimization_mode != "repeated_cpu":
        for method in SUPPORTED_METHODS:
            backend = (
                "torch"
                if args.optimization_mode == "cache_plus_gpu0" and method != "structural_basics_only"
                else "cpu"
            )
            pairwise_cache[method] = build_pairwise_score_matrix(
                dataset,
                records_by_model_id,
                method=method,
                backend=backend,
                device=args.device,
            )
    for target in targets:
        baseline_oof, baseline_summary = _baseline_rows(
            dataset,
            target=target,
            cv_splits=args.cv_splits,
        )
        baseline_by_model_id = {
            str(model_id): float(prediction)
            for model_id, prediction in zip(
                baseline_oof["model_id"],
                baseline_oof["prediction"],
                strict=False,
            )
        }
        oof_frames.append(baseline_oof)
        result_frames.append(baseline_summary)

        for method in SUPPORTED_METHODS[1:]:
            matrix, model_index = pairwise_cache.get(method, (None, None))
            target_oof, target_results = _spectral_rows(
                dataset,
                records_by_model_id,
                target=target,
                method=method,
                cv_splits=args.cv_splits,
                pairwise_scores=matrix,
                model_index=model_index,
                baseline_by_model_id=baseline_by_model_id,
            )
            oof_frames.extend(target_oof)
            result_frames.extend(target_results)
```

- [ ] **Step 4: Run the full prediction test module to verify the optimization path did not break existing behavior**

Run: `conda run -n esd_ind python -m unittest tests.test_prediction -v`

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 add \
  spectral_research/prediction.py \
  scripts/spectral_prediction_research.py \
  tests/test_prediction.py
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 commit -m "feat: add optimized prediction run modes"
```

## Task 3: Run The Tiny Bakeoff, Keep The Simplest Winner, And Update Findings

**Files:**
- Modify: `docs/superpowers/specs/2026-04-18-spectral-family-retrieval-preliminary-findings.md`

- [ ] **Step 1: Kill stale prediction runners and verify the test suite before timing**

Run:

```bash
pkill -f 'scripts/spectral_prediction_research.py' || true
conda run -n esd_ind python -m unittest tests.test_prediction tests.test_pipeline tests.test_heterogeneity -v
```

Expected:

- no stale `spectral_prediction_research.py` processes remain
- test output ends with `OK`

- [ ] **Step 2: Run the short timing bakeoff on one target**

Run:

```bash
/usr/bin/time -f 'repeated_cpu %E %M' \
  conda run -n esd_ind python scripts/spectral_prediction_research.py \
  --targets "Average ⬆️" \
  --optimization-mode repeated_cpu \
  --output-dir research_runs/performance_prediction_bakeoff_repeated

/usr/bin/time -f 'cpu_cache_only %E %M' \
  conda run -n esd_ind python scripts/spectral_prediction_research.py \
  --targets "Average ⬆️" \
  --optimization-mode cpu_cache_only \
  --output-dir research_runs/performance_prediction_bakeoff_cpu_cache

CUDA_VISIBLE_DEVICES=0 /usr/bin/time -f 'cache_plus_gpu0 %E %M' \
  conda run -n esd_ind python scripts/spectral_prediction_research.py \
  --targets "Average ⬆️" \
  --optimization-mode cache_plus_gpu0 \
  --device cuda:0 \
  --output-dir research_runs/performance_prediction_bakeoff_gpu0
```

Expected:

- all three runs finish
- the cache-based modes are materially faster than `repeated_cpu`
- if `cache_plus_gpu0` is only marginally better or not better at all, prefer `cpu_cache_only`

- [ ] **Step 3: Compare numerical consistency and pick the simplest promising mode**

Run:

```bash
conda run -n esd_ind python - <<'PY'
import pandas as pd

key = ["target", "method", "mode", "bucket", "k", "weighted", "n"]
repeated = pd.read_csv("research_runs/performance_prediction_bakeoff_repeated/prediction_results.csv")
cached = pd.read_csv("research_runs/performance_prediction_bakeoff_cpu_cache/prediction_results.csv")
gpu = pd.read_csv("research_runs/performance_prediction_bakeoff_gpu0/prediction_results.csv")

for name, frame in [("cached", cached), ("gpu", gpu)]:
    merged = repeated.merge(frame, on=key, suffixes=("_base", f"_{name}"))
    print(name, len(merged), len(repeated), len(frame))
    print(name, "max_abs_r2", (merged["r2_base"] - merged[f"r2_{name}"]).abs().max())
    print(name, "max_abs_mae", (merged["mae_base"] - merged[f"mae_{name}"]).abs().max())
    print(name, "max_abs_spearman", (merged["spearman_base"] - merged[f"spearman_{name}"]).abs().max())
PY
```

Expected:

- merged row counts match
- metric deltas are only tiny floating-point noise
- choose the simplest faster mode, usually `cpu_cache_only` or `cache_plus_gpu0`

- [ ] **Step 4: Rerun the full pilot once with the chosen mode**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 \
  conda run -n esd_ind python scripts/spectral_prediction_research.py \
  --targets "Average ⬆️,MMLU-PRO,MATH Lvl 5" \
  --optimization-mode cache_plus_gpu0 \
  --device cuda:0 \
  --output-dir research_runs/performance_prediction_cycle1_optimized
```

If `cpu_cache_only` won the bakeoff, replace the mode and omit `CUDA_VISIBLE_DEVICES`.

Expected:

- full run finishes without hanging for hours
- output files exist:
  - `research_runs/performance_prediction_cycle1_optimized/prediction_dataset.csv`
  - `research_runs/performance_prediction_cycle1_optimized/prediction_results.csv`
  - `research_runs/performance_prediction_cycle1_optimized/oof_predictions.csv`
  - `research_runs/performance_prediction_cycle1_optimized/report.md`

- [ ] **Step 5: Update the findings note and commit**

```markdown
## Prediction Run Optimization Note

- tested `repeated_cpu`, `cpu_cache_only`, and `cache_plus_gpu0`
- kept `<chosen_mode>` because it was the simplest materially faster path
- confirmed optimized outputs matched the original pilot up to tiny floating-point noise
- this was a run-cost optimization only, not a change in the prediction experiment
```

Run:

```bash
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 add \
  docs/superpowers/specs/2026-04-18-spectral-family-retrieval-preliminary-findings.md \
  research_runs/performance_prediction_bakeoff_repeated \
  research_runs/performance_prediction_bakeoff_cpu_cache \
  research_runs/performance_prediction_bakeoff_gpu0 \
  research_runs/performance_prediction_cycle1_optimized
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 commit -m "run optimized performance prediction pilot"
```

## Self-Review

- Spec coverage:
  - pairwise-cache optimization is implemented in Task 1
  - optional GPU precompute path is implemented in Task 2
  - tiny bakeoff, simplest-winner selection, and one final rerun are implemented in Task 3
  - findings-note update is included in Task 3
- Placeholder scan:
  - no `TODO`, `TBD`, or deferred implementation placeholders remain
- Type consistency:
  - the plan consistently uses `build_pairwise_score_matrix`, `pairwise_scores`, `model_index`, `baseline_by_model_id`, `--optimization-mode`, and `--device`
