# Lean Mixed-Zoo Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a parallel `lean` mixed-zoo benchmark mode that removes architecture/params from method inputs, broadens supported slices to `confirmed + provisional`, and reuses the current spectral ladder on a broader zoo slice.

**Architecture:** Extend the existing mixed-zoo runner instead of creating a second script. Add one lean structural baseline helper in `spectral_research/heterogeneity.py`, add one small benchmark-mode switch and supported-slice policy in `scripts/mixed_zoo_retrieval_research.py`, and add focused tests to prove the lean mode uses broader support and omits architecture/params from its baseline features.

**Tech Stack:** Python 3, `numpy`, `pandas`, `unittest`, existing `conda run -n esd_ind` workflow, current `spectral_research` package, local `svd_results` and CSV artifacts.

---

## File Structure

- Modify: `spectral_research/heterogeneity.py`
  - add `structural_basics_feature_map`
  - keep `intrinsic_basics_feature_map` unchanged for strict mode
- Modify: `tests/test_heterogeneity.py`
  - add focused tests for the new lean structural baseline
- Modify: `scripts/mixed_zoo_retrieval_research.py`
  - add `--benchmark-mode {strict,lean}`
  - add lean supported-slice selection that allows `confirmed + provisional`
  - switch baseline method name and feature map by mode
  - keep the spectral ladder unchanged
- Modify: `tests/test_pipeline.py`
  - add mode-parsing and lean-slice tests
  - add a small test for lean method lists

### Task 1: Add The Lean Structural Baseline Helper

**Files:**
- Modify: `spectral_research/heterogeneity.py`
- Modify: `tests/test_heterogeneity.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_heterogeneity.py`:

```python
def test_structural_basics_feature_map_excludes_architecture_and_params():
    manifest = pd.DataFrame(
        [
            {
                "model_id": "model/a",
                "Architecture": "LlamaForCausalLM",
                "#Params (B)": 8.0,
                "num_layers": 32,
                "schema_name": "common_llm_7",
                "raw_module_count": 7,
                "common_schema": True,
            },
            {
                "model_id": "model/b",
                "Architecture": "QwenForCausalLM",
                "#Params (B)": 72.0,
                "num_layers": 32,
                "schema_name": "common_llm_7",
                "raw_module_count": 7,
                "common_schema": True,
            },
        ]
    )

    vectors = structural_basics_feature_map(manifest)
    np.testing.assert_allclose(vectors["model/a"], vectors["model/b"])


def test_structural_basics_feature_map_uses_structural_fields():
    manifest = pd.DataFrame(
        [
            {
                "model_id": "model/a",
                "num_layers": 32,
                "schema_name": "common_llm_7",
                "raw_module_count": 7,
                "common_schema": True,
            },
            {
                "model_id": "model/b",
                "num_layers": 40,
                "schema_name": "hetero_llm_5",
                "raw_module_count": 5,
                "common_schema": False,
            },
        ]
    )

    vectors = structural_basics_feature_map(manifest)
    assert vectors["model/a"].shape == vectors["model/b"].shape
    assert not np.allclose(vectors["model/a"], vectors["model/b"])
```

- [ ] **Step 2: Run the test file to confirm failure**

Run:

```bash
conda run -n esd_ind python -m unittest tests.test_heterogeneity -v
```

Expected:
- FAIL with `NameError` or `ImportError` for `structural_basics_feature_map`

- [ ] **Step 3: Implement the helper**

Modify `spectral_research/heterogeneity.py`:

```python
_STRUCTURAL_COLUMNS = (
    "num_layers",
    "schema_name",
    "raw_module_count",
    "common_schema",
)


def structural_basics_feature_map(manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    frame = manifest.copy()
    if "model_id" not in frame.columns:
        return {}
    if frame.empty:
        return {}

    columns = [column for column in _STRUCTURAL_COLUMNS if column in frame.columns]
    if not columns:
        raise ValueError("no structural feature columns are available")
    structural = frame.loc[:, columns].copy()

    numeric_columns = ["num_layers", "raw_module_count"]
    numeric = pd.DataFrame(index=frame.index)
    for column in numeric_columns:
        if column in structural.columns:
            numeric[column] = pd.to_numeric(structural[column], errors="coerce").fillna(-1.0)

    categorical_columns = [column for column in structural.columns if column not in numeric.columns]
    if categorical_columns:
        categorical = structural.loc[:, categorical_columns].fillna("unknown").astype(str)
        categorical = pd.get_dummies(categorical, prefix=categorical_columns, dtype=float)
    else:
        categorical = pd.DataFrame(index=frame.index)

    features = pd.concat([numeric, categorical], axis=1)
    vectors: dict[str, np.ndarray] = {}
    for row in pd.concat([frame[["model_id"]], features], axis=1).itertuples(index=False):
        vectors[str(row.model_id)] = np.asarray(row[1:], dtype=float)
    return vectors
```

Update `__all__`:

```python
    "structural_basics_feature_map",
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run:

```bash
conda run -n esd_ind python -m unittest tests.test_heterogeneity -v
```

Expected:
- PASS for the new structural baseline tests

- [ ] **Step 5: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 add spectral_research/heterogeneity.py tests/test_heterogeneity.py
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 commit -m "feat: add lean structural baseline"
```

### Task 2: Add Lean Mode Parsing And Supported-Slice Policy

**Files:**
- Modify: `scripts/mixed_zoo_retrieval_research.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_pipeline.py`:

```python
def test_mixed_zoo_parse_args_accepts_benchmark_mode():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "mixed_zoo_retrieval_research.py"
    )
    spec = importlib.util.spec_from_file_location("mixed_zoo_retrieval_research_cli", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    args = module.parse_args(["--benchmark-mode", "lean", "--output-dir", "out"])
    assert args.benchmark_mode == "lean"
    assert args.output_dir == Path("out")


def test_build_supported_slices_allows_provisional_labels_in_lean_mode():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "mixed_zoo_retrieval_research.py"
    )
    spec = importlib.util.spec_from_file_location("mixed_zoo_retrieval_research_cli", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    manifest = pd.DataFrame(
        [
            {
                "model_id": f"hetero-{i}",
                "audited_base_family": "fam/x",
                "label_status": "provisional",
                "query_bucket": "supported_candidate",
                "common_schema": False,
            }
            for i in range(5)
        ]
    )

    strict = module.build_supported_candidate_slices(manifest, min_family_size=5, benchmark_mode="strict")
    lean = module.build_supported_candidate_slices(manifest, min_family_size=5, benchmark_mode="lean")

    assert strict["heterogeneous_supported"].empty
    assert len(lean["heterogeneous_supported"]) == 5
```

- [ ] **Step 2: Run the test file to confirm failure**

Run:

```bash
conda run -n esd_ind python -m unittest tests.test_pipeline -v
```

Expected:
- FAIL because `--benchmark-mode` and `build_supported_candidate_slices` do not exist

- [ ] **Step 3: Implement the small mode switch and supported-slice builder**

Modify `scripts/mixed_zoo_retrieval_research.py`:

```python
from spectral_research.heterogeneity import (
    intrinsic_basics_feature_map,
    structural_basics_feature_map,
)
```

Add parser option:

```python
    parser.add_argument(
        "--benchmark-mode",
        choices=("strict", "lean"),
        default="strict",
    )
```

Add helper:

```python
def build_supported_candidate_slices(
    manifest: pd.DataFrame,
    *,
    min_family_size: int,
    benchmark_mode: str,
) -> dict[str, pd.DataFrame]:
    if benchmark_mode == "strict":
        supported_statuses = {"confirmed"}
    elif benchmark_mode == "lean":
        supported_statuses = {"confirmed", "provisional"}
    else:
        raise ValueError(f"unsupported benchmark mode: {benchmark_mode}")

    supported = manifest[
        manifest["label_status"].isin(supported_statuses)
        & manifest["query_bucket"].eq("supported_candidate")
        & manifest["audited_base_family"].fillna("").astype(str).ne("")
    ].copy()

    def _apply_floor(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        counts = frame["audited_base_family"].value_counts()
        keep = counts[counts >= min_family_size].index
        return frame[frame["audited_base_family"].isin(keep)].copy()

    homogeneous_supported = _apply_floor(
        supported[supported["common_schema"].fillna(False).astype(bool)].copy()
    )
    heterogeneous_supported = _apply_floor(
        supported[~supported["common_schema"].fillna(False).astype(bool)].copy()
    )
    unsupported_special = manifest[manifest["query_bucket"].eq("unsupported_special")].copy()
    unsupported_unknown = manifest[manifest["query_bucket"].eq("unsupported_unknown")].copy()
    return {
        "homogeneous_supported": homogeneous_supported.sort_values("model_id").reset_index(drop=True),
        "heterogeneous_supported": heterogeneous_supported.sort_values("model_id").reset_index(drop=True),
        "unsupported_special": unsupported_special.sort_values("model_id").reset_index(drop=True),
        "unsupported_unknown": unsupported_unknown.sort_values("model_id").reset_index(drop=True),
    }
```

In `main`, replace:

```python
    slices = build_mixed_zoo_slices(audited_manifest, min_family_size=args.min_family_size)
```

with:

```python
    slices = build_supported_candidate_slices(
        audited_manifest,
        min_family_size=args.min_family_size,
        benchmark_mode=args.benchmark_mode,
    )
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run:

```bash
conda run -n esd_ind python -m unittest tests.test_pipeline -v
```

Expected:
- PASS for the new mode and slice-policy tests

- [ ] **Step 5: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 add scripts/mixed_zoo_retrieval_research.py tests/test_pipeline.py
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 commit -m "feat: add lean mixed-zoo slice mode"
```

### Task 3: Switch The Baseline Method By Benchmark Mode

**Files:**
- Modify: `scripts/mixed_zoo_retrieval_research.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_pipeline.py`:

```python
def test_mixed_zoo_methods_use_structural_baseline_in_lean_mode():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "mixed_zoo_retrieval_research.py"
    )
    spec = importlib.util.spec_from_file_location("mixed_zoo_retrieval_research_cli", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.methods_for_benchmark_mode("strict")[0] == "intrinsic_basics_only"
    assert module.methods_for_benchmark_mode("lean")[0] == "structural_basics_only"
    assert module.methods_for_benchmark_mode("lean")[1:] == (
        "pl_alpha_scalar",
        "full_module_trace",
        "shared_role_trace",
        "role_aggregated_trace",
    )
```

- [ ] **Step 2: Run the test file to confirm failure**

Run:

```bash
conda run -n esd_ind python -m unittest tests.test_pipeline -v
```

Expected:
- FAIL because `methods_for_benchmark_mode` does not exist

- [ ] **Step 3: Implement the mode-dependent baseline**

Modify `scripts/mixed_zoo_retrieval_research.py`:

```python
STRICT_METHODS = (
    "intrinsic_basics_only",
    "pl_alpha_scalar",
    "full_module_trace",
    "shared_role_trace",
    "role_aggregated_trace",
)
LEAN_METHODS = (
    "structural_basics_only",
    "pl_alpha_scalar",
    "full_module_trace",
    "shared_role_trace",
    "role_aggregated_trace",
)


def methods_for_benchmark_mode(benchmark_mode: str) -> tuple[str, ...]:
    if benchmark_mode == "strict":
        return STRICT_METHODS
    if benchmark_mode == "lean":
        return LEAN_METHODS
    raise ValueError(f"unsupported benchmark mode: {benchmark_mode}")
```

Use:

```python
    methods = methods_for_benchmark_mode(args.benchmark_mode)
```

Add helper:

```python
def metadata_vectors_for_benchmark_mode(
    audited_manifest: pd.DataFrame,
    benchmark_mode: str,
) -> dict[str, np.ndarray]:
    if benchmark_mode == "strict":
        return intrinsic_basics_feature_map(audited_manifest)
    if benchmark_mode == "lean":
        return structural_basics_feature_map(audited_manifest)
    raise ValueError(f"unsupported benchmark mode: {benchmark_mode}")
```

Use:

```python
    metadata_vectors = metadata_vectors_for_benchmark_mode(
        audited_manifest,
        args.benchmark_mode,
    )
```

Everywhere the script currently uses `MIXED_ZOO_METHODS`, replace it with `methods`.

- [ ] **Step 4: Run the tests to confirm they pass**

Run:

```bash
conda run -n esd_ind python -m unittest tests.test_pipeline -v
```

Expected:
- PASS for the lean baseline-method test

- [ ] **Step 5: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 add scripts/mixed_zoo_retrieval_research.py tests/test_pipeline.py
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 commit -m "feat: add lean mixed-zoo structural baseline"
```

### Task 4: Make The Report And Gate Mode-Aware

**Files:**
- Modify: `scripts/mixed_zoo_retrieval_research.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_pipeline.py`:

```python
def test_write_prediction_bridge_gate_mentions_no_rows_when_heterogeneous_slice_is_empty():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "mixed_zoo_retrieval_research.py"
    )
    spec = importlib.util.spec_from_file_location("mixed_zoo_retrieval_research_cli", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        pd.DataFrame(
            [
                {
                    "method": "structural_basics_only",
                    "slice": "heterogeneous_supported",
                    "num_queries": 0,
                    "top1_accuracy": np.nan,
                    "mrr": np.nan,
                }
            ]
        ).to_csv(out / "retrieval_metrics.csv", index=False)

        module.write_prediction_bridge_gate(out)

        gate = (out / "prediction_bridge_gate.md").read_text(encoding="utf-8")
        assert "No heterogeneous-supported retrieval rows were produced." in gate
```

- [ ] **Step 2: Run the test file to confirm failure or mismatch**

Run:

```bash
conda run -n esd_ind python -m unittest tests.test_pipeline -v
```

Expected:
- FAIL if the gate still assumes `full_module_trace` only or does not behave consistently across modes

- [ ] **Step 3: Make report/gate mode-aware without adding complexity**

Modify `scripts/mixed_zoo_retrieval_research.py`:

```python
def gate_method_for_benchmark_mode(benchmark_mode: str) -> str:
    return "full_module_trace"
```

Change:

```python
def write_prediction_bridge_gate(output_dir: Path, *, benchmark_mode: str = "strict") -> None:
    gate_method = gate_method_for_benchmark_mode(benchmark_mode)
    metrics = pd.read_csv(output_dir / "retrieval_metrics.csv")
    hetero = metrics[
        (metrics["slice"] == "heterogeneous_supported")
        & (metrics["method"] == gate_method)
    ]
```

Change the report title block:

```python
def report_title_for_mode(benchmark_mode: str) -> str:
    if benchmark_mode == "lean":
        return "# Lean Mixed-Zoo Benchmark Report"
    return "# Mixed-Zoo Retrieval Cycle 1 Report"
```

Use that title in `write_mixed_zoo_report`, and include:

```python
        f"- benchmark_mode: {benchmark_mode}",
```

Call the gate writer with:

```python
        write_prediction_bridge_gate(args.output_dir, benchmark_mode=args.benchmark_mode)
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run:

```bash
conda run -n esd_ind python -m unittest tests.test_pipeline -v
```

Expected:
- PASS for the new mode-aware gate/report behavior

- [ ] **Step 5: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 add scripts/mixed_zoo_retrieval_research.py tests/test_pipeline.py
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 commit -m "feat: add lean mixed-zoo reporting mode"
```

### Task 5: Run The Lean Benchmark And Verify The Output

**Files:**
- Modify: `scripts/mixed_zoo_retrieval_research.py`
- Output: `research_runs/lean_mixed_zoo_benchmark/`

- [ ] **Step 1: Run the full touched test set**

Run:

```bash
conda run -n esd_ind python -m unittest tests.test_heterogeneity tests.test_pipeline -v
```

Expected:
- PASS for all touched tests

- [ ] **Step 2: Run the lean benchmark**

Run:

```bash
conda run -n esd_ind python scripts/mixed_zoo_retrieval_research.py --benchmark-mode lean --output-dir research_runs/lean_mixed_zoo_benchmark
```

Expected:
- run completes successfully
- artifacts are written under `research_runs/lean_mixed_zoo_benchmark`

- [ ] **Step 3: Review the key lean artifacts**

Run:

```bash
sed -n '1,220p' research_runs/lean_mixed_zoo_benchmark/report.md
sed -n '1,80p' research_runs/lean_mixed_zoo_benchmark/prediction_bridge_gate.md
python3 - <<'PY'
import pandas as pd
root = "research_runs/lean_mixed_zoo_benchmark"
retr = pd.read_csv(f"{root}/retrieval_metrics.csv")
print(sorted(retr["slice"].unique()))
print(retr[retr["slice"] == "heterogeneous_supported"].to_string(index=False))
PY
```

Expected:
- report title identifies the lean benchmark
- gate note exists
- retrieval metrics include `heterogeneous_supported`
- heterogeneous-supported row count is larger than the strict benchmark or at least nonzero if the broadened slice worked

- [ ] **Step 4: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 add scripts/mixed_zoo_retrieval_research.py tests/test_pipeline.py tests/test_heterogeneity.py
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 commit -m "feat: add lean mixed-zoo benchmark mode"
```
