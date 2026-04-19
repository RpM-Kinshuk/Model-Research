# Mixed-Zoo Retrieval Cycle 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a mixed-zoo retrieval benchmark that tests model-intrinsic and spectral methods on homogeneous, heterogeneous, and unsupported model slices.

**Architecture:** Extend the current audited retrieval pipeline rather than replacing it. Add mixed-zoo slice construction in `benchmark.py`, add one focused `heterogeneity.py` helper module for intrinsic-basics and role-aware comparisons, integrate the new methods into `pipeline.py`, and keep cycle-specific orchestration in a new script so the existing strict benchmark flow remains stable.

**Tech Stack:** Python 3, `numpy`, `pandas`, `unittest`, existing `conda run -n esd_ind` workflow, current `spectral_research` package, local `svd_results` and CSV artifacts.

---

## File Structure

- Create: `spectral_research/heterogeneity.py`
  - Canonical role mapping
  - Shared-role and role-aggregated trace helpers
  - `intrinsic_basics_only` feature-vector construction
- Modify: `spectral_research/benchmark.py`
  - Build mixed-zoo slices from the audited manifest
- Modify: `spectral_research/pipeline.py`
  - Add `intrinsic_basics_only`, `shared_role_trace`, and `role_aggregated_trace`
  - Add slice-aware reporting for mixed-zoo runs
- Create: `scripts/mixed_zoo_retrieval_research.py`
  - Cycle-1 orchestration and artifact writing
- Create: `tests/test_heterogeneity.py`
  - Unit tests for canonical roles, intrinsic-basics vectors, and overlap-aware traces
- Modify: `tests/test_benchmark.py`
  - Mixed-zoo slice tests
- Modify: `tests/test_pipeline.py`
  - Method integration tests and CLI smoke coverage for the new script

### Task 1: Add Mixed-Zoo Slice Construction

**Files:**
- Modify: `spectral_research/benchmark.py`
- Modify: `tests/test_benchmark.py`

- [ ] **Step 1: Write the failing slice-construction tests**

Add to `tests/test_benchmark.py`:

```python
import pandas as pd

from spectral_research.benchmark import build_mixed_zoo_slices


def test_build_mixed_zoo_slices_separates_homogeneous_heterogeneous_and_unsupported():
    manifest = pd.DataFrame(
        [
            {
                "model_id": "base--a",
                "audited_base_family": "fam/a",
                "label_status": "confirmed",
                "query_bucket": "supported_candidate",
                "common_schema": True,
                "base_model_relation": "finetune",
                "bucket_override": "",
            },
            {
                "model_id": "base--b",
                "audited_base_family": "fam/a",
                "label_status": "confirmed",
                "query_bucket": "supported_candidate",
                "common_schema": False,
                "base_model_relation": "finetune",
                "bucket_override": "",
            },
            {
                "model_id": "special--c",
                "audited_base_family": "",
                "label_status": "unknown",
                "query_bucket": "unsupported_special",
                "common_schema": True,
                "base_model_relation": "merge",
                "bucket_override": "",
            },
        ]
    )

    slices = build_mixed_zoo_slices(manifest, min_family_size=1)

    assert slices["homogeneous_supported"]["model_id"].tolist() == ["base--a"]
    assert slices["heterogeneous_supported"]["model_id"].tolist() == ["base--b"]
    assert slices["unsupported_special"]["model_id"].tolist() == ["special--c"]


def test_build_mixed_zoo_slices_respects_family_size_floor_inside_each_supported_slice():
    manifest = pd.DataFrame(
        [
            {
                "model_id": "common--1",
                "audited_base_family": "fam/common",
                "label_status": "confirmed",
                "query_bucket": "supported_candidate",
                "common_schema": True,
                "base_model_relation": "finetune",
                "bucket_override": "",
            },
            {
                "model_id": "hetero--1",
                "audited_base_family": "fam/hetero",
                "label_status": "confirmed",
                "query_bucket": "supported_candidate",
                "common_schema": False,
                "base_model_relation": "finetune",
                "bucket_override": "",
            },
        ]
    )

    slices = build_mixed_zoo_slices(manifest, min_family_size=2)

    assert slices["homogeneous_supported"].empty
    assert slices["heterogeneous_supported"].empty
```

- [ ] **Step 2: Run the tests to confirm failure**

Run:

```bash
conda run -n esd_ind python -m unittest tests.test_benchmark -v
```

Expected:
- FAIL with `ImportError` or `AttributeError` for `build_mixed_zoo_slices`

- [ ] **Step 3: Implement the mixed-zoo slice builder**

Add to `spectral_research/benchmark.py`:

```python
def build_mixed_zoo_slices(manifest: pd.DataFrame, min_family_size: int) -> dict[str, pd.DataFrame]:
    frame = manifest.copy()
    supported = frame[
        (frame["label_status"] == "confirmed")
        & (frame["query_bucket"] == "supported_candidate")
        & frame["audited_base_family"].fillna("").astype(str).ne("")
    ].copy()
    homogeneous = supported[supported["common_schema"].fillna(False).astype(bool)].copy()
    heterogeneous = supported[~supported["common_schema"].fillna(False).astype(bool)].copy()

    def _keep_supported_families(slice_frame: pd.DataFrame) -> pd.DataFrame:
        if slice_frame.empty:
            return slice_frame
        counts = slice_frame["audited_base_family"].value_counts()
        keep = counts[counts >= min_family_size].index
        return slice_frame[slice_frame["audited_base_family"].isin(keep)].copy()

    homogeneous = _keep_supported_families(homogeneous)
    heterogeneous = _keep_supported_families(heterogeneous)

    return {
        "homogeneous_supported": homogeneous.sort_values("model_id").reset_index(drop=True),
        "heterogeneous_supported": heterogeneous.sort_values("model_id").reset_index(drop=True),
        "unsupported_special": frame[frame["query_bucket"] == "unsupported_special"]
        .sort_values("model_id")
        .reset_index(drop=True),
        "unsupported_unknown": frame[frame["query_bucket"] == "unsupported_unknown"]
        .sort_values("model_id")
        .reset_index(drop=True),
    }
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run:

```bash
conda run -n esd_ind python -m unittest tests.test_benchmark -v
```

Expected:
- PASS for the new slice-construction tests

- [ ] **Step 5: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 add tests/test_benchmark.py spectral_research/benchmark.py
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 commit -m "feat: add mixed-zoo benchmark slices"
```

### Task 2: Add Intrinsic-Basics And Heterogeneity Helpers

**Files:**
- Create: `spectral_research/heterogeneity.py`
- Create: `tests/test_heterogeneity.py`

- [ ] **Step 1: Write the failing helper tests**

Create `tests/test_heterogeneity.py`:

```python
import numpy as np
import pandas as pd

from spectral_research.heterogeneity import (
    canonical_role,
    intrinsic_basics_feature_map,
    role_aggregated_trace_score,
    shared_role_trace_score,
)


def test_canonical_role_maps_attention_and_mlp_suffixes():
    assert canonical_role("self_attn.q_proj") == "attn_q"
    assert canonical_role("self_attn.o_proj") == "attn_o"
    assert canonical_role("mlp.down_proj") == "mlp_down"
    assert canonical_role("totally_unknown") is None


def test_intrinsic_basics_feature_map_ignores_external_catalog_fields():
    manifest = pd.DataFrame(
        [
            {
                "model_id": "model/a",
                "Architecture": "LlamaForCausalLM",
                "#Params (B)": 8.03,
                "num_layers": 32,
                "schema_name": "common_llm_7",
                "raw_module_count": 7,
                "common_schema": True,
                "Type": "chat",
                "Generation": 9,
            },
            {
                "model_id": "model/b",
                "Architecture": "LlamaForCausalLM",
                "#Params (B)": 8.03,
                "num_layers": 32,
                "schema_name": "common_llm_7",
                "raw_module_count": 7,
                "common_schema": True,
                "Type": "merge",
                "Generation": 0,
            },
        ]
    )

    vectors = intrinsic_basics_feature_map(manifest)
    np.testing.assert_allclose(vectors["model/a"], vectors["model/b"])


def test_shared_role_trace_score_uses_only_overlapping_roles():
    query_alpha = np.array([[1.0, 2.0], [2.0, 3.0]])
    query_modules = ("self_attn.q_proj", "mlp.down_proj")
    proto_alpha = np.array([[1.0, 99.0, 2.0], [2.0, 98.0, 3.0]])
    proto_modules = ("self_attn.q_proj", "self_attn.k_proj", "mlp.down_proj")

    score = shared_role_trace_score(
        query_alpha=query_alpha,
        query_modules=query_modules,
        proto_alpha=proto_alpha,
        proto_modules=proto_modules,
        overlap_penalty=0.2,
    )

    assert np.isfinite(score)
    assert score > -1.0


def test_role_aggregated_trace_score_handles_schema_mismatch():
    query_alpha = np.array([[1.0, 2.0], [2.0, 4.0]])
    query_modules = ("self_attn.q_proj", "self_attn.o_proj")
    proto_alpha = np.array([[1.0, 1.5, 3.0], [2.0, 2.5, 4.0]])
    proto_modules = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj")

    score = role_aggregated_trace_score(
        query_alpha=query_alpha,
        query_modules=query_modules,
        proto_alpha=proto_alpha,
        proto_modules=proto_modules,
        overlap_penalty=0.2,
    )

    assert np.isfinite(score)
```

- [ ] **Step 2: Run the tests to confirm failure**

Run:

```bash
conda run -n esd_ind python -m unittest tests.test_heterogeneity -v
```

Expected:
- FAIL because `spectral_research.heterogeneity` does not exist yet

- [ ] **Step 3: Implement the helper module**

Create `spectral_research/heterogeneity.py`:

```python
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


def canonical_role(module_name: str) -> str | None:
    for suffix, role in ROLE_MAP.items():
        if module_name.endswith(suffix):
            return role
    return None


def intrinsic_basics_feature_map(manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    frame = pd.DataFrame({"model_id": manifest["model_id"].astype(str)})
    frame["params_b"] = pd.to_numeric(manifest.get("#Params (B)"), errors="coerce").fillna(-1.0)
    frame["num_layers"] = pd.to_numeric(manifest.get("num_layers"), errors="coerce").fillna(-1.0)
    frame["raw_module_count"] = pd.to_numeric(manifest.get("raw_module_count"), errors="coerce").fillna(-1.0)
    categorical = pd.DataFrame(
        {
            "Architecture": manifest.get("Architecture", pd.Series(index=manifest.index, dtype=object)),
            "schema_name": manifest.get("schema_name", pd.Series(index=manifest.index, dtype=object)),
            "common_schema": manifest.get("common_schema", pd.Series(index=manifest.index, dtype=object)),
        }
    ).fillna("unknown")
    dummies = pd.get_dummies(categorical.astype(str), prefix=["arch", "schema", "common"], dtype=float)
    matrix = pd.concat([frame, dummies, frame[["params_b", "num_layers", "raw_module_count"]]], axis=1)
    cols = [c for c in matrix.columns if c != "model_id"]
    return {row.model_id: values for row, values in zip(matrix.itertuples(index=False), matrix[cols].to_numpy(dtype=float), strict=False)}


def _role_series(alpha: np.ndarray, modules: tuple[str, ...]) -> dict[str, np.ndarray]:
    out: dict[str, list[np.ndarray]] = {}
    for idx, module in enumerate(modules):
        role = canonical_role(module)
        if role is None:
            continue
        out.setdefault(role, []).append(np.asarray(alpha[:, idx], dtype=float))
    return {role: np.nanmean(np.vstack(values), axis=0) for role, values in out.items()}


def _negative_l2(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if not mask.any():
        return float("nan")
    delta = left[mask] - right[mask]
    return float(-np.mean(delta * delta))


def shared_role_trace_score(*, query_alpha: np.ndarray, query_modules: tuple[str, ...], proto_alpha: np.ndarray, proto_modules: tuple[str, ...], overlap_penalty: float) -> float:
    query_roles = _role_series(query_alpha, query_modules)
    proto_roles = _role_series(proto_alpha, proto_modules)
    shared = sorted(set(query_roles) & set(proto_roles))
    if not shared:
        return float("nan")
    scores = [_negative_l2(query_roles[role], proto_roles[role]) for role in shared]
    overlap = len(shared) / max(len(query_roles), len(proto_roles))
    return float(np.nanmean(scores) - overlap_penalty * (1.0 - overlap))


def role_aggregated_trace_score(*, query_alpha: np.ndarray, query_modules: tuple[str, ...], proto_alpha: np.ndarray, proto_modules: tuple[str, ...], overlap_penalty: float) -> float:
    query_roles = _role_series(query_alpha, query_modules)
    proto_roles = _role_series(proto_alpha, proto_modules)
    groups = {
        "attention": ("attn_q", "attn_k", "attn_v", "attn_o"),
        "mlp": ("mlp_down", "mlp_gate", "mlp_up"),
    }
    query_grouped = {}
    proto_grouped = {}
    for group, roles in groups.items():
        q = [query_roles[role] for role in roles if role in query_roles]
        p = [proto_roles[role] for role in roles if role in proto_roles]
        if q:
            query_grouped[group] = np.nanmean(np.vstack(q), axis=0)
        if p:
            proto_grouped[group] = np.nanmean(np.vstack(p), axis=0)
    shared = sorted(set(query_grouped) & set(proto_grouped))
    if not shared:
        return float("nan")
    scores = [_negative_l2(query_grouped[group], proto_grouped[group]) for group in shared]
    overlap = len(shared) / max(len(query_grouped), len(proto_grouped))
    return float(np.nanmean(scores) - overlap_penalty * (1.0 - overlap))
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run:

```bash
conda run -n esd_ind python -m unittest tests.test_heterogeneity -v
```

Expected:
- PASS for all four helper tests

- [ ] **Step 5: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 add tests/test_heterogeneity.py spectral_research/heterogeneity.py
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 commit -m "feat: add mixed-zoo heterogeneity helpers"
```

### Task 3: Integrate Mixed-Zoo Methods Into The Retrieval Pipeline

**Files:**
- Modify: `spectral_research/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing pipeline tests**

Add to `tests/test_pipeline.py`:

```python
import numpy as np
import pandas as pd

from spectral_research.pipeline import _score_query


def test_score_query_supports_intrinsic_basics_only():
    row = pd.Series({"model_id": "query/a"})
    prototype = type(
        "Proto",
        (),
        {
            "family": "fam/a",
            "metadata_vector": np.array([9.0, 9.0]),
            "scalar_vector": np.array([0.0]),
            "summary_vector": np.array([0.0]),
            "full_alpha": np.array([[1.0]]),
            "modules": ("self_attn.q_proj",),
            "architecture": "LlamaForCausalLM",
            "size_bucket": "medium",
        },
    )()
    score = _score_query(
        method="intrinsic_basics_only",
        row=row,
        record=None,
        prototype=prototype,
        metadata_vectors={"query/a": np.array([1.0, 2.0])},
        parent_scores=None,
    )
    assert np.isfinite(score)


def test_score_query_supports_shared_role_trace():
    row = pd.Series({"model_id": "query/a"})
    record = type(
        "Record",
        (),
        {
            "model_id": "query/a",
            "raw_modules": ("self_attn.q_proj", "mlp.down_proj"),
            "alpha": np.array([[1.0, 2.0], [2.0, 3.0]]),
        },
    )()
    prototype = type(
        "Proto",
        (),
        {
            "family": "fam/a",
            "modules": ("self_attn.q_proj", "self_attn.k_proj", "mlp.down_proj"),
            "full_alpha": np.array([[1.0, 9.0, 2.0], [2.0, 8.0, 3.0]]),
            "metadata_vector": np.array([0.0]),
            "scalar_vector": np.array([0.0]),
            "summary_vector": np.array([0.0]),
            "architecture": "LlamaForCausalLM",
            "size_bucket": "medium",
        },
    )()
    score = _score_query(
        method="shared_role_trace",
        row=row,
        record=record,
        prototype=prototype,
        metadata_vectors={},
        parent_scores=None,
    )
    assert np.isfinite(score)
```

- [ ] **Step 2: Run the tests to confirm failure**

Run:

```bash
conda run -n esd_ind python -m unittest tests.test_pipeline -v
```

Expected:
- FAIL because the new methods are not recognized by `_score_query`

- [ ] **Step 3: Implement the new pipeline methods**

Modify `spectral_research/pipeline.py`:

```python
from spectral_research.heterogeneity import (
    intrinsic_basics_feature_map,
    role_aggregated_trace_score,
    shared_role_trace_score,
)

DEFAULT_METHODS = (
    "intrinsic_basics_only",
    "pl_alpha_scalar",
    "attention_trace",
    "full_module_trace",
    "shared_role_trace",
    "role_aggregated_trace",
    "summary_module",
)
```

Replace the existing metadata feature-map call inside the main benchmark flow:

```python
metadata_vectors = intrinsic_basics_feature_map(audited_manifest)
```

Extend `_score_query`:

```python
if method == "intrinsic_basics_only":
    vector = metadata_vectors.get(str(row["model_id"]))
    if vector is None:
        return float("nan")
    return _negative_l2(vector, prototype.metadata_vector)
if method == "shared_role_trace":
    return shared_role_trace_score(
        query_alpha=np.asarray(record.alpha, dtype=float),
        query_modules=tuple(record.raw_modules),
        proto_alpha=np.asarray(prototype.full_alpha, dtype=float),
        proto_modules=tuple(prototype.modules),
        overlap_penalty=0.2,
    )
if method == "role_aggregated_trace":
    return role_aggregated_trace_score(
        query_alpha=np.asarray(record.alpha, dtype=float),
        query_modules=tuple(record.raw_modules),
        proto_alpha=np.asarray(prototype.full_alpha, dtype=float),
        proto_modules=tuple(prototype.modules),
        overlap_penalty=0.2,
    )
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run:

```bash
conda run -n esd_ind python -m unittest tests.test_heterogeneity tests.test_pipeline -v
```

Expected:
- PASS for the new pipeline-method coverage

- [ ] **Step 5: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 add spectral_research/pipeline.py tests/test_pipeline.py
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 commit -m "feat: add mixed-zoo retrieval methods"
```

### Task 4: Add The Cycle-1 Mixed-Zoo Runner And Reporting

**Files:**
- Create: `scripts/mixed_zoo_retrieval_research.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing CLI smoke test**

Add to `tests/test_pipeline.py`:

```python
from pathlib import Path
import subprocess
import sys


def test_mixed_zoo_retrieval_cli_accepts_output_dir():
    script = Path("scripts/mixed_zoo_retrieval_research.py")
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--output-dir" in result.stdout
```

- [ ] **Step 2: Run the test to confirm failure**

Run:

```bash
conda run -n esd_ind python -m unittest tests.test_pipeline -v
```

Expected:
- FAIL because the new runner script does not exist yet

- [ ] **Step 3: Implement the cycle runner**

Create `scripts/mixed_zoo_retrieval_research.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_research.pipeline import parse_args as parse_pipeline_args, main as pipeline_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run mixed-zoo retrieval cycle 1.")
    parser.add_argument("--metrics-dir", type=Path, default=REPO_ROOT / "svd_results")
    parser.add_argument("--lineage-csv", type=Path, default=REPO_ROOT / "data" / "sampled_metadata.csv")
    parser.add_argument("--leaderboard-csv", type=Path, default=REPO_ROOT / "data" / "sampled_leaderboard.csv")
    parser.add_argument("--freeze-csv", type=Path, default=REPO_ROOT / "data" / "manual_audits" / "spectral_family_retrieval_label_freeze.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "research_runs" / "mixed_zoo_retrieval_cycle1")
    parser.add_argument("--min-family-size", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return pipeline_main(
        [
            "--metrics-dir",
            str(args.metrics_dir),
            "--lineage-csv",
            str(args.lineage_csv),
            "--leaderboard-csv",
            str(args.leaderboard_csv),
            "--freeze-csv",
            str(args.freeze_csv),
            "--output-dir",
            str(args.output_dir),
            "--min-family-size",
            str(args.min_family_size),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the smoke test and the cycle runner**

Run:

```bash
conda run -n esd_ind python -m unittest tests.test_pipeline -v
conda run -n esd_ind python scripts/mixed_zoo_retrieval_research.py --output-dir research_runs/mixed_zoo_retrieval_cycle1_smoke
```

Expected:
- CLI test passes
- the script writes a report and CSV artifacts under `research_runs/mixed_zoo_retrieval_cycle1_smoke`

- [ ] **Step 5: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 add scripts/mixed_zoo_retrieval_research.py tests/test_pipeline.py
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 commit -m "feat: add mixed-zoo retrieval runner"
```

### Task 5: Record The Cycle-1 Gate For The Prediction Bridge

**Files:**
- Modify: `scripts/mixed_zoo_retrieval_research.py`
- Create: `research_runs/mixed_zoo_retrieval_cycle1/prediction_bridge_gate.md`

- [ ] **Step 1: Add the gate writer**

Extend `scripts/mixed_zoo_retrieval_research.py` with:

```python
import pandas as pd


def write_prediction_bridge_gate(output_dir: Path) -> None:
    metrics = pd.read_csv(output_dir / "retrieval_metrics.csv")
    hetero = metrics[(metrics["slice"] == "heterogeneous_supported") & (metrics["method"] == "full_module_trace")]
    if hetero.empty:
        decision = "HOLD"
        reason = "No heterogeneous-supported retrieval rows were produced."
    else:
        top1 = float(hetero.iloc[0]["top1_accuracy"])
        mrr = float(hetero.iloc[0]["mrr"])
        if top1 >= 0.50 and mrr >= 0.65:
            decision = "GO"
            reason = "Mixed-zoo spectral neighborhoods are strong enough for a tiny prediction pilot."
        else:
            decision = "HOLD"
            reason = "Mixed-zoo retrieval is not yet strong enough to justify the prediction bridge."
    (output_dir / "prediction_bridge_gate.md").write_text(
        f\"# Prediction Bridge Gate\\n\\n- Decision: `{decision}`\\n- Reason: {reason}\\n\",
        encoding=\"utf-8\",
    )
```

Call it after the pipeline run completes.

- [ ] **Step 2: Run the cycle runner again**

Run:

```bash
conda run -n esd_ind python scripts/mixed_zoo_retrieval_research.py --output-dir research_runs/mixed_zoo_retrieval_cycle1
```

Expected:
- `prediction_bridge_gate.md` exists in the output directory
- the file contains either `GO` or `HOLD`

- [ ] **Step 3: Verify the full test set**

Run:

```bash
conda run -n esd_ind python -m unittest tests.test_benchmark tests.test_heterogeneity tests.test_pipeline -v
```

Expected:
- PASS for all new and touched tests

- [ ] **Step 4: Review the key artifacts**

Run:

```bash
sed -n '1,220p' research_runs/mixed_zoo_retrieval_cycle1/report.md
sed -n '1,80p' research_runs/mixed_zoo_retrieval_cycle1/prediction_bridge_gate.md
```

Expected:
- report contains mixed-zoo retrieval metrics
- gate note states whether the tiny prediction pilot is justified

- [ ] **Step 5: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 add scripts/mixed_zoo_retrieval_research.py
git -C /scratch/kinshuk/Model-Research/.worktrees/spectral-family-retrieval-prelim-v2 commit -m "feat: add mixed-zoo prediction gate artifact"
```
