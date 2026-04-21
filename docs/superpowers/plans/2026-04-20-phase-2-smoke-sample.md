# Phase 2 Smoke Sample Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one very small workflow that prepares a reproducible `100`-model mixed-zoo smoke sample from the canonical curated table without starting the analysis run.

**Architecture:** Add one small Python script under `fetch_data/` in `Model-Research` so the logic stays near the existing curation code, but have it write its outputs into a disposable sandbox such as `/tmp/model_zoo_phase2_smoke/`. Keep the workflow human-readable: one command builds the sample CSV and one short markdown summary that a person can inspect before later launching a run in `Model-Zoo`.

**Tech Stack:** Python, pandas, pathlib, pytest

---

### Task 1: Add a tiny smoke-sample builder module

**Files:**
- Create: `fetch_data/build_phase2_smoke_sample.py`
- Test: `tests/test_build_phase2_smoke_sample.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_build_phase2_smoke_sample.py` with focused coverage for:

```python
from pathlib import Path

import pandas as pd

from fetch_data.build_phase2_smoke_sample import (
    filter_supported_smoke_candidates,
    proportional_sample_by_type,
    write_smoke_summary,
)


def _frame():
    rows = []
    rows.extend(
        {
            "model_id": f"finetune_{i}",
            "primary_type_bucket": "finetune",
            "lineage_status": "known" if i < 6 else "unknown",
            "eval_status": "eval_present" if i % 2 == 0 else "eval_missing",
            "loader_scenario": "standard_transformers",
            "probe_failure_class": "",
        }
        for i in range(10)
    )
    rows.extend(
        {
            "model_id": f"adapter_{i}",
            "primary_type_bucket": "adapter",
            "lineage_status": "known",
            "eval_status": "eval_missing",
            "loader_scenario": "adapter_requires_base",
            "probe_failure_class": "",
        }
        for i in range(6)
    )
    rows.extend(
        {
            "model_id": f"gguf_{i}",
            "primary_type_bucket": "quantized",
            "lineage_status": "unknown",
            "eval_status": "eval_missing",
            "loader_scenario": "quantized_alt_format",
            "probe_failure_class": "",
        }
        for i in range(3)
    )
    rows.append(
        {
            "model_id": "broken_probe",
            "primary_type_bucket": "merge",
            "lineage_status": "known",
            "eval_status": "eval_present",
            "loader_scenario": "standard_transformers",
            "probe_failure_class": "repo_missing",
        }
    )
    return pd.DataFrame(rows)


def test_filter_supported_smoke_candidates_removes_structurally_unsupported_rows():
    df = _frame()
    filtered = filter_supported_smoke_candidates(df)
    assert "quantized_alt_format" not in set(filtered["loader_scenario"])
    assert "repo_missing" not in set(filtered["probe_failure_class"].fillna(""))
    assert "broken_probe" not in set(filtered["model_id"])


def test_proportional_sample_by_type_hits_target_and_preserves_major_type_mix():
    df = filter_supported_smoke_candidates(_frame())
    sampled = proportional_sample_by_type(df, target_size=8, random_seed=7)
    assert len(sampled) == 8
    counts = sampled["primary_type_bucket"].value_counts().to_dict()
    assert "finetune" in counts
    assert "adapter" in counts


def test_write_smoke_summary_contains_core_sections(tmp_path: Path):
    df = filter_supported_smoke_candidates(_frame())
    sampled = proportional_sample_by_type(df, target_size=8, random_seed=7)
    out = tmp_path / "summary.md"
    write_smoke_summary(
        sampled_df=sampled,
        excluded_count=len(df) - len(sampled),
        output_path=out,
    )
    text = out.read_text()
    assert "total sampled rows" in text
    assert "primary_type_bucket" in text
    assert "lineage_status" in text
    assert "eval_status" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
pytest /scratch/kinshuk/Model-Research/tests/test_build_phase2_smoke_sample.py -q
```

Expected:
- fail because `fetch_data.build_phase2_smoke_sample` does not exist yet

- [ ] **Step 3: Write the minimal implementation**

Create `fetch_data/build_phase2_smoke_sample.py` with:

- `filter_supported_smoke_candidates(df)`
- `proportional_sample_by_type(df, target_size, random_seed)`
- `write_smoke_summary(sampled_df, excluded_count, output_path)`

Use this minimal implementation shape:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd


UNSUPPORTED_LOADER_SCENARIOS = {"quantized_alt_format"}
UNSUPPORTED_PROBE_FAILURES = {
    "repo_missing",
    "repo_gated",
    "repo_private",
    "adapter_base_unresolved",
}


def filter_supported_smoke_candidates(df: pd.DataFrame) -> pd.DataFrame:
    probe_failures = df.get("probe_failure_class", pd.Series("", index=df.index)).fillna("")
    mask = ~df["loader_scenario"].isin(UNSUPPORTED_LOADER_SCENARIOS)
    mask &= ~probe_failures.isin(UNSUPPORTED_PROBE_FAILURES)
    mask &= df["model_id"].fillna("").ne("")
    return df.loc[mask].copy()


def proportional_sample_by_type(
    df: pd.DataFrame,
    target_size: int = 100,
    random_seed: int = 7,
) -> pd.DataFrame:
    counts = df["primary_type_bucket"].value_counts(dropna=False)
    shares = counts / counts.sum()
    quotas = (shares * target_size).round().astype(int)
    if quotas.sum() == 0:
        raise ValueError("No rows available for smoke sampling")

    while quotas.sum() < target_size:
        quotas.iloc[0] += 1
    while quotas.sum() > target_size:
        largest = quotas.sort_values(ascending=False).index[0]
        quotas.loc[largest] -= 1

    samples = []
    for bucket, quota in quotas.items():
        bucket_df = df[df["primary_type_bucket"] == bucket]
        if bucket_df.empty or quota <= 0:
            continue
        take = min(len(bucket_df), quota)
        samples.append(bucket_df.sample(n=take, random_state=random_seed))

    sampled = pd.concat(samples, ignore_index=True).drop_duplicates(subset=["model_id"])
    if len(sampled) > target_size:
        sampled = sampled.sample(n=target_size, random_state=random_seed).reset_index(drop=True)
    return sampled


def write_smoke_summary(sampled_df: pd.DataFrame, excluded_count: int, output_path: Path) -> None:
    lines = [
        "# Phase 2 Smoke Sample",
        "",
        f"- total sampled rows: {len(sampled_df)}",
        f"- excluded clearly unsupported rows before sampling: {excluded_count}",
        "",
        "## Counts by `primary_type_bucket`",
        sampled_df["primary_type_bucket"].value_counts().to_string(),
        "",
        "## Counts by `lineage_status`",
        sampled_df["lineage_status"].value_counts(dropna=False).to_string(),
        "",
        "## Counts by `eval_status`",
        sampled_df["eval_status"].value_counts(dropna=False).to_string(),
        "",
    ]
    output_path.write_text("\\n".join(lines) + "\\n")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
pytest /scratch/kinshuk/Model-Research/tests/test_build_phase2_smoke_sample.py -q
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git -C /scratch/kinshuk/Model-Research add \
  fetch_data/build_phase2_smoke_sample.py \
  tests/test_build_phase2_smoke_sample.py
git -C /scratch/kinshuk/Model-Research commit -m "feat: add phase 2 smoke sample builder"
```

### Task 2: Add the tiny CLI entrypoint behavior and sandbox README output

**Files:**
- Modify: `fetch_data/build_phase2_smoke_sample.py`
- Test: `tests/test_build_phase2_smoke_sample.py`

- [ ] **Step 1: Add a failing CLI-style test**

Extend `tests/test_build_phase2_smoke_sample.py` with:

```python
from pathlib import Path

import pandas as pd

from fetch_data.build_phase2_smoke_sample import build_smoke_sample


def test_build_smoke_sample_writes_csv_and_readme(tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "model_id": f"m{i}",
                "primary_type_bucket": "finetune" if i < 8 else "adapter",
                "lineage_status": "known",
                "eval_status": "eval_present" if i % 2 == 0 else "eval_missing",
                "loader_scenario": "standard_transformers" if i < 8 else "adapter_requires_base",
                "probe_failure_class": "",
            }
            for i in range(12)
        ]
    )
    source = tmp_path / "phase2.csv"
    df.to_csv(source, index=False)

    out_dir = tmp_path / "sandbox"
    build_smoke_sample(
        source_csv=source,
        output_dir=out_dir,
        target_size=10,
        random_seed=3,
        run_name="smoke_100_mixed_v1",
    )

    assert (out_dir / "smoke_100_mixed_v1.csv").is_file()
    assert (out_dir / "smoke_100_mixed_v1_summary.md").is_file()
    assert (out_dir / "README.md").is_file()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest /scratch/kinshuk/Model-Research/tests/test_build_phase2_smoke_sample.py -q
```

Expected:
- fail because `build_smoke_sample` does not exist yet

- [ ] **Step 3: Implement the minimal orchestration and CLI**

Extend `fetch_data/build_phase2_smoke_sample.py` with:

- `build_smoke_sample(...)`
- a small `main()` using `argparse`

Use this minimal additional shape:

```python
import argparse


def write_sandbox_readme(output_dir: Path, run_name: str) -> None:
    text = f"""# Phase 2 Smoke Sample

Rebuild the sample with:

```bash
python fetch_data/build_phase2_smoke_sample.py \\
  --source-csv Model-Zoo/data/curated/model_zoo_phase2.csv \\
  --output-dir {output_dir} \\
  --target-size 100 \\
  --run-name {run_name}
```

Files written here:

- `{run_name}.csv`
- `{run_name}_summary.md`

Planned later run target:

- `Model-Zoo/analysis_runs/phase2/{run_name}/`

After the run, check:

- overall success rate
- success/failure by `primary_type_bucket`
- failure reasons by `loader_scenario`
"""
    (output_dir / "README.md").write_text(text)


def build_smoke_sample(
    source_csv: Path,
    output_dir: Path,
    target_size: int,
    random_seed: int,
    run_name: str,
) -> None:
    df = pd.read_csv(source_csv, low_memory=False)
    filtered = filter_supported_smoke_candidates(df)
    sampled = proportional_sample_by_type(filtered, target_size=target_size, random_seed=random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(output_dir / f"{run_name}.csv", index=False)
    write_smoke_summary(
        sampled_df=sampled,
        excluded_count=len(df) - len(filtered),
        output_path=output_dir / f"{run_name}_summary.md",
    )
    write_sandbox_readme(output_dir=output_dir, run_name=run_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--run-name", type=str, default="smoke_100_mixed_v1")
    args = parser.parse_args()
    build_smoke_sample(
        source_csv=args.source_csv,
        output_dir=args.output_dir,
        target_size=args.target_size,
        random_seed=args.random_seed,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
pytest /scratch/kinshuk/Model-Research/tests/test_build_phase2_smoke_sample.py -q
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git -C /scratch/kinshuk/Model-Research add \
  fetch_data/build_phase2_smoke_sample.py \
  tests/test_build_phase2_smoke_sample.py
git -C /scratch/kinshuk/Model-Research commit -m "feat: add smoke sample CLI flow"
```

### Task 3: Run one local smoke build into `/tmp` and verify the artifacts

**Files:**
- Reuse: `fetch_data/build_phase2_smoke_sample.py`
- Output only: `/tmp/model_zoo_phase2_smoke/`

- [ ] **Step 1: Run the builder against the real curated table**

Run:

```bash
python /scratch/kinshuk/Model-Research/fetch_data/build_phase2_smoke_sample.py \
  --source-csv /scratch/kinshuk/Model-Research/Model-Zoo/data/curated/model_zoo_phase2.csv \
  --output-dir /tmp/model_zoo_phase2_smoke \
  --target-size 100 \
  --run-name smoke_100_mixed_v1
```

Expected:
- writes `/tmp/model_zoo_phase2_smoke/smoke_100_mixed_v1.csv`
- writes `/tmp/model_zoo_phase2_smoke/smoke_100_mixed_v1_summary.md`
- writes `/tmp/model_zoo_phase2_smoke/README.md`

- [ ] **Step 2: Inspect the generated summary and sample shape**

Run:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
out = Path('/tmp/model_zoo_phase2_smoke')
df = pd.read_csv(out / 'smoke_100_mixed_v1.csv', low_memory=False)
print('rows', len(df))
print(df['primary_type_bucket'].value_counts().to_string())
print(df['lineage_status'].value_counts(dropna=False).to_string())
print(df['eval_status'].value_counts(dropna=False).to_string())
PY
sed -n '1,200p' /tmp/model_zoo_phase2_smoke/smoke_100_mixed_v1_summary.md
```

Expected:
- exactly `100` sampled rows
- visible type counts
- visible lineage/eval counts
- readable markdown summary

- [ ] **Step 3: Run the targeted tests again**

Run:

```bash
pytest /scratch/kinshuk/Model-Research/tests/test_build_phase2_smoke_sample.py -q
```

Expected:
- PASS

- [ ] **Step 4: Commit**

```bash
git -C /scratch/kinshuk/Model-Research add \
  fetch_data/build_phase2_smoke_sample.py \
  tests/test_build_phase2_smoke_sample.py
git -C /scratch/kinshuk/Model-Research commit -m "test: verify smoke sample build on curated zoo"
```
