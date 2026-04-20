# Phase Contract And Repo Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Model-Zoo` the clean operational source of truth for phases 1 and 2, while keeping `Model-Research` the clean experimental consumer for phase 3.

**Architecture:** Keep the system explicitly split into three phases with stable handoff artifacts. Use `Model-Zoo` for canonical curated data, operational documentation, and analysis outputs. Use `Model-Research` for benchmark methods, research runs, and paper-facing documents. Avoid refactoring the Model-Zoo analysis engine; prefer path/docs/contract cleanup.

**Tech Stack:** Markdown docs, Python path/config cleanup, existing `Model-Zoo` ESD runner, existing `Model-Research` spectral research code, pytest

---

### Task 1: Add Model-Zoo operational docs as the primary human entrypoint

**Files:**
- Create: `Model-Zoo/docs/operations/README.md`
- Create: `Model-Zoo/docs/operations/phase-contract.md`
- Create: `Model-Zoo/docs/operations/curation.md`
- Create: `Model-Zoo/docs/operations/analysis.md`
- Test/Verify: manual readback and path consistency checks

- [ ] **Step 1: Write the operational README**

Create `Model-Zoo/docs/operations/README.md` with content that makes the repo readable in one pass:

```md
# Model-Zoo Operations

Start here for the operational pipeline.

This repo owns:

1. data curation and gathering
2. ESD analysis

It does not own the phase-3 research methods. Those live in `Model-Research`.

Core documents:

- [Phase Contract](./phase-contract.md)
- [Curation](./curation.md)
- [Analysis](./analysis.md)

Canonical operational locations:

- curated tables: `data/curated/`
- manual audit inputs: `data/manual_audits/`
- analysis outputs: `analysis_runs/phase2/`
- analysis engine: `esd_experiment/`
```

- [ ] **Step 2: Write the phase contract doc**

Create `Model-Zoo/docs/operations/phase-contract.md`:

```md
# Phase Contract

The system is split into three phases:

1. Phase 1: data curation and gathering
2. Phase 2: ESD analysis
3. Phase 3: experimental analysis

`Model-Zoo` owns phases 1 and 2.
`Model-Research` owns phase 3.

Phase 1 canonical output:
- `data/curated/model_zoo_phase2.csv`
- derived views under `data/curated/`

Phase 2 canonical output:
- `analysis_runs/phase2/<run_name>/stats/`
- `analysis_runs/phase2/<run_name>/metrics/`
- `analysis_runs/phase2/<run_name>/logs/`

Phase 3 consumes those outputs and writes research runs in `Model-Research/research_runs/`.
```

- [ ] **Step 3: Write the curation doc**

Create `Model-Zoo/docs/operations/curation.md`:

```md
# Curation

Phase 1 owns the canonical curated model table.

Canonical location:
- `data/curated/model_zoo_phase2.csv`

Derived views:
- `data/curated/model_zoo_phase2_analysis_ready.csv`
- `data/curated/model_zoo_phase2_prediction_ready.csv`
- `data/curated/model_zoo_phase2_probe_failures.csv`

Supporting inputs may include:
- atlas-derived metadata
- leaderboard enrichment
- Hugging Face supplements
- manual audits

Phase 1 stops at:
- curation
- enrichment
- probe/fetchability status
```

- [ ] **Step 4: Write the analysis doc**

Create `Model-Zoo/docs/operations/analysis.md`:

```md
# Analysis

Phase 2 reads the canonical curated table and writes structured ESD outputs.

Canonical input:
- `data/curated/model_zoo_phase2.csv`

Canonical output root:
- `analysis_runs/phase2/<run_name>/`

Expected contents:
- `stats/*.csv`
- `metrics/*.h5`
- `logs/failure_records.jsonl`
- `logs/failed_models.txt`

Completion rule:
- a model is complete only when both its final stats CSV and metrics H5 exist
```

- [ ] **Step 5: Verify the new docs exist and are coherent**

Run:

```bash
test -f /scratch/kinshuk/Model-Research/Model-Zoo/docs/operations/README.md
test -f /scratch/kinshuk/Model-Research/Model-Zoo/docs/operations/phase-contract.md
test -f /scratch/kinshuk/Model-Research/Model-Zoo/docs/operations/curation.md
test -f /scratch/kinshuk/Model-Research/Model-Zoo/docs/operations/analysis.md
sed -n '1,200p' /scratch/kinshuk/Model-Research/Model-Zoo/docs/operations/README.md
```

Expected:
- all files exist
- README clearly points to phases 1 and 2 and their canonical locations

- [ ] **Step 6: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/Model-Zoo add \
  docs/operations/README.md \
  docs/operations/phase-contract.md \
  docs/operations/curation.md \
  docs/operations/analysis.md
git -C /scratch/kinshuk/Model-Research/Model-Zoo commit -m "docs: add operational phase docs"
```

### Task 2: Move the canonical curated outputs into their final simple location in Model-Zoo

**Files:**
- Create: `Model-Zoo/data/curated/README.md`
- Create/Move: `Model-Zoo/data/curated/model_zoo_phase2.csv`
- Create/Move: `Model-Zoo/data/curated/model_zoo_phase2_analysis_ready.csv`
- Create/Move: `Model-Zoo/data/curated/model_zoo_phase2_prediction_ready.csv`
- Create/Move: `Model-Zoo/data/curated/model_zoo_phase2_probe_failures.csv`
- Optional archive/keep: `Model-Research/data/curated/phase2_v3/*` only as transitional synced copies if needed
- Test: path and row-count verification

- [ ] **Step 1: Copy the current canonical curation outputs into Model-Zoo**

Run:

```bash
mkdir -p /scratch/kinshuk/Model-Research/Model-Zoo/data/curated
cp /scratch/kinshuk/Model-Research/data/curated/phase2_v3/model_zoo_phase2.csv \
   /scratch/kinshuk/Model-Research/Model-Zoo/data/curated/model_zoo_phase2.csv
cp /scratch/kinshuk/Model-Research/data/curated/phase2_v3/model_zoo_phase2_analysis_ready.csv \
   /scratch/kinshuk/Model-Research/Model-Zoo/data/curated/model_zoo_phase2_analysis_ready.csv
cp /scratch/kinshuk/Model-Research/data/curated/phase2_v3/model_zoo_phase2_prediction_ready.csv \
   /scratch/kinshuk/Model-Research/Model-Zoo/data/curated/model_zoo_phase2_prediction_ready.csv
cp /scratch/kinshuk/Model-Research/data/curated/phase2_v3/model_zoo_phase2_probe_failures.csv \
   /scratch/kinshuk/Model-Research/Model-Zoo/data/curated/model_zoo_phase2_probe_failures.csv
```

- [ ] **Step 2: Write the curated-data README**

Create `Model-Zoo/data/curated/README.md`:

```md
# Curated Data

This directory contains the canonical phase-1 outputs used by the operational pipeline.

Canonical table:
- `model_zoo_phase2.csv`

Derived views:
- `model_zoo_phase2_analysis_ready.csv`
- `model_zoo_phase2_prediction_ready.csv`
- `model_zoo_phase2_probe_failures.csv`

These tables are the default phase-1 inputs for the ESD analysis runner and downstream research consumers.
```

- [ ] **Step 3: Verify the moved/copied files**

Run:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
root = Path('/scratch/kinshuk/Model-Research/Model-Zoo/data/curated')
for name in [
    'model_zoo_phase2.csv',
    'model_zoo_phase2_analysis_ready.csv',
    'model_zoo_phase2_prediction_ready.csv',
    'model_zoo_phase2_probe_failures.csv',
]:
    p = root / name
    df = pd.read_csv(p, low_memory=False)
    print(name, len(df))
PY
```

Expected:
- files exist
- row counts match the copied source versions

- [ ] **Step 4: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/Model-Zoo add \
  data/curated/README.md \
  data/curated/model_zoo_phase2.csv \
  data/curated/model_zoo_phase2_analysis_ready.csv \
  data/curated/model_zoo_phase2_prediction_ready.csv \
  data/curated/model_zoo_phase2_probe_failures.csv
git -C /scratch/kinshuk/Model-Research/Model-Zoo commit -m "data: add canonical curated phase outputs"
```

### Task 3: Introduce the canonical analysis output root in Model-Zoo with minimal runner support

**Files:**
- Create: `Model-Zoo/analysis_runs/phase2/.gitkeep`
- Modify: `Model-Zoo/esd_experiment/README.md`
- Modify: `Model-Zoo/README.md`
- Optional small modify: `Model-Zoo/esd_experiment/src/run_experiment.py` only if a default/example path needs cleanup
- Test: CLI/help/docs/path consistency

- [ ] **Step 1: Create the canonical analysis output root**

Run:

```bash
mkdir -p /scratch/kinshuk/Model-Research/Model-Zoo/analysis_runs/phase2
touch /scratch/kinshuk/Model-Research/Model-Zoo/analysis_runs/phase2/.gitkeep
```

- [ ] **Step 2: Update the top-level Model-Zoo README to point to the canonical output root**

Edit `Model-Zoo/README.md` so examples and text prefer:

```md
python run_experiment.py \
    --model_list data/curated/model_zoo_phase2.csv \
    --output_dir analysis_runs/phase2/example_run \
    --gpus 0 1 2 3
```

And mention:

```md
Canonical phase-2 outputs live under `analysis_runs/phase2/<run_name>/`.
```

- [ ] **Step 3: Update the esd_experiment README to prefer the canonical input and output paths**

Edit `Model-Zoo/esd_experiment/README.md` so it uses:

```md
--model_list ../data/curated/model_zoo_phase2.csv
--output_dir ../analysis_runs/phase2/example_run
```

Also mention that:

- the runner accepts curated-table columns directly
- successful outputs are written to `stats/` and `metrics/`
- failures are written to `logs/`

- [ ] **Step 4: Verify the docs/examples mention the new paths**

Run:

```bash
rg -n "analysis_runs/phase2|data/curated/model_zoo_phase2.csv" \
  /scratch/kinshuk/Model-Research/Model-Zoo/README.md \
  /scratch/kinshuk/Model-Research/Model-Zoo/esd_experiment/README.md
```

Expected:
- both docs reference the canonical curated table and canonical analysis output root

- [ ] **Step 5: Commit**

```bash
git -C /scratch/kinshuk/Model-Research/Model-Zoo add \
  analysis_runs/phase2/.gitkeep \
  README.md \
  esd_experiment/README.md
git -C /scratch/kinshuk/Model-Research/Model-Zoo commit -m "docs: define canonical analysis output root"
```

### Task 4: Point Model-Research at Model-Zoo as the operational source of truth

**Files:**
- Modify: `docs/research/preliminary/README.md`
- Modify: `docs/research/preliminary/phase-1-wrap-up.md`
- Modify: `docs/research/preliminary/experiment-family.md`
- Create: `docs/research/scaleup/README.md`
- Test: reader-facing path consistency checks

- [ ] **Step 1: Update the preliminary README**

Edit `docs/research/preliminary/README.md` to add a short section:

```md
Operational source of truth:

- curated tables live in `Model-Zoo/data/curated/`
- ESD analysis outputs live in `Model-Zoo/analysis_runs/phase2/`

This directory is the research-facing entrypoint, not the operational source of truth.
```

- [ ] **Step 2: Update the phase-1 wrap-up note**

Edit `docs/research/preliminary/phase-1-wrap-up.md` to clarify that:

- phase-1 findings remain in `Model-Research`
- but canonical phase-1 and phase-2 operational artifacts now belong to `Model-Zoo`

Add a short explicit note like:

```md
For scale-up, `Model-Zoo` is the canonical home of curated data and analysis outputs.
`Model-Research` consumes those artifacts for phase-3 experiments.
```

- [ ] **Step 3: Update the experiment family note**

Edit `docs/research/preliminary/experiment-family.md` to add a short “stable inputs” section:

```md
Stable inputs for scale-up:
- curated table: `Model-Zoo/data/curated/model_zoo_phase2.csv`
- analysis outputs: `Model-Zoo/analysis_runs/phase2/<run_name>/`
```

- [ ] **Step 4: Create the scale-up research README**

Create `docs/research/scaleup/README.md`:

```md
# Scale-Up Research

This directory is the research-facing entrypoint for the scaled experiment phase.

Operational source of truth:
- curated tables: `Model-Zoo/data/curated/`
- analysis outputs: `Model-Zoo/analysis_runs/phase2/`

This repo owns:
- retrieval and mixed-zoo experiments
- performance prediction experiments
- reports and paper-facing outputs
```

- [ ] **Step 5: Verify the research docs point to Model-Zoo**

Run:

```bash
rg -n "Model-Zoo/data/curated|Model-Zoo/analysis_runs/phase2" \
  /scratch/kinshuk/Model-Research/docs/research/preliminary \
  /scratch/kinshuk/Model-Research/docs/research/scaleup
```

Expected:
- the research entrypoints clearly point to Model-Zoo as the operational source of truth

- [ ] **Step 6: Commit**

```bash
git -C /scratch/kinshuk/Model-Research add \
  docs/research/preliminary/README.md \
  docs/research/preliminary/phase-1-wrap-up.md \
  docs/research/preliminary/experiment-family.md \
  docs/research/scaleup/README.md
git -C /scratch/kinshuk/Model-Research commit -m "docs: point research layer to operational source of truth"
```

### Task 5: Add a small stable-input reader surface in Model-Research for future scale-up

**Files:**
- Create: `spectral_research/io_contract.py`
- Modify: existing phase-3 scripts that hardcode old local curated paths, only if needed
- Test: `tests/test_io_contract.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_io_contract.py`:

```python
from pathlib import Path

from spectral_research.io_contract import (
    model_zoo_curated_table_path,
    model_zoo_analysis_run_path,
)


def test_model_zoo_curated_table_path_points_to_canonical_location():
    path = model_zoo_curated_table_path()
    assert path.as_posix().endswith("Model-Zoo/data/curated/model_zoo_phase2.csv")


def test_model_zoo_analysis_run_path_builds_phase2_run_root():
    path = model_zoo_analysis_run_path("example_run")
    assert path.as_posix().endswith("Model-Zoo/analysis_runs/phase2/example_run")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest /scratch/kinshuk/Model-Research/tests/test_io_contract.py -q
```

Expected:
- FAIL because the helper module does not exist yet

- [ ] **Step 3: Write the minimal implementation**

Create `spectral_research/io_contract.py`:

```python
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ZOO_ROOT = PROJECT_ROOT / "Model-Zoo"


def model_zoo_root() -> Path:
    return MODEL_ZOO_ROOT


def model_zoo_curated_table_path() -> Path:
    return MODEL_ZOO_ROOT / "data" / "curated" / "model_zoo_phase2.csv"


def model_zoo_analysis_run_path(run_name: str) -> Path:
    return MODEL_ZOO_ROOT / "analysis_runs" / "phase2" / run_name
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest /scratch/kinshuk/Model-Research/tests/test_io_contract.py -q
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git -C /scratch/kinshuk/Model-Research add \
  spectral_research/io_contract.py \
  tests/test_io_contract.py
git -C /scratch/kinshuk/Model-Research commit -m "feat: add stable cross-repo IO contract helpers"
```

### Task 6: Final verification and cleanup review

**Files:**
- Verify both repos

- [ ] **Step 1: Verify Model-Zoo docs and loader/runner tests**

Run:

```bash
pytest /scratch/kinshuk/Model-Research/Model-Zoo/esd_experiment/tests -q
```

Expected:
- passing suite
- warnings are acceptable only if they are pre-existing and unrelated to this cleanup

- [ ] **Step 2: Verify Model-Research tests affected by the new IO contract**

Run:

```bash
pytest /scratch/kinshuk/Model-Research/tests/test_io_contract.py -q
```

Expected:
- PASS

- [ ] **Step 3: Verify both repos are readable at the entrypoints**

Run:

```bash
sed -n '1,160p' /scratch/kinshuk/Model-Research/Model-Zoo/docs/operations/README.md
sed -n '1,160p' /scratch/kinshuk/Model-Research/docs/research/preliminary/README.md
sed -n '1,160p' /scratch/kinshuk/Model-Research/docs/research/scaleup/README.md
```

Expected:
- Model-Zoo entrypoint clearly covers phases 1 and 2
- Model-Research entrypoints clearly cover phase 3 and point to Model-Zoo for operational inputs

- [ ] **Step 4: Verify both repos are clean**

Run:

```bash
git -C /scratch/kinshuk/Model-Research status --short
git -C /scratch/kinshuk/Model-Research/Model-Zoo status --short
```

Expected:
- both repos clean

- [ ] **Step 5: Create the final wrap-up commit in each repo if needed**

If there are uncommitted changes remaining in either repo:

```bash
git -C /scratch/kinshuk/Model-Research add <files>
git -C /scratch/kinshuk/Model-Research commit -m "docs: finalize cross-repo phase contract"

git -C /scratch/kinshuk/Model-Research/Model-Zoo add <files>
git -C /scratch/kinshuk/Model-Research/Model-Zoo commit -m "docs: finalize operational phase layout"
```
