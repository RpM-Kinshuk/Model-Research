# Repository Structure Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up the repository structure so the active preliminary research pipeline, canonical findings, and preserved legacy material are easier to understand and maintain without changing experiment logic.

**Architecture:** Keep active code at the root, add one canonical reader-facing preliminary research area, move clearly old material into `archive/`, and regroup existing experiment outputs under `research_runs/preliminary/` by purpose. Do not refactor active imports or redesign data-curation/output contracts in this phase.

**Tech Stack:** Git, Markdown, shell file operations, Python import smoke checks, pytest

---

## File Structure Map

### Active areas that stay at the root

- `README.md`
  - top-level orientation for the repository
- `alpha_cluster_dashboard/`
  - active dependency used by `spectral_research/`, `scripts/`, `tests/`, and `app.py`
- `spectral_research/`
  - active research package
- `scripts/`
  - active experiment entrypoints
- `data/`
  - active curated and sampled metadata inputs
- `svd_results/`
  - active spectral metric source data
- `tests/`
  - active regression/unit tests
- `research_runs/`
  - experiment artifact root after regrouping
- `docs/`
  - process history plus the new canonical research entrypoint

### New canonical research docs area

- Create: `docs/research/preliminary/README.md`
- Create: `docs/research/preliminary/artifact-index.md`
- Create: `docs/research/preliminary/experiment-family.md`
- Create: `docs/research/preliminary/preliminary-research-draft.md`
- Create: `docs/research/preliminary/preliminary-findings.md`
- Create: `docs/research/preliminary/preliminary-paper-outline.md`

### New archive area

- Create: `archive/README.md`
- Move: `reference-work/` -> `archive/reference-work/`
- Move: `correlation_runs/` -> `archive/correlation_runs/`

### New navigation docs

- Create: `scripts/README.md`
- Create: `spectral_research/README.md`
- Create: `research_runs/README.md`
- Modify: `README.md`

### Research run regrouping

- Create:
  - `research_runs/preliminary/retrieval/`
  - `research_runs/preliminary/rejection/`
  - `research_runs/preliminary/prediction/`
  - `research_runs/preliminary/optimization/`
  - `research_runs/preliminary/exploratory/`

- Move current run directories into those buckets:
  - retrieval:
    - `research_runs/lean_mixed_zoo_benchmark`
    - `research_runs/mixed_zoo_retrieval_cycle1`
    - `research_runs/spectral_family_retrieval`
    - `research_runs/spectral_family_retrieval_curated_v1`
    - `research_runs/spectral_family_retrieval_root_base_only_v1`
    - `research_runs/spectral_family_retrieval_strict_headline_v2`
    - `research_runs/spectral_family_retrieval_strict_metadata_backfill_v1`
  - rejection:
    - `research_runs/rejection_confidence_study_strict_v1`
  - prediction:
    - `research_runs/performance_prediction_cycle1`
    - `research_runs/performance_prediction_cycle1_optimized`
    - `research_runs/performance_prediction_cycle2`
    - `research_runs/performance_trend_diagnostics`
  - optimization:
    - `research_runs/performance_prediction_bakeoff_cpu_cache`
    - `research_runs/performance_prediction_bakeoff_gpu0`
    - `research_runs/performance_prediction_bakeoff_repeated`
  - exploratory:
    - `research_runs/mixed_zoo_retrieval_cycle1_smoke`
    - `research_runs/mixed_zoo_retrieval_cycle1_smoke_v2`
    - `research_runs/spectral_family_retrieval_exploratory`
    - `research_runs/spectral_family_retrieval_exploratory_verify`
    - `research_runs/spectral_family_retrieval_root_base_expansion_v1`
    - `research_runs/spectral_family_retrieval_smoke`
    - `research_runs/spectral_family_retrieval_smoke_verify`
    - `research_runs/spectral_family_retrieval_strict_headline_min5_v1`

### Validation targets

- `pytest -q`
- import smoke checks:
  - `python -c "import spectral_research.pipeline"`
  - `python -c "import spectral_research.prediction"`
  - `python -c "import app"`

---

### Task 1: Create Canonical Preliminary Research Docs

**Files:**
- Create: `docs/research/preliminary/README.md`
- Create: `docs/research/preliminary/artifact-index.md`
- Create: `docs/research/preliminary/experiment-family.md`
- Create: `docs/research/preliminary/preliminary-research-draft.md`
- Create: `docs/research/preliminary/preliminary-findings.md`
- Create: `docs/research/preliminary/preliminary-paper-outline.md`
- Source from:
  - `docs/superpowers/specs/2026-04-19-preliminary-research-draft.md`
  - `docs/superpowers/specs/2026-04-18-spectral-family-retrieval-preliminary-findings.md`
  - `docs/superpowers/specs/2026-04-19-preliminary-paper-outline.md`

- [ ] **Step 1: Create the directory skeleton**

Run:

```bash
mkdir -p docs/research/preliminary
```

Expected: directory exists at `docs/research/preliminary`

- [ ] **Step 2: Copy the three canonical narrative docs into the new reader-facing area**

Run:

```bash
cp docs/superpowers/specs/2026-04-19-preliminary-research-draft.md \
   docs/research/preliminary/preliminary-research-draft.md
cp docs/superpowers/specs/2026-04-18-spectral-family-retrieval-preliminary-findings.md \
   docs/research/preliminary/preliminary-findings.md
cp docs/superpowers/specs/2026-04-19-preliminary-paper-outline.md \
   docs/research/preliminary/preliminary-paper-outline.md
```

Expected: the new files exist and preserve the current canonical narrative content without altering the process-history originals

- [ ] **Step 3: Add the new preliminary navigation docs**

Write these files with `apply_patch`.

`docs/research/preliminary/README.md`

```md
# Preliminary Research

Start here for the current preliminary phase.

Core documents:

- [Research Draft](./preliminary-research-draft.md)
- [Findings](./preliminary-findings.md)
- [Paper Outline](./preliminary-paper-outline.md)
- [Artifact Index](./artifact-index.md)
- [Experiment Family](./experiment-family.md)

This directory is the reader-facing entrypoint for the preliminary study.
Process history remains in `docs/superpowers/`.
```

`docs/research/preliminary/artifact-index.md`

```md
# Preliminary Artifact Index

This file lists the canonical artifacts for the preliminary phase.

## Canonical run families

- strict homogeneous retrieval control
- lean mixed-zoo retrieval
- rejection mini-study
- performance prediction cycles
- descriptive trend diagnostics

## Canonical run roots

- `research_runs/preliminary/retrieval/`
- `research_runs/preliminary/rejection/`
- `research_runs/preliminary/prediction/`
- `research_runs/preliminary/optimization/`
- `research_runs/preliminary/exploratory/`

Use the grouped run directories above rather than the old flat layout.
```

`docs/research/preliminary/experiment-family.md`

```md
# Preliminary Experiment Family

This preliminary phase keeps one fixed experiment family:

1. strict homogeneous control retrieval
2. lean mixed-zoo retrieval
3. rejection mini-study as side evidence
4. performance prediction cycles
5. trend diagnostics as descriptive-only support

Out-of-scope for this phase:

- data-curation redesign
- output-contract redesign
- replacing `alpha_cluster_dashboard/`
- new large method families beyond the current ones
```

- [ ] **Step 4: Verify the canonical docs layer exists and is readable**

Run:

```bash
find docs/research/preliminary -maxdepth 1 -type f | sort
```

Expected output contains:

- `docs/research/preliminary/README.md`
- `docs/research/preliminary/artifact-index.md`
- `docs/research/preliminary/experiment-family.md`
- `docs/research/preliminary/preliminary-research-draft.md`
- `docs/research/preliminary/preliminary-findings.md`
- `docs/research/preliminary/preliminary-paper-outline.md`

- [ ] **Step 5: Commit**

```bash
git add docs/research/preliminary
git commit -m "docs: add canonical preliminary research docs"
```

---

### Task 2: Add Navigation Docs For Active Areas

**Files:**
- Modify: `README.md`
- Create: `scripts/README.md`
- Create: `spectral_research/README.md`
- Create: `research_runs/README.md`

- [ ] **Step 1: Replace the root README framing so it describes the active repository, not just the old dashboard**

Update `README.md` with content in this shape:

```md
# Model Research

This repository contains the active preliminary research pipeline for organizing a mixed zoo of models using spectral structure, plus preserved legacy materials from earlier phases.

## Start here

- Current preliminary narrative: `docs/research/preliminary/`
- Active package: `spectral_research/`
- Active scripts: `scripts/`
- Active artifacts: `research_runs/`
- Legacy preserved material: `archive/`

## Active areas

- `spectral_research/`: retrieval, prediction, rejection, heterogeneity, and trend analysis logic
- `alpha_cluster_dashboard/`: active data/loading utilities still used by the research pipeline
- `scripts/`: runnable experiment entrypoints
- `data/`: sampled and manually audited metadata inputs
- `svd_results/`: spectral metric sources
- `tests/`: regression and unit tests
- `research_runs/`: saved experiment outputs

## Notes

- `docs/superpowers/` contains design and planning history.
- `docs/research/preliminary/` is the reader-facing summary of the current preliminary phase.
- `archive/` preserves older materials that are no longer part of the active prelim workflow.
```

- [ ] **Step 2: Add `scripts/README.md`**

Create `scripts/README.md` with content in this shape:

```md
# Active Scripts

These are the active experiment entrypoints for the current preliminary phase.

## Retrieval

- `spectral_similarity_research.py`
- `mixed_zoo_retrieval_research.py`
- `spectral_family_retrieval_cli.py`

## Rejection

- `rejection_confidence_study.py`

## Prediction

- `spectral_prediction_research.py`
- `performance_trend_diagnostics.py`

## Older exploratory scripts still kept

- `alpha_leaderboard_correlation.py`
- `alpha_near_two_hypothesis_check.py`
- `common_schema_performance_research.py`
- `find_merge_models_with_valid_analysis.py`
```

- [ ] **Step 3: Add `spectral_research/README.md` and `research_runs/README.md`**

Create `spectral_research/README.md`:

```md
# spectral_research

Active package for the preliminary spectral-model research workflow.

Main modules:

- `benchmark.py`: benchmark and slice construction
- `pipeline.py`: retrieval and scoring flow
- `heterogeneity.py`: schema-aware and role-aware comparisons
- `rejection.py`: lightweight rejection utilities
- `prediction.py`: neighborhood-based prediction logic
- `trend_diagnostics.py`: descriptive trend analyses
- `evaluation.py`: metrics and reporting helpers
- `features.py`: feature extraction and summaries
```

Create `research_runs/README.md`:

```md
# Research Runs

This directory stores experiment artifacts.

## Grouping

- `preliminary/retrieval/`
- `preliminary/rejection/`
- `preliminary/prediction/`
- `preliminary/optimization/`
- `preliminary/exploratory/`

Use the grouped layout above for the current preliminary phase.
Older process-history documents may still mention the old flat paths.
```

- [ ] **Step 4: Verify the new navigation docs exist**

Run:

```bash
test -f scripts/README.md
test -f spectral_research/README.md
test -f research_runs/README.md
rg -n "docs/research/preliminary|archive/" README.md scripts/README.md spectral_research/README.md research_runs/README.md
```

Expected: all files exist and `rg` returns matches for the new canonical navigation points

- [ ] **Step 5: Commit**

```bash
git add README.md scripts/README.md spectral_research/README.md research_runs/README.md
git commit -m "docs: add repository navigation docs"
```

---

### Task 3: Archive Clearly Old Material

**Files:**
- Create: `archive/README.md`
- Move:
  - `reference-work/` -> `archive/reference-work/`
  - `correlation_runs/` -> `archive/correlation_runs/`
- Clean or ignore:
  - `tmp/`

- [ ] **Step 1: Confirm no active code imports the target archive directories**

Run:

```bash
rg -n "reference-work|correlation_runs" spectral_research scripts tests app.py README.md
```

Expected: either no matches, or only documentation/process-history references that do not affect active imports

- [ ] **Step 2: Create the archive root and README**

Create `archive/README.md` with content:

```md
# Archive

This directory preserves older material that is not part of the active preliminary pipeline.

Contents:

- `reference-work/`: older paper drafts and related materials
- `correlation_runs/`: older correlation analysis outputs

These are preserved for provenance, not treated as the current working surface.
```

- [ ] **Step 3: Move the clearly old directories**

Run:

```bash
mkdir -p archive
mv reference-work archive/reference-work
mv correlation_runs archive/correlation_runs
```

If `tmp/` exists and only contains local temporary output, remove it:

```bash
rm -rf tmp
```

- [ ] **Step 4: Verify archive layout**

Run:

```bash
find archive -maxdepth 2 -mindepth 1 | sort
```

Expected output includes:

- `archive/README.md`
- `archive/reference-work`
- `archive/correlation_runs`

And the old root paths should no longer exist:

```bash
test ! -e reference-work
test ! -e correlation_runs
```

- [ ] **Step 5: Commit**

```bash
git add archive
git add -u
git commit -m "chore: archive legacy research material"
```

---

### Task 4: Regroup Experiment Artifacts Under `research_runs/preliminary`

**Files:**
- Modify by move: `research_runs/*`
- Modify: `docs/research/preliminary/artifact-index.md`
- Modify: `docs/research/preliminary/experiment-family.md`
- Modify: `research_runs/README.md`

- [ ] **Step 1: Create the grouped preliminary artifact directories**

Run:

```bash
mkdir -p research_runs/preliminary/retrieval
mkdir -p research_runs/preliminary/rejection
mkdir -p research_runs/preliminary/prediction
mkdir -p research_runs/preliminary/optimization
mkdir -p research_runs/preliminary/exploratory
```

- [ ] **Step 2: Move the current run directories into the grouped layout**

Run these commands exactly:

```bash
mv research_runs/lean_mixed_zoo_benchmark research_runs/preliminary/retrieval/
mv research_runs/mixed_zoo_retrieval_cycle1 research_runs/preliminary/retrieval/
mv research_runs/spectral_family_retrieval research_runs/preliminary/retrieval/
mv research_runs/spectral_family_retrieval_curated_v1 research_runs/preliminary/retrieval/
mv research_runs/spectral_family_retrieval_root_base_only_v1 research_runs/preliminary/retrieval/
mv research_runs/spectral_family_retrieval_strict_headline_v2 research_runs/preliminary/retrieval/
mv research_runs/spectral_family_retrieval_strict_metadata_backfill_v1 research_runs/preliminary/retrieval/

mv research_runs/rejection_confidence_study_strict_v1 research_runs/preliminary/rejection/

mv research_runs/performance_prediction_cycle1 research_runs/preliminary/prediction/
mv research_runs/performance_prediction_cycle1_optimized research_runs/preliminary/prediction/
mv research_runs/performance_prediction_cycle2 research_runs/preliminary/prediction/
mv research_runs/performance_trend_diagnostics research_runs/preliminary/prediction/

mv research_runs/performance_prediction_bakeoff_cpu_cache research_runs/preliminary/optimization/
mv research_runs/performance_prediction_bakeoff_gpu0 research_runs/preliminary/optimization/
mv research_runs/performance_prediction_bakeoff_repeated research_runs/preliminary/optimization/

mv research_runs/mixed_zoo_retrieval_cycle1_smoke research_runs/preliminary/exploratory/
mv research_runs/mixed_zoo_retrieval_cycle1_smoke_v2 research_runs/preliminary/exploratory/
mv research_runs/spectral_family_retrieval_exploratory research_runs/preliminary/exploratory/
mv research_runs/spectral_family_retrieval_exploratory_verify research_runs/preliminary/exploratory/
mv research_runs/spectral_family_retrieval_root_base_expansion_v1 research_runs/preliminary/exploratory/
mv research_runs/spectral_family_retrieval_smoke research_runs/preliminary/exploratory/
mv research_runs/spectral_family_retrieval_smoke_verify research_runs/preliminary/exploratory/
mv research_runs/spectral_family_retrieval_strict_headline_min5_v1 research_runs/preliminary/exploratory/
```

- [ ] **Step 3: Update the canonical artifact docs to the new grouped paths**

Edit these files with `apply_patch`:

`docs/research/preliminary/artifact-index.md`

Replace the short placeholder list with concrete grouped references in this shape:

```md
# Preliminary Artifact Index

## Canonical retrieval artifacts

- `research_runs/preliminary/retrieval/spectral_family_retrieval_strict_headline_v2/`
- `research_runs/preliminary/retrieval/lean_mixed_zoo_benchmark/`
- `research_runs/preliminary/retrieval/mixed_zoo_retrieval_cycle1/`

## Canonical rejection artifact

- `research_runs/preliminary/rejection/rejection_confidence_study_strict_v1/`

## Canonical prediction artifacts

- `research_runs/preliminary/prediction/performance_prediction_cycle1_optimized/`
- `research_runs/preliminary/prediction/performance_prediction_cycle2/`
- `research_runs/preliminary/prediction/performance_trend_diagnostics/`

## Supporting artifacts

- `research_runs/preliminary/optimization/`
- `research_runs/preliminary/exploratory/`
```

`docs/research/preliminary/experiment-family.md`

Append one short section in this shape:

```md
## Canonical grouped artifact layout

- retrieval: `research_runs/preliminary/retrieval/`
- rejection: `research_runs/preliminary/rejection/`
- prediction: `research_runs/preliminary/prediction/`
- optimization: `research_runs/preliminary/optimization/`
- exploratory: `research_runs/preliminary/exploratory/`
```

`research_runs/README.md`

Expand it so it names the concrete canonical grouped paths and says the old flat layout has been retired for current prelim work.

- [ ] **Step 4: Verify the regrouped artifact layout**

Run:

```bash
find research_runs/preliminary -maxdepth 2 -mindepth 1 -type d | sort
```

Expected output includes all five grouped directories and the moved run directories underneath them.

Also check the old flat directories are gone for a few representative examples:

```bash
test ! -e research_runs/lean_mixed_zoo_benchmark
test ! -e research_runs/performance_prediction_cycle2
test ! -e research_runs/rejection_confidence_study_strict_v1
```

- [ ] **Step 5: Commit**

```bash
git add research_runs docs/research/preliminary
git commit -m "chore: regroup preliminary research artifacts"
```

---

### Task 5: Final Validation And Smoke Checks

**Files:**
- Verify:
  - `README.md`
  - `docs/research/preliminary/*`
  - `archive/*`
  - `research_runs/preliminary/*`
  - active Python imports

- [ ] **Step 1: Run import smoke checks**

Run:

```bash
python -c "import spectral_research.pipeline; print('pipeline ok')"
python -c "import spectral_research.prediction; print('prediction ok')"
python -c "import app; print('app ok')"
```

Expected output:

```text
pipeline ok
prediction ok
app ok
```

- [ ] **Step 2: Run the test suite**

Run:

```bash
conda run -n esd_ind env PYTHONPATH=. pytest -q
```

Expected output includes:

```text
98 passed, 2 skipped
```

If the counts differ because of unrelated repo evolution, require that the suite is still green with no new failures caused by the cleanup.

- [ ] **Step 3: Spot-check the new navigation path**

Run:

```bash
sed -n '1,120p' README.md
sed -n '1,160p' docs/research/preliminary/README.md
sed -n '1,200p' research_runs/README.md
```

Expected: all three clearly point a new reader to:

- `docs/research/preliminary/`
- grouped `research_runs/preliminary/` paths
- `archive/`

- [ ] **Step 4: Check git status stays focused**

Run:

```bash
git status --short
```

Expected: only the intended cleanup moves and docs are staged or committed; no stray cache or temp output should remain

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: finalize repository structure cleanup"
```

---

## Self-Review

### Spec coverage

Covered:

- active vs canonical vs archive split
- `docs/research/preliminary/` entrypoint
- preserving `alpha_cluster_dashboard/` as active
- archive move for `reference-work/` and `correlation_runs/`
- grouped `research_runs/preliminary/*`
- lightweight navigation docs
- validation with import smoke checks and pytest

Deferred on purpose:

- data curation
- experiment output contract redesign
- replacing `alpha_cluster_dashboard/`
- deeper code modularization

### Placeholder scan

No `TODO`, `TBD`, or “implement later” placeholders remain. All file paths and move targets are concrete.

### Type and path consistency

- Canonical docs live under `docs/research/preliminary/`
- grouped run artifacts live under `research_runs/preliminary/*`
- archive content moves under `archive/`
- active code remains at the root

