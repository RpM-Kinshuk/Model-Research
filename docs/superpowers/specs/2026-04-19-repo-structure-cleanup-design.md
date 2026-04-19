# Repository Structure Cleanup Design

Date: 2026-04-19

## Goal

Clean up the repository so the current preliminary research work is easier to navigate, maintain, and reuse for the scaled-up study, without throwing away methods, results, or process history.

This is a **phase-1 structure cleanup only**. It is not a data-curation pass, an experiment-contract redesign, or a major code refactor.

## Why This Cleanup Exists

The repository currently mixes together:

- active research code
- active experiment artifacts
- planning/spec history
- older paper material
- older correlation and dashboard work

That makes it harder to answer basic questions like:

- what is the current active pipeline?
- which results are canonical?
- which folders are historical?
- what should be reused for the scaled-up rerun?

The cleanup should make those answers obvious.

## Principles

1. Preserve reproducibility.
   - Do not delete important runs, methods, notes, or labels.

2. Separate active, canonical, and archived material.
   - A reader should be able to tell what is current without reading every file.

3. Avoid hidden refactors.
   - Phase 1 should improve structure and navigation, not quietly rewrite the system.

4. Keep the current preliminary story intact.
   - Zoo management first.
   - Mixed-zoo retrieval as the main experimental center.
   - Performance prediction as a secondary extension.

5. Defer data-curation and output-contract tightening.
   - That belongs to the next research phase, not this one.

## Important Constraint

`alpha_cluster_dashboard/` must remain active in phase 1.

Even though the name suggests older dashboard code, active modules still import it:

- `spectral_research/`
- `scripts/`
- `tests/`
- `app.py`

So phase 1 must **not** archive or move `alpha_cluster_dashboard/` unless those imports are refactored first. That refactor is out of scope here.

## Scope

Phase 1 includes:

- clarifying the top-level repo structure
- separating canonical preliminary docs from planning history
- separating active artifacts from clearly old material
- reorganizing `research_runs/` into clearer groups
- adding lightweight navigation docs

Phase 1 does not include:

- data curation
- redesign of benchmark contracts
- replacing `alpha_cluster_dashboard/`
- major script rewrites
- changing experiment logic
- deleting exploratory history that may still matter

## Target Repository Shape

### Active Root-Level Areas

These remain at the root because they are part of the active runnable system:

- `spectral_research/`
- `alpha_cluster_dashboard/`
- `scripts/`
- `data/`
- `svd_results/`
- `tests/`
- `research_runs/`
- `docs/`

### Canonical Research Narrative Area

Add a new reader-facing area:

- `docs/research/preliminary/`

This becomes the main place to understand the current preliminary phase.

It should contain:

- a short `README.md`
- the current preliminary research draft
- the consolidated findings note
- the paper outline
- an artifact index
- a short experiment-family summary

This area should point to the real files and canonical runs. It should not become a second planning log.

### Process History Area

Keep:

- `docs/superpowers/specs/`
- `docs/superpowers/plans/`

These remain as working history and design records.

They are still useful, but they should no longer be the main reader-facing entrypoint for the research state.

### Archive Area

Create a clear archive area:

- `archive/reference-work/`
- `archive/correlation_runs/`

Move these there:

- `reference-work/`
- `correlation_runs/`

Reason:

- preserve them
- remove them from the active surface area
- make it obvious they are not the current preliminary pipeline

### Temporary / Local Junk

Temporary or machine-local clutter should be cleaned or ignored:

- `tmp/`
- cache artifacts such as `__pycache__/`, `.pyc`, and similar local noise if any remain tracked or unignored

## Research Run Reorganization

Keep `research_runs/` as the artifact root, but group runs by purpose rather than leaving everything flat.

Target shape:

- `research_runs/preliminary/retrieval/`
- `research_runs/preliminary/rejection/`
- `research_runs/preliminary/prediction/`
- `research_runs/preliminary/optimization/`
- `research_runs/preliminary/exploratory/`
- `research_runs/legacy/` only if older non-preliminary runs need to stay under `research_runs/`

The current canonical preliminary run families should map roughly like this:

- retrieval
  - `spectral_family_retrieval_*`
  - `mixed_zoo_retrieval_cycle1`
  - `lean_mixed_zoo_benchmark`

- rejection
  - `rejection_confidence_study_strict_v1`

- prediction
  - `performance_prediction_cycle1`
  - `performance_prediction_cycle1_optimized`
  - `performance_prediction_cycle2`
  - `performance_trend_diagnostics`

- optimization
  - `performance_prediction_bakeoff_*`

- exploratory
  - smoke runs
  - exploratory verification runs
  - one-off sensitivity runs not part of the final preliminary story

Within these groups, existing run directory names should be preserved as much as possible so links and provenance remain understandable.

## Canonical Preliminary Artifact Set

The canonical preliminary phase should be presented as one fixed experiment family:

1. strict homogeneous control retrieval
2. lean mixed-zoo retrieval
3. rejection mini-study as side evidence
4. performance prediction cycles
5. trend diagnostics as descriptive-only support

Everything else should be marked as one of:

- exploratory
- optimization-only
- archived history

## Documentation Additions

Add lightweight docs only. No big framework.

Required files:

- root `README.md` update
- `docs/research/preliminary/README.md`
- `docs/research/preliminary/artifact-index.md`
- `docs/research/preliminary/experiment-family.md`
- `research_runs/README.md`
- `scripts/README.md`
- `spectral_research/README.md`

Each should answer a simple question:

- root `README.md`
  - what is active, what is archive, where to start

- preliminary `README.md`
  - what the current prelim phase is trying to show

- artifact index
  - which run outputs are canonical and why

- experiment family
  - which experiments belong to the prelim set

- `research_runs/README.md`
  - how runs are grouped and how to find a result

- `scripts/README.md`
  - which scripts are still active and what they are for

- `spectral_research/README.md`
  - what the active package contains and how it relates to the runs

## Migration Rules

1. Prefer moving whole directories over rewriting their contents.
2. Preserve filenames inside run directories where possible.
3. Do not break active imports in phase 1.
4. If a document exists only as process history but is now canonical, surface it from `docs/research/preliminary/` instead of deleting the original.
5. If an old item is still actively imported, keep it active even if the name is imperfect.

## Validation

Phase 1 is successful if:

1. The active repo root is visibly cleaner.
2. A new reader can find the current preliminary narrative in one obvious place.
3. A new reader can find the canonical run outputs in one obvious place.
4. Active tests still pass.
5. No experiment logic changes are required just to keep the repo running.

Verification should include:

- `pytest -q`
- spot-checking moved docs and run links
- checking that active scripts still import and run at least at smoke level if paths changed

## Risks And Mitigations

### Risk: Breaking active imports

Mitigation:

- do not move `alpha_cluster_dashboard/` in phase 1
- avoid import-path refactors

### Risk: Losing provenance

Mitigation:

- preserve run directory names
- keep planning/spec history in `docs/superpowers/`
- move legacy content instead of deleting it

### Risk: Creating a cleaner-looking repo that is still confusing

Mitigation:

- add explicit indexes and READMEs
- define a canonical preliminary artifact set
- separate active from archive clearly

## Deferred To Phase 2

These are intentionally not part of this cleanup:

- data-curation workflow
- experiment output contract for scale-up
- script standardization around a shared run schema
- refactoring away from `alpha_cluster_dashboard/`
- deeper code modularization beyond navigation docs

## Acceptance Criteria

This design is complete when phase 1 produces:

- a cleaner top-level layout
- a clear `archive/` area
- a clear `docs/research/preliminary/` area
- grouped `research_runs/`
- lightweight navigation docs for active code and runs
- unchanged scientific logic
- preserved reproducibility

That gives the repo a cleaner shape now, while keeping the later data-curation and scale-up contract work for the next phase where it belongs.
