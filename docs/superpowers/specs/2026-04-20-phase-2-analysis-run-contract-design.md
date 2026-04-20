# Phase 2 Analysis Run Contract Design

## Goal

Define one small, stable contract for phase 2 analysis runs so that:

- `Model-Zoo` remains the operational source of truth for curation and ESD analysis
- `Model-Research` can consume phase-2 outputs without ad hoc joins or path hunting
- the scaled-up study can reuse the same experiment family without redesigning the pipeline again

This is a contract-definition step only. It does not start a run, redesign the scheduler, or widen the analysis engine.

## Recommendation

Use a two-document shape:

1. a **canonical operational contract** in `Model-Zoo`
2. a **short research-facing summary** in `Model-Research`

The canonical contract should live in:

- `Model-Zoo/docs/operations/analysis-run-contract.md`

The short consumer summary should live in:

- `docs/research/scaleup/analysis-run-contract-summary.md`

The `Model-Zoo` document is the source of truth.
The `Model-Research` document is only a concise handoff note for phase-3 work.

## Why This Contract Exists

We now have a cleaner three-phase split:

1. phase 1: curation and gathering
2. phase 2: ESD analysis
3. phase 3: retrieval, mixed-zoo analysis, prediction, and paper-facing work

That split is only useful if the phase-2 handoff is boring and reliable.

So the contract should answer only these questions:

- what is the default phase-2 input?
- where does a run write?
- what files mean success?
- what files mean failure?
- what should phase 3 read, and what should it ignore?

Anything beyond that is out of scope for this step.

## Canonical Input Contract

The default phase-2 input is:

- `Model-Zoo/data/curated/model_zoo_phase2.csv`

Optional narrower inputs may exist:

- `Model-Zoo/data/curated/model_zoo_phase2_analysis_ready.csv`
- `Model-Zoo/data/curated/model_zoo_phase2_prediction_ready.csv`

But those are subset views, not the default contract.

The contract should state clearly:

- normal operational runs start from `model_zoo_phase2.csv`
- the narrower subset files are opt-in
- manual audit files may support curation, but they are not phase-2 run outputs

## Canonical Output Contract

Each analysis run writes to:

- `Model-Zoo/analysis_runs/phase2/<run_name>/`

`run_name` should be treated as one short path component.
No nested paths should be part of the contract.

Required layout under the run root:

- `stats/`
- `metrics/`
- `logs/`

Expected files:

- `stats/*.csv`
- `metrics/*.h5`
- `logs/failed_models.txt`
- `logs/failure_records.jsonl`

Expected convenience file:

- `summary.csv`

`summary.csv` is useful and should be documented, but it should not be treated as the canonical proof that a model completed successfully.

## Completion Rule

A model counts as complete only when both of these exist:

- `stats/<model_key>.csv`
- `metrics/<model_key>.h5`

If one exists without the other, the model is incomplete.
That is true even if the run produced a summary row.

This is the main rule phase 3 should rely on.

No empty placeholder file should count as success.

## Failure Rule

Failures must be visible in structured form.

The contract should treat these as the canonical failure artifacts:

- `logs/failed_models.txt`
- `logs/failure_records.jsonl`

Their role is:

- `failed_models.txt` for quick operator scanning
- `failure_records.jsonl` for structured downstream diagnosis

Phase 3 should use these for triage and filtering, not as substitutes for missing success artifacts.

## Consumer Rule For Model-Research

`Model-Research` should consume phase-2 runs with the following assumptions:

1. the curated join key comes from the canonical table in `Model-Zoo/data/curated/`
2. per-model success is determined by paired `stats/` and `metrics/` outputs
3. `summary.csv` is a convenience artifact, not the primary truth source
4. `logs/failure_records.jsonl` is diagnostic, not a scientific label source

In plain English:

- phase 3 should read the run directory directly
- phase 3 should not depend on local copies of the curated table
- phase 3 should not infer success from summaries alone

## Phase Boundary

The contract should say explicitly that phase 2 stops at:

- ESD outputs
- run summary
- failure logs

It does **not** include:

- lineage retrieval outputs
- mixed-zoo benchmark outputs
- prediction outputs
- paper-facing reports

Those belong to phase 3 in `Model-Research`.

## Non-Goals

This contract should not introduce:

- a scheduler redesign
- a loader redesign
- a new orchestration layer
- a giant manifest system
- a run kickoff workflow
- new scientific methods

If a small provenance file is added later, it should be a narrow extension, not part of the minimum contract for this step.

## Minimal Deliverables

The implementation that follows this spec should stay small:

1. add `Model-Zoo/docs/operations/analysis-run-contract.md`
2. add `docs/research/scaleup/analysis-run-contract-summary.md`
3. link both from the existing reader-facing docs where useful
4. do not change the runner behavior unless a wording or path cleanup is absolutely required

## Success Criteria

This step is successful if, after implementation:

- a human can open one `Model-Zoo` doc and understand the phase-2 run contract in one pass
- a phase-3 reader can open one short `Model-Research` summary and know what to consume
- there is no ambiguity about:
  - default input table
  - canonical run root
  - required success artifacts
  - required failure artifacts
  - where phase 2 ends and phase 3 begins

## Keep / Kill Rules

Keep:

- `Model-Zoo` as the operational source of truth
- `Model-Research` as the experimental consumer
- `stats + metrics` pairing as the completion rule
- `failed_models.txt + failure_records.jsonl` as the failure surface

Kill:

- duplicate contract docs with equal authority
- ad hoc phase-2 output locations
- treating `summary.csv` as the only source of truth
- blending phase-2 operational outputs with phase-3 research outputs
