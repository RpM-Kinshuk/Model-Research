# Phase Contract And Repo Cleanup Design

Date: 2026-04-20

## Goal

Make the overall system clean, stable, and easy to pick up by explicitly separating it into three phases:

1. data curation and gathering
2. ESD analysis
3. experimental analysis

The main operational source of truth should become `Model-Zoo`, while `Model-Research` should remain the research and paper-facing consumer.

This phase is about:

- freezing the cross-phase contract
- making the filesystem layout reflect that contract
- reducing ambiguity for a human operator
- preserving reproducibility without turning the system into one huge pipeline

This phase is not about:

- redesigning the hand-built Model-Zoo execution engine
- rewriting the experimental methods
- merging both repos into one

## Why This Change Is Needed

Right now the research direction is coherent, but the operational layout is still easy to misread:

- some canonical artifacts live in `Model-Research`
- the ESD runner lives in `Model-Zoo`
- some outputs and contracts are implied rather than explicit
- a new human reader still has to infer which repo owns what

That is workable during exploration, but it will become a maintenance problem as soon as the larger-scale run begins.

The system needs one simple mental model:

- `Model-Zoo` runs the operational pipeline
- `Model-Research` reads stable outputs and does the science

## Design Principles

1. Keep the three phases explicit.
   - Do not collapse them into one giant pipeline.

2. Keep `Model-Zoo` as the operational source of truth.
   - It should own canonical curated data and canonical analysis outputs.

3. Keep `Model-Research` as the experimental consumer.
   - It should own methods, evaluation, research runs, and paper-facing outputs.

4. Prefer stable handoff artifacts over internal coupling.
   - Each phase should produce a clear output that the next phase reads.

5. Keep it readable for a human.
   - The right file should be easy to find without reading lots of docs.

6. Do not over-touch `Model-Zoo`.
   - Only small changes are justified there unless a direct contract or cleanliness issue requires more.

## Repo Roles

### `Model-Zoo`

`Model-Zoo` should be the operational home for:

- phase-1 curated data
- phase-2 ESD analysis
- the canonical contract between input tables and analysis outputs
- the canonical operational documentation for those phases

In plain English:

- if a human wants to know what data is being analyzed and where analysis outputs go, `Model-Zoo` should be the first place to look

### `Model-Research`

`Model-Research` should be the home for:

- phase-3 experimental methods
- retrieval, mixed-zoo, and prediction pipelines
- research runs and reports
- paper and narrative docs

In plain English:

- if a human wants to know how we test lineage or prediction ideas, `Model-Research` should be the first place to look

## Canonical Three-Phase Contract

### Phase 1: Data Curation And Gathering

Owner: `Model-Zoo`

Input sources may include:

- atlas-derived metadata
- leaderboard enrichment
- Hugging Face supplements
- manual audit files

Canonical output:

- one canonical curated model table
- a few derived views from that same table
- lightweight availability / probe outputs

This phase should stop at:

- curated data shape
- enrichment
- probe/fetchability status

It should not include:

- persistent model downloads
- ESD analysis
- retrieval or prediction experiments

### Phase 2: ESD Analysis

Owner: `Model-Zoo`

Input:

- canonical curated table from phase 1

Canonical output:

- `stats/*.csv`
- `metrics/*.h5`
- `logs/failure_records.jsonl`
- `logs/failed_models.txt`

This phase should remain the existing Model-Zoo runner flow:

- curated table in
- structured analysis outputs out

It should not include:

- lineage retrieval logic
- prediction logic
- paper-facing analysis

### Phase 3: Experimental Analysis

Owner: `Model-Research`

Input:

- curated tables and derived metadata views from `Model-Zoo`
- analysis outputs from `Model-Zoo`

Canonical output:

- research runs
- benchmark reports
- comparison tables
- paper-facing documents

This phase should remain extensible:

- new methods should plug into stable loaders and evaluation flows
- adding a new method should not require changing phases 1 or 2

## Canonical Directory Layout

### In `Model-Zoo`

The following structure should be treated as canonical:

```text
Model-Zoo/
├── data/
│   ├── curated/
│   │   ├── model_zoo_phase2.csv
│   │   ├── model_zoo_phase2_analysis_ready.csv
│   │   ├── model_zoo_phase2_prediction_ready.csv
│   │   ├── model_zoo_phase2_probe_failures.csv
│   │   └── README.md
│   └── manual_audits/
│
├── analysis_runs/
│   └── phase2/
│       └── <run_name>/
│           ├── stats/
│           ├── metrics/
│           └── logs/
│
├── esd_experiment/
│   └── ...
│
└── docs/
    └── operations/
        ├── README.md
        ├── phase-contract.md
        ├── curation.md
        └── analysis.md
```

Notes:

- `data/curated/` should be the canonical phase-1 output location
- `analysis_runs/phase2/` should be the canonical phase-2 output root
- `docs/operations/` should become the reader-facing entrypoint for operational use

### In `Model-Research`

The following structure should be treated as canonical:

```text
Model-Research/
├── spectral_research/
│   └── ...
│
├── research_runs/
│   ├── preliminary/
│   └── scaleup/
│
└── docs/
    └── research/
        ├── preliminary/
        └── scaleup/
```

Notes:

- `Model-Research` should stop being the canonical home of curated operational data
- it can keep local convenience copies only if clearly marked as derived or synced, not canonical

## Canonical File Contracts

### Canonical Curated Table

Canonical location:

- `Model-Zoo/data/curated/model_zoo_phase2.csv`

This should be the default source of truth for phase 1.

Minimum important fields:

- `model_id`
- `revision_norm`
- `source_model`
- `base_model_relation`
- `primary_type_bucket`
- `loader_scenario`
- `lineage_status`
- `eval_status`

Important supporting fields may also include:

- `candidate_source`
- availability / probe columns
- analysis status columns
- leaderboard enrichment columns

### Canonical Analysis Output

Canonical root:

- `Model-Zoo/analysis_runs/phase2/<run_name>/`

Expected structure:

- `stats/`
- `metrics/`
- `logs/`

Expected files:

- `stats/<safe_model_id>.csv`
- `metrics/<safe_model_id>.h5`
- `logs/failure_records.jsonl`
- `logs/failed_models.txt`

Resume semantics:

- a model is considered complete only if both the final stats and metrics artifacts exist

### Canonical Research Input Contract

`Model-Research` should consume:

- the curated table from `Model-Zoo`
- the analysis output root from `Model-Zoo`

The experimental code should not depend on:

- ad hoc intermediate exports
- manually assembled joins scattered across the repo
- ambiguous run directories

## Human-Facing Entry Points

The system should have one obvious starting point per repo.

### `Model-Zoo`

Primary entrypoint:

- `docs/operations/README.md`

This should answer:

- where the canonical curated table lives
- how to build or refresh it
- where analysis outputs go
- how to run ESD analysis
- where failures are recorded

### `Model-Research`

Primary entrypoint:

- `docs/research/preliminary/README.md`
- later `docs/research/scaleup/README.md`

This should answer:

- what experiments exist
- what stable inputs they expect from `Model-Zoo`
- where research reports and outputs go

## Minimal Cleanup Plan

### In `Model-Zoo`

Do:

- add `docs/operations/`
- make `data/curated/` clearly canonical
- introduce `analysis_runs/phase2/` as the canonical phase-2 output root
- keep `esd_experiment/` mostly intact
- make small path/config/doc changes only where needed

Do not:

- refactor the GPU dispatcher architecture
- redesign the threading/worker system
- build a new orchestration framework

### In `Model-Research`

Do:

- remove operational ambiguity in docs
- point clearly to `Model-Zoo` for curated data and analysis artifacts
- keep research outputs grouped under `research_runs/`
- keep method modules and evaluation flows easy to extend

Do not:

- keep growing a second operational truth for curated data
- duplicate canonical phase-1/phase-2 outputs unless clearly marked as temporary or synced

## Extensibility Rule For Experimental Methods

Phase 3 should stay easy to extend.

That means:

- stable input readers
- stable benchmark/slice definitions
- method modules with a small registration surface
- stable evaluation/report code

In practice:

- adding a new retrieval or prediction method should mostly mean adding one method module and plugging it into an existing evaluation flow

It should not mean:

- changing the curated data shape
- changing the Model-Zoo runner
- inventing a new output format

## Success Criteria

This cleanup is successful if:

1. a human can open `Model-Zoo/docs/operations/README.md` and understand phases 1 and 2 quickly
2. a human can open `Model-Research/docs/research/...` and understand phase 3 quickly
3. the canonical curated table has one obvious home
4. the canonical analysis outputs have one obvious home
5. `Model-Research` reads those canonical outputs without ambiguous joins or ad hoc copies
6. the current experiment family can be rerun at scale without another structural cleanup first

## Recommendation

Adopt this split:

- `Model-Zoo` is the operational source of truth
- `Model-Research` is the experimental and paper-facing consumer

This is the cleanest way to preserve the hand-built value in Model-Zoo, keep the research layer extensible, and reduce overload for a human reader.
