# Phase 1 Wrap-Up

## What Is Closed

Phase 1 is now closed as a preliminary research phase.

The repository has a stable active surface, a canonical preliminary narrative, a grouped experiment-artifact layout, and a preserved archive/legacy area. The goal of phase 1 was not to finish the whole research program. It was to answer whether the current spectral approach has a real and useful signal for organizing models.

That question has a clear answer:

- yes for retrieval and zoo organization
- yes in a meaningful first way for heterogeneous handling
- partly for performance prediction
- not yet for strong rejection or general prediction claims

## Frozen Canonical Preliminary Artifact Set

The preliminary phase should now be read through this canonical set:

1. strict homogeneous retrieval control
2. lean mixed-zoo retrieval
3. rejection mini-study as side evidence
4. performance prediction cycles
5. descriptive trend diagnostics

Canonical reader-facing docs:

- [README](./README.md)
- [Findings](./preliminary-findings.md)
- [Research Draft](./preliminary-research-draft.md)
- [Paper Outline](./preliminary-paper-outline.md)
- [Artifact Index](./artifact-index.md)
- [Experiment Family](./experiment-family.md)

Model-Research preliminary research artifact roots:

- `research_runs/preliminary/retrieval/`
- `research_runs/preliminary/rejection/`
- `research_runs/preliminary/prediction/`

Support-only preliminary research artifact roots:

- `research_runs/preliminary/optimization/`
- `research_runs/preliminary/exploratory/`

## What Holds Up

These are the parts we should carry forward confidently.

- `full_module_trace` is the strongest general retrieval method on the clean control benchmark.
- `role_aggregated_trace` is the strongest current heterogeneity-aware retrieval direction.
- mixed-zoo retrieval is real, not hypothetical.
- spectral neighborhoods carry real performance signal on some slices.
- trend-style summaries are useful only as secondary features.

## What Stays Secondary Or Unresolved

These points should stay explicit and should not be upgraded during scale-up unless the evidence changes.

- rejection is still a side result
- heterogeneous prediction is still weak/uneven
- trend diagnostics are descriptive support, not the main method
- metadata-heavy controls are useful confound checks, not the main zoo method

## What Phase 2 Starts With

Phase 2 should begin with data curation for the larger model pool in `Model-Zoo`.

The point is not to invent a new experiment family. The point is to scale the same family of experiments onto a larger and better-curated set of models while keeping the current method hierarchy stable.

Operational source of truth:

- curated tables: `Model-Zoo/data/curated/`
- analysis outputs: `Model-Zoo/analysis_runs/phase2/`

Ownership split:

- `Model-Research` keeps the phase-1 findings, narrative, and reader-facing research artifacts:
  - `docs/research/preliminary/`
  - `research_runs/preliminary/`
- `Model-Zoo` owns the canonical operational scale-up artifacts:
  - curated tables in `Model-Zoo/data/curated/`
  - analysis outputs in `Model-Zoo/analysis_runs/phase2/`

`Model-Research` remains the phase-3 experimental consumer and report layer, but it is no longer the canonical home for the scale-up operational artifacts themselves.

So phase 2 should start with:

1. larger-pool data curation
2. the same grouped experiment family
3. the same distinction between canonical artifacts and support artifacts
4. the same high-level claim structure

## Immediate Next Step

The next work item is the scale-up handoff:

`prepare the scale-up/data-curation phase without reopening the phase-1 method story`

That means:

- keep the phase-1 claims fixed
- keep the phase-1 artifact set fixed
- define what new data must be curated for scale-up
- rerun the same experiment family on the larger pool
- read curated inputs from `Model-Zoo/data/curated/`
- write analysis outputs under `Model-Zoo/analysis_runs/phase2/`
- keep `Model-Research` focused on phase-3 experiments and reports

This is the clean handoff point between the preliminary phase and the scale-up phase.
