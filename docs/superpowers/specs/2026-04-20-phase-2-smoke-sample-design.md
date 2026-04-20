# Phase 2 Smoke Sample Design

## Goal

Create one very small, reproducible workflow to prepare a `100`-model mixed-zoo smoke sample for phase 2.

This step should:

- use the already-curated phase-2 table
- preserve the broad type mix of the larger zoo
- avoid clearly unsupported rows
- produce one sampled CSV and one small summary artifact

It should **not**:

- start the analysis run
- redesign the runner
- introduce a new framework
- replace the full curated table

## Recommendation

Use a tiny manual sandbox that does not pollute committed repo state, for example:

- `/tmp/model_zoo_phase2_smoke/`
- or a small gitignored local sandbox if that is more convenient

Put only two things there:

- `build_smoke_sample.py`
- `README.md`

That keeps the workflow:

- close to the operational repo
- reproducible
- easy to delete or replace later

without turning it into a new system.

## Input

Read from the full curated table:

- `Model-Zoo/data/curated/model_zoo_phase2.csv`

Do **not** sample from `model_zoo_phase2_analysis_ready.csv` by default, because that would bias the smoke pass toward models that are already easier.

## Filtering Rule

Before sampling, drop only rows that are already clearly unsupported for this pass.

Examples:

- `loader_scenario == quantized_alt_format`
- rows already marked by strong probe-failure style signals that make this pass structurally pointless
- rows that are obviously missing essential identity fields

The filter should stay conservative.
The point is not to pre-clean away all risk.
The point is just to avoid knowingly feeding the runner rows we already expect to fail for structural reasons.

## Sampling Rule

Target size:

- `100` rows

Primary stratification:

- `primary_type_bucket`

Main objective:

- preserve rough dataset proportions across the major type buckets

Secondary objective:

- within larger buckets, keep some diversity across:
  - `lineage_status`
  - `eval_status`

This should be a soft preference, not a hard balancing problem.

If a type bucket is small, just take what is available up to its proportional share.

## Output

Write the sample to a disposable sandbox location, for example:

- `/tmp/model_zoo_phase2_smoke/smoke_100_mixed_v1.csv`

Write one small summary artifact next to it, for example:

- `/tmp/model_zoo_phase2_smoke/smoke_100_mixed_v1_summary.md`

The summary should include:

- total sampled rows
- counts by `primary_type_bucket`
- counts by `lineage_status`
- counts by `eval_status`
- short note on excluded clearly unsupported rows

## Human Workflow

The sandbox README should make the next step obvious and keep the workflow easy to inspect by hand.

It should contain:

1. how to rebuild the sample
2. where the sample CSV is written
3. the later planned run target:
   - `Model-Zoo/analysis_runs/phase2/smoke_100_mixed_v1/`
4. the success checks we care about after the run:
   - overall success rate
   - success/failure by `primary_type_bucket`
   - failure reasons by `loader_scenario`

But the README should stop short of actually starting the run.

## Tracking Rule

This smoke-sample workflow is intentionally **not** part of the committed `Model-Zoo` repo surface.

That means:

- do not commit the sandbox files into `Model-Zoo`
- using `.gitignore` is acceptable if it helps keep the sandbox local and disposable
- keep the sandbox easy to inspect and remove manually

## Keep / Kill Rules

Keep:

- one small script
- one small README
- one sampled CSV
- one small summary file

Kill:

- a new pipeline layer
- a generalized sampler framework
- too many output artifacts
- any coupling to phase-3 research code

## Success Criteria

This step is successful if:

- a human can regenerate the `100`-model sample in one command
- the sample roughly reflects the larger type mix
- clearly unsupported rows are not dominating the sample
- the output is small and easy to inspect
