# Preliminary Research Draft Design

## Goal

Write a markdown-first research draft that captures the current preliminary study honestly and usefully, without pretending it is a submission-ready paper.

The draft should do three things:

1. state the real research goal clearly
2. summarize what the preliminary work has actually established
3. make the remaining work before scale-up explicit

## Why This Format

A standard paper skeleton is premature for the current stage. It would push the project toward polished-paper language before the evidence and scaled-up study are settled.

A purely chronological lab note is also not ideal. It preserves the history of the work, but it does not present the current research argument cleanly.

So the right format is a `claims and evidence draft`:

- centered on the main research aim
- organized by what survived and what did not
- explicit about method hierarchy
- explicit about limits
- easy to revise after the scaled-up experiments

## Draft Shape

The markdown draft should be one document with the following sections.

### 1. North Star

State the real project aim in plain language:

- managing a large zoo of models
- requiring support for both homogeneous and heterogeneous mixtures
- treating performance prediction as a secondary but important direction

This section should make clear that retrieval and organization come first, and that prediction matters only if it strengthens the zoo-management story.

### 2. Why We Started Narrow

Explain why the preliminary phase began with a strict homogeneous benchmark:

- to test whether the spectral signal was real
- to avoid mixing clean family recovery with noisy lineage, derived-model effects, and schema mismatch
- to create a trustworthy first result before broadening the benchmark

This section should explicitly say that the narrow benchmark was a control, not the final destination.

### 3. What We Built

Summarize the actual preliminary system:

- strict homogeneous control benchmark
- lean mixed-zoo benchmark
- retrieval methods
- heterogeneity-aware methods
- neighborhood-based prediction probes
- descriptive trend diagnostics

This should read like a compact system overview, not an experiment log.

### 4. What Holds Up

State only the strongest supported findings:

- `full_module_trace` is the strongest general retrieval method
- `role_aggregated_trace` is the strongest current heterogeneity-aware method
- mixed-zoo retrieval is now real, not hypothetical
- spectral neighborhoods carry real but uneven prediction signal

This section should contain the clearest representative numbers, but not become a metric dump.

### 5. What Stayed Secondary Or Weak

Make the weaker parts explicit:

- rejection is still weak
- metadata-heavy controls are not the real zoo method
- trend-only signals are not the main prediction story
- heterogeneous prediction remains difficult

This section should help prevent overclaiming later.

### 6. Current Method Hierarchy

Make the current ranking of methods explicit.

Primary:
- `full_module_trace`
- `role_aggregated_trace`

Secondary:
- `attention_trace`
- `pl_alpha_scalar`
- spectral-neighborhood prediction
- trend-style descriptive features

Side-result only:
- rejection
- metadata-heavy controls

This section should make later scale-up decisions easier and reduce method sprawl.

### 7. Canonical Preliminary Artifact Set

List the small set of notes and experiment outputs that should be treated as the canonical preliminary record:

- consolidated findings note
- strict benchmark report
- lean mixed-zoo report
- prediction cycle report
- trend diagnostics report

This is meant to keep the branch reproducible but less conceptually messy.

### 8. What Needs To Happen Before Scale-Up

State the remaining work in disciplined form:

- fix the preliminary narrative in writing
- preserve the current experiment family
- do not add side quests
- rerun the same family of experiments on a larger model pool

This section should close the draft with direction, not speculation.

## Style Constraints

The draft should be:

- in markdown
- plain English first
- concise and argument-driven
- not written in fake submission tone
- not overloaded with metrics
- easy to revise after larger-scale experiments

The draft should not:

- claim open-world identification is solved
- claim rejection is solved
- claim general performance prediction is solved
- revert to the broader older-paper framing

## Output Location

Write the draft to:

`docs/superpowers/specs/2026-04-19-preliminary-research-draft.md`

## Success Condition

The draft is successful if:

- it reflects the current preliminary findings accurately
- it is organized around the real research goal
- it keeps the method hierarchy disciplined
- it is reusable as the base text for the scaled-up phase
