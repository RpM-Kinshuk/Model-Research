# Current conclusion in plain language

These are the completed September 2026 findings. The
[October 23 plan](research/scaleup/model_zoo_frontier_research_plan_2026-10-23.md)
proposes the next model-selection experiments; it does not add prediction results.

We rechecked the reference paper, collection code, saved matrix measurements,
data joins, predictions, and reporting. We also tested alternatives on the same
models. The source data was not changed.

## What holds up

**Broad family grouping works.** Comparing how alpha changes through layers
gives 17 repeatable groups across 10,946 models. Agreement with 2,981 broad-family
labels is 0.924 by adjusted Rand index, where 1 means identical grouping. The
174 human-reviewed models are one validation subset, not the whole analysis.
Grouping covers 87.2% of the 12,553 models with saved analyses, or 55.5% of the
19,713 selected records. Missing analyses account for much of that difference.
The method handles different layer counts and tested missing-role patterns;
accuracy across unfamiliar architectures remains untested.

**Known-base matching depends on the comparison.** Comparing layer-wise
matrix norms raises first-choice accuracy from 82.30% to 85.19% on the same 3,802
queries and 29 candidate bases. The correct base is in the first three for
99.29%. However, excluding queries tied between candidates reverses the overall
first-choice ordering: 85.88% for alpha ranks, 84.73% for norms. This changes the
models and roots tested; it is a sensitivity check, not proof that one method is
better. Neither identifies historical parents or reliably detects missing bases.

**Performance prediction has more signal than the old method showed.** Keeping
alpha ranks alone discards useful measurement levels. When near-identical model
groups are kept out of training together, the old neighbors' average-score
errors are 3.726 modern and 5.227 legacy. Small trees using spectral measurements
without alpha reduce these to 3.211 and 3.068.

## What changed

- **Alpha is not the demonstrated source of that gain.** Adding alpha back to
  the same tree model slightly worsens those errors, to 3.266 and 3.109. Keep it
  as a comparison, not an assumed quality indicator.
- **Broader is not always better.** Summarizing all saved matrices covers 12,365
  models without role requirements, but broad-family grouping agreement falls
  to 0.278. Preserve layer-wise grouping for now.
- **Generalization is limited.** Removing an entire weight-derived group makes
  score prediction harder. Linear models retain some signal but can produce
  impossible scores in collection transfer. No method is deployment-validated.
- **Checkpoint identity is unresolved.** Scores and weights are joined by
  repository, but saved analyses lack the resolved commit. Some source commits
  conflict. We cannot verify that every score describes the exact weights used.

```text
Weights -> group or compare models -> save answers -> check lineage labels

Weights + other models' training scores -> predict score
                                      -> save answer -> check held-out score
```

Family and architecture annotations are not prediction inputs. Metadata still
helps construct collections and evaluate results. Score prediction needs other
models' training scores; it does not train on the held-out model's score.

## Decision

Keep the existing family groups, add the norm-based base comparison, and use
non-alpha spectral measurements as the stronger simple performance baseline.
Do not claim we beat the paper: the populations and evaluations differ.

The newer collector now records pinned revisions, matrix coverage, and full
computed spectra. Its new records have not yet been used in the prediction
results above. The next experiment should join scores to those exact checkpoints
and test whether the measurements improve model choices for particular tasks.
The plan retains the weight-only comparison and proposes a hybrid using
documentation and historical evaluations. The missing candidate bases and a
matched test of the paper's alpha estimator remain useful follow-ups.

See [the study](HTSR_STUDY.md) for the process, comparisons, and limitations.
