# Method and claim guide

This note defines the claims used by the deck. Numbers come from the newest
complete run under `research_runs/htsr_weight_only/runs/`; the current completed
run is `weight_only_2a5abfeb98dc33d9662c`.

## Experimental boundary

Descriptive metadata is excluded from prediction:

- lineage and family labels;
- architecture and model type;
- parameter count;
- source collection;
- ancestry graph.

Lineage discovery fits its groups from complete seven-role alpha summaries, then
adds partial summaries only when a label-free masking test passes. Supplied-
gallery routing also consumes the operator-provided candidate IDs. Performance
prediction consumes observed-role summaries and benchmark scores from training
models. Held-out scores and all descriptive metadata are joined only after
predictions are frozen.

## Collection and extraction

The collector checks model loading before dispatch, then gives each worker a
configurable set of GPUs. It waits for allowed GPUs to remain below a memory
threshold and also reserves assigned GPUs inside the runner. This reduces overlap
risk; it is not a cluster-wide reservation against unrelated schedulers.

Each worker has its own log, heartbeat, terminal status, and temporary Hugging Face
cache. Retryable load, analysis, and save failures can be retried; access and known
unsupported-model failures stop without repeated downloads. Runtime signals can
reload GPU settings, drain pending work, or stop workers. The cache is removed when
the worker ends, which bounds local storage but may repeat network transfers later.

For one model, eligible layer matrices are ordered by estimated cost and assigned
across the visible GPUs. This can reduce wall-clock time even for a model that fits
on one GPU, because independent matrix decompositions run concurrently. It does not
guarantee fewer total GPU-seconds. The collector supports DKS, a fixed half-tail
cutoff, or a histogram-peak cutoff; direct singular-value decomposition or a Gram
matrix; and optional eigenvalue storage. Per-model CSV and HDF5 files are finalized
atomically for downstream analysis.

## Spectral representation

The stored scalar alpha values use WeightWatcher's DKS estimator. For each of
seven common matrix roles, model depth is normalized to `[0, 1]`, alpha is
interpolated at 32 positions, converted to within-role ranks, centered, and
normalized. The strict paper comparison joins all seven vectors. The general
method compares observed roles and never fills missing roles.

The reference paper instead uses fixed-tail Hill alpha with `k = n / 2`. It uses
role-wise Spearman similarity for equal-depth models and, in Section 5.4,
z-scored injective order-preserving dynamic programming for unequal depths.

## Experiment 1: unlabeled discovery

MiniBatch K-means is run on 10,431 complete profiles. Fourteen candidate cluster counts
from 2 to 103 are compared using cosine silhouette in the original profile
space. Seventeen clusters maximize silhouette. No labels select the count or fit
the clusters.

Each cluster's representative is the actual core member with the highest cosine
similarity to the cluster mean. It is not claimed to be a historical root.

Each observed missing-role pattern is also applied to 1,200 complete profiles.
The pattern is admitted only when the complete-profile nearest model remains in
the masked top ten at least 90% of the time. This adds 515 models without
changing the core groups, for 10,946 lineage assignments.

Internal results:

| Measure | Value |
| --- | ---: |
| Cosine silhouette | 0.6068 |
| Davies-Bouldin index | 1.5409 |
| Mean seed stability, adjusted Rand index | 0.9441 |
| Minimum seed stability, adjusted Rand index | 0.8835 |

Post-freeze label agreement:

| Labels | Models | Adjusted Rand | Normalized mutual information | Purity |
| --- | ---: | ---: | ---: | ---: |
| Manual, 11 families | 174 | 0.9127 | 0.9313 | 0.9310 |
| Metadata consensus, 11 families | 2,981 | 0.9238 | 0.9184 | 0.9389 |
| Exact graph root, 529 roots | 5,013 | 0.6000 | 0.7538 | 0.6599 |

These are validation results, not training results. They support broad grouping,
not exact parent recovery or edge direction.

## Experiment 2: supplied-gallery routing

The candidate gallery is externally supplied, as in the paper's closed-set
classification design. Public provenance is prior experimental knowledge, but it
is not a numeric feature. Of 38 supplied bases, 29 have complete seven-role
summaries.

Predictions are frozen for 10,402 non-gallery models. Post-freeze graph labels
allow 3,836 queries to be scored because their labelled root is one of the 29
profiled bases.

| Measure | Value |
| --- | ---: |
| Overall top-1 | 0.8227 |
| Root-balanced top-1 | 0.8407 |
| Overall top-3 | 0.9841 |
| Root-balanced top-3 | 0.9863 |
| Root-balanced mean reciprocal rank | 0.9083 |

The paper reports 98.44% overall top-1 for 128 derivatives and eight supplied
bases. Our result is lower and not directly comparable because the corpus,
candidate set, alpha estimator, and unequal-depth method differ.

## Experiment 3: performance prediction

Masking checks select a minimum of four observed roles without using scores.
Five test parts are assigned from exact observed-role-summary hashes, so
identical summaries cannot cross parts. A held-out model is predicted from a
distance-weighted average of its three nearest training models, each sharing at
least four roles.

| Scores | Models / profile groups | MAE | Spearman | Training-mean MAE |
| --- | ---: | ---: | ---: | ---: |
| Modern | 2,537 / 2,410 | 3.043 | 0.875 | 8.458 |
| Legacy | 3,515 / 3,363 | 3.188 | 0.897 | 10.530 |

This is an interpolation test. Very similar, non-identical models may still be
in different folds.

The harder test assigns complete weight-derived clusters to folds:

| Scores | kNN MAE | Spearman | Training-mean MAE |
| --- | ---: | ---: | ---: |
| Modern | 7.721 | 0.479 | 9.102 |
| Legacy | 14.169 | 0.032 | 13.182 |

Modern prediction retains limited signal. Legacy prediction fails to beat the
training mean. Production use therefore needs a distance-based coverage check
and abstention outside represented spectral regions.

## Recommended two-stage lineage system

Use HTSR as the scalable first stage for broad clustering and retrieval. For
exact within-family verification, collect a second weight-geometry signal. Three
current candidates are:

- arXiv:2608.07786: spectral energy plus singular-subspace alignment;
- arXiv:2608.14929: centered residual signatures;
- arXiv:2510.06738 (AWM): assignment plus unbiased centered kernel alignment.

The present cache stores scalar eigenvalue summaries, not singular vectors or
centered weight matrices. These methods are future work, not tested results.
