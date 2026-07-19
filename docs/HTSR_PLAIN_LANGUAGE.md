# Spectral Model Lineage And Performance Prediction

## What This Study Does

A language model contains many large tables of numbers called weight matrices.
Training and fine-tuning change the internal structure of these matrices. We study
that structure to produce a spectral fingerprint for each model.

The same measurements are used for two separate tasks:

1. **Lineage discovery:** find models that probably belong to the same broad base-model family.
2. **Performance prediction:** estimate benchmark scores without running the benchmarks.

These are statistical predictions. They do not prove that one model was directly
created from another model.

## What A Spectral Fingerprint Is

Every weight matrix has a spectrum that describes how its information is spread
across different directions. Heavy-Tailed Self-Regularization, or HTSR, summarizes
parts of this spectrum using measurements such as alpha.

We measure seven common matrix roles in every transformer layer:

- attention query, key, value, and output
- MLP gate, up, and down projections

The location of a measurement matters. Two models can have the same average alpha
but very different patterns from their first layer to their last layer.

## The Large-Scale Data

The Phase 2 collection contains:

- **15,000** model records
- **8,892** models with spectral artifacts
- **7,534** models passing strict lineage quality checks
- **8,284** models in the broader exploratory lineage population
- **650** modern benchmark-labelled models suitable for ancestry-grouped alpha prediction
- **348** legacy benchmark-labelled models with sufficiently reliable ancestry information

Not every model can be used for every question. A model may have spectral data but
no benchmark score, or a benchmark score but no reliable ancestry metadata.

## Lineage Discovery

### Method

Each model is represented by its matrix sizes and spectral behavior across its
layers. Models with similar fingerprints are grouped without giving the clustering
algorithm their known family names. We then compare the anonymous clusters with
the family labels available in the metadata.

The strict analysis excludes incomplete or incompatible layer layouts. The broad
analysis accepts more models, but its additional assignments are used only for
coverage checks and manual investigation.

### Results

The strict analysis clustered **7,534 models**.

| Measure | Result | Plain meaning |
| --- | ---: | --- |
| ARI | 0.850 | Models from the same known families were usually grouped together |
| NMI | 0.905 | The anonymous clusters contained strong information about family identity |
| Matched-family recall | 0.847 | About 85% of each known family was recovered on average |

Known-family retrieval reached approximately **96.9% top-1 accuracy** on 174
labelled queries from 11 families. This is promising, but the labelled population
is much smaller than the complete model collection.

Exact-parent prediction reached only **6.5% top-1 accuracy**. The fingerprints are
therefore useful for broad family tracing, not for proving the exact parent
checkpoint or reconstructing a complete derivation chain.

The broad clustering covers **8,284 models**. Its family suggestions are
distance-based manual-triage candidates, not probabilities or ancestry ground truth.

## Performance Prediction

### Main Method

The leading predictor uses each model's depth-wise alpha profile:

1. Scale model depth from zero at the first layer to one at the last layer.
2. Sample alpha at 32 positions for each of the seven matrix roles.
3. Replace the raw values with their within-profile ranks. This preserves where
   alpha rises and falls while reducing sensitivity to model scale.
4. Find the five labelled models with the most similar profiles.
5. Predict the score using a similarity-weighted average of their scores.

Clustering is **not** an input to this predictor. Known ancestry is used only to
make evaluation harder and more honest. Entire related groups are held out
together, and the five neighbours must come from distinct ancestry groups.

### Comparison Methods

The alpha-profile predictor was compared with ridge regression using progressively
larger feature blocks:

- **M:** layer count, matrix dimensions, coverage, and parameter scale
- **S:** weight scale, spectral norm, entropy, and stable rank
- **H:** alpha, alpha-weighted norm, and related HTSR summaries

### Modern Benchmark Results

The paired comparison uses the same 649 models and 139 ancestry components for
every method.

| Method | Component-weighted MAE | Spearman rank correlation |
| --- | ---: | ---: |
| Model structure only | 3.711 | 0.666 |
| Structure and non-HTSR spectra | 3.562 | 0.726 |
| Structure, spectra, and HTSR summaries | 3.455 | 0.734 |
| **Direct alpha profile** | **3.082** | **0.767** |

MAE is measured in benchmark percentage points. An MAE of 3.082 means the
predicted six-task average was about 3.1 points away from the real score on
average after giving each ancestry component equal weight.

The direct alpha profile improved MAE over the strongest ridge baseline by
**0.373 points**. Its estimated 95% improvement interval was **0.037 to 0.727
points**, with `p=0.037`.

### Legacy Benchmark Results

| Method | Component-weighted MAE |
| --- | ---: |
| Structure, spectra, and HTSR summaries | 5.463 |
| **Direct alpha profile** | **4.135** |

The direct alpha profile improved the six-task average by **1.328 points**. Its
estimated 95% improvement interval was **0.554 to 2.104 points**, with `p=0.0013`.
Five of the six individual legacy tasks showed a reliable improvement after
multiple-comparison correction; MMLU was the exception.

## Why We Do Not Lead With The Lowest Error

A paper-like random split produced a lower modern MAE of 2.805 across 942 models.
However, nearly half of its test models had a known relative in the training data.
That measures interpolation among related models more than generalization to new
families. The ancestry-grouped result is harder and is therefore the main result.

## What Did Not Help

Using alpha only as a residual correction on top of ridge regression did not
reliably improve the modern result. A small fixed-fold follow-up also found no
useful gain from raw alpha magnitude or larger depth profiles containing every
available spectral metric. The rank-normalized alpha shape remains the simplest
and strongest current choice.

## Conclusion And Limits

- Spectral fingerprints work well for **broad model-family discovery**.
- They do not reliably identify the **exact parent checkpoint**.
- The best current performance method is the direct, five-neighbour,
  rank-normalized alpha-profile predictor.
- The large-scale performance gains are ancestry-grouped internal evidence, not a
  genuinely prospective external validation.
- The stored alpha values use a DKS power-law estimator. They are valid HTSR
  features, but they are not an exact reproduction of the paper's fixed-tail Hill
  estimator.

The practical decision is to use strict spectral clustering for lineage views and
direct alpha-profile nearest neighbours for performance prediction. Broad lineage
routes should remain clearly marked suggestions for human review.
