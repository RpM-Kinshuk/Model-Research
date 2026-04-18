# Rejection Confidence Mini-Study

## Goal

Run a small follow-up study on the current strict family-retrieval benchmark to see whether a better **confidence rule** can improve rejection behavior without changing the underlying fingerprint methods.

This is not a new main contribution. It is a short support study.

## Scope

Keep fixed:

- benchmark slice: `spectral_family_retrieval_strict_headline_v2`
- family labels and splits
- fingerprint methods

Do not add:

- new spectral features
- new benchmark slices
- heterogeneous-first analysis
- broad method search

## Methods Included

Only evaluate:

- `full_module_trace`
- `attention_trace`
- `pl_alpha_scalar`
- `metadata_only`

`summary_module` is not needed for this mini-study.

## Confidence Rules

For each method, compare only these confidence rules:

1. `top_score`
- confidence = best family score

2. `margin`
- confidence = best family score minus second-best family score

3. `score_plus_margin`
- confidence = small normalized combination of best score and margin

No larger ensembles or learned calibrators.

## Main Question

If a method makes a family guess, which simple confidence rule is best at telling whether that guess should be trusted?

## Evaluation

Use the current strict benchmark only.

Main reporting:

- `AUROC`
- `AUPRC`
- accepted precision at fixed coverage levels:
  - `10%`
  - `25%`
  - `50%`

Secondary reporting for the best rule only:

- supported vs unsupported pooled
- unsupported bucket breakdown:
  - `unsupported_unknown`
  - `unsupported_special`
  - `heterogeneous`

## Success Criteria

The study is useful only if at least one simple confidence rule clearly improves rejection over `top_score` for one of the two main methods:

- `full_module_trace`
- `attention_trace`

Useful means:

- higher `AUROC` or `AUPRC`
- and/or better accepted precision at the same coverage

If improvements are tiny or unstable, the conclusion is simply:

`rejection remains weak under simple confidence rules`

## Expected Outcome

Most likely outcomes:

1. `margin` helps somewhat because ambiguity between the top two families matters.
2. `score_plus_margin` helps a little more than either alone.
3. Even with that improvement, rejection will still remain secondary to the main retrieval result.

## Deliverables

- one small extension to the current pipeline or one short helper script
- one output folder with rejection-comparison tables
- one short written summary comparing the rules

## Keep / Kill Rules

Keep:

- the best confidence rule if it gives a clean, interpretable gain

Kill:

- any rule that does not beat `top_score`
- any design that starts adding learned calibration or many extra heuristics
