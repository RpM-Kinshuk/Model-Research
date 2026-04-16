# CSV Alpha Correlation Summary

## Dataset

- Leaderboard rows loaded: 936
- Unique leaderboard models after deduplication: 903
- Direct `model_id` overlap with `svd_results/*.csv`: 686
- Usable matched models after skipping empty / malformed SVD CSVs: 374
- Clean 7-module common-schema matched models: 344

## Main results

- Canonical view, strongest size-adjusted per-module signal:
  - `attn_k__start` vs `MATH Lvl 5`
  - partial Spearman = `0.645`
  - `n = 353`
- Canonical view, strongest aggregate alpha summary signal:
  - `alpha_start_across_modules` vs `MATH Lvl 5`
  - raw Spearman = `0.456`
  - partial Spearman = `0.546`
- Common-schema alpha-only best signals by target:
  - `Average ⬆️`: `alpha_median_attn_minus_mlp` (`partial = 0.442`)
  - `IFEval`: `alpha_global_fraction_gt_2` (`partial = 0.265`)
  - `BBH`: `alpha_mean_attn_minus_mlp` (`partial = 0.438`)
  - `MATH Lvl 5`: `alpha_median_module_spread` (`partial = 0.578`)
  - `GPQA`: `alpha_delta_attn_mean` (`partial = -0.261`)
  - `MUSR`: `alpha_end_module_spread` (`partial = -0.371`)
  - `MMLU-PRO`: `alpha_median_module_spread` (`partial = 0.448`)

## Interpretation notes

- In this matched subset, the alpha summaries are overwhelmingly above `2`.
- `alpha_global_mean > 2` for `100%` of matched models.
- `alpha_global_median > 2` for `100%` of matched models.
- Because of that, the signed and absolute distance-to-2 features are effectively the same ranking in this dataset.
- Practically, the data looks much more like an `alpha > 2` regime than a mixed overtrained/undertrained regime.
- The strongest alpha-performance relationship in this sample is with `MATH Lvl 5`, especially through early-layer / attention-key behavior and cross-module alpha spread.

## Predictive family takeaways

- For `Average ⬆️`, `BBH`, `MATH Lvl 5`, `MUSR`, and `MMLU-PRO`, alpha features improved out-of-fold prediction over non-alpha metadata alone.
- For `GPQA`, non-alpha metrics were still the best predictive family.
- For `IFEval`, alpha modules alone were the best family, but the effect size was smaller than for `MATH Lvl 5` and `MMLU-PRO`.

## Output files

- Canonical report: `correlation_runs/csv_alpha_leaderboard/report.md`
- Common-schema report: `correlation_runs/csv_common_schema_research/report.md`
- Visuals:
  - `correlation_runs/csv_alpha_visuals/common_alpha_median_module_spread_vs_math_lvl5.png`
  - `correlation_runs/csv_alpha_visuals/partial_attn_k_start_vs_math_lvl5.png`
  - `correlation_runs/csv_alpha_visuals/selected_alpha_signals_heatmap.png`
