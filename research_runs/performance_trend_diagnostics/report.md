# Performance Trend Diagnostics

## Best Absolute-Trend Rows
| bucket | target | feature | control_mode | n | spearman | r2 | mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| heterogeneous | Average ⬆️ | trend_global_mean | structural_adjusted | 21 | 0.149 | 0.324 | 2.500 |
| heterogeneous | Average ⬆️ | trend_global_end | structural_adjusted | 21 | 0.151 | 0.321 | 2.555 |
| heterogeneous | Average ⬆️ | trend_global_end_minus_start | structural_adjusted | 21 | 0.153 | 0.320 | 2.612 |
| heterogeneous | MATH Lvl 5 | trend_global_mean | structural_adjusted | 21 | 0.177 | 0.401 | 4.252 |
| heterogeneous | MATH Lvl 5 | trend_global_end | structural_adjusted | 21 | 0.274 | 0.398 | 4.338 |
| heterogeneous | MATH Lvl 5 | trend_global_start | structural_adjusted | 21 | 0.461 | 0.394 | 4.429 |
| all_scored | MMLU-PRO | trend_global_mean | structural_adjusted | 374 | 0.038 | 0.284 | 6.553 |
| all_scored | MMLU-PRO | trend_global_start | structural_adjusted | 374 | 0.058 | 0.282 | 6.523 |
| all_scored | MMLU-PRO | trend_global_end | structural_adjusted | 374 | 0.025 | 0.279 | 6.505 |

## Best Exact-Parent Rows
| bucket | target | feature | control_mode | n | spearman | r2 | mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| heterogeneous | Average ⬆️_delta | trend_global_mean_delta | structural_adjusted | 12 | -0.748 | 0.870 | 1.367 |
| heterogeneous | Average ⬆️_delta | trend_global_end_delta | structural_adjusted | 12 | -0.601 | 0.527 | 2.546 |
| heterogeneous | Average ⬆️_delta | trend_global_end_minus_start_delta | structural_adjusted | 12 | -0.154 | 0.270 | 3.382 |
| heterogeneous | MATH Lvl 5_delta | trend_global_mean_delta | structural_adjusted | 12 | -0.790 | 0.761 | 4.971 |
| heterogeneous | MATH Lvl 5_delta | trend_global_end_delta | structural_adjusted | 12 | -0.692 | 0.514 | 6.709 |
| all_scored | MATH Lvl 5_delta | trend_global_start_delta | structural_adjusted | 71 | 0.144 | 0.114 | 5.860 |
| heterogeneous | MMLU-PRO_delta | trend_global_start_delta | structural_adjusted | 12 | 0.280 | 0.646 | 1.136 |
| heterogeneous | MMLU-PRO_delta | trend_global_end_minus_start_delta | structural_adjusted | 12 | -0.357 | 0.628 | 1.205 |
| heterogeneous | MMLU-PRO_delta | trend_global_mean_delta | structural_adjusted | 12 | -0.091 | 0.282 | 1.730 |

## Best Family-Relative Rows
| bucket | target | feature | control_mode | n | spearman | r2 | mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all_scored | Average ⬆️_delta | trend_global_mean_delta | structural_adjusted | 155 | 0.070 | 0.217 | 2.593 |
| all_scored | Average ⬆️_delta | trend_global_mean_delta | raw | 155 | 0.070 | 0.211 | 2.581 |
| heterogeneous | Average ⬆️_delta | trend_global_mean_delta | structural_adjusted | 18 | -0.110 | 0.050 | 2.339 |
| all_scored | MATH Lvl 5_delta | trend_global_mean_delta | structural_adjusted | 155 | -0.008 | 0.167 | 3.510 |
| all_scored | MATH Lvl 5_delta | trend_global_mean_delta | raw | 155 | -0.008 | 0.162 | 3.478 |
| all_scored | MATH Lvl 5_delta | trend_global_end_delta | structural_adjusted | 155 | 0.090 | 0.019 | 3.654 |
| all_scored | MMLU-PRO_delta | trend_global_mean_delta | structural_adjusted | 155 | 0.035 | 0.173 | 2.178 |
| all_scored | MMLU-PRO_delta | trend_global_mean_delta | raw | 155 | 0.035 | 0.169 | 2.174 |
| heterogeneous | MMLU-PRO_delta | trend_global_mean_delta | structural_adjusted | 18 | -0.467 | 0.048 | 1.112 |

## Notes
- Exact-parent and family-relative evidence are reported separately on purpose.
- Raw and lightweight structural-adjusted variants are both included when enough rows are available.
