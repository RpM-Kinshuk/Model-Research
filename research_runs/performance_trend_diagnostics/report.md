# Performance Trend Diagnostics

## Top Absolute-Trend Rows By Target
| bucket | target | feature | control_mode | n | spearman |
| --- | --- | --- | --- | --- | --- |
| heterogeneous | Average ⬆️ | trend_global_start | raw | 21 | 0.357 |
| heterogeneous | Average ⬆️ | trend_global_end | structural_adjusted | 21 | -0.338 |
| all_scored | Average ⬆️ | trend_global_start | structural_adjusted | 374 | 0.256 |
| all_scored | MATH Lvl 5 | trend_global_start | structural_adjusted | 374 | 0.468 |
| heterogeneous | MATH Lvl 5 | trend_global_start | raw | 21 | 0.461 |
| all_scored | MATH Lvl 5 | trend_global_start | raw | 374 | 0.453 |
| heterogeneous | MMLU-PRO | trend_global_end_minus_start | raw | 21 | 0.503 |
| heterogeneous | MMLU-PRO | trend_global_start | structural_adjusted | 21 | -0.475 |
| heterogeneous | MMLU-PRO | trend_global_end_minus_start | structural_adjusted | 21 | 0.453 |

## Top Exact-Parent Rows By Target
| bucket | target | feature | control_mode | n | spearman |
| --- | --- | --- | --- | --- | --- |
| heterogeneous | Average ⬆️_delta | trend_global_mean_delta | structural_adjusted | 12 | -0.871 |
| heterogeneous | Average ⬆️_delta | trend_global_mean_delta | raw | 12 | -0.748 |
| heterogeneous | Average ⬆️_delta | trend_global_end_delta | structural_adjusted | 12 | -0.660 |
| heterogeneous | MATH Lvl 5_delta | trend_global_mean_delta | structural_adjusted | 12 | -0.876 |
| heterogeneous | MATH Lvl 5_delta | trend_global_mean_delta | raw | 12 | -0.790 |
| heterogeneous | MATH Lvl 5_delta | trend_global_end_delta | structural_adjusted | 12 | -0.741 |
| heterogeneous | MMLU-PRO_delta | trend_global_end_delta | structural_adjusted | 12 | 0.561 |
| heterogeneous | MMLU-PRO_delta | trend_global_end_minus_start_delta | raw | 12 | -0.357 |
| heterogeneous | MMLU-PRO_delta | trend_global_start_delta | raw | 12 | 0.280 |

## Top Family-Relative Rows By Target
| bucket | target | feature | control_mode | n | spearman |
| --- | --- | --- | --- | --- | --- |
| heterogeneous | Average ⬆️_delta | trend_global_end_minus_start_delta | structural_adjusted | 18 | -0.728 |
| heterogeneous | Average ⬆️_delta | trend_global_end_minus_start_delta | raw | 18 | -0.725 |
| heterogeneous | Average ⬆️_delta | trend_global_start_delta | structural_adjusted | 18 | 0.690 |
| heterogeneous | MATH Lvl 5_delta | trend_global_end_minus_start_delta | raw | 18 | -0.507 |
| heterogeneous | MATH Lvl 5_delta | trend_global_end_minus_start_delta | structural_adjusted | 18 | -0.506 |
| heterogeneous | MATH Lvl 5_delta | trend_global_start_delta | structural_adjusted | 18 | 0.454 |
| heterogeneous | MMLU-PRO_delta | trend_global_mean_delta | raw | 18 | -0.467 |
| heterogeneous | MMLU-PRO_delta | trend_global_mean_delta | structural_adjusted | 18 | -0.447 |
| heterogeneous | MMLU-PRO_delta | trend_global_start_delta | raw | 18 | -0.170 |

## Notes
- Exact-parent and family-relative evidence are reported separately on purpose.
- Raw and lightweight structural-adjusted variants are both included when enough rows are available.
