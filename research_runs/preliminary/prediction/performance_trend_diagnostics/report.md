# Performance Trend Diagnostics

## Largest-Magnitude Absolute-Trend Rows By Target And Bucket
| bucket | target | feature | control_mode | n | spearman |
| --- | --- | --- | --- | --- | --- |
| supported_candidate | Average ⬆️ | trend_global_mean | structural_adjusted | 41 | 0.604 |
| supported_candidate | MATH Lvl 5 | trend_global_start | structural_adjusted | 41 | 0.649 |
| supported_candidate | MMLU-PRO | trend_global_start | structural_adjusted | 41 | 0.646 |
| heterogeneous | Average ⬆️ | trend_global_start | raw | 21 | 0.357 |
| heterogeneous | MATH Lvl 5 | trend_global_start | raw | 21 | 0.461 |
| heterogeneous | MMLU-PRO | trend_global_end_minus_start | raw | 21 | 0.503 |
| all_scored | Average ⬆️ | trend_global_start | structural_adjusted | 374 | 0.256 |
| all_scored | MATH Lvl 5 | trend_global_start | structural_adjusted | 374 | 0.468 |
| all_scored | MMLU-PRO | trend_global_start | structural_adjusted | 374 | 0.199 |

## Largest-Magnitude Exact-Parent Rows By Target And Bucket
| bucket | target | feature | control_mode | n | spearman |
| --- | --- | --- | --- | --- | --- |
| supported_candidate | Average ⬆️_delta | trend_global_end_minus_start_delta | structural_adjusted | 13 | 0.278 |
| supported_candidate | MATH Lvl 5_delta | trend_global_start_delta | structural_adjusted | 13 | 0.575 |
| supported_candidate | MMLU-PRO_delta | trend_global_end_delta | structural_adjusted | 13 | 0.263 |
| heterogeneous | Average ⬆️_delta | trend_global_mean_delta | structural_adjusted | 12 | -0.871 |
| heterogeneous | MATH Lvl 5_delta | trend_global_mean_delta | structural_adjusted | 12 | -0.876 |
| heterogeneous | MMLU-PRO_delta | trend_global_end_delta | structural_adjusted | 12 | 0.561 |
| all_scored | Average ⬆️_delta | trend_global_end_minus_start_delta | structural_adjusted | 71 | 0.127 |
| all_scored | MATH Lvl 5_delta | trend_global_start_delta | structural_adjusted | 71 | 0.157 |
| all_scored | MMLU-PRO_delta | trend_global_start_delta | raw | 71 | 0.266 |

## Largest-Magnitude Family-Relative Rows By Target And Bucket
| bucket | target | feature | control_mode | n | spearman |
| --- | --- | --- | --- | --- | --- |
| supported_candidate | Average ⬆️_delta | trend_global_end_minus_start_delta | structural_adjusted | 41 | 0.179 |
| supported_candidate | MATH Lvl 5_delta | trend_global_end_minus_start_delta | structural_adjusted | 41 | 0.099 |
| supported_candidate | MMLU-PRO_delta | trend_global_mean_delta | raw | 41 | -0.163 |
| heterogeneous | Average ⬆️_delta | trend_global_end_minus_start_delta | structural_adjusted | 18 | -0.728 |
| heterogeneous | MATH Lvl 5_delta | trend_global_end_minus_start_delta | raw | 18 | -0.507 |
| heterogeneous | MMLU-PRO_delta | trend_global_mean_delta | raw | 18 | -0.467 |
| all_scored | Average ⬆️_delta | trend_global_start_delta | structural_adjusted | 155 | 0.152 |
| all_scored | MATH Lvl 5_delta | trend_global_end_delta | structural_adjusted | 155 | 0.095 |
| all_scored | MMLU-PRO_delta | trend_global_start_delta | structural_adjusted | 155 | 0.157 |

## Notes
- Rows are selected by absolute Spearman magnitude within each bucket/target pair, so negative associations can appear when their magnitude is larger.
- `family_relative` anchors are family medians over all scored rows for that family, not bucket-local medians.
- `structural_adjusted` means partial Spearman adjusted for `num_layers`, `raw_module_count`, `common_schema`, and schema dummies.
- This branch is descriptive-only, not predictive.
- `supported_candidate` is the primary managed slice and should be read first.
- `heterogeneous` is the broader cross-zoo context slice.
- `all_scored` is a mixed-zoo aggregate across every scored row, so it should be read as context rather than the primary managed slice (current composition: `supported_candidate`=41, `heterogeneous`=21, `unsupported_special`=97, `unsupported_unknown`=215).
- Exact-parent and family-relative evidence are reported separately on purpose.
- Raw and lightweight structural-adjusted variants are both included when enough rows are available.
