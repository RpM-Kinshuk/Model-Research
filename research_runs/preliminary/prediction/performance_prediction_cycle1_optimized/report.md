# Performance Prediction Cycle 1

## Best rows by target
| target | method | mode | bucket | k | weighted | n | r2 | mae | spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Average ⬆️ | full_module_trace | raw_spectral | unsupported_special | 5 | true | 97 | 0.703 | 2.933 | 0.820 |
| Average ⬆️ | full_module_trace | raw_spectral | unsupported_special | 3 | true | 97 | 0.660 | 3.039 | 0.795 |
| Average ⬆️ | full_module_trace | spectral_plus_structural | unsupported_special | 5 | false | 97 | 0.637 | 3.445 | 0.792 |
| MATH Lvl 5 | full_module_trace | raw_spectral | unsupported_special | 5 | true | 97 | 0.779 | 4.544 | 0.848 |
| MATH Lvl 5 | full_module_trace | raw_spectral | unsupported_special | 3 | true | 97 | 0.767 | 4.653 | 0.841 |
| MATH Lvl 5 | role_aggregated_trace | raw_spectral | unsupported_special | 5 | true | 97 | 0.761 | 4.888 | 0.820 |
| MMLU-PRO | full_module_trace | raw_spectral | supported_candidate | 3 | true | 41 | 0.831 | 3.209 | 0.894 |
| MMLU-PRO | full_module_trace | raw_spectral | supported_candidate | 5 | true | 41 | 0.809 | 3.469 | 0.889 |
| MMLU-PRO | full_module_trace | raw_spectral | unsupported_special | 5 | true | 97 | 0.806 | 2.590 | 0.872 |

## Notes
- Raw spectral rows measure direct neighborhood signal.
- Controlled rows add the structural baseline back after residual neighborhood prediction.
