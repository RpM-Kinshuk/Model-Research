# Performance Prediction Cycle 1

## Best rows by target
| target | method | mode | bucket | k | weighted | n | r2 | mae | spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Average ⬆️ | full_module_trace | raw_spectral | unsupported_special | 5 | true | 97 | 0.703 | 2.933 | 0.820 |
| Average ⬆️ | full_module_trace | raw_spectral | unsupported_special | 3 | true | 97 | 0.660 | 3.039 | 0.795 |
| Average ⬆️ | full_module_trace | spectral_plus_structural | unsupported_special | 5 | false | 97 | 0.637 | 3.445 | 0.792 |

## Notes
- Raw spectral rows measure direct neighborhood signal.
- Controlled rows add the structural baseline back after residual neighborhood prediction.
