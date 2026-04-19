# Rejection Confidence Mini-Study

This study keeps the strict benchmark and the fingerprint methods fixed.
It only compares simple confidence rules: `top_score`, `margin`, and `score_plus_margin`.

## Best all-slice AUROC by method
- full_module_trace: score_plus_margin (AUROC 0.598, AUPRC 0.033, P@10 0.028, P@25 0.040, P@50 0.030)
- attention_trace: score_plus_margin (AUROC 0.583, AUPRC 0.035, P@10 0.033, P@25 0.032, P@50 0.029)
- pl_alpha_scalar: top_score (AUROC 0.655, AUPRC 0.035, P@10 0.023, P@25 0.037, P@50 0.033)
- metadata_only: score_plus_margin (AUROC 0.573, AUPRC 0.028, P@10 0.014, P@25 0.017, P@50 0.008)

## Bucket breakdown
| method | rule | bucket | num_examples | auroc | auprc | coverage_10_precision | coverage_25_precision | coverage_50_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_module_trace | score_plus_margin | unsupported_unknown | 1638 | 0.585 | 0.043 | 0.037 | 0.046 | 0.037 |
| full_module_trace | score_plus_margin | unsupported_special | 406 | 0.574 | 0.150 | 0.049 | 0.157 | 0.143 |
| full_module_trace | score_plus_margin | heterogeneous | 178 | 0.831 | 0.568 | 0.500 | 0.711 | 0.483 |
| attention_trace | score_plus_margin | unsupported_unknown | 1638 | 0.576 | 0.058 | 0.049 | 0.037 | 0.035 |
| attention_trace | score_plus_margin | unsupported_special | 400 | 0.547 | 0.157 | 0.050 | 0.150 | 0.140 |
| attention_trace | score_plus_margin | heterogeneous | 172 | 0.782 | 0.554 | 0.556 | 0.721 | 0.465 |
