# Spectral Family Retrieval Preliminary Report

Headline claim tested: module-aware spectral candidate generation for base-family retrieval with abstention.

## Included methods
- metadata_only
- pl_alpha_scalar
- attention_trace
- full_module_trace
- summary_module

## Split summary
- seed: 0
- min_family_size: 5
- gallery_supported: 165
- queries_supported: 76
- queries_unsupported: 1845
- queries_heterogeneous: 97

## Retrieval metrics
| method | slice | num_queries | top1_accuracy | mrr | recall@1 | recall@3 | recall@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| metadata_only | all | 42 | 0.214 | 0.353 | 0.214 | 0.357 | 0.381 |
| metadata_only | architecture_matched | 18 | 0.556 | 0.769 | 0.556 | 1.000 | 1.000 |
| metadata_only | size_matched | 18 | 0.500 | 0.706 | 0.500 | 0.889 | 0.944 |
| pl_alpha_scalar | all | 42 | 0.786 | 0.852 | 0.786 | 0.881 | 0.929 |
| pl_alpha_scalar | architecture_matched | 18 | 0.722 | 0.852 | 0.722 | 1.000 | 1.000 |
| pl_alpha_scalar | size_matched | 18 | 0.722 | 0.852 | 0.722 | 1.000 | 1.000 |
| attention_trace | all | 42 | 0.881 | 0.914 | 0.881 | 0.929 | 0.952 |
| attention_trace | architecture_matched | 18 | 0.833 | 0.917 | 0.833 | 1.000 | 1.000 |
| attention_trace | size_matched | 18 | 0.833 | 0.917 | 0.833 | 1.000 | 1.000 |
| full_module_trace | all | 42 | 0.881 | 0.914 | 0.881 | 0.929 | 0.929 |
| full_module_trace | architecture_matched | 18 | 0.833 | 0.917 | 0.833 | 1.000 | 1.000 |
| full_module_trace | size_matched | 18 | 0.833 | 0.917 | 0.833 | 1.000 | 1.000 |
| summary_module | all | 42 | 0.738 | 0.820 | 0.738 | 0.881 | 0.952 |
| summary_module | architecture_matched | 18 | 0.667 | 0.824 | 0.667 | 1.000 | 1.000 |
| summary_module | size_matched | 18 | 0.667 | 0.824 | 0.667 | 1.000 | 1.000 |

## Rejection metrics
| method | slice | num_examples | threshold | accepted_precision | accepted_coverage | auroc | auprc | fpr95 | aurc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metadata_only | all | 1887 | -0.408 | 0.004 | 0.891 | 0.501 | 0.022 | 0.912 | 0.995 |
| metadata_only | architecture_matched | 1805 | -0.408 | 0.004 | 0.845 | 0.421 | 0.009 | 0.978 | 0.995 |
| metadata_only | size_matched | 1811 | -0.408 | 0.004 | 0.865 | 0.417 | 0.008 | 0.983 | 0.995 |
| pl_alpha_scalar | all | 1887 | n/a | n/a | 0.000 | 0.480 | 0.023 | 0.993 | 0.980 |
| pl_alpha_scalar | architecture_matched | 1805 | n/a | n/a | 0.000 | 0.674 | 0.024 | 0.997 | 0.985 |
| pl_alpha_scalar | size_matched | 1811 | n/a | n/a | 0.000 | 0.673 | 0.024 | 0.996 | 0.985 |
| attention_trace | all | 1882 | n/a | n/a | 0.000 | 0.532 | 0.029 | 0.979 | 0.974 |
| attention_trace | architecture_matched | 1801 | n/a | n/a | 0.000 | 0.721 | 0.034 | 0.997 | 0.981 |
| attention_trace | size_matched | 1807 | n/a | n/a | 0.000 | 0.721 | 0.034 | 0.996 | 0.981 |
| full_module_trace | all | 1887 | -1.974 | 0.021 | 0.686 | 0.514 | 0.027 | 0.971 | 0.975 |
| full_module_trace | architecture_matched | 1805 | -1.974 | 0.031 | 0.177 | 0.720 | 0.030 | 0.997 | 0.982 |
| full_module_trace | size_matched | 1811 | -1.974 | 0.031 | 0.177 | 0.721 | 0.030 | 0.996 | 0.982 |
| summary_module | all | 1887 | n/a | n/a | 0.000 | 0.510 | 0.024 | 0.995 | 0.978 |
| summary_module | architecture_matched | 1805 | n/a | n/a | 0.000 | 0.661 | 0.023 | 0.997 | 0.986 |
| summary_module | size_matched | 1811 | n/a | n/a | 0.000 | 0.661 | 0.023 | 0.996 | 0.986 |

## Family neighborhood top-k
| method | family | neighbor | rank | score |
| --- | --- | --- | --- | --- |
| metadata_only | Qwen/Qwen2-0.5B | Qwen/Qwen2.5-0.5B | 1 | -0.015 |
| metadata_only | Qwen/Qwen2-0.5B | Qwen/Qwen2.5-1.5B | 2 | -0.032 |
| metadata_only | Qwen/Qwen2-0.5B | mistralai/Mistral-7B-v0.1 | 3 | -0.081 |
| metadata_only | Qwen/Qwen2-0.5B | meta-llama/Meta-Llama-3-8B | 4 | -0.092 |
| metadata_only | Qwen/Qwen2-0.5B | meta-llama/Llama-3.1-8B | 5 | -0.096 |
| metadata_only | Qwen/Qwen2-7B | mistralai/Mistral-Nemo-Base-2407 | 1 | -0.140 |
| metadata_only | Qwen/Qwen2-7B | Qwen/Qwen2.5-7B | 2 | -0.544 |
| metadata_only | Qwen/Qwen2-7B | mistralai/Mistral-7B-v0.3 | 3 | -0.698 |
| metadata_only | Qwen/Qwen2-7B | meta-llama/Llama-3.1-8B | 4 | -0.895 |
| metadata_only | Qwen/Qwen2-7B | meta-llama/Meta-Llama-3-8B | 5 | -0.897 |
| metadata_only | Qwen/Qwen2.5-0.5B | Qwen/Qwen2.5-1.5B | 1 | -0.011 |
| metadata_only | Qwen/Qwen2.5-0.5B | Qwen/Qwen2-0.5B | 2 | -0.015 |
| metadata_only | Qwen/Qwen2.5-0.5B | mistralai/Mistral-7B-v0.1 | 3 | -0.036 |
| metadata_only | Qwen/Qwen2.5-0.5B | meta-llama/Llama-2-7b-hf | 4 | -0.063 |
| metadata_only | Qwen/Qwen2.5-0.5B | meta-llama/Llama-3.2-1B | 5 | -0.063 |

## Heterogeneous stress bucket
- rows: 97
