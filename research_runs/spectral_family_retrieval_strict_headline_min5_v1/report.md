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
- gallery_supported: 153
- queries_supported: 72
- queries_unsupported: 1861
- queries_heterogeneous: 97

## Retrieval metrics
| method | slice | num_queries | top1_accuracy | mrr | recall@1 | recall@3 | recall@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| metadata_only | all | 49 | 0.184 | 0.323 | 0.184 | 0.224 | 0.469 |
| metadata_only | architecture_matched | 17 | 0.647 | 0.814 | 0.647 | 1.000 | 1.000 |
| metadata_only | size_matched | 17 | 0.529 | 0.644 | 0.529 | 0.647 | 0.765 |
| pl_alpha_scalar | all | 49 | 0.857 | 0.903 | 0.857 | 0.939 | 0.939 |
| pl_alpha_scalar | architecture_matched | 17 | 0.824 | 0.902 | 0.824 | 1.000 | 1.000 |
| pl_alpha_scalar | size_matched | 17 | 0.824 | 0.902 | 0.824 | 1.000 | 1.000 |
| attention_trace | all | 49 | 0.918 | 0.946 | 0.918 | 0.959 | 0.980 |
| attention_trace | architecture_matched | 17 | 0.941 | 0.971 | 0.941 | 1.000 | 1.000 |
| attention_trace | size_matched | 17 | 0.941 | 0.971 | 0.941 | 1.000 | 1.000 |
| full_module_trace | all | 49 | 0.918 | 0.950 | 0.918 | 0.980 | 1.000 |
| full_module_trace | architecture_matched | 17 | 0.941 | 0.971 | 0.941 | 1.000 | 1.000 |
| full_module_trace | size_matched | 17 | 0.941 | 0.971 | 0.941 | 1.000 | 1.000 |
| summary_module | all | 49 | 0.796 | 0.873 | 0.796 | 0.939 | 0.959 |
| summary_module | architecture_matched | 17 | 0.706 | 0.843 | 0.706 | 1.000 | 1.000 |
| summary_module | size_matched | 17 | 0.706 | 0.843 | 0.706 | 1.000 | 1.000 |

## Rejection metrics
| method | slice | num_examples | threshold | accepted_precision | accepted_coverage | auroc | auprc | fpr95 | aurc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metadata_only | all | 1910 | 0.000 | 0.004 | 0.831 | 0.475 | 0.024 | 0.911 | 0.995 |
| metadata_only | architecture_matched | 1823 | 0.000 | 0.004 | 0.828 | 0.300 | 0.006 | 0.979 | 0.995 |
| metadata_only | size_matched | 1829 | 0.000 | 0.004 | 0.825 | 0.301 | 0.006 | 0.976 | 0.995 |
| pl_alpha_scalar | all | 1910 | n/a | n/a | 0.000 | 0.549 | 0.029 | 0.914 | 0.974 |
| pl_alpha_scalar | architecture_matched | 1823 | n/a | n/a | 0.000 | 0.772 | 0.030 | 0.970 | 0.983 |
| pl_alpha_scalar | size_matched | 1829 | n/a | n/a | 0.000 | 0.772 | 0.030 | 0.965 | 0.983 |
| attention_trace | all | 1905 | n/a | n/a | 0.000 | 0.538 | 0.031 | 0.956 | 0.971 |
| attention_trace | architecture_matched | 1819 | n/a | n/a | 0.000 | 0.751 | 0.031 | 0.931 | 0.982 |
| attention_trace | size_matched | 1825 | n/a | n/a | 0.000 | 0.754 | 0.031 | 0.930 | 0.982 |
| full_module_trace | all | 1910 | -2.082 | 0.026 | 0.654 | 0.527 | 0.029 | 0.933 | 0.973 |
| full_module_trace | architecture_matched | 1823 | -2.082 | 0.032 | 0.172 | 0.755 | 0.027 | 0.888 | 0.982 |
| full_module_trace | size_matched | 1829 | -2.082 | 0.032 | 0.172 | 0.758 | 0.027 | 0.880 | 0.982 |
| summary_module | all | 1910 | n/a | n/a | 0.000 | 0.516 | 0.026 | 0.959 | 0.977 |
| summary_module | architecture_matched | 1823 | n/a | n/a | 0.000 | 0.750 | 0.026 | 0.996 | 0.986 |
| summary_module | size_matched | 1829 | n/a | n/a | 0.000 | 0.750 | 0.026 | 0.994 | 0.986 |

## Family neighborhood top-k
| method | family | neighbor | rank | score |
| --- | --- | --- | --- | --- |
| metadata_only | Qwen/Qwen2-0.5B | Qwen/Qwen2.5-0.5B | 1 | -0.015 |
| metadata_only | Qwen/Qwen2-0.5B | mistralai/Mistral-7B-v0.1 | 2 | -0.081 |
| metadata_only | Qwen/Qwen2-0.5B | meta-llama/Meta-Llama-3-8B | 3 | -0.092 |
| metadata_only | Qwen/Qwen2-0.5B | meta-llama/Llama-3.1-8B | 4 | -0.096 |
| metadata_only | Qwen/Qwen2-0.5B | meta-llama/Llama-2-7b-hf | 5 | -0.134 |
| metadata_only | Qwen/Qwen2-7B | mistralai/Mistral-Nemo-Base-2407 | 1 | -0.140 |
| metadata_only | Qwen/Qwen2-7B | Qwen/Qwen2.5-7B | 2 | -0.544 |
| metadata_only | Qwen/Qwen2-7B | meta-llama/Llama-3.1-8B | 3 | -0.895 |
| metadata_only | Qwen/Qwen2-7B | meta-llama/Meta-Llama-3-8B | 4 | -0.897 |
| metadata_only | Qwen/Qwen2-7B | Qwen/Qwen2-0.5B | 5 | -1.174 |
| metadata_only | Qwen/Qwen2.5-0.5B | Qwen/Qwen2-0.5B | 1 | -0.015 |
| metadata_only | Qwen/Qwen2.5-0.5B | mistralai/Mistral-7B-v0.1 | 2 | -0.036 |
| metadata_only | Qwen/Qwen2.5-0.5B | meta-llama/Llama-2-7b-hf | 3 | -0.063 |
| metadata_only | Qwen/Qwen2.5-0.5B | meta-llama/Llama-3.2-1B | 4 | -0.063 |
| metadata_only | Qwen/Qwen2.5-0.5B | meta-llama/Llama-3.2-3B | 5 | -0.063 |

## Heterogeneous stress bucket
- rows: 97
