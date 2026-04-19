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
- min_family_size: 8
- gallery_supported: 159
- queries_supported: 76
- queries_unsupported: 1851
- queries_heterogeneous: 97

## Retrieval metrics
| method | slice | num_queries | top1_accuracy | mrr | recall@1 | recall@3 | recall@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| metadata_only | all | 52 | 0.192 | 0.336 | 0.192 | 0.231 | 0.462 |
| metadata_only | architecture_matched | 18 | 0.611 | 0.796 | 0.611 | 1.000 | 1.000 |
| metadata_only | size_matched | 18 | 0.556 | 0.661 | 0.556 | 0.667 | 0.778 |
| pl_alpha_scalar | all | 52 | 0.827 | 0.885 | 0.827 | 0.923 | 0.942 |
| pl_alpha_scalar | architecture_matched | 18 | 0.833 | 0.907 | 0.833 | 1.000 | 1.000 |
| pl_alpha_scalar | size_matched | 18 | 0.833 | 0.907 | 0.833 | 1.000 | 1.000 |
| attention_trace | all | 52 | 0.885 | 0.923 | 0.885 | 0.942 | 0.962 |
| attention_trace | architecture_matched | 18 | 0.889 | 0.944 | 0.889 | 1.000 | 1.000 |
| attention_trace | size_matched | 18 | 0.889 | 0.944 | 0.889 | 1.000 | 1.000 |
| full_module_trace | all | 52 | 0.885 | 0.922 | 0.885 | 0.962 | 0.962 |
| full_module_trace | architecture_matched | 18 | 0.889 | 0.944 | 0.889 | 1.000 | 1.000 |
| full_module_trace | size_matched | 18 | 0.889 | 0.944 | 0.889 | 1.000 | 1.000 |
| summary_module | all | 52 | 0.788 | 0.867 | 0.788 | 0.942 | 0.962 |
| summary_module | architecture_matched | 18 | 0.722 | 0.852 | 0.722 | 1.000 | 1.000 |
| summary_module | size_matched | 18 | 0.722 | 0.852 | 0.722 | 1.000 | 1.000 |

## Rejection metrics
| method | slice | num_examples | threshold | accepted_precision | accepted_coverage | auroc | auprc | fpr95 | aurc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metadata_only | all | 1903 | 0.000 | 0.004 | 0.831 | 0.482 | 0.026 | 0.910 | 0.997 |
| metadata_only | architecture_matched | 1820 | 0.000 | 0.005 | 0.828 | 0.316 | 0.007 | 0.979 | 0.994 |
| metadata_only | size_matched | 1826 | 0.000 | 0.005 | 0.825 | 0.317 | 0.007 | 0.976 | 0.997 |
| pl_alpha_scalar | all | 1903 | n/a | n/a | 0.000 | 0.542 | 0.030 | 0.914 | 0.973 |
| pl_alpha_scalar | architecture_matched | 1820 | n/a | n/a | 0.000 | 0.771 | 0.029 | 0.970 | 0.983 |
| pl_alpha_scalar | size_matched | 1826 | n/a | n/a | 0.000 | 0.772 | 0.029 | 0.965 | 0.983 |
| attention_trace | all | 1898 | n/a | n/a | 0.000 | 0.538 | 0.033 | 0.965 | 0.969 |
| attention_trace | architecture_matched | 1816 | n/a | n/a | 0.000 | 0.755 | 0.029 | 0.928 | 0.981 |
| attention_trace | size_matched | 1822 | n/a | n/a | 0.000 | 0.758 | 0.029 | 0.927 | 0.981 |
| full_module_trace | all | 1903 | -2.462 | 0.029 | 0.668 | 0.529 | 0.032 | 0.936 | 0.970 |
| full_module_trace | architecture_matched | 1820 | -2.462 | 0.034 | 0.178 | 0.761 | 0.027 | 0.884 | 0.982 |
| full_module_trace | size_matched | 1826 | -2.462 | 0.034 | 0.177 | 0.764 | 0.027 | 0.877 | 0.982 |
| summary_module | all | 1903 | n/a | n/a | 0.000 | 0.509 | 0.028 | 0.954 | 0.976 |
| summary_module | architecture_matched | 1820 | n/a | n/a | 0.000 | 0.740 | 0.025 | 0.996 | 0.986 |
| summary_module | size_matched | 1826 | n/a | n/a | 0.000 | 0.740 | 0.025 | 0.994 | 0.986 |

## Family neighborhood top-k
| method | family | neighbor | rank | score |
| --- | --- | --- | --- | --- |
| metadata_only | Qwen/Qwen2-0.5B | Qwen/Qwen2.5-0.5B | 1 | -0.021 |
| metadata_only | Qwen/Qwen2-0.5B | mistralai/Mistral-7B-v0.1 | 2 | -0.081 |
| metadata_only | Qwen/Qwen2-0.5B | meta-llama/Llama-3.1-8B | 3 | -0.093 |
| metadata_only | Qwen/Qwen2-0.5B | meta-llama/Meta-Llama-3-8B | 4 | -0.108 |
| metadata_only | Qwen/Qwen2-0.5B | Qwen/Qwen2.5-7B | 5 | -0.131 |
| metadata_only | Qwen/Qwen2-7B | mistralai/Mistral-Nemo-Base-2407 | 1 | -0.140 |
| metadata_only | Qwen/Qwen2-7B | Qwen/Qwen2.5-7B | 2 | -0.651 |
| metadata_only | Qwen/Qwen2-7B | meta-llama/Meta-Llama-3-8B | 3 | -0.797 |
| metadata_only | Qwen/Qwen2-7B | meta-llama/Llama-3.1-8B | 4 | -0.923 |
| metadata_only | Qwen/Qwen2-7B | Qwen/Qwen2-0.5B | 5 | -1.174 |
| metadata_only | Qwen/Qwen2.5-0.5B | Qwen/Qwen2-0.5B | 1 | -0.021 |
| metadata_only | Qwen/Qwen2.5-0.5B | mistralai/Mistral-7B-v0.1 | 2 | -0.030 |
| metadata_only | Qwen/Qwen2.5-0.5B | meta-llama/Llama-2-7b-hf | 3 | -0.051 |
| metadata_only | Qwen/Qwen2.5-0.5B | meta-llama/Llama-3.2-1B | 4 | -0.051 |
| metadata_only | Qwen/Qwen2.5-0.5B | meta-llama/Llama-3.2-3B | 5 | -0.051 |

## Heterogeneous stress bucket
- rows: 97
