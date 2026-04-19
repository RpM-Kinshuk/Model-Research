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
- gallery_supported: 153
- queries_supported: 72
- queries_unsupported: 1861
- queries_heterogeneous: 97

## Retrieval metrics
| method | slice | num_queries | top1_accuracy | mrr | recall@1 | recall@3 | recall@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| metadata_only | all | 49 | 0.918 | 0.959 | 0.918 | 1.000 | 1.000 |
| metadata_only | architecture_matched | 49 | 0.918 | 0.959 | 0.918 | 1.000 | 1.000 |
| metadata_only | size_matched | 49 | 0.918 | 0.959 | 0.918 | 1.000 | 1.000 |
| pl_alpha_scalar | all | 49 | 0.857 | 0.903 | 0.857 | 0.939 | 0.939 |
| pl_alpha_scalar | architecture_matched | 49 | 0.857 | 0.919 | 0.857 | 0.980 | 1.000 |
| pl_alpha_scalar | size_matched | 49 | 0.857 | 0.905 | 0.857 | 0.939 | 0.980 |
| attention_trace | all | 49 | 0.918 | 0.946 | 0.918 | 0.959 | 0.980 |
| attention_trace | architecture_matched | 49 | 0.939 | 0.964 | 0.939 | 0.980 | 1.000 |
| attention_trace | size_matched | 49 | 0.918 | 0.948 | 0.918 | 0.959 | 1.000 |
| full_module_trace | all | 49 | 0.918 | 0.950 | 0.918 | 0.980 | 1.000 |
| full_module_trace | architecture_matched | 49 | 0.939 | 0.964 | 0.939 | 0.980 | 1.000 |
| full_module_trace | size_matched | 49 | 0.918 | 0.951 | 0.918 | 0.980 | 1.000 |
| summary_module | all | 49 | 0.796 | 0.873 | 0.796 | 0.939 | 0.959 |
| summary_module | architecture_matched | 49 | 0.796 | 0.888 | 0.796 | 0.980 | 1.000 |
| summary_module | size_matched | 49 | 0.796 | 0.876 | 0.796 | 0.939 | 0.980 |

## Rejection metrics
| method | slice | num_examples | threshold | accepted_precision | accepted_coverage | auroc | auprc | fpr95 | aurc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metadata_only | all | 1910 | -0.015 | 0.654 | 0.027 | 0.983 | 0.711 | 0.103 | 0.898 |
| metadata_only | architecture_matched | 355 | -0.015 | 0.654 | 0.146 | 0.898 | 0.711 | 0.624 | 0.667 |
| metadata_only | size_matched | 361 | -0.015 | 0.654 | 0.144 | 0.900 | 0.711 | 0.612 | 0.669 |
| pl_alpha_scalar | all | 1910 | n/a | n/a | 0.000 | 0.549 | 0.029 | 0.914 | 0.974 |
| pl_alpha_scalar | architecture_matched | 355 | n/a | n/a | 0.000 | 0.520 | 0.140 | 0.918 | 0.872 |
| pl_alpha_scalar | size_matched | 361 | n/a | n/a | 0.000 | 0.534 | 0.140 | 0.817 | 0.872 |
| attention_trace | all | 1905 | n/a | n/a | 0.000 | 0.538 | 0.031 | 0.956 | 0.971 |
| attention_trace | architecture_matched | 355 | n/a | n/a | 0.000 | 0.524 | 0.157 | 0.948 | 0.853 |
| attention_trace | size_matched | 361 | n/a | n/a | 0.000 | 0.536 | 0.158 | 0.894 | 0.853 |
| full_module_trace | all | 1910 | -2.082 | 0.026 | 0.654 | 0.527 | 0.029 | 0.933 | 0.973 |
| full_module_trace | architecture_matched | 355 | -2.082 | 0.143 | 0.648 | 0.515 | 0.151 | 0.915 | 0.858 |
| full_module_trace | size_matched | 361 | -2.082 | 0.143 | 0.640 | 0.526 | 0.151 | 0.833 | 0.858 |
| summary_module | all | 1910 | n/a | n/a | 0.000 | 0.516 | 0.026 | 0.959 | 0.977 |
| summary_module | architecture_matched | 355 | n/a | n/a | 0.000 | 0.463 | 0.122 | 0.961 | 0.896 |
| summary_module | size_matched | 361 | n/a | n/a | 0.000 | 0.475 | 0.122 | 0.904 | 0.896 |

## Family neighborhood top-k
| method | family | neighbor | rank | score |
| --- | --- | --- | --- | --- |
| metadata_only | Qwen/Qwen2-0.5B | Qwen/Qwen2.5-0.5B | 1 | -0.004 |
| metadata_only | Qwen/Qwen2-0.5B | meta-llama/Llama-3.2-1B | 2 | -0.145 |
| metadata_only | Qwen/Qwen2-0.5B | meta-llama/Llama-3.2-3B | 3 | -0.547 |
| metadata_only | Qwen/Qwen2-0.5B | meta-llama/Llama-2-7b-hf | 4 | -1.983 |
| metadata_only | Qwen/Qwen2-0.5B | mistralai/Mistral-7B-v0.1 | 5 | -2.274 |
| metadata_only | Qwen/Qwen2-7B | Qwen/Qwen2.5-7B | 1 | -0.008 |
| metadata_only | Qwen/Qwen2-7B | meta-llama/Meta-Llama-3-8B | 2 | -0.111 |
| metadata_only | Qwen/Qwen2-7B | mistralai/Mistral-7B-v0.1 | 3 | -0.117 |
| metadata_only | Qwen/Qwen2-7B | meta-llama/Llama-3.1-8B | 4 | -0.130 |
| metadata_only | Qwen/Qwen2-7B | meta-llama/Llama-2-7b-hf | 5 | -0.155 |
| metadata_only | Qwen/Qwen2.5-0.5B | Qwen/Qwen2-0.5B | 1 | -0.004 |
| metadata_only | Qwen/Qwen2.5-0.5B | meta-llama/Llama-3.2-1B | 2 | -0.129 |
| metadata_only | Qwen/Qwen2.5-0.5B | meta-llama/Llama-3.2-3B | 3 | -0.525 |
| metadata_only | Qwen/Qwen2.5-0.5B | meta-llama/Llama-2-7b-hf | 4 | -1.950 |
| metadata_only | Qwen/Qwen2.5-0.5B | mistralai/Mistral-7B-v0.1 | 5 | -2.243 |

## Heterogeneous stress bucket
- rows: 97
