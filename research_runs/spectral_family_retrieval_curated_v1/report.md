# Spectral Family Retrieval Preliminary Report

Headline claim tested: module-aware spectral candidate generation for base-family retrieval with abstention.

## Included methods
- metadata_only
- pl_alpha_scalar
- attention_trace
- full_module_trace
- summary_module
- hybrid_lite

## Split summary
- seed: 0
- min_family_size: 8
- gallery_supported: 355
- queries_supported: 165
- queries_unsupported: 1566
- queries_heterogeneous: 97

## Retrieval metrics
| method | slice | num_queries | top1_accuracy | mrr | recall@1 | recall@3 | recall@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| metadata_only | all | 107 | 0.028 | 0.148 | 0.028 | 0.103 | 0.271 |
| metadata_only | architecture_matched | 35 | 0.114 | 0.376 | 0.114 | 0.514 | 0.857 |
| metadata_only | size_matched | 35 | 0.086 | 0.333 | 0.086 | 0.371 | 0.829 |
| pl_alpha_scalar | all | 107 | 0.449 | 0.694 | 0.449 | 0.935 | 0.953 |
| pl_alpha_scalar | architecture_matched | 35 | 0.686 | 0.822 | 0.686 | 0.943 | 1.000 |
| pl_alpha_scalar | size_matched | 35 | 0.686 | 0.822 | 0.686 | 0.943 | 1.000 |
| attention_trace | all | 107 | 0.692 | 0.822 | 0.692 | 0.944 | 0.953 |
| attention_trace | architecture_matched | 35 | 0.743 | 0.854 | 0.743 | 0.943 | 1.000 |
| attention_trace | size_matched | 35 | 0.743 | 0.854 | 0.743 | 0.943 | 1.000 |
| full_module_trace | all | 107 | 0.720 | 0.833 | 0.720 | 0.935 | 0.953 |
| full_module_trace | architecture_matched | 35 | 0.829 | 0.896 | 0.829 | 0.943 | 0.971 |
| full_module_trace | size_matched | 35 | 0.829 | 0.896 | 0.829 | 0.943 | 0.971 |
| summary_module | all | 107 | 0.421 | 0.667 | 0.421 | 0.897 | 0.944 |
| summary_module | architecture_matched | 35 | 0.629 | 0.789 | 0.629 | 0.943 | 1.000 |
| summary_module | size_matched | 35 | 0.629 | 0.789 | 0.629 | 0.943 | 1.000 |
| hybrid_lite | all | 107 | 0.495 | 0.700 | 0.495 | 0.888 | 0.925 |
| hybrid_lite | architecture_matched | 35 | 0.657 | 0.810 | 0.657 | 0.943 | 0.971 |
| hybrid_lite | size_matched | 35 | 0.629 | 0.789 | 0.629 | 0.914 | 0.971 |

## Rejection metrics
| method | slice | num_examples | threshold | accepted_precision | accepted_coverage | auroc | auprc | fpr95 | aurc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metadata_only | all | 1673 | -0.444 | 0.002 | 0.925 | 0.530 | 0.068 | 0.931 | 0.997 |
| metadata_only | architecture_matched | 1531 | -0.444 | 0.002 | 0.845 | 0.441 | 0.020 | 0.992 | 0.997 |
| metadata_only | size_matched | 1536 | -0.444 | 0.002 | 0.874 | 0.439 | 0.020 | 0.996 | 0.997 |
| pl_alpha_scalar | all | 1673 | n/a | n/a | 0.000 | 0.675 | 0.141 | 0.907 | 0.942 |
| pl_alpha_scalar | architecture_matched | 1531 | n/a | n/a | 0.000 | 0.745 | 0.199 | 0.980 | 0.953 |
| pl_alpha_scalar | size_matched | 1536 | n/a | n/a | 0.000 | 0.747 | 0.199 | 0.966 | 0.953 |
| attention_trace | all | 1668 | n/a | n/a | 0.000 | 0.608 | 0.100 | 0.935 | 0.934 |
| attention_trace | architecture_matched | 1529 | n/a | n/a | 0.000 | 0.721 | 0.094 | 0.863 | 0.964 |
| attention_trace | size_matched | 1534 | n/a | n/a | 0.000 | 0.723 | 0.094 | 0.851 | 0.964 |
| full_module_trace | all | 1673 | -3.527 | 0.049 | 0.783 | 0.605 | 0.096 | 0.911 | 0.933 |
| full_module_trace | architecture_matched | 1531 | -3.527 | 0.025 | 0.653 | 0.716 | 0.088 | 0.814 | 0.961 |
| full_module_trace | size_matched | 1536 | -3.527 | 0.025 | 0.648 | 0.717 | 0.088 | 0.808 | 0.961 |
| summary_module | all | 1673 | n/a | n/a | 0.000 | 0.653 | 0.116 | 0.941 | 0.949 |
| summary_module | architecture_matched | 1531 | n/a | n/a | 0.000 | 0.745 | 0.115 | 0.987 | 0.962 |
| summary_module | size_matched | 1536 | n/a | n/a | 0.000 | 0.745 | 0.115 | 0.986 | 0.962 |
| hybrid_lite | all | 1673 | 0.744 | 0.034 | 0.926 | 0.471 | 0.061 | 0.879 | 0.968 |
| hybrid_lite | architecture_matched | 1531 | 0.744 | 0.009 | 0.150 | 0.556 | 0.027 | 1.000 | 0.975 |
| hybrid_lite | size_matched | 1536 | 0.744 | 0.007 | 0.094 | 0.537 | 0.031 | 1.000 | 0.977 |

## Family neighborhood top-k
| method | family | neighbor | rank | score |
| --- | --- | --- | --- | --- |
| metadata_only | Qwen/Qwen2-0.5B | Qwen/Qwen2.5-0.5B | 1 | -0.021 |
| metadata_only | Qwen/Qwen2-0.5B | Qwen/Qwen2.5-1.5B-Instruct | 2 | -0.036 |
| metadata_only | Qwen/Qwen2-0.5B | meta-llama/Llama-3.2-3B-Instruct | 3 | -0.063 |
| metadata_only | Qwen/Qwen2-0.5B | Qwen/Qwen2.5-0.5B-Instruct | 4 | -0.063 |
| metadata_only | Qwen/Qwen2-0.5B | mistralai/Mistral-7B-v0.1 | 5 | -0.081 |
| metadata_only | Qwen/Qwen2-0.5B-Instruct | meta-llama/Llama-2-7b-hf | 1 | -0.000 |
| metadata_only | Qwen/Qwen2-0.5B-Instruct | meta-llama/Llama-3.2-1B | 2 | -0.000 |
| metadata_only | Qwen/Qwen2-0.5B-Instruct | meta-llama/Llama-3.2-1B-Instruct | 3 | -0.000 |
| metadata_only | Qwen/Qwen2-0.5B-Instruct | meta-llama/Llama-3.2-3B | 4 | -0.000 |
| metadata_only | Qwen/Qwen2-0.5B-Instruct | meta-llama/Meta-Llama-3-8B-Instruct | 5 | -0.000 |
| metadata_only | Qwen/Qwen2-7B | mistralai/Mistral-Nemo-Base-2407 | 1 | -0.140 |
| metadata_only | Qwen/Qwen2-7B | mistralai/Mistral-Nemo-Instruct-2407 | 2 | -0.141 |
| metadata_only | Qwen/Qwen2-7B | Qwen/Qwen2.5-7B-Instruct | 3 | -0.445 |
| metadata_only | Qwen/Qwen2-7B | Qwen/Qwen2.5-7B | 4 | -0.651 |
| metadata_only | Qwen/Qwen2-7B | meta-llama/Meta-Llama-3-8B | 5 | -0.797 |

## Heterogeneous stress bucket
- rows: 97
