# Spectral Family Retrieval Preliminary Report

Headline claim tested: module-aware spectral candidate generation for base-family retrieval with abstention.

## Included methods
- intrinsic_basics_only
- pl_alpha_scalar
- attention_trace
- full_module_trace
- shared_role_trace
- role_aggregated_trace
- summary_module
- hybrid_lite

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
| intrinsic_basics_only | all | 49 | 0.327 | 0.589 | 0.327 | 0.735 | 1.000 |
| intrinsic_basics_only | architecture_matched | 17 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| intrinsic_basics_only | size_matched | 17 | 0.647 | 0.753 | 0.647 | 0.765 | 1.000 |
| pl_alpha_scalar | all | 49 | 0.857 | 0.903 | 0.857 | 0.939 | 0.939 |
| pl_alpha_scalar | architecture_matched | 17 | 0.824 | 0.902 | 0.824 | 1.000 | 1.000 |
| pl_alpha_scalar | size_matched | 17 | 0.824 | 0.902 | 0.824 | 1.000 | 1.000 |
| attention_trace | all | 49 | 0.918 | 0.946 | 0.918 | 0.959 | 0.980 |
| attention_trace | architecture_matched | 17 | 0.941 | 0.971 | 0.941 | 1.000 | 1.000 |
| attention_trace | size_matched | 17 | 0.941 | 0.971 | 0.941 | 1.000 | 1.000 |
| full_module_trace | all | 49 | 0.918 | 0.950 | 0.918 | 0.980 | 1.000 |
| full_module_trace | architecture_matched | 17 | 0.941 | 0.971 | 0.941 | 1.000 | 1.000 |
| full_module_trace | size_matched | 17 | 0.941 | 0.971 | 0.941 | 1.000 | 1.000 |
| shared_role_trace | all | 49 | 0.918 | 0.950 | 0.918 | 0.980 | 1.000 |
| shared_role_trace | architecture_matched | 17 | 0.941 | 0.971 | 0.941 | 1.000 | 1.000 |
| shared_role_trace | size_matched | 17 | 0.941 | 0.971 | 0.941 | 1.000 | 1.000 |
| role_aggregated_trace | all | 49 | 0.939 | 0.966 | 0.939 | 1.000 | 1.000 |
| role_aggregated_trace | architecture_matched | 17 | 0.941 | 0.971 | 0.941 | 1.000 | 1.000 |
| role_aggregated_trace | size_matched | 17 | 0.941 | 0.971 | 0.941 | 1.000 | 1.000 |
| summary_module | all | 49 | 0.796 | 0.873 | 0.796 | 0.939 | 0.959 |
| summary_module | architecture_matched | 17 | 0.706 | 0.843 | 0.706 | 1.000 | 1.000 |
| summary_module | size_matched | 17 | 0.706 | 0.843 | 0.706 | 1.000 | 1.000 |
| hybrid_lite | all | 49 | 0.918 | 0.950 | 0.918 | 0.980 | 1.000 |
| hybrid_lite | architecture_matched | 17 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hybrid_lite | size_matched | 17 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Rejection metrics
| method | slice | num_examples | threshold | accepted_precision | accepted_coverage | auroc | auprc | fpr95 | aurc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| intrinsic_basics_only | all | 1910 | -0.023 | 0.006 | 0.654 | 0.508 | 0.026 | 0.962 | 0.995 |
| intrinsic_basics_only | architecture_matched | 1823 | -0.023 | 0.006 | 0.615 | 0.443 | 0.008 | 0.879 | 0.994 |
| intrinsic_basics_only | size_matched | 1829 | -0.023 | 0.006 | 0.613 | 0.451 | 0.008 | 0.883 | 0.995 |
| pl_alpha_scalar | all | 1910 | n/a | n/a | 0.000 | 0.549 | 0.029 | 0.914 | 0.974 |
| pl_alpha_scalar | architecture_matched | 1823 | n/a | n/a | 0.000 | 0.772 | 0.030 | 0.970 | 0.983 |
| pl_alpha_scalar | size_matched | 1829 | n/a | n/a | 0.000 | 0.772 | 0.030 | 0.965 | 0.983 |
| attention_trace | all | 1905 | n/a | n/a | 0.000 | 0.538 | 0.031 | 0.956 | 0.971 |
| attention_trace | architecture_matched | 1819 | n/a | n/a | 0.000 | 0.751 | 0.031 | 0.931 | 0.982 |
| attention_trace | size_matched | 1825 | n/a | n/a | 0.000 | 0.754 | 0.031 | 0.930 | 0.982 |
| full_module_trace | all | 1910 | -2.082 | 0.026 | 0.654 | 0.527 | 0.029 | 0.933 | 0.973 |
| full_module_trace | architecture_matched | 1823 | -2.082 | 0.032 | 0.172 | 0.755 | 0.027 | 0.888 | 0.982 |
| full_module_trace | size_matched | 1829 | -2.082 | 0.032 | 0.172 | 0.758 | 0.027 | 0.880 | 0.982 |
| shared_role_trace | all | 1910 | n/a | n/a | 0.000 | 0.527 | 0.030 | 0.933 | 0.973 |
| shared_role_trace | architecture_matched | 1823 | n/a | n/a | 0.000 | 0.756 | 0.027 | 0.888 | 0.982 |
| shared_role_trace | size_matched | 1829 | n/a | n/a | 0.000 | 0.758 | 0.027 | 0.880 | 0.982 |
| role_aggregated_trace | all | 1910 | n/a | n/a | 0.000 | 0.537 | 0.030 | 0.940 | 0.972 |
| role_aggregated_trace | architecture_matched | 1823 | n/a | n/a | 0.000 | 0.774 | 0.030 | 0.674 | 0.981 |
| role_aggregated_trace | size_matched | 1829 | n/a | n/a | 0.000 | 0.776 | 0.030 | 0.663 | 0.981 |
| summary_module | all | 1910 | n/a | n/a | 0.000 | 0.516 | 0.026 | 0.959 | 0.977 |
| summary_module | architecture_matched | 1823 | n/a | n/a | 0.000 | 0.750 | 0.026 | 0.996 | 0.986 |
| summary_module | size_matched | 1829 | n/a | n/a | 0.000 | 0.750 | 0.026 | 0.994 | 0.986 |
| hybrid_lite | all | 1910 | 0.989 | 0.027 | 0.770 | 0.577 | 0.032 | 0.858 | 0.973 |
| hybrid_lite | architecture_matched | 1823 | 0.989 | 0.013 | 0.659 | 0.640 | 0.013 | 0.922 | 0.991 |
| hybrid_lite | size_matched | 1829 | 0.989 | 0.009 | 0.606 | 0.450 | 0.009 | 0.948 | 0.995 |

## Family neighborhood top-k
| method | family | neighbor | rank | score |
| --- | --- | --- | --- | --- |
| intrinsic_basics_only | Qwen/Qwen2-0.5B | Qwen/Qwen2.5-0.5B | 1 | -0.005 |
| intrinsic_basics_only | Qwen/Qwen2-0.5B | meta-llama/Llama-3.2-3B | 2 | -0.775 |
| intrinsic_basics_only | Qwen/Qwen2-0.5B | Qwen/Qwen2.5-7B | 3 | -0.865 |
| intrinsic_basics_only | Qwen/Qwen2-0.5B | Qwen/Qwen2-7B | 4 | -1.865 |
| intrinsic_basics_only | Qwen/Qwen2-0.5B | mistralai/Mistral-7B-v0.1 | 5 | -2.928 |
| intrinsic_basics_only | Qwen/Qwen2-7B | Qwen/Qwen2.5-7B | 1 | -0.503 |
| intrinsic_basics_only | Qwen/Qwen2-7B | meta-llama/Llama-3.2-3B | 2 | -1.540 |
| intrinsic_basics_only | Qwen/Qwen2-7B | meta-llama/Llama-3.1-8B | 3 | -1.552 |
| intrinsic_basics_only | Qwen/Qwen2-7B | meta-llama/Meta-Llama-3-8B | 4 | -1.557 |
| intrinsic_basics_only | Qwen/Qwen2-7B | Qwen/Qwen2-0.5B | 5 | -1.865 |
| intrinsic_basics_only | Qwen/Qwen2.5-0.5B | Qwen/Qwen2-0.5B | 1 | -0.005 |
| intrinsic_basics_only | Qwen/Qwen2.5-0.5B | meta-llama/Llama-3.2-3B | 2 | -0.750 |
| intrinsic_basics_only | Qwen/Qwen2.5-0.5B | Qwen/Qwen2.5-7B | 3 | -0.897 |
| intrinsic_basics_only | Qwen/Qwen2.5-0.5B | Qwen/Qwen2-7B | 4 | -1.974 |
| intrinsic_basics_only | Qwen/Qwen2.5-0.5B | mistralai/Mistral-7B-v0.1 | 5 | -2.920 |

## Heterogeneous stress bucket
- rows: 97
