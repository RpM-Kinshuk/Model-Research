# Mixed-Zoo Retrieval Cycle 1 Report

Headline claim tested: model-intrinsic and spectral retrieval on a mixed zoo of homogeneous, heterogeneous, and unsupported models.

## Included methods
- intrinsic_basics_only
- pl_alpha_scalar
- full_module_trace
- shared_role_trace
- role_aggregated_trace

## Slice summary
- seed: 0
- min_family_size: 5
- gallery_supported: 153
- queries_supported: 72
- test_queries_homogeneous_supported: 49
- test_queries_heterogeneous_supported: 0
- unsupported_special: 312
- unsupported_unknown: 1549

## Retrieval metrics
| method | slice | num_queries | top1_accuracy | mrr | recall@1 | recall@3 | recall@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| intrinsic_basics_only | all_supported | 49 | 0.327 | 0.589 | 0.327 | 0.735 | 1.000 |
| intrinsic_basics_only | homogeneous_supported | 49 | 0.327 | 0.589 | 0.327 | 0.735 | 1.000 |
| intrinsic_basics_only | heterogeneous_supported | 0 | n/a | n/a | n/a | n/a | n/a |
| pl_alpha_scalar | all_supported | 49 | 0.857 | 0.903 | 0.857 | 0.939 | 0.939 |
| pl_alpha_scalar | homogeneous_supported | 49 | 0.857 | 0.903 | 0.857 | 0.939 | 0.939 |
| pl_alpha_scalar | heterogeneous_supported | 0 | n/a | n/a | n/a | n/a | n/a |
| full_module_trace | all_supported | 49 | 0.918 | 0.950 | 0.918 | 0.980 | 1.000 |
| full_module_trace | homogeneous_supported | 49 | 0.918 | 0.950 | 0.918 | 0.980 | 1.000 |
| full_module_trace | heterogeneous_supported | 0 | n/a | n/a | n/a | n/a | n/a |
| shared_role_trace | all_supported | 49 | 0.918 | 0.950 | 0.918 | 0.980 | 1.000 |
| shared_role_trace | homogeneous_supported | 49 | 0.918 | 0.950 | 0.918 | 0.980 | 1.000 |
| shared_role_trace | heterogeneous_supported | 0 | n/a | n/a | n/a | n/a | n/a |
| role_aggregated_trace | all_supported | 49 | 0.939 | 0.966 | 0.939 | 1.000 | 1.000 |
| role_aggregated_trace | homogeneous_supported | 49 | 0.939 | 0.966 | 0.939 | 1.000 | 1.000 |
| role_aggregated_trace | heterogeneous_supported | 0 | n/a | n/a | n/a | n/a | n/a |

## Rejection metrics
| method | slice | num_examples | threshold | accepted_precision | accepted_coverage | auroc | auprc | fpr95 | aurc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| intrinsic_basics_only | all_supported | 1910 | -0.023 | 0.006 | 0.654 | 0.508 | 0.026 | 0.962 | 0.993 |
| intrinsic_basics_only | homogeneous_supported | 1910 | -0.023 | 0.006 | 0.654 | 0.508 | 0.026 | 0.962 | 0.993 |
| intrinsic_basics_only | heterogeneous_supported | 1861 | -0.023 | 0.000 | 0.653 | n/a | n/a | n/a | 0.999 |
| pl_alpha_scalar | all_supported | 1910 | -0.023 | 0.031 | 0.513 | 0.549 | 0.029 | 0.914 | 0.974 |
| pl_alpha_scalar | homogeneous_supported | 1910 | -0.023 | 0.031 | 0.513 | 0.549 | 0.029 | 0.914 | 0.974 |
| pl_alpha_scalar | heterogeneous_supported | 1861 | -0.023 | 0.000 | 0.510 | n/a | n/a | n/a | 0.999 |
| full_module_trace | all_supported | 1910 | -2.082 | 0.026 | 0.654 | 0.527 | 0.029 | 0.933 | 0.973 |
| full_module_trace | homogeneous_supported | 1910 | -2.082 | 0.026 | 0.654 | 0.527 | 0.029 | 0.933 | 0.973 |
| full_module_trace | heterogeneous_supported | 1861 | -2.082 | 0.000 | 0.653 | n/a | n/a | n/a | 0.999 |
| shared_role_trace | all_supported | 1910 | -2.082 | 0.026 | 0.654 | 0.527 | 0.030 | 0.933 | 0.972 |
| shared_role_trace | homogeneous_supported | 1910 | -2.082 | 0.026 | 0.654 | 0.527 | 0.030 | 0.933 | 0.972 |
| shared_role_trace | heterogeneous_supported | 1861 | -2.082 | 0.000 | 0.653 | n/a | n/a | n/a | 0.999 |
| role_aggregated_trace | all_supported | 1910 | -0.513 | 0.027 | 0.652 | 0.537 | 0.030 | 0.940 | 0.972 |
| role_aggregated_trace | homogeneous_supported | 1910 | -0.513 | 0.027 | 0.652 | 0.537 | 0.030 | 0.940 | 0.972 |
| role_aggregated_trace | heterogeneous_supported | 1861 | -0.513 | 0.000 | 0.651 | n/a | n/a | n/a | 0.999 |

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
