# Lean Mixed-Zoo Benchmark Report

Headline claim tested: model-intrinsic and spectral retrieval on a mixed zoo of homogeneous, heterogeneous, and unsupported models.

## Included methods
- structural_basics_only
- pl_alpha_scalar
- full_module_trace
- shared_role_trace
- role_aggregated_trace

## Slice summary
- benchmark_mode: lean
- seed: 0
- min_family_size: 5
- gallery_supported: 184
- queries_supported: 86
- test_queries_homogeneous_supported: 58
- test_queries_heterogeneous_supported: 14
- unsupported_special: 312
- unsupported_unknown: 1549

## Retrieval metrics
| method | slice | num_queries | top1_accuracy | mrr | recall@1 | recall@3 | recall@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| structural_basics_only | all_supported | 72 | 0.347 | 0.587 | 0.347 | 0.792 | 1.000 |
| structural_basics_only | homogeneous_supported | 58 | 0.259 | 0.524 | 0.259 | 0.741 | 1.000 |
| structural_basics_only | heterogeneous_supported | 14 | 0.714 | 0.845 | 0.714 | 1.000 | 1.000 |
| pl_alpha_scalar | all_supported | 72 | 0.819 | 0.892 | 0.819 | 0.958 | 0.958 |
| pl_alpha_scalar | homogeneous_supported | 58 | 0.828 | 0.892 | 0.828 | 0.948 | 0.948 |
| pl_alpha_scalar | heterogeneous_supported | 14 | 0.786 | 0.893 | 0.786 | 1.000 | 1.000 |
| full_module_trace | all_supported | 72 | 0.861 | 0.910 | 0.861 | 0.958 | 0.972 |
| full_module_trace | homogeneous_supported | 58 | 0.862 | 0.910 | 0.862 | 0.966 | 0.966 |
| full_module_trace | heterogeneous_supported | 14 | 0.857 | 0.907 | 0.857 | 0.929 | 1.000 |
| shared_role_trace | all_supported | 72 | 0.861 | 0.910 | 0.861 | 0.958 | 0.986 |
| shared_role_trace | homogeneous_supported | 58 | 0.862 | 0.911 | 0.862 | 0.966 | 0.983 |
| shared_role_trace | heterogeneous_supported | 14 | 0.857 | 0.907 | 0.857 | 0.929 | 1.000 |
| role_aggregated_trace | all_supported | 72 | 0.903 | 0.946 | 0.903 | 0.986 | 1.000 |
| role_aggregated_trace | homogeneous_supported | 58 | 0.897 | 0.941 | 0.897 | 0.983 | 1.000 |
| role_aggregated_trace | heterogeneous_supported | 14 | 0.929 | 0.964 | 0.929 | 1.000 | 1.000 |

## Rejection metrics
| method | slice | num_examples | threshold | accepted_precision | accepted_coverage | auroc | auprc | fpr95 | aurc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| structural_basics_only | all_supported | 1933 | 0.000 | 0.014 | 0.917 | 0.536 | 0.040 | 0.914 | 0.989 |
| structural_basics_only | homogeneous_supported | 1919 | 0.000 | 0.009 | 0.917 | 0.543 | 0.033 | 0.914 | 0.991 |
| structural_basics_only | heterogeneous_supported | 1875 | 0.000 | 0.006 | 0.914 | 0.505 | 0.008 | 0.995 | 0.993 |
| pl_alpha_scalar | all_supported | 1933 | -0.961 | 0.030 | 0.976 | 0.606 | 0.057 | 0.979 | 0.957 |
| pl_alpha_scalar | homogeneous_supported | 1919 | -0.961 | 0.025 | 0.977 | 0.614 | 0.040 | 0.918 | 0.966 |
| pl_alpha_scalar | heterogeneous_supported | 1875 | -0.961 | 0.005 | 0.976 | 0.571 | 0.043 | 0.997 | 0.990 |
| full_module_trace | all_supported | 1933 | -3.190 | 0.041 | 0.726 | 0.590 | 0.068 | 0.904 | 0.948 |
| full_module_trace | homogeneous_supported | 1919 | -3.190 | 0.033 | 0.725 | 0.576 | 0.043 | 0.894 | 0.962 |
| full_module_trace | heterogeneous_supported | 1875 | -3.190 | 0.008 | 0.723 | 0.646 | 0.065 | 1.000 | 0.984 |
| shared_role_trace | all_supported | 1933 | -3.190 | 0.041 | 0.725 | 0.584 | 0.076 | 0.902 | 0.947 |
| shared_role_trace | homogeneous_supported | 1919 | -3.190 | 0.033 | 0.724 | 0.577 | 0.044 | 0.880 | 0.962 |
| shared_role_trace | heterogeneous_supported | 1875 | -3.190 | 0.008 | 0.723 | 0.611 | 0.099 | 1.000 | 0.983 |
| role_aggregated_trace | all_supported | 1933 | -0.998 | 0.039 | 0.763 | 0.579 | 0.070 | 0.957 | 0.947 |
| role_aggregated_trace | homogeneous_supported | 1919 | -0.998 | 0.032 | 0.762 | 0.584 | 0.048 | 0.949 | 0.960 |
| role_aggregated_trace | heterogeneous_supported | 1875 | -0.998 | 0.008 | 0.759 | 0.559 | 0.056 | 1.000 | 0.986 |

## Family neighborhood top-k
| method | family | neighbor | rank | score |
| --- | --- | --- | --- | --- |
| structural_basics_only | Qwen/Qwen2-0.5B | Qwen/Qwen2.5-0.5B | 1 | -0.000 |
| structural_basics_only | Qwen/Qwen2-0.5B | Qwen/Qwen2-7B | 2 | -1.333 |
| structural_basics_only | Qwen/Qwen2-0.5B | Qwen/Qwen2.5-7B | 3 | -1.333 |
| structural_basics_only | Qwen/Qwen2-0.5B | meta-llama/Llama-3.2-3B | 4 | -1.333 |
| structural_basics_only | Qwen/Qwen2-0.5B | meta-llama/Llama-2-7b-hf | 5 | -5.333 |
| structural_basics_only | Qwen/Qwen2-7B | Qwen/Qwen2.5-7B | 1 | -0.000 |
| structural_basics_only | Qwen/Qwen2-7B | meta-llama/Llama-3.2-3B | 2 | -0.000 |
| structural_basics_only | Qwen/Qwen2-7B | Qwen/Qwen2-0.5B | 3 | -1.333 |
| structural_basics_only | Qwen/Qwen2-7B | Qwen/Qwen2.5-0.5B | 4 | -1.333 |
| structural_basics_only | Qwen/Qwen2-7B | meta-llama/Llama-2-7b-hf | 5 | -1.333 |
| structural_basics_only | Qwen/Qwen2.5-0.5B | Qwen/Qwen2-0.5B | 1 | -0.000 |
| structural_basics_only | Qwen/Qwen2.5-0.5B | Qwen/Qwen2-7B | 2 | -1.333 |
| structural_basics_only | Qwen/Qwen2.5-0.5B | Qwen/Qwen2.5-7B | 3 | -1.333 |
| structural_basics_only | Qwen/Qwen2.5-0.5B | meta-llama/Llama-3.2-3B | 4 | -1.333 |
| structural_basics_only | Qwen/Qwen2.5-0.5B | meta-llama/Llama-2-7b-hf | 5 | -5.333 |
