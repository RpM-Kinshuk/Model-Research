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
- gallery_supported: 476
- queries_supported: 225
- queries_unsupported: 687
- queries_heterogeneous: 97

## Retrieval metrics
| method | slice | num_queries | top1_accuracy | mrr | recall@1 | recall@3 | recall@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| metadata_only | all | 173 | 0.110 | 0.206 | 0.110 | 0.179 | 0.266 |
| metadata_only | architecture_matched | 67 | 0.313 | 0.493 | 0.313 | 0.537 | 0.851 |
| metadata_only | size_matched | 67 | 0.284 | 0.441 | 0.284 | 0.507 | 0.687 |
| pl_alpha_scalar | all | 173 | 0.509 | 0.706 | 0.509 | 0.896 | 0.931 |
| pl_alpha_scalar | architecture_matched | 67 | 0.776 | 0.867 | 0.776 | 0.955 | 0.985 |
| pl_alpha_scalar | size_matched | 67 | 0.776 | 0.867 | 0.776 | 0.955 | 0.985 |
| attention_trace | all | 173 | 0.734 | 0.830 | 0.734 | 0.908 | 0.965 |
| attention_trace | architecture_matched | 67 | 0.806 | 0.886 | 0.806 | 0.955 | 1.000 |
| attention_trace | size_matched | 67 | 0.806 | 0.887 | 0.806 | 0.955 | 1.000 |
| full_module_trace | all | 173 | 0.659 | 0.793 | 0.659 | 0.902 | 0.948 |
| full_module_trace | architecture_matched | 67 | 0.791 | 0.881 | 0.791 | 0.970 | 1.000 |
| full_module_trace | size_matched | 67 | 0.791 | 0.881 | 0.791 | 0.970 | 1.000 |
| summary_module | all | 173 | 0.590 | 0.741 | 0.590 | 0.873 | 0.913 |
| summary_module | architecture_matched | 67 | 0.776 | 0.851 | 0.776 | 0.896 | 0.970 |
| summary_module | size_matched | 67 | 0.776 | 0.848 | 0.776 | 0.896 | 0.955 |

## Rejection metrics
| method | slice | num_examples | threshold | accepted_precision | accepted_coverage | auroc | auprc | fpr95 | aurc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metadata_only | all | 860 | -0.402 | 0.021 | 0.874 | 0.539 | 0.216 | 0.910 | 0.981 |
| metadata_only | architecture_matched | 734 | -0.402 | 0.027 | 0.793 | 0.437 | 0.081 | 0.979 | 0.985 |
| metadata_only | size_matched | 738 | -0.402 | 0.026 | 0.822 | 0.440 | 0.081 | 0.969 | 0.988 |
| pl_alpha_scalar | all | 860 | n/a | n/a | 0.000 | 0.629 | 0.317 | 0.908 | 0.848 |
| pl_alpha_scalar | architecture_matched | 734 | n/a | n/a | 0.000 | 0.737 | 0.346 | 0.843 | 0.834 |
| pl_alpha_scalar | size_matched | 738 | n/a | n/a | 0.000 | 0.738 | 0.346 | 0.844 | 0.835 |
| attention_trace | all | 855 | n/a | n/a | 0.000 | 0.610 | 0.253 | 0.933 | 0.816 |
| attention_trace | architecture_matched | 732 | n/a | n/a | 0.000 | 0.732 | 0.231 | 0.878 | 0.857 |
| attention_trace | size_matched | 736 | n/a | n/a | 0.000 | 0.733 | 0.231 | 0.862 | 0.857 |
| full_module_trace | all | 860 | -6.458 | 0.137 | 0.951 | 0.604 | 0.246 | 0.936 | 0.838 |
| full_module_trace | architecture_matched | 734 | -6.458 | 0.086 | 0.812 | 0.733 | 0.234 | 0.793 | 0.859 |
| full_module_trace | size_matched | 738 | -6.458 | 0.086 | 0.805 | 0.735 | 0.234 | 0.787 | 0.860 |
| summary_module | all | 860 | n/a | n/a | 0.000 | 0.578 | 0.232 | 0.955 | 0.865 |
| summary_module | architecture_matched | 734 | n/a | n/a | 0.000 | 0.699 | 0.183 | 0.937 | 0.875 |
| summary_module | size_matched | 738 | n/a | n/a | 0.000 | 0.702 | 0.183 | 0.914 | 0.875 |

## Family neighborhood top-k
| method | family | neighbor | rank | score |
| --- | --- | --- | --- | --- |
| metadata_only | HuggingFaceH4/mistral-7b-sft-beta | Qwen/Qwen2-0.5B-Instruct | 1 | -0.000 |
| metadata_only | HuggingFaceH4/mistral-7b-sft-beta | meta-llama/Llama-2-7b-hf | 2 | -0.000 |
| metadata_only | HuggingFaceH4/mistral-7b-sft-beta | meta-llama/Llama-3.2-1B | 3 | -0.000 |
| metadata_only | HuggingFaceH4/mistral-7b-sft-beta | meta-llama/Llama-3.2-1B-Instruct | 4 | -0.000 |
| metadata_only | HuggingFaceH4/mistral-7b-sft-beta | mistralai/Mistral-7B-Instruct-v0.2 | 5 | -0.000 |
| metadata_only | Qwen/Qwen2-0.5B | Qwen/Qwen2.5-0.5B | 1 | -0.021 |
| metadata_only | Qwen/Qwen2-0.5B | Qwen/Qwen2.5-1.5B-Instruct | 2 | -0.042 |
| metadata_only | Qwen/Qwen2-0.5B | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 3 | -0.061 |
| metadata_only | Qwen/Qwen2-0.5B | meta-llama/Llama-3.2-3B-Instruct | 4 | -0.063 |
| metadata_only | Qwen/Qwen2-0.5B | Qwen/Qwen2.5-0.5B-Instruct | 5 | -0.063 |
| metadata_only | Qwen/Qwen2-0.5B-Instruct | HuggingFaceH4/mistral-7b-sft-beta | 1 | -0.000 |
| metadata_only | Qwen/Qwen2-0.5B-Instruct | meta-llama/Llama-2-7b-hf | 2 | -0.000 |
| metadata_only | Qwen/Qwen2-0.5B-Instruct | meta-llama/Llama-3.2-1B | 3 | -0.000 |
| metadata_only | Qwen/Qwen2-0.5B-Instruct | meta-llama/Llama-3.2-1B-Instruct | 4 | -0.000 |
| metadata_only | Qwen/Qwen2-0.5B-Instruct | mistralai/Mistral-7B-Instruct-v0.2 | 5 | -0.000 |

## Heterogeneous stress bucket
- rows: 97
